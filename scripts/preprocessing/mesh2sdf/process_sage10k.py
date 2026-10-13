#!/usr/bin/env python3
"""Generate scene-level surface and UDF samples for rendered SAGE-10K meshes."""

import argparse
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


UDF_SCRIPT = Path(__file__).with_name("udf_sample.py").resolve()


def process_scene(
    mesh_path,
    output_prefix,
    overwrite,
    color_max_distance=0.05,
    sample_num=524288,
    omit_sharp_surface=True,
    progress=None,
    batch_size=None,
):
    outputs = (
        Path(f"{output_prefix}_surface.npz"),
        Path(f"{output_prefix}_udf.npz"),
    )
    if not overwrite and all(path.is_file() for path in outputs):
        if progress is not None:
            progress(4)
        return "skipped"

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(UDF_SCRIPT),
        "--input_obj",
        str(mesh_path),
        "--output_prefix",
        str(output_prefix),
        "--color_max_distance",
        str(color_max_distance),
        "--sample_num",
        str(sample_num),
        "--progress_events",
    ]
    if omit_sharp_surface:
        command.append("--omit_sharp_surface")
    if batch_size is not None:
        command.extend(("--batch_size", str(batch_size)))
    try:
        completed_stages = 0
        output_lines = []
        with tempfile.TemporaryDirectory(
            prefix=f".{output_prefix.name}_", dir=output_prefix.parent
        ) as work_dir:
            with subprocess.Popen(
                command, cwd=work_dir, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            ) as process:
                for line in process.stdout:
                    if line.strip() == "SAGE_UDF_PROGRESS":
                        completed_stages += 1
                        if progress is not None:
                            progress(1)
                    else:
                        output_lines.append(line.rstrip())
                exit_code = process.wait()
            if exit_code:
                raise subprocess.CalledProcessError(exit_code, command)
        return "processed"
    except Exception as error:
        tqdm.write(f"[Error] {mesh_path}: {error}\n" + "\n".join(output_lines[-10:]))
        return "failed"
    finally:
        if progress is not None:
            progress(max(0, 4 - completed_stages))


def main(args):
    meshes = sorted(args.data_root.resolve().glob("*/scene.glb"))
    if args.limit is not None:
        meshes = meshes[: args.limit]
    if not meshes:
        raise ValueError(f"No <scene>/scene.glb meshes found under {args.data_root}")

    jobs = [
        (mesh, args.output_root.resolve() / mesh.parent.name / "scene")
        for mesh in meshes
    ]
    if args.dry_run:
        for mesh, prefix in jobs:
            print(f"{mesh} -> {prefix}_{{surface,udf}}.npz")
        return

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    completed_stages = 0
    progress_lock = threading.Lock()
    with tqdm(total=len(jobs), desc="Processing SAGE-10K scenes", unit="scene") as progress_bar, ThreadPoolExecutor(max_workers=args.workers) as executor:
        def report_stages(amount):
            nonlocal completed_stages
            with progress_lock:
                completed_stages += amount
                progress_bar.set_postfix_str(
                    f"steps={completed_stages}/{4 * len(jobs)}", refresh=False
                )
                progress_bar.update(0)

        futures = {
            executor.submit(
                process_scene,
                mesh,
                prefix,
                args.overwrite,
                args.color_max_distance,
                args.sample_num,
                not args.include_sharp_surface,
                report_stages,
                args.batch_size,
            ): mesh
            for mesh, prefix in jobs
        }
        for future in as_completed(futures):
            counts[future.result()] += 1
            progress_bar.update(1)

    print(
        f"Complete: {counts['processed']} processed, {counts['skipped']} skipped, "
        f"{counts['failed']} failed."
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run UDF sampling on SAGE-10K scene.glb files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing <scene>/scene.glb files.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum UDF distance queries per batch (default: all points).")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--color-max-distance",
        type=float,
        default=0.05,
        help="Near-surface RGB band as a fraction of each mesh's sampling scale.",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=524288,
        help="Random-surface and sharp-surface samples per scene.",
    )
    parser.add_argument(
        "--include-sharp-surface",
        action="store_true",
        help="Also save sharp-edge surface points for models that use them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()
    if parsed.workers < 1:
        parser.error("--workers must be at least 1")
    if parsed.batch_size is not None and parsed.batch_size < 1:
        parser.error("--batch-size must be positive")
    if parsed.limit is not None and parsed.limit < 1:
        parser.error("--limit must be at least 1")
    if parsed.color_max_distance < 0:
        parser.error("--color-max-distance must be non-negative")
    if parsed.sample_num < 1:
        parser.error("--sample-num must be positive")
    main(parsed)
