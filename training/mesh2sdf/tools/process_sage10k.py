#!/usr/bin/env python3
"""Generate scene-level surface and UDF samples for rendered SAGE-10K meshes."""

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


UDF_SCRIPT = Path(__file__).with_name("udf_sample.py").resolve()


def process_scene(mesh_path, output_prefix, overwrite, color_max_distance=0.05):
    outputs = (
        Path(f"{output_prefix}_surface.npz"),
        Path(f"{output_prefix}_udf.npz"),
    )
    if not overwrite and all(path.is_file() for path in outputs):
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
    ]
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output_prefix.name}_", dir=output_prefix.parent
        ) as work_dir:
            subprocess.run(command, check=True, cwd=work_dir)
        return "processed"
    except Exception as error:
        tqdm.write(f"[Error] {mesh_path}: {error}")
        return "failed"


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
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_scene,
                mesh,
                prefix,
                args.overwrite,
                args.color_max_distance,
            ): mesh
            for mesh, prefix in jobs
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing SAGE-10K"
        ):
            counts[future.result()] += 1

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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--color-max-distance",
        type=float,
        default=0.05,
        help="Near-surface RGB band as a fraction of each mesh's sampling scale.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()
    if parsed.workers < 1:
        parser.error("--workers must be at least 1")
    if parsed.limit is not None and parsed.limit < 1:
        parser.error("--limit must be at least 1")
    if parsed.color_max_distance < 0:
        parser.error("--color-max-distance must be non-negative")
    main(parsed)
