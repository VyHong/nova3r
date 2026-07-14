import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


UDF_PROCESSING_SCRIPT = Path(__file__).with_name("udf_sample.py").resolve()
DEFAULT_DATA_ROOT = Path("/tmp/datasets/scannetpp/scannetpp_scans/data")
DEFAULT_OUTPUT_ROOT = Path("/mnt/projects/theses/vyhong/scannetpp/udf")
MESH_NAME = "mesh_aligned_0.05.ply"


def find_meshes(data_root):
    return sorted(data_root.glob(f"*/scans/{MESH_NAME}"))


def output_prefix_for(mesh_path, data_root, output_root):
    relative_path = mesh_path.relative_to(data_root)
    return (output_root / relative_path).with_suffix("")


def process_mesh(mesh_path, output_prefix, overwrite, normalize_mode):
    expected_surface = Path(f"{output_prefix}_surface.npz")
    expected_udf = Path(f"{output_prefix}_udf.npz")

    if not overwrite and expected_surface.exists() and expected_udf.exists():
        return "skipped"

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(UDF_PROCESSING_SCRIPT),
        "--input_obj",
        str(mesh_path),
        "--output_prefix",
        str(output_prefix),
        "--normalize_mode",
        normalize_mode,
    ]

    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_prefix.name}_", dir=output_prefix.parent) as work_dir:
            subprocess.run(command, check=True, cwd=work_dir)
        return "processed"
    except subprocess.CalledProcessError as error:
        tqdm.write(f"[Error] {mesh_path} exited with code {error.returncode}")
        return "failed"
    except Exception as error:
        tqdm.write(f"[Error] {mesh_path}: {error}")
        return "failed"


def main(data_root, output_root, workers, overwrite, dry_run, limit, normalize_mode):
    data_root = data_root.resolve()
    output_root = output_root.resolve()

    if not data_root.is_dir():
        raise ValueError(f"Data root directory does not exist: {data_root}")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")

    mesh_paths = find_meshes(data_root)
    if limit is not None:
        mesh_paths = mesh_paths[:limit]

    print(f"Found {len(mesh_paths)} ScanNet++ meshes under {data_root}.")
    if not mesh_paths:
        return

    jobs = [(mesh_path, output_prefix_for(mesh_path, data_root, output_root)) for mesh_path in mesh_paths]

    if dry_run:
        for mesh_path, output_prefix in jobs:
            print(f"{mesh_path} -> {output_prefix}_{{surface,udf}}.npz")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    counts = {"processed": 0, "skipped": 0, "failed": 0}

    print(f"Processing with {workers} concurrent worker(s)...")
    print(f"normalize_mode={normalize_mode}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_mesh, mesh_path, output_prefix, overwrite, normalize_mode): mesh_path
            for mesh_path, output_prefix in jobs
        }
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing ScanNet++ meshes",
            unit="mesh",
        )

        for future in progress:
            mesh_path = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as error:
                counts["failed"] += 1
                tqdm.write(f"[Fatal] {mesh_path}: {error}")

            progress.set_postfix(
                processed=counts["processed"],
                skipped=counts["skipped"],
                failed=counts["failed"],
            )

    print("Complete: " f"{counts['processed']} processed, " f"{counts['skipped']} skipped, " f"{counts['failed']} failed.")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UDF sampling on ScanNet++ aligned meshes.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"ScanNet++ data root containing <scene>/scans/{MESH_NAME}.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Writable output root. The <scene>/scans structure is preserved.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent udf_sample.py subprocesses. Default: 1 due to high memory use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess meshes whose surface and UDF NPZ files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected inputs and outputs without processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N meshes, useful for a small test run.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="none",
        help="Passed to udf_sample.py. Default keeps ScanNet++ points in aligned scan coordinates.",
    )

    args = parser.parse_args()
    main(
        data_root=args.data_root,
        output_root=args.output_root,
        workers=args.workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        normalize_mode=args.normalize_mode,
    )
