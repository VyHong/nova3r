import argparse
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WATERTIGHT_PROCESSING_SCRIPT = Path(__file__).with_name("watertight_and_sample.py").resolve()
UDF_PROCESSING_SCRIPT = Path(__file__).with_name("udf_sample.py").resolve()


def output_prefix_for(file_path, data_root, output_root):
    if output_root is None:
        return file_path.with_suffix("")

    relative_path = file_path.relative_to(data_root)
    return (output_root / relative_path).with_suffix("")


def process_mesh(file_path, output_prefix, udf_only):
    """Worker function to process a single mesh file."""
    processing_script = UDF_PROCESSING_SCRIPT if udf_only else WATERTIGHT_PROCESSING_SCRIPT
    expected_outputs = (
        [Path(f"{output_prefix}_udf.npz")]
        if udf_only
        else [Path(f"{output_prefix}_surface.npz"), Path(f"{output_prefix}_sdf.npz")]
    )

    if all(output.exists() for output in expected_outputs):
        print(f"[Skipping] Files already exist for: {file_path.name}")
        return True

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(processing_script),
        "--input_obj",
        str(file_path),
        "--output_prefix",
        str(output_prefix),
    ]

    print(f"[Running] {' '.join(command)}")

    try:
        # Run the command and wait for it to finish
        subprocess.run(command, check=True)
        print(f"[Success] Processed {file_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Processing failed for {file_path}. Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"[Exception] Unexpected error for {file_path}: {e}")
        return False


def main(data_root, output_root, num_workers, udf_only):
    root_path = data_root.resolve()
    output_root = output_root.resolve() if output_root is not None else None

    if not root_path.is_dir():
        raise ValueError(f"Data root directory does not exist: {root_path}")
    if num_workers < 1:
        raise ValueError("--workers must be at least 1")

    # 1. Find all .ply files and group them by their parent folder
    print(f"Scanning {root_path} for .ply files...")
    ply_files = root_path.rglob("*.ply")
    files_by_dir = defaultdict(list)

    for file_path in ply_files:
        files_by_dir[file_path.parent].append(file_path)

    files_to_process = []

    # 2. Apply filtering logic per directory
    for directory, files in files_by_dir.items():
        cropped_files = [f for f in files if f.name.endswith("_cropped.ply")]
        aligned_files = [f for f in files if f.name.endswith("_aligned.ply")]

        if cropped_files:
            # If cropped exist in this folder, ONLY take the cropped ones
            files_to_process.extend(cropped_files)
        elif aligned_files:
            # Otherwise, fall back to aligned files
            files_to_process.extend(aligned_files)

    files_to_process.sort()
    print(f"Found {len(files_to_process)} valid meshes to process based on rules.")
    for f in files_to_process:
        output_prefix = output_prefix_for(f, root_path, output_root)
        output_suffixes = "udf" if udf_only else "surface,sdf"
        print(f" - {f.relative_to(root_path)} -> {output_prefix}_{{{output_suffixes}}}.npz")
    if not files_to_process:
        print("No files to process. Exiting.")
        return

    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    # 3. Execute the processing pipeline concurrently
    print(f"\nStarting processing with {num_workers} concurrent workers...\n" + "-" * 50)

    successful_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the executor
        future_to_file = {
            executor.submit(
                process_mesh,
                file_path,
                output_prefix_for(file_path, root_path, output_root),
                udf_only,
            ): file_path
            for file_path in files_to_process
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                success = future.result()
                if success:
                    successful_count += 1
            except Exception as exc:
                print(f"[Fatal Error] Task for {file_path.name} generated an exception: {exc}")

    print("-" * 50)
    print(f"Processing complete! Successfully handled {successful_count}/{len(files_to_process)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process PLY files based on cropped/aligned hierarchy.")
    parser.add_argument("--data_root", type=Path, required=True, help="Path to the root Replica data directory.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional output directory that mirrors the structure under --data_root. "
        "By default, outputs are written beside each input mesh.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of concurrent subprocesses to run.")
    parser.add_argument(
        "--udf-only",
        action="store_true",
        help="Sample unsigned distance data with udf_sample.py instead of running watertight_and_sample.py.",
    )
    args = parser.parse_args()

    main(args.data_root, args.output_root, args.workers, args.udf_only)
