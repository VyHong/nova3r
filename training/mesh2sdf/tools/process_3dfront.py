import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROCESSING_SCRIPT = Path(__file__).with_name("watertight_and_sample.py").resolve()


def output_prefix_for(mesh_path, data_root, output_root):
    relative_path = mesh_path.relative_to(data_root)
    return (output_root / relative_path).with_suffix("")


def load_room_ids(json_paths):
    room_ids = []

    for json_path in json_paths:
        json_path = Path(json_path)
        with open(json_path, "r") as f:
            ids = json.load(f)

        if not isinstance(ids, list):
            raise ValueError(f"{json_path} must contain a JSON list of room ids")

        room_ids.extend(ids)

    # remove duplicates while preserving order
    return list(dict.fromkeys(room_ids))


def find_selected_meshes(data_root, room_ids):
    mesh_paths = []
    missing_rooms = []
    missing_meshes = []

    for room_id in room_ids:
        room_dir = data_root / room_id

        if not room_dir.is_dir():
            missing_rooms.append(room_id)
            continue

        # The full room mesh is next to the room directory:
        # scene_id/RoomName-1234_full.glb
        full_mesh = room_dir.parent / f"{room_dir.name}_full.glb"

        if not full_mesh.is_file():
            missing_meshes.append(room_id)
            continue

        mesh_paths.append(full_mesh)

    mesh_paths = list(dict.fromkeys(mesh_paths))
    return mesh_paths, missing_rooms, missing_meshes


def process_mesh(mesh_path, output_prefix, overwrite):
    expected_surface = Path(f"{output_prefix}_surface.npz")
    expected_sdf = Path(f"{output_prefix}_sdf.npz")

    if not overwrite and expected_surface.exists() and expected_sdf.exists():
        return "skipped"

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROCESSING_SCRIPT),
        "--input_obj",
        str(mesh_path),
        "--output_prefix",
        str(output_prefix),
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


def main(
    data_root,
    output_root,
    workers,
    overwrite,
    dry_run,
    limit,
    room_id_json,
):
    data_root = data_root.resolve()
    output_root = output_root.resolve()

    if not data_root.is_dir():
        raise ValueError(f"Data root directory does not exist: {data_root}")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")

    room_ids = load_room_ids(room_id_json)

    print(f"Loaded {len(room_ids)} room ids from:")
    for path in room_id_json:
        print(f"  {path}")

    print(f"Searching selected rooms under {data_root}...")
    mesh_paths, missing_rooms, missing_meshes = find_selected_meshes(data_root, room_ids)

    if limit is not None:
        mesh_paths = mesh_paths[:limit]

    print(f"Found {len(mesh_paths)} selected full GLB meshes.")

    if missing_rooms:
        print(f"Warning: {len(missing_rooms)} room directories were missing.")
        for room_id in missing_rooms[:10]:
            print(f"  missing dir: {room_id}")
        if len(missing_rooms) > 10:
            print("  ...")

    if missing_meshes:
        print(f"Warning: {len(missing_meshes)} room directories had no *_full.glb.")
        for room_id in missing_meshes[:10]:
            print(f"  no mesh: {room_id}")
        if len(missing_meshes) > 10:
            print("  ...")

    if not mesh_paths:
        return

    jobs = [(mesh_path, output_prefix_for(mesh_path, data_root, output_root)) for mesh_path in mesh_paths]

    if dry_run:
        for mesh_path, output_prefix in jobs:
            print(f"{mesh_path} -> {output_prefix}_{{surface,sdf}}.npz")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    counts = {"processed": 0, "skipped": 0, "failed": 0}

    print(f"Processing with {workers} concurrent worker(s)...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_mesh, mesh_path, output_prefix, overwrite): mesh_path for mesh_path, output_prefix in jobs}

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing selected 3D-FRONT rooms",
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
    parser = argparse.ArgumentParser(description="Run watertight_and_sample.py on selected 3D-FRONT room meshes.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Mounted 3D-FRONT scene directory, e.g. /tmp/datasets/3dfront/3D-FRONT-SCENE.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Writable output directory. The input directory structure is preserved.",
    )
    parser.add_argument(
        "--room_id_json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSON files containing room ids, e.g. midi_room_ids.json validate_room_ids.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent subprocesses. Default: 1 due to high memory use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess meshes whose surface and SDF NPZ files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected inputs and outputs without processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected meshes, useful for a small test run.",
    )

    args = parser.parse_args()

    main(
        data_root=args.data_root,
        output_root=args.output_root,
        workers=args.workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        room_id_json=args.room_id_json,
    )
