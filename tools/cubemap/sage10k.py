"""Generate cubemap faces from rendered SAGE-10k equirectangular panoramas."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


GENERATOR = Path(__file__).with_name("generate.py")
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def output_is_complete(output_dir, require_depth=False):
    rgb_complete = (output_dir / "camera_data.json").is_file() and all(
        (output_dir / f"{face:04d}.jpg").is_file() for face in range(6)
    )
    if not require_depth:
        return rgb_complete
    depth_dir = output_dir.parent / "depth" / output_dir.name
    return rgb_complete and all(
        (depth_dir / f"{face:04d}.png").is_file() for face in range(6)
    )


def add_world_extrinsics(output_dir, panorama_w2c):
    metadata_path = output_dir / "camera_data.json"
    metadata = json.loads(metadata_path.read_text())
    panorama_w2c = np.asarray(panorama_w2c, dtype=np.float64)
    if panorama_w2c.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 panorama extrinsic, got {panorama_w2c.shape}")

    for face in metadata.values():
        face_w2c = np.asarray(face["extrinsics"], dtype=np.float64)
        world_w2c = face_w2c @ BLENDER_TO_OPENCV @ panorama_w2c
        face["world_extrinsics"] = world_w2c.tolist()
    metadata_path.write_text(json.dumps(metadata, indent=4))


def process_panorama(panorama_path, scene_dir, output_root, args):
    relative_scene = scene_dir.relative_to(args.input_root)
    output_dir = (
        output_root
        / relative_scene
        / f"{panorama_path.stem}_{args.shape}_{int(args.fov)}"
    )
    depth_path = scene_dir / args.depth_dir_name / panorama_path.name
    if args.require_depth and not depth_path.is_file():
        raise FileNotFoundError(
            f"No rendered depth for {panorama_path}; expected {depth_path}"
        )
    if not args.overwrite and output_is_complete(
        output_dir, require_depth=depth_path.is_file()
    ):
        print(f"Skipping complete panorama: {panorama_path}")
        return

    command = [
        sys.executable,
        str(GENERATOR),
        "--image",
        str(panorama_path),
        "--image_out",
        str(output_root / relative_scene / panorama_path.stem),
        "--shape",
        args.shape,
        "--fov",
        str(args.fov),
        "--res",
        str(args.res),
    ]
    if depth_path.is_file():
        command.extend(
            [
                "--depth",
                str(depth_path),
                "--depth_out",
                str(output_root / relative_scene / "depth" / panorama_path.stem),
            ]
        )
    subprocess.run(command, check=True)

    extrinsics_path = scene_dir / "extrinsics.json"
    if extrinsics_path.is_file():
        panorama_extrinsics = json.loads(extrinsics_path.read_text())
        if panorama_path.name not in panorama_extrinsics:
            raise KeyError(f"{panorama_path.name} is missing from {extrinsics_path}")
        add_world_extrinsics(
            output_dir, panorama_extrinsics[panorama_path.name]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate cubemap faces for rendered SAGE-10k panoramas."
    )
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Root containing SAGE scene directories"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="Process at most this many panoramas")
    parser.add_argument("--shape", default="cube")
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--res", type=int, default=518)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--depth-dir-name",
        default="depth",
        help="Depth panorama directory within each scene (default: depth)",
    )
    parser.add_argument(
        "--require-depth",
        action="store_true",
        help="Fail when a panorama has no corresponding rendered depth PNG",
    )
    args = parser.parse_args()
    if not args.input_root.is_dir():
        parser.error(f"--input-root does not exist: {args.input_root}")
    if args.workers < 1:
        parser.error("--workers must be positive")

    args.input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    panoramas = sorted(args.input_root.glob("*/*_panorama_*.png"))
    if args.limit is not None:
        panoramas = panoramas[: args.limit]
    if not panoramas:
        parser.error(f"no SAGE panorama PNGs found below {args.input_root}")

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_panorama, path, path.parent, output_root, args
            ): path
            for path in panoramas
        }
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((path, error))
                print(f"Failed: {path}: {error}", file=sys.stderr)

    print(f"Completed {len(panoramas) - len(failures)}/{len(panoramas)} panoramas")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
