"""Generate cubemap faces from rendered SAGE-10K or 3D-FRONT panoramas."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from nova3r.utils.three_d_front import GLTF_TO_BLENDER


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


def add_world_extrinsics(
    output_dir, panorama_w2c, normalization=None, dataset_layout="sage10k"
):
    metadata_path = output_dir / "camera_data.json"
    metadata = json.loads(metadata_path.read_text())
    panorama_w2c = np.asarray(panorama_w2c, dtype=np.float64)
    if panorama_w2c.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 panorama extrinsic, got {panorama_w2c.shape}")

    for face in metadata.values():
        face_w2c = np.asarray(face["extrinsics"], dtype=np.float64)
        world_w2c = face_w2c @ BLENDER_TO_OPENCV @ panorama_w2c
        if dataset_layout == "3dfront":
            # The room mesh and its UDF samples retain glTF Y-up coordinates;
            # the panorama pose is expressed after Blender's glTF import.
            world_w2c = world_w2c @ GLTF_TO_BLENDER
        face["world_extrinsics"] = world_w2c.tolist()
        if normalization is not None:
            # x_normalized = scale * (x_gltf - center). Keep rotations rigid;
            # normalized camera coordinates are scaled metric camera coordinates.
            center = np.asarray(normalization["center_gltf"], dtype=np.float64)
            scale = normalization["scale"]
            normalized_w2c = world_w2c.copy()
            normalized_w2c[:3, 3] = scale * (
                world_w2c[:3, :3] @ center + world_w2c[:3, 3]
            )
            face["normalized_world_extrinsics"] = normalized_w2c.tolist()
    metadata_path.write_text(json.dumps(metadata, indent=4))
    if normalization is not None:
        (output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")


def projection_signature(panorama_path, scene_dir, shape, fov, res, depth_dir_name="depth"):
    # Resume only if the source image, poses, normalization, and projection
    # settings still match. In particular, never reuse faces of an old pose.
    signature = {"shape": shape, "fov": fov, "res": res}
    for path in (panorama_path, scene_dir / depth_dir_name / panorama_path.name,
                 scene_dir / "extrinsics.json", scene_dir / "normalization.json"):
        if path.is_file():
            stat = path.stat()
            signature[str(path.relative_to(scene_dir))] = [stat.st_size, stat.st_mtime_ns]
    render_marker = scene_dir / ".complete.json"
    if render_marker.is_file():
        settings = json.loads(render_marker.read_text()).get("render_settings")
        if settings is not None:
            signature["render_settings"] = settings
    return signature


def process_panorama(panorama_path, scene_dir, output_root, args):
    relative_scene = scene_dir.relative_to(args.input_root)
    output_dir = (
        output_root
        / relative_scene
        / f"{panorama_path.stem}_{args.shape}_{int(args.fov)}"
    )
    depth_path = scene_dir / args.depth_dir_name / panorama_path.name
    extrinsics_path = scene_dir / "extrinsics.json"
    normalization_path = scene_dir / "normalization.json"
    signature = projection_signature(
        panorama_path, scene_dir, args.shape, args.fov, args.res, args.depth_dir_name,
    )
    marker = output_dir / ".complete.json"
    try:
        matching_signature = json.loads(marker.read_text()) == signature
    except (OSError, ValueError):
        matching_signature = False
    if args.require_depth and not depth_path.is_file():
        raise FileNotFoundError(
            f"No rendered depth for {panorama_path}; expected {depth_path}"
        )
    if not args.overwrite and matching_signature and output_is_complete(
        output_dir, require_depth=depth_path.is_file()
    ) and (not normalization_path.is_file() or (output_dir / "normalization.json").is_file()):
        print(f"Skipping complete panorama: {panorama_path}")
        return
    marker.unlink(missing_ok=True)

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

    if extrinsics_path.is_file():
        panorama_extrinsics = json.loads(extrinsics_path.read_text())
        if panorama_path.name not in panorama_extrinsics:
            raise KeyError(f"{panorama_path.name} is missing from {extrinsics_path}")
        add_world_extrinsics(
            output_dir, panorama_extrinsics[panorama_path.name],
            json.loads(normalization_path.read_text()) if normalization_path.is_file() else None,
            args.dataset_layout,
        )
    if not output_is_complete(output_dir, require_depth=depth_path.is_file()):
        raise RuntimeError(f"Cubemap generator left incomplete output: {output_dir}")
    marker.write_text(json.dumps(signature, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cubemap faces for rendered SAGE-10K or 3D-FRONT panoramas."
    )
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Root containing SAGE scene directories"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-layout", choices=("sage10k", "3dfront"), default="sage10k"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="Process at most this many panoramas")
    parser.add_argument(
        "--sample-list",
        type=Path,
        help=(
            "Optional JSON manifest. SAGE entries are 'SCENE FRAME'; 3D-FRONT "
            "entries are panorama paths relative to --input-root."
        ),
    )
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
    if args.sample_list is not None:
        entries = json.loads(args.sample_list.read_text())
        panoramas = []
        for entry in entries:
            if args.dataset_layout == "sage10k":
                scene, frame = entry.split()
                path = args.input_root / scene / f"{scene}_panorama_{int(frame):03d}.png"
            else:
                path = args.input_root / entry
            if not path.is_file():
                parser.error(f"manifest panorama does not exist: {path}")
            panoramas.append(path)
    else:
        panoramas = sorted(args.input_root.rglob("*_panorama_*.png"))
        panoramas = [
            path for path in panoramas
            if path.is_file() and not any(part in {"depth", "logs"} for part in path.parts)
        ]
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
