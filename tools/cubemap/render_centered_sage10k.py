#!/usr/bin/env python3
"""Render one centered panorama and RGB cubemap per distinct listed scene.

Keep the panorama, six cubemap faces, and camera metadata. No surface files or
median normalization are needed.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

from sage10k import output_is_complete, projection_signature


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "training/mesh2sdf/tools/sage-10k/render_panoramas_batched.sh"
CONVERTER = Path(__file__).with_name("sage10k.py")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_scenes(path):
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Scene list must be a nonempty JSON list")
    scenes = []
    seen = set()
    for entry in entries:
        fields = entry.split() if isinstance(entry, str) else []
        if (len(fields) != 2 or not fields[1].isdigit()
                or fields[0] in {".", ".."} or "/" in fields[0] or "\\" in fields[0]):
            raise ValueError(f"Expected 'SCENE FRAME' entry, got {entry!r}")
        if fields[0] not in seen:
            scenes.append(fields[0])
            seen.add(fields[0])
    return scenes, len(entries)


def run(command):
    print("+ " + shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=False).returncode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path,
                        default=REPO_ROOT / "data/sage-10k/surface_vae_latent_list.json")
    parser.add_argument("--scene-root", type=Path, default=Path("/tmp/datasets/sage-10k/scenes"))
    parser.add_argument("--surface-root", type=Path, help="Deprecated; accepted for old commands but unused")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-env", default="blender", help="Blender rendering environment")
    parser.add_argument("--projection-env", default="nova3r", help="Environment with cv2 and utils3d")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4,
                        help="Total concurrent rendering processes across all selected GPUs")
    parser.add_argument("--gpus", default="0,1",
                        help="Comma-separated rendering GPU IDs (default: 0,1); batches alternate across GPUs")
    parser.add_argument("--device", choices=("AUTO", "CPU", "OPTIX", "CUDA"), default="OPTIX")
    parser.add_argument("--workers", type=int, default=4, help="Cubemap conversion workers")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--res", type=int, default=518, help="Cubemap face size; FOV is fixed at 95 degrees")
    parser.add_argument("--camera-height", type=float,
                        help="Override half-height with this many meters above the floor")
    parser.add_argument("--limit", type=int, help="First N unique scenes in list order")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate scene coverage without rendering")
    parser.add_argument("--convert-only", action="store_true",
                        help="Convert existing panoramas and clean intermediates without launching Blender")
    args = parser.parse_args()
    for key in ("batch_size", "jobs", "workers", "width", "height", "samples", "res", "limit"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.width != 2 * args.height:
        parser.error("Panorama width must be twice its height")
    if args.camera_height is not None and (not np.isfinite(args.camera_height) or args.camera_height <= 0):
        parser.error("--camera-height must be finite and positive")
    return args


def render_settings(args):
    return {
        "camera_position": "center", "camera_height": args.camera_height,
        "world_coordinate_system": "gltf_y_up", "width": args.width,
        "height": args.height, "samples": args.samples,
        "adaptive_threshold": 0.02, "seed": 42, "room_id": None,
    }


def cube_path(root, scene):
    return root / "cubemaps" / scene / f"{scene}_panorama_000_cube_95"


def cubemap_complete(root, scene, args):
    """Resume from faces even after their source panorama has been deleted."""
    cube = cube_path(root, scene)
    if not output_is_complete(cube):
        return False
    try:
        marker = json.loads((cube / ".complete.json").read_text())
        metadata = json.loads((cube / "camera_data.json").read_text())
        if set(metadata) != {f"{i:04d}" for i in range(6)} or not all(
            "world_extrinsics" in face for face in metadata.values()
        ):
            return False
        if any(marker.get(k) != v for k, v in {"shape": "cube", "fov": 95, "res": args.res}.items()):
            return False
        if marker.get("render_settings") == render_settings(args):
            return True
        # Reuse RGB from the older normalized workflow without reprojecting it.
        # Normalization only added metadata; the original images are identical.
        pano = root / "panoramas" / scene
        source_marker = json.loads((pano / ".complete.json").read_text())
        if source_marker.get("render_settings") != render_settings(args):
            return False
        signature = projection_signature(pano / f"{scene}_panorama_000.png", pano, "cube", 95, args.res)
        return all(marker.get(k) == v for k, v in signature.items()
                   if k not in {"normalization.json", "render_settings"})
    except (OSError, ValueError, TypeError):
        return False


def cleanup_completed(root, scene, args):
    """Remove obsolete normalization metadata while retaining RGB and poses."""
    cube = cube_path(root, scene)
    metadata_path = cube / "camera_data.json"
    metadata = json.loads(metadata_path.read_text())
    for face in metadata.values():
        face.pop("normalized_world_extrinsics", None)
    atomic_json(metadata_path, metadata)
    marker_path = cube / ".complete.json"
    marker = json.loads(marker_path.read_text())
    marker["render_settings"] = render_settings(args)
    marker.pop("normalization.json", None)
    atomic_json(marker_path, marker)
    (cube / "normalization.json").unlink(missing_ok=True)
    pano = root / "panoramas" / scene
    (pano / "normalization.json").unlink(missing_ok=True)


def main():
    args = parse_args()
    scenes, entry_count = load_scenes(args.scene_list)
    print(f"{entry_count} list entries -> {len(scenes)} unique scenes", flush=True)
    if args.limit is not None:
        scenes = scenes[:args.limit]
    args.scene_root = args.scene_root.resolve()
    root = args.output_root.resolve()
    ready = {scene for scene in scenes if not args.overwrite and cubemap_complete(root, scene, args)}
    pending = [scene for scene in scenes if scene not in ready]
    if not args.convert_only:
        missing = [str(args.scene_root / f"{scene}.zip") for scene in pending
                   if not (args.scene_root / f"{scene}.zip").is_file()]
        if missing:
            raise SystemExit(f"Missing {len(missing)} scene ZIPs; first 20:\n" + "\n".join(missing[:20]))
    print(f"{len(ready)} complete cubemaps; {len(pending)} scenes pending. Output: {root}", flush=True)
    if args.dry_run:
        return 0
    environments = []
    if pending:
        environments.append((args.projection_env, "cv2, utils3d, numpy"))
        if not args.convert_only:
            environments.append((args.conda_env, "bpy, numpy, trimesh, open3d"))
    for env, modules in environments:
        if run(["conda", "run", "-n", env, "python", "-c", f"import {modules}"]):
            raise SystemExit(f"Required dependencies are unavailable in conda environment {env!r}")

    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "scene_list.json", [f"{scene} 00000" for scene in scenes])
    failures = {}
    process_failed = False
    with tempfile.TemporaryDirectory(prefix="sage-centered-") as temporary:
        selected = Path(temporary) / "scenes"
        selected.mkdir()
        if pending and not args.convert_only:
            for scene in pending:
                (selected / f"{scene}.zip").symlink_to(args.scene_root / f"{scene}.zip")
            command = [
                "bash", LAUNCHER, selected, root / "panoramas",
                "--batch-size", args.batch_size, "--jobs", args.jobs,
                "--gpus", args.gpus, "--conda-env", args.conda_env, "--",
                "--num-panoramas", 1, "--camera-position", "center",
                "--world-coordinate-system", "gltf_y_up", "--device", args.device,
                "--width", args.width, "--height", args.height, "--samples", args.samples,
            ]
            if args.camera_height is not None:
                command.extend(["--camera-height", args.camera_height])
            if args.overwrite:
                command.append("--overwrite")
            status = run(command)
            process_failed = status != 0
            if process_failed:
                print(f"Render launcher exited with status {status}; inspect {root / 'panoramas/logs-batched'}", flush=True)

        projection_input = Path(temporary) / "panoramas"
        projection_input.mkdir()
        for scene in pending:
            pano = root / "panoramas" / scene
            try:
                marker = json.loads((pano / ".complete.json").read_text())
                if (marker.get("num_panoramas") != 1
                        or marker.get("world_coordinate_system") != "gltf_y_up"
                        or marker.get("render_settings") != render_settings(args)
                        or not (pano / f"{scene}_panorama_000.png").is_file()):
                    raise ValueError("No complete centered panorama with matching settings")
                staged = projection_input / scene
                staged.mkdir()
                for filename in (f"{scene}_panorama_000.png", "extrinsics.json", ".complete.json"):
                    (staged / filename).symlink_to(pano / filename)
            except Exception as error:
                failures[scene] = f"{error}; inspect {pano / 'failure.log'} and batch logs"
        if len(failures) < len(pending):
            command = [
                "conda", "run", "--no-capture-output", "-n", args.projection_env,
                "python", CONVERTER, "--input-root", projection_input,
                "--output-root", root / "cubemaps", "--shape", "cube",
                "--fov", 95, "--res", args.res, "--workers", args.workers,
            ]
            if args.overwrite:
                command.append("--overwrite")
            process_failed = (run(command) != 0) or process_failed

    completed = []
    for scene in scenes:
        if scene in failures:
            continue
        if not cubemap_complete(root, scene, args):
            failures[scene] = "Incomplete cubemap; intermediate panorama retained for retry"
            continue
        try:
            cleanup_completed(root, scene, args)
            completed.append(f"{scene} 00000")
        except Exception as error:
            failures[scene] = f"Cubemap created but cleanup failed: {error}"
    atomic_json(root / "completed_list.json", completed)
    atomic_json(root / "failures.json", failures)
    print(f"Completed {len(completed)}/{len(scenes)} cubemaps. Failures: {root / 'failures.json'}")
    return 1 if failures or process_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
