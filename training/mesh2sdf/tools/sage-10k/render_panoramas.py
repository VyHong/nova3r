#!/usr/bin/env python3
"""Render collision-safe equirectangular panoramas from SAGE-10k scenes.

The input can be one extracted layout JSON, one scene ZIP, or a directory of
scene ZIPs.  Directory runs are resumable: a scene is skipped only after its
completion marker and all expected images are present.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import traceback
import zipfile

import numpy as np
import trimesh
from tqdm import tqdm

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "Blender's Python module is required. Install kits/requirements-panorama.txt "
        "in the nova3r environment."
    ) from exc

# Reuse the scene loading, lighting, and collision implementation used by the
# video renderer rather than maintaining a subtly different safety model.
from render_video import (  # noqa: E402
    CAMERA_RADIUS,
    CameraPlannerEnv,
    clear_blender_scene,
    get_room_meshes,
    load_scene_meshes_into_blender,
    setup_camera_look_at,
    setup_scene_lighting,
)
from tex_utils_local import dict_to_floor_plan  # noqa: E402
from glb_utils import SAGE_TO_GLTF  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 10 safe equirectangular panoramas per SAGE scene"
    )
    parser.add_argument("input", help="Layout JSON, scene ZIP, or ZIP directory")
    parser.add_argument("--output-dir", default="panoramas")
    parser.add_argument("--num-panoramas", type=int, default=10)
    parser.add_argument("--room-id", help="Room to render (default: first room)")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=0.02,
        help="Cycles adaptive sampling noise threshold (higher is faster)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--camera-height",
        type=float,
        help="Use one fixed camera height, overriding the min/max range",
    )
    parser.add_argument(
        "--camera-height-min",
        type=float,
        default=1.6,
        help="Minimum camera height above the room floor (default: 1.6)",
    )
    parser.add_argument(
        "--camera-height-max",
        type=float,
        default=1.8,
        help="Maximum camera height above the room floor (default: 1.8)",
    )
    parser.add_argument("--min-separation", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--device", choices=("AUTO", "CPU", "OPTIX", "CUDA"), default="AUTO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument(
        "--save-scene-ply",
        action="store_true",
        help="Also export merged scene geometry (debugging; disabled by default)",
    )
    parser.add_argument(
        "--save-scene-glb",
        action="store_true",
        help="Export the rendered layout as scene.glb beside the panoramas",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many scenes")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--allow-scene-failures",
        action="store_true",
        help=(
            "Exit successfully after processing every scene even if individual "
            "scenes failed (failures are still logged)"
        ),
    )
    if "--" in sys.argv:
        return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    return parser.parse_args()


def scene_inputs(path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(
            f"Input does not exist: {path}. Check that the dataset path or "
            "shell variable used to construct it is set."
        )
    if path.is_dir():
        inputs = sorted(path.glob("*.zip"))
        if not inputs:
            raise ValueError(f"Input directory contains no scene ZIP files: {path}")
        return inputs
    if path.suffix.lower() in {".zip", ".json"}:
        return [path]
    raise ValueError(f"Input must be a layout JSON, scene ZIP, or ZIP directory: {path}")


def safe_extract(archive, destination):
    """Extract without allowing absolute paths, traversal, or symlink entries."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            mode = info.external_attr >> 16
            if (mode & 0o170000) == 0o120000:
                raise ValueError(f"Symlink archive member is not allowed: {info.filename}")
        zf.extractall(root)


def complete(scene_dir, count, require_scene_glb=False, camera_height_range=(1.6, 1.8)):
    marker = scene_dir / ".complete.json"
    extrinsics_path = scene_dir / "extrinsics.json"
    if not marker.is_file() or not extrinsics_path.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
        extrinsics = json.loads(extrinsics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_files = {f"{scene_dir.name}_panorama_{i:03d}.png" for i in range(count)}
    return (
        data.get("num_panoramas") == count
        and data.get("camera_height_range") == list(camera_height_range)
        and set(extrinsics) == expected_files
        and (
            not require_scene_glb
            or (
                (scene_dir / "scene.glb").is_file()
                and data.get("world_coordinate_system") == "gltf_y_up"
            )
        )
        and all(
        (scene_dir / filename).is_file() for filename in expected_files
        )
    )


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def choose_positions(
    env, count, rng, min_separation, max_attempts, camera_height_range
):
    """Rejection-sample positions with the video renderer's collision margin."""
    lower = env.bounds[:3] + CAMERA_RADIUS
    upper = env.bounds[3:] - CAMERA_RADIUS
    if np.any(lower >= upper):
        raise RuntimeError("Room is too small for the camera safety radius")
    height_min, height_max = camera_height_range
    camera_z_min = env.bounds[2] + height_min
    camera_z_max = env.bounds[2] + height_max
    if camera_z_min < lower[2] or camera_z_max > upper[2]:
        safe_min = lower[2] - env.bounds[2]
        safe_max = upper[2] - env.bounds[2]
        raise RuntimeError(
            f"Camera height range [{height_min:.3f}, {height_max:.3f}] m is "
            f"outside the room's safe "
            f"range [{safe_min:.3f}, {safe_max:.3f}] m"
        )

    positions = []
    attempts = 0
    separation = min_separation
    # Relax only diversity, never the collision or room-boundary safeguards.
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(lower, upper)
        candidate[2] = env.bounds[2] + rng.uniform(height_min, height_max)
        if not env.is_valid_location(candidate, min_dist=CAMERA_RADIUS):
            continue
        if positions and min(np.linalg.norm(candidate - p) for p in positions) < separation:
            if attempts % max(100, max_attempts // 5) == 0:
                separation *= 0.75
            continue
        positions.append(candidate)
    if len(positions) != count:
        raise RuntimeError(
            f"Found only {len(positions)}/{count} collision-safe positions after {attempts} attempts"
        )
    return positions


def configure_cycles(scene, requested_device, samples, adaptive_threshold):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = adaptive_threshold
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 1
    # Geometry and materials remain fixed for all panoramas in a scene.
    scene.render.use_persistent_data = True

    if requested_device == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    choices = ("OPTIX", "CUDA") if requested_device == "AUTO" else (requested_device,)
    for backend in choices:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            usable = [d for d in prefs.devices if d.type == backend]
            if usable:
                for device in prefs.devices:
                    device.use = device in usable
                scene.cycles.device = "GPU"
                return backend
        except Exception as exc:
            print(f"Warning: {backend} unavailable ({exc}); trying fallback", file=sys.stderr)
    scene.cycles.device = "CPU"
    return "CPU"


def render_scene(layout_path, scene_dir, args, scene_seed):
    with layout_path.open() as f:
        layout = dict_to_floor_plan(json.load(f))
    room = next((r for r in layout.rooms if r.id == args.room_id), None) if args.room_id else layout.rooms[0]
    if room is None:
        raise ValueError(f"Room {args.room_id!r} was not found")

    bounds = np.array([
        room.position.x, room.position.y, room.position.z,
        room.position.x + room.dimensions.width,
        room.position.y + room.dimensions.length,
        room.position.z + room.dimensions.height,
    ])
    all_meshes, _, mesh_info_dict = get_room_meshes(layout, str(layout_path.parent))
    if not all_meshes:
        raise RuntimeError("Scene contains no exportable meshes")
    if args.save_scene_ply:
        scene_ply = scene_dir / "scene.ply"
        scene_ply_tmp = scene_dir / ".scene.tmp.ply"
        trimesh.util.concatenate(all_meshes).export(scene_ply_tmp, file_type="ply")
        if not scene_ply_tmp.is_file() or scene_ply_tmp.stat().st_size == 0:
            raise RuntimeError(f"Failed to export scene geometry to {scene_ply_tmp}")
        os.replace(scene_ply_tmp, scene_ply)
    env = CameraPlannerEnv(
        bounds,
        all_meshes,
        collision_samples=100000,
        build_ray_intersector=False,
    )
    rng = np.random.default_rng(scene_seed)
    positions = choose_positions(
        env,
        args.num_panoramas,
        rng,
        args.min_separation,
        args.max_attempts,
        args.camera_height_range,
    )

    load_scene_meshes_into_blender(
        room, layout, str(layout_path.parent), mesh_info_dict=mesh_info_dict
    )
    scene = bpy.context.scene
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "PanoramaCamera"
    camera.data.type = "PANO"
    if hasattr(camera.data, "panorama_type"):  # Blender 5+
        camera.data.panorama_type = "EQUIRECTANGULAR"
    else:
        camera.data.cycles.panorama_type = "EQUIRECTANGULAR"
    scene.camera = camera
    backend = configure_cycles(
        scene, args.device, args.samples, args.adaptive_threshold
    )
    setup_scene_lighting(scene, {
        "width": room.dimensions.width,
        "length": room.dimensions.length,
        "height": room.dimensions.height,
    })
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    # One unified mapping with exactly one W2C extrinsic per image. The scene
    # ID in every filename preserves provenance if images are moved elsewhere.
    extrinsics = {}
    base_yaw = rng.uniform(0.0, 2.0 * np.pi)
    for index, position in enumerate(positions):
        final_path = scene_dir / f"{scene_dir.name}_panorama_{index:03d}.png"
        tmp_path = scene_dir / f".{scene_dir.name}_panorama_{index:03d}.tmp.png"
        # Match render_video.py's convention: camera-local -Z is forward and
        # local +Y is up. Evenly spaced yaw headings guarantee that panorama
        # seams progress horizontally from left to right across the set.
        yaw = base_yaw + (2.0 * np.pi * index / len(positions))
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        setup_camera_look_at(camera, position, position + heading)
        bpy.context.view_layer.update()
        world_to_camera = np.linalg.inv(np.asarray(camera.matrix_world, dtype=np.float64))
        if args.save_scene_glb:
            # p_gltf = SAGE_TO_GLTF @ p_sage, hence
            # p_camera = W2C_sage @ inv(SAGE_TO_GLTF) @ p_gltf.
            world_to_camera = world_to_camera @ SAGE_TO_GLTF.T
        scene.render.filepath = str(tmp_path)
        print(f"Rendering {final_path.name} at {position.tolist()} ({backend})")
        bpy.ops.render.render(write_still=True)
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Renderer did not create {tmp_path}")
        os.replace(tmp_path, final_path)
        extrinsics[final_path.name] = world_to_camera.tolist()
        atomic_json(scene_dir / "extrinsics.json", extrinsics)
    atomic_json(scene_dir / ".complete.json", {
        "num_panoramas": args.num_panoramas,
        "room_id": room.id,
        "camera_height_range": list(args.camera_height_range),
        "seed": scene_seed,
        "world_coordinate_system": "gltf_y_up" if args.save_scene_glb else "sage_z_up",
        "extrinsics_convention": "row_major_world_to_camera_blender_axes",
    })
    clear_blender_scene()


def main():
    args = parse_args()
    if not args.output_dir.strip():
        raise SystemExit(
            "--output-dir cannot be empty; check that its shell variable is set"
        )
    if args.num_panoramas < 1 or args.width < 1 or args.height < 1 or args.samples < 1:
        raise SystemExit("Counts, dimensions, and samples must be positive")
    if args.adaptive_threshold <= 0:
        raise SystemExit("--adaptive-threshold must be positive")
    if args.camera_height is not None:
        args.camera_height_min = args.camera_height
        args.camera_height_max = args.camera_height
    if args.camera_height_min <= 0 or args.camera_height_max < args.camera_height_min:
        raise SystemExit(
            "Camera heights must satisfy 0 < --camera-height-min <= --camera-height-max"
        )
    args.camera_height_range = (args.camera_height_min, args.camera_height_max)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require --shard-count >= 1 and 0 <= --shard-index < --shard-count")
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        inputs = scene_inputs(args.input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    inputs = inputs[args.shard_index :: args.shard_count]
    if args.limit is not None:
        inputs = inputs[: args.limit]

    failures = []
    shard_label = f"{args.shard_index + 1}/{args.shard_count}"
    progress = tqdm(
        inputs,
        desc=f"Scenes (shard {shard_label})",
        unit="scene",
        dynamic_ncols=True,
    )
    for index, source in enumerate(progress, 1):
        scene_name = source.stem
        progress.set_postfix_str(scene_name, refresh=False)
        scene_dir = output_root / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and complete(
            scene_dir,
            args.num_panoramas,
            require_scene_glb=args.save_scene_glb,
            camera_height_range=args.camera_height_range,
        ):
            print(f"[{index}/{len(inputs)}] Skipping complete scene {scene_name}")
            continue
        if args.overwrite:
            for old in scene_dir.glob("*panorama_*.png"):
                old.unlink()
            for old in (
                scene_dir / "metadata.json",
                scene_dir / "extrinsics.json",
                scene_dir / "scene.ply",
                scene_dir / "scene.glb",
                scene_dir / ".complete.json",
            ):
                old.unlink(missing_ok=True)

        work_dir = None
        try:
            print(f"[{index}/{len(inputs)}] Processing {source.name}")
            if source.suffix.lower() == ".zip":
                if args.keep_extracted:
                    work_dir = output_root / "_extracted" / scene_name
                    work_dir.mkdir(parents=True, exist_ok=True)
                    safe_extract(source, work_dir)
                else:
                    work_dir = Path(tempfile.mkdtemp(prefix=f"sage-{scene_name}-"))
                    safe_extract(source, work_dir)
                layouts = sorted(work_dir.glob("*.json"))
                if len(layouts) != 1:
                    raise RuntimeError(f"Expected one root layout JSON, found {len(layouts)}")
                layout_path = layouts[0]
            else:
                layout_path = source
            if args.save_scene_glb:
                from export_glb import export_glb

                export_glb(str(layout_path), str(scene_dir), output_filename="scene.glb")
            digest = hashlib.sha256(scene_name.encode()).digest()
            scene_seed = args.seed + int.from_bytes(digest[:4], "little")
            random.seed(scene_seed)
            np.random.seed(scene_seed & 0xFFFFFFFF)
            render_scene(layout_path, scene_dir, args, scene_seed)
            (scene_dir / "failure.log").unlink(missing_ok=True)
        except Exception as exc:
            failures.append((str(source), str(exc)))
            progress.set_postfix(failed=len(failures))
            (scene_dir / "failure.log").write_text(traceback.format_exc())
            print(f"ERROR: {source}: {exc}", file=sys.stderr)
        finally:
            if work_dir and not args.keep_extracted:
                shutil.rmtree(work_dir, ignore_errors=True)
            clear_blender_scene()

    if failures:
        print(f"Completed with {len(failures)} failed scene(s):", file=sys.stderr)
        for source, error in failures:
            print(f"  {source}: {error}", file=sys.stderr)
        return 0 if args.allow_scene_failures else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
