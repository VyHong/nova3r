#!/usr/bin/env python3
"""Render exterior/opening masks at existing 3D-FRONT panorama poses.

Each room must contain ``extrinsics.json`` beside its rendered panoramas.  The
mask follows the ScanNet++ convention: white means valid retained room
geometry; black means that the ray escaped through an opening or hit an object
identified as window/glass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import traceback

import bpy
from mathutils import Matrix
from PIL import Image


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("panorama_root", type=Path)
    parser.add_argument("--mask-dir", default="anon_mask")
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Writable root for masks when panorama_root is read-only. The output "
            "keeps the SCENE/ROOM/anon_mask layout. By default masks are written "
            "inside each panorama room."
        ),
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require --shard-count >= 1 and 0 <= --shard-index < --shard-count")
    return args


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def png_size(path: Path):
    with path.open("rb") as file:
        header = file.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def emission_material(name, value):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (value, value, value, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def configure_scene(width, height):
    scene = bpy.context.scene
    # EEVEE does not reliably honor panoramic cameras in headless mode.  A
    # one-sample emission-only Cycles pass is still cheap and reproduces the
    # exact equirectangular camera used for the source renders.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    # Blender's headless BW writer may encode scene-linear 1.0 as integer 1;
    # RGB reliably maps the binary render to 0/255 and consumers already load
    # anonymization masks as grayscale.
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    world = bpy.data.worlds.new("MaskWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0, 0, 0, 1)
    background.inputs["Strength"].default_value = 1
    scene.world = world
    return scene


def render_room(
    scene_root: Path,
    panorama_root: Path,
    room_dir: Path,
    mask_dir_name: str,
    output_root: Path | None,
    overwrite: bool,
):
    poses = json.loads((room_dir / "extrinsics.json").read_text())
    if not poses:
        return 0
    first_image = room_dir / next(iter(poses))
    width, height = png_size(first_image)
    relative = room_dir.relative_to(room_dir.parents[1])
    mesh = scene_root / relative.parent / f"{relative.name}_full.glb"
    if not mesh.is_file():
        raise FileNotFoundError(mesh)

    relative_room = room_dir.relative_to(panorama_root)
    output_dir = (
        output_root / relative_room / mask_dir_name
        if output_root is not None
        else room_dir / mask_dir_name
    )
    pending = [(name, matrix) for name, matrix in poses.items()
               if overwrite or not (output_dir / name).is_file()]
    if not pending:
        return 0

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(mesh), merge_vertices=True)
    interior = emission_material("InteriorMask", 1.0)
    transparent_keywords = ("window", "glass", "pane", "glazing")
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        labels = " ".join([obj.name, *(m.name for m in obj.data.materials if m)]).lower()
        if any(keyword in labels for keyword in transparent_keywords):
            obj.hide_render = True
        else:
            obj.data.materials.clear()
            obj.data.materials.append(interior)

    scene = configure_scene(width, height)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "PANO"
    if hasattr(camera.data, "panorama_type"):
        camera.data.panorama_type = "EQUIRECTANGULAR"
    else:
        camera.data.cycles.panorama_type = "EQUIRECTANGULAR"
    scene.camera = camera
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, world_to_camera in pending:
        camera.matrix_world = Matrix(world_to_camera).inverted()
        final = output_dir / name
        temporary = output_dir / f".{name}.tmp.png"
        scene.render.filepath = str(temporary)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Blender did not create {temporary}")
        # Color management can encode black background pixels as either 0 or
        # 1. Canonicalize the render so downstream code receives a genuinely
        # binary grayscale mask: valid geometry=255, outside/opening=0.
        with Image.open(temporary) as rendered:
            binary = rendered.convert("L").point(lambda value: 255 if value >= 128 else 0)
            binary.save(temporary)
        os.replace(temporary, final)
    return len(pending)


def main():
    args = arguments()
    rooms = sorted(path.parent for path in args.panorama_root.glob("*/*/extrinsics.json"))
    rooms = rooms[args.shard_index::args.shard_count]
    if args.limit is not None:
        rooms = rooms[:args.limit]
    completed = failed = 0
    for index, room in enumerate(rooms, 1):
        try:
            count = render_room(
                args.scene_root,
                args.panorama_root,
                room,
                args.mask_dir,
                args.output_root,
                args.overwrite,
            )
            completed += count
            print(f"[{index}/{len(rooms)}] {room}: rendered {count}", flush=True)
        except Exception as error:
            failed += 1
            print(f"[{index}/{len(rooms)}] Failed {room}: {error}", file=sys.stderr, flush=True)
            traceback.print_exc()
    print(f"Rendered {completed} masks; {failed} rooms failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
