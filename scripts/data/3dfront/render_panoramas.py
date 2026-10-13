#!/usr/bin/env python3
"""Render equirectangular RGB panoramas from room-level 3D-FRONT GLBs.

Run with Blender's Python (or a Python environment which provides ``bpy``):

    blender -b --python render_panoramas.py -- DATA_ROOT OUTPUT_ROOT

``DATA_ROOT`` may also be one GLB.  Dataset directories are searched
recursively for ``*_full.glb``; in the released scene layout, each UUID
directory is a dwelling/scene and each GLB within it is one room.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One room GLB or a 3D-FRONT scene root")
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-panoramas", type=int, default=10)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--threads", type=int,
        help="Fixed Cycles CPU thread count (also use taskset to constrain native libraries)",
    )
    parser.add_argument(
        "--light-energy", type=float, default=80.0,
        help="Ceiling fill-light energy in watts (default: 80 for normalized rooms)",
    )
    parser.add_argument(
        "--lamp-energy", type=float, default=35.0,
        help="Base energy for lights placed inside detected Lighting_* fixtures (default: 35)",
    )
    parser.add_argument(
        "--minimum-fill-energy", type=float, default=160.0,
        help=("Minimum ceiling fill-light energy when no Lighting_* fixture is detected "
              "(default: 160)"),
    )
    parser.add_argument(
        "--camera-height", type=float,
        help="Absolute height above the room floor (default: use --camera-height-ratio)",
    )
    parser.add_argument(
        "--camera-height-ratio", type=float, default=0.58,
        help="Camera height as a fraction of room height for normalized GLBs (default: 0.58)",
    )
    parser.add_argument(
        "--clearance", type=float, default=0.2,
        help=("Minimum camera-to-surface distance in normalized scene units "
              "(default: 0.2; normalized-scene equivalent of SAGE-10K's "
              "0.5-meter camera safety radius)"),
    )
    parser.add_argument("--min-separation", type=float, default=0.35)
    parser.add_argument(
        "--camera-poses", type=Path,
        help=("Optional extrinsics.json whose world-to-camera matrices fix the camera "
              "positions instead of sampling new positions"),
    )
    parser.add_argument(
        "--max-attempts", type=int, default=100000,
        help="Maximum camera-placement attempts (default: 100000, as in SAGE-10K)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture-material", choices=("original", "random-realistic", "ambientcg"),
        default="original",
        help="Keep GLB materials, generate procedural materials, or use ambientCG PBR materials",
    )
    parser.add_argument(
        "--pbr-material-root", type=Path,
        help="Directory produced by 'blenderproc download cc_textures' (required for ambientcg)",
    )
    parser.add_argument(
        "--exterior-ground", choices=("flat", "pbr-plane", "none"), default="flat",
        help="Exterior below the sky horizon; pbr-plane uses an ambientCG outdoor material",
    )
    parser.add_argument(
        "--background", choices=("sky", "black", "world"), default="sky",
        help="Exterior/background model; sky is randomized deterministically per room (default: sky)",
    )
    parser.add_argument(
        "--time-of-day", choices=("day", "night", "random"), default="random",
        help="Sky and lighting period; random selects night for about 20%% of rooms",
    )
    parser.add_argument("--device", choices=("AUTO", "CPU", "CUDA", "OPTIX"), default="AUTO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Skip this many rooms in the globally sorted input list (default: 0)",
    )
    parser.add_argument(
        "--target-complete", type=int,
        help="Stop once OUTPUT contains this many complete rooms",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def clear_scene() -> None:
    # The anonymization pass uses a view-layer material override. Clear it
    # before importing the next room or its RGB render would be solid white.
    for view_layer in bpy.context.scene.view_layers:
        view_layer.material_override = None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                       bpy.data.cameras, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def inputs(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file() and path.suffix.lower() == ".glb":
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input is neither a GLB nor a directory: {path}")
    rooms = sorted(path.rglob("*_full.glb"))
    if not rooms:
        raise ValueError(f"No *_full.glb room meshes found below {path}")
    return rooms


def output_dir(mesh: Path, input_root: Path, root: Path) -> Path:
    # Preserve the dwelling UUID and room name, avoiding collisions between
    # common names such as Bedroom-1234 in different scene directories.
    if input_root.is_dir():
        relative = mesh.relative_to(input_root)
        return root / relative.parent / mesh.stem.removesuffix("_full")
    return root / mesh.stem.removesuffix("_full")


def mesh_bounds() -> tuple[Vector, Vector]:
    corners = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        raise RuntimeError("Imported GLB contains no mesh objects")
    return (Vector(tuple(min(p[i] for p in corners) for i in range(3))),
            Vector(tuple(max(p[i] for p in corners) for i in range(3))))


def cast(origin: Vector, direction: Vector, distance: float):
    return bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(), origin, direction.normalized(),
        distance=distance,
    )


def scene_surface_bvh() -> BVHTree:
    """Build one exact nearest-surface index over evaluated scene geometry."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    polygons = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            offset = len(vertices)
            vertices.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
            polygons.extend(
                tuple(offset + index for index in polygon.vertices)
                for polygon in mesh.polygons
                if len(polygon.vertices) >= 3
            )
        finally:
            evaluated.to_mesh_clear()
    if not polygons:
        raise RuntimeError("Imported GLB contains no polygonal collision geometry")
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)


def valid_position(position: Vector, clearance: float, surface_bvh: BVHTree) -> bool:
    # Match SAGE-10K's nearest-surface safety model. An exact BVH query is used
    # instead of its 100k-point KD tree, avoiding gaps between sampled points.
    # The downward ray additionally rejects AABB points outside the room.
    down_hit, _, _, _, _, _ = cast(position, Vector((0, 0, -1)), 3.0)
    if not down_hit:
        return False
    nearest = surface_bvh.find_nearest(position)
    return nearest is None or nearest[3] >= clearance


def camera_positions(bounds, count, height, clearance, separation, attempts, rng,
                     surface_bvh):
    low, high = bounds
    margin = clearance
    if any(high[axis] - low[axis] <= 2 * margin for axis in range(3)):
        raise RuntimeError("Room bounds are too small for the requested clearance")
    camera_z = low.z + height
    if not low.z + margin < camera_z < high.z - margin:
        raise RuntimeError(
            f"Camera height {height:.4f} places z={camera_z:.4f} outside the "
            f"collision-safe range ({low.z + margin:.4f}, {high.z - margin:.4f})"
        )
    positions = []
    current_separation = separation
    progress_interval = max(100, attempts // 5)
    for attempt in range(attempts):
        if attempt and attempt % progress_interval == 0:
            current_separation *= 0.75
            print(
                f"Camera sampling: {len(positions)}/{count} positions after "
                f"{attempt}/{attempts} attempts; relaxing separation "
                f"to {current_separation:.4f}",
                flush=True,
            )
        x = rng.uniform(low.x + margin, high.x - margin)
        y = rng.uniform(low.y + margin, high.y - margin)
        # Architectural room meshes use the global minimum Z as their floor.
        # Starting at eye height avoids mistaking the ceiling (or a tabletop)
        # for the floor, which happens when casting down from above the room.
        position = Vector((x, y, camera_z))
        if not valid_position(position, clearance, surface_bvh):
            continue
        if any((position - other).length < current_separation for other in positions):
            continue
        positions.append(position)
        if len(positions) == count:
            print(
                f"Camera sampling: found {count}/{count} positions in "
                f"{attempt + 1} attempts (final separation "
                f"{current_separation:.4f})",
                flush=True,
            )
            return positions
    raise RuntimeError(f"Found only {len(positions)}/{count} safe camera positions in {attempts} attempts")


def configure_cycles(scene, requested: str, samples: int, threads: int | None) -> str:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.max_bounces = 3
    # Restore settings changed by the preceding room's one-sample mask pass.
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.transparent_max_bounces = 3
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    if threads is not None:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = threads
    if requested == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in (("OPTIX", "CUDA") if requested == "AUTO" else (requested,)):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            devices = [device for device in preferences.devices if device.type == backend]
            if devices:
                for device in preferences.devices:
                    device.use = device in devices
                scene.cycles.device = "GPU"
                return backend
        except Exception as error:
            print(f"Warning: {backend} unavailable ({error})", file=sys.stderr)
    scene.cycles.device = "CPU"
    return "CPU"


def render_anon_masks(scene, camera, positions, destination: Path, names: list[str]) -> None:
    """Render binary valid-geometry masks at the panorama camera poses.

    White denotes retained room geometry. Black denotes rays escaping through
    openings or hitting objects identified as windows/glass, matching the
    ScanNet++ anonymization-mask convention used by the training loaders.
    """
    mask_dir = destination / "anon_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_material = bpy.data.materials.new("AnonMaskInterior")
    mask_material.use_nodes = True
    nodes = mask_material.node_tree.nodes
    links = mask_material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    emission.inputs["Strength"].default_value = 1
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    transparent_keywords = ("window", "glass", "pane", "glazing")
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        labels = " ".join([obj.name, *(material.name for material in obj.data.materials
                                       if material)]).lower()
        if obj.name == "ExteriorGround" or any(
                keyword in labels for keyword in transparent_keywords):
            obj.hide_render = True

    scene.view_layers[0].material_override = mask_material
    world = bpy.data.worlds.new("AnonMaskWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0, 0, 0, 1)
    background.inputs["Strength"].default_value = 1
    scene.world = world
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    for index, (position, name) in enumerate(zip(positions, names)):
        camera.location = position
        camera.rotation_euler = (math.pi / 2, 0, 2 * math.pi * index / len(positions))
        final = mask_dir / name
        temporary = mask_dir / f".{name}.tmp.png"
        scene.render.filepath = str(temporary)
        bpy.context.view_layer.update()
        print(f"Rendering anonymization mask {final}", flush=True)
        bpy.ops.render.render(write_still=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Blender did not create {temporary}")
        with Image.open(temporary) as rendered:
            binary = rendered.convert("L").point(lambda value: 255 if value >= 128 else 0)
            binary.save(temporary)
        os.replace(temporary, final)


def color_temperature(kelvin: float) -> tuple[float, float, float]:
    """Approximate a black-body color as display RGB for Blender light colors."""
    temperature = kelvin / 100.0
    red = 255.0 if temperature <= 66 else 329.698727446 * ((temperature - 60) ** -0.1332047592)
    green = (99.4708025861 * math.log(temperature) - 161.1195681661
             if temperature <= 66 else 288.1221695283 * ((temperature - 60) ** -0.0755148492))
    if temperature >= 66:
        blue = 255.0
    elif temperature <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(temperature - 10) - 305.044792731
    return tuple(max(0.0, min(255.0, value)) / 255.0 for value in (red, green, blue))


def object_center(obj) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def fixture_meshes():
    return [obj for obj in bpy.context.scene.objects
            if obj.type == "MESH" and "lighting" in obj.name.lower()]


def add_fixture_lights(rng: random.Random, base_energy: float, fixtures=None) -> list[dict]:
    """Place warm practical lights at furniture meshes whose names identify lamps."""
    fixtures = fixture_meshes() if fixtures is None else fixtures
    records = []
    for index, fixture in enumerate(fixtures[:8]):
        center = object_center(fixture)
        dimensions = fixture.dimensions
        kelvin = rng.uniform(2600.0, 3800.0)
        energy = base_energy * rng.uniform(0.65, 1.35)
        bpy.ops.object.light_add(type="POINT", location=center)
        light = bpy.context.object
        light.name = f"Practical_{fixture.name}_{index}"
        light.data.energy = energy
        light.data.color = color_temperature(kelvin)
        light.data.shadow_soft_size = max(0.03, min(0.25, max(dimensions) * 0.15))
        records.append({
            "fixture": fixture.name,
            "location": list(center),
            "energy": energy,
            "temperature_kelvin": kelvin,
        })
    return records


def add_lighting(bounds, background: str, light_energy: float,
                 lamp_energy: float, minimum_fill_energy: float,
                 seed: int, requested_period: str) -> dict:
    low, high = bounds
    rng = random.Random(seed ^ 0x5A17C9E3)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    period = requested_period
    if period == "random":
        period = "night" if rng.random() < 0.2 else "day"
    fixtures = fixture_meshes() if lamp_energy > 0 else []
    unlit_fallback = not fixtures
    # A night room without a detected lamp has no plausible key light. Avoid
    # generating an unusably black training image by falling back to daylight.
    if unlit_fallback and period == "night":
        period = "day"
    output = nodes.new("ShaderNodeOutputWorld")
    ambient = nodes.new("ShaderNodeBackground")
    ambient.name = "EnvironmentBackground"
    sky_settings = None
    if background == "sky":
        sky = nodes.new("ShaderNodeTexSky")
        sky.name = "ProceduralSky"
        sky_types = {item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items}
        sky.sky_type = "NISHITA" if "NISHITA" in sky_types else "MULTIPLE_SCATTERING"
        elevation = rng.uniform(-12.0, -2.0) if period == "night" else rng.uniform(8.0, 65.0)
        sky.sun_elevation = math.radians(elevation)
        sky.sun_rotation = rng.uniform(0.0, 2.0 * math.pi)
        sky.altitude = rng.uniform(0.0, 800.0)
        sky.air_density = rng.uniform(0.8, 1.25)
        aerosol_density = rng.uniform(0.5, 2.5)
        if hasattr(sky, "dust_density"):
            sky.dust_density = aerosol_density
        elif hasattr(sky, "aerosol_density"):
            sky.aerosol_density = aerosol_density
        sky.ozone_density = rng.uniform(0.8, 1.2)
        sky.sun_intensity = (rng.uniform(0.03, 0.15) if period == "night"
                             else rng.uniform(0.8, 1.3))
        sky.sun_size = math.radians(rng.uniform(0.35, 0.8))
        if hasattr(sky, "ground_albedo"):
            sky.ground_albedo = rng.uniform(0.15, 0.4)
        strength = rng.uniform(0.025, 0.08) if period == "night" else rng.uniform(0.18, 0.5)
        if unlit_fallback:
            strength = max(strength, 0.5)
        ambient.inputs["Strength"].default_value = strength
        # Atmospheric sky models return black below the horizon. Blend in a
        # muted procedural ground so window pixels below eye-level remain
        # plausible even though these room GLBs contain no outdoor geometry.
        ground_colors = ((0.008, 0.012, 0.018, 1.0), (0.012, 0.015, 0.012, 1.0)) \
            if period == "night" else (
                (0.12, 0.16, 0.08, 1.0), (0.16, 0.13, 0.09, 1.0),
                (0.11, 0.14, 0.14, 1.0), (0.18, 0.17, 0.14, 1.0),
            )
        ground_color = rng.choice(ground_colors)
        coordinates = nodes.new("ShaderNodeTexCoord")
        separate = nodes.new("ShaderNodeSeparateXYZ")
        horizon = nodes.new("ShaderNodeMapRange")
        horizon.clamp = True
        horizon.inputs["From Min"].default_value = -0.08
        horizon.inputs["From Max"].default_value = 0.04
        horizon.inputs["To Min"].default_value = 0.0
        horizon.inputs["To Max"].default_value = 1.0
        ground_mix = nodes.new("ShaderNodeMixRGB")
        ground_mix.blend_type = "MIX"
        ground_mix.inputs[2].default_value = ground_color
        links.new(coordinates.outputs["Normal"], separate.inputs["Vector"])
        links.new(separate.outputs["Z"], horizon.inputs["Value"])
        links.new(horizon.outputs["Result"], ground_mix.inputs[0])
        links.new(sky.outputs["Color"], ground_mix.inputs[1])
        links.new(ground_mix.outputs["Color"], ambient.inputs["Color"])
        sky_settings = {
            "time_of_day": period,
            "model": sky.sky_type,
            "sun_elevation_degrees": math.degrees(sky.sun_elevation),
            "sun_rotation_degrees": math.degrees(sky.sun_rotation),
            "sun_size_degrees": math.degrees(sky.sun_size),
            "sun_intensity": sky.sun_intensity,
            "altitude": sky.altitude,
            "air_density": sky.air_density,
            "aerosol_density": (sky.dust_density if hasattr(sky, "dust_density")
                                else sky.aerosol_density),
            "ozone_density": sky.ozone_density,
            "ground_albedo": sky.ground_albedo if hasattr(sky, "ground_albedo") else None,
            "ground_color": list(ground_color),
            "strength": strength,
        }
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.object
        sun.name = "ProceduralSun"
        sun.rotation_euler = (math.pi / 2 - sky.sun_elevation, 0.0, sky.sun_rotation)
        sun.data.energy = (rng.uniform(0.01, 0.05) if period == "night"
                           else rng.uniform(0.3, 1.2))
        if period == "night":
            sun.data.color = color_temperature(rng.uniform(7000.0, 10000.0))
        sun.data.angle = math.radians(rng.uniform(1.0, 8.0))
    else:
        ambient.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
        ambient.inputs["Strength"].default_value = 0.35
    if background == "black":
        # Use the gray world for indirect lighting, but return exact black for
        # rays that leave the room and reach the camera through an opening.
        black = nodes.new("ShaderNodeBackground")
        black.name = "CameraBackground"
        black.inputs["Color"].default_value = (0, 0, 0, 1)
        black.inputs["Strength"].default_value = 0.0
        light_path = nodes.new("ShaderNodeLightPath")
        mix = nodes.new("ShaderNodeMixShader")
        links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
        links.new(ambient.outputs["Background"], mix.inputs[1])
        links.new(black.outputs["Background"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Surface"])
    else:
        links.new(ambient.outputs["Background"], output.inputs["Surface"])
    bpy.ops.object.light_add(type="AREA", location=((low.x + high.x) / 2,
                                                     (low.y + high.y) / 2,
                                                     high.z - 0.15))
    light = bpy.context.object
    light.name = "CeilingFill"
    fill_multiplier = 2.0 if unlit_fallback else (0.12 if period == "night" else 1.0)
    fill_energy = light_energy * fill_multiplier
    if unlit_fallback:
        fill_energy = max(fill_energy, minimum_fill_energy)
    light.data.energy = fill_energy
    light.data.color = color_temperature(rng.uniform(3800.0, 5200.0))
    light.data.shape = "DISK"
    light.data.size = max(1.0, min(high.x - low.x, high.y - low.y) * 0.5)
    practicals = add_fixture_lights(rng, lamp_energy, fixtures) if fixtures else []
    settings = {
        "background": background,
        "time_of_day": period,
        "sky": sky_settings,
        "ceiling_fill_energy": light.data.energy,
        "ceiling_fill_color": list(light.data.color),
        "practical_lights": practicals,
        "unlit_fallback": unlit_fallback,
        "minimum_fill_energy": minimum_fill_energy,
        "fill_multiplier": fill_multiplier,
    }
    print(f"Lighting: background={background}, period={period}, "
          f"practical fixtures={len(practicals)}", flush=True)
    return settings


def procedural_material(seed: int, category: str):
    """Create a seeded material appropriate for one architectural category."""
    rng = random.Random(seed ^ sum(ord(c) << (i % 16) for i, c in enumerate(category)))
    # Repeated names deliberately weight common finishes more heavily than
    # visually strong ones, while still yielding substantially varied rooms.
    styles = {
        "wall": ("paint", "paint", "stucco", "linen-wallpaper", "striped-wallpaper",
                 "geometric-wallpaper", "brick", "stone-block"),
        "floor": ("oak-planks", "walnut-planks", "herringbone", "ceramic-tile",
                  "stone-tile", "terrazzo", "woven-carpet", "concrete"),
        "ceil": ("matte-ceiling", "fine-plaster", "tinted-ceiling", "concrete"),
        "others": ("painted-trim", "painted-trim", "dark-wood", "light-wood",
                   "stone-trim", "brushed-metal"),
        "ground": ("concrete", "stone-tile"),
    }
    style = rng.choice(styles[category])

    def rgba(hex_color: str):
        value = hex_color.lstrip("#")
        return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)

    # Curated pairs avoid arbitrary neon combinations. Neutral choices are
    # repeated so a large corpus stays plausible while still containing muted
    # greens, blues, clay, ochre and pink interiors.
    wall_pairs = (
        ("756f65", "eee9df"), ("81796d", "f3ede2"), ("777a72", "e2e1d8"),
        ("758071", "d7dfd2"), ("557068", "c7d9d2"), ("566b78", "cedae1"),
        ("465974", "c9d2df"), ("79665e", "decac0"), ("925f4c", "e5c1ae"),
        ("80656d", "ddc8ce"), ("897238", "e0cf99"), ("596044", "c9cbae"),
        ("44464b", "b8b9ba"), ("756f65", "eee9df"), ("81796d", "f3ede2"),
    )
    wallpaper_pairs = (
        ("334b4b", "d5ddd5"), ("455779", "d9dcea"), ("744c55", "ead8d8"),
        ("79642c", "eadcae"), ("4f6041", "dbe0cb"), ("875849", "ead0bf"),
        ("5d536c", "ded7e7"), ("53656a", "d3e0df"),
    )
    finish_pairs = {
        "oak-planks": (("56351f", "c6975f"), ("684228", "d2ac72"), ("49301f", "b98550")),
        "herringbone": (("51321e", "c08b51"), ("66462d", "d0a66f")),
        "light-wood": (("74543a", "d4b486"), ("65462d", "cba574")),
        "walnut-planks": (("24150f", "71452f"), ("2e1c15", "80543b")),
        "dark-wood": (("211713", "634536"), ("181514", "55463d")),
        "stone-tile": (("4d504f", "b8b6ae"), ("5c5750", "c9c0b3"), ("4e5558", "acb5b7")),
        "stone-block": (("595852", "c1bcae"), ("51585a", "b8c0be")),
        "stone-trim": (("454746", "aaa9a2"), ("514b45", "b7aa9c")),
        "concrete": (("555856", "aaa9a2"), ("555b5e", "abb2b3"), ("625d56", "bcb2a4")),
        "ceramic-tile": (("405c65", "d6e1df"), ("746153", "eadfd2"),
                          ("45594b", "d5ded3"), ("6c5454", "e1d3d0")),
        "terrazzo": (("55514c", "e1d8ca"), ("4b565b", "d9e0df"), ("655158", "e5d5d8")),
        "woven-carpet": (("3c4650", "8796a2"), ("594b43", "9e897d"),
                           ("46533e", "89977d"), ("664c52", "a9868e"),
                           ("5f552f", "a99b62")),
        "brick": (("642f24", "b96549"), ("704537", "c18165"), ("4d3b35", "9c7769")),
        "brushed-metal": (("25292c", "8f989d"), ("423b32", "a6967e")),
        "matte-ceiling": (("b3afa5", "f3f0e8"), ("afb3ae", "eff2ec")),
        "fine-plaster": (("aaa69d", "eeeae1"), ("a9adaa", "edf0ec")),
    }
    if style in finish_pairs:
        pair = rng.choice(finish_pairs[style])
    elif style in {"linen-wallpaper", "striped-wallpaper", "geometric-wallpaper"}:
        pair = rng.choice(wallpaper_pairs)
    elif style == "tinted-ceiling":
        pair = rng.choice(wall_pairs[:8])
    else:
        pair = rng.choice(wall_pairs)
    dark, light = rgba(pair[0]), rgba(pair[1])

    # A small per-room shift prevents exact duplicate colors without moving a
    # curated palette into an implausible hue or saturation range.
    shift = rng.uniform(-0.025, 0.025)
    dark = tuple(max(0.0, min(1.0, c + shift)) for c in dark[:3]) + (1.0,)
    light = tuple(max(0.0, min(1.0, c + shift)) for c in light[:3]) + (1.0,)

    material = bpy.data.materials.new(f"Procedural_{category}_{style}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texture_coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(texture_coordinates.outputs["Generated"], mapping.inputs["Vector"])

    if style in {"oak-planks", "walnut-planks", "herringbone", "striped-wallpaper",
                 "linen-wallpaper", "brushed-metal"}:
        texture = nodes.new("ShaderNodeTexWave")
        texture.wave_type = "BANDS"
        texture.bands_direction = "X" if style not in {"striped-wallpaper", "linen-wallpaper"} else "Z"
        texture.inputs["Scale"].default_value = {
            "striped-wallpaper": rng.uniform(5.0, 10.0),
            "linen-wallpaper": rng.uniform(35.0, 60.0),
            "brushed-metal": rng.uniform(45.0, 80.0),
        }.get(style, rng.uniform(7.0, 16.0))
        texture.inputs["Distortion"].default_value = rng.uniform(1.8, 5.0) if "planks" in style or style == "herringbone" else rng.uniform(0.02, 0.18)
        texture.inputs["Detail"].default_value = rng.uniform(2.0, 6.0)
        if style == "herringbone":
            mapping.inputs["Rotation"].default_value[2] = math.radians(45)
    elif style in {"stone-tile", "stone-block", "terrazzo"}:
        texture = nodes.new("ShaderNodeTexVoronoi")
        texture.distance = "EUCLIDEAN"
        texture.inputs["Scale"].default_value = rng.uniform(5.0, 10.0) if style != "terrazzo" else rng.uniform(28.0, 55.0)
    elif style in {"ceramic-tile", "geometric-wallpaper", "brick"}:
        texture = nodes.new("ShaderNodeTexBrick")
        texture.inputs["Scale"].default_value = rng.uniform(4.0, 8.0) if style == "brick" else rng.uniform(8.0, 18.0)
        texture.inputs["Mortar Size"].default_value = 0.025 if style == "brick" else 0.012
        texture.inputs["Color1"].default_value = dark
        texture.inputs["Color2"].default_value = light
        texture.inputs["Mortar"].default_value = (0.08, 0.08, 0.075, 1)
    else:
        texture = nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = rng.uniform(2.5, 8.0) if style != "woven-carpet" else rng.uniform(35.0, 70.0)
        texture.inputs["Detail"].default_value = rng.uniform(4.0, 7.0)
        texture.inputs["Roughness"].default_value = 0.72
        texture.inputs["Distortion"].default_value = rng.uniform(0.08, 0.25)
    ramp = nodes.new("ShaderNodeValToRGB")
    bump = nodes.new("ShaderNodeBump")
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = dark
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = light
    shader.inputs["Roughness"].default_value = 0.42 if style in {"ceramic-tile", "brushed-metal"} else 0.82
    if style == "brushed-metal" and "Metallic" in shader.inputs:
        shader.inputs["Metallic"].default_value = 0.72
    bump.inputs["Strength"].default_value = 0.08 if category == "ceil" else rng.uniform(0.14, 0.32)
    bump.inputs["Distance"].default_value = 0.012 if category == "ceil" else 0.028
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    fac_output = "Distance" if style in {"stone-tile", "stone-block", "terrazzo"} else "Fac"
    links.new(texture.outputs[fac_output], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(texture.outputs[fac_output], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    return material, style


PBR_ASSET_CACHE = {}


def texture_map(asset: Path, names: tuple[str, ...]) -> Path | None:
    images = sorted(path for path in asset.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for name in names:
        marker = f"_{name.lower()}."
        for image in images:
            if marker in image.name.lower():
                return image
    return None


def ambientcg_assets(root: Path) -> dict[str, list[Path]]:
    """Index usable ambientCG assets once, grouped by indoor surface category."""
    cache_key = str(root.resolve())
    if cache_key in PBR_ASSET_CACHE:
        return PBR_ASSET_CACHE[cache_key]
    keywords = {
        "wall": ("paint", "plaster", "brick", "concrete", "wall"),
        "floor": ("woodfloor", "wood", "plank", "tile", "carpet", "marble",
                  "terrazzo", "concrete", "parquet"),
        "ceil": ("paint", "plaster"),
        "others": ("wood", "metal", "marble", "plastic", "paint"),
        "ground": ("grass", "ground", "soil", "gravel", "asphalt", "paving",
                   "cobble", "sand", "forestfloor", "rock"),
    }
    exclusions = {
        "wall": ("wood", "metal", "floor", "road"),
        "floor": ("wall", "roof"),
        "ceil": ("wood", "metal", "floor", "brick", "tile", "road"),
        "others": ("floor", "paving", "asphalt", "grass", "soil", "gravel"),
        "ground": ("wall", "roof", "paint", "plaster"),
    }
    folders = [path for path in sorted(root.iterdir())
               if path.is_dir() and texture_map(path, ("Color",)) is not None]
    grouped = {}
    for category, category_keywords in keywords.items():
        if category == "ceil":
            grouped[category] = [path for path in folders
                                 if path.name.lower().startswith("plaster")]
        else:
            grouped[category] = [
                path for path in folders
                if any(word in path.name.lower() for word in category_keywords)
                and not any(word in path.name.lower() for word in exclusions[category])
            ]
    PBR_ASSET_CACHE[cache_key] = grouped
    print("Indexed ambientCG materials: " + ", ".join(
        f"{category}={len(paths)}" for category, paths in grouped.items()), flush=True)
    return grouped


def ambientcg_material(seed: int, category: str, root: Path):
    """Build a Blender material from one deterministic, category-appropriate PBR set."""
    rng = random.Random(seed ^ sum(ord(c) << (i % 16) for i, c in enumerate(category)))
    candidates = ambientcg_assets(root)[category]
    if not candidates:
        print(f"Warning: no usable ambientCG {category} materials; using procedural fallback",
              file=sys.stderr)
        return procedural_material(seed, category)
    asset = rng.choice(candidates)
    maps = {
        "color": texture_map(asset, ("Color",)),
        "roughness": texture_map(asset, ("Roughness",)),
        "metallic": texture_map(asset, ("Metalness", "Metallic")),
        "normal": texture_map(asset, ("NormalGL", "Normal")),
        "height": texture_map(asset, ("Displacement", "Height")),
        "ao": texture_map(asset, ("AmbientOcclusion", "AO")),
    }
    material = bpy.data.materials.new(f"ambientCG_{category}_{asset.name}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    scale = rng.uniform(24.0, 48.0) if category == "ground" else rng.uniform(2.0, 5.0)
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])

    image_nodes = {}
    for map_name, path in maps.items():
        if path is None:
            continue
        node = nodes.new("ShaderNodeTexImage")
        node.name = f"PBR_{map_name}"
        node.image = bpy.data.images.load(str(path), check_existing=True)
        node.extension = "REPEAT"
        node.projection = "BOX"
        node.projection_blend = 0.2
        if map_name != "color":
            node.image.colorspace_settings.name = "Non-Color"
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        image_nodes[map_name] = node

    color_output = image_nodes["color"].outputs["Color"]
    if "ao" in image_nodes:
        multiply = nodes.new("ShaderNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 0.7
        links.new(color_output, multiply.inputs[1])
        links.new(image_nodes["ao"].outputs["Color"], multiply.inputs[2])
        color_output = multiply.outputs["Color"]
    links.new(color_output, shader.inputs["Base Color"])
    if "roughness" in image_nodes:
        links.new(image_nodes["roughness"].outputs["Color"], shader.inputs["Roughness"])
    else:
        shader.inputs["Roughness"].default_value = 0.7
    if "metallic" in image_nodes:
        links.new(image_nodes["metallic"].outputs["Color"], shader.inputs["Metallic"])

    normal_output = None
    if "normal" in image_nodes:
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = 0.65
        links.new(image_nodes["normal"].outputs["Color"], normal.inputs["Color"])
        normal_output = normal.outputs["Normal"]
    if "height" in image_nodes:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.18 if category == "ceil" else 0.3
        bump.inputs["Distance"].default_value = 0.03
        links.new(image_nodes["height"].outputs["Color"], bump.inputs["Height"])
        if normal_output is not None:
            links.new(normal_output, bump.inputs["Normal"])
        normal_output = bump.outputs["Normal"]
    if normal_output is not None:
        links.new(normal_output, shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material, f"ambientCG:{asset.name}"


def architecture_material(seed: int, category: str, pbr_root: Path | None):
    return (ambientcg_material(seed, category, pbr_root) if pbr_root is not None
            else procedural_material(seed, category))


def add_exterior_ground(bounds, seed: int, pbr_root: Path) -> dict:
    """Add sunlit textured ground outside the room, slightly below its floor."""
    low, high = bounds
    room_span = max(high.x - low.x, high.y - low.y)
    size = max(20.0, room_span * 30.0)
    location = ((low.x + high.x) / 2, (low.y + high.y) / 2, low.z - 0.025)
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    ground = bpy.context.object
    ground.name = "ExteriorGround"
    material, style = ambientcg_material(seed ^ 0x31D5A9B7, "ground", pbr_root)
    ground.data.materials.append(material)
    print(f"Exterior ground: {style}, size={size:.2f}", flush=True)
    return {"mode": "pbr-plane", "material": style, "size": size, "height": location[2]}


def import_room_with_procedural_architecture(mesh: Path, seed: int,
                                             pbr_root: Path | None = None) -> bool:
    """Import furniture and separate architecture so each gets a suitable material."""
    room_name = mesh.stem.removesuffix("_full")
    furniture = mesh.with_name(f"{room_name}.glb")
    component_dir = mesh.parent / room_name
    components = {category: component_dir / f"{category}.glb"
                  for category in ("wall", "floor", "ceil", "others")}
    available = {category: path for category, path in components.items() if path.is_file()}
    # Some rooms contain only ``others.glb``, which is a single merged shell;
    # treating that as trim would incorrectly texture the whole room. Use the
    # combined-GLB fallback unless at least walls or floors are separable.
    if not furniture.is_file() or not ({"wall", "floor"} & available.keys()):
        return False
    bpy.ops.import_scene.gltf(filepath=str(furniture), merge_vertices=True)
    applied = []
    for category, path in available.items():
        before = {obj.as_pointer() for obj in bpy.context.scene.objects}
        bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=True)
        imported = [obj for obj in bpy.context.scene.objects
                    if obj.type == "MESH" and obj.as_pointer() not in before]
        material, style = architecture_material(seed, category, pbr_root)
        for obj in imported:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        applied.append(f"{category}={style} ({len(imported)} mesh)")
    print("Architecture materials: " + ", ".join(applied))
    return True


def add_procedural_architecture_material(seed: int, pbr_root: Path | None = None) -> None:
    """Fallback for rooms without separable architectural component GLBs."""
    material, style = architecture_material(seed, "wall", pbr_root)

    def structural(name: str) -> bool:
        return name.startswith("None") or name.startswith("geometry_") or (name and name[0].isdigit())

    replaced = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not structural(obj.name):
            continue
        # The processed architectural materials have no image nodes. Preserve
        # any unexpectedly textured structural object rather than overwriting it.
        if any(
            node.type == "TEX_IMAGE" and node.image is not None
            for old in obj.data.materials if old and old.use_nodes
            for node in old.node_tree.nodes
        ):
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)
        replaced += 1
    print(f"Applied fallback {style} material to {replaced} structural mesh object(s)")


def render_room(mesh: Path, destination: Path, args, seed: int) -> None:
    clear_scene()
    separated = False
    replace_architecture = args.architecture_material != "original"
    pbr_root = args.pbr_material_root if args.architecture_material == "ambientcg" else None
    if replace_architecture:
        separated = import_room_with_procedural_architecture(mesh, seed, pbr_root)
    if not separated:
        bpy.ops.import_scene.gltf(filepath=str(mesh), merge_vertices=True)
    bpy.context.view_layer.update()
    if replace_architecture and not separated:
        add_procedural_architecture_material(seed, pbr_root)
    bounds = mesh_bounds()
    camera_height = args.camera_height
    if camera_height is None:
        camera_height = (bounds[1].z - bounds[0].z) * args.camera_height_ratio
    if args.camera_poses is not None:
        saved_extrinsics = json.loads(args.camera_poses.read_text())
        expected_names = [
            f"{mesh.stem.removesuffix('_full')}_panorama_{i:03d}.png"
            for i in range(args.num_panoramas)
        ]
        missing = [name for name in expected_names if name not in saved_extrinsics]
        if missing:
            raise RuntimeError(f"Camera pose file is missing entries: {missing}")
        positions = [Matrix(saved_extrinsics[name]).inverted().translation.copy()
                     for name in expected_names]
    else:
        positions = camera_positions(
            bounds, args.num_panoramas, camera_height, args.clearance,
            args.min_separation, args.max_attempts, random.Random(seed),
            scene_surface_bvh(),
        )
    exterior_ground = None
    if args.exterior_ground == "pbr-plane":
        exterior_ground = add_exterior_ground(bounds, seed, args.pbr_material_root)
    lighting = add_lighting(
        bounds, args.background, args.light_energy, args.lamp_energy,
        args.minimum_fill_energy, seed, args.time_of_day,
    )
    lighting["exterior_ground"] = exterior_ground or {"mode": args.exterior_ground}
    atomic_json(destination / "lighting.json", lighting)
    scene = bpy.context.scene
    backend = configure_cycles(scene, args.device, args.samples, args.threads)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "PanoramaCamera"
    camera.data.type = "PANO"
    if hasattr(camera.data, "panorama_type"):
        camera.data.panorama_type = "EQUIRECTANGULAR"
    else:
        camera.data.cycles.panorama_type = "EQUIRECTANGULAR"
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True

    extrinsics = {}
    panorama_names = []
    for index, position in enumerate(positions):
        camera.location = position
        camera.rotation_euler = (math.pi / 2, 0, 0)  # level; pano yaw only moves its seam
        camera.rotation_euler.z = 2 * math.pi * index / len(positions)
        bpy.context.view_layer.update()
        name = f"{mesh.stem.removesuffix('_full')}_panorama_{index:03d}.png"
        final = destination / name
        temporary = destination / f".{name}.tmp.png"
        scene.render.filepath = str(temporary)
        print(f"Rendering {final} at {list(position)} ({backend})", flush=True)
        bpy.ops.render.render(write_still=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Blender did not create {temporary}")
        os.replace(temporary, final)
        extrinsics[name] = [list(row) for row in camera.matrix_world.inverted()]
        panorama_names.append(name)
        atomic_json(destination / "extrinsics.json", extrinsics)

    render_anon_masks(scene, camera, positions, destination, panorama_names)


def main() -> int:
    args = arguments()
    if min(args.num_panoramas, args.width, args.height, args.samples, args.shard_count) < 1:
        raise SystemExit("Counts and image dimensions must be positive")
    if args.threads is not None and args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.camera_height is not None and args.camera_height <= 0:
        raise SystemExit("--camera-height must be positive")
    if not 0 < args.camera_height_ratio < 1:
        raise SystemExit("--camera-height-ratio must be between 0 and 1")
    if args.clearance <= 0 or args.min_separation < 0:
        raise SystemExit("--clearance must be positive and --min-separation non-negative")
    if min(args.light_energy, args.lamp_energy, args.minimum_fill_energy) < 0:
        raise SystemExit(
            "--light-energy, --lamp-energy, and --minimum-fill-energy must be non-negative"
        )
    if args.architecture_material == "ambientcg":
        if args.pbr_material_root is None:
            raise SystemExit("--pbr-material-root is required with --architecture-material ambientcg")
        args.pbr_material_root = args.pbr_material_root.resolve()
        if not args.pbr_material_root.is_dir():
            raise SystemExit(f"PBR material directory does not exist: {args.pbr_material_root}")
    if args.exterior_ground == "pbr-plane":
        if args.pbr_material_root is None:
            raise SystemExit("--pbr-material-root is required with --exterior-ground pbr-plane")
        args.pbr_material_root = args.pbr_material_root.resolve()
        if not args.pbr_material_root.is_dir():
            raise SystemExit(f"PBR material directory does not exist: {args.pbr_material_root}")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Require 0 <= --shard-index < --shard-count")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.target_complete is not None and args.target_complete < 1:
        raise SystemExit("--target-complete must be positive")
    source_root = args.input.resolve()
    rooms = inputs(source_root)
    rooms = rooms[args.start_index:]
    if args.limit is not None:
        rooms = rooms[:args.limit]
    rooms = rooms[args.shard_index::args.shard_count]
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    complete_count = sum(1 for marker in output_root.rglob(".complete.json")
                         if marker.is_file())
    if args.target_complete is not None:
        print(f"Found {complete_count}/{args.target_complete} completed rooms in {output_root}",
              flush=True)
    failures = []
    settings = {key: getattr(args, key) for key in (
        "num_panoramas", "width", "height", "samples", "threads", "light_energy", "lamp_energy",
        "minimum_fill_energy", "camera_height",
        "camera_height_ratio",
        "clearance", "min_separation", "seed", "architecture_material", "background", "exterior_ground",
        "time_of_day",
    )}
    settings["anon_masks"] = True
    settings["camera_poses"] = (str(args.camera_poses.resolve())
                                if args.camera_poses is not None else None)
    settings["pbr_material_root"] = (str(args.pbr_material_root)
                                     if args.pbr_material_root is not None else None)
    for index, mesh in enumerate(rooms, 1):
        if args.target_complete is not None:
            # Other shards write completion markers into the same output root.
            # Refresh the shared count before claiming more work so a parallel
            # launch stops near the requested global target instead of making
            # every shard independently attempt that many rooms.
            complete_count = sum(1 for marker in output_root.rglob(".complete.json")
                                 if marker.is_file())
            if complete_count >= args.target_complete:
                print(f"Reached target of {args.target_complete} completed rooms", flush=True)
                break
        destination = output_dir(mesh, source_root, output_root)
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / ".complete.json"
        try:
            previous = json.loads(marker.read_text()) if marker.is_file() else None
        except (OSError, json.JSONDecodeError):
            previous = None
        expected = [destination / f"{mesh.stem.removesuffix('_full')}_panorama_{i:03d}.png"
                    for i in range(args.num_panoramas)]
        expected_masks = [destination / "anon_mask" / path.name for path in expected]
        if not args.overwrite and previous and previous.get("render_settings") == settings \
                and all(path.is_file() for path in expected + expected_masks):
            print(f"[{index}/{len(rooms)}] Skipping complete room {mesh}")
            continue
        marker.unlink(missing_ok=True)
        try:
            digest = hashlib.sha256(str(mesh).encode()).digest()
            room_seed = args.seed + int.from_bytes(digest[:4], "little")
            print(f"[{index}/{len(rooms)}] Processing {mesh}")
            render_room(mesh, destination, args, room_seed)
            atomic_json(marker, {
                "source": str(mesh),
                "scene_id": mesh.parent.name,
                "room_id": mesh.stem.removesuffix("_full"),
                "render_settings": settings,
                "coordinate_system": "blender_z_up_after_gltf_import",
                "extrinsics_convention": "row_major_world_to_camera_blender_axes",
            })
            complete_count += 1
            (destination / "failure.log").unlink(missing_ok=True)
        except Exception as error:
            failures.append((mesh, error))
            (destination / "failure.log").write_text(traceback.format_exc())
            print(f"ERROR: {mesh}: {error}", file=sys.stderr)
        finally:
            clear_scene()
    if failures:
        print(f"Failed rooms: {len(failures)}/{len(rooms)}", file=sys.stderr)
    if args.target_complete is not None:
        complete_count = sum(1 for marker in output_root.rglob(".complete.json")
                             if marker.is_file())
        if complete_count >= args.target_complete:
            return 0
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
