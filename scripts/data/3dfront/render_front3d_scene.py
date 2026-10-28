#!/usr/bin/env python3
"""Render one target room with full-scene context and export a room-only GLB.

Required environment variables: FRONT3D_SCENE, FRONT3D_ROOM, FRONT3D_OUTPUT,
and FRONT3D_HDRI. Optional lighting variables are documented in
docs/runbooks/3dFront_rendering.md.
"""

import builtins
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("USE_EXTERNAL_BPY_MODULE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/blenderproc-matplotlib")

import bpy
import imageio.v3 as iio
import numpy as np
import blenderproc as bproc
from mathutils import Vector


SCENE = Path(os.environ["FRONT3D_SCENE"])
FUTURE = Path("/tmp/datasets/3dfront/future_model")
TEXTURES = Path("/tmp/datasets/3dfront/3D-FRONT-texture")
OUTPUT = Path(os.environ["FRONT3D_OUTPUT"])
CC_TEXTURES = Path("/tmp/datasets/3dfront/cctextures")
HDRI = Path(os.environ["FRONT3D_HDRI"])
HDRI_ROTATION = math.radians(float(os.environ.get("FRONT3D_HDRI_ROTATION", "0")))
HDRI_STRENGTH = float(os.environ.get("FRONT3D_HDRI_STRENGTH", "1"))
CEILING_EMISSION_STRENGTH = float(os.environ.get("FRONT3D_CEILING_EMISSION_STRENGTH", "0.8"))
LAMP_EMISSION_STRENGTH = float(os.environ.get("FRONT3D_LAMP_EMISSION_STRENGTH", "7.0"))
DOOR_OPEN_PROBABILITY = float(os.environ.get("FRONT3D_OPEN_DOOR_PROBABILITY", "0.45"))
DOOR_FULLY_OPEN_PROBABILITY = float(os.environ.get("FRONT3D_FULLY_OPEN_DOOR_PROBABILITY", "0.30"))
DOOR_MIN_ANGLE = float(os.environ.get("FRONT3D_DOOR_MIN_ANGLE", "12"))
DOOR_MAX_ANGLE = float(os.environ.get("FRONT3D_DOOR_MAX_ANGLE", "85"))
if not (0 <= DOOR_OPEN_PROBABILITY <= 1 and 0 <= DOOR_FULLY_OPEN_PROBABILITY <= 1):
    raise ValueError("Door probabilities must be in [0, 1]")
if not (0 < DOOR_MIN_ANGLE <= DOOR_MAX_ANGLE <= 90):
    raise ValueError("Door angles must satisfy 0 < min <= max <= 90 degrees")
ROOM_INSTANCE = os.environ["FRONT3D_ROOM"]


data = json.loads(SCENE.read_text())
room_index, room = next(
    (index, room)
    for index, room in enumerate(data["scene"]["room"])
    if room.get("instanceid") == ROOM_INSTANCE
)
mesh_by_uid = {mesh["uid"]: mesh for mesh in data["mesh"]}
room_mesh_uids = {
    child["ref"] for child in room["children"]
    if child["ref"] in mesh_by_uid
}
window_uids = {
    uid for uid in room_mesh_uids if mesh_by_uid[uid].get("type") == "Window"
}
if not window_uids:
    print(f"{ROOM_INSTANCE} has no windows; rendering with environment and interior lighting", flush=True)

# Source coordinates are X/Y-up/Z; BlenderProc converts them to Blender X/Y/Z.
# Retain the target floor footprint for camera placement and for discovering
# doors assigned to an adjacent room but located on this room's boundary.
floor_vertices = []
floor_edge_counts = {}
for uid in room_mesh_uids:
    record = mesh_by_uid[uid]
    if record.get("type", "").lower() == "floor":
        vertices = np.asarray(record["xyz"], dtype=float).reshape(-1, 3)
        floor_vertices.append(vertices)
        for triangle in np.asarray(record["faces"]).reshape(-1, 3):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted(tuple(np.round(vertices[triangle[i], [0, 2]], 5)) for i in (a, b)))
                floor_edge_counts[edge] = floor_edge_counts.get(edge, 0) + 1
if not floor_vertices:
    raise RuntimeError(f"No floor mesh found for {ROOM_INSTANCE}")
floor_vertices = np.concatenate(floor_vertices)
adjacent_door_margin = float(os.environ.get("FRONT3D_ADJACENT_DOOR_MARGIN", "0.05"))
floor_boundary = [(np.array(a), np.array(b)) for (a, b), count in floor_edge_counts.items() if count == 1]


def door_touches_floor_boundary(center, width, depth, width_axis, normal_axis):
    """Require the opening's central span to follow a real floor boundary."""
    for fraction in (-0.25, 0.0, 0.25):
        point = center[:2].copy()
        point[width_axis] += fraction * width
        touches = False
        for start, end in floor_boundary:
            direction = end - start
            length_squared = float(direction @ direction)
            if length_squared < 1e-10 or abs(direction[width_axis]) < 3 * abs(direction[normal_axis]):
                continue
            closest = start + np.clip(float((point - start) @ direction) / length_squared, 0, 1) * direction
            if np.linalg.norm(point - closest) <= depth / 2 + adjacent_door_margin:
                touches = True
                break
        if not touches:
            return False
    return True

scene_cache = getattr(builtins, "_nova3r_front3d_scene_cache", None)
reuse_scene = scene_cache is not None and scene_cache["scene"] == str(SCENE.resolve())
if not reuse_scene:
    # Blender 5 removed the legacy compositor node tree used by BlenderProc's
    # default Intel denoiser. Denoising is configured explicitly below after
    # the requested Cycles device has been selected.
    from blenderproc.python.utility.DefaultConfig import DefaultConfig
    DefaultConfig.denoiser = None
    bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
if reuse_scene:
    loaded = scene_cache["loaded"]
    scene_cache["rooms_processed"] += 1
    print(f"Reusing loaded 3D-FRONT scene {SCENE.stem}", flush=True)
else:
    loaded = bproc.loader.load_front3d(
        json_path=str(SCENE),
        future_model_path=str(FUTURE),
        front_3D_texture_path=str(TEXTURES),
        label_mapping=mapping,
        # These are BlenderProc's native 3D-FRONT lighting controls. The loader
        # turns ceiling and lamp materials into camera-visible emissive shaders.
        ceiling_light_strength=CEILING_EMISSION_STRENGTH,
        lamp_light_strength=LAMP_EMISSION_STRENGTH,
    )
    builtins._nova3r_front3d_scene_cache = {
        "scene": str(SCENE.resolve()),
        "loaded": loaded,
        "rooms_processed": 1,
    }

# BlenderProc intends to make 3D-FRONT lamps emissive, but its loader checks
# for "lamp" while many released assets are categorized as "lighting". Apply
# the same Material.make_emissive() operation to those missed fixtures.
emissive_fixture_materials = set()
emissive_fixture_objects = []
for obj in loaded:
    fixture_label = obj.get_name().lower()
    is_fixture = any(
        token in fixture_label for token in ("lamp", "lighting", "chandelier", "pendant")
    )
    changed = False
    for material in obj.get_materials():
        if material is None:
            continue
        # A persistent worker may assign different lighting profiles to rooms
        # in the same dwelling. Refresh existing emission strengths instead of
        # retaining the first room's values in the cached scene.
        for node in material.get_nodes_with_type("Emission"):
            node.inputs["Strength"].default_value = (
                LAMP_EMISSION_STRENGTH if is_fixture else CEILING_EMISSION_STRENGTH
            )
        if not is_fixture:
            continue
        material_key = material.blender_obj.as_pointer()
        if material_key in emissive_fixture_materials:
            continue
        if not material.get_nodes_with_type("Emission"):
            material.make_emissive(LAMP_EMISSION_STRENGTH)
        emissive_fixture_materials.add(material_key)
        changed = True
    if changed:
        emissive_fixture_objects.append(obj.get_name())

# Match BlenderProc's front_3d_with_improved_mat example: random CC materials
# on floors, wood-floor materials on doors/baseboards, and occasional marble
# walls. A fixed seed makes this artifact reproducible.
random.seed(42)
np.random.seed(42)
if reuse_scene:
    cc_materials = scene_cache["cc_materials"]
else:
    cc_materials = bproc.loader.load_ccmaterials(
        str(CC_TEXTURES), ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
    )
    scene_cache = builtins._nova3r_front3d_scene_cache
    scene_cache["cc_materials"] = cc_materials
floors = bproc.filter.by_attr(loaded, "name", "Floor.*", regex=True)
for floor in floors:
    for material_index in range(len(floor.get_materials())):
        if np.random.uniform(0, 1) <= 0.95:
            floor.set_material(material_index, random.choice(cc_materials))

baseboards_and_doors = bproc.filter.by_attr(
    loaded, "name", "Baseboard.*|Door.*", regex=True
)
wood_floor_materials = bproc.filter.by_cp(
    cc_materials, "asset_name", "WoodFloor.*", regex=True
)
for obj in baseboards_and_doors:
    for material_index in range(len(obj.get_materials())):
        obj.set_material(material_index, random.choice(wood_floor_materials))

walls = bproc.filter.by_attr(loaded, "name", "Wall.*", regex=True)
marble_materials = bproc.filter.by_cp(
    cc_materials, "asset_name", "Marble.*", regex=True
)
for wall in walls:
    for material_index in range(len(wall.get_materials())):
        if np.random.uniform(0, 1) <= 0.1:
            wall.set_material(material_index, random.choice(marble_materials))

# Front3DLoader returns architectural meshes first, in data["mesh"] order.
architecture = loaded[:len(data["mesh"])]
for obj, mesh_record in zip(architecture, data["mesh"]):
    obj.set_cp("front3d_mesh_uid", mesh_record["uid"])

target_room_objects = []
target_sampler_objects = []
excluded_target_mesh_types = []
for obj in loaded:
    mesh_uid = obj.get_cp("front3d_mesh_uid") if obj.has_cp("front3d_mesh_uid") else None
    mesh_type = mesh_by_uid.get(mesh_uid, {}).get("type")
    target_architecture = (
        mesh_uid in room_mesh_uids
        and mesh_type != "SlabSide"
    )
    if mesh_uid in room_mesh_uids and mesh_type == "SlabSide":
        excluded_target_mesh_types.append(mesh_type)
    target_furniture = obj.has_cp("room_id") and obj.get_cp("room_id") == room_index
    if target_architecture or target_furniture:
        target_room_objects.append(obj.blender_obj)
        target_sampler_objects.append(obj)

# Keep every loaded object for panorama rendering. Adjacent rooms then remain
# visible through doors and open-plan connections. Only target_room_objects is
# selected during the GLB export below.


def make_principled_material(name, color, roughness=0.4, metallic=0.0,
                             transmission=0.0, alpha=1.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, alpha)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["IOR"].default_value = 1.45
    shader.inputs["Transmission Weight"].default_value = transmission
    shader.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        material.surface_render_method = "DITHERED"
        material.use_transparency_overlap = False
    return material


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        np.array([min(c[i] for c in corners) for i in range(3)]),
        np.array([max(c[i] for c in corners) for i in range(3)]),
    )


def add_box(name, center, dimensions, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    box = bpy.context.object
    box.name = name
    box.dimensions = dimensions
    box.data.materials.append(material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return box


generated = []
target_generated = []
glass_settings = []
door_settings = []
adjacent_door_openings = []
window_index = 0
door_index = 0
for opening in list(bpy.context.scene.objects):
    if opening.type != "MESH":
        continue
    lower_name = opening.name.lower()
    if "window" not in lower_name and "door" not in lower_name:
        continue
    low, high = world_bounds(opening)
    extent = high - low
    center = 0.5 * (low + high)
    # 3D-FRONT is Z-up after loading. The thinner horizontal dimension is the
    # wall normal; the other horizontal dimension spans the opening.
    normal_axis = 0 if extent[0] < extent[1] else 1
    width_axis = 1 - normal_axis
    width, height, depth = extent[width_axis], extent[2], extent[normal_axis]

    if "window" in lower_name and height > 0.5 and width > 0.5:
        opening_uid = opening.get("front3d_mesh_uid")
        belongs_to_target = opening_uid in room_mesh_uids
        # Deterministic per-window variation. Alpha is the visual opacity used
        # by Blender's dithered surface; transmission, roughness, and IOR keep
        # the result in the range of ordinary architectural glass.
        glass_rng = random.Random(f"{SCENE.stem}:{ROOM_INSTANCE}:{opening.name}")
        opacity = glass_rng.uniform(0.18, 0.38)
        transmission = glass_rng.uniform(0.88, 0.98)
        roughness = glass_rng.uniform(0.03, 0.12)
        ior = glass_rng.uniform(1.45, 1.52)
        tint = (
            glass_rng.uniform(0.32, 0.42),
            glass_rng.uniform(0.55, 0.68),
            glass_rng.uniform(0.68, 0.82),
        )
        glass_material = make_principled_material(
            f"WindowGlass_{window_index}", tint, roughness=roughness,
            transmission=transmission, alpha=opacity,
        )
        glass_material.node_tree.nodes["Principled BSDF"].inputs["IOR"].default_value = ior
        styles = ["picture", "double_casement", "triple_panel", "grid",
                  "transom", "awning", "arched_transom"]
        style = os.environ.get("FRONT3D_WINDOW_STYLE") or glass_rng.choice(styles)
        if style not in styles:
            raise ValueError(f"Unknown window style: {style}")
        finish = glass_rng.choice(["charcoal_aluminum", "white", "oak", "bronze"])
        finish_colors = {"charcoal_aluminum": (0.045, 0.055, 0.065),
                         "white": (0.82, 0.80, 0.75), "oak": (0.30, 0.15, 0.055),
                         "bronze": (0.18, 0.10, 0.045)}
        window_frame_material = make_principled_material(
            f"WindowFinish_{window_index}_{finish}", finish_colors[finish],
            roughness=0.38 if finish in {"white", "oak"} else 0.25,
            metallic=0.0 if finish in {"white", "oak"} else 0.72,
        )
        window_parts = []
        bar = min(glass_rng.uniform(0.045, 0.085), width * 0.06, height * 0.06)
        frame_depth = glass_rng.uniform(0.065, 0.12)

        def window_box(name, u, z, w, h, material, thickness=frame_depth):
            position = center.copy()
            position[width_axis] = u
            position[2] = z
            dimensions = np.zeros(3)
            dimensions[width_axis] = w
            dimensions[normal_axis] = thickness
            dimensions[2] = h
            part = add_box(f"Window{name}_{window_index}_{len(window_parts)}",
                           position, dimensions, material)
            window_parts.append(part)
            return part

        for side, u in (("Left", low[width_axis] + bar / 2),
                        ("Right", high[width_axis] - bar / 2)):
            window_box(f"Frame{side}", u, center[2], bar, height, window_frame_material)
        for side, z in (("Bottom", low[2] + bar / 2), ("Top", high[2] - bar / 2)):
            window_box(f"Frame{side}", center[width_axis], z, width, bar, window_frame_material)
        u0, u1 = low[width_axis] + bar, high[width_axis] - bar
        z0, z1 = low[2] + bar, high[2] - bar
        columns = {"double_casement": 2, "triple_panel": 3, "grid": 3}.get(style, 1)
        rows = 2 if style in {"grid", "transom", "awning", "arched_transom"} else 1
        us = np.linspace(u0, u1, columns + 1)
        zs = np.array([z0, z0 + (z1 - z0) * 0.72, z1]) if rows == 2 and style != "grid" else np.linspace(z0, z1, rows + 1)
        mullion_bar = bar * 0.7
        for u in us[1:-1]:
            window_box("Mullion", u, center[2], mullion_bar, z1-z0, window_frame_material)
        for z in zs[1:-1]:
            window_box("Transom", center[width_axis], z, u1-u0, mullion_bar, window_frame_material)
        for column in range(columns):
            for row in range(rows):
                pane_width = us[column+1] - us[column] - mullion_bar
                pane_height = zs[row+1] - zs[row] - mullion_bar
                pane = window_box("Glass", (us[column]+us[column+1])/2,
                                  (zs[row]+zs[row+1])/2, pane_width, pane_height,
                                  glass_material, thickness=0.012)
                if style == "awning" and row == rows-1:
                    # Rotate pane and its inset sash around the upper hinge.
                    sash_parts = [pane]
                    pu = (us[column]+us[column+1])/2
                    pz = (zs[row]+zs[row+1])/2
                    for u in (pu-pane_width/2, pu+pane_width/2):
                        sash_parts.append(window_box("AwningSash", u, pz, bar*.5,
                                                      pane_height, window_frame_material))
                    for z in (pz-pane_height/2, pz+pane_height/2):
                        sash_parts.append(window_box("AwningSash", pu, z, pane_width,
                                                      bar*.5, window_frame_material))
                    from mathutils import Matrix
                    hinge = Vector(pane.location)
                    hinge.z += pane_height/2
                    rotation = Matrix.Rotation(math.radians(12), 4, 'X' if width_axis == 0 else 'Y')
                    transform = Matrix.Translation(hinge) @ rotation @ Matrix.Translation(-hinge)
                    for part in sash_parts:
                        part.matrix_world = transform @ part.matrix_world
        if style == "arched_transom":
            # Elliptical arch mullion inside the original rectangular opening.
            arch_base = float(zs[1])
            radius_u = (u1-u0)*0.48
            radius_z = (z1-arch_base)*0.9
            vertices, faces = [], []
            segments = 32
            for step in range(segments+1):
                angle = math.pi * step/segments
                for radial, normal in ((-1,-1), (1,-1), (1,1), (-1,1)):
                    point = center.copy()
                    point[width_axis] += (radius_u+radial*bar*.3)*math.cos(angle)
                    point[2] = arch_base+(radius_z+radial*bar*.3)*math.sin(angle)
                    point[normal_axis] += normal*frame_depth/2
                    vertices.append(point.tolist())
            for step in range(segments):
                for side in range(4):
                    a=step*4+side; b=step*4+(side+1)%4
                    faces.append((a,b,b+4,a+4))
            faces.extend([(3,2,1,0),tuple(segments*4+i for i in range(4))])
            mesh=bpy.data.meshes.new(f"WindowArchMesh_{window_index}")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            arch=bpy.data.objects.new(f"WindowArch_{window_index}", mesh)
            bpy.context.collection.objects.link(arch)
            arch.data.materials.append(window_frame_material)
            window_parts.append(arch)
        generated.extend(window_parts)
        if belongs_to_target:
            target_generated.extend(window_parts)
            glass_settings.append({
                "opening_uid": opening_uid,
                "exported_objects": [part.name for part in window_parts],
                "style": style,
                "frame_finish": finish,
                "frame_width_meters": bar,
                "frame_depth_meters": frame_depth,
                "opacity": opacity,
                "transmission": transmission,
                "roughness": roughness,
                "ior": ior,
                "tint_rgb": tint,
            })
        window_index += 1

    # Ignore the separate zero-height threshold meshes. Tall Door meshes define
    # the jamb/header bounds and receive one inset leaf each.
    if "door" in lower_name and height > 1.5 and width > 0.45:
        opening_uid = opening.get("front3d_mesh_uid")
        belongs_to_target = opening_uid in room_mesh_uids
        touches_target = door_touches_floor_boundary(center, width, depth, width_axis, normal_axis)
        export_with_target = touches_target
        if not export_with_target and opening in target_room_objects:
            target_room_objects.remove(opening)
        gap = 0.025
        slab_depth = 0.045
        leaf_width = width - 2 * gap
        door_rng = random.Random(f"door:{SCENE.stem}:{opening.name}")
        is_open = door_rng.random() < DOOR_OPEN_PROBABILITY
        # Keep open doors only slightly ajar so the neighboring room provides
        # context without exposing most of its contents.
        if not is_open:
            angle_degrees = 0.0
        elif door_rng.random() < DOOR_FULLY_OPEN_PROBABILITY:
            angle_degrees = 90.0
        else:
            angle_degrees = door_rng.uniform(DOOR_MIN_ANGLE, DOOR_MAX_ANGLE)
        hinge_at_high = door_rng.random() < 0.5
        swing_sign = -1.0 if door_rng.random() < 0.5 else 1.0

        width_direction = np.zeros(2)
        width_direction[width_axis] = -1.0 if hinge_at_high else 1.0
        normal_direction = np.array([-width_direction[1], width_direction[0]])
        angle = math.radians(angle_degrees) * swing_sign
        leaf_direction = (
            math.cos(angle) * width_direction
            + math.sin(angle) * normal_direction
        )
        hinge = center[:2].copy()
        hinge[width_axis] = (high[width_axis] - gap if hinge_at_high
                             else low[width_axis] + gap)
        leaf_center = center.copy()
        leaf_center[:2] = hinge + leaf_direction * leaf_width / 2
        leaf_center[2] = low[2] + (height - gap) / 2
        door_styles = ["flush", "two_panel", "six_panel", "glazed", "louvered"]
        door_style = os.environ.get("FRONT3D_DOOR_STYLE") or door_rng.choice(door_styles)
        if door_style not in door_styles:
            raise ValueError(f"Unknown door style: {door_style}")
        door_finish = door_rng.choice(["walnut", "oak", "painted_white", "sage", "charcoal"])
        door_colors = {"walnut": (0.18, 0.065, 0.022), "oak": (0.42, 0.25, 0.11),
                       "painted_white": (0.78, 0.76, 0.70), "sage": (0.22, 0.30, 0.24),
                       "charcoal": (0.045, 0.055, 0.065)}
        leaf_material = make_principled_material(
            f"DoorFinish_{door_index}_{door_finish}", door_colors[door_finish], roughness=0.38)
        hardware_finish = door_rng.choice(["brass", "steel", "black"])
        hardware_colors = {"brass": (0.42, 0.20, 0.055), "steel": (0.45, 0.48, 0.50),
                           "black": (0.025, 0.028, 0.03)}
        hardware_material = make_principled_material(
            f"DoorHardware_{door_index}", hardware_colors[hardware_finish],
            roughness=0.25, metallic=0.8)
        door_parts = []
        leaf_height = height-gap
        leaf_normal = np.array([-leaf_direction[1], leaf_direction[0]])
        leaf_angle = math.atan2(leaf_direction[1], leaf_direction[0])

        def door_box(name, x, y, z, w, d, h, material=leaf_material, tilt=0):
            position = leaf_center.copy()
            position[:2] += leaf_direction*x + leaf_normal*y
            position[2] += z
            part = add_box(f"Door{name}_{door_index}_{len(door_parts)}", position,
                           (w, d, h), material)
            part.rotation_euler = (tilt, 0, leaf_angle)
            door_parts.append(part)
            return part

        if door_style in {"flush", "two_panel", "six_panel"}:
            door_box("Leaf", 0, 0, 0, leaf_width, slab_depth, leaf_height)
            if door_style != "flush":
                columns = 2 if door_style == "six_panel" else 1
                rows = 3 if door_style == "six_panel" else 2
                panel_w = (leaf_width-0.20)/columns
                panel_h = (leaf_height-0.28)/rows
                for col in range(columns):
                    for row in range(rows):
                        x = (col-(columns-1)/2)*panel_w
                        z = (row-(rows-1)/2)*panel_h
                        for side in (-1, 1):
                            # Raised panels and narrow surrounding molding on both faces.
                            door_box("Panel", x, side*(slab_depth/2+0.004), z,
                                     panel_w-0.045, 0.008, panel_h-0.065)
                            for sign in (-1, 1):
                                door_box("PanelMolding", x+sign*(panel_w-0.045)/2,
                                         side*(slab_depth/2+0.01), z,
                                         0.018, 0.018, panel_h-0.065)
                                door_box("PanelMolding", x, side*(slab_depth/2+0.01),
                                         z+sign*(panel_h-0.065)/2,
                                         panel_w-0.045, 0.018, 0.018)
        else:
            stile = min(0.12, leaf_width*0.14)
            rail = 0.13
            inner_w = leaf_width-2*stile
            for sign in (-1, 1):
                door_box("Stile", sign*(leaf_width-stile)/2, 0, 0,
                         stile, slab_depth, leaf_height)
                door_box("Rail", 0, 0, sign*(leaf_height-rail)/2,
                         inner_w, slab_depth, rail)
            inner_h = leaf_height-2*rail
            if door_style == "glazed":
                panel_h = inner_h*0.30
                door_box("LowerPanel", 0, 0, -inner_h/2+panel_h/2,
                         inner_w, slab_depth, panel_h)
                door_box("GlassRail", 0, 0, -inner_h/2+panel_h,
                         inner_w, slab_depth, 0.065)
                glass_h = inner_h-panel_h-0.065
                glass_z = inner_h/2-glass_h/2
                door_glass = make_principled_material(
                    f"DoorGlassMaterial_{door_index}", (0.74, 0.82, 0.78),
                    roughness=0.20, transmission=0.92, alpha=0.32)
                door_box("Glass", 0, 0, glass_z, inner_w, 0.012, glass_h, door_glass)
                door_box("GlassMullion", 0, 0, glass_z, 0.035, slab_depth, glass_h)
            else:
                slat_count = max(6, int(inner_h/0.075))
                pitch = inner_h/slat_count
                for slat in range(slat_count):
                    door_box("Louver", 0, 0, -inner_h/2+(slat+0.5)*pitch,
                             inner_w, 0.018, pitch*1.12, tilt=math.radians(28))
        # Lever and backplate on each face, attached to the moving leaf.
        handle_x = leaf_width/2-0.15
        handle_z = low[2]+1.0-leaf_center[2]
        for side in (-1, 1):
            door_box("HandlePlate", handle_x, side*(slab_depth/2+0.006), handle_z,
                     0.045, 0.012, 0.16, hardware_material)
            door_box("HandleStem", handle_x, side*(slab_depth/2+0.028), handle_z,
                     0.025, 0.05, 0.025, hardware_material)
            door_box("HandleLever", handle_x-0.045, side*(slab_depth/2+0.055), handle_z,
                     0.12, 0.025, 0.025, hardware_material)
        generated.extend(door_parts)
        if export_with_target:
            target_generated.extend(door_parts)
            if opening not in target_room_objects:
                target_room_objects.append(opening)
            if not belongs_to_target:
                adjacent_door_openings.append(opening_uid)
            door_settings.append({
                "opening_uid": opening_uid,
                "assignment": "target_room" if belongs_to_target else "adjacent_room",
                "style": door_style,
                "finish": door_finish,
                "hardware_finish": hardware_finish,
                "exported_objects": [part.name for part in door_parts],
                "state": "open" if is_open else "closed",
                "angle_degrees": angle_degrees * swing_sign,
                "hinge_side": "high" if hinge_at_high else "low",
            })
        door_index += 1

render_width = int(os.environ.get("FRONT3D_WIDTH", "1036"))
render_samples = int(os.environ.get("FRONT3D_SAMPLES", "128"))
encode_workers = int(os.environ.get("FRONT3D_ENCODE_WORKERS", "4"))
num_views = int(os.environ.get("FRONT3D_VIEWS", "1"))
requested_view_count = num_views
adaptive_views = os.environ.get("FRONT3D_ADAPTIVE_VIEWS", "0") == "1"
room_floor_area = 0.0
if adaptive_views:
    # Project floor polygons into world XY; this avoids counting wall/roof area.
    for obj in target_sampler_objects:
        if not obj.get_name().lower().startswith("floor"):
            continue
        mesh_obj = obj.blender_obj
        for polygon in mesh_obj.data.polygons:
            coords = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in polygon.vertices]
            room_floor_area += abs(sum(
                a.x * b.y - b.x * a.y
                for a, b in zip(coords, coords[1:] + coords[:1])
            )) * 0.5
    if room_floor_area > 0:
        num_views = min(num_views, 3 if room_floor_area < 8 else 5 if room_floor_area < 16 else num_views)
    print(f"Adaptive views: floor area={room_floor_area:.2f} m², views={num_views}", flush=True)
if min(render_width, render_samples, encode_workers, num_views) < 1:
    raise ValueError("render width, samples, encode workers, and views must be positive")
camera_height_min = float(os.environ.get("FRONT3D_CAMERA_HEIGHT_MIN", "1.4"))
camera_height_max = float(os.environ.get("FRONT3D_CAMERA_HEIGHT_MAX", "1.7"))
camera_clearance = float(os.environ.get("FRONT3D_CAMERA_CLEARANCE", "0.45"))
camera_min_separation = float(os.environ.get("FRONT3D_CAMERA_MIN_SEPARATION", "0.6"))
camera_max_tries = int(os.environ.get("FRONT3D_CAMERA_MAX_TRIES", "10000"))
camera_seed = int(os.environ.get("FRONT3D_CAMERA_SEED", "42"))
camera_sampler = os.environ.get("FRONT3D_CAMERA_SAMPLER", "legacy").lower()
if camera_sampler not in {"legacy", "quality"}:
    raise ValueError("FRONT3D_CAMERA_SAMPLER must be 'legacy' or 'quality'")

# Follow BlenderProc's 3D-FRONT example: sample points above the target floor,
# then reject poses using its BVH obstacle and scene-coverage checks. Validate
# four perspective headings because the final camera is equirectangular.
random.seed(camera_seed)
np.random.seed(camera_seed)
bproc.camera.set_resolution(512, 512)
camera = bpy.context.scene.camera
camera.data.type = "PERSP"
camera.data.lens = 24
class FloorTriangleSampler:
    """Sample target floor triangles by projected area, without furniture filters."""

    def __init__(self, triangles):
        triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        edges_a = triangles[:, 1, :2] - triangles[:, 0, :2]
        edges_b = triangles[:, 2, :2] - triangles[:, 0, :2]
        areas = np.abs(edges_a[:, 0] * edges_b[:, 1] - edges_a[:, 1] * edges_b[:, 0]) * 0.5
        valid = np.isfinite(triangles).all(axis=(1, 2)) & np.isfinite(areas) & (areas > 1e-10)
        self.triangles = triangles[valid]
        areas = areas[valid]
        if not len(areas):
            raise RuntimeError("Target room has no nondegenerate floor triangles")
        self.probabilities = areas / areas.sum()

    def sample(self, height):
        triangle = self.triangles[np.random.choice(len(self.triangles), p=self.probabilities)]
        u, v = np.random.random(2)
        root_u = np.sqrt(u)
        point = (1 - root_u) * triangle[0] + root_u * (1 - v) * triangle[1] + root_u * v * triangle[2]
        point[2] += height
        return point


floor_triangles = []
for obj in target_sampler_objects:
    if not obj.get_name().lower().startswith("floor"):
        continue
    mesh_obj = obj.blender_obj
    mesh_obj.data.calc_loop_triangles()
    for triangle in mesh_obj.data.loop_triangles:
        floor_triangles.append([
            tuple(mesh_obj.matrix_world @ mesh_obj.data.vertices[index].co)
            for index in triangle.vertices
        ])
point_sampler = FloorTriangleSampler(floor_triangles)
print("floor_sampler=projected_triangle_area_v1", flush=True)
bvh_tree = bproc.object.create_bvh_tree_multi_objects([
    obj for obj in loaded if isinstance(obj, bproc.types.MeshObject)
])
special_objects = [
    obj.get_cp("category_id") for obj in target_sampler_objects
    if obj.has_cp("category_id") and any(
        token in obj.get_name().lower() for token in ("chair", "sofa", "table", "bed")
    )
]
camera_poses = []
camera_coverage_scores = []
sampling_attempts = 0
camera_sampling_started_at = time.perf_counter()
separation = camera_min_separation
while sampling_attempts < camera_max_tries and len(camera_poses) < num_views:
    sampling_attempts += 1
    if sampling_attempts % 2000 == 0:
        separation *= 0.8
    location = point_sampler.sample(random.uniform(camera_height_min, camera_height_max))
    if any(np.linalg.norm(location - np.asarray(pose)[:3, 3]) < separation for pose in camera_poses):
        continue
    yaw = random.uniform(0, 2 * math.pi)
    heading_checks = []
    coverage_scores = []
    if camera_sampler == "legacy":
        # The earlier dataset renderer accepted positions using one exact BVH
        # nearest-surface clearance query. FloorTriangleSampler already
        # guarantees that the XY point lies on the target room's floor.
        nearest = bvh_tree.find_nearest(Vector(location))
        if nearest is not None and nearest[3] < camera_clearance:
            continue
    else:
        for heading in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
            validation_pose = bproc.math.build_transformation_mat(
                location, [math.pi / 2, 0, yaw + heading]
            )
            heading_checks.append(bproc.camera.perform_obstacle_in_view_check(
                validation_pose, {"min": camera_clearance}, bvh_tree,
                sqrt_number_of_rays=8,
            ))
            coverage_scores.append(bproc.camera.scene_coverage_score(
                validation_pose, special_objects, special_objects_weight=5.0,
                sqrt_number_of_rays=8,
            ))
        if (not all(heading_checks) or max(coverage_scores) <= 0.4
                or sum(score > 0.1 for score in coverage_scores) < 2):
            continue
    camera_pose = bproc.math.build_transformation_mat(location, [math.pi / 2, 0, yaw])
    camera_poses.append(camera_pose)
    camera_coverage_scores.append(coverage_scores)
    bproc.camera.add_camera_pose(camera_pose)
if len(camera_poses) != num_views:
    raise RuntimeError(
        f"Found only {len(camera_poses)}/{num_views} valid camera poses in "
        f"{sampling_attempts} attempts for {ROOM_INSTANCE}"
    )
camera_sampling_seconds = time.perf_counter() - camera_sampling_started_at
print(
    f"Sampled {len(camera_poses)} camera poses in {camera_sampling_seconds:.3f}s "
    f"({sampling_attempts} attempts)",
    flush=True,
)

bproc.camera.set_resolution(render_width, render_width // 2)
camera.data.type = "PANO"
camera.data.panorama_type = "EQUIRECTANGULAR"

render_device = os.environ.get("FRONT3D_RENDER_DEVICE", "GPU").upper()
if render_device == "GPU":
    gpu_backend = os.environ.get("FRONT3D_GPU_BACKEND", "OPTIX").upper()
    if gpu_backend not in {"OPTIX", "CUDA"}:
        raise ValueError("FRONT3D_GPU_BACKEND must be OPTIX or CUDA")
    bproc.renderer.set_render_devices(desired_gpu_device_type=[gpu_backend])
    if bpy.context.scene.cycles.device != "GPU":
        raise RuntimeError("GPU rendering requested but no Cycles GPU was selected")
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        if device.type == "CPU":
            device.use = False
elif render_device == "CPU":
    bpy.context.scene.cycles.device = "CPU"
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        device.use = device.type == "CPU"
else:
    raise ValueError("FRONT3D_RENDER_DEVICE must be GPU or CPU")
bproc.renderer.set_max_amount_of_samples(render_samples)
denoiser = os.environ.get("FRONT3D_DENOISER", "OPTIX").upper()
bproc.renderer.set_denoiser(None if denoiser == "NONE" else denoiser)
bproc.renderer.set_light_bounces(diffuse_bounces=6, glossy_bounces=6, max_bounces=12,
                                 transmission_bounces=8, transparent_max_bounces=16)
world = bpy.context.scene.world or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bproc.world.set_world_background_hdr_img(
    str(HDRI), strength=HDRI_STRENGTH, rotation_euler=[0, 0, HDRI_ROTATION]
)

# BlenderProc's 3D-FRONT loader has already made ceilings and lamp materials
# emissive. Record the affected loaded objects and rely on them plus the HDRI;
# do not add a camera-centered fill light that flattens the illumination.
emissive_objects = []
for obj in loaded:
    emissive_materials = []
    for material in obj.blender_obj.data.materials:
        if material and material.use_nodes and any(
                node.type == "EMISSION" for node in material.node_tree.nodes):
            emissive_materials.append(material.name)
    if emissive_materials:
        mesh_uid = obj.get_cp("front3d_mesh_uid") if obj.has_cp("front3d_mesh_uid") else None
        emissive_objects.append({
            "object": obj.get_name(),
            "materials": emissive_materials,
            "room_id": obj.get_cp("room_id") if obj.has_cp("room_id") else None,
            "mesh_uid": mesh_uid,
        })

OUTPUT.mkdir(parents=True, exist_ok=True)
panorama_names = [
    f"{ROOM_INSTANCE}_panorama_{index:03d}.png" for index in range(num_views)
]
if os.environ.get("FRONT3D_EXPORT_ONLY") != "1" or any(
        not (OUTPUT / name).is_file() for name in panorama_names):
    rendered = bproc.renderer.render()

    def write_panorama(item):
        panorama_name, color = item
        iio.imwrite(OUTPUT / panorama_name, color)

    # PNG compression is independent for every view and runs after Cycles has
    # returned all frames. Keep this bounded because batch mode may have one
    # renderer active per GPU.
    with ThreadPoolExecutor(max_workers=min(encode_workers, num_views)) as executor:
        list(executor.map(write_panorama, zip(panorama_names, rendered["colors"])))
    # Preserve the original one-view filename for older dataset readers.
    if num_views == 1:
        iio.imwrite(OUTPUT / "panorama.png", rendered["colors"][0])

# Match Front3DSDFDataset and the existing cubemap tools: keys are panorama
# filenames and matrices are row-major Blender-axis world-to-camera poses.
extrinsics = {
    name: np.linalg.inv(np.asarray(pose)).tolist()
    for name, pose in zip(panorama_names, camera_poses)
}
(OUTPUT / "extrinsics.json").write_text(json.dumps(extrinsics, indent=2) + "\n")

# Export only the requested room, while the panorama above used the complete
# scene. Generated frames, panes, and door leaves are included only when their
# source opening belongs to the target room.
skip_glb = os.environ.get("FRONT3D_SKIP_GLB") == "1"
glb_texture_max_size = int(os.environ.get("FRONT3D_GLB_TEXTURE_MAX_SIZE", "0"))
glb_image_format = os.environ.get("FRONT3D_GLB_IMAGE_FORMAT", "AUTO").upper()
glb_image_quality = int(os.environ.get("FRONT3D_GLB_IMAGE_QUALITY", "80"))
glb_resized_images = []
if not skip_glb:
    # Rendering is already complete, so resizing these in-memory images only
    # affects the shared GLB and never the panorama quality.
    target_images = {}
    for obj in target_room_objects + target_generated:
        if not obj or obj.type != "MESH":
            continue
        for material in obj.data.materials:
            if not material or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    target_images[node.image.as_pointer()] = node.image
    if glb_texture_max_size > 0:
        for image in target_images.values():
            width, height = image.size[:]
            longest = max(width, height)
            if longest <= glb_texture_max_size:
                continue
            scale = glb_texture_max_size / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image.scale(*new_size)
            glb_resized_images.append({
                "image": image.name,
                "original_size": [width, height],
                "export_size": list(new_size),
            })
    bpy.ops.object.select_all(action="DESELECT")
    for obj in target_room_objects + target_generated:
        if obj and obj.name in bpy.context.scene.objects:
            obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUTPUT / "room.glb"),
        export_format="GLB",
        use_selection=True,
        export_image_format=glb_image_format,
        export_image_quality=glb_image_quality,
        export_image_webp_fallback=False,
    )

metadata = {
    "door_selection_version": 2,
    "window_geometry_version": 2,
    "door_geometry_version": 2,
    "lighting_version": 3,
    "time_of_day": os.environ.get("FRONT3D_TIME_OF_DAY", "day"),
    "environment": os.environ.get("FRONT3D_ENVIRONMENT", "unspecified"),
    "lighting_pipeline": "BlenderProc Front3DLoader emissive ceilings and lamps plus HDRI",
    "door_selection_method": "actual_floor_boundary_three_points",
    "render_device": bpy.context.scene.cycles.device,
    "render_samples": render_samples,
    "panorama_encode_workers": encode_workers,
    "resolution": [render_width, render_width // 2],
    "scene": SCENE.stem,
    "persistent_scene_reused": reuse_scene,
    "persistent_scene_room_number": scene_cache["rooms_processed"],
    "room": ROOM_INSTANCE,
    "room_index": room_index,
    "window_mesh_uids": sorted(window_uids),
    "camera_sampling": ("legacy Front3DPointInRoomSampler plus BVH clearance"
                        if camera_sampler == "legacy" else
                        "BlenderProc Front3DPointInRoomSampler plus obstacle and coverage checks"),
    "camera_sampler": camera_sampler,
    "floor_sampler": "projected_triangle_area_v1",
    "camera_sampling_attempts": sampling_attempts,
    "camera_sampling_seconds": camera_sampling_seconds,
    "camera_clearance_meters": camera_clearance,
    "camera_min_separation_meters": camera_min_separation,
    "camera_heading_coverage_scores": camera_coverage_scores,
    "camera_poses_blender_cam2world": [np.asarray(pose).tolist() for pose in camera_poses],
    "extrinsics": "extrinsics.json",
    "extrinsics_convention": "row_major_world_to_camera_blender_axes",
    "mesh_object_count": sum(obj.type == "MESH" for obj in bpy.context.scene.objects),
    "panorama": panorama_names[0],
    "panoramas": panorama_names,
    "view_count": num_views,
    "requested_view_count": requested_view_count,
    "adaptive_views": adaptive_views,
    "room_floor_area_m2": room_floor_area if adaptive_views else None,
    "glb": None if skip_glb else "room.glb",
    "glb_texture_max_size": glb_texture_max_size or None,
    "glb_image_format": glb_image_format,
    "glb_image_quality": glb_image_quality,
    "glb_resized_images": glb_resized_images,
    "panorama_geometry_scope": "full_scene",
    "glb_geometry_scope": "target_room_only",
    "glb_excluded_architecture_types": sorted(set(excluded_target_mesh_types)),
    "material_pipeline": "BlenderProc front_3d_with_improved_mat",
    "cc_texture_root": str(CC_TEXTURES),
    "generated_window_assemblies_in_scene": window_index,
    "generated_door_leaves_in_scene": door_index,
    "target_window_glass": glass_settings,
    "target_doors": door_settings,
    "adjacent_door_margin_meters": adjacent_door_margin,
    "adjacent_door_openings_in_glb": sorted(set(adjacent_door_openings)),
    "hdri": str(HDRI),
    "hdri_rotation_degrees": float(os.environ.get("FRONT3D_HDRI_ROTATION", "0")),
    "hdri_strength": HDRI_STRENGTH,
    "camera_centered_fill_light": False,
    "ceiling_emission_strength": CEILING_EMISSION_STRENGTH,
    "lamp_emission_strength": LAMP_EMISSION_STRENGTH,
    "emissive_object_count_in_scene": len(emissive_objects),
    "fixture_label_fallbacks_made_emissive": sorted(set(emissive_fixture_objects)),
    "target_room_emissive_objects": [
        entry for entry in emissive_objects
        if entry["room_id"] == room_index or entry["mesh_uid"] in room_mesh_uids
    ],
}
(OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
if os.environ.get("FRONT3D_PERSISTENT_WORKER") == "1":
    # Preserve the loaded dwelling but remove geometry and animation created
    # specifically for this room before the next room is processed.
    print("Cleaning per-room generated geometry", flush=True)
    for obj in generated:
        if obj and obj.name in bpy.data.objects:
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0 and material.name.startswith(("Window", "Door")):
            bpy.data.materials.remove(material)
    bproc.utility.reset_keyframes()
    print("Per-room cleanup complete", flush=True)
else:
    bproc.clean_up()
