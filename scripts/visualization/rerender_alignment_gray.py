"""Rerender the saved aligned-mesh closeup in Blender with neutral hidden vertices."""
import json
import math
from pathlib import Path
import bpy
import numpy as np
import trimesh
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(globals().get('ALIGNMENT_BASE', ROOT / 'artifacts/scannetpp_09c1414f1b_00003_4/manual_panorama_alignment'))
OUT = BASE / 'qualitative_view04_closeups/view04_alt_06_az235_el33_mesh_gray_preprocessed.png'
mesh = trimesh.load(BASE / 'mesh_panorama_aligned.ply', process=False)
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.faces)
c = np.array(mesh.visual.vertex_colors)
hidden = np.all(c[:, :3] == [112, 139, 157], axis=1)
c[hidden, :3] = [178, 178, 178]
f = f[np.all(v[f, 1] <= 0.7705303430557251, axis=1)]
tri = v[f]
valid = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1) > 1e-12
removed_degenerate = int((~valid).sum())
f = f[valid]
used, inv = np.unique(f, return_inverse=True)
v = v[used][:, [0, 2, 1]]
v[:, 1] *= -1
c = c[used].astype(np.float32) / 255
bpy.ops.wm.read_factory_settings(use_empty=True)
data = bpy.data.meshes.new('Roof-cut aligned mesh')
data.from_pydata(v.tolist(), [], inv.reshape(-1, 3).tolist())
data.update()
data.polygons.foreach_set('use_smooth', np.ones(len(data.polygons), dtype=bool))
obj = bpy.data.objects.new('Panorama mesh, gray occlusions', data)
bpy.context.collection.objects.link(obj)
attr = data.color_attributes.new(name='Color', type='BYTE_COLOR', domain='POINT')
# Blender stores the byte attribute in sRGB; use its explicit sRGB accessor.
attr.data.foreach_set('color_srgb', c.ravel())
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 64
scene.cycles.use_denoising = True
mat = bpy.data.materials.new('Vertex colors with soft illumination')
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
color = nodes.new('ShaderNodeVertexColor')
color.layer_name = 'Color'
shader = nodes.new('ShaderNodeBsdfDiffuse')
output = nodes.new('ShaderNodeOutputMaterial')
mat.node_tree.links.new(color.outputs['Color'], shader.inputs['Color'])
mat.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
obj.data.materials.append(mat)
scene.display.shading.light = 'STUDIO'
scene.display.shading.studiolight_rotate_z = 0
scene.display.shading.color_type = 'VERTEX'
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = False
scene.display.shading.show_specular_highlight = False
scene.display.shading.background_type = 'WORLD'
scene.world = bpy.data.worlds.new('White')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.94,0.94,0.94,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.8
scene.view_settings.view_transform = 'Standard'
scene.render.resolution_x = 1600
scene.render.resolution_y = 1400
scene.render.resolution_percentage = 100
lo, hi = v.min(axis=0), v.max(axis=0)
center = (lo + hi) / 2
az, el = math.radians(235), math.radians(33)
direction = Vector((math.cos(el)*math.sin(az), -math.cos(el)*math.cos(az), math.sin(el)))
bpy.ops.object.camera_add(location=Vector(center) + direction * 15)
cam = bpy.context.object
cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
cam.data.type = 'ORTHO'
# Fit the cropped geometry in the camera plane, using the saved closeup factor.
rot = cam.rotation_euler.to_matrix()
projected = (v - center) @ np.array(rot)
span = np.ptp(projected, axis=0)
cam.data.ortho_scale = max(span[1], span[0] / (1600/1400)) * 1.19348
# Reference corner registration: corrected projection center in image pixels.
units_per_pixel = cam.data.ortho_scale / 1600
cam.location += rot @ Vector((0.02 * units_per_pixel, -26.1 * units_per_pixel, 0))
scene.camera = cam
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(OUT)
scene.display.render_aa = '32'
bpy.ops.render.render(write_still=True)
OUT.with_suffix('.json').write_text(json.dumps(dict(source=str(BASE/'mesh_panorama_aligned.ply'), blender=bpy.app.version_string, hidden_rgb=[178,178,178], replaced_vertices=int(hidden.sum()), preprocessing=dict(degenerate_faces_removed=removed_degenerate, smooth_shading=True, vertex_positions_unchanged=True), roof_cut_y=0.7705303430557251, azimuth=235, elevation=33, framing_scale=1.19348, camera_location=list(cam.location), ortho_scale=cam.data.ortho_scale), indent=2)+'\n')
print(OUT)
