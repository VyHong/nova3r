"""Render an Open3D pick_mesh_view.py camera JSON with Blender Cycles."""
import argparse
import json
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('view', type=Path)
parser.add_argument('--mesh', type=Path)
parser.add_argument('--output', type=Path)
parser.add_argument('--transparent', action='store_true')
parser.add_argument('--soft-lighting', action='store_true', help='Use broad frontal fill and gentler shadows')
parser.add_argument('--exposure', type=float, default=-0.4, help='Exposure adjustment in stops')
parser.add_argument('--multi-fill', action='store_true', help='Add broad room and camera fill lights to lift shadows')
args = parser.parse_args()
view = json.loads(args.view.read_text())
mesh_path = args.mesh or args.view.parent/'test_mesh.ply'
output = args.output or args.view.with_name(args.view.stem+'_render.png')
preprocessing_path = mesh_path.with_suffix('.json')
preprocessing = json.loads(preprocessing_path.read_text()) if preprocessing_path.exists() else None
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=str(mesh_path.resolve()))
obj = bpy.context.object
obj.matrix_world = Matrix(view['mesh_matrix_world'])
obj.data.polygons.foreach_set('use_smooth',np.ones(len(obj.data.polygons),dtype=bool))
mat = bpy.data.materials.new('Medium grey')
mat.use_nodes = True
shader = mat.node_tree.nodes.get('Principled BSDF')
shader.inputs['Base Color'].default_value = (0.24,0.24,0.24,1)
shader.inputs['Roughness'].default_value = 0.85
obj.data.materials.clear()
obj.data.materials.append(mat)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64
scene.cycles.use_denoising = True
if args.soft_lighting:
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 8
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
gpus = [d for d in prefs.devices if d.type == 'OPTIX']
if not gpus:
    raise RuntimeError('No OptiX GPU available')
for device in prefs.devices:
    device.use = device.type == 'OPTIX'
scene.cycles.device = 'GPU'
scene.world = bpy.data.worlds.new('Soft neutral world')
scene.world.use_nodes = True
background = scene.world.node_tree.nodes['Background']
background.inputs['Color'].default_value = (0.8,0.8,0.8,1)
background.inputs['Strength'].default_value = 0.55 if args.soft_lighting else 0.35
scene.view_settings.view_transform = 'AgX'
scene.view_settings.exposure = args.exposure
width,height = view['width'],view['height']
k = np.asarray(view['intrinsic_matrix'])
fx,fy,cx,cy = k[0,0],k[1,1],k[0,2],k[1,2]
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = 1
scene.render.pixel_aspect_y = float(fx/fy)
bpy.ops.object.camera_add()
cam = bpy.context.object
cam.matrix_world = Matrix(view['blender_camera_matrix_world'])
cam.data.type = 'PERSP'
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36
cam.data.lens = float(fx*36/width)
# Open3D pixel centers are indexed from zero; Blender's frame uses pixel edges.
cam.data.shift_x = float((width/2-(cx+0.5))/width)
cam.data.shift_y = float(((cy+0.5)-height/2)*fx/fy/width)
cam.data.clip_start = 0.01
cam.data.clip_end = 1000
scene.camera = cam
bpy.context.view_layer.update()
# Check image-space framing against Open3D's calibration before rendering.
from bpy_extras.object_utils import world_to_camera_view
for local in [(0,0,3),(1,0.5,4),(-0.4,-0.2,2)]:
    world = cam.matrix_world @ Vector((local[0],-local[1],-local[2]))
    projected = world_to_camera_view(scene,cam,world)
    actual = np.array([projected.x*width-0.5,(1-projected.y)*height-0.5])
    expected = np.array([fx*local[0]/local[2]+cx,fy*local[1]/local[2]+cy])
    assert np.max(np.abs(actual-expected)) < 0.01,(actual,expected)
print('Saved camera reprojection verified within 0.01 pixels',flush=True)

def area(name, location, target, power, size):
    bpy.ops.object.light_add(type='AREA',location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = power
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (Vector(target)-light.location).to_track_quat('-Z','Y').to_euler()

position = cam.matrix_world.translation
forward = -cam.matrix_world.col[2].to_3d()
area('Soft camera fill',position+Vector((0,0,0.2)),position+forward*4,
     200 if args.soft_lighting else 120, 5 if args.soft_lighting else 2.5)
coords = np.asarray([obj.matrix_world@Vector(c) for c in obj.bound_box])
center = (coords.min(0)+coords.max(0))/2
area('Ceiling fill',(center[0],center[1],coords[:,2].max()-0.2),(center[0],center[1],coords[:,2].min()),
     80 if args.soft_lighting else 250, 5 if args.soft_lighting else 3)
if args.soft_lighting:
    # A small diffuse emission contribution lifts deep creases without changing
    # the mesh geometry or eliminating the shape cues from the area lights.
    shader.inputs['Emission Color'].default_value = (0.24,0.24,0.24,1)
    shader.inputs['Emission Strength'].default_value = 0.12
if args.multi_fill:
    right = cam.matrix_world.col[0].to_3d()
    up = cam.matrix_world.col[1].to_3d()
    target = position + forward*3
    # Place the softboxes just inside the viewing direction rather than
    # behind the camera, which can put lights outside the enclosing walls.
    for name, offset, power, size in [
        ('Broad frontal fill',forward*0.45,450,3.0),
        ('Left fill',forward*0.6-right*0.6+up*0.15,200,2.5),
        ('Right fill',forward*0.6+right*0.6+up*0.15,200,2.5),
        ('Upper fill',forward*0.6+up*0.35,150,2.5),
    ]:
        area(name,position+offset,target,power,size)
    # Spread ceiling illumination over the room instead of one strong key.
    for i, fraction in enumerate([0.25,0.5,0.75]):
        location = coords.min(0)+(coords.max(0)-coords.min(0))*fraction
        location[2] = coords[:,2].max()-0.3
        area(f'Room fill {i+1}',location,(location[0],location[1],coords[:,2].min()),120,2.5)
scene.render.film_transparent = args.transparent
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA' if args.transparent else 'RGB'
scene.render.filepath = str(output.resolve())
output.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.render.render(write_still=True)
output.with_suffix('.settings.json').write_text(json.dumps({
    'view_source':str(args.view),'mesh':str(mesh_path),'camera_matrix_world':[list(r) for r in cam.matrix_world],
    'resolution':[width,height],'lens_mm':cam.data.lens,'grey_linear':0.24,
    'transparent_background':args.transparent,'samples':64,'smooth_shading':True,
    'soft_lighting':args.soft_lighting,
    'exposure':args.exposure,
    'multi_fill':args.multi_fill,
    'geometry_modified':bool(preprocessing), 'preprocessing':preprocessing,'gpu':[d.name for d in gpus],
},indent=2)+'\n')
print(output,flush=True)
