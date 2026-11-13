"""Render neutral-grey interior and roof-cut exterior views of a Y-up room mesh."""
import argparse
import json
import math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mesh', type=Path)
parser.add_argument('--skip-existing', action='store_true')
parser.add_argument('--output-name', default='blender_grey_views_medium')
parser.add_argument('--camera-fill', type=float, default=0, help='Soft interior light power in watts at each viewpoint')
args = parser.parse_args()
folder = args.mesh.parent / args.output_name
folder.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=str(args.mesh.resolve()))
obj = bpy.context.object
obj.rotation_euler.x = math.pi / 2
mat = bpy.data.materials.new('Neutral grey clay')
mat.diffuse_color = (0.24, 0.24, 0.24, 1)
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (0.24, 0.24, 0.24, 1)
bsdf.inputs['Roughness'].default_value = 0.85
obj.data.materials.clear()
obj.data.materials.append(mat)
obj.data.polygons.foreach_set('use_smooth', np.ones(len(obj.data.polygons), dtype=bool))
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 48
scene.cycles.use_denoising = True
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
gpus = [d for d in prefs.devices if d.type == 'OPTIX']
if not gpus:
    raise RuntimeError('No OptiX GPU available')
for d in prefs.devices:
    d.use = d.type == 'OPTIX'
scene.cycles.device = 'GPU'
print('GPU:', [d.name for d in gpus], flush=True)
scene.world = bpy.data.worlds.new('Soft studio')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.35
scene.view_settings.view_transform = 'AgX'
scene.view_settings.exposure = -0.4
scene.render.resolution_x = 1400
scene.render.resolution_y = 1050
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'

def area(name, location, energy, size, target):
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (Vector(target)-light.location).to_track_quat('-Z', 'Y').to_euler()
    return light

bpy.context.view_layer.update()
coords = np.array([obj.matrix_world @ Vector(c) for c in obj.bound_box])
lo,hi = coords.min(0),coords.max(0)
center = (lo+hi)/2
interior_lights = [area('Interior fill', (0,0,0.55), 180, 1.8, (0,0,-1)), area('Ceiling fill', (center[0],center[1],hi[2]-0.2), 250, 3, (center[0],center[1],lo[2]))]
studio_lights = [area('Studio key',center+np.array((-4,-5,8)),1600,7,center), area('Studio fill',center+np.array((5,3,6)),1100,6,center)]
for light in studio_lights:
    light.hide_render = True
bpy.ops.object.camera_add(location=(0,0,0))
cam = bpy.context.object
scene.camera = cam
cam.data.clip_start = 0.04
cam.data.clip_end = 200
manifest = {'source': str(args.mesh), 'gpu': [d.name for d in gpus], 'grey_linear': 0.24, 'exposure': -0.4, 'world_strength': 0.35, 'views': {}}
camera_fill = area('Camera soft fill', (0,0,0), args.camera_fill, 2.5, (0,1,0)) if args.camera_fill else None
manifest['camera_fill_watts'] = args.camera_fill

def render(name, location, target, ortho=None):
    cam.location = location
    cam.rotation_euler = (Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type = 'ORTHO' if ortho else 'PERSP'
    if ortho:
        cam.data.ortho_scale = ortho
    else:
        cam.data.lens = 20
    if camera_fill:
        camera_fill.hide_render = bool(ortho)
        camera_fill.location = cam.location + Vector((0,0,0.2))
        camera_fill.rotation_euler = (Vector(target)-camera_fill.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath = str((folder/(name+'.png')).resolve())
    if not args.skip_existing or not Path(scene.render.filepath).exists():
        bpy.ops.render.render(write_still=True)
    manifest['views'][name] = {'camera_location':list(cam.location),'target':list(target),'ortho_scale':ortho}
    (folder/'render_settings.json').write_text(json.dumps(manifest,indent=2,default=float)+'\n')

# Find independent eye-level positions with floor support and wall clearance.
depsgraph = bpy.context.evaluated_depsgraph_get()
eye_z = float(lo[2]+1.5)
candidates = []
for x in np.linspace(lo[0]+0.7,hi[0]-0.7,13):
    for y in np.linspace(lo[1]+0.7,hi[1]-0.7,13):
        pos = Vector((float(x),float(y),eye_z))
        hit,point,*_ = scene.ray_cast(depsgraph,pos,Vector((0,0,-1)),distance=2)
        if not hit or not 1.1 < pos.z-point.z < 1.9:
            continue
        distances = []
        for az in np.linspace(0,2*math.pi,16,endpoint=False):
            direction = Vector((math.cos(az),math.sin(az),0))
            hit,point,*_ = scene.ray_cast(depsgraph,pos,direction,distance=12)
            distances.append((point-pos).length if hit else 12)
        if min(distances) > 0.6 and sum(d<11.9 for d in distances) >= 12:
            candidates.append((pos,distances))
if len(candidates) < 4:
    raise RuntimeError('Could not find four clear interior camera positions')
selected = []
for i in range(4):
    if selected:
        pos,distances = max(candidates,key=lambda c:min((c[0]-p).length for p in selected))
    else:
        pos,distances = max(candidates,key=lambda c:sum(min(d,5) for d in c[1]))
    selected.append(pos)
    candidates = [c for c in candidates if (c[0]-pos).length > 0.1]
    # Open windows must not score as long unobstructed views into the room.
    distances = [d if d < 11.9 else 0.5 for d in distances]
    az = int(np.argmax([min(distances[(j+k)%16] for k in [-2,-1,0,1,2])*4 + sum(min(distances[(j+k)%16],6) for k in [-2,-1,0,1,2]) for j in range(16)]))*2*math.pi/16
    target = pos+Vector((math.cos(az)*4,math.sin(az)*4,-0.35))
    render(f'inside_{i+1:02d}',pos,target)
for light in interior_lights:
    light.hide_render = True
for light in studio_lights:
    light.hide_render = False
# Keep complete triangles below the ceiling; leave the source PLY untouched.
mesh = obj.data
verts = np.empty(len(mesh.vertices)*3,dtype=np.float32)
mesh.vertices.foreach_get('co',verts)
verts = verts.reshape(-1,3)
indices = np.empty(len(mesh.polygons)*3,dtype=np.int32)
mesh.loops.foreach_get('vertex_index',indices)
faces = indices.reshape(-1,3)
cut_height = float(lo[2]+0.84*(hi[2]-lo[2]))
faces = faces[np.all(verts[faces,1] < cut_height,axis=1)]
used,inv = np.unique(faces,return_inverse=True)
cut = bpy.data.meshes.new('Ceiling removed')
cut.from_pydata(verts[used].tolist(),[],inv.reshape(-1,3).tolist())
cut.update()
obj.data = cut
cut.materials.append(mat)
cut.polygons.foreach_set('use_smooth',np.ones(len(cut.polygons),dtype=bool))
manifest['exterior_ceiling_cut_height'] = cut_height
visible_world = verts[used][:,[0,2,1]].copy()
visible_world[:,1] *= -1
fit_lo,fit_hi = np.percentile(visible_world,[1,99],axis=0)
center = (fit_lo+fit_hi)/2
for i,az in enumerate([225,45]):
    a = math.radians(az)
    direction = Vector((math.sin(a)*0.67,math.cos(a)*0.67,0.74)).normalized()
    cam.rotation_euler = (-direction).to_track_quat('-Z','Y').to_euler()
    projected = (visible_world-center) @ np.array(cam.rotation_euler.to_matrix())
    projected_lo,projected_hi = np.percentile(projected,[0.5,99.5],axis=0)
    span = projected_hi-projected_lo
    scale = max(span[0],span[1]*1400/1050)*1.12
    render(f'outside_{i+1:02d}_cutaway',Vector(center)+direction*18,center,scale)
from PIL import Image,ImageDraw
canvas = Image.new('RGB',(1800,960),'white')
draw = ImageDraw.Draw(canvas)
for i,name in enumerate(manifest['views']):
    x,y = (i%3)*600,(i//3)*480
    with Image.open(folder/(name+'.png')) as im:
        canvas.paste(im.resize((600,450)),(x,y+30))
    draw.text((x+15,y+8),name.replace('_',' ').title(),fill='black')
canvas.save(folder/'comparison.jpg',quality=95)
print(folder/'comparison.jpg',flush=True)
