"""Blender sidebar tool: open a mesh, frame a camera, and export a reproducible view.

Run in Blender's Scripting workspace with Run Script (Alt-P). In the 3D
viewport press N and open the Mesh View tab. No additional packages needed.

Headless replay:
  blender --background --python mesh_view_picker.py -- --view view.json \
      --render image.png
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Matrix, Vector

CAMERA_NAME = 'MeshView_Camera'


def camera(scene):
    obj = bpy.data.objects.get(CAMERA_NAME)
    if obj is None:
        obj = bpy.data.objects.new(CAMERA_NAME, bpy.data.cameras.new(CAMERA_NAME))
        scene.collection.objects.link(obj)
    scene.camera = obj
    return obj


def load_mesh(context, filepath, y_up=True):
    path = Path(filepath).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    bpy.ops.object.select_all(action='DESELECT')
    suffix = path.suffix.lower()
    if suffix == '.ply':
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == '.obj':
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis='NEGATIVE_Z', up_axis='Y')
    elif suffix in {'.glb', '.gltf'}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == '.stl':
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        raise ValueError('Supported formats: PLY, OBJ, GLB/GLTF, STL')
    objects = list(context.selected_objects)
    # Format importers already convert the axes of OBJ and glTF.
    rotate = y_up and suffix in {'.ply', '.stl'}
    transform = Matrix.Rotation(math.pi / 2, 4, 'X') if rotate else Matrix.Identity(4)
    roots = [obj for obj in objects if obj.parent not in objects]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world
    material = bpy.data.materials.get('MeshView_Grey') or bpy.data.materials.new('MeshView_Grey')
    material.diffuse_color = (0.24, 0.24, 0.24, 1)
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = material.diffuse_color
    shader.inputs['Roughness'].default_value = 0.85
    for index, obj in enumerate(objects):
        obj['mesh_view_source'] = str(path)
        obj['mesh_view_index'] = index
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(material)
    context.scene.mesh_view_source = str(path)
    context.scene.mesh_view_y_up = y_up
    context.scene.mesh_view_captured = False
    context.view_layer.update()
    return objects


def view_payload(scene):
    cam = camera(scene)
    data = cam.data
    return {
        'schema': 'nova3r.mesh_view.v1',
        'mesh_path': scene.mesh_view_source,
        'source_y_up': scene.mesh_view_y_up,
        'coordinate_system': 'Blender world, Z up; camera local -Z forward, +Y up',
        'objects': [
            {'name': obj.name, 'matrix_world': [list(row) for row in obj.matrix_world]}
            for obj in sorted(scene.objects, key=lambda o: o.get('mesh_view_index', -1))
            if obj.get('mesh_view_source') == scene.mesh_view_source
        ],
        'camera': {
            'matrix_world': [list(row) for row in cam.matrix_world],
            'location': list(cam.location),
            'rotation_euler_degrees': [math.degrees(a) for a in cam.rotation_euler],
            'type': data.type,
            'lens_mm': data.lens,
            'sensor_width_mm': data.sensor_width,
            'sensor_height_mm': data.sensor_height,
            'sensor_fit': data.sensor_fit,
            'shift_x': data.shift_x,
            'shift_y': data.shift_y,
            'ortho_scale': data.ortho_scale,
            'clip_start': data.clip_start,
            'clip_end': data.clip_end,
        },
        'render': {
            'resolution_x': scene.render.resolution_x,
            'resolution_y': scene.render.resolution_y,
            'resolution_percentage': scene.render.resolution_percentage,
            'pixel_aspect_x': scene.render.pixel_aspect_x,
            'pixel_aspect_y': scene.render.pixel_aspect_y,
        },
    }


def restore_view(context, payload, mesh_override=None):
    objects = load_mesh(context, mesh_override or payload['mesh_path'], payload['source_y_up'])
    saved = payload.get('objects', [])
    if len(saved) != len(objects):
        raise ValueError('Imported object count differs from saved view')
    for obj, state in zip(objects, saved):
        obj.matrix_world = Matrix(state['matrix_world'])
    cam = camera(context.scene)
    state = payload['camera']
    cam.matrix_world = Matrix(state['matrix_world'])
    for key in ['type', 'sensor_fit', 'shift_x', 'shift_y', 'ortho_scale', 'clip_start', 'clip_end']:
        setattr(cam.data, key, state[key])
    for key, source in [('lens', 'lens_mm'), ('sensor_width', 'sensor_width_mm'), ('sensor_height', 'sensor_height_mm')]:
        setattr(cam.data, key, state[source])
    for key, value in payload['render'].items():
        setattr(context.scene.render, key, value)
    context.view_layer.update()
    return cam


class MESHVIEW_OT_open(bpy.types.Operator, ImportHelper):
    bl_idname = 'mesh_view.open'
    bl_label = 'Open Mesh'
    filename_ext = '.ply'
    filter_glob: StringProperty(default='*.ply;*.obj;*.glb;*.gltf;*.stl', options={'HIDDEN'})
    y_up: BoolProperty(name='Source is Y-up (PLY/STL)', default=True)

    def execute(self, context):
        try:
            # Isolate the mesh from the startup cube and any existing project.
            scene = bpy.data.scenes.new('Mesh View')
            context.window.scene = scene
            scene.render.resolution_x = 1400
            scene.render.resolution_y = 1050
            scene.render.resolution_percentage = 100
            load_mesh(context, self.filepath, self.y_up)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    region = next(r for r in area.regions if r.type == 'WINDOW')
                    with context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_selected(use_all_regions=False)
                    area.spaces.active.clip_end = 1000
                    area.spaces.active.shading.color_type = 'MATERIAL'
            self.report({'INFO'}, 'Mesh opened. Navigate, then Capture Current View.')
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class MESHVIEW_OT_capture(bpy.types.Operator):
    bl_idname = 'mesh_view.capture'
    bl_label = 'Capture Current View'
    bl_description = 'Align the export camera to this viewport; then adjust within the camera frame'

    def execute(self, context):
        scene = context.scene
        cam = camera(scene)
        space = context.space_data
        if space.region_3d.view_perspective != 'CAMERA':
            cam.data.lens = space.lens
            cam.data.type = 'ORTHO' if space.region_3d.view_perspective == 'ORTHO' else 'PERSP'
            region = next(r for r in context.area.regions if r.type == 'WINDOW')
            with context.temp_override(region=region):
                bpy.ops.view3d.camera_to_view()
        space.lock_camera = True
        space.region_3d.view_perspective = 'CAMERA'
        scene.mesh_view_captured = True
        self.report({'INFO'}, 'Camera frame is the exact output. Navigate to adjust, then Export View JSON.')
        return {'FINISHED'}


class MESHVIEW_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = 'mesh_view.export'
    bl_label = 'Export View JSON'
    filename_ext = '.json'
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})

    def execute(self, context):
        if not context.scene.mesh_view_captured:
            self.report({'ERROR'}, 'Capture Current View first, then check the camera frame')
            return {'CANCELLED'}
        context.view_layer.update()
        payload = view_payload(context.scene)
        Path(self.filepath).write_text(json.dumps(payload, indent=2)+'\n')
        context.window_manager.clipboard = json.dumps(payload, indent=2)
        self.report({'INFO'}, 'View saved; JSON also copied to clipboard')
        return {'FINISHED'}


class MESHVIEW_PT_panel(bpy.types.Panel):
    bl_label = 'Exact Mesh View'
    bl_idname = 'MESHVIEW_PT_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mesh View'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.operator('mesh_view.open', icon='FILE_FOLDER')
        layout.prop(scene.render, 'resolution_x', text='Width')
        layout.prop(scene.render, 'resolution_y', text='Height')
        layout.operator('mesh_view.capture', icon='CAMERA_DATA')
        layout.prop(context.space_data, 'lock_camera', text='Navigate camera')
        cam = scene.camera
        if cam:
            layout.prop(cam.data, 'type', text='Projection')
            if cam.data.type == 'ORTHO':
                layout.prop(cam.data, 'ortho_scale')
            else:
                layout.prop(cam.data, 'lens', text='Lens (mm)')
            layout.label(text='Position (Blender world):')
            layout.prop(cam, 'location', text='')
            layout.label(text='Rotation:')
            layout.prop(cam, 'rotation_euler', text='')
        layout.operator('mesh_view.export', icon='EXPORT')
        layout.label(text='Share the JSON file or clipboard text.')
        layout.label(text='Only the camera frame is exported.')


CLASSES = [MESHVIEW_OT_open, MESHVIEW_OT_capture, MESHVIEW_OT_export, MESHVIEW_PT_panel]


def register():
    for cls in CLASSES:
        old = getattr(bpy.types, cls.__name__, None)
        if issubclass(cls, bpy.types.Operator):
            prefix, suffix = cls.bl_idname.split('.')
            old = getattr(bpy.types, prefix.upper() + '_OT_' + suffix, None) or old
        if old is None and cls.is_registered:
            old = cls
        if old:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_view_source = StringProperty(subtype='FILE_PATH')
    bpy.types.Scene.mesh_view_y_up = BoolProperty(default=True)
    bpy.types.Scene.mesh_view_captured = BoolProperty(default=False)
    scene = bpy.context.scene
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1050
    scene.render.resolution_percentage = 100


if __name__ == '__main__':
    register()
    if '--' in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--view', type=Path, required=True)
        parser.add_argument('--mesh', help='Override mesh path when moving between machines')
        parser.add_argument('--render', type=Path)
        args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
        bpy.ops.wm.read_factory_settings(use_empty=True)
        register()
        restore_view(bpy.context, json.loads(args.view.read_text()), args.mesh)
        if args.render:
            scene = bpy.context.scene
            scene.render.engine = 'BLENDER_WORKBENCH'
            scene.display.shading.light = 'STUDIO'
            scene.display.shading.color_type = 'MATERIAL'
            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = str(args.render.resolve())
            bpy.ops.render.render(write_still=True)
    elif not bpy.app.background:
        bpy.ops.mesh_view.open('INVOKE_DEFAULT')
