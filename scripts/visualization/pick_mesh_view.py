#!/usr/bin/env python3
"""Open a mesh in Open3D; press P to save the current view for Blender rendering."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def camera_payload(parameters, mesh_path, mesh_transform):
    """Open3D uses world-to-camera OpenCV axes; Blender cameras look along -Z."""
    extrinsic = np.asarray(parameters.extrinsic)
    camera_to_world = np.linalg.inv(extrinsic)
    return {
        'mesh_path': str(mesh_path),
        'mesh_matrix_world': mesh_transform.tolist(),
        'width': parameters.intrinsic.width,
        'height': parameters.intrinsic.height,
        'intrinsic_matrix': np.asarray(parameters.intrinsic.intrinsic_matrix).tolist(),
        'extrinsic_world_to_camera': extrinsic.tolist(),
        'camera_position': camera_to_world[:3, 3].tolist(),
        'look_direction': camera_to_world[:3, 2].tolist(),
        'up_direction': (-camera_to_world[:3, 1]).tolist(),
        'blender_camera_matrix_world': (camera_to_world @ np.diag([1., -1., -1., 1.])).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path, help='Mesh file, e.g. test_mesh.ply')
    parser.add_argument('--output', type=Path, help='Output JSON (default: mesh folder/view.json)')
    parser.add_argument('--z-up', action='store_true', help='Mesh is already Z-up; skip NOVA3R Y-up conversion')
    args = parser.parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.is_file():
        parser.error(f'Mesh not found: {mesh_path}')
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or not mesh.has_triangles():
        parser.error('File does not contain a triangle mesh')
    transform = np.eye(4)
    if not args.z_up:
        transform[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        mesh.transform(transform)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    output = (args.output or mesh_path.parent / 'view.json').expanduser().resolve()
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    if not viewer.create_window('Mesh view — P: save camera | Esc: close', width=1280, height=960):
        raise RuntimeError('Could not open a window. Run this on a desktop with a display.')
    viewer.add_geometry(mesh)
    options = viewer.get_render_option()
    options.mesh_show_back_face = True
    options.background_color = np.asarray([0.18, 0.18, 0.18])
    control = viewer.get_view_control()
    control.set_lookat(mesh.get_center())
    control.set_up([0, 0, 1])
    control.set_front([1, -1, 0.8])
    control.set_zoom(0.7)

    def save_view(vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if params.intrinsic.width <= 0 or not np.isfinite(params.extrinsic).all():
            print('Could not capture this camera. Use a perspective view and try again.')
            return False
        payload = camera_payload(params, mesh_path, transform)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2)+'\n')
        print(f'\nSaved camera: {output}\n'
              f'Position: {payload["camera_position"]}\n'
              f'Look direction: {payload["look_direction"]}\n'
              f'Up direction: {payload["up_direction"]}\n'
              'Send the JSON file to reproduce this exact view.\n', flush=True)
        return False

    viewer.register_key_callback(ord('P'), save_view)
    print('Left drag: orbit | Ctrl + left drag: pan | Wheel: zoom\n'
          'Press P to save the view; Esc to close. Saving again replaces view.json.', flush=True)
    try:
        viewer.run()
    finally:
        viewer.destroy_window()


if __name__ == '__main__':
    main()
