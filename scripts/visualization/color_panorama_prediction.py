"""Project a saved inference panorama onto its mesh without changing geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import torch
import trimesh
import open3d as o3d

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--visibility-tolerance', type=float, default=0.002,
                        help='First-hit distance tolerance in recovered camera units (default: 0.002)')
    args = parser.parse_args()
    if args.visibility_tolerance <= 0 or not np.isfinite(args.visibility_tolerance):
        parser.error('visibility tolerance must be positive and finite')
    folder = args.directory.resolve()
    config = json.loads((folder / 'run.json').read_text())
    # Recover the camera transform from the saved point cloud and verify its
    # agreement with the quantized normalized conditioning surface.
    saved = torch.load(folder / 'latents.pt', map_location='cpu', weights_only=True)
    target = saved['surface'][0, :, :3].float().numpy()
    data = np.load(folder / 'pager_geometry.npz')
    points = data['points'].copy()
    points[~data['valid']] = np.nan
    raw = points.reshape(-1, 3)
    raw = raw[np.isfinite(raw).all(1)]
    mode = config['normalization']
    if mode in ('cube', 'unit_box'):
        center = (raw.min(0) + raw.max(0)) * 0.5
        scale = 2.0 / ((raw.max(0) - raw.min(0)).max() * (1.01 if mode == 'unit_box' else 1.0))
    elif mode in ('none', 'raw', 'skip'):
        center, scale = np.zeros(3), 1.0
    else:
        raise ValueError(f'Unsupported normalization: {mode}')
    distances, _ = cKDTree((raw - center) * scale).query(target)
    error = float(np.percentile(distances, 99))
    if error > 0.01:
        raise RuntimeError(f'Camera normalization does not match saved surface: {error}')
    mesh = trimesh.load(folder / 'prediction.ply', process=False)
    camera_vertices = np.asarray(mesh.vertices) / scale + center
    # Cast in recovered OpenCV camera coordinates: the panorama camera is at
    # the origin, not at the normalized decoder origin.
    vertices = np.asarray(camera_vertices, dtype=np.float32)
    distance = np.linalg.norm(vertices, axis=1)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices),
                        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    visible = np.zeros(len(vertices), dtype=bool)
    depth_gap = np.full(len(vertices), np.inf, dtype=np.float32)
    for start in range(0, len(vertices), 150000):
        stop = min(start + 150000, len(vertices))
        directions = vertices[start:stop] / np.maximum(distance[start:stop, None], 1e-10)
        rays = np.column_stack((np.zeros_like(directions), directions)).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)['t_hit'].numpy()
        depth_gap[start:stop] = np.abs(hit - distance[start:stop])
        visible[start:stop] = (np.isfinite(hit) &
            (depth_gap[start:stop] <= args.visibility_tolerance) &
            (distance[start:stop] > 1e-6))
    image = np.asarray(Image.open(folder / 'panorama.png').convert('RGB'))
    h, w = image.shape[:2]
    x, y, z = camera_vertices.T
    u = (np.arctan2(x, z) / (2 * np.pi) + 0.5) * w - 0.5
    v = (np.arctan2(y, np.hypot(x, z)) / np.pi + 0.5) * h - 0.5
    # Add seam columns for bilinear interpolation across ERP longitude wrap.
    image = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    coordinates = [np.clip(v, 0, h - 1), np.mod(u, w) + 1]
    colors = np.stack([map_coordinates(image[:, :, i], coordinates, order=1, mode='nearest') for i in range(3)], axis=1)
    hidden_rgb = [235, 114, 132]  # Match render_alignment_color_options.py coral.
    colors[~visible] = hidden_rgb
    mesh.visual.vertex_colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    np.savez_compressed(folder / 'prediction_colored_visibility.npz',
                        visible=visible, depth_gap=depth_gap)
    for extension in ('ply', 'glb'):
        mesh.export(folder / f'prediction_colored.{extension}')
    report = dict(source=str(folder / 'prediction.ply'), panorama=str(folder / 'panorama.png'),
                  method='ERP vertex color projection in recovered OpenCV camera coordinates',
                  normalization_center=center.tolist(), normalization_scale=float(scale),
                  normalization_validation_p99_distance=error, vertices=len(mesh.vertices), faces=len(mesh.faces),
                  geometry_modified=False, visibility='reconstructed_mesh_first_hit_vertex_distance',
                  visibility_tolerance=args.visibility_tolerance, camera_center=[0, 0, 0],
                  visible_vertices=int(visible.sum()), hidden_vertices=int((~visible).sum()),
                  visible_fraction=float(visible.mean()), hidden_rgb=hidden_rgb,
                  note='Panorama RGB on camera-visible vertices only; occluded vertices use Blender coral. No color propagation onto hidden geometry.')
    (folder / 'prediction_colored.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
