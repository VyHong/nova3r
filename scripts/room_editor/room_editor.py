#!/usr/bin/env python3
"""Portable ScanNet++ room selection tool. No repository imports required."""
import argparse
import json
from pathlib import Path

import numpy as np


def points_inside(points, polygon):
    points, polygon = np.asarray(points), np.asarray(polygon)
    inside = np.zeros(len(points), dtype=bool)
    boundary = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = end - start
        cross = (x - start[0]) * dy - (y - start[1]) * dx
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        boundary |= ((np.abs(cross) <= 1e-9 * length)
                     & (x >= min(start[0], end[0]) - 1e-9)
                     & (x <= max(start[0], end[0]) + 1e-9)
                     & (y >= min(start[1], end[1]) - 1e-9)
                     & (y <= max(start[1], end[1]) + 1e-9))
        if dy != 0:
            inside ^= (((start[1] > y) != (end[1] > y))
                       & (x < start[0] + (y - start[1]) * dx / dy))
    return inside | boundary


def room_face_mask(vertices, triangles, polygon, min_height, max_height):
    centers = vertices[triangles].mean(axis=1)
    return (points_inside(centers[:, :2], polygon)
            & (centers[:, 2] >= min_height) & (centers[:, 2] <= max_height))


class RoomEditor:
    def __init__(self, root, output, max_points):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox, PolygonSelector, CheckButtons
        import open3d as o3d
        self.plt, self.o3d = plt, o3d
        self.output, self.max_points = output, max_points
        self.scenes = sorted({p.parent for p in root.glob('*/test_camera_poses.json')})
        if (root / 'test_camera_poses.json').exists():
            self.scenes = [root]
        if not self.scenes:
            raise ValueError(f'No scene folders with test_camera_poses.json found in {root}')
        self.index = 0
        self.polygon = None
        self.saved_polygons = []
        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        self.fig.subplots_adjust(left=.07, right=.79, bottom=.1, top=.91)
        self.widgets = []
        def button(y, title, callback):
            w = Button(self.fig.add_axes([.81, y, .17, .045]), title)
            w.on_clicked(callback)
            self.widgets.append(w)
        self.name = TextBox(self.fig.add_axes([.85, .84, .13, .04]), 'Room ', initial='room_01')
        self.min_z = TextBox(self.fig.add_axes([.86, .77, .12, .04]), 'Min Z ')
        self.max_z = TextBox(self.fig.add_axes([.86, .70, .12, .04]), 'Max Z ')
        button(.62, 'Save room', self.save)
        button(.56, 'Preview room 3D', lambda _: self.preview(True))
        button(.50, 'Preview scene 3D', lambda _: self.preview(False))
        button(.44, 'Clear polygon', self.clear)
        button(.34, 'Previous scene', lambda _: self.change(-1))
        button(.28, 'Next scene', lambda _: self.change(1))
        self.roof = CheckButtons(self.fig.add_axes([.81, .16, .17, .07]), ['Hide top 0.3 m'], [True])
        self.roof.on_clicked(lambda _: self.draw())
        self.status = self.fig.text(.07, .035, '', fontsize=10, wrap=True)
        self.load()

    def message(self, text):
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def load(self):
        folder = self.scenes[self.index]
        mesh_path = folder / 'scans/mesh_aligned_0.05.ply'
        if not mesh_path.exists():
            mesh_path = folder / 'mesh_aligned_0.05.ply'
        self.mesh = self.o3d.io.read_triangle_mesh(str(mesh_path))
        if not self.mesh.has_triangles():
            raise ValueError(f'No triangle mesh in {mesh_path}')
        self.vertices = np.asarray(self.mesh.vertices)
        self.triangles = np.asarray(self.mesh.triangles)
        self.data = json.loads((folder / 'test_camera_poses.json').read_text())
        self.ids = list(self.data['frames'])
        self.is_test = np.array([self.data['frames'][key].get('is_test', True) for key in self.ids], dtype=bool)
        self.cameras = np.array([np.asarray(self.data['frames'][k]['scanner_camera_to_world'])[:3, 3]
                                 for k in self.ids])
        self.min_z.set_val(f'{self.vertices[:, 2].min():.5f}')
        self.max_z.set_val(f'{self.vertices[:, 2].max():.5f}')
        self.polygon = None
        self.saved_polygons = []
        self.name.set_val('room_01')
        self.draw()

    def draw(self):
        from matplotlib.widgets import PolygonSelector
        if hasattr(self, 'selector'):
            self.selector.disconnect_events()
        self.ax.clear()
        z = self.vertices[:, 2]
        limit = z.max() - .3 if self.roof.get_status()[0] else z.max() + 1
        eligible = np.flatnonzero(z <= limit)
        if len(eligible) > self.max_points:
            eligible = np.random.default_rng(42).choice(eligible, self.max_points, replace=False)
        xyz = self.vertices[eligible]
        if self.mesh.has_vertex_colors():
            colors = np.asarray(self.mesh.vertex_colors)[eligible]
        else:
            colors = xyz[:, 2]
        self.ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=.5, alpha=.6, rasterized=True)
        for mask, color, label in [(self.is_test, 'red', 'Test cameras'), (~self.is_test, 'dodgerblue', 'Other cameras')]:
            self.ax.scatter(self.cameras[mask, 0], self.cameras[mask, 1], marker='^', s=65,
                            c=color, edgecolors='white', zorder=5, label=label)
        self.ax.legend(loc='upper right')
        for frame, point in zip(self.ids, self.cameras):
            self.ax.annotate(frame, point[:2], xytext=(5, 5), textcoords='offset points',
                             color='darkred' if self.data['frames'][frame].get('is_test', True) else 'blue', fontsize=9, zorder=6)
        for polygon, name in self.saved_polygons:
            ring = np.vstack([polygon, polygon[0]])
            self.ax.plot(ring[:, 0], ring[:, 1], color='tab:green')
            self.ax.text(*np.mean(polygon, axis=0), name, color='tab:green')
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Scan X (m)')
        self.ax.set_ylabel('Scan Y (m)')
        self.ax.set_title(f'{self.scenes[self.index].name} ({self.index + 1}/{len(self.scenes)})\n'
                          'Click room boundary; click first vertex to close. Red = test cameras; blue = other cameras. Labels = frame IDs.')
        self.selector = PolygonSelector(self.ax, self.select, useblit=True)
        self.polygon = None
        self.message('Draw a room polygon. Selection extends vertically between Min Z and Max Z. '
                     'Roof hiding affects display only. Disable toolbar pan/zoom before drawing.')

    def select(self, polygon):
        self.polygon = np.asarray(polygon)
        self.message('Polygon ready. Set room name, then preview or save. Escape restarts drawing.')

    def selection(self):
        if self.polygon is None or len(self.polygon) < 3:
            raise ValueError('Close a polygon first by clicking its starting vertex.')
        lo, hi = float(self.min_z.text), float(self.max_z.text)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError('Min Z and Max Z must be finite; Max Z must exceed Min Z.')
        mask = room_face_mask(self.vertices, self.triangles, self.polygon, lo, hi)
        if not mask.any():
            raise ValueError('The selected region contains no mesh triangles.')
        return mask, lo, hi

    def selected_mesh(self, mask):
        import copy
        mesh = copy.deepcopy(self.mesh)
        mesh.remove_triangles_by_mask(~mask)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    def save(self, _):
        try:
            mask, lo, hi = self.selection()
            name = self.name.text.strip()
            if not name or name in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
                raise ValueError('Use letters, numbers, underscores or hyphens for the room name.')
            dest = self.output / self.scenes[self.index].name / name
            if dest.exists():
                raise ValueError(f'{dest} already exists; choose a different room name.')
            camera_mask = (points_inside(self.cameras[:, :2], self.polygon)
                           & (self.cameras[:, 2] >= lo) & (self.cameras[:, 2] <= hi))
            payload = dict(self.data)
            payload['frames'] = {k: self.data['frames'][k] for k, keep in zip(self.ids, camera_mask) if keep}
            payload['room'] = name
            payload['selection'] = {'polygon_xy': self.polygon.tolist(), 'min_z': lo, 'max_z': hi,
                                    'mesh_rule': 'Keep triangles whose centroids lie inside the vertical polygon prism.'}
            dest.mkdir(parents=True)
            if not self.o3d.io.write_triangle_mesh(str(dest / 'mesh.ply'), self.selected_mesh(mask)):
                raise OSError('Mesh export failed.')
            (dest / 'test_camera_poses.json').write_text(json.dumps(payload, indent=2) + '\n')
            (dest / 'test_images.txt').write_text(''.join(f'{self.scenes[self.index].name} {k}\n' for k in payload['frames'] if payload['frames'][k].get('is_test', True)))
            self.saved_polygons.append((self.polygon.copy(), name))
            self.draw()
            self.name.set_val(f'room_{len(self.saved_polygons) + 1:02d}')
            self.message(f'Saved {name}: {int(mask.sum())} triangles, {int(camera_mask.sum())} test images. {dest}')
        except (ValueError, OSError) as error:
            self.message(str(error))

    def preview(self, selected):
        try:
            mask, lo, hi = self.selection() if selected else (None, None, None)
            mesh = self.selected_mesh(mask) if selected else self.mesh
            geometries = [mesh]
            for frame, xyz in zip(self.ids, self.cameras):
                if selected and not (points_inside(xyz[None, :2], self.polygon)[0] and lo <= xyz[2] <= hi):
                    continue
                ball = self.o3d.geometry.TriangleMesh.create_sphere(radius=.07)
                ball.translate(xyz)
                ball.paint_uniform_color([1, 0, 0] if self.data['frames'][frame].get('is_test', True) else [0, .4, 1])
                geometries.append(ball)
            self.message('3D preview: red = test cameras; blue = other cameras. Close its window to resume editing.')
            self.o3d.visualization.draw_geometries(geometries, window_name='Room preview — red = test cameras; blue = other cameras',
                                                  mesh_show_back_face=True)
        except (ValueError, OSError) as error:
            self.message(str(error))

    def clear(self, _):
        self.draw()

    def change(self, delta):
        self.index = (self.index + delta) % len(self.scenes)
        try:
            self.load()
        except (ValueError, OSError) as error:
            self.message(str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path, help='Scenes parent folder or one scene folder')
    parser.add_argument('--output', type=Path, help='Default: <folder>/rooms')
    parser.add_argument('--max-points', type=int, default=250000, help='Top-down display point limit only')
    args = parser.parse_args()
    if args.folder is None:
        import tkinter as tk
        from tkinter.filedialog import askdirectory
        window = tk.Tk()
        window.withdraw()
        chosen = askdirectory(title='Select ScanNet++ test_scenes folder')
        window.destroy()
        if not chosen:
            return
        args.folder = Path(chosen)
    if args.max_points < 1:
        parser.error('--max-points must be positive')
    import matplotlib.pyplot as plt
    editor = RoomEditor(args.folder.resolve(), (args.output or args.folder / 'rooms').resolve(), args.max_points)
    plt.show()


if __name__ == '__main__':
    main()
