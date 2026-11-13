#!/usr/bin/env python3
"""Manually align an ERP panorama camera to a mesh; never modify mesh geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

def sample_rgb(rgb,u,v):
    h,w=rgb.shape[:2]
    x=(u*w-.5)%w; y=np.clip(v*h-.5,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=(x0+1)%w;y1=np.minimum(y0+1,h-1)
    a=(x-x0)[:,None];b=(y-y0)[:,None]
    return np.rint((1-b)*((1-a)*rgb[y0,x0]+a*rgb[y0,x1])+b*((1-a)*rgb[y1,x0]+a*rgb[y1,x1])).astype(np.uint8)

HIDDEN_RGB = [112, 139, 157]  # Muted slate blue (#708b9d).


def extend_color_to_local_occlusions(vertices, visible, depth_gap,
                                     radius=.08, max_gap=.10):
    """Color shallow occluded points within 8 cm of a visible surface point."""
    from scipy.spatial import cKDTree
    result = visible.copy()
    candidates = np.flatnonzero(~visible & np.isfinite(depth_gap) & (depth_gap <= max_gap))
    tree = cKDTree(vertices[visible])
    for start in range(0, len(candidates), 150000):
        ids = candidates[start:start+150000]
        distances, _ = tree.query(vertices[ids], distance_upper_bound=radius, workers=4)
        result[ids[np.isfinite(distances)]] = True
    return result


def color_small_hidden_islands(vertices, faces, visible, max_area=.005, radius=.04):
    """Fill small hidden color islands near RGB surfaces, independent of ray depth."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    hidden = np.flatnonzero(~visible)
    index = np.full(len(vertices), -1, np.int32)
    index[hidden] = np.arange(len(hidden), dtype=np.int32)
    rows, cols = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        x, y = index[faces[:, a]], index[faces[:, b]]
        keep = (x >= 0) & (y >= 0)
        rows.append(x[keep]); cols.append(y[keep])
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row), bool), (row, col)), shape=(len(hidden), len(hidden))).tocsr()
    count, labels = connected_components(graph, directed=False)
    # Associate each vertex with a third of its adjacent triangle areas.
    vertex_area = np.zeros(len(vertices), np.float64)
    for start in range(0, len(faces), 250000):
        f = faces[start:start+250000]
        tri = vertices[f]
        area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)/6
        for j in range(3):
            np.add.at(vertex_area, f[:,j], area)
    areas = np.bincount(labels, weights=vertex_area[hidden], minlength=count)
    candidates = hidden[areas[labels] <= max_area]
    tree = cKDTree(vertices[visible])
    result = visible.copy()
    for start in range(0, len(candidates), 150000):
        ids = candidates[start:start+150000]
        distance, _ = tree.query(vertices[ids], distance_upper_bound=radius, workers=4)
        result[ids[np.isfinite(distance)]] = True
    return result



def project(vertices, center, angles):
    """Camera rotation: scipy xyz Euler degrees, Hunyuan camera to mesh frame."""
    relative = (vertices - center) @ Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    distance = np.linalg.norm(relative, axis=1)
    cv = relative * [1, -1, -1]
    u = (np.arctan2(cv[:, 0], cv[:, 2]) / (2*np.pi) + .5) % 1
    v = np.arcsin(np.clip(cv[:, 1]/np.maximum(distance, 1e-10), -1, 1))/np.pi + .5
    return u, v


def fit_landmark_pose(points, bearings, initial):
    """Robust spherical pose fit; bearings are fixed in panorama camera axes."""
    from scipy.optimize import least_squares
    points, bearings = np.asarray(points, float), np.asarray(bearings, float)
    if points.shape != bearings.shape or points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        raise ValueError('Pick at least 6 complete landmark pairs (8–12 recommended).')
    if not np.isfinite(points).all() or not np.isfinite(bearings).all():
        raise ValueError('Landmarks must be finite.')
    if np.any(np.linalg.norm(bearings, axis=1) < 1e-8):
        raise ValueError('Panorama directions must be nonzero.')
    bearings = bearings / np.linalg.norm(bearings, axis=1)[:, None]
    if np.linalg.matrix_rank(points-points.mean(0), tol=1e-6) < 2:
        raise ValueError('Mesh landmarks are collinear; spread them around the room.')
    start = np.r_[initial[:3], Rotation.from_euler('xyz', initial[3:], degrees=True).as_rotvec()]
    def residual(p):
        relative = (points-p[:3]) @ Rotation.from_rotvec(p[3:]).as_matrix()
        relative /= np.maximum(np.linalg.norm(relative, axis=1)[:, None], 1e-10)
        return (relative-bearings).ravel()
    result = least_squares(residual, start, loss='soft_l1', f_scale=np.deg2rad(2),
                           x_scale='jac', max_nfev=1000)
    if not result.success:
        raise ValueError(f'Camera fit failed: {result.message}')
    pose = np.r_[result.x[:3], Rotation.from_rotvec(result.x[3:]).as_euler('xyz', degrees=True)]
    difference = residual(result.x).reshape(-1, 3)
    errors = np.rad2deg(2*np.arcsin(np.clip(np.linalg.norm(difference, axis=1)/2, 0, 1)))
    return pose, errors


class Editor:
    def __init__(self, args):
        import trimesh
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, CheckButtons
        self.args, self.plt = args, plt
        print('Loading mesh...', flush=True)
        self.mesh = trimesh.load(args.mesh, process=False, force='mesh')
        if not len(self.mesh.faces):
            raise ValueError('Input must contain triangle faces')
        self.rgb = np.asarray(Image.open(args.png).convert('RGB'), dtype=np.float32)
        h, w = self.rgb.shape[:2]
        if w != 2*h:
            print('Warning: panorama is not 2:1. Use a full ERP with bottom padding for cropped ScanNet++ inputs.')
        self.initial = np.zeros(6)
        metrics = args.mesh.parent/'metrics.json'
        if metrics.exists():
            record = json.loads(metrics.read_text())
            if record.get('mesh_alignment_applied'):
                self.initial[:3] = record['alignment_shift']
        if args.pose:
            pose = json.loads(args.pose.read_text())
            self.initial[:3] = pose['camera_center']
            self.initial[3:] = pose.get('rotation_xyz_degrees', [0, 0, 0])
        self.values = self.initial.copy()
        import open3d as o3d
        self.o3d = o3d
        print('Building full triangle preview (no point sampling)...', flush=True)
        self.ray_scene = o3d.t.geometry.RaycastingScene()
        self.ray_scene.add_triangles(
            o3d.core.Tensor(np.asarray(self.mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(self.mesh.faces, dtype=np.uint32)))
        self.view_angles = np.zeros(2)  # yaw, pitch; viewing controls only
        self.fov = 80.
        self.drag = None
        self.landmarks = []
        self.pending_bearing = None
        self.landmark_artists = []
        self.fit_errors = None
        self.updating_pose = False
        self.fig = plt.figure(figsize=(15, 9))
        self.ax_image = self.fig.add_axes([.04, .38, .43, .52])
        self.ax_image.set_title('Panorama perspective at the same viewing direction')
        self.ax_image.axis('off')
        self.ax = self.fig.add_axes([.52, .38, .44, .52])
        self.ax.set_title('Solid mesh: drag to look around; scroll to change FOV')
        self.ax.axis('off')
        blank = np.zeros((args.preview_height, args.preview_width, 3), np.uint8)
        self.image_view = self.ax_image.imshow(blank)
        self.mesh_view = self.ax.imshow(blank)
        self.fig.canvas.mpl_connect('button_press_event', self.mouse_down)
        self.fig.canvas.mpl_connect('button_release_event', self.mouse_up)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.scroll)
        self.sliders = []
        labels = ['Camera X (m)', 'Camera Y (m)', 'Camera Z (m)', 'Pitch X (deg)', 'Yaw Y (deg)', 'Roll Z (deg)']
        for i, label in enumerate(labels):
            lo, hi = ((self.initial[i]-args.move_range, self.initial[i]+args.move_range)
                      if i < 3 else (-180, 180))
            slider = Slider(self.fig.add_axes([.17, .28-i*.035, .58, .022]), label,
                            lo, hi, valinit=self.initial[i], valfmt='%.3f')
            slider.on_changed(self.changed)
            self.sliders.append(slider)
        self.geometry_only = CheckButtons(self.fig.add_axes([.79, .23, .18, .055]), ['Geometry only'], [False])
        self.geometry_only.on_clicked(self.changed)
        self.buttons = []
        for x, label, callback in [( .04, 'Save pose', self.save_pose), (.26, 'Export colored mesh', self.export),
                                    (.52, 'Reset pose', self.reset)]:
            button = Button(self.fig.add_axes([x, .025, .21, .04]), label)
            button.on_clicked(callback); self.buttons.append(button)
        self.status = self.fig.text(.04, .955, '', fontsize=10)
        for x, label, callback in [(.04, 'Fit camera', self.fit_camera),
                                    (.27, 'Undo landmark', self.undo_landmark),
                                    (.50, 'Save landmarks', self.save_landmarks)]:
            button = Button(self.fig.add_axes([x, .315, .21, .04]), label)
            button.on_clicked(callback); self.buttons.append(button)
        self.landmark_status = self.fig.text(.04, .365, '', fontsize=10)
        landmark_path = getattr(args, 'landmarks', None)
        if landmark_path:
            data = json.loads(landmark_path.read_text())
            if Path(data['mesh']).resolve() != args.mesh.resolve() or Path(data['panorama']).resolve() != args.png.resolve():
                raise ValueError('Landmark file belongs to a different mesh or panorama.')
            self.landmarks = data['pairs']
        self.changed()
        plt.show()

    def mouse_down(self, event):
        if event.inaxes in (self.ax, self.ax_image) and event.button == 1:
            if event.key and 'shift' in event.key:
                self.pick_landmark(event)
            else:
                self.drag = (event.x, event.y)

    def mouse_up(self, event):
        self.drag = None

    def mouse_move(self, event):
        if self.drag is None or event.inaxes not in (self.ax, self.ax_image):
            return
        x, y = self.drag
        self.view_angles += [(x-event.x)*.25, (event.y-y)*.25]
        self.view_angles[1] = np.clip(self.view_angles[1], -89, 89)
        self.drag = (event.x, event.y)
        self.changed()

    def scroll(self, event):
        if event.inaxes in (self.ax, self.ax_image):
            self.fov = float(np.clip(self.fov-event.step*5, 25, 120))
            self.changed()

    def changed(self, _=None):
        if self.updating_pose:
            return
        self.values = np.array([s.val for s in self.sliders])
        width, height = self.args.preview_width, self.args.preview_height
        tangent = np.tan(np.deg2rad(self.fov)/2)
        x = ((np.arange(width)+.5)/width*2-1)*tangent
        y = (1-(np.arange(height)+.5)/height*2)*tangent*height/width
        xx, yy = np.meshgrid(x, y)
        directions = np.column_stack((xx.ravel(), yy.ravel(), -np.ones(width*height)))
        rotation = Rotation.from_euler('yx', self.view_angles, degrees=True).as_matrix()
        directions = (directions @ rotation.T).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        center = self.values[:3].astype(np.float32)
        rays = np.column_stack((np.broadcast_to(center, directions.shape), directions))
        hit = self.ray_scene.cast_rays(self.o3d.core.Tensor(rays))
        distance = hit['t_hit'].numpy()
        valid = np.isfinite(distance)
        # Image projection depends only on direction, so no-hit pixels also show the ERP.
        u, v = project(center+directions, center, self.values[3:])
        rgb = sample_rgb(self.rgb, u, v)
        self.image_view.set_data(rgb.reshape(height, width, 3))
        normals = hit['primitive_normals'].numpy()
        # Two-sided headlight shading makes walls and object boundaries legible.
        diffuse = np.abs(np.sum(normals*directions, axis=1))
        light = .3+.7*diffuse
        base = np.full_like(rgb, 210) if self.geometry_only.get_status()[0] else rgb
        rendered = np.clip(base.astype(float)*light[:, None], 0, 255).astype(np.uint8)
        rendered[~valid] = [35, 40, 48]
        self.mesh_view.set_data(rendered.reshape(height, width, 3))
        self.status.set_text(f'Perspective from camera position; FOV {self.fov:.0f}°. '
                             'Drag either pane to turn the view. Pose sliders align the panorama; Geometry only reveals shape.')
        self.draw_landmarks()
        self.fig.canvas.draw_idle()

    def pixel_direction(self, x, y):
        width, height = self.args.preview_width, self.args.preview_height
        tangent = np.tan(np.deg2rad(self.fov)/2)
        direction = np.array([((x+.5)/width*2-1)*tangent,
                              (1-(y+.5)/height*2)*tangent*height/width, -1.])
        direction = Rotation.from_euler('yx', self.view_angles, degrees=True).apply(direction)
        return direction/np.linalg.norm(direction)

    def pick_landmark(self, event):
        if event.xdata is None or event.ydata is None:
            return
        direction = self.pixel_direction(event.xdata, event.ydata)
        if event.inaxes == self.ax_image:
            # Freeze the selected image direction in panorama coordinates.
            rotation = Rotation.from_euler('xyz', self.values[3:], degrees=True)
            self.pending_bearing = rotation.inv().apply(direction).tolist()
        elif self.pending_bearing is None:
            self.status.set_text('Shift-click the panorama first, then the matching mesh corner.')
            self.fig.canvas.draw_idle()
            return
        else:
            ray = np.r_[self.values[:3], direction].astype(np.float32)[None]
            distance = self.ray_scene.cast_rays(self.o3d.core.Tensor(ray))['t_hit'].numpy()[0]
            if not np.isfinite(distance):
                self.status.set_text('No mesh at this pixel. Pick a visible mesh surface.')
                self.fig.canvas.draw_idle()
                return
            point = self.values[:3]+float(distance)*direction
            self.landmarks.append(dict(panorama_direction=self.pending_bearing, mesh_point=point.tolist()))
            self.pending_bearing = None
            self.fit_errors = None
            self.save_landmarks()
        self.draw_landmarks()
        self.fig.canvas.draw_idle()

    def draw_landmarks(self):
        for artist in self.landmark_artists:
            artist.remove()
        self.landmark_artists = []
        view = Rotation.from_euler('yx', self.view_angles, degrees=True).as_matrix()
        pano = Rotation.from_euler('xyz', self.values[3:], degrees=True).as_matrix()
        width, height = self.args.preview_width, self.args.preview_height
        focal = width/(2*np.tan(np.deg2rad(self.fov)/2))
        def marker(ax, direction, label, color):
            d = np.asarray(direction) @ view
            if d[2] >= -1e-8:
                return
            x, y = width/2-.5 + focal*d[0]/(-d[2]), height/2-.5-focal*d[1]/(-d[2])
            if -.5 <= x < width-.5 and -.5 <= y < height-.5:
                artist = ax.text(x, y, label, color='white', fontsize=10, ha='center', va='center',
                                 bbox=dict(boxstyle='circle,pad=.15', facecolor=color, alpha=.8), clip_on=True)
                self.landmark_artists.append(artist)
        for i, pair in enumerate(self.landmarks):
            marker(self.ax_image, np.asarray(pair['panorama_direction']) @ pano.T, str(i+1), 'darkorange')
            marker(self.ax, np.asarray(pair['mesh_point'])-self.values[:3], str(i+1), 'teal')
        if self.pending_bearing is not None:
            marker(self.ax_image, np.asarray(self.pending_bearing) @ pano.T, '+', 'red')
        self.landmark_status.set_text(f'{len(self.landmarks)} pairs. Shift-click panorama, then matching mesh point. '
                                      + ('Waiting for mesh point.' if self.pending_bearing is not None else 'Drag to look around.'))

    def undo_landmark(self, _=None):
        if self.pending_bearing is not None:
            self.pending_bearing = None
        elif self.landmarks:
            self.landmarks.pop()
        self.fit_errors = None
        self.save_landmarks()
        self.draw_landmarks()
        self.fig.canvas.draw_idle()

    def save_landmarks(self, _=None):
        self.args.output.mkdir(parents=True, exist_ok=True)
        data = dict(mesh=str(self.args.mesh.resolve()), panorama=str(self.args.png.resolve()),
                    pairs=self.landmarks, angular_errors_degrees=self.fit_errors)
        path = self.args.output/'landmarks.json'
        path.write_text(json.dumps(data, indent=2)+'\n')
        self.status.set_text(f'Saved {len(self.landmarks)} landmark pairs: {path}')
        self.fig.canvas.draw_idle()

    def fit_camera(self, _=None):
        try:
            pose, errors = fit_landmark_pose([p['mesh_point'] for p in self.landmarks],
                                            [p['panorama_direction'] for p in self.landmarks], self.values)
        except ValueError as exc:
            self.status.set_text(str(exc)); self.fig.canvas.draw_idle(); return
        self.updating_pose = True
        for i, (slider, value) in enumerate(zip(self.sliders, pose)):
            if value < slider.valmin or value > slider.valmax:
                margin = self.args.move_range if i < 3 else 30
                slider.valmin, slider.valmax = min(slider.valmin, value-margin), max(slider.valmax, value+margin)
                slider.ax.set_xlim(slider.valmin, slider.valmax)
            slider.set_val(value)
        self.updating_pose = False
        self.changed()
        self.fit_errors = errors.tolist()
        self.save_pose()
        self.save_landmarks()
        self.status.set_text(f'Fit saved: median error {np.median(errors):.2f}°, max {np.max(errors):.2f}°. '
                             'Large errors suggest mismatched landmarks or geometry differences.')
        self.fig.canvas.draw_idle()

    def reset(self, _):
        for slider, value in zip(self.sliders, self.initial):
            slider.set_val(value)

    def save_pose(self, _=None):
        self.args.output.mkdir(parents=True, exist_ok=True)
        pose = dict(mesh=str(self.args.mesh.resolve()), panorama=str(self.args.png.resolve()),
                    camera_center=self.values[:3].tolist(), rotation_xyz_degrees=self.values[3:].tolist(),
                    rotation_convention='scipy extrinsic xyz Euler degrees; camera to mesh',
                    camera_axes='Hunyuan: x right, y up, z backward', mesh_geometry_changed=False)
        (self.args.output/'panorama_pose.json').write_text(json.dumps(pose, indent=2)+'\n')
        self.status.set_text(f'Saved {self.args.output / "panorama_pose.json"}')
        self.fig.canvas.draw_idle()

    def export(self, _=None):
        import open3d as o3d
        import trimesh
        self.save_pose()
        self.status.set_text('Exporting full mesh with visibility cleanup; see terminal for progress...')
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        faces = np.asarray(self.mesh.faces, dtype=np.uint32)
        center = self.values[:3].astype(np.float32)
        relative = vertices-center
        distance = np.linalg.norm(relative, axis=1)
        directions = relative/np.maximum(distance[:, None], 1e-10)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
        visible = np.zeros(len(vertices), bool)
        gap = np.full(len(vertices), np.inf, np.float32)
        for start in range(0, len(vertices), 150000):
            stop = min(start+150000, len(vertices))
            rays = np.column_stack((np.broadcast_to(center, (stop-start, 3)), directions[start:stop]))
            hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            gap[start:stop] = np.abs(hit-distance[start:stop])
            visible[start:stop] = np.isfinite(hit) & (gap[start:stop] <= .002) & (distance[start:stop] > 1e-6)
        raw = visible.copy()
        if visible.any():
            visible = extend_color_to_local_occlusions(vertices, visible, gap)
            visible = color_small_hidden_islands(vertices, faces, visible)
        colors = np.full((len(vertices), 4), [*HIDDEN_RGB, 255], np.uint8)
        u, v = project(vertices[visible], self.values[:3], self.values[3:])
        colors[visible, :3] = sample_rgb(self.rgb, u, v)
        mesh = self.mesh.copy()
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        for suffix in ('ply', 'glb'):
            path = self.args.output/f'mesh_panorama_aligned.{suffix}'
            mesh.export(path); print(f'Saved {path}', flush=True)
        np.savez_compressed(self.args.output/'visibility.npz', visible=visible, raw_visible=raw, depth_gap=gap)
        self.status.set_text(f'Exported PLY + GLB; {visible.mean():.1%} colored. Pose saved.')
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--png', type=Path, required=True, help='Full 2:1 ERP panorama')
    parser.add_argument('--output', type=Path, help='Default: <mesh folder>/manual_panorama_alignment')
    parser.add_argument('--pose', type=Path, help='Reload panorama_pose.json')
    parser.add_argument('--landmarks', type=Path, help='Reload landmarks.json for this mesh and panorama')
    parser.add_argument('--preview-width', type=int, default=480)
    parser.add_argument('--preview-height', type=int, default=360)
    parser.add_argument('--preview-points', type=int, default=25000, help=argparse.SUPPRESS)
    parser.add_argument('--move-range', type=float, default=2., help='Translation slider range around initial camera, metres')
    args = parser.parse_args()
    if min(args.preview_width, args.preview_height) < 1 or args.move_range <= 0:
        parser.error('preview dimensions and move-range must be positive')
    args.output = args.output or args.mesh.parent/'manual_panorama_alignment'
    Editor(args)


if __name__ == '__main__':
    main()
