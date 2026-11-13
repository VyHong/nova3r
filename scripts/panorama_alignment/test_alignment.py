"""Run with MPLBACKEND=Agg python -m unittest discover -s scripts/panorama_alignment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('alignment', Path(__file__).with_name('align_panorama.py'))
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class AlignmentTests(unittest.TestCase):
    def test_recovers_spherical_camera_pose(self):
        rng = np.random.default_rng(23)
        bearings = rng.normal(size=(24, 3))
        bearings /= np.linalg.norm(bearings, axis=1)[:, None]
        center = np.array([.3, -.15, .2])
        rotation = Rotation.from_euler('xyz', [7, -13, 4], degrees=True)
        points = center+rotation.apply(bearings)*rng.uniform(2, 5, (24, 1))
        pose, errors = alignment.fit_landmark_pose(points, bearings, np.zeros(6))
        np.testing.assert_allclose(pose[:3], center, atol=1e-5)
        np.testing.assert_allclose(Rotation.from_euler('xyz', pose[3:], degrees=True).as_matrix(),
                                   rotation.as_matrix(), atol=1e-5)
        self.assertLess(errors.max(), .001)
        corrupted = bearings.copy(); corrupted[0] = -corrupted[0]
        _, errors = alignment.fit_landmark_pose(points, corrupted, np.zeros(6))
        self.assertLess(np.median(errors[1:]), .5)
        self.assertGreater(errors[0], 100)

    def test_rejects_insufficient_and_collinear_pairs(self):
        with self.assertRaises(ValueError):
            alignment.fit_landmark_pose(np.ones((5, 3)), np.ones((5, 3)), np.zeros(6))
        with self.assertRaises(ValueError):
            alignment.fit_landmark_pose(np.arange(18).reshape(6, 3), np.ones((6, 3)), np.zeros(6))

    def test_picking_undo_save_reload_and_fit_controls(self):
        import trimesh
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trimesh.creation.box(extents=[6, 4, 8]).export(root/'mesh.ply')
            Image.new('RGB', (128, 64), (160, 100, 40)).save(root/'image.png')
            args = SimpleNamespace(mesh=root/'mesh.ply', png=root/'image.png', output=root/'output',
                                   pose=None, landmarks=None, move_range=2., preview_width=100, preview_height=80)
            editor = alignment.Editor(args)
            click = lambda ax: SimpleNamespace(inaxes=ax, xdata=49.5, ydata=39.5)
            editor.pick_landmark(click(editor.ax_image))
            np.testing.assert_allclose(editor.pending_bearing, [0, 0, -1])
            # Change viewing direction between the image and mesh picks.
            editor.view_angles[0] = 10
            editor.changed()
            editor.pick_landmark(click(editor.ax))
            self.assertEqual(len(editor.landmarks), 1)
            self.assertIsNone(editor.pending_bearing)
            self.assertEqual(len(editor.landmark_artists), 2)
            np.testing.assert_allclose(editor.landmarks[0]['panorama_direction'], [0, 0, -1])
            editor.undo_landmark()
            self.assertEqual(len(editor.landmarks), 0)
            rng = np.random.default_rng(3)
            bearings = rng.normal(size=(12, 3)); bearings /= np.linalg.norm(bearings, axis=1)[:, None]
            points = np.array([.1, .2, -.1])+Rotation.from_euler('xyz', [2, 5, -3], degrees=True).apply(bearings)*3
            editor.landmarks = [dict(mesh_point=p.tolist(), panorama_direction=b.tolist()) for p, b in zip(points, bearings)]
            editor.fit_camera()
            np.testing.assert_allclose(editor.values, [.1, .2, -.1, 2, 5, -3], atol=1e-4)
            self.assertTrue((args.output/'panorama_pose.json').exists())
            args.landmarks = args.output/'landmarks.json'
            restored = alignment.Editor(args)
            self.assertEqual(restored.landmarks, editor.landmarks)
            editor.plt.close('all')


if __name__ == '__main__':
    unittest.main()
