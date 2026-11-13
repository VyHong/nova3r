"""Analytic coverage examples for regional evaluation."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from eval.baselines.common import compute_visibility_metrics, build_evaluation_summary, write_evaluation_results


class VisibilityMetricsTests(unittest.TestCase):
    def test_geometry_only_glb_preserves_textured_scene_arrays(self):
        import trimesh
        from PIL import Image
        from eval.baselines.common import SageSample, load_visibility_geometry
        scene = trimesh.Scene()
        for index, color in enumerate(('red', 'blue')):
            mesh = trimesh.creation.box()
            mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.zeros((len(mesh.vertices), 2)), image=Image.new('RGB', (8, 8), color))
            transform = np.eye(4)
            transform[0, 3] = index * 3
            scene.add_geometry(mesh, node_name=f'box{index}', transform=transform)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.glb'
            scene.export(path)
            original = trimesh.load(path, process=False).dump(concatenate=True)
            sample = SageSample('scene 00000', 'scene', 0, path.parent,
                                path.parent, path, 'sage10k')
            vertices, faces, evaluated, _ = load_visibility_geometry(sample)
            np.testing.assert_array_equal(vertices, np.asarray(original.vertices, dtype=np.float32))
            np.testing.assert_array_equal(faces, original.faces)
            np.testing.assert_array_equal(evaluated, original.faces)

    def test_mesh_blocks_rear_points(self):
        from eval.baselines.common import panorama_visibility_mask
        vertices = np.array([[-2, -2, 1], [2, -2, 1], [0, 2, 1]], dtype=np.float32)
        points = np.array([[0, 0, 1], [0, 0, 2], [3, 0, 1]], dtype=np.float32)
        visible = panorama_visibility_mask(points, np.zeros(3), vertices, [[0, 1, 2]])
        np.testing.assert_array_equal(visible, [True, False, False])

    def test_anonymized_faces_excluded_but_retained_as_occluders(self):
        import trimesh
        from eval.baselines.common import SageSample, load_visibility_geometry
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh_aligned_0.05.ply'
            mesh = trimesh.Trimesh(vertices=[[0,0,1], [1,0,1], [0,1,1],
                                             [0,0,2], [1,0,2], [0,1,2]],
                                   faces=[[0,1,2], [3,4,5]], process=False)
            mesh.export(path)
            path.with_name('mesh_aligned_0.05_mask.txt').write_text('0\n')
            sample = SageSample('scene 00000', 'scene', 0, Path(directory),
                                Path(directory), path, 'scannetpp')
            _, occluders, evaluated, _ = load_visibility_geometry(sample)
            self.assertEqual(len(occluders), 2)
            np.testing.assert_array_equal(evaluated, [[3,4,5]])

    def test_room_precedence_and_reindexed_anonymization(self):
        import trimesh
        from eval.baselines.common import SageSample, load_visibility_geometry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = root / 'scene' / 'scans'
            scan.mkdir(parents=True)
            vertices = np.array([[0,0,1], [1,0,1], [0,1,1],
                                 [0,0,2], [1,0,2], [0,1,2]], dtype=float)
            trimesh.Trimesh(vertices=vertices, faces=[[0,1,2],[3,4,5]], process=False).export(scan / 'mesh_aligned_0.05.ply')
            (scan / 'mesh_aligned_0.05_mask.txt').write_text('0\n')
            room = root / 'rooms' / 'scene' / 'room_01'
            room.mkdir(parents=True)
            # Reindex: masked original vertex 0 is room vertex 3.
            trimesh.Trimesh(vertices=vertices[[3,4,5,0,1,2]], faces=[[0,1,2],[3,4,5]], process=False).export(room / 'mesh.ply')
            sample = SageSample('scene 00000', 'scene', 0, root, root, room / 'mesh.ply', 'scannetpp')
            _, faces, evaluated, path = load_visibility_geometry(sample, mesh_root=root)
            self.assertEqual(path, room / 'mesh.ply')
            self.assertEqual(len(faces), 2)
            np.testing.assert_array_equal(evaluated, [[0,1,2]])

    def test_room_local_mask_works_without_original_scan(self):
        import trimesh
        from eval.baselines.common import SageSample, load_visibility_geometry
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.ply'
            trimesh.Trimesh(vertices=[[0,0,1], [1,0,1], [0,1,1],
                                     [0,0,2], [1,0,2], [0,1,2]],
                            faces=[[0,1,2],[3,4,5]], process=False).export(path)
            path.with_name('mesh_mask.txt').write_text('3\n')
            root = Path(directory)
            sample = SageSample('scene 00000', 'scene', 0, root, root, path, 'scannetpp')
            _, occluders, evaluated, _ = load_visibility_geometry(sample)
            self.assertEqual(len(occluders), 2)
            np.testing.assert_array_equal(evaluated, [[0,1,2]])

    def test_missing_occluded_surface_is_penalized(self):
        prediction = np.array([[0., 0., 0.]])
        target = np.array([[0., 0., 0.], [1., 0., 0.]])
        metrics = compute_visibility_metrics(prediction, target, [True, False], [0.1, 0.05, 0.02])
        self.assertEqual(metrics['visible_cd'], 0)
        self.assertEqual(metrics['visible_apd_0.1'], 1)
        self.assertEqual(metrics['occluded_cd'], 1)
        self.assertEqual(metrics['occluded_apd_0.1'], 0)
        self.assertEqual(metrics['complete_cd'], 0.25)
        self.assertEqual(metrics['visible_chamfer_forward'], 0)
        self.assertEqual(metrics['visible_chamfer_backward'], 0)
        self.assertEqual(metrics['occluded_chamfer_forward'], 1)
        self.assertEqual(metrics['occluded_chamfer_backward'], 1)
        self.assertEqual(metrics['occluded_chamfer_mean'], 1)

    def test_regional_forward_uses_entire_prediction(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        metrics = compute_visibility_metrics(points, points, [True, False], [0.1, 0.05, 0.02])
        for region in ('visible', 'occluded'):
            self.assertEqual(metrics[f'{region}_chamfer_forward'], 0.5)
            self.assertEqual(metrics[f'{region}_chamfer_backward'], 0.)
            self.assertEqual(metrics[f'{region}_chamfer_mean'], 0.25)

    def test_empty_region_is_null(self):
        points = np.array([[0., 0., 0.]])
        metrics = compute_visibility_metrics(points, points, [True], [0.1])
        self.assertIsNone(metrics['occluded_chamfer_forward'])
        self.assertIsNone(metrics['occluded_chamfer_backward'])
        self.assertIsNone(metrics['occluded_cd'])
        self.assertIsNone(metrics['occluded_apd_0.1'])

    def test_table_units_and_summary(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.]])
        metrics = compute_visibility_metrics(points, points, [True, False], [0.1, 0.05, 0.02])
        record = dict(status='ok', chamfer_forward=0., chamfer_backward=0.,
                      chamfer_mean=0., f_score_0=0.)
        record.update(metrics)
        record.update({'f_score_0.1_full': 1., 'f_score_0.05_full': 1., 'f_score_0.02_full': 1.})
        summary = build_evaluation_summary([record], model='pager', thresholds=[0.1, 0.05, 0.02],
                                           alignment_iterations=300, alignment_lr=0.1)
        with tempfile.TemporaryDirectory() as directory:
            write_evaluation_results([record], summary, Path(directory))
            for column in ('Visible APD0.02↑', 'Occluded APD0.02↑', 'Complete FS0.02↑'):
                self.assertIn(column, (Path(directory) / 'table.md').read_text())
            self.assertEqual(summary['visible_chamfer_forward'], 0.5)
            self.assertEqual(summary['occluded_chamfer_backward'], 0.)
            self.assertEqual(summary['visible_apd_0.02'], 1.)
            self.assertEqual(summary['occluded_apd_0.02'], 1.)
            self.assertIn('| pager | 0.00 | 100.00 |', (Path(directory) / 'table.md').read_text())


if __name__ == '__main__':
    unittest.main()
