"""Check saved test geometry against the alignment used by evaluation."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import traceback
import unittest

import numpy as np
import torch
import trimesh


class AlignedTestMeshTests(unittest.TestCase):
    def test_export_uses_metric_alignment_after_world_scaling(self):
        tree = ast.parse(Path('training/mesh2sdf/lightning_module.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'Mesh2SDFLightningModule')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'test_step')
        for world_scale in (1.0, 3.0):
            with self.subTest(world_scale=world_scale), tempfile.TemporaryDirectory() as root:
                mesh = trimesh.creation.box()
                mesh.visual.vertex_colors = [20, 80, 140, 255]
                original_vertices = mesh.vertices.copy()
                shift = np.array([0.25, -0.5, 1.0])
                scale = 1.75
                sampled = []

                def sample_surface(vertices, faces, count, seed):
                    # Evaluation must sample the unaligned world-scale mesh.
                    np.testing.assert_array_equal(vertices, original_vertices * world_scale)
                    sampled.append(True)
                    return np.asarray(vertices).copy()

                def evaluate(pred, target, **kwargs):
                    return pred * scale + shift, dict(
                        alignment_scale=scale, alignment_shift=shift.tolist())

                namespace = dict(torch=torch, np=np, Path=Path, json=json,
                                 traceback=traceback, sample_mesh_surface=sample_surface,
                                 sample_points=lambda points, count, seed: points,
                                 evaluate_point_clouds=evaluate)
                exec(compile(ast.Module(body=[method], type_ignores=[]),
                             '<test-step>', 'exec'), namespace)
                instance = SimpleNamespace(
                    normalize_surface_and_geo_points=lambda batch, **kw: batch,
                    validation_options=SimpleNamespace(normalization='none'),
                    model=SimpleNamespace(lightning_forward=lambda *a, **kw:
                                          (None, None, torch.zeros(1, 1, 1))),
                    device=torch.device('cpu'), cfg={}, val_criterion=None,
                    _latent_decoder=lambda: SimpleNamespace(rgb_decoder=None),
                    _decode_latents_for_geometry=lambda latents, decoder: latents,
                    current_log_dir=root, trainer=SimpleNamespace(world_size=1),
                    global_rank=0, _test_sample_name=lambda *a: 'example',
                    _reconstruction_bounds=lambda *a: None,
                    _surface_world_scale=lambda *a: world_scale,
                    mesh_from_latents=lambda *a, **kw: mesh,
                    mesh_options=SimpleNamespace(method='mesh_udf'),
                    _test_sample_seed=lambda *a: 43,
                    evaluation_options=SimpleNamespace(points=8, fscore_thresholds=[.1],
                                                       alignment_iterations=1, alignment_lr=.01),
                    test_results=[], print=print)
                records = namespace['test_step'](
                    instance, {'surface': torch.zeros(1, 8, 3)}, 0)
                self.assertEqual(records[0]['status'], 'ok')
                self.assertTrue(sampled)
                saved = trimesh.load(records[0]['mesh'], process=False)
                np.testing.assert_allclose(saved.vertices,
                                           original_vertices * world_scale * scale + shift)
                np.testing.assert_array_equal(saved.faces, mesh.faces)
                np.testing.assert_array_equal(saved.visual.vertex_colors, mesh.visual.vertex_colors)
                np.testing.assert_array_equal(mesh.vertices, original_vertices)
                metadata = json.loads((Path(records[0]['mesh']).parent/'metrics.json').read_text())
                self.assertTrue(metadata['mesh_alignment_applied'])
                self.assertEqual(metadata['alignment_scale'], scale)
                self.assertEqual(metadata['alignment_shift'], shift.tolist())


if __name__ == '__main__':
    unittest.main()
