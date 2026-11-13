"""Exercise the Lightning target builder without loading the model/CUDA stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import trimesh
from eval.baselines.common import (SageSample, load_visibility_geometry,
                                   sample_mesh_surface, panorama_visibility_mask)


class VisibilityTargetTests(unittest.TestCase):
    def test_mesh_in_decoder_frame_and_normalization_offset(self):
        source = Path('training/mesh2sdf/lightning_module.py').read_text()
        tree = ast.parse(source)
        module = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == 'Mesh2SDFLightningModule')
        method = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_test_visibility_target')
        namespace = dict(Path=Path, np=np, SageSample=SageSample,
                         load_visibility_geometry=load_visibility_geometry,
                         sample_mesh_surface=sample_mesh_surface,
                         panorama_visibility_mask=panorama_visibility_mask)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<target-builder>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / 'scene'
            scene.mkdir()
            mesh = trimesh.creation.box(extents=[2, 2, 2])
            mesh.apply_translation([5, 0, 0])
            mesh.export(scene / 'scene.ply')
            transform = torch.eye(4)
            transform[0, 3] = -5
            batch = dict(evaluation_layout=['sage10k'], evaluation_scene=['scene'],
                         evaluation_frame=torch.tensor([0]),
                         evaluation_surface_path=[str(scene / 'scene_surface.npz')],
                         evaluation_pose_path=['unused.json'],
                         evaluation_world_to_geometry=transform[None],
                         surface_geo_norm_center=torch.tensor([[[0.25, 0., 0.]]]))
            model = SimpleNamespace(cfg={'test_visibility_mesh_roots': {'sage10k': directory}},
                                    _batch_item=lambda value, index: value[index])
            points, visible, path = namespace['_test_visibility_target'](model, batch, 0, 0, 200)
            self.assertEqual(path, scene / 'scene.ply')
            self.assertEqual(points.shape, (200, 3))
            self.assertTrue(visible.all())
            self.assertGreaterEqual(points[:, 0].min(), -1.25001)
            self.assertLessEqual(points[:, 0].max(), 0.75001)


if __name__ == '__main__':
    unittest.main()
