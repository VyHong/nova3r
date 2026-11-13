"""Regression coverage for empty structure chunks without loading model weights."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'third_party/GenRecon'))
pipeline = importlib.import_module('genrecon.pipelines.full_scene_images_to_3d')


class EmptyChunkTests(unittest.TestCase):
    def run_pipeline(self, counts):
        n = len(counts)
        coords = [torch.tensor([[0, 0, 0, 0]] * count, dtype=torch.int32).reshape(-1, 4) for count in counts]
        translations = [torch.tensor([i + 2., 0., 0.]) for i in range(n)]
        cond_features = torch.arange(n).reshape(1, n, 1, 1).float()
        model = SimpleNamespace(dtype=torch.float32, in_channels=1, resolution=32)
        captured = {}
        def build_cond(*args, **kwargs):
            selected_coords, offsets = args[5:7]
            self.assertTrue(all(len(c) for c in selected_coords))
            captured['features'] = args[2]
            captured['offsets'] = offsets
            return [dict(cond=None, neg_cond=None) for _ in selected_coords]
        def decode(slats, res, offsets, **kwargs):
            self.assertTrue(all(len(s.coords) for s in slats))
            captured['decode_offsets'] = offsets
            return SimpleNamespace(fill_holes=lambda: None), None, None, None, None
        fake = SimpleNamespace(
            default_pipeline_type='512', device=torch.device('cpu'), low_vram=False,
            models={'sparse_structure_flow_model': model, 'shape_slat_flow_model_512': model},
            sparse_structure_sampler=None, shape_slat_sampler=None,
            sparse_structure_sampler_params={}, shape_slat_sampler_params={},
            shape_slat_normalization={'std': [1.], 'mean': [0.]},
            _require_run_components=lambda kind: None,
            _encode_scene=lambda images, res: cond_features,
            _prep_extrinsics_intrinsics=lambda *args: (None, None),
            _build_global_dense_cond=lambda *args, **kwargs: [dict(cond=None, neg_cond=None)] * n,
            joint_decode_sparse_structure=lambda *args: None,
            extract_coords_from_occ_logits=lambda *args, **kwargs: coords,
            _build_global_sparse_cond=build_cond, joint_decode_shape_slats=decode,
        )
        sel = SimpleNamespace(chunk_indices=list(range(n)), scene_images_512=torch.zeros(n, 1),
                              cond2d_images_512=[torch.zeros(1)] * n,
                              scene_extrinsics_c0=None, scene_intrinsics=None)
        class Sampler:
            def __init__(self, sampler): pass
            def sample(self, model, noise, *args, **kwargs): return noise
        with patch.object(pipeline, 'MultiDiffusionOrchestrator', Sampler), patch.object(torch.cuda, 'is_available', return_value=False):
            pipeline.FullSceneImagesTo3DPipeline.run(fake, sel, translations, geometry_only=True)
        return captured

    def test_empty_first_chunk_keeps_original_world_offsets_and_conditioning(self):
        captured = self.run_pipeline([0, 1, 2])
        self.assertEqual(captured['features'].flatten().tolist(), [1., 2.])
        self.assertEqual([t[0].item() for t in captured['offsets']], [3., 4.])
        self.assertEqual([t[0].item() for t in captured['decode_offsets']], [3., 4.])

    def test_empty_middle_chunk_preserves_matching_images(self):
        captured = self.run_pipeline([1, 0, 1])
        self.assertEqual(captured['features'].flatten().tolist(), [0., 2.])
        self.assertEqual([t[0].item() for t in captured['decode_offsets']], [2., 4.])

    def test_all_empty_reports_structure_failure(self):
        with self.assertRaisesRegex(ValueError, 'no occupied voxels in any chunk'):
            self.run_pipeline([0, 0])

    def test_nonempty_chunks_are_unchanged(self):
        captured = self.run_pipeline([1, 2])
        self.assertEqual(captured['features'].flatten().tolist(), [0., 1.])
        self.assertEqual([t[0].item() for t in captured['decode_offsets']], [2., 3.])


if __name__ == '__main__':
    unittest.main()
