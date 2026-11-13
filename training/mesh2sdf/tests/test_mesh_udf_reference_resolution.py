"""A finer extraction grid must be able to retain a learned UDF residual."""
import contextlib
import io
import unittest

import numpy as np
import torch

from training.mesh2sdf.tools.mesh_udf.mesh_udf import (
    _reference_distance_scale, get_mesh_udf_fast,
)
from training.mesh2sdf.module_options import MeshOptions


class ResidualPlane(torch.nn.Module):
    def forward(self, xyz):
        return (xyz[:, :1] - .13).abs() + .025


class ReferenceResolutionTests(unittest.TestCase):
    def test_reference_retains_surface_lost_to_finer_grid_cutoff(self):
        options = dict(decoder=ResidualPlane(), latent_vec=None, gradient=False,
                       device='cpu', streamed_grid=True, query_batch=32768,
                       surface_threshold=.1, fast_cleanup=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, coarse = get_mesh_udf_fast(N_MC=33, **options)
            with self.assertRaisesRegex(RuntimeError, 'No surface'):
                get_mesh_udf_fast(N_MC=65, **options)
            _, _, fine = get_mesh_udf_fast(N_MC=65, reference_resolution=33, **options)
        self.assertGreater(len(coarse.faces), 0)
        self.assertGreater(len(fine.faces), 0)
        self.assertLess(float(np.abs(fine.vertices[:, 0] - .13).max()), .02)

    def test_reference_never_tightens_coarser_grid_cutoffs(self):
        self.assertEqual(_reference_distance_scale((33, 17, 25), None), 1.)
        self.assertEqual(_reference_distance_scale((33, 17, 25), 65), 1.)
        self.assertEqual(_reference_distance_scale((65, 33, 49), 33), .5)
        for value in (0, 1, 2.5, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _reference_distance_scale((65, 33, 49), value)

    def test_tuned_dense_defaults_and_explicit_legacy_overrides(self):
        defaults = MeshOptions.from_config({'mesh_method': 'mesh_udf'})
        self.assertEqual(defaults.udf_resolution, 768)
        self.assertEqual(defaults.udf_reference_resolution, 512)
        self.assertEqual(defaults.surface_threshold, .01)
        self.assertEqual(defaults.udf_query_batch_size, 262144)
        self.assertEqual(defaults.vertex_batch_size, 131072)
        self.assertTrue(defaults.udf_streamed_grid)
        self.assertTrue(defaults.udf_fast_cleanup)
        self.assertEqual(defaults.bounds_padding, .05)
        self.assertEqual(defaults.udf_normal_band_voxels, 2.)
        self.assertFalse(defaults.smooth_borders)
        legacy = MeshOptions.from_config(dict(mesh_method='mesh_udf', mesh_mc=512,
            mesh_udf_reference_resolution=None, mesh_udf_surface_threshold=.008,
            mesh_udf_vertex_batch_size=65536, mesh_udf_streamed_grid=False,
            mesh_udf_fast_cleanup=False))
        self.assertIsNone(legacy.udf_reference_resolution)
        self.assertEqual(legacy.udf_resolution, 512)
        self.assertEqual(legacy.surface_threshold, .008)
        self.assertEqual(legacy.vertex_batch_size, 65536)
        self.assertFalse(legacy.udf_streamed_grid)
        self.assertFalse(legacy.udf_fast_cleanup)
        sparse = MeshOptions.from_config(dict(mesh_method='mesh_udf', mesh_udf_sparse_bricks=True))
        self.assertIsNone(sparse.udf_reference_resolution)

    def test_config_rejects_unsupported_sparse_reference(self):
        self.assertIsNone(MeshOptions.from_config({}).udf_reference_resolution)
        self.assertEqual(MeshOptions.from_config({'mesh_udf_reference_resolution': 512}).udf_reference_resolution, 512)
        with self.assertRaisesRegex(ValueError, 'dense'):
            MeshOptions.from_config({'mesh_udf_reference_resolution': 512, 'mesh_udf_sparse_bricks': True})


if __name__ == '__main__': unittest.main()
