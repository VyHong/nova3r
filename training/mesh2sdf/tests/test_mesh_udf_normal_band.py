import unittest
import torch
from training.mesh2sdf.tools.mesh_udf.mesh_udf import (
    get_udf_normals_grid_slow, get_udf_normals_grid_streamed,
)

class OffsetPlane(torch.nn.Module):
    def forward(self, xyz):
        return xyz[:, :1] + 1.1

class NormalBandTests(unittest.TestCase):
    def test_wider_band_adds_normals_without_changing_field(self):
        decoder = OffsetPlane()
        field, default, _ = get_udf_normals_grid_slow(decoder, None, N=9, max_batch=64, device='cpu')
        for band in (2., 3., 4.):
            dense_field, dense_normals, _ = get_udf_normals_grid_slow(
                decoder, None, N=9, max_batch=64, device='cpu', normal_band_voxels=band)
            streamed_field, streamed_normals = get_udf_normals_grid_streamed(
                decoder, None, N=9, max_batch=64, device='cpu', normal_band_voxels=band)
            torch.testing.assert_close(dense_field, field)
            torch.testing.assert_close(streamed_field, field)
            torch.testing.assert_close(streamed_normals, dense_normals)
            selected = field < band * .25
            torch.testing.assert_close(dense_normals.norm(dim=-1), selected.float())
            torch.testing.assert_close(dense_normals[selected][:, 0], -torch.ones(selected.sum()))
            if band == 2:
                torch.testing.assert_close(default, dense_normals)
            else:
                self.assertGreater(int(selected.sum()), int((default.norm(dim=-1) > 0).sum()))

    def test_scattered_gradient_points_match_streamed_grid(self):
        class Sphere(torch.nn.Module):
            def forward(self, xyz):
                return (xyz.square().sum(dim=-1, keepdim=True).sqrt() - .55).abs() + .03
        bounds = [-1., -.5, -.75, 1., .5, .75]
        args = dict(N=(13, 9, 11), max_batch=32, grid_bounds=bounds, device='cpu', normal_band_voxels=3.)
        dense_field, dense_normals, _ = get_udf_normals_grid_slow(Sphere(), None, **args)
        streamed_field, streamed_normals = get_udf_normals_grid_streamed(Sphere(), None, **args)
        torch.testing.assert_close(dense_field, streamed_field)
        torch.testing.assert_close(dense_normals, streamed_normals)

    def test_invalid_bands_fail(self):
        for band in (0., -1., float('nan'), float('inf')):
            with self.subTest(band=band), self.assertRaisesRegex(ValueError, 'positive and finite'):
                get_udf_normals_grid_streamed(OffsetPlane(), None, N=3, max_batch=64,
                                             device='cpu', normal_band_voxels=band)

if __name__ == '__main__':
    unittest.main()
