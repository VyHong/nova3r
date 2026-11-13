"""Native UDF extraction must use the same distance cutoff after tiling."""
import contextlib
import io
import unittest
import numpy as np
from scipy.spatial import cKDTree
from training.mesh2sdf.tools.mesh_udf.custom_mc._marching_cubes_lewiner import udf_mc_lewiner


class TileScaleTests(unittest.TestCase):
    def test_enclosed_sphere_matches_full_grid(self):
        n = 49
        xyz = np.stack(np.meshgrid(*([np.linspace(-1, 1, n)] * 3), indexing='ij'), -1)
        radius = np.linalg.norm(xyz, axis=-1)
        signed = radius - .23
        field = np.abs(signed).astype(np.float32)
        normals = (-np.sign(signed)[..., None] * xyz / np.maximum(radius[..., None], 1e-9)).astype(np.float32)
        spacing = (2 / (n - 1),) * 3
        start, stop = 12, 36
        tile_field = field[start:stop+1, start:stop+1, start:stop+1]
        tile_normals = normals[start:stop+1, start:stop+1, start:stop+1]
        with contextlib.redirect_stdout(io.StringIO()):
            full, _, _, _ = udf_mc_lewiner(field, normals, spacing=spacing)
            tile, _, _, _ = udf_mc_lewiner(tile_field * ((n - 1) / (stop - start)), tile_normals, spacing=spacing)
        tile += start * spacing[0]
        self.assertGreater(len(full), 0)
        self.assertGreater(len(tile), 0)
        self.assertLess(cKDTree(full).query(tile)[0].max(), 1e-5)
        self.assertLess(cKDTree(tile).query(full)[0].max(), 1e-5)


if __name__ == '__main__':
    unittest.main()
