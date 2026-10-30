import unittest
import numpy as np
import torch
from torch import nn

from training.mesh2sdf.oriented_meshing import surface_oriented_bounds, decoder_in_box_frame


class Field(nn.Module):
    def forward(self, queries, latents=None):
        return queries.square().sum(-1)


class OrientedMeshingTests(unittest.TestCase):
    def test_rotated_box_padding_and_query_frame(self):
        angle = 0.63
        rotation = np.array([[np.cos(angle), 0, -np.sin(angle)],
                             [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
        center = np.array([0.4, -0.2, 0.7])
        corners = np.array([[x,y,z] for x in (-1,1)
                            for y in (-0.4,0.4) for z in (-0.2,0.2)])
        scene = corners @ rotation.T + center
        frame = surface_oriented_bounds(torch.tensor(scene), 0.2)
        np.testing.assert_allclose(np.sort(frame.bounds[3:]-frame.bounds[:3]),
                                   np.sort(np.array([2,0.8,0.4])*1.4), atol=1e-6)
        local = (scene-frame.to_scene[:3,3]) @ frame.to_scene[:3,:3]
        self.assertTrue((local >= frame.bounds[:3]-1e-6).all())
        self.assertTrue((local <= frame.bounds[3:]+1e-6).all())
        decoder = nn.Module()
        decoder.geo_decoder = Field()
        original = decoder.geo_decoder
        wrapped = decoder_in_box_frame(decoder, frame)
        self.assertIs(decoder.geo_decoder, original)
        queries = torch.tensor(local, requires_grad=True)
        values = wrapped.geo_decoder(queries=queries)
        torch.testing.assert_close(values, torch.tensor((scene**2).sum(-1)))
        values.sum().backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        import trimesh
        mesh = trimesh.Trimesh(vertices=local, faces=[], process=False)
        frame.transform_mesh(mesh)
        np.testing.assert_allclose(mesh.vertices, scene, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
