import unittest
import numpy as np
import torch
from torch import nn

from training.mesh2sdf.oriented_meshing import surface_oriented_bounds, decoder_in_box_frame


class Field(nn.Module):
    def forward(self, queries, latents=None):
        return queries.square().sum(-1)


class OrientedMeshingTests(unittest.TestCase):
    def test_axis_aligned_grid_and_whole_triangle_snug_clip(self):
        import trimesh
        from training.mesh2sdf.oriented_meshing import (
            OrientedMeshBounds, enclosing_axis_aligned_bounds,
            remove_triangles_outside_oriented_bounds,
        )
        angle = 0.63
        transform = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        transform[:3, 3] = [0.4, -0.2, 0.7]
        frame = OrientedMeshBounds(np.array([-1, -.4, -.2, 1, .4, .2]), transform)
        corners = np.array([[x, y, z] for x in (-1, 1)
                            for y in (-.4, .4) for z in (-.2, .2)])
        scene_corners = trimesh.transform_points(corners, transform)
        bounds = enclosing_axis_aligned_bounds(frame)
        np.testing.assert_allclose(bounds[:3], scene_corners.min(axis=0))
        np.testing.assert_allclose(bounds[3:], scene_corners.max(axis=0))
        local = np.array([[0, 0, 0], [.5, 0, 0], [0, .2, 0], [1.1, 0, 0]])
        scene = trimesh.transform_points(local, transform)
        mesh = trimesh.Trimesh(scene, [[0, 1, 2], [1, 3, 2]], process=False)
        clipped = remove_triangles_outside_oriented_bounds(mesh, frame)
        self.assertEqual(len(clipped.faces), 1)
        self.assertEqual(len(clipped.vertices), 3)
        np.testing.assert_allclose(clipped.vertices, scene[:3])
        self.assertEqual(len(mesh.faces), 2)
        np.testing.assert_allclose(mesh.vertices, scene)
        outside = trimesh.Trimesh(scene, [[1, 3, 2]], process=False)
        empty = remove_triangles_outside_oriented_bounds(outside, frame)
        self.assertEqual(len(empty.faces), 0)
        self.assertEqual(len(empty.vertices), 0)

    def test_clip_preserves_custom_geometry_inside_cube(self):
        import trimesh
        from training.mesh2sdf.oriented_meshing import clip_mesh_to_bounds
        mesh = trimesh.creation.box(extents=[6, 2, 2])
        original = mesh.vertices.copy()
        clipped = clip_mesh_to_bounds(mesh, [-2, -2, -2, 2, 2, 2])
        self.assertTrue((clipped.vertices >= -2 - 1e-8).all())
        self.assertTrue((clipped.vertices <= 2 + 1e-8).all())
        np.testing.assert_allclose(mesh.vertices, original)
        np.testing.assert_allclose(clipped.bounds, [[-2, -1, -1], [2, 1, 1]])

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
