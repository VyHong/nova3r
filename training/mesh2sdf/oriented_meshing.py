"""Extract in a mesh-aligned frame while querying the field in scene coordinates."""
from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn


@dataclass
class OrientedMeshBounds:
    bounds: np.ndarray
    to_scene: np.ndarray

    def transform_mesh(self, mesh):
        if mesh is not None:
            mesh.apply_transform(self.to_scene)
        return mesh


def surface_oriented_bounds(surface, padding_fraction):
    import trimesh

    points = surface[..., :3].detach().float().cpu().numpy()
    to_box, extents = trimesh.bounds.oriented_bounds(points)
    half_extents = np.asarray(extents) * (0.5 + padding_fraction)
    return OrientedMeshBounds(np.r_[-half_extents, half_extents], np.linalg.inv(to_box))


class SceneQueryDecoder(nn.Module):
    def __init__(self, decoder, to_scene):
        super().__init__()
        self.decoder = decoder
        self.register_buffer('to_scene', torch.as_tensor(to_scene, dtype=torch.float64))

    def forward(self, queries, *args, **kwargs):
        transform = self.to_scene.to(device=queries.device, dtype=queries.dtype)
        queries = queries @ transform[:3, :3].T + transform[:3, 3]
        return self.decoder(queries, *args, **kwargs)


def decoder_in_box_frame(decoder, frame):
    # Share weights without changing the original decoder or its module registry.
    result = copy.copy(decoder)
    result._modules = decoder._modules.copy()
    result.geo_decoder = SceneQueryDecoder(decoder.geo_decoder, frame.to_scene)
    return result


def clip_mesh_to_bounds(mesh, bounds):
    """Trim a reconstructed mesh in decoder coordinates without changing its sampling frame."""
    bounds = np.asarray(bounds, dtype=np.float64)
    result = mesh.copy()
    for axis in range(3):
        for side, sign in ((0, 1.0), (3, -1.0)):
            origin = np.zeros(3)
            origin[axis] = bounds[axis + side]
            normal = np.zeros(3)
            normal[axis] = sign
            result = result.slice_plane(origin, normal, cap=False)
            if not len(result.faces):
                return result
    result.remove_unreferenced_vertices()
    return result


def enclosing_axis_aligned_bounds(frame):
    """Enclose an oriented box in the decoder coordinate system."""
    corners = np.array([[x, y, z] for x in frame.bounds[[0, 3]]
                        for y in frame.bounds[[1, 4]]
                        for z in frame.bounds[[2, 5]]])
    scene = corners @ frame.to_scene[:3, :3].T + frame.to_scene[:3, 3]
    return np.r_[scene.min(axis=0), scene.max(axis=0)]


def remove_triangles_outside_oriented_bounds(mesh, frame, tolerance=1e-7):
    """Keep whole triangles only when all vertices lie inside the snug box.

    Unlike plane slicing, this creates no new vertices or boundary triangles.
    The input mesh and its coordinate system are preserved.
    """
    result = mesh.copy()
    to_box = np.linalg.inv(frame.to_scene)
    vertices = np.asarray(result.vertices)
    local = vertices @ to_box[:3, :3].T + to_box[:3, 3]
    inside = ((local >= frame.bounds[:3] - tolerance) &
              (local <= frame.bounds[3:] + tolerance)).all(axis=1)
    result.update_faces(inside[np.asarray(result.faces)].all(axis=1))
    result.remove_unreferenced_vertices()
    return result
