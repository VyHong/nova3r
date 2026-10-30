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
