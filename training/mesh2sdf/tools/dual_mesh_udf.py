"""DualMeshUDF extraction for Nova3R's implicit unsigned-distance decoder."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


class DualMeshUDFDecoder:
    """Expose a ShapeVAE geometry decoder as a normalized UDF on [-1, 1]^3."""

    def __init__(self, geo_decoder, latent, bounds, udf_scale=128.0):
        self.geo_decoder = geo_decoder
        self.latent = latent
        self.udf_scale = float(udf_scale)
        bounds = torch.as_tensor(bounds, dtype=torch.float32, device=latent.device)
        mins, maxs = bounds[:3], bounds[3:]
        extents = maxs - mins
        if not torch.isfinite(bounds).all() or torch.any(extents <= 0):
            raise ValueError(f"Mesh bounds must be finite with positive extents, got {bounds}")
        self.center = (mins + maxs) * 0.5
        self.half_extent = (extents.max() * 0.5).clamp_min(1e-6)

    def _distance(self, unit_queries: torch.Tensor) -> torch.Tensor:
        world_queries = unit_queries.to(
            device=self.latent.device, dtype=self.latent.dtype
        ) * self.half_extent.to(self.latent.dtype) + self.center.to(self.latent.dtype)
        logits = self.geo_decoder(
            queries=world_queries.unsqueeze(0), latents=self.latent
        )
        return F.softplus(logits.float()).reshape(-1, 1) / (
            self.udf_scale * self.half_extent
        )

    def values(self, coords: np.ndarray) -> np.ndarray:
        points = torch.as_tensor(coords, dtype=torch.float32, device=self.latent.device)
        with torch.no_grad():
            values = self._distance(points)
        return values.detach().cpu().numpy()

    def values_and_gradients(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = torch.as_tensor(
            coords, dtype=torch.float32, device=self.latent.device
        ).requires_grad_(True)
        with torch.enable_grad():
            values = self._distance(points)
            gradients = torch.autograd.grad(values.sum(), points, only_inputs=True)[0]
        return (
            values.detach().cpu().numpy(),
            gradients.detach().cpu().numpy(),
        )

    def mesh_to_world(self, vertices: np.ndarray) -> np.ndarray:
        return (
            np.asarray(vertices, dtype=np.float32)
            * float(self.half_extent.detach().cpu())
            + self.center.detach().cpu().numpy()
        )


def extract_dual_mesh_udf(
    geo_decoder,
    latent: torch.Tensor,
    bounds,
    *,
    max_depth: int = 7,
    batch_size: int = 65536,
    reliability_threshold: float = 0.002,
    subdivision_threshold: float | None = None,
    projection_threshold: float | None = None,
    qef_threshold: float | None = None,
) -> trimesh.Trimesh:
    """Extract a triangle mesh with the installed DualMeshUDF implementation."""
    if max_depth < 1:
        raise ValueError("DualMeshUDF max_depth must be positive")
    # Upstream DualMeshUDF uses signed 32-bit flattened grid IDs. It allocates
    # a grid with (2 ** (max_depth + 1) + 1) cells along each dimension, so a
    # depth of 10 (1024 output resolution) requires 2049 ** 3 IDs and overflows
    # before extraction starts. Reject it here rather than allowing the native
    # extension to segfault the Python process.
    if max_depth > 9:
        raise ValueError(
            "DualMeshUDF max_depth > 9 is unsupported by the upstream native "
            "core: its signed 32-bit grid IDs overflow at resolution 1024"
        )
    if batch_size < 1:
        raise ValueError("DualMeshUDF batch_size must be positive")
    if not np.isfinite(reliability_threshold) or reliability_threshold <= 0:
        raise ValueError("DualMeshUDF reliability_threshold must be positive and finite")
    subdivision_threshold = float(
        reliability_threshold if subdivision_threshold is None else subdivision_threshold
    )
    projection_threshold = float(
        reliability_threshold if projection_threshold is None else projection_threshold
    )
    qef_threshold = float(reliability_threshold if qef_threshold is None else qef_threshold)
    for name, value in (
        ("subdivision_threshold", subdivision_threshold),
        ("projection_threshold", projection_threshold),
        ("qef_threshold", qef_threshold),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"DualMeshUDF {name} must be positive and finite")
    try:
        import igl
        from DualMeshUDF.extract_mesh import query_udf, query_udf_and_grad
        from DualMeshUDF_core import Octree, triangulate_faces
    except ImportError as exc:
        raise ImportError(
            "DualMeshUDF and libigl are required for mesh_method=dualmeshudf"
        ) from exc

    adapter = DualMeshUDFDecoder(geo_decoder, latent, bounds)
    # This is the small upstream extraction loop, reproduced here because the
    # reference package hard-codes 0.002 for all three reliability checks.
    octree = Octree(
        max_depth=max_depth,
        min_corner=np.array([[-1.0], [-1.0], [-1.0]]),
        max_corner=np.array([[1.0], [1.0], [1.0]]),
        sampling_depth=1,
    )
    for _ in range(max_depth + 1):
        centroids = octree.centroids_of_new_nodes().astype(np.float32)
        distances, gradients = query_udf_and_grad(
            adapter.values_and_gradients, centroids, batch_size
        )
        octree.adaptive_subdivide(distances, gradients, subdivision_threshold)

    grid_indices, grid_coords = octree.get_samples_of_new_nodes()
    distances, gradients = query_udf_and_grad(
        adapter.values_and_gradients, grid_coords.astype(np.float32), batch_size
    )
    octree.set_new_grid_data(grid_indices, distances, gradients)

    indices, projections = octree.get_projections_for_checking_validity()
    projection_distances = query_udf(adapter.values, projections, batch_size)
    octree.set_grid_validity(indices, projection_distances < projection_threshold)
    octree.batch_solve(qef_threshold, 1.0, 1.0, 0.15, 0.08)
    octree.generate_mesh()

    triangle_faces = triangulate_faces(
        octree.mesh_v, octree.mesh_f, octree.v_type, octree.mesh_v_dir
    )
    vertices, _, _, faces = igl.remove_duplicate_vertices(
        np.asarray(octree.mesh_v), triangle_faces, 1e-7
    )
    vertices, faces, _, _ = igl.remove_unreferenced(vertices, faces)
    return trimesh.Trimesh(
        vertices=adapter.mesh_to_world(vertices),
        faces=np.asarray(faces),
        process=False,
    )
