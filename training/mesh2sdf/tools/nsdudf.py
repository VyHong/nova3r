"""Adapter for the official ECCV 2024 NSDUDF surface detector."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import trimesh


def extract_nsdudf(
    udf_decoder,
    *,
    resolution: int = 129,
    query_batch: int = 10000,
    max_avg_distance_factor: float = 1.2,
    max_max_distance_factor: float = 2.0,
    target_faces: int | None = None,
) -> trimesh.Trimesh:
    if resolution < 17 or query_batch < 1:
        raise ValueError("NSDUDF resolution must be >=17 and query_batch must be positive")

    root = Path(__file__).resolve().parents[3]
    nsd_root = root / "third_party" / "nsdudf"
    for path in (nsd_root / "custom_mc", nsd_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from core.meshing import compute_pseudo_sdf, mesh_marching_cubes
    from core.utils import load_model

    device = udf_decoder.latent.device
    model = load_model(str(nsd_root / "model.pt"), str(device))

    def udf_and_grad(points: torch.Tensor):
        values_out, gradients_out = [], []
        for start in range(0, len(points), query_batch):
            xyz = points[start : start + query_batch].to(device=device, dtype=torch.float32)
            xyz.requires_grad_(True)
            values = udf_decoder(xyz).reshape(-1)
            gradients = torch.autograd.grad(values.sum(), xyz, only_inputs=True)[0]
            values_out.append(values.detach().cpu())
            gradients_out.append(F.normalize(gradients, dim=1).detach().cpu())
        return torch.cat(values_out), torch.cat(gradients_out)

    pseudo_sdf = compute_pseudo_sdf(
        model,
        udf_and_grad,
        n_grid_samples=resolution,
        batch_size=query_batch,
        max_avg_distance_factor=max_avg_distance_factor,
        max_max_distance_factor=max_max_distance_factor,
    )
    mesh = mesh_marching_cubes(pseudo_sdf)
    if mesh is None:
        raise RuntimeError("NSDUDF found no surface")
    if target_faces is not None and target_faces > 0 and len(mesh.faces) > target_faces:
        import pymeshlab

        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(pymeshlab.Mesh(mesh.vertex_matrix(), mesh.face_matrix()))
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preservenormal=True,
            preservetopology=True,
            autoclean=True,
        )
        simplified = mesh_set.current_mesh()
        mesh = trimesh.Trimesh(
            vertices=simplified.vertex_matrix(),
            faces=simplified.face_matrix(),
            process=False,
        )
    return mesh
