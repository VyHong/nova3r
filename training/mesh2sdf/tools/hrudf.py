"""Adapter for the official NeurIPS 2025 HRUDF mesher."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import trimesh


def extract_hrudf(
    udf_decoder,
    *,
    resolution: int = 129,
    query_batch: int = 10000,
    recursions: int = 5,
    clamp_distance: float = 0.1,
) -> trimesh.Trimesh:
    if resolution < 17 or query_batch < 1 or recursions < 0:
        raise ValueError("HRUDF resolution must be >=17, query_batch positive, and recursions nonnegative")

    root = Path(__file__).resolve().parents[3]
    hrudf_root = root / "third_party" / "hrudf"
    for path in (hrudf_root / "custom_mc", hrudf_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # HRUDF and NSDUDF both use a top-level package named ``core``. Ensure the
    # official HRUDF package wins if both adapters run in the same interpreter.
    for name in tuple(sys.modules):
        if name == "core" or name.startswith("core."):
            del sys.modules[name]

    from core.meshing import compute_pseudo_sdf, mesh_marching_cubes
    from core.utils import load_model

    device = udf_decoder.latent.device
    model, config = load_model(str(hrudf_root / "weights"), str(device))

    def udf_and_grad(points: torch.Tensor):
        xyz = points.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
        values = udf_decoder(xyz).reshape(-1)
        gradients = torch.autograd.grad(values.sum(), xyz, only_inputs=True)[0]
        return values.detach().cpu(), F.normalize(gradients, dim=1).detach().cpu()

    pseudo_sdfs = compute_pseudo_sdf(
        model,
        udf_and_grad_f=udf_and_grad,
        n_grid_samples=resolution,
        batch_size_udf=query_batch,
        batch_size_pseudosdf=query_batch,
        num_recursions=recursions,
        recursion_type=config["recursion_type"],
        clamp_distance=clamp_distance,
        export_recursions="last",
    )
    mesh = mesh_marching_cubes(pseudo_sdfs[-1])
    if mesh is None:
        raise RuntimeError("HRUDF found no surface")
    return mesh
