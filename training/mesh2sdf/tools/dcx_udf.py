"""Adapter from Nova3R's learned UDF to the official DCx extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


def extract_dcx_udf(
    udf_decoder,
    *,
    resolution: int = 128,
    iterations: int = 25,
    max_batch: int = 262144,
    threshold: float = 0.005,
    learning_rate: float = 0.0005,
    warm_up_end: int = 25,
    scan_block_size: int = 1,
    scan_lipschitz_safety: float = 1.5,
    projection_batch_size: int = 50000,
    thinning: bool = True,
    postprocessing: bool = True,
    work_dir: str | Path = "/tmp/nova3r_dcx",
) -> trimesh.Trimesh:
    if resolution < 16 or iterations < 1 or max_batch < 1 or learning_rate <= 0 or warm_up_end < 0:
        raise ValueError("DCx resolution must be >=16 and iterations/batch must be positive")
    root = Path(__file__).resolve().parents[3]
    for path in (root / ".deps" / "dcx", root / "third_party" / "DCx"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import dcx_pkg
        from DCX.upsampling_speedup_according_to_gradients_of_initial_points_directly_removefar import DCX
    except ImportError as exc:
        raise ImportError("DCx core must be built into .deps/dcx") from exc

    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # DCx operates in [-0.5, 0.5]^3. Nova3R's adapter operates in [-1, 1]^3;
    # coordinates and distances therefore both differ by a factor of two.
    def query(points):
        return udf_decoder(points * 2.0) * 0.5

    evaluator = DCX(
        query,
        resolution,
        threshold,
        dest_dir=str(destination) + "/",
        model_name="nova3r",
        iter_max=iterations,
        bound_min=np.full(3, -0.5, dtype=np.float32),
        bound_max=np.full(3, 0.5, dtype=np.float32),
        max_batch=max_batch,
        learning_rate=learning_rate,
        warm_up_end=min(warm_up_end, iterations),
        report_freq=max(1, iterations // 5),
        scan_block_size=scan_block_size,
        scan_lipschitz_safety=scan_lipschitz_safety,
        projection_batch_size=projection_batch_size,
    )
    points, bbox = evaluator.optimize1()
    if not len(points):
        raise RuntimeError("DCx found no near-surface points")

    voxel_ids, voxel_points, bbox, orders = dcx_pkg.points_to_voxels(
        points=np.asarray(points, dtype=np.float32),
        bbox=np.asarray(bbox, dtype=np.float32).reshape(-1),
        res=resolution,
    )
    cube_ids, cube_types = dcx_pkg.get_cube_types(
        voxel_ids=voxel_ids,
        voxel_points=voxel_points,
        orders=orders,
        res=resolution,
    )
    if thinning:
        voxel_ids, voxel_points, cube_ids, cube_types = dcx_pkg.thinning(
            voxel_ids=voxel_ids,
            voxel_points=voxel_points,
            cube_ids=cube_ids,
            cube_types=cube_types,
            orders=orders,
            res=resolution,
        )
    _, vertices, faces = dcx_pkg.reconstruction(
        voxel_ids=voxel_ids,
        voxel_points=voxel_points,
        cube_ids=cube_ids,
        cube_types=cube_types,
        orders=orders,
        res=resolution,
        pattern=0,
        enable_postprocessing=postprocessing,
        dataname="nova3r",
    )
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32) * 2.0,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
