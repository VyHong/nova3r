"""Typed option groups for mesh reconstruction and evaluation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.baselines.common import (
    DEFAULT_ALIGNMENT_ITERATIONS,
    DEFAULT_ALIGNMENT_LR,
    DEFAULT_FSCORE_THRESHOLDS,
)


@dataclass(frozen=True)
class ValidationOptions:
    num_save: int
    normalization: str
    use_camera_translation: bool
    rgb_lpips: bool
    rgb_lpips_net: str
    rgb_ray_chunk_size: int

    @classmethod
    def from_config(cls, cfg):
        net = str(cfg.get("val_rgb_lpips_net", "alex"))
        if net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("val_rgb_lpips_net must be alex, vgg, or squeeze")
        ray_chunk_size = int(cfg.get("val_rgb_ray_chunk_size", 262144))
        if ray_chunk_size < 1:
            raise ValueError("val_rgb_ray_chunk_size must be positive")
        return cls(
            num_save=int(cfg.get("val_num_save", 1)),
            normalization=str(cfg.get("surface_geo_normalize_mode", "median")),
            use_camera_translation=bool(cfg.get("surface_geo_use_camera_translation", False)),
            rgb_lpips=bool(cfg.get("val_rgb_lpips", True)),
            rgb_lpips_net=net,
            rgb_ray_chunk_size=ray_chunk_size,
        )


@dataclass(frozen=True)
class MeshOptions:
    level: Any
    method: str
    octree_resolution: int
    algorithm: str
    num_chunks: int
    udf_resolution: int
    mind_resolution: int
    mind_r2: float
    mind_sample_points: int
    mind_sample_iterations: int
    mind_mesh_iterations: int
    dualmesh_max_depth: int
    dualmesh_batch_size: int
    dualmesh_reliability_threshold: float
    dualmesh_subdivision_threshold: float
    dualmesh_projection_threshold: float
    dualmesh_qef_threshold: float
    smooth_borders: bool
    fourier: bool
    vertex_batch_size: int
    rgb_vertex_batch_size: int
    bounds_padding: float
    bounds: np.ndarray | None
    surface_threshold: float | None
    udf_sparse_bricks: bool
    udf_brick_cells: int
    udf_cull_margin: float
    udf_query_batch_size: int
    udf_streamed_grid: bool
    udf_fast_cleanup: bool
    udf_mc_tile_cells: int

    @classmethod
    def from_config(cls, cfg):
        threshold = cfg.get("mesh_udf_surface_threshold", 0.008)
        mind_resolution = int(cfg.get("mesh_mind_resolution", 256))
        mind_r2 = float(cfg.get("mesh_mind_r2", 0.01))
        if mind_resolution < 2:
            raise ValueError("mesh_mind_resolution must be at least 2")
        if not np.isfinite(mind_r2) or mind_r2 <= 0:
            raise ValueError("mesh_mind_r2 must be a positive finite distance")
        mind_sample_points = int(cfg.get("mesh_mind_sample_points", 250000))
        mind_sample_iterations = int(cfg.get("mesh_mind_sample_iterations", 25))
        mind_mesh_iterations = int(cfg.get("mesh_mind_mesh_iterations", 50))
        if mind_sample_points < 1 or mind_sample_iterations < 0 or mind_mesh_iterations < 0:
            raise ValueError("MIND sample points must be positive and iteration counts non-negative")
        dualmesh_max_depth = int(cfg.get("mesh_dualmesh_max_depth", 7))
        dualmesh_batch_size = int(cfg.get("mesh_dualmesh_batch_size", 65536))
        dualmesh_reliability_threshold = float(cfg.get("mesh_dualmesh_reliability_threshold", 0.002))
        dualmesh_subdivision_threshold = float(
            cfg.get("mesh_dualmesh_subdivision_threshold", dualmesh_reliability_threshold)
        )
        dualmesh_projection_threshold = float(
            cfg.get("mesh_dualmesh_projection_threshold", dualmesh_reliability_threshold)
        )
        dualmesh_qef_threshold = float(
            cfg.get("mesh_dualmesh_qef_threshold", dualmesh_reliability_threshold)
        )
        if dualmesh_max_depth < 1 or dualmesh_batch_size < 1:
            raise ValueError("DualMeshUDF max depth and batch size must be positive")
        if not np.isfinite(dualmesh_reliability_threshold) or dualmesh_reliability_threshold <= 0:
            raise ValueError("DualMeshUDF reliability threshold must be positive and finite")
        if not all(
            np.isfinite(value) and value > 0
            for value in (
                dualmesh_subdivision_threshold,
                dualmesh_projection_threshold,
                dualmesh_qef_threshold,
            )
        ):
            raise ValueError("DualMeshUDF thresholds must be positive and finite")
        return cls(
            level=cfg.get("mesh_mc_level", "auto"),
            method=str(cfg.get("mesh_method", "latent2mesh")),
            octree_resolution=int(cfg.get("mesh_octree_resolution", 512)),
            algorithm=str(cfg.get("mesh_mc_algo", "mc")).lower(),
            num_chunks=int(cfg.get("mesh_num_chunks", 40000)),
            udf_resolution=int(cfg.get("mesh_mc", 768)),
            mind_resolution=mind_resolution,
            mind_r2=mind_r2,
            mind_sample_points=mind_sample_points,
            mind_sample_iterations=mind_sample_iterations,
            mind_mesh_iterations=mind_mesh_iterations,
            dualmesh_max_depth=dualmesh_max_depth,
            dualmesh_batch_size=dualmesh_batch_size,
            dualmesh_reliability_threshold=dualmesh_reliability_threshold,
            dualmesh_subdivision_threshold=dualmesh_subdivision_threshold,
            dualmesh_projection_threshold=dualmesh_projection_threshold,
            dualmesh_qef_threshold=dualmesh_qef_threshold,
            smooth_borders=bool(cfg.get("smooth_borders", False)),
            fourier=bool(cfg.get("mesh_fourier", False)),
            vertex_batch_size=int(cfg.get("mesh_udf_vertex_batch_size", 65536)),
            rgb_vertex_batch_size=int(cfg.get("mesh_rgb_vertex_batch_size", 65536)),
            bounds_padding=float(cfg.get("mesh_bounds_padding", 0.2)),
            bounds=cls._validate_bounds(cfg.get("mesh_bounds")),
            surface_threshold=None if threshold is None else float(threshold),
            udf_sparse_bricks=bool(cfg.get("mesh_udf_sparse_bricks", False)),
            udf_brick_cells=int(cfg.get("mesh_udf_brick_cells", 64)),
            udf_cull_margin=float(cfg.get("mesh_udf_cull_margin", 1.25)),
            udf_query_batch_size=int(cfg.get("mesh_udf_query_batch_size", 262144)),
            udf_streamed_grid=bool(cfg.get("mesh_udf_streamed_grid", False)),
            udf_fast_cleanup=bool(cfg.get("mesh_udf_fast_cleanup", False)),
            udf_mc_tile_cells=int(cfg.get("mesh_udf_mc_tile_cells", 0)),
        )

    @staticmethod
    def _validate_bounds(bounds):
        if bounds is None:
            return None
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (6,):
            raise ValueError("mesh_bounds must contain [xmin, ymin, zmin, xmax, ymax, zmax], " f"got shape {bounds.shape}")
        if not np.isfinite(bounds).all() or np.any(bounds[3:] <= bounds[:3]):
            raise ValueError("mesh_bounds must be finite and each maximum must be greater " f"than its minimum, got {bounds.tolist()}")
        return bounds


@dataclass(frozen=True)
class EvaluationOptions:
    points: int | None
    fscore_thresholds: tuple[float, ...]
    alignment_iterations: int
    alignment_lr: float
    seed: int

    @classmethod
    def from_config(cls, cfg):
        return cls(
            points=(
                None if cfg.get("test_eval_points") is None
                else int(cfg["test_eval_points"])
            ),
            fscore_thresholds=tuple(
                float(value)
                for value in cfg.get(
                    "test_fscore_thresholds",
                    DEFAULT_FSCORE_THRESHOLDS,
                )
            ),
            alignment_iterations=int(
                cfg.get(
                    "test_alignment_iterations",
                    DEFAULT_ALIGNMENT_ITERATIONS,
                )
            ),
            alignment_lr=float(cfg.get("test_alignment_lr", DEFAULT_ALIGNMENT_LR)),
            seed=int(cfg.get("test_seed", 0)),
        )
