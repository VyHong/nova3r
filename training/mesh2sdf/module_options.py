"""Typed option groups for mesh reconstruction and evaluation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.baselines.common import (
    DEFAULT_ALIGNMENT_ITERATIONS,
    DEFAULT_ALIGNMENT_LR,
    DEFAULT_EVAL_POINTS,
    DEFAULT_FSCORE_THRESHOLDS,
)


@dataclass(frozen=True)
class ValidationOptions:
    num_save: int
    normalization: str
    use_camera_translation: bool

    @classmethod
    def from_config(cls, cfg):
        return cls(
            num_save=int(cfg.get("val_num_save", 1)),
            normalization=str(cfg.get("surface_geo_normalize_mode", "median")),
            use_camera_translation=bool(
                cfg.get("surface_geo_use_camera_translation", False)
            ),
        )


@dataclass(frozen=True)
class MeshOptions:
    level: Any
    method: str
    octree_resolution: int
    algorithm: str
    num_chunks: int
    udf_resolution: int
    smooth_borders: bool
    fourier: bool
    vertex_batch_size: int
    bounds_padding: float
    bounds: np.ndarray | None
    surface_threshold: float | None

    @classmethod
    def from_config(cls, cfg):
        threshold = cfg.get("mesh_udf_surface_threshold", 0.008)
        return cls(
            level=cfg.get("mesh_mc_level", "auto"),
            method=str(cfg.get("mesh_method", "latent2mesh")),
            octree_resolution=int(cfg.get("mesh_octree_resolution", 512)),
            algorithm=str(cfg.get("mesh_mc_algo", "mc")).lower(),
            num_chunks=int(cfg.get("mesh_num_chunks", 40000)),
            udf_resolution=int(cfg.get("mesh_mc", 768)),
            smooth_borders=bool(cfg.get("smooth_borders", False)),
            fourier=bool(cfg.get("mesh_fourier", False)),
            vertex_batch_size=int(cfg.get("mesh_udf_vertex_batch_size", 65536)),
            bounds_padding=float(cfg.get("mesh_bounds_padding", 0.1)),
            bounds=cls._validate_bounds(cfg.get("mesh_bounds")),
            surface_threshold=None if threshold is None else float(threshold),
        )

    @staticmethod
    def _validate_bounds(bounds):
        if bounds is None:
            return None
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (6,):
            raise ValueError(
                "mesh_bounds must contain [xmin, ymin, zmin, xmax, ymax, zmax], "
                f"got shape {bounds.shape}"
            )
        if not np.isfinite(bounds).all() or np.any(bounds[3:] <= bounds[:3]):
            raise ValueError(
                "mesh_bounds must be finite and each maximum must be greater "
                f"than its minimum, got {bounds.tolist()}"
            )
        return bounds


@dataclass(frozen=True)
class EvaluationOptions:
    points: int
    fscore_thresholds: tuple[float, ...]
    alignment_iterations: int
    alignment_lr: float
    seed: int

    @classmethod
    def from_config(cls, cfg):
        return cls(
            points=int(cfg.get("test_eval_points", DEFAULT_EVAL_POINTS)),
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
