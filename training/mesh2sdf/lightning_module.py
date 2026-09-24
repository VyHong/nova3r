import hashlib
import json
import traceback
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric, SumMetric
from demo_nova3r import render_360_video
from nova3r.heads.hunyuan_model.autoencoders.model import DiagonalGaussianDistribution
from nova3r.losses import L21, FMVelocity, Pts3D_Regr3D_CD_V4, ReconstructionLoss, SDFReconstructionLoss, MSE
from eval.baselines.common import (
    build_evaluation_summary,
    evaluate_point_clouds,
    sample_mesh_surface,
    sample_points,
    write_evaluation_results,
)

from nova3r.heads.hunyuan_model.pipelines import (
    colorize_trimesh_vertices,
    export_to_trimesh,
)
from nova3r.utils.batch_debug import save_batch_debug
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import math
from torch.optim.lr_scheduler import LRScheduler

from training.mesh2sdf.tools.mesh_udf.mesh_udf import get_mesh_udf_fast
from training.mesh2sdf.module_options import (
    EvaluationOptions,
    MeshOptions,
    ValidationOptions,
)
from training.module_setup import (
    FlowTrainingOptions,
    OptimizerOptions,
    build_frozen_shape_decoder,
    freeze_layers,
)


class UDFGeoDecoderAdapter(nn.Module):
    def __init__(self, geo_decoder, latent, bounds, udf_scale=128.0):
        super().__init__()
        self.geo_decoder = geo_decoder
        self.latent = latent
        self.udf_scale = float(udf_scale)
        bounds = torch.as_tensor(bounds, dtype=latent.dtype, device=latent.device)
        mins, maxs = bounds[:3], bounds[3:]
        extents = maxs - mins
        if not torch.isfinite(bounds).all() or torch.any(extents <= 0):
            raise ValueError(f"Mesh bounds must be finite with positive extents, got {bounds}")
        self.center = (mins + maxs) * 0.5
        self.half_extent = (extents.max() * 0.5).clamp_min(1e-6)
        self.grid_half_extents = extents * (0.5 / self.half_extent)

    def forward(self, decoder_input):
        unit_queries = decoder_input[:, -3:].view(1, -1, 3).to(device=self.latent.device, dtype=self.latent.dtype)
        queries = unit_queries * self.half_extent + self.center
        logits = self.geo_decoder(queries=queries, latents=self.latent)
        # Training predicts UDF distances scaled by 128; MC expects distances in unit-grid coordinates.
        return (F.softplus(logits) / (self.udf_scale * self.half_extent)).view(-1, 1)

    def mesh_to_world(self, mesh):
        mesh.vertices = mesh.vertices * float(self.half_extent.detach().cpu()) + self.center.detach().cpu().numpy()
        return mesh

    def mesh_udf_grid(self, max_resolution):
        """Return tight normalized bounds with near-uniform world-space voxel density."""
        if max_resolution < 2:
            raise ValueError("MeshUDF resolution must be at least 2")
        half_extents = self.grid_half_extents.detach().float().cpu().numpy()
        grid_bounds = np.concatenate([-half_extents, half_extents])
        target_spacing = 2.0 / (max_resolution - 1)
        intervals = np.maximum(1, np.ceil((2.0 * half_extents) / target_spacing).astype(np.int64))
        resolution = tuple((intervals + 1).tolist())
        return grid_bounds, resolution


class Mesh2SDFLightningModule(pl.LightningModule):
    FLOW_TIMESTEP_BUCKETS = (
        (0.0, 0.1),
        (0.1, 0.3),
        (0.3, 0.5),
        (0.5, 0.8),
        (0.8, 1.0),
    )

    def __init__(self, cfg, model):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.cfg = cfg

        self.model = model
        self.optimizer_options = OptimizerOptions.from_config(cfg)
        self.first_stage = build_frozen_shape_decoder(cfg, model)
        self.kl_weight = float(cfg.kl_weight)

        FlowTrainingOptions.from_config(cfg).apply_to(model)
        freeze_layers(model, cfg.get("layers_to_freeze", ()))

        self.train_criterion = SDFReconstructionLoss(MSE)
        self.val_criterion = SDFReconstructionLoss(MSE)
        self.test_criterion = None

        self.validation_options = ValidationOptions.from_config(cfg)
        self.mesh_options = MeshOptions.from_config(cfg)
        self.evaluation_options = EvaluationOptions.from_config(cfg)
        self._init_flow_metrics()

    def _init_flow_metrics(self):
        self.flow_bucket_velocity_mse = nn.ModuleDict()
        self.flow_bucket_reconstruction_mse = nn.ModuleDict()
        self.flow_bucket_sample_count = nn.ModuleDict()
        for lower, upper in self.FLOW_TIMESTEP_BUCKETS:
            label = self._flow_timestep_bucket_label(lower, upper)
            self.flow_bucket_velocity_mse[label] = MeanMetric()
            self.flow_bucket_reconstruction_mse[label] = MeanMetric()
            self.flow_bucket_sample_count[label] = SumMetric()

    @staticmethod
    def _median_normalization_target(mode):
        if mode == "median":
            return 1.0
        return float(mode.split("_")[-1])

    def _resolve_mesh_mc_level(self, bounds, octree_resolution):
        cfg_value = self.mesh_options.level
        if not (isinstance(cfg_value, str) and cfg_value.lower() == "auto"):
            return float(cfg_value)

        bounds = np.asarray(bounds, dtype=np.float64)
        extent = np.max(bounds[3:] - bounds[:3])
        voxel_world_size = extent / max(int(octree_resolution) - 1, 1)
        return float(np.clip(voxel_world_size * 128.0, 1e-4, 1.0))

    def normalize_surface_and_geo_points(self, batch, mode):
        """Normalize surface XYZ, query XYZ, and distances with a shared transform.

        Geometry supplied by the SDF datasets is in the first camera's frame,
        making the camera center the origin. When camera translation is enabled,
        normalization preserves that origin. Cube mode still derives its scale
        from the bounding cube; median mode translates to the camera before
        deriving its radial scale.
        """
        if mode == "none" or "surface" not in batch:
            return batch

        surface = batch["surface"]
        geo_points = batch.get("geo_points")

        camera_center = surface.new_zeros((surface.shape[0], 1, 3))
        center = camera_center
        if mode == "cube":
            lower = surface[..., :3].amin(dim=1, keepdim=True)
            upper = surface[..., :3].amax(dim=1, keepdim=True)
            cube_center = (lower + upper) * 0.5
            norm_factor = (upper - lower).amax(dim=-1).squeeze(1) * 0.5
            if not torch.isfinite(norm_factor).all() or torch.any(norm_factor <= 0):
                raise ValueError("Cannot cube-normalize a non-finite or zero-extent surface")
            coord_scale = norm_factor.reciprocal().view(-1, 1, 1)
            # Applying the cube transform and then translating its transformed
            # camera center back to zero is equivalent to scaling around the
            # camera center.
            center = (
                camera_center
                if self.validation_options.use_camera_translation
                else cube_center
            )
        else:
            target_median = self._median_normalization_target(mode)
            if self.validation_options.use_camera_translation:
                center = camera_center
            distances = torch.linalg.norm(surface[..., :3] - center, dim=-1)
            norm_factor = torch.median(distances, dim=1).values
            norm_factor = torch.clamp(norm_factor, min=0.01, max=100.0)
            coord_scale = (surface.new_tensor(float(target_median)) / norm_factor).view(-1, 1, 1)

        normalized_surface = surface.clone()
        normalized_surface[..., :3] = torch.clamp(
            (normalized_surface[..., :3] - center) * coord_scale,
            min=-1000.0,
            max=1000.0,
        )

        normalized_batch = dict(batch)
        normalized_batch["surface"] = normalized_surface
        if geo_points is not None:
            normalized_geo_points = geo_points.clone()
            normalized_geo_points[..., :3] = torch.clamp(
                (normalized_geo_points[..., :3] - center) * coord_scale,
                min=-1000.0,
                max=1000.0,
            )
            normalized_geo_points[..., 3:4] = (
                normalized_geo_points[..., 3:4] * coord_scale
            )
            normalized_batch["geo_points"] = normalized_geo_points
        normalized_batch["surface_geo_norm_center"] = center
        normalized_batch["surface_geo_norm_factor"] = norm_factor
        return normalized_batch

    def training_step(self, batch, batch_idx):
        batch = self.normalize_surface_and_geo_points(
            batch,
            mode=self.validation_options.normalization,
        )
        loss, details, _ = self.model.lightning_forward(
            batch,
            self.train_criterion,
            kl_weight=self.kl_weight,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if "rgb_loss" in details:
            self.log(
                "train_rgb_loss",
                details["rgb_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.normalize_surface_and_geo_points(
            batch,
            mode=self.validation_options.normalization,
        )
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end

        loss, details, _ = self.model.lightning_forward(
            batch,
            self.val_criterion,
            kl_weight=0.0,
            deterministic_seed=42 + batch_idx,
        )

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        bucket_samples = details.pop("timestep_bucket_samples", None)
        if bucket_samples is not None:
            self._accumulate_flow_timestep_buckets(bucket_samples)

        for name, value in details.items():
            self.log(
                f"val_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def on_validation_epoch_start(self):
        for metrics in (
            self.flow_bucket_velocity_mse,
            self.flow_bucket_reconstruction_mse,
            self.flow_bucket_sample_count,
        ):
            for metric in metrics.values():
                metric.reset()

    @staticmethod
    def _flow_timestep_bucket_label(lower, upper):
        return f"t_{lower:g}_{upper:g}".replace(".", "p")

    @torch.no_grad()
    def _accumulate_flow_timestep_buckets(self, samples):
        t = samples["t"].to(device=self.device)
        velocity_mse = samples["velocity_mse"].to(device=self.device)
        reconstruction_mse = samples["reconstruction_mse"].to(device=self.device)

        for lower, upper in self.FLOW_TIMESTEP_BUCKETS:
            # Include t=1 in the final bucket. torch.rand normally produces
            # t<1, but this keeps the definition complete for explicit tests.
            mask = (t >= lower) & (t <= upper if upper == 1.0 else t < upper)
            if not mask.any():
                continue
            label = self._flow_timestep_bucket_label(lower, upper)
            self.flow_bucket_velocity_mse[label].update(velocity_mse[mask])
            self.flow_bucket_reconstruction_mse[label].update(reconstruction_mse[mask])
            self.flow_bucket_sample_count[label].update(mask.sum())

    def _log_flow_timestep_buckets(self):
        for lower, upper in self.FLOW_TIMESTEP_BUCKETS:
            label = self._flow_timestep_bucket_label(lower, upper)
            self.log(
                f"val_velocity_mse/{label}",
                self.flow_bucket_velocity_mse[label],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val_reconstruction_mse/{label}",
                self.flow_bucket_reconstruction_mse[label],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val_timestep_samples/{label}",
                self.flow_bucket_sample_count[label],
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        self._log_flow_timestep_buckets()
        # Validation runs on every DDP process, but mesh extraction and rendering
        # should only consume memory on the global-zero GPU.
        if not self.trainer.is_global_zero:
            self.val_batch_to_log = None
            return

        decoder = self._latent_decoder()
        if decoder is None:
            self.val_batch_to_log = None
            return

        val_batch_to_log = self._slice_batch(
            self.val_batch_to_log,
            self.validation_options.num_save,
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, recon_details, latents = self.model.lightning_forward(
                val_batch_to_log,
                self.val_criterion,
                kl_weight=0.0,
                recon_latents=True,
                deterministic_seed=42,
            )
            latents = self._decode_latents_for_geometry(latents, decoder)
            surface = val_batch_to_log["surface"]

            sample_dir = f"{self.current_log_dir}/val_points/epoch_{self.current_epoch}"
            os.makedirs(sample_dir, exist_ok=True)

            # save_batch_debug(
            #     val_batch_to_log,
            #     sample_dir,
            #     max_samples=self.validation_options.num_save,
            #     batched=True,
            # )

            meshes_to_render = []
            validation_evaluations = []
            for sample_idx in range(latents.shape[0]):
                bounds = self._reconstruction_bounds(surface[sample_idx])

                mesh_dir = os.path.join(
                    sample_dir,
                    f"{val_batch_to_log["seq_name"][sample_idx]}_{val_batch_to_log["id"][sample_idx]}_{val_batch_to_log["subseq_ids"][sample_idx][0]}",
                )
                os.makedirs(mesh_dir, exist_ok=True)
                try:
                    mesh = self.mesh_from_latents(latents, sample_idx, bounds, decoder)
                except RuntimeError as exc:
                    if "No surface found at the given iso value" not in str(exc):
                        raise
                    mesh = None

                if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    glb_path = os.path.join(mesh_dir, "val_mesh.ply")
                    mesh.export(glb_path)
                    meshes_to_render.append((glb_path, mesh_dir))

                    sample_name = f"validation_sample_{sample_idx}"
                    try:
                        sample_name = self._test_sample_name(
                            val_batch_to_log,
                            sample_idx,
                        )
                        seed = self._test_sample_seed(sample_name)
                        world_scale = self._surface_world_scale(
                            val_batch_to_log,
                            sample_idx,
                        )
                        pred = sample_mesh_surface(
                            np.asarray(mesh.vertices) * world_scale,
                            mesh.faces,
                            self.evaluation_options.points,
                            seed,
                        )
                        target = val_batch_to_log["surface"][
                            sample_idx, :, :3
                        ].detach().float().cpu().numpy()
                        target = sample_points(
                            target * world_scale,
                            self.evaluation_options.points,
                            seed,
                        )
                        with torch.autocast(
                            device_type=self.device.type,
                            enabled=False,
                        ):
                            _, evaluation = evaluate_point_clouds(
                                pred,
                                target,
                                thresholds=self.evaluation_options.fscore_thresholds,
                                device=str(self.device),
                                seed=seed,
                                alignment_iterations=self.evaluation_options.alignment_iterations,
                                alignment_lr=self.evaluation_options.alignment_lr,
                            )
                        validation_evaluations.append(evaluation)
                    except Exception as exc:
                        self.print(
                            f"Validation metric calculation failed for "
                            f"{sample_name}: {exc}"
                        )

        self._log_validation_reconstruction_metrics(validation_evaluations)

        for glb_path, mesh_dir in meshes_to_render:
            try:
                render_360_video(glb_path, mesh_dir)
            except Exception as exc:
                self.print(f"Validation preview rendering failed for {glb_path}: {exc}")

        self.val_batch_to_log = None

    def _log_validation_reconstruction_metrics(self, evaluations):
        """Log means over the meshes reconstructed for validation previews."""
        if not evaluations:
            return

        metric_names = (
            "chamfer_forward",
            "chamfer_backward",
            "chamfer_mean",
            *(
                f"f_score_{threshold}_full"
                for threshold in self.evaluation_options.fscore_thresholds
            ),
        )
        for name in metric_names:
            values = [evaluation[name] for evaluation in evaluations if name in evaluation]
            if values:
                self.log(
                    f"val_{name}",
                    float(np.mean(values)),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )

    def _latent_decoder(self):
        if hasattr(self.model, "first_stage") and self.model.first_stage is not None:
            return self.model.first_stage
        if self.first_stage is not None:
            return self.first_stage
        if hasattr(self.model, "latents2mesh") or hasattr(self.model, "geo_decoder"):
            return self.model
        return None

    @staticmethod
    def _decode_latents_for_geometry(latents, decoder):
        """Convert raw VAE latents [B, N, C] to decoder-width tokens once."""
        if latents is None:
            return None
        latent_shape = getattr(decoder, "latent_shape", None)
        if latent_shape is not None and latents.shape[-1] == latent_shape[-1]:
            return decoder.decode(latents)
        return latents

    def mesh_from_latents(self, latent_batch, sample_idx, bounds, decoder, method=None):
        method = self.mesh_options.method if method is None else method
        mesh = None
        if method == "latent2mesh":
            mc_level = self._resolve_mesh_mc_level(
                bounds,
                self.mesh_options.octree_resolution,
            )

            if isinstance(self.mesh_options.level, str):
                self.print(f"Auto mesh_mc_level={mc_level:.6g} " f"for bounds extent={float(np.max(bounds[3:] - bounds[:3])):.6g}, ")

            mesh_outputs = decoder.latents2mesh(
                latent_batch[sample_idx : sample_idx + 1],
                output_type="trimesh",
                bounds=bounds,
                mc_level=mc_level,
                num_chunks=self.mesh_options.num_chunks,
                octree_resolution=self.mesh_options.octree_resolution,
                mc_algo=self.mesh_options.algorithm,
            )
            meshes = export_to_trimesh(mesh_outputs)
            mesh = meshes[0] if isinstance(meshes, list) and meshes else meshes
        elif method == "mesh_udf":
            with torch.inference_mode(False), torch.enable_grad():
                latent_target = latent_batch[sample_idx : sample_idx + 1].float().detach().clone()
                udf_decoder = UDFGeoDecoderAdapter(decoder.geo_decoder, latent_target, bounds)
                grid_bounds, grid_resolution = udf_decoder.mesh_udf_grid(
                    self.mesh_options.udf_resolution
                )
                _, _, mesh = get_mesh_udf_fast(
                    udf_decoder,
                    None,
                    N_MC=grid_resolution,
                    gradient=False,
                    smooth_borders=self.mesh_options.smooth_borders,
                    fourier=self.mesh_options.fourier,
                    vertex_max_batch=self.mesh_options.vertex_batch_size,
                    surface_threshold=self.mesh_options.surface_threshold,
                    grid_bounds=grid_bounds,
                    device=latent_target.device,
                )
                if mesh is not None:
                    mesh = udf_decoder.mesh_to_world(mesh)
        else:
            raise ValueError(f"Unknown mesh extraction method: {method}")

        return colorize_trimesh_vertices(
            mesh,
            decoder,
            latent_batch[sample_idx : sample_idx + 1],
            batch_size=self.mesh_options.rgb_vertex_batch_size,
        )

    @staticmethod
    def _slice_batch(batch, max_elements):
        sliced = {}
        for key, value in batch.items():
            if key in ["images", "intrinsics", "extrinsics"] and isinstance(value, list):
                sliced[key] = [Mesh2SDFLightningModule._slice_batch_value(item, max_elements) for item in value]
            elif isinstance(value, torch.Tensor):
                sliced[key] = value[:max_elements]
            elif isinstance(value, list):
                sliced[key] = value[:max_elements]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _slice_batch_value(value, max_elements):
        if isinstance(value, torch.Tensor):
            return value[:max_elements]
        if isinstance(value, list):
            return [Mesh2SDFLightningModule._slice_batch_value(item, max_elements) for item in value]
        return value

    @staticmethod
    def _surface_bounds(surface, padding_fraction=0.1):
        points = surface[..., :3]
        minimum = points.min(dim=0).values.detach().cpu().numpy()
        maximum = points.max(dim=0).values.detach().cpu().numpy()
        padding = (maximum - minimum) * padding_fraction
        return np.concatenate([minimum - padding, maximum + padding])

    def _reconstruction_bounds(self, surface):
        if self.mesh_options.bounds is not None:
            return self.mesh_options.bounds.copy()
        return self._surface_bounds(surface, self.mesh_options.bounds_padding)

    @staticmethod
    def _batch_item(value, index):
        if isinstance(value, torch.Tensor):
            value = value[index]
            return value.item() if value.ndim == 0 else value
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    def _test_sample_name(self, batch, sample_idx):
        seq_name = str(self._batch_item(batch.get("seq_name", "sample"), sample_idx))
        sample_id = str(self._batch_item(batch.get("id", sample_idx), sample_idx))
        subseq_ids = self._batch_item(batch["subseq_ids"], sample_idx) if "subseq_ids" in batch else 0
        if isinstance(subseq_ids, torch.Tensor):
            first_subseq_id = subseq_ids.reshape(-1)[0].item()
        elif isinstance(subseq_ids, (list, tuple)):
            first_subseq_id = subseq_ids[0]
        else:
            first_subseq_id = subseq_ids
        name = f"{seq_name}_{sample_id}_{first_subseq_id}"
        return name.replace("/", "_").replace(os.sep, "_")

    def _test_sample_seed(self, sample_name):
        digest = hashlib.sha256(sample_name.encode("utf-8")).digest()
        return self.evaluation_options.seed + int.from_bytes(digest[:4], "little")

    def _surface_world_scale(self, batch, sample_idx):
        if "surface_geo_norm_factor" not in batch:
            return 1.0
        norm_factor = self._batch_item(batch["surface_geo_norm_factor"], sample_idx)
        mode = self.validation_options.normalization
        target = 1.0 if mode == "cube" else self._median_normalization_target(mode)
        if isinstance(norm_factor, torch.Tensor):
            norm_factor = norm_factor.detach().cpu()
        return float(norm_factor) / target

    def on_test_epoch_start(self):
        self.test_results = []

    def test_step(self, batch, batch_idx):
        batch = self.normalize_surface_and_geo_points(
            batch,
            mode=self.validation_options.normalization,
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, _, latents = self.model.lightning_forward(
                batch,
                self.val_criterion,
                kl_weight=0.0,
                recon_latents=True,
                deterministic_seed=42 + batch_idx,
            )

        if latents is None:
            raise RuntimeError("The model did not return reconstruction latents during testing")
        decoder = self._latent_decoder()
        if decoder is None:
            raise RuntimeError(
                "Testing generated latents requires a VAE decoder. Configure "
                "experiment.vae and experiment.vae_ckpt."
            )
        latents = self._decode_latents_for_geometry(latents, decoder)
        output_dir = Path(self.current_log_dir) / "test"
        mesh_root = output_dir / "meshes"
        if self.trainer.world_size > 1:
            mesh_root = mesh_root / f"rank_{self.global_rank:03d}"
        mesh_root.mkdir(parents=True, exist_ok=True)

        records = []
        for sample_idx in range(latents.shape[0]):
            sample_name = self._test_sample_name(batch, sample_idx)
            sample_dir = mesh_root / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            record = {"sample": sample_name, "rank": self.global_rank}

            try:
                bounds = self._reconstruction_bounds(batch["surface"][sample_idx])

                mesh = self.mesh_from_latents(
                    latents,
                    sample_idx,
                    bounds,
                    decoder,
                    method="mesh_udf",
                )
                if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    raise ValueError("MeshUDF did not produce a non-empty mesh")

                world_scale = self._surface_world_scale(batch, sample_idx)
                world_mesh = mesh.copy()
                world_mesh.vertices = np.asarray(world_mesh.vertices) * world_scale
                mesh_path = sample_dir / "test_mesh.ply"
                world_mesh.export(mesh_path)

                seed = self._test_sample_seed(sample_name)
                pred = sample_mesh_surface(
                    world_mesh.vertices,
                    world_mesh.faces,
                    self.evaluation_options.points,
                    seed,
                )
                target = batch["surface"][sample_idx, :, :3].detach().float().cpu().numpy()
                target = sample_points(
                    target * world_scale,
                    self.evaluation_options.points,
                    seed,
                )
                _, evaluation = evaluate_point_clouds(
                    pred,
                    target,
                    thresholds=self.evaluation_options.fscore_thresholds,
                    device=str(self.device),
                    seed=seed,
                    alignment_iterations=self.evaluation_options.alignment_iterations,
                    alignment_lr=self.evaluation_options.alignment_lr,
                )
                record.update(
                    {
                        "status": "ok",
                        "mesh": str(mesh_path),
                        **evaluation,
                    }
                )
            except Exception as exc:
                record.update({"status": "error", "error": str(exc)})
                self.print(f"MeshUDF test failed for {sample_name}: {exc}")
                traceback.print_exc()

            records.append(record)
            (sample_dir / "metrics.json").write_text(json.dumps(record, indent=2))

        self.test_results.extend(records)
        return records

    def on_test_epoch_end(self):
        results = self.test_results
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, results)
            results = [record for rank_results in gathered for record in rank_results]

        if not self.trainer.is_global_zero:
            return

        # A DistributedSampler may pad its shards; keep a single record per sample.
        results = list({record["sample"]: record for record in results}.values())
        output_dir = Path(self.current_log_dir) / "test"
        summary = build_evaluation_summary(
            results,
            model=type(self.model).__name__,
            thresholds=self.evaluation_options.fscore_thresholds,
            alignment_iterations=self.evaluation_options.alignment_iterations,
            alignment_lr=self.evaluation_options.alignment_lr,
        )
        write_evaluation_results(results, summary, output_dir)
        self.print(json.dumps(summary, indent=2))

    @property
    def current_log_dir(self):
        if not self.logger:
            return "logs"
        return getattr(self.logger, "log_dir", self.logger.save_dir)

    def configure_optimizers(self):
        options = self.optimizer_options
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=options.learning_rate, weight_decay=self.cfg.weight_decay)

        scheduler_epochs = max(1, int(self.trainer.max_epochs) - options.warmup_epochs)
        eta_min_factor = min(1.0, max(1e-12, options.eta_min / max(options.learning_rate, 1e-12)))

        if options.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
        elif options.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_epochs,
                eta_min=options.eta_min,
            )
        elif options.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=eta_min_factor,
                total_iters=scheduler_epochs,
            )
        elif options.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=eta_min_factor ** (1.0 / scheduler_epochs),
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {options.scheduler!r}")

        if options.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.02, total_iters=options.warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[options.warmup_epochs])

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1000)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            },
            {"scheduler": plateau, "monitor": "train_loss_epoch", "interval": "epoch"},
        ]
