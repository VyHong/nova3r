import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from demo_nova3r import render_360_video
from nova3r.heads.hunyuan_model.autoencoders.model import DiagonalGaussianDistribution
from nova3r.losses import L21, FMVelocity, Pts3D_Regr3D_CD_V4, ReconstructionLoss, SDFReconstructionLoss, MSE
from eval.mv_recon.metric import SSI3DScore_Scene_Multi

from nova3r.heads.hunyuan_model.pipelines import export_to_trimesh
from nova3r.utils.batch_debug import save_batch_debug
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import math
from torch.optim.lr_scheduler import LRScheduler

from training.mesh2sdf.tools.mesh_udf.mesh_udf import get_mesh_udf_fast


class UDFGeoDecoderAdapter(nn.Module):
    def __init__(self, geo_decoder, latent, bounds, udf_scale=128.0):
        super().__init__()
        self.geo_decoder = geo_decoder
        self.latent = latent
        self.udf_scale = float(udf_scale)
        bounds = torch.as_tensor(bounds, dtype=latent.dtype, device=latent.device)
        mins, maxs = bounds[:3], bounds[3:]
        self.center = (mins + maxs) * 0.5
        self.half_extent = ((maxs - mins).max() * 0.5).clamp_min(1e-6)

    def forward(self, decoder_input):
        unit_queries = decoder_input[:, -3:].view(1, -1, 3).to(device=self.latent.device, dtype=self.latent.dtype)
        queries = unit_queries * self.half_extent + self.center
        logits = self.geo_decoder(queries=queries, latents=self.latent)
        # Training predicts UDF distances scaled by 128; MC expects distances in unit-grid coordinates.
        return (F.softplus(logits) / (self.udf_scale * self.half_extent)).view(-1, 1)

    def mesh_to_world(self, mesh):
        mesh.vertices = mesh.vertices * float(self.half_extent.detach().cpu()) + self.center.detach().cpu().numpy()
        return mesh


class Mesh2SDFLightningModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.cfg = cfg

        self.learning_rate = cfg.lr
        self.lr_scheduler_name = str(getattr(cfg, "lr_scheduler", "exponential"))
        self.warmup_epochs = int(getattr(cfg, "warmup_epochs", 0))
        self.lr_scheduler_eta_min = float(getattr(cfg, "lr_scheduler_eta_min", 1e-6))

        self.model = model
        self.kl_weight = cfg.kl_weight

        layers_to_freeze = getattr(cfg, "layers_to_freeze", [])
        frozen_layers_logged = set()
        for name, param in model.named_parameters():
            matched_layer = next((layer for layer in layers_to_freeze if layer in name), None)
            if matched_layer is not None:
                param.requires_grad = False
                if matched_layer not in frozen_layers_logged:
                    print(f"Froze: {matched_layer}")
                    frozen_layers_logged.add(matched_layer)
            else:
                # print(f"Kept active: {name}")
                pass

        self.train_criterion = SDFReconstructionLoss(MSE)
        self.val_criterion = SDFReconstructionLoss(MSE)
        self.test_criterion = None

        self.val_num_save = int(getattr(cfg, "val_num_save", 1))

        self.surface_geo_normalize_mode = str(getattr(cfg, "surface_geo_normalize_mode", "median"))

        self.mesh_mc_level = getattr(cfg, "mesh_mc_level", "auto")
        self.mesh_method = str(getattr(cfg, "mesh_method", "latent2mesh"))
        self.octree_resolution = int(getattr(cfg, "mesh_octree_resolution", 512))
        self.mesh_mc_algo = str(getattr(cfg, "mesh_mc_algo", "mc")).lower()
        self.mesh_num_chunks = int(getattr(cfg, "mesh_num_chunks", 40000))

        self.mesh_mc = int(getattr(cfg, "mesh_mc", 512))
        self.mesh_smooth_borders = bool(getattr(cfg, "smooth_borders", False))
        self.mesh_fourier = bool(getattr(cfg, "mesh_fourier", False))
        self.mesh_udf_vertex_batch_size = int(getattr(cfg, "mesh_udf_vertex_batch_size", 65536))

        self.mesh_udf_surface_threshold = getattr(cfg, "mesh_udf_surface_threshold", 0.002)
        if self.mesh_udf_surface_threshold is not None:
            self.mesh_udf_surface_threshold = float(self.mesh_udf_surface_threshold)

    @staticmethod
    def _median_normalization_target(mode):
        if mode == "median":
            return 1.0
        return float(mode.split("_")[-1])

    def _resolve_mesh_mc_level(self, bounds, octree_resolution):
        cfg_value = self.mesh_mc_level
        if not (isinstance(cfg_value, str) and cfg_value.lower() == "auto"):
            return float(cfg_value)

        bounds = np.asarray(bounds, dtype=np.float64)
        extent = np.max(bounds[3:] - bounds[:3])
        voxel_world_size = extent / max(int(octree_resolution) - 1, 1)
        return float(np.clip(voxel_world_size * 128.0, 1e-4, 1.0))

    def normalize_surface_and_geo_points(self, batch, mode):
        "median normalization of surface and geo_points to a target median distance from origin"
        if self.surface_geo_normalize_mode == "none":
            return batch

        target_median = self._median_normalization_target(mode)

        surface = batch["surface"]
        geo_points = batch["geo_points"]

        distances = torch.linalg.norm(surface[..., :3], dim=-1)
        norm_factor = torch.median(distances, dim=1).values
        norm_factor = torch.clamp(norm_factor, min=0.01, max=100.0)
        coord_scale = (surface.new_tensor(float(target_median)) / norm_factor).view(-1, 1, 1)

        normalized_surface = surface.clone()
        normalized_surface[..., :3] = torch.clamp(
            normalized_surface[..., :3] * coord_scale,
            min=-1000.0,
            max=1000.0,
        )

        normalized_geo_points = geo_points.clone()
        normalized_geo_points[..., :3] = torch.clamp(
            normalized_geo_points[..., :3] * coord_scale,
            min=-1000.0,
            max=1000.0,
        )
        normalized_geo_points[..., 3:4] = normalized_geo_points[..., 3:4] * coord_scale

        normalized_batch = dict(batch)
        normalized_batch["surface"] = normalized_surface
        normalized_batch["geo_points"] = normalized_geo_points
        normalized_batch["surface_geo_norm_factor"] = norm_factor
        return normalized_batch

    def training_step(self, batch, batch_idx):
        batch = self.normalize_surface_and_geo_points(batch, mode=self.surface_geo_normalize_mode)
        loss, details, _ = self.model.lightning_forward(
            batch,
            self.train_criterion,
            kl_weight=self.kl_weight,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.normalize_surface_and_geo_points(batch, mode=self.surface_geo_normalize_mode)
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end

        reconstruction_loss, details, _ = self.model.lightning_forward(
            batch,
            self.val_criterion,
            kl_weight=0.0,
            deterministic_seed=42 + batch_idx,
        )

        self.log("val_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in details.items():
            self.log(
                f"val_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        return reconstruction_loss

    def on_validation_epoch_end(self):

        val_batch_to_log = self._slice_batch(self.val_batch_to_log, self.val_num_save)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, recon_details, latents = self.model.lightning_forward(
                val_batch_to_log,
                self.val_criterion,
                kl_weight=0.0,
                recon_latents=True,
                deterministic_seed=42,
            )
            surface = val_batch_to_log["surface"]

            if hasattr(self.model, "first_stage"):
                decoder = self.model.first_stage
            else:
                decoder = self.model

            sample_dir = f"{self.current_log_dir}/val_points/epoch_{self.current_epoch}"
            os.makedirs(sample_dir, exist_ok=True)

            # save_batch_debug(
            #     val_batch_to_log,
            #     sample_dir,
            #     max_samples=self.val_num_save,
            #     batched=True,
            # )

            meshes_to_render = []
            for sample_idx in range(latents.shape[0]):
                sample_surface = surface[sample_idx, :, :3]
                sample_min = sample_surface.min(dim=0).values.detach().cpu().numpy()
                sample_max = sample_surface.max(dim=0).values.detach().cpu().numpy()
                sample_padding = (sample_max - sample_min) * 0.1
                bounds = np.concatenate([sample_min - sample_padding, sample_max + sample_padding], axis=0)

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

        for glb_path, mesh_dir in meshes_to_render:
            render_360_video(glb_path, mesh_dir)

    def mesh_from_latents(self, latent_batch, sample_idx, bounds, decoder):
        if self.mesh_method == "latent2mesh":
            mc_level = self._resolve_mesh_mc_level(bounds, self.octree_resolution)

            if isinstance(self.mesh_mc_level, str):
                self.print(f"Auto mesh_mc_level={mc_level:.6g} " f"for bounds extent={float(np.max(bounds[3:] - bounds[:3])):.6g}, ")

            mesh_outputs = decoder.latents2mesh(
                latent_batch[sample_idx : sample_idx + 1],
                output_type="trimesh",
                bounds=bounds,
                mc_level=mc_level,
                num_chunks=self.mesh_num_chunks,
                octree_resolution=self.octree_resolution,
                mc_algo=self.mesh_mc_algo,
            )
            meshes = export_to_trimesh(mesh_outputs)
            return meshes[0] if isinstance(meshes, list) and meshes else meshes

        if self.mesh_method == "mesh_udf":
            with torch.inference_mode(False), torch.enable_grad():
                latent_target = latent_batch[sample_idx : sample_idx + 1].float().detach().clone()
                udf_decoder = UDFGeoDecoderAdapter(decoder.geo_decoder, latent_target, bounds)
                _, _, mesh = get_mesh_udf_fast(
                    udf_decoder,
                    None,
                    N_MC=self.mesh_mc,
                    gradient=False,
                    smooth_borders=self.mesh_smooth_borders,
                    fourier=self.mesh_fourier,
                    vertex_max_batch=self.mesh_udf_vertex_batch_size,
                    surface_threshold=self.mesh_udf_surface_threshold,
                )
                if mesh is not None:
                    mesh = udf_decoder.mesh_to_world(mesh)
            return mesh

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

    def test_step(self, batch, batch_idx):
        pass

    @property
    def current_log_dir(self):
        if not self.logger:
            return "logs"
        return getattr(self.logger, "log_dir", self.logger.save_dir)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.cfg.weight_decay)

        scheduler_epochs = max(1, int(self.trainer.max_epochs) - self.warmup_epochs)
        eta_min_factor = min(1.0, max(1e-12, self.lr_scheduler_eta_min / max(float(self.learning_rate), 1e-12)))

        if self.lr_scheduler_name == "constant":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
        elif self.lr_scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_epochs,
                eta_min=self.lr_scheduler_eta_min,
            )
        elif self.lr_scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=eta_min_factor,
                total_iters=scheduler_epochs,
            )
        elif self.lr_scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=eta_min_factor ** (1.0 / scheduler_epochs),
            )

        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.02, total_iters=self.warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[self.warmup_epochs])

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            },
            {"scheduler": plateau, "monitor": "train_loss_epoch", "interval": "epoch"},
        ]
