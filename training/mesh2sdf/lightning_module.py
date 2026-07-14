import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from demo_nova3r import render_360_video
from nova3r.heads.hunyuan_model.autoencoders.model import DiagonalGaussianDistribution
from nova3r.losses import L21, FMVelocity, Pts3D_Regr3D_CD_V4, ReconstructionLoss, SDFReconstructionLoss, MSE
from eval.mv_recon.metric import SSI3DScore_Scene_Multi

from nova3r.heads.hunyuan_model.pipelines import export_to_trimesh
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

class PowerCosineAnnealingLR(LRScheduler):
    """
    Drop-in replacement for torch.optim.lr_scheduler.CosineAnnealingLR,
    but with a power parameter controlling decay aggressiveness.

    power = 1.0  -> normal cosine
    power > 1.0  -> faster decay early, stays lower longer
    power < 1.0  -> slower decay early
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0.0,
        power=1.0,
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_max <= 0:
            raise ValueError("T_max must be > 0")

        progress = min(self.last_epoch / self.T_max, 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = cosine ** self.power

        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]

class Mesh2SDFLightningModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.cfg = cfg

        self.learning_rate = cfg.lr
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

    @staticmethod
    def _median_normalization_target(mode):
        if mode == "median":
            return 1.0
        return float(mode.split("_")[-1])

    def _resolve_mesh_mc_level(self, bounds, octree_resolution):
        cfg_value = getattr(self.cfg, "mesh_mc_level", "auto")
        if not (isinstance(cfg_value, str) and cfg_value.lower() == "auto"):
            return float(cfg_value)

        bounds = np.asarray(bounds, dtype=np.float64)
        extent = np.max(bounds[3:] - bounds[:3])
        voxel_world_size = extent / max(int(octree_resolution) - 1, 1)
        return float(np.clip(voxel_world_size * 128.0 , 1e-4, 1.0))

    def normalize_surface_and_geo_points(self, batch, mode=None):
        if mode is None:
            mode = str(getattr(self.cfg, "surface_geo_normalize_mode", "none"))
        if mode in ("none", "raw", "skip"):
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
        batch = self.normalize_surface_and_geo_points(batch)
        loss, details, _ = self.model.lightning_forward(
            batch,
            self.train_criterion,
            kl_weight=self.kl_weight,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.normalize_surface_and_geo_points(batch)
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end
        reconstruction_loss, _, _ = self.model.lightning_forward(
            batch,
            self.val_criterion,
            kl_weight=0.0,
        )

        self.log(
            "val_loss",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return reconstruction_loss

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            return

        max_elements = max(1, int(getattr(self.cfg, "val_num_save", 1)))
        val_batch_to_log = self._slice_batch(self.val_batch_to_log, max_elements)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, recon_details, latents = self.model.lightning_forward(
                val_batch_to_log,
                self.val_criterion,
                kl_weight=0.0,
                recon_latents=True,
            )

            surface = val_batch_to_log["surface"]

            if hasattr(self.model, "decoder_head"):
                decoder = self.model.decoder_head
            elif hasattr(self.model, "first_stage"):
                decoder = self.model.first_stage
            else:
                decoder = self.model

            sample_dir = f"{self.current_log_dir}/val_points/epoch_{self.current_epoch}"
            os.makedirs(sample_dir, exist_ok=True)

            def mesh_from_latents(latent_batch, sample_idx, bounds):
                mesh_method = str(getattr(self.cfg, "mesh_method", "old")).lower()
                if mesh_method in {"old", "latent2mesh", "latents2mesh", "hunyuan"}:
                    octree_resolution = int(getattr(self.cfg, "mesh_octree_resolution", 512))
                    mc_level = self._resolve_mesh_mc_level(bounds, octree_resolution)
                    if isinstance(getattr(self.cfg, "mesh_mc_level", "auto"), str):
                        self.print(
                            f"Auto mesh_mc_level={mc_level:.6g} "
                            f"for bounds extent={float(np.max(bounds[3:] - bounds[:3])):.6g}, "
                            f"octree_resolution={octree_resolution}"
                        )
                    mesh_outputs = decoder.latents2mesh(
                        latent_batch[sample_idx : sample_idx + 1],
                        output_type="trimesh",
                        bounds=bounds,
                        mc_level=mc_level,
                        num_chunks=int(getattr(self.cfg, "mesh_num_chunks", 40000)),
                        octree_resolution=octree_resolution,
                        mc_algo=str(getattr(self.cfg, "mesh_mc_algo", "mc")),
                        enable_pbar=bool(getattr(self.cfg, "mesh_enable_pbar", True)),
                    )
                    meshes = export_to_trimesh(mesh_outputs)
                    return meshes[0] if isinstance(meshes, list) and meshes else meshes

                if mesh_method in {"udf", "mesh_udf", "new"}:
                    MC = int(getattr(self.cfg, "mesh_mc", 192))
                    smooth_borders = bool(getattr(self.cfg, "smooth_borders", False))
                    fourier = bool(getattr(self.cfg, "mesh_fourier", False))
                    with torch.inference_mode(False), torch.enable_grad():
                        latent_target = latent_batch[sample_idx : sample_idx + 1].float().detach().clone()
                        udf_decoder = UDFGeoDecoderAdapter(decoder.geo_decoder, latent_target, bounds)
                        _, _, mesh = get_mesh_udf_fast(
                            udf_decoder,
                            None,
                            N_MC=MC,
                            gradient=False,
                            smooth_borders=smooth_borders,
                            fourier=fourier,
                        )
                        if mesh is not None:
                            mesh = udf_decoder.mesh_to_world(mesh)
                    return mesh

                raise ValueError(f"Unknown mesh_method '{mesh_method}'. Use 'udf' or 'latents2mesh'.")

            meshes_to_render = []
            for sample_idx in range(latents.shape[0]):
                sample_surface = surface[sample_idx, :, :3]
                sample_min = sample_surface.min(dim=0).values.detach().cpu().numpy()
                sample_max = sample_surface.max(dim=0).values.detach().cpu().numpy()
                sample_padding = (sample_max - sample_min) * 0.1
                bounds = np.concatenate(
                    [sample_min - sample_padding, sample_max + sample_padding],
                    axis=0,
                )

                mesh_dir = os.path.join(sample_dir, f"{val_batch_to_log["seq_name"][sample_idx]}_{val_batch_to_log["subseq_ids"][sample_idx][0]}")
                os.makedirs(mesh_dir, exist_ok=True)
                try:
                    mesh = mesh_from_latents(latents, sample_idx, bounds)
                except RuntimeError as exc:
                    if "No surface found at the given iso value" not in str(exc):
                        raise
                    self.print(f"Skipping validation mesh for sample {sample_idx}: no surface found.")
                    mesh = None

                if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    glb_path = os.path.join(mesh_dir, "val_mesh.ply")
                    mesh.export(glb_path)
                    meshes_to_render.append((glb_path, mesh_dir))

        for glb_path, mesh_dir in meshes_to_render:
            render_360_video(glb_path, mesh_dir)

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
        if isinstance(self.logger, pl.loggers.WandbLogger):
            # For WandB, the actual run directory is available via experiment.dir
            return self.logger.experiment.dir if hasattr(self.logger.experiment, "dir") else self.logger.save_dir
        # For TensorBoard, log_dir gives the specific version_X folder
        return getattr(self.logger, "log_dir", self.logger.save_dir)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.cfg.weight_decay)

        scheduler_name = str(getattr(self.cfg, "lr_scheduler", "exponential")).lower().replace("-", "_")
        warmup_epochs = max(0, int(getattr(self.cfg, "warmup_epochs", 0)))
        scheduler_epochs = max(1, int(self.trainer.max_epochs) - warmup_epochs)
        eta_min = float(getattr(self.cfg, "lr_scheduler_eta_min", 1e-6))
        eta_min_factor = min(1.0, max(1e-12, eta_min / max(float(self.learning_rate), 1e-12)))

        if scheduler_name == "constant":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_epochs,
                eta_min=eta_min,
            )
        elif scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=eta_min_factor,
                total_iters=scheduler_epochs,
            )
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=eta_min_factor ** (1.0 / scheduler_epochs),
            )
        elif scheduler_name in {"power_cosine", "powercosine"}:
            scheduler = PowerCosineAnnealingLR(
                optimizer,
                T_max=scheduler_epochs,
                eta_min=eta_min,
                power=float(getattr(self.cfg, "lr_scheduler_power", 2.0)),
            )
        else:
            raise ValueError("lr_scheduler must be one of: constant, cosine, linear, exponential, power_cosine")

        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_epochs])

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            },
            {"scheduler": plateau, "monitor": "train_loss_epoch", "interval": "epoch"},
        ]


def save_surface_to_ply(surface, filename):
    """
    Saves a 7-dimensional surface tensor/array to a binary PLY file.

    Args:
        surface: numpy array or torch Tensor of shape (N, 7).
                 Columns: [x, y, z, nx, ny, nz, sharp_label]
        filename: str, path to save the .ply file (e.g., 'output.ply')
    """
    # 1. Convert torch tensor to numpy if necessary
    if isinstance(surface, torch.Tensor):
        surface = surface.detach().cpu().numpy()

    if surface.ndim != 2 or surface.shape[1] != 7:
        raise ValueError(f"Expected surface shape (N, 7), got {surface.shape}")

    num_points = surface.shape[0]

    # 2. Extract components
    # Using float32 for geometry/normals is standard
    xyz = surface[:, 0:3].astype(np.float32)
    normals = surface[:, 3:6].astype(np.float32)

    # Cast the sharp label to an unsigned 8-bit integer (0 or 1) to save file space
    sharp = surface[:, 6].astype(np.uint8)

    # 3. Create a structured array matching the exact byte layout of the PLY file
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),  # 'f4' = 32-bit float
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("sharp", "u1"),  # 'u1' = 8-bit unsigned integer
    ]

    vertex_data = np.empty(num_points, dtype=vertex_dtype)
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["nx"] = normals[:, 0]
    vertex_data["ny"] = normals[:, 1]
    vertex_data["nz"] = normals[:, 2]
    vertex_data["sharp"] = sharp

    # 4. Construct the standard PLY text header
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar sharp\n"
        "end_header\n"
    )

    # 5. Write everything to the file
    with open(filename, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(vertex_data.tobytes())


def save_sdf_to_ply(sdf_points, filename):
    """
    Saves SDF samples and labels to a binary PLY file.

    Args:
        sdf_points: numpy array or torch Tensor of shape (N, 4) or (N, 5).
                    Columns: [x, y, z, sdf] or [x, y, z, sdf, sample_label]
        filename: str, path to save the .ply file (e.g., 'output.ply')
    """
    if isinstance(sdf_points, torch.Tensor):
        sdf_points = sdf_points.detach().cpu().numpy()

    if sdf_points.ndim != 2 or sdf_points.shape[1] not in (4, 5):
        raise ValueError(f"Expected SDF shape (N, 4) or (N, 5), got {sdf_points.shape}")

    num_points = sdf_points.shape[0]
    has_sample_label = sdf_points.shape[1] == 5

    xyz = sdf_points[:, 0:3].astype(np.float32)
    sdf = sdf_points[:, 3].astype(np.float32)
    colors = np.empty((num_points, 3), dtype=np.uint8)
    colors[sdf > 0] = (220, 70, 70)
    colors[sdf < 0] = (70, 120, 220)
    colors[sdf == 0] = (180, 180, 180)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("sdf", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if has_sample_label:
        vertex_dtype.append(("sample_label", "u1"))

    vertex_data = np.empty(num_points, dtype=vertex_dtype)
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["sdf"] = sdf
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    if has_sample_label:
        vertex_data["sample_label"] = sdf_points[:, 4].astype(np.uint8)

    properties = (
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float sdf\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
    )
    if has_sample_label:
        properties += "property uchar sample_label\n"

    header = "ply\n" "format binary_little_endian 1.0\n" f"element vertex {num_points}\n" f"{properties}" "end_header\n"

    with open(filename, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(vertex_data.tobytes())
