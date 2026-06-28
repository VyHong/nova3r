import pytorch_lightning as pl
import torch
from demo_nova3r import render_360_video
from nova3r.heads.hunyuan_model.autoencoders.model import DiagonalGaussianDistribution
from nova3r.losses import L21, FMVelocity, Pts3D_Regr3D_CD_V4, ReconstructionLoss, SDFReconstructionLoss, MSE
from eval.mv_recon.metric import SSI3DScore_Scene_Multi

from nova3r.heads.hunyuan_model.pipelines import export_to_trimesh
import os
import numpy as np
import matplotlib.pyplot as plt


class Mesh2SDFLightningModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.cfg = cfg

        self.learning_rate = cfg.lr
        self.model = model
        self.kl_weight = cfg.kl_weight

        layers_to_freeze = getattr(cfg, "layers_to_freeze", [])

        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                print(f"Froze: {name}")
            else:
                # print(f"Kept active: {name}")
                pass

        self.train_criterion = SDFReconstructionLoss(MSE)
        self.val_criterion = SDFReconstructionLoss(MSE)
        self.test_criterion = None

    # def on_after_backward(self):
    #     if self.global_step % 50 != 0:
    #         return

    #     groups = {
    #         "da3": "da3_aggregator",
    #         "img_proj": "img_token_proj",
    #         "fm_head": "pts3d_head",
    #         "first_stage": "first_stage",
    #     }

    #     for label, key in groups.items():
    #         norms = []
    #         for name, p in self.model.named_parameters():
    #             if key in name and p.grad is not None:
    #                 norms.append(p.grad.detach().float().norm().item())
    #         if norms:
    #             print(f"grad/{label}: mean={sum(norms)/len(norms):.3e}, max={max(norms):.3e}, n={len(norms)}")
    #         else:
    #             print(f"grad/{label}: NONE")

    def training_step(self, batch, batch_idx):
        # loss, details, _ = self.model.lightning_forward(batch, self.train_criterion, kl_weight=self.kl_weight)
        loss, details, _ = self.model.lightning_forward(
            batch,
            self.train_criterion,
            kl_weight=self.kl_weight,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end
        # reconstruction_loss, _, _ = self.model.lightning_forward(batch, self.val_criterion, kl_weight=0.0)
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
            # _, _, latents = self.model.lightning_forward(self.val_batch_to_log, self.val_criterion, kl_weight=0.0)
            _, _, latents = self.model.lightning_forward(
                val_batch_to_log,
                self.val_criterion,
                kl_weight=0.0,
                recon_latents=True,
            )

            surface = val_batch_to_log["surface"]

            # save_sdf_to_ply(self.val_batch_to_log["geo_points"][0], filename="debug_points/sdf_hole.ply")
            mins = surface[:, :, :3].min(dim=1).values.detach().cpu().numpy()
            maxs = surface[:, :, :3].max(dim=1).values.detach().cpu().numpy()
            box_size = maxs - mins
            padding = box_size * 0.2

            if hasattr(self.model, "decoder_head"):
                decoder = self.model.decoder_head
            elif hasattr(self.model, "first_stage"):
                decoder = self.model.first_stage
            else:
                decoder = self.model

            sample_dir = f"{self.current_log_dir}/val_points/epoch_{self.current_epoch}"
            os.makedirs(sample_dir, exist_ok=True)

            meshes_to_render = []
            for sample_idx in range(latents.shape[0]):
                bounds = np.concatenate(
                    [mins[sample_idx] - padding[sample_idx], maxs[sample_idx] + padding[sample_idx]],
                    axis=0,
                )
                mesh = decoder.latents2mesh(
                    latents[sample_idx : sample_idx + 1],
                    output_type="trimesh",
                    bounds=bounds,
                    mc_level=0.0,
                    num_chunks=20000,
                    octree_resolution=512,
                    mc_algo="mc",
                    enable_pbar=True,
                )
                mesh = export_to_trimesh(mesh)[0]
                if mesh is not None:
                    mesh_dir = os.path.join(sample_dir, f"{val_batch_to_log["seq_name"][sample_idx]}_{sample_idx:03d}")
                    glb_path = os.path.join(mesh_dir, "val_mesh.ply")
                    os.makedirs(mesh_dir, exist_ok=True)
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
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        exponential = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

        if self.cfg.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=self.cfg.warmup_epochs)

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine], milestones=[self.cfg.warmup_epochs]
            )
        else:
            scheduler = exponential
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
