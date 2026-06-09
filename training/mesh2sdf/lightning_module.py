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

        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.learning_rate = cfg.lr
        self.model = model

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

    def forward(self, batch, criterion):
        surface = batch["surface"]
        # np.save("debug_points/surface_sample.npy", surface.cpu().numpy())
        # save_surface_to_ply(surface[0].cpu(), f"debug_points/surface_sample.ply")

        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, _ = self.model.encoder(pc, feats)
        moments = self.model.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        latents = posterior.sample()

        latents = self.model.decode(latents)

        geo_points = batch["geo_points"]
        geo_points_coords = geo_points[:, :, :3]
        geo_points_label = geo_points[:, :, 3:4]

        # test_scales = [64,80,90,96,100,110,128, 140 ]
        # scale_losses = []
        # logits = self.model.geo_decoder(queries=geo_points_coords, latents=latents)
        # kl_loss = posterior.kl(dims=(0, 1, 2))
        # for scale in test_scales:
        #     geo_points_label_test = geo_points_label * scale
        #     geo_points_label_test = geo_points_label_test.clamp(-1.0, 1.0)

        #     gt_list = {"sdf_target": geo_points_label_test}
        #     pred_list = {"sdf_pred": logits}

        #     loss, details = self.train_criterion(gt_list, pred_list)
        #     scale_losses.append(loss.item())

        # # --- Plotting the Results ---
        # plt.figure(figsize=(8, 5))
        # plt.plot(test_scales, scale_losses, marker='o', linestyle='-', color='#1f77b4', linewidth=2)

        # # Formatting the plot
        # plt.title('Loss vs. Scale', fontsize=14)
        # plt.xlabel('Scale', fontsize=12)
        # plt.ylabel('Loss', fontsize=12)

        # # Using a log scale for the X-axis is usually best since your scales double each time
        # plt.xscale('log', base=2)
        # plt.xticks(test_scales, test_scales) # Force x-ticks to display your exact scale values

        # plt.grid(True, which="both", linestyle="--", alpha=0.6)
        # plt.tight_layout()
        # plt.savefig("loss_vs_scale.png")
        # plt.close()

        geo_points_label = geo_points_label * 128
        geo_points_label = geo_points_label.clamp(-1.0, 1.0)

        logits = self.model.geo_decoder(queries=geo_points_coords, latents=latents)
        kl_loss = posterior.kl(dims=(0, 1, 2))

        gt_list = {"sdf_target": geo_points_label}
        pred_list = {"sdf_pred": logits}

        loss, details = self.train_criterion(gt_list, pred_list)

        loss += kl_loss * 0.00
        details["kl_loss"] = kl_loss.item()
        return loss, details, latents

    def training_step(self, batch, batch_idx):
        loss, details, _ = self.forward(batch, self.train_criterion)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end
        loss, details, _ = self.forward(batch, self.val_criterion)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if hasattr(self, "val_batch_to_log") and self.val_batch_to_log is not None:
            max_elements = 1
            for k, v in self.val_batch_to_log.items():
                if k in ["images", "intrinsics", "extrinsics"] and isinstance(v, list):
                    new_v = []
                    for item in v:
                        if isinstance(item, torch.Tensor):
                            new_v.append(item[:max_elements])
                        elif isinstance(item, list):
                            new_v.append([t[:max_elements] if isinstance(t, torch.Tensor) else t for t in item])
                        else:
                            new_v.append(item)
                    self.val_batch_to_log[k] = new_v
                elif isinstance(v, torch.Tensor) or isinstance(v, list):
                    self.val_batch_to_log[k] = v[:max_elements]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss, details, latents = self.forward(self.val_batch_to_log, self.val_criterion)
            # surface = self.val_batch_to_log["surface"]
            # latents = self.model.encode(surface)
            # latents = self.model.decode(latents)
            mesh = self.model.latents2mesh(
                latents, output_type="trimesh", bounds=1.01, mc_level=0.0, num_chunks=20000, octree_resolution=512, mc_algo="mc", enable_pbar=True
            )

        mesh = export_to_trimesh(mesh)[0]
        sample_dir = f"{self.current_log_dir}/val_points/epoch_{self.current_epoch}"
        glb_path = os.path.join(sample_dir, "val_mesh.ply")
        os.makedirs(sample_dir, exist_ok=True)
        mesh.export(glb_path)
        render_360_video(glb_path, sample_dir)

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

            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine], milestones=[self.cfg.warmup_epochs])
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
