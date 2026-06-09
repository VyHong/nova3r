import pytorch_lightning as pl
import torch
from training.nova3r.training_utils import get_all_pts3d, loss_of_one_batch_train, normalize_input, save_points_ply
from training.nova3r.validation_utils import (
    generate_example,
    generate_pointcloud,
    run_test_score,
    run_validation_loss,
    save_test_scores_to_csv,
    normalize_pointclouds,
)
from nova3r.losses import L21, FMVelocity, Pts3D_Regr3D_CD_V4, ReconstructionLoss
from eval.mv_recon.metric import SSI3DScore_Scene_Multi


class Nova3RLightningModule(pl.LightningModule):
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

        self.train_criterion = FMVelocity(L21) + 0.1 * ReconstructionLoss(L21)
        self.val_criterion = Pts3D_Regr3D_CD_V4(L21)
        self.test_criterion = SSI3DScore_Scene_Multi(
            num_eval_pts=16384,
            fs_thres=[0.1, 0.05, 0.02],
            pts_sampling_mode="uniform",
            alignment="none",
            use_cd_align=True,
        ).to(self.device)

    def forward(self, batch, criterion):
        loss, details = loss_of_one_batch_train(
            args=self.cfg,
            batch=batch,
            model=self.model,
            criterion=criterion,  # Criterion is integrated into the model's forward pass
            device=self.device,
        )
        return loss, details

    def training_step(self, batch, batch_idx):
        loss, details = self.forward(batch, self.train_criterion)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pts3d_src_norm, valid_src, pts3d_trg_norm, valid_trg, normals_src, normals_trg = normalize_pointclouds(self.cfg, batch)
        pts3d_trg_norm = pts3d_trg_norm.to(dtype=torch.float32)

        gt_pts3d = pts3d_trg_norm
        gt_valid = valid_trg
        batch["pts3d_src_norm"] = pts3d_src_norm
        batch["valid_src"] = valid_src
        batch["pts3d_trg_norm"] = pts3d_trg_norm
        batch["valid_trg"] = valid_trg
        batch["normals_src"] = normals_src
        batch["normals_trg"] = normals_trg
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end

        pts3d = generate_pointcloud(
            cfg=self.cfg,
            model=self.model,
            batch=batch,
            num_queries=gt_pts3d.shape[1],
            device=self.device,
        )

        loss, details = run_validation_loss(gt_pts3d=gt_pts3d, gt_valid=gt_valid, pts3d=pts3d, criterion=self.val_criterion, device=self.device)

        self.log("val_loss", details["Pts3D_Regr3D_CD_V4_completeness"], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @property
    def current_log_dir(self):
        if not self.logger:
            return "logs"
        if isinstance(self.logger, pl.loggers.WandbLogger):
            # For WandB, the actual run directory is available via experiment.dir
            return self.logger.experiment.dir if hasattr(self.logger.experiment, "dir") else self.logger.save_dir
        # For TensorBoard, log_dir gives the specific version_X folder
        return getattr(self.logger, "log_dir", self.logger.save_dir)

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

            pts3d = generate_pointcloud(
                cfg=self.cfg,
                model=self.model,
                batch=self.val_batch_to_log,
                num_queries=self.cfg.decoder_sample_size,
                device=self.device,
                stage="val",
            )

            generate_example(
                batch=self.val_batch_to_log,
                pts3d=pts3d,
                log_dir=self.current_log_dir,
                current_epoch=self.current_epoch,
            )
            self.val_batch_to_log = None

    def test_step(self, batch, batch_idx):
        pts3d_src_norm, valid_src, pts3d_trg_norm, valid_trg, normals_src, normals_trg = normalize_pointclouds(self.cfg, batch)

        batch["pts3d_src_norm"] = pts3d_src_norm
        batch["valid_src"] = valid_src
        batch["pts3d_trg_norm"] = pts3d_trg_norm
        batch["valid_trg"] = valid_trg
        batch["normals_src"] = normals_src
        batch["normals_trg"] = normals_trg
        pts3d = generate_pointcloud(
            cfg=self.cfg, model=self.model, batch=batch, num_queries=self.cfg.decoder_sample_size, device=self.device, stage="test"
        )
        # save_points_ply(batch["cam_points"],"./debug_points/gt.ply")
        # save_points_ply(pts3d,"./debug_points/pred.ply")

        log_dir = self.logger.log_dir if self.logger else None

        if not hasattr(self, "test_results"):
            self.test_results = []

        data, details = run_test_score(batch, pts3d, self.test_criterion, self.device)
        res = save_test_scores_to_csv(batch, details, log_dir)
        self.test_results.append(res)

        if self.cfg.save_test_examples:
            generate_example(
                batch=batch,
                pts3d=pts3d,
                log_dir=self.current_log_dir,
                current_epoch=self.current_epoch,
            )

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
