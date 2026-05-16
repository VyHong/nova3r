import pytorch_lightning as pl
import torch
from training.training_utils import loss_of_one_batch_train
from training.validation_utils import generate_example
from nova3r.losses import L21, FMVelocity, Pts3D_Regr3D_CD_V4
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
                print(f"Kept active: {name}")

        self.train_criterion = FMVelocity(L21)
        self.test_criterion = None  # Criterion is integrated into the model's forward pass

    def move_batch_to_device(self, batch, device):
        """Move all tensors in the batch to the specified device."""
        keys = ["images", "extrinsics", "intrinsics"]  # , "cam_points", "world_points", "point_masks"]
        for key, value in batch.items():
            if key not in keys:
                continue
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                batch[key] = self.move_batch_to_device(value, device)
            elif isinstance(value, list):
                batch[key] = [self.move_batch_to_device(item, device) if isinstance(item, dict) else item.to(device) for item in value]
        return batch

    def forward(self, batch,criterion):
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
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end

        loss, details = self.forward(batch, self.train_criterion)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if hasattr(self, "val_batch_to_log") and self.val_batch_to_log is not None:
            generate_example(
                cfg=self.cfg,
                model=self.model,
                images=[t[0:1, ...] for t in self.val_batch_to_log["images"]],
                num_queries=200000,
                log_dir=self.logger.log_dir if self.logger else "logs",
                device=self.device,
                current_epoch=self.current_epoch,
            )
            self.val_batch_to_log = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate,weight_decay=self.cfg.weight_decay)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=self.cfg.warmup_epochs)
        # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.cfg.warmup_epochs])
        
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            },
            # {"scheduler": cosine, "interval": "epoch"},
            {"scheduler": plateau, "monitor": "train_loss_epoch", "interval": "epoch"},
        ]
