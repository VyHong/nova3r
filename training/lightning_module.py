import pytorch_lightning as pl
import torch
import os
from training.training_utils import loss_of_one_batch_train
from training.validation_utils import generate_example
import tracemalloc
tracemalloc.start()

def check_leak():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 5 Memory Consumers ]")
    for stat in top_stats[:5]:
        print(stat)


class Nova3RLightningModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.learning_rate = cfg.lr
        self.model = model

        # Freeze decoder parameters
        for param in self.model.pts3d_head.parameters():
            param.requires_grad = True
        # Freeze patch embedding parameters
        # for param in self.model.vggt_aggregator.patch_embed.parameters():
        #     param.requires_grad = True
        self.criterion = None  # Criterion is integrated into the model's forward pass

    def move_batch_to_device(self, batch, device):
        """Move all tensors in the batch to the specified device."""

        keys = ["images","extrinsics", "intrinsics"]#, "cam_points", "world_points", "point_masks"] 
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

    def forward(self, batch):
        
        batch = self.move_batch_to_device(batch, self.device)
        output_dict = loss_of_one_batch_train(
            args=self.cfg,
            batch = batch,
            model=self.model,
            criterion=None,  # Criterion is integrated into the model's forward pass
            device=self.device,
        )

        return output_dict

    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        self.log("train_loss", output_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return output_dict["loss"]

    def on_validation_epoch_start(self):
        self.val_batch_to_log = None

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "val_batch_to_log") or self.val_batch_to_log is None:
            self.val_batch_to_log = batch  # Store the first batch for logging at epoch end
        output_dict = self.forward(batch)
        self.log("val_loss", output_dict["loss"], on_epoch=True, prog_bar=True)
        return output_dict["loss"]

    def on_validation_epoch_end(self):
        if hasattr(self, "val_batch_to_log") and self.val_batch_to_log is not None:
            generate_example(
                cfg=self.cfg,
                model=self.model,
                images=[t[0:1,...] for t in self.val_batch_to_log["images"]],
                num_queries=200000,
                log_dir=self.logger.log_dir if self.logger else "logs",
                device=self.device,
                current_epoch=self.current_epoch
            )
            self.val_batch_to_log = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
