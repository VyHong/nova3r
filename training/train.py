import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
from omegaconf import OmegaConf
import torch
from lightning_module import Nova3RLightningModule
from lightning_data import Nova3RDataModule
from demo_nova3r import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--ckpt", default="checkpoints/hunyuan_mini/model.fp16.ckpt", help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--aggregator_ckpt", default="./checkpoints/da3/model.safetensors", help="Aggregator type (default: DepthAnything3Net)")
    parser.add_argument("--wandb", default=False, action="store_true", help="Use Weights and Biases logger")
    parser.add_argument("--wandb_project", default="nova3r", help="WandB project name")
    args = parser.parse_args()

    return args


def load_data_config(ckpt_path):
    config_dir = os.path.join(os.path.dirname(ckpt_path), ".hydra")
    if os.path.exists(os.path.join(config_dir, "config.yaml")):
        cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
        return cfg.data
    else:
        raise FileNotFoundError(f"No .hydra/config.yaml found at {config_dir}. " f"Please ensure the checkpoint directory contains the Hydra config.")


def main():
    # torch.cuda.memory._record_memory_history()
    torch.set_float32_matmul_precision("medium")
    args = parse_args()

    model, cfg = load_model(args.ckpt, args.device, aggregator_ckpt=args.aggregator_ckpt, strict=False)
    data_cfg = load_data_config(args.ckpt)
    cfg.data = data_cfg

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = Nova3RLightningModule(cfg, model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    if args.wandb:
        logger = WandbLogger(project=args.wandb_project, name="nova3r_training", save_dir=cfg.output_dir)
    else:
        logger = TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.output_dir, "checkpoints"),
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss_epoch",
        mode="min",
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(cfg.output_dir, "checkpoints"),
        monitor="epoch",
        mode="max",
        save_top_k=1,
        filename="last",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=args.device,
        devices=cfg.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, last_checkpoint, lr_monitor],
        precision=cfg.amp_dtype,
        log_every_n_steps=1,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
