import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
from omegaconf import OmegaConf
import torch
from lightning_module import Nova3RLightningModule
from lightning_data import Nova3RDataModule
from demo_nova3r import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--ckpt", default="checkpoints/hunyuan_dims/12_layers.ckpt", help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--aggregator_ckpt", default="./checkpoints/da3_giant/model.safetensors", help="Aggregator type (default: DepthAnything3Net)")
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
    args = parse_args()

    model, cfg = load_model(args.ckpt, args.device, aggregator_ckpt=args.aggregator_ckpt, stage="test")
    data_cfg = load_data_config(args.ckpt)
    cfg.data = data_cfg

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = Nova3RLightningModule(cfg, model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training")

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=cfg.gpus,
        logger=logger,
        precision=32,
    )
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
