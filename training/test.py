import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import torch
from training.nova3r.lightning_module import Nova3RLightningModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from lightning_data import Nova3RDataModule
from demo_nova3r import load_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--config", default="checkpoints/hunyuan/.hydra/config.yaml", help="Model and evaluation config")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    return args


def main():

    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    pl.seed_everything(42, workers=True)

    model, cfg = load_model(device=args.device, config_path=args.config, stage="test")
    data_cfg = cfg.data

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = eval(cfg.lightning_module)(cfg, model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training")

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=cfg.gpus,
        logger=logger,
        precision=cfg.amp_dtype,
    )
    trainer.validate(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
