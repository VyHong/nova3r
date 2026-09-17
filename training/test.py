import os
import sys
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.distributed_env import configure, validate

CONFIGURED_DEVICES = configure()

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import argparse
import torch
from training.nova3r.lightning_module import Nova3RLightningModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from lightning_data import Nova3RDataModule
from demo_nova3r import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--config", default="checkpoints/da3_large_hunyuan_mini/.hydra/config_agg.yaml", help="Model and evaluation config")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--devices", type=int, default=CONFIGURED_DEVICES, help="Number of GPUs; defaults to experiment.gpus")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    return args


def main():

    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    pl.seed_everything(42, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device in {"cuda", "gpu"}:
        torch.cuda.set_device(local_rank)
        model_device = f"cuda:{local_rank}"
    else:
        model_device = args.device
    model, cfg = load_model(device=model_device, config_path=args.config, stage="test")
    data_cfg = cfg.data
    devices = args.devices if args.devices is not None else int(cfg.gpus)
    validate(torch, devices, args.device)

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = eval(cfg.lightning_module)(cfg, model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training")

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=devices,
        strategy=(DDPStrategy(timeout=timedelta(seconds=args.ddp_timeout_seconds)) if devices > 1 else "auto"),
        logger=logger,
        precision=cfg.amp_dtype,
    )
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
