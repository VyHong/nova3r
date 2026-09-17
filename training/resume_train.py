import os
import sys
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.distributed_env import configure, validate
from training.eval_freq import resolve_eval_freq

configure()

from demo_nova3r import load_model
from lightning_data import Nova3RDataModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.nova3r.lightning_module import Nova3RLightningModule
import torch
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--config", default="checkpoints/dinov3/config_dinov3_panorama_visible_latent_conditioned_cube.yaml", help="Model and training config")
    parser.add_argument("--device", default="gpu", help="Device (default: cuda)")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices; defaults to experiment.gpus")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    return args


def main():
    # torch.cuda.memory._record_memory_history()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(43, workers=True)
    args = parse_args()

    model, cfg = load_model(
        device="cpu",
        config_path=args.config,
        dtype=torch.float32,
        load_weights=False,
    )
    data_cfg = cfg.data

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = eval(cfg.lightning_module)(cfg, model)

    os.makedirs(cfg.output_dir, exist_ok=True)
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

    devices = args.devices if args.devices is not None else int(cfg.gpus)
    validate(torch, devices, args.device)
    check_val_every_n_epoch, val_check_interval = resolve_eval_freq(cfg.eval_freq)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=args.device,
        devices=devices,
        strategy=(
            DDPStrategy(
                timeout=timedelta(seconds=args.ddp_timeout_seconds),
                find_unused_parameters=True,
            )
            if devices > 1
            else "auto"
        ),
        logger=logger,
        callbacks=[checkpoint_callback, last_checkpoint, lr_monitor],
        precision=cfg.amp_dtype,
        gradient_clip_val=getattr(cfg, "grad_clip_norm", None),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accum_iter,
        log_every_n_steps=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )

    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=cfg.get("resume_checkpoint"),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
