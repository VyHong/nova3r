import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_nova3r import load_model
from lightning_data import Nova3RDataModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.nova3r.lightning_module import Nova3RLightningModule
import torch
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--config", default="checkpoints/hunyuan/.hydra/config.yaml", help="Model and training config")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices; defaults to experiment.gpus")
    args = parser.parse_args()

    return args


def main():
    # torch.cuda.memory._record_memory_history()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(43, workers=True)
    args = parse_args()

    model, cfg = load_model(
        device=args.device,
        config_path=args.config,
        strict=False,
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
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=args.device,
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback, last_checkpoint, lr_monitor],
        precision=cfg.amp_dtype,
        gradient_clip_val=getattr(cfg, "grad_clip_norm", None),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accum_iter,
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.eval_freq,
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
