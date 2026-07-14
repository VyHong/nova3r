import argparse
import os
import sys
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple NOVA3R learning-rate sweep.")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4,3e-4,3e-5,1e-5], help="Learning rates to try.")
    parser.add_argument("--ckpt", default="checkpoints/da3_base/checkpoint-last.pth", help="Path to model checkpoint.")
    parser.add_argument("--device", default="cuda", help="Trainer accelerator/device type.")
    parser.add_argument("--aggregator_ckpt", default="./checkpoints/da3_base/model.safetensors", help="DA3 aggregator checkpoint.")
    parser.add_argument("--vae_ckpt", default="./checkpoints/hunyuan_mini/last_no_norm.ckpt", help="ShapeVAE checkpoint.")
    parser.add_argument("--output_root", default="exp_output/lr_sweep", help="Root directory for sweep outputs.")
    parser.add_argument("--tag", default="da3_base", help="Tag added to logger names and output folders.")
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg.epochs for each run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used before each LR run.")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Print planned runs without training.")
    return parser.parse_args()


def lr_name(lr):
    return f"lr_{lr:.0e}".replace("-", "m").replace("+", "")


def load_data_config(ckpt_path):
    from omegaconf import OmegaConf

    config_dir = os.path.join(os.path.dirname(ckpt_path), ".hydra")
    config_path = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        return cfg.data
    raise FileNotFoundError(f"No .hydra/config.yaml found at {config_dir}.")


def print_parameter_tree(module, name="model", max_depth=3, depth=0):
    indent = "  " * depth
    own_params = sum(param.numel() for param in module.parameters(recurse=False))
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    print(f"{indent}{name}: own={own_params:,}, total={total_params:,} " f"({total_params / 1e6:.2f}M), trainable={trainable_params:,}")

    if depth >= max_depth:
        return

    for child_name, child_module in module.named_children():
        print_parameter_tree(child_module, f"{name}.{child_name}", max_depth=max_depth, depth=depth + 1)


def build_logger(args, cfg, run_name):
    from pytorch_lightning.loggers import TensorBoardLogger

    return TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training", version=run_name)


def run_one_lr(args, base_data_cfg, lr):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    from demo_nova3r import load_model
    from training.lightning_data import Nova3RDataModule
    from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
    from training.nova3r.lightning_module import Nova3RLightningModule

    lightning_modules = {
        "Mesh2SDFLightningModule": Mesh2SDFLightningModule,
        "Nova3RLightningModule": Nova3RLightningModule,
    }

    run_name = f"{args.tag}_{lr_name(lr)}"
    pl.seed_everything(args.seed, workers=True)

    model, cfg = load_model(args.ckpt, args.device, aggregator_ckpt=args.aggregator_ckpt, vae_ckpt=args.vae_ckpt, strict=False)#, stage="test")
    cfg.lr = lr
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.output_dir = os.path.join(args.output_root, run_name)
    cfg.data = deepcopy(base_data_cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"\n=== LR sweep run: {run_name} ===")
    print(f"lr={cfg.lr} epochs={cfg.epochs} output_dir={cfg.output_dir}")
    print_parameter_tree(model)

    datamodule = Nova3RDataModule(cfg.data)
    module_cls = lightning_modules.get(cfg.lightning_module)
    if module_cls is None:
        raise ValueError(f"Unknown lightning module: {cfg.lightning_module}")
    module = module_cls(cfg, model)

    logger = build_logger(args, cfg, run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss_epoch",
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            monitor="epoch",
            mode="max",
            save_top_k=1,
            filename="last",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=args.device,
        devices=cfg.gpus,
        logger=logger,
        callbacks=callbacks,
        precision=cfg.amp_dtype,
        gradient_clip_val=getattr(cfg, "grad_clip_norm", None),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accum_iter,
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.eval_freq,
    )
    trainer.fit(module, datamodule=datamodule)


def main():
    import torch

    torch.set_float32_matmul_precision("medium")
    args = parse_args()
    base_data_cfg = None if args.dry_run else load_data_config(args.ckpt)

    for lr in args.lrs:
        run_name = f"{args.tag}_{lr_name(lr)}"
        output_dir = os.path.join(args.output_root, run_name)
        if args.dry_run:
            print(f"Would run lr={lr} output_dir={output_dir}")
            continue
        run_one_lr(args, base_data_cfg, lr)


if __name__ == "__main__":
    main()
