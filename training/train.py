import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
import argparse
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from training.nova3r.lightning_module import Nova3RLightningModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.nova3r.benchmark import BenchmarkTimingCallback
from lightning_data import Nova3RDataModule
from demo_nova3r import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--ckpt", default="checkpoints/da3_large/checkpoint-last.pth", help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--aggregator_ckpt", default="./checkpoints/da3_large/model.safetensors", help="Aggregator type (default: DepthAnything3Net)")
    parser.add_argument("--vae_ckpt", default="./checkpoints/hunyuan_mini/last_3d_front.ckpt", help="VAE checkpoint path")
    parser.add_argument("--wandb", default=False, action="store_true", help="Use Weights and Biases logger")
    parser.add_argument("--wandb_project", default="nova3r", help="WandB project name")
    parser.add_argument("--benchmark", default=False, action="store_true", help="Print CUDA-synchronized training timing summaries")
    parser.add_argument("--benchmark_inner", default=False, action="store_true", help="Time the main Nova3R forward/loss sub-sections")
    parser.add_argument("--benchmark_every_n_steps", type=int, default=10, help="How often to print benchmark averages")
    parser.add_argument("--benchmark_warmup_steps", type=int, default=2, help="Number of initial steps to skip in benchmark averages")
    parser.add_argument("--benchmark_steps", type=int, default=0, help="Stop after this many training steps when benchmarking; 0 disables")
    parser.add_argument(
        "--lightning_profiler",
        choices=["none", "simple", "advanced", "pytorch"],
        default="none",
        help="Optional PyTorch Lightning profiler to write under the logger directory",
    )
    args = parser.parse_args()

    return args


def build_profiler(name, output_dir):
    if name == "none":
        return None
    profiler_dir = os.path.join(output_dir, "profilers")
    os.makedirs(profiler_dir, exist_ok=True)
    if name == "simple":
        return SimpleProfiler(dirpath=profiler_dir, filename="simple_profiler")
    if name == "advanced":
        return AdvancedProfiler(dirpath=profiler_dir, filename="advanced_profiler")
    if name == "pytorch":
        return PyTorchProfiler(dirpath=profiler_dir, filename="pytorch_profiler")
    raise ValueError(f"Unknown profiler: {name}")


def update_cfg(cfg, key, value):
    try:
        OmegaConf.update(cfg, key, value, merge=True, force_add=True)
    except TypeError:
        setattr(cfg, key, value)


def load_data_config(ckpt_path):
    config_dir = os.path.join(os.path.dirname(ckpt_path), ".hydra")
    if os.path.exists(os.path.join(config_dir, "config.yaml")):
        cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
        return cfg.data
    else:
        raise FileNotFoundError(f"No .hydra/config.yaml found at {config_dir}. " f"Please ensure the checkpoint directory contains the Hydra config.")


def print_parameter_tree(module: nn.Module, name: str = "model", max_depth: int = 3, depth: int = 0):
    """Print parameter counts for a module and its children up to max_depth."""
    indent = "  " * depth
    own_params = sum(param.numel() for param in module.parameters(recurse=False))
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    print(f"{indent}{name}: own={own_params:,}, total={total_params:,} " f"({total_params / 1e6:.2f}M), trainable={trainable_params:,}")

    if depth >= max_depth:
        return

    for child_name, child_module in module.named_children():
        print_parameter_tree(child_module, f"{name}.{child_name}", max_depth=max_depth, depth=depth + 1)


def main():
    # torch.cuda.memory._record_memory_history()
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42, workers=True)
    args = parse_args()

    model, cfg = load_model(args.ckpt, args.device, aggregator_ckpt=args.aggregator_ckpt, vae_ckpt=args.vae_ckpt, strict=False)

    print_parameter_tree(model)

    data_cfg = load_data_config(args.ckpt)
    cfg.data = data_cfg
    update_cfg(cfg, "benchmark_inner", args.benchmark_inner)
    update_cfg(cfg, "benchmark_every_n_steps", args.benchmark_every_n_steps)

    datamodule = Nova3RDataModule(
        data_cfg,
    )
    module = eval(cfg.lightning_module)(cfg, model)

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
    callbacks = [checkpoint_callback, last_checkpoint, lr_monitor]
    if args.benchmark:
        callbacks.append(
            BenchmarkTimingCallback(
                every_n_steps=args.benchmark_every_n_steps,
                warmup_steps=args.benchmark_warmup_steps,
                max_steps=args.benchmark_steps,
            )
        )
    profiler = build_profiler(args.lightning_profiler, cfg.output_dir)

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
        max_steps=args.benchmark_steps if args.benchmark_steps > 0 else -1,
        profiler=profiler,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
