import os
import sys
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keep the repository ahead of site-packages even when PROJECT_ROOT was
# already added through PYTHONPATH.  Otherwise an installed, incompatible
# depth_anything_3 package can shadow the local implementation.
try:
    sys.path.remove(PROJECT_ROOT)
except ValueError:
    pass
sys.path.insert(0, PROJECT_ROOT)

from training.distributed_env import configure, validate
from training.eval_freq import resolve_eval_freq

configure()

from demo_nova3r import load_model
from lightning_data import Nova3RDataModule
from training.mesh2sdf.lightning_module import Mesh2SDFLightningModule
from training.nova3r.lightning_module import Nova3RLightningModule
import torch.nn as nn
import torch
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl


class BenchmarkProfiler(SimpleProfiler):
    """Collect timings without printing Lightning's full profiler table."""

    def describe(self):
        pass


class BenchmarkReporter(pl.Callback):
    """Print a compact timing summary every few optimizer steps."""

    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = every_n_steps
        self._last_reported_step = 0
        self._seen_durations = {}

    def _new_durations(self, profiler, predicate):
        values = []
        for action, durations in profiler.recorded_durations.items():
            if not predicate(action):
                continue
            start = self._seen_durations.get(action, 0)
            values.extend(durations[start:])
            self._seen_durations[action] = len(durations)
        return values

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _print_timing(name, duration, total):
        percentage = 100 * duration / total if total else 0.0
        print(f"  {name:<20} {duration * 1000:>8.1f} ms  ({percentage:>5.1f}%)")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if trainer.is_global_zero and step > 0 and step % self.every_n_steps == 0 and step != self._last_reported_step:
            profiler = trainer.profiler
            dataloader = self._mean(
                self._new_durations(
                    profiler,
                    lambda name: name.endswith(".train_dataloader_next"),
                )
            )
            training_step = self._mean(
                self._new_durations(
                    profiler,
                    lambda name: name.endswith(".training_step"),
                )
            )
            backward = self._mean(
                self._new_durations(
                    profiler,
                    lambda name: name.endswith(".backward"),
                )
            )
            compute = self._mean(
                self._new_durations(
                    profiler,
                    lambda name: name.startswith("[LightningModule]") and name.endswith(".optimizer_step"),
                )
            )
            progress = self._mean(self._new_durations(profiler, lambda name: "RichProgressBar.on_train_batch_end" in name))
            optimizer_and_other = max(0.0, compute - training_step - backward)
            total = dataloader + compute + progress

            first_step = self._last_reported_step + 1
            print(f"\nBenchmark steps {first_step}-{step} (mean per step):")
            self._print_timing("data loading", dataloader, total)
            self._print_timing("forward + loss", training_step, total)
            self._print_timing("backward", backward, total)
            self._print_timing("optimizer + other", optimizer_and_other, total)
            self._print_timing("progress output", progress, total)
            print(f"  {'total':<20} {total * 1000:>8.1f} ms")
            self._last_reported_step = step


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--config", default="checkpoints/dinov3/config_dinov3_panorama_visible_latent_conditioned_cube.yaml", help="Model and training config")
    parser.add_argument("--device", default="gpu", help="Device (default: cuda)")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices; defaults to experiment.gpus")
    parser.add_argument("--ddp-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Profile dataloading, training, backward, and optimizer time",
    )
    args = parser.parse_args()

    return args


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

    model, cfg = load_model(
        device="cpu",
        config_path=args.config,
        dtype=torch.float32,
        strict=False,
    )

    print_parameter_tree(model)

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
    callbacks = [checkpoint_callback, last_checkpoint, lr_monitor]
    if args.benchmark:
        callbacks.append(BenchmarkReporter(every_n_steps=10))

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
        callbacks=callbacks,
        precision=cfg.amp_dtype,
        gradient_clip_val=getattr(cfg, "grad_clip_norm", None),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.accum_iter,
        profiler=BenchmarkProfiler() if args.benchmark else None,
        log_every_n_steps=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
