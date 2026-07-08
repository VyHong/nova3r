import time
from collections import defaultdict
from contextlib import contextmanager

import pytorch_lightning as pl
import torch


def cuda_synchronize_if_needed(device=None):
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_section(timings, name, device=None):
    cuda_synchronize_if_needed(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        cuda_synchronize_if_needed(device)
        timings[name] += time.perf_counter() - start


class BenchmarkTimingCallback(pl.Callback):
    def __init__(self, every_n_steps=10, warmup_steps=2, max_steps=0):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_steps = max(0, int(max_steps))
        self.reset()

    def reset(self):
        self.step_count = 0
        self.last_batch_end = None
        self.batch_start = None
        self.before_backward = None
        self.after_backward = None
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @staticmethod
    def _is_rank_zero(trainer):
        return getattr(trainer, "global_rank", 0) == 0

    @staticmethod
    def _sync(pl_module):
        cuda_synchronize_if_needed(getattr(pl_module, "device", None))

    def _record(self, name, value):
        if value < 0:
            return
        self.totals[name] += value
        self.counts[name] += 1

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync(pl_module)
        now = time.perf_counter()
        if self.last_batch_end is not None:
            self._record("benchmark/data_wait", now - self.last_batch_end)
        self.batch_start = now
        self.before_backward = None
        self.after_backward = None

    def on_before_backward(self, trainer, pl_module, loss):
        self._sync(pl_module)
        self.before_backward = time.perf_counter()

    def on_after_backward(self, trainer, pl_module):
        self._sync(pl_module)
        self.after_backward = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync(pl_module)
        now = time.perf_counter()
        self.last_batch_end = now
        if self.batch_start is None:
            return

        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            return

        total = now - self.batch_start
        self._record("benchmark/train_batch_total", total)
        if self.before_backward is not None:
            self._record("benchmark/forward_loss", self.before_backward - self.batch_start)
        if self.before_backward is not None and self.after_backward is not None:
            self._record("benchmark/backward", self.after_backward - self.before_backward)
            self._record("benchmark/optimizer_logging", now - self.after_backward)

        if self.step_count % self.every_n_steps == 0:
            self._emit(trainer, pl_module)

        if self.max_steps and self.step_count >= self.max_steps:
            trainer.should_stop = True

    def _emit(self, trainer, pl_module):
        averages = {
            name: self.totals[name] / max(1, self.counts[name])
            for name in sorted(self.totals)
        }
        for name, value in averages.items():
            pl_module.log(name, value, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)

        if not self._is_rank_zero(trainer):
            return

        total = averages.get("benchmark/train_batch_total")
        parts = []
        for name, value in averages.items():
            pct = ""
            if total and name != "benchmark/train_batch_total":
                pct = f" ({value / total * 100:.1f}%)"
            parts.append(f"{name.replace('benchmark/', '')}={value * 1000:.1f}ms{pct}")
        print(f"[benchmark] step={self.step_count} " + ", ".join(parts), flush=True)
