"""Convert epoch-based configuration to optimizer-step schedule durations."""

import math


def scheduler_step_counts(trainer, warmup_epochs):
    # Accessing this property sets up the train loader in configure_optimizers.
    estimated_steps = trainer.estimated_stepping_batches
    batches = trainer.num_training_batches
    if not math.isfinite(batches) or not math.isfinite(estimated_steps):
        raise ValueError("Step-based LR scheduling requires a finite training length")
    if not math.isfinite(warmup_epochs) or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be finite and non-negative")
    steps_per_epoch = math.ceil(batches / trainer.accumulate_grad_batches)
    warmup_steps = math.ceil(warmup_epochs * steps_per_epoch)
    total_steps = int(estimated_steps)
    if total_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_epochs must leave at least one optimizer step for decay")
    return warmup_steps, total_steps - warmup_steps
