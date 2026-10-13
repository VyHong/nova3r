# Training

Run commands from the repository root. The shell launchers intentionally remain
at the root so existing scheduler jobs and personal workflows keep working.

## Launchers

- `train.sh`: single-GPU containerized training.
- `train_ddp.sh`: distributed training.
- `test.sh`: evaluation launcher.
- `mount_datasets.sh`: mount all configured datasets concurrently.

The corresponding Python entry points live under `training/`:

- `training/train.py`: start a new training run.
- `training/resume_train.py`: resume a full Lightning checkpoint.
- `training/test.py`: evaluate a trained checkpoint.
- `training/lr_sweep.py`: learning-rate experiments.

Machine-specific examples and older exploratory commands are retained in the
[operations runbook](runbooks/OPERATIONS.md).
