"""Checkpoint callbacks shared by training entry points."""

import os

import pytorch_lightning as pl


class LatestCheckpoint(pl.Callback):
    """Save resumable checkpoints without replacing another run's latest file."""

    def __init__(self, dirpath, filename="last.ckpt"):
        super().__init__()
        self.dirpath = os.fspath(dirpath)
        self.filename = filename
        self.checkpoint_path = None

    def on_fit_start(self, trainer, pl_module):
        # Only the resumed file may be replaced. Choose once so subsequent
        # validations in this run update the same file.
        checkpoint_path = None
        if trainer.is_global_zero:
            preferred = os.path.join(self.dirpath, self.filename)
            resumed = trainer.ckpt_path
            if (
                resumed
                and os.path.dirname(os.path.realpath(resumed))
                == os.path.realpath(self.dirpath)
                and os.path.basename(resumed).startswith(
                    os.path.splitext(self.filename)[0] + "-v"
                )
                and os.path.splitext(resumed)[1] == os.path.splitext(self.filename)[1]
            ):
                preferred = os.fspath(resumed)
            checkpoint_path = preferred
            stem, extension = os.path.splitext(os.path.join(self.dirpath, self.filename))
            version = 1
            while os.path.lexists(checkpoint_path) and not (
                resumed and os.path.samefile(checkpoint_path, resumed)
            ):
                checkpoint_path = f"{stem}-v{version}{extension}"
                version += 1
        self.checkpoint_path = trainer.strategy.broadcast(checkpoint_path)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        trainer.save_checkpoint(
            self.checkpoint_path,
            weights_only=False,
        )
