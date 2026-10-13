"""Checkpoint callbacks shared by training entry points."""

import os

import pytorch_lightning as pl


class LatestCheckpoint(pl.Callback):
    """Write one full resumable checkpoint after every validation run."""

    def __init__(self, dirpath, filename="last.ckpt"):
        super().__init__()
        self.dirpath = os.fspath(dirpath)
        self.filename = filename

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        trainer.save_checkpoint(
            os.path.join(self.dirpath, self.filename),
            weights_only=False,
        )
