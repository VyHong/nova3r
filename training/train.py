import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os

from lightning_module import Nova3RLightningModule
from lightning_data import Nova3RDataModule


def main(args):
    # Set up data
    datamodule = Nova3RDataModule(
        train_list=args.train_list, val_list=args.val_list, root_dir=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Set up model
    model_kwargs = {
        # Populate with actual model configuration
    }
    model = Nova3RLightningModule(model_kwargs=model_kwargs, learning_rate=args.lr)

    # Set up logger & callbacks
    os.makedirs(args.log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=args.log_dir, name="nova3r_training")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16 if args.use_fp16 else 32,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, required=True, help="Path to training data list JSON")
    parser.add_argument("--val_list", type=str, required=True, help="Path to validation data list JSON")
    parser.add_argument("--data_root", type=str, default=".", help="Root directory for dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--use_fp16", action="store_true", help="Use 16-bit mixed precision")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and checkpoints")

    args = parser.parse_args()
    main(args)
