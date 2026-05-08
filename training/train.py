import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
from omegaconf import OmegaConf
import torch
from lightning_module import Nova3RLightningModule
from lightning_data import Nova3RDataModule
from demo_nova3r import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R: 3D reconstruction from images")
    parser.add_argument("--ckpt", default="checkpoints/scene_n2/checkpoint-last.pth", help="Path to model checkpoint")
    #parser.add_argument("--output_dir", default="demo/outputs/", help="Output directory (default: demo/outputs/)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--aggregator_ckpt", default="./checkpoints/scene_n2/model.safetensors", help="Aggregator type (default: DepthAnything3Net)")
    args = parser.parse_args()

    return args

def load_data_config(ckpt_path):
    config_dir = os.path.join(os.path.dirname(ckpt_path), ".hydra")
    if os.path.exists(os.path.join(config_dir, "config.yaml")):
        cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
        return cfg.data
    else:
        raise FileNotFoundError(f"No .hydra/config.yaml found at {config_dir}. "
                               f"Please ensure the checkpoint directory contains the Hydra config.")


def main():
    # Set up data
    torch.cuda.memory._record_memory_history()
    args = parse_args()

    model, cfg = load_model(args.ckpt, args.device, aggregator_ckpt=args.aggregator_ckpt)

    data_cfg = load_data_config(args.ckpt)

    datamodule = Nova3RDataModule(
       data_cfg,
    )

    module = Nova3RLightningModule(cfg, model)

    # Set up logger & callbacks
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=cfg.output_dir, name="nova3r_training")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.output_dir, "checkpoints"),
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=args.device,
        devices=cfg.gpus,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=cfg.amp_dtype,
        log_every_n_steps=10,
    )

    # # --- THE ULTIMATE SMOKE TEST ---
    # print("1. Initializing Model & Data...")
    # device = torch.device("cuda")
    # module.to(device)

    # print("2. Testing DataLoader Initialization...")
    # datamodule.setup()
    # train_loader = datamodule.train_dataloader() # or your datamodule.train_dataloader()
    # iterator = iter(train_loader)

    # print("3. Attempting to fetch ONE batch...")
    # # This is where 90% of "100% CPU" hangs happen
    # batch = next(iterator) 
    # print("Batch fetched successfully!")

    # print("4. Attempting ONE forward pass...")
    # # This is the other 10%
    # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    #     module.train()
    #     output = module.training_step(batch, 0)
    #     print("Forward pass successful!")
    # # -------------------------------
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
