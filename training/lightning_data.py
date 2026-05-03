import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import json
import torch
import os
from PIL import Image
import torchvision.transforms as T


class Nova3RDataset(Dataset):
    def __init__(self, data_list_path, root_dir, transform=None):
        self.root_dir = root_dir
        with open(data_list_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Implement actual loading logic here based on your data structure
        # E.g., loading images, depths, cam poses

        # Dummy implementations
        sample = {
            # "image": load_image(item['image_path']),
            "dummy_input": torch.randn(3, 224, 224)
        }
        return sample


class Nova3RDataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, root_dir, batch_size=4, num_workers=4):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Nova3RDataset(self.train_list, self.root_dir)
            self.val_dataset = Nova3RDataset(self.val_list, self.root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
