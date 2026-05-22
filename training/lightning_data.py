import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import json
import torch
import os
from PIL import Image
import torchvision.transforms as T
from training.data.datasets.replica_dataset import ReplicaPanoDataset  



class Nova3RDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.data_cfg = data_cfg
        self.root_dir = data_cfg.root_dir
        self.train_list = data_cfg.train_list_path
        self.val_list = data_cfg.val_list_path
        self.test_list = data_cfg.test_list_path
        self.batch_size = data_cfg.batch_size
        self.test_batch_size = data_cfg.test_batch_size
        self.num_workers = data_cfg.num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ReplicaPanoDataset(common_conf=self.data_cfg, scenes_list_path=self.train_list, data_root=self.root_dir)
            self.val_dataset = ReplicaPanoDataset(common_conf=self.data_cfg, scenes_list_path=self.val_list, data_root=self.root_dir)
        if stage == "validate" or stage is None:
            self.val_dataset = ReplicaPanoDataset(common_conf=self.data_cfg, scenes_list_path=self.val_list, data_root=self.root_dir)
        if stage == "test" or stage is None:
            self.test_dataset = ReplicaPanoDataset(common_conf=self.data_cfg, scenes_list_path=self.test_list, data_root=self.root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,collate_fn=self.train_dataset.dynamic_pad_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=self.val_dataset.dynamic_pad_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=self.test_dataset.dynamic_pad_collate_fn)
