import inspect

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import OmegaConf

from training.nova3r.data.datasets.front3d_dataset import Front3DDataset
from training.nova3r.data.datasets.replica_dataset import ReplicaPanoDataset
from training.mesh2sdf.data.front3d_dataset import Front3DSDFDataset
from training.mesh2sdf.data.replica_dataset import ReplicaPanoSDFDataset


class Nova3RDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.data_cfg = data_cfg
        self.batch_size = data_cfg.batch_size
        self.test_batch_size = data_cfg.test_batch_size
        self.num_workers = data_cfg.num_workers

        self.dataset_registry = {
            "Front3DDataset": Front3DDataset,
            "Front3DSDFDataset": Front3DSDFDataset,
            "ReplicaPanoDataset": ReplicaPanoDataset,
            "ReplicaPanoSDFDataset": ReplicaPanoSDFDataset,
        }

        self.train_collate_fn = None
        self.val_collate_fn = None
        self.test_collate_fn = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset, self.train_collate_fn = self._build_concat_dataset("train_datasets", split="train")
            self.val_dataset, self.val_collate_fn = self._build_concat_dataset("val_datasets", split="val")
        if stage == "validate" or stage is None:
            self.val_dataset, self.val_collate_fn = self._build_concat_dataset("val_datasets", split="val")
        if stage == "test" or stage is None:
            self.test_dataset, self.test_collate_fn = self._build_concat_dataset("test_datasets", split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_collate_fn,
        )

    def _build_dataset(self, dataset_cfg, split):
        dataset_name = dataset_cfg.name
        if dataset_name not in self.dataset_registry:
            raise ValueError(f"Unknown dataset class in config: {dataset_name}")

        dataset_cls = self.dataset_registry[dataset_name]
        params = OmegaConf.to_container(dataset_cfg.params, resolve=True) if "params" in dataset_cfg else {}

        kwargs = {
            "common_conf": self.data_cfg,
            "split": split,
            "data_root": params.pop("root_dir"),
        }

        split_list_key = f"{split}_list_path"
        kwargs["samples_list_path"] = params.pop(split_list_key)
        kwargs.update(params)
        valid_params = set(inspect.signature(dataset_cls.__init__).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return dataset_cls(**filtered_kwargs)

    def _build_concat_dataset(self, config_key, split):
        datasets_cfg = getattr(self.data_cfg, config_key)
        datasets = []
        for _, dataset_cfg in datasets_cfg.items():
            datasets.append(self._build_dataset(dataset_cfg, split=split))

        collate_fn = getattr(datasets[0], "dynamic_pad_collate_fn", None)
        return ConcatDataset(datasets), collate_fn

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_collate_fn,
        )
