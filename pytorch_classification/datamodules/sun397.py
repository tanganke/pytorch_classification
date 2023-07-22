import os
import re
from typing import List

import lightning.pytorch as pl
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class SUN397DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__()

        self.train_dir = os.path.join(root, "sun397", "train")
        self.test_dir = os.path.join(root, "sun397", "test")

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.test_transform)

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classes = [idx_to_class[i][2:].replace("_", " ") for i in range(len(idx_to_class))]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
