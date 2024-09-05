"""
Construct Unclear Dataset.

    (ICLR 2021) Huang etc. Unlearnable Examples: Making Personal Data Unexploitable
"""

import random
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SamplewiseUncleanDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        noise: Tensor,
        perturb_ratio: float = 1,
    ):
        super().__init__()
        assert len(dataset) == len(noise)
        self.dataset = dataset
        self.noise = noise
        self.perburt_ratio = perturb_ratio

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        if random.random() <= self.perburt_ratio:
            x = (x + self.noise[index]).clamp(0, 1)
        return x, y

    def __len__(self):
        return len(self.dataset)


class ClasswiseUncleanDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        noise: Tensor,
        if_perturb: bool | List[bool] = True,
        perturb_ratio: float = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.noise = noise
        self.if_perturb = if_perturb
        self.perburt_ratio = perturb_ratio

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        if self.if_perturb or self.if_perturb[y]:
            if random.random() <= self.perburt_ratio:
                x = (x + self.noise[y]).clamp(0, 1)
        return x, y

    def __len__(self):
        return len(self.dataset)
