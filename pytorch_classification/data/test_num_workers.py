import multiprocessing as mp
from time import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

__all__ = ['test_num_workers']

def test_num_workers(
    dataset: Dataset,
    num_workers_set: Iterable[int] = range(2, mp.cpu_count(), 2),
    test_epoch: int = 3,
    shuffle: bool = True,
    batch_size: int = 64,
    **loader_kwargs,
):
    for num_workers in num_workers_set:
        train_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            **loader_kwargs,
        )
        start = time()
        for epoch_idx in range(test_epoch):
            for batch_idx, data in enumerate(tqdm(train_loader, f"epoch {epoch_idx+1}/{test_epoch}")):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
