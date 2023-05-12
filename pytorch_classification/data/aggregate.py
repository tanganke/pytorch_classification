from typing import Any, Callable, List

import numpy as np
import torch
import torch.utils.data
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm


@torch.no_grad()
def forward_large_batch(
    model: nn.Module,
    large_batch: Tensor,
    small_batch_size: int,
    device: torch.device,
    use_tqdm: bool = False,
):
    outputs = []
    pbar = torch.split(large_batch, small_batch_size)
    if use_tqdm:
        pbar = tqdm(pbar)
    for small_batch in pbar:
        small_batch = small_batch.to(device)
        out = model(small_batch).to(torch.device("cpu"))
        outputs.append(out)
    return torch.cat(outputs, dim=0)


def collect_random_samples(dataset: Dataset, n: int):
    samples = np.random.choice(len(dataset), n, replace=False)
    samples = [dataset[i] for i in samples]
    return samples


def collect_random_batch(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Callable[[List[Any]], Any] = torch.utils.data.default_collate,
):
    samples = collect_random_samples(dataset, batch_size)
    batch = collate_fn(samples)
    return batch
