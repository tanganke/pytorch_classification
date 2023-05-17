from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

__all__ = [
    "collect_index_by_label",
    "subset_by_label",
]


def collect_index_by_label(dataset: Dataset, labels: List[int]):
    indices = []
    for i, (x, y) in enumerate(dataset):
        if y in labels:
            indices.append(i)
    return indices


def subset_by_label(dataset: Dataset, labels: List[int]):
    """
    Attention:
        the label space of the returned subset is the same as labels, but not [0, `len(labels)`)

    Args:
        dataset (Dataset): _description_
        labels (List[int]): _description_

    Returns:
        Subset: _description_
    """
    indices = collect_index_by_label(dataset, labels)
    subset = Subset(dataset, indices=indices)
    return subset
