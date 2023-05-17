import torch
from torch import Tensor


def accuracy(logits: Tensor, targets: Tensor):
    assert logits.dim() == 2
    assert len(logits) == len(targets)
    predictions = logits.argmax(-1)
    acc = (predictions == targets).sum() / len(targets)
    return acc
