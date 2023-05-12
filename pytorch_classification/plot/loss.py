R"""
visualize the loss landscape.

Examples:
    
    1. https://github.com/marcellodebernardi/loss-landscapes/blob/master/examples/core-features.ipynb

Referneces:
    
    [1] Visualizing the Loss Landscape of Neural Nets.

"""
from typing import Tuple

import loss_landscapes
import torch
from loss_landscapes import linear_interpolation, random_plane
from loss_landscapes.metrics import Loss
from torch import Tensor, nn
from torch.nn import functional as F
from torch.types import _device


def get_metric(
    batch: Tuple[Tensor, Tensor],
    loss_fn=F.cross_entropy,
    device: _device = None,
):
    inputs, targets = batch
    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)

    metric = Loss(loss_fn, inputs, targets)
    return metric
