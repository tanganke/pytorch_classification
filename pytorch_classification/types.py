import numpy as np
import torch
from torch import Tensor
from torch.types import _device, _dtype

__all__ = [
    "as_numpy",
    "as_tensor",
]


def as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, Tensor):
        return x.cpu().numpy()
    else:
        return np.asarray(x)


def as_tensor(
    x,
    device: _device = None,
    dtype: _dtype = None,
) -> Tensor:
    if isinstance(x, Tensor):
        x = x
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.as_tensor(x)

    if device is not None or dtype is not None:
        x = x.to(device=device, dtype=dtype)
    return x
