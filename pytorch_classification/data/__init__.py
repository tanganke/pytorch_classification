from torch import Tensor
from torch.types import _device
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import first
from .subset import *
from .test_num_workers import *


def num_samples(dataloader: DataLoader):
    n = 0
    batch = first(batch)
    if isinstance(batch, (tuple, list)):
        for batch in tqdm(dataloader, "counting samples"):
            x = batch[0]
            batch_size = len(x)
            n += batch_size
    elif isinstance(batch, dict):
        for batch in tqdm(dataloader, "counting samples"):
            for k, v in batch.items():
                batch_size = len(v)
                n += batch_size
                break
    else:
        raise NotImplementedError(f"unsupported batch type, {type(batch)}")

    return n


def to_device(data, device: _device):
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(to_device(x, device) for x in data)
    elif isinstance(data, dict):
        ret = {}
        for k, x in data.items():
            ret[k] = to_device(x, device)
    else:
        raise NotImplementedError(f"unsupported type: `{type(data)}`")
