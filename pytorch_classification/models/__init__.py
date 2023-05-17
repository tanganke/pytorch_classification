import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from . import cifar_models
from .mlp import MLP


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


def load_model(
    model_fn,
    state_dict_path: str,
    device: Optional[torch.device] = None,
    model_args: Tuple[Any] = [],
    model_kwargs: Dict[str, Any] = {},
) -> nn.Module:
    R"""
    instantiate model from `model_fn` and load state dict from `state_dict_path`.

    Args:
        model_fn (_type_): instantiate model by calling `model_fn(*model_args, **model_kwargs)`.
        state_dict_path (str): state dict to load.
        device (Optional[torch.device], optional): _description_. Defaults to None.
        model_args (Tuple[Any], optional): args pass to `model_fn`. Defaults to [].
        model_kwargs (Dict[str, Any], optional): kwargs pass to `model_fn`. Defaults to {}.

    Raises:
        FileNotFoundError: _description_

    Returns:
        nn.Module: _description_
    """
    # create model instance
    model: nn.Module = model_fn(*model_args, **model_kwargs)

    # try load state dict
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"State dict path {state_dict_path} does not exist")
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

    if device is not None:
        model = model.to(device=device)
    return model
