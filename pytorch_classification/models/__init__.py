import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from . import cifar_models
from .mlp import MLP


def load_model(
    model_fn,
    state_dict_path: str,
    device: Optional[torch.device] = None,
    model_args: Tuple[Any] = [],
    model_kwargs: Dict[str, Any] = {},
):
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
