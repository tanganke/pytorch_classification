import os

import torch

from . import cifar_models


def load_model(model_fn, state_dict_path: str, device: torch.device):
    model = model_fn()
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"State dict path {state_dict_path} does not exist")
    else:
        state_dict = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    model = model.to(device=device)
    return model
