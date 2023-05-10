from torch import nn, Tensor
import torch
from copy import deepcopy


def linear_interp(model_x: nn.Module, model_y: nn.Module, alpha: float):
    R"""
    weight interpolation
    return a new model whose parameters are the linear interpolation of `model_x` and `model_y`.

        returned_model = alpha * model_x + (1-alpha) * model_y

    Args:
        model_x (nn.Module): _description_
        model_y (nn.Module): _description_
        alpha (float): _description_

    Returns:
        nn.Module
    """
    assert 0 <= alpha <= 1
    assert type(model_x) == type(model_y)

    state_dict_x = model_x.state_dict()
    state_dict_y = model_y.state_dict()
    state_dict = {}
    for k in state_dict_x:
        state_dict[k] = alpha * state_dict_x[k] + (1 - alpha) * state_dict_y[k]

    returned_model = deepcopy(model_x)
    returned_model.load_state_dict(state_dict)
    return returned_model
