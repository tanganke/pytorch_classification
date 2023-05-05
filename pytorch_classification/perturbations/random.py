from typing import Tuple

import torch
from torch import Tensor


def generate_uniform_noise(*sizes, epsilon: float = 8 / 255, device=torch.device("cpu")):
    "generate uniform random noise from -`epsilon` to `epsilon`"
    noise = torch.FloatTensor(*sizes, device=device).uniform_(-epsilon, epsilon)
    return noise
