"""
[1] (ICLR 2023.) 
    L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodola, 
    “RELATIVE REPRESENTATIONS ENABLE ZERO-SHOT LATENT SPACE COMMUNICATION,” 
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.aggregate import forward_large_batch


def relative_projection(x: Tensor, anchors: Tensor):
    """
    Computes the relative representation of the input tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    anchors : Tensor
        The anchors tensor.

    Returns
    -------
    Tensor
        The relative representation of the input tensor.
    """
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)
    return torch.einsum("bm, am -> ba", x, anchors)
