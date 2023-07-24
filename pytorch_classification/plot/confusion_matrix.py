from typing import Tuple, Union

import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor

from ..types import as_numpy

__all__ = [
    "confusion_matrix",
    "plot_confusion_matrix",
]


def plot_confusion_matrix(
    predictions: Union[Tensor, np.ndarray],
    targets: Union[Tensor, np.ndarray],
    **plot_args,
):
    """
    Plot a confusion matrix for the given predictions and targets.

    Args:
        predictions (Union[Tensor, np.ndarray]): The predicted labels.
        targets (Union[Tensor, np.ndarray]): The true labels.
        **plot_args: Additional arguments to pass to the plot function.

    Returns:
        Tuple: A tuple containing the figure and axis objects of the plot.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    predictions = as_numpy(predictions)
    targets = as_numpy(targets)

    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(**plot_args)

    return disp.figure_, disp.ax_
