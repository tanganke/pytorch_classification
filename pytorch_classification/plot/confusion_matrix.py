import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor

from ..types import as_numpy

__all__ = [
    "confusion_matrix",
    "plot_confusion_matrix",
]


def plot_confusion_matrix(predictions, targets, **plot_args):
    from sklearn.metrics import ConfusionMatrixDisplay

    predictions = as_numpy(predictions)
    targets = as_numpy(targets)

    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(**plot_args)

    return disp.figure_, disp.ax_
