r"""
The `pytorch_classification` package provides modules for training and evaluating PyTorch models for classification tasks.

This package contains the following submodules:
- `data`: utilities for loading and preprocessing datasets
- `models`: implementations of various classification models
- `pl_modules`: PyTorch Lightning modules for training and evaluating models
- `plot`: utilities for visualizing model performance
- `transforms`: data augmentation and preprocessing transforms
- `utils`: miscellaneous utilities

In addition, this package defines several type aliases in the `types` module.

Examples:
    >>> import pytorch_classification

    # Use the `plot` module to visualize model performance
    >>> pytorch_classification.plot.plot_confusion_matrix(predictions, targets)
"""
from . import data, models, pl_modules, plot, transforms, utils
from .types import *
