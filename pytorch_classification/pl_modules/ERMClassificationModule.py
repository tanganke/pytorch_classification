"""
empirical risk minimization
"""

import logging
import os
from typing import Callable, Optional, Tuple, TypeVar, Union

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy

log = logging.getLogger(__name__)


def get_tensorboard_logger(module: pl.LightningModule) -> Union[SummaryWriter, None]:
    logger = module.trainer.logger
    if isinstance(logger, pl_loggers.TensorBoardLogger):
        tb_logger = logger.experiment
        return tb_logger
    else:
        return None


def log_confusion_matrix(tb_logger: SummaryWriter, tag: str, global_step: int, predictions: Tensor, targets: Tensor):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    tb_logger.add_figure(tag, disp.figure_, global_step)


Batch = TypeVar("Batch", bound=Tuple[Tensor, Tensor])


class ERMClassificationModule(pl.LightningModule):
    """
    Empirical Risk Minimization module for classification,
        minimize the cross entropy loss.

    Args:
        model (nn.Module): PyTorch model to use for classification.
        num_classes (int): Number of classes in the classification problem.
        optim_cfg (DictConfig): Configuration for the optimizer and learning rate scheduler.
        argumentation_fn (Optional[Callable[[Batch], Batch]]): Function to use for data argumentation.

    Attributes:
        train_acc (MulticlassAccuracy): Multiclass accuracy metric for training set.
        val_acc (MulticlassAccuracy): Multiclass accuracy metric for validation set.
        test_acc (MulticlassAccuracy): Multiclass accuracy metric for test set.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optim_cfg: Optional[DictConfig] = None,
        argumentation_fn: Optional[Callable[[Batch], Batch]] = None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.optim_cfg = optim_cfg
        self.argumentation_fn = argumentation_fn

        # metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute unnormalized probabilities (logits).

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Unnormalized probabilities (logits).
        """
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optim = {}
        if "optimizer" in self.optim_cfg:
            optim["optimizer"]: torch.optim.Optimizer = instantiate(
                self.optim_cfg["optimizer"],
                params=self.parameters(),
            )
        if "lr_scheduler" in self.optim_cfg:
            optim["lr_scheduler"]: torch.optim.lr_scheduler.LRScheduler = instantiate(
                self.optim_cfg["lr_scheduler"],
                optimizer=optim["optimizer"],
            )
        log.info(f"{'configure_optimizers':=^50}")
        log.info(optim)
        return optim

    def training_step(self, batch: Batch, batch_idx: int):
        # data argumentation
        if self.argumentation_fn is not None:
            batch = self.argumentation_fn(batch)

        x, y = batch
        logits: Tensor = self.forward(x)
        loss = F.cross_entropy(logits, y)

        # compute accuracy
        pred = logits.softmax(-1)
        self.train_acc(pred, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        x, y = batch

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.softmax(-1)
        self.val_acc(pred, y)

        self.log("val/accuracy", self.val_acc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch: Batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.softmax(-1)
        self.test_acc(pred, y)

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/accuracy", self.test_acc, prog_bar=True)
