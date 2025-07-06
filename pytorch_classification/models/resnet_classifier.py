from typing import List, Literal

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


# Define LightningModule
class ResNetClassifier(L.LightningModule):
    model_mapping = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }
    model_weights_mapping = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        "resnet101": ResNet101_Weights.IMAGENET1K_V1,
        "resnet152": ResNet152_Weights.IMAGENET1K_V1,
    }

    def __init__(
        self,
        class_names: List[str],
        model_name: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        pretrained: bool = False,
        max_epochs: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.lr = lr

        self.class_names = class_names
        self.num_classes = num_classes = len(class_names)
        self.model = self.model_mapping[model_name](
            weights=(self.model_weights_mapping[model_name] if pretrained else None),
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Initialize Accuracy metric
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _step(self, batch):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate and log accuracy
        acc = self.train_accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)


def save_model(model, train_dataset, path):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": train_dataset.features["label"].names,
        },
        path,
    )


def load_model(model_name: str, path):
    checkpoint = torch.load(path)
    model = ResNetClassifier(
        model_name,
        pretrained=False,
        class_names=checkpoint["class_names"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
