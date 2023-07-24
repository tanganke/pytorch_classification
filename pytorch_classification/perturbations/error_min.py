"""
This module contains functions for generating sample-wise and class-wise error-minimizing noise for a given PyTorch model.

    (ICLR 2021) Huang etc. Unlearnable Examples: Making Personal Data Unexploitable
"""
import itertools
import logging
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, autograd, nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from ..data import num_samples


def to_device(
    batch: Iterable[Tensor],
    device: torch.device,
):
    """
    Move tensors in batch to the specified device.

    Args:
    - batch: Iterable[Tensor]: A batch of tensors.
    - device: torch.device: The device to move the tensors to.

    Returns:
    - Tuple[Tensor]: A tuple of tensors moved to the specified device.
    """
    return tuple(t.to(device) for t in batch)


def train_batch(
    model: nn.Module,
    batch: Tuple[Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
):
    """
    Trains the given PyTorch model on a batch of data.
    optimize the the model with respect to the cross-entropy loss.

    Args:
        model (nn.Module): The PyTorch model to train.
        batch (Tuple[Tensor, Tensor]): A tuple containing the input data and its corresponding labels.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.

    Returns:
        None
    """
    images, labels = batch
    logits = model(images)
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def min_min_attack_batch(
    model: nn.Module,
    batch: Tuple[Tensor, Tensor],
    noise: Tensor,
    epsilon: float,
    num_steps: int = 1,
    step_size: float = 0.8,
):
    images, labels = batch
    for _ in range(num_steps):
        perturb_img = (images + noise).clamp_(0, 1).requires_grad_(True)

        logits = model(perturb_img)
        loss = F.cross_entropy(logits, labels)

        [grad] = autograd.grad(loss, [perturb_img])

        with torch.no_grad():
            perturb_img = perturb_img - step_size * grad.sign()
            noise = torch.clamp(perturb_img - images, -epsilon, epsilon)

    return noise


def samplewise_perturbation_eval(
    model: nn.Module,
    dataloader: DataLoader,
    noise: Tensor,
    device,
):
    "error rate on dataloader"
    model.eval()
    avg_acc = torchmetrics.MeanMetric()
    image_id = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, "evaluation")):
        batch = to_device(batch, device)
        images, labels = batch
        batch_size = images.size(0)
        batch_noise = noise[image_id : image_id + batch_size].to(device)
        perturb_images = images + batch_noise

        logits = model(perturb_images)
        preds = logits.softmax(-1).max(-1).indices
        acc = (preds == labels).sum() / batch_size
        avg_acc.update(acc.item(), batch_size)
        image_id += batch_size

    avg_acc = avg_acc.compute().item()
    return 1 - avg_acc


def generate_samplewise_error_min_noise(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epsilon: float = 8 / 255,
    tolerance: float = 0.01,
    train_steps: int = 10,
    perturb_steps: int = 20,
    perturb_step_size: float = 0.8 / 255,
    device=torch.device("cpu"),
):
    """
        generate sample-wise error-min noise.
        algorithm terminates until error rate less than `tolerance`.

    Returns:
        Tensor: noise ( #samples, *image_sizes )
    """
    assert isinstance(dataloader.sampler, SequentialSampler), "shuffle must be `False`"

    # initialize noise
    for images, labels in dataloader:
        image = images[0]
        break
    noise = torch.zeros(num_samples(dataloader), *image.size(), dtype=torch.float32)

    data_iter = iter(range(0))
    for loop_idx in itertools.count():
        logging.info(f"iterating: {loop_idx}")
        # ---- Train Batch for min-min noise ----
        model.train()
        for _ in tqdm(range(train_steps), "optimization steps"):
            try:
                (images, labels) = next(data_iter)
            except StopIteration:
                train_image_id = 0
                data_iter = iter(dataloader)
                (images, labels) = next(data_iter)

            images, labels = images.to(device), labels.to(device)
            # Add Sample-wise Noise to each sample
            for i, (image, label) in enumerate(zip(images, labels)):
                images[i] = images[i] + noise[train_image_id].to(device)
                train_image_id += 1
            train_batch(model, (images, labels), optimizer)

        # ---- Search For Noise ----
        image_id = 0
        model.eval()
        for i, batch in enumerate(tqdm(dataloader, "search noise")):
            batch = to_device(batch, device)
            images, labels = batch
            batch_size = images.size(0)

            batch_noise = noise[image_id : image_id + batch_size].to(device)

            # Update sample-wise perturbation
            batch_noise = min_min_attack_batch(model, batch, batch_noise, epsilon=epsilon, num_steps=perturb_steps, step_size=perturb_step_size)

            noise[image_id : image_id + batch_size] = batch_noise.cpu()
            image_id += batch_size

        # Eval termination conditions
        error_rate = samplewise_perturbation_eval(model, dataloader, noise, device)
        logging.info(f"average error rate: {error_rate:.2f}%")

        if error_rate < tolerance:
            break

    return noise


def classwise_perturbation_eval(
    model: nn.Module,
    dataloader: DataLoader,
    noise: Tensor,
    device: torch.device,
):
    "error rate on dataloader"
    model.eval()
    avg_acc = torchmetrics.MeanMetric()
    for batch_idx, batch in enumerate(tqdm(dataloader, "evaluation")):
        batch = to_device(batch, device)
        images, labels = batch
        batch_size = images.size(0)
        batch_noise = noise[labels]
        perturb_images = images + batch_noise

        logits = model(perturb_images)
        preds = logits.softmax(-1).max(-1).indices
        acc = (preds == labels).sum() / batch_size
        avg_acc.update(acc.item(), batch_size)

    avg_acc = avg_acc.compute().item()
    return 1 - avg_acc


def generate_classwise_error_min_noise(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    epsilon: float = 8 / 255,
    tolerance: float = 0.01,
    train_steps: int = 10,
    perturb_steps: int = 20,
    perturb_step_size: float = 0.8 / 255,
    device=torch.device("cpu"),
):
    assert isinstance(dataloader.sampler, SequentialSampler), "shuffle must be `False`"

    # initialize noise
    for images, labels in dataloader:
        image = images[0]
        break
    noise = torch.zeros(num_classes, *image.size(), dtype=torch.float32).to(device)

    data_iter = iter(range(0))
    for loop_idx in itertools.count():
        logging.info(f"iterating: {loop_idx}")
        # ---- Train Batch for min-min noise ----
        model.train()
        for _ in tqdm(range(train_steps), "optimization steps"):
            try:
                (images, labels) = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                (images, labels) = next(data_iter)

            images, labels = images.to(device), labels.to(device)
            # Add Sample-wise Noise to each sample
            images = images + noise[labels]
            train_batch(model, (images, labels), optimizer)

        # ---- Search For Noise ----
        model.eval()
        for i, batch in enumerate(tqdm(dataloader, "search noise")):
            batch = to_device(batch, device)
            images, labels = batch

            batch_noise = noise[labels]

            # Update sample-wise perturbation
            batch_noise = min_min_attack_batch(model, batch, batch_noise, epsilon=epsilon, num_steps=perturb_steps, step_size=perturb_step_size)

            for y in range(num_classes):
                class_noise = batch_noise[labels == y]
                if len(class_noise) > 0:
                    noise[y] = class_noise.mean(0)

        # Eval termination conditions
        error_rate = classwise_perturbation_eval(model, dataloader, noise, device)
        logging.info(f"average error rate: {error_rate:.2f}%")

        if error_rate < tolerance:
            break

    return noise
