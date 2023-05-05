import torch
from torch import Tensor

__all__ = [
    "channel_shuffle",
    "batch_channel_shuffle",
]


def _channel_shuffle(image: Tensor):
    assert image.dim() == 3, "`images` must be 3 dimensions"
    device = image.device
    num_channels, height, width = image.size()
    return image[torch.randperm(num_channels)]


def channel_shuffle(images: Tensor):
    assert images.dim() == 3 or images.dim() == 4, "`images` must be 3 or 4 dimensions"
    if images.dim() == 3:
        return _channel_shuffle(images)

    shuffled_images = torch.empty_like(images)
    for i in range(images.size(0)):
        shuffled_images[i] = _channel_shuffle(images[i])
    return shuffled_images


def batch_channel_shuffle(images: Tensor):
    assert images.dim() == 4, "`images` must be 4 dimensions"
    batch_size, num_channels, height, width = images.size()
    shuffled_images = torch.empty_like(images)
    indices = torch.randperm(num_channels)
    shuffled_images[:, indices] = images[:, indices]
    return shuffled_images
