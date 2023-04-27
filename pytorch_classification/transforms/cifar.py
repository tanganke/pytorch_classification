"""
Standard CIFAR Argumentation
"""
from torchvision import transforms

_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize,
    ]
)

val_transform = test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        _normalize,
    ]
)
