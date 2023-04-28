"""
Standard CIFAR Argumentation
"""
from torchvision import transforms

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        to_tensor,
        normalize,
    ]
)

val_transform = test_transform = transforms.Compose(
    [
        to_tensor,
        normalize,
    ]
)
