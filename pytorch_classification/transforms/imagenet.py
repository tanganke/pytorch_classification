"""
Standard ImageNet Argumentation

https://github.com/pytorch/examples/blob/main/imagenet/main.py
"""

from torchvision import transforms

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        to_tensor,
        normalize,
    ]
)

test_transform = val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        to_tensor,
        normalize,
    ]
)
