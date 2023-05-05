"""
https://github.com/pytorch/examples/blob/main/mnist/main.py
"""
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
