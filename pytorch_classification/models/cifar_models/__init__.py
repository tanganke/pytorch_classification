"""
references:
    https://github.com/chenyaofo/pytorch-cifar-model
"""
from .mobilenetv2 import (
    cifar10_mobilenetv2_x0_5,
    cifar10_mobilenetv2_x0_75,
    cifar10_mobilenetv2_x1_0,
    cifar10_mobilenetv2_x1_4,
    cifar100_mobilenetv2_x0_5,
    cifar100_mobilenetv2_x0_75,
    cifar100_mobilenetv2_x1_0,
    cifar100_mobilenetv2_x1_4,
)
from .repvgg import (
    cifar10_repvgg_a0,
    cifar10_repvgg_a1,
    cifar10_repvgg_a2,
    cifar100_repvgg_a0,
    cifar100_repvgg_a1,
    cifar100_repvgg_a2,
)
from .resnet import (
    cifar10_resnet20,
    cifar10_resnet32,
    cifar10_resnet44,
    cifar10_resnet56,
    cifar100_resnet20,
    cifar100_resnet32,
    cifar100_resnet44,
    cifar100_resnet56,
)
from .shufflenetv2 import (
    cifar10_shufflenetv2_x0_5,
    cifar10_shufflenetv2_x1_0,
    cifar10_shufflenetv2_x1_5,
    cifar10_shufflenetv2_x2_0,
    cifar100_shufflenetv2_x0_5,
    cifar100_shufflenetv2_x1_0,
    cifar100_shufflenetv2_x1_5,
    cifar100_shufflenetv2_x2_0,
)
from .vgg import (
    cifar10_vgg11_bn,
    cifar10_vgg13_bn,
    cifar10_vgg16_bn,
    cifar10_vgg19_bn,
    cifar100_vgg11_bn,
    cifar100_vgg13_bn,
    cifar100_vgg16_bn,
    cifar100_vgg19_bn,
)
from .vit import (
    cifar10_vit_b16,
    cifar10_vit_b32,
    cifar10_vit_h14,
    cifar10_vit_l16,
    cifar10_vit_l32,
    cifar100_vit_b16,
    cifar100_vit_b32,
    cifar100_vit_h14,
    cifar100_vit_l16,
    cifar100_vit_l32,
)
