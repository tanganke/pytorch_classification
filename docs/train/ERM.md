# ERM Training

General Command:

```bash
python3 scripts/train_ERM.py --config-name CONFIG_NAME [options]
```

config layout:

```yaml title='yaml config'
data:
  train_loader      # instantiatable
  val_loader: ???   # instantiatable (optional)
  test_loader: ???  # instantiatable (optional)

model           # instantiatable

trainer         # instantiatable
```

## MNIST

```bash
python3 scripts/train_ERM.py --config-name mnist-mlp
```

## CIFAR10/CIFAR100

CIFAR10

```bash
python3 scripts/train_ERM.py --config-name cifar10 [model=cifar_resnet50]
```

CIFAR100

```bash
python3 scripts/train_ERM.py --config-name cifar100 [model=cifar_resnet50]
```

## ImageNet

```bash
python3 scripts/train_ERM.py --config-name imagenet-resnet50
```
