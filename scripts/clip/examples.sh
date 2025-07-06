python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset svhn
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset tanganke/gtsrb
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset tanganke/stanford_cars
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset tanganke/resisc45
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset tanganke/dtd
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset tanganke/eurosat
# python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset tanganke/sun397 # very large dataset
# python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset mnist
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset cifar10
python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset cifar100

python scripts/clip/finetune.py --model openai/clip-vit-large-patch14 --devices 4 --dataset "$HOME/datasets/tanganke/sun397"
python scripts/clip/finetune.py --dataset nateraw/rendered-sst2
python scripts/clip/finetune.py --dataset tanganke/stl10
python scripts/clip/finetune.py --dataset dpdl-benchmark/oxford_flowers102
python scripts/clip/finetune.py --dataset timm/oxford-iiit-pet

