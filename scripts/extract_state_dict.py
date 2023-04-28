"""
extrate plain state dict from pytorch lightning checkpoint file.

for example, a checkpoint file containing

    {
        'state_dict': {
            'model.xxx': <Tensor>,
            ...
        }
    }

this will be converted into

    {
        'xxx': <Tensor>,
        ...
    }
"""
import argparse
import os

import torch

from pytorch_classification.utils.logging import logging

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="model.")
    parser.add_argument("ckpt", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.ckpt), "Cannot find ckeckpoint file"
    if os.path.exists(args.output_path):
        log.warn("output_path already exists, overwriting it")

    ckpt = torch.load(args.ckpt, torch.device("cpu"))
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        assert k.startswith(args.prefix)
        new_state_dict[k[len(args.prefix) :]] = v

    torch.save(new_state_dict, args.output_path)
