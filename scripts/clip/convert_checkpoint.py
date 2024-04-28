"""
convert checkpoint from pytorch lightning to huggingface
"""

import argparse
import logging
import os
from typing import Optional
from transformers import CLIPProcessor, CLIPModel
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)

    checkpoint = torch.load(args.checkpoint)
    # remove the prefix from the keys
    state_dict = {}
    for key in checkpoint["state_dict"]:
        prefix = "model.clip_model."
        assert key.startswith(prefix)
        state_dict[key[len(prefix) :]] = checkpoint["state_dict"][key]

    model.load_state_dict(state_dict)

    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    logging.info(f"converted checkpoint to {args.output}")
