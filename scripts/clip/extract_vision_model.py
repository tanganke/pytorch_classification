# Extract the vision model
import argparse
import logging
import os
from typing import Optional
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vision_model = CLIPVisionModel.from_pretrained(args.input)

    vision_model.save_pretrained(args.output)
    logging.info(f"extracted vision model to {args.output}")
