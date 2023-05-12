from typing import Iterable

import clip
import torch
from torch import Tensor
from torch.types import _device


def load_clip_features(
    class_names: Iterable[str],
    device: _device,
    clip_model_name: str = "ViT-B/32",
):
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    model, preprocess = clip.load(clip_model_name, device)
    with torch.no_grad():
        text_features: Tensor = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features
