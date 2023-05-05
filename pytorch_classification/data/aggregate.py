import torch
from torch import Tensor, nn
from tqdm import tqdm


@torch.no_grad()
def forward_large_batch(
    model: nn.Module,
    large_batch: Tensor,
    small_batch_size: int,
    device: torch.device,
    use_tqdm: bool = False,
):
    outputs = []
    pbar = torch.split(large_batch, small_batch_size)
    if use_tqdm:
        pbar = tqdm(pbar)
    for small_batch in pbar:
        small_batch = small_batch.to(device)
        out = model(small_batch).to(torch.device("cpu"))
        outputs.append(out)
    return torch.cat(outputs, dim=0)
