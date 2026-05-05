"""Test-time augmentation: 4-view (identity + h-flip + v-flip + rot90) probability mean.

For chip images (200x200 pure pattern, no orientation prior), all 4 views are valid.
Output: averaged sigmoid probabilities — call this AFTER temperature scaling on logits,
or pass logits and TTA returns mean sigmoid post-T.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


@torch.no_grad()
def tta_logits(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    n_views: int = 4,
) -> np.ndarray:
    """Run model under 4 views, return averaged logits (NOT sigmoid).

    img_tensor: (N, C, H, W) on CPU, float in [0,1] or normalized — caller's choice.
    Returns: (N, num_classes) numpy logits, mean across views.
    """
    if n_views not in (1, 2, 4):
        raise ValueError(f"n_views must be 1/2/4, got {n_views}")
    views: list[Callable[[torch.Tensor], torch.Tensor]] = [lambda x: x]
    if n_views >= 2:
        views.append(lambda x: torch.flip(x, dims=[-1]))
    if n_views >= 4:
        views.append(lambda x: torch.flip(x, dims=[-2]))
        views.append(lambda x: torch.rot90(x, k=1, dims=[-2, -1]))
    out = None
    for vfn in views:
        x = vfn(img_tensor).to(device, non_blocking=True)
        logits = model(x).detach().float().cpu().numpy()
        out = logits if out is None else (out + logits)
    return out / float(len(views))
