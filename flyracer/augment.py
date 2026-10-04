"""Image augmentation for pixel-based RL."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_shift(obs: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Shift each frame stack by a random whole number of pixels in [-pad, pad] on each axis.

    The same shift applies to all stacked frames of one sample (so motion between
    frames is preserved); edges repeat the border pixels. Returns float, same scale.
    """
    b, _, h, w = obs.shape
    x = obs.float()
    shift = torch.randint(-pad, pad + 1, (b, 2), device=x.device).float()
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift[:, 0] / w
    theta[:, 1, 2] = 2.0 * shift[:, 1] / h
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="border", align_corners=False)
