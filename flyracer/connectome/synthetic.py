"""Synthetic Tier S connectome with the same shape as the real extraction.

Used when no neuPrint token is available, and in tests. It is NOT fly wiring:
it is a layered sparse graph (bilateral inputs -> recurrent core -> bilateral
outputs) with heavy-tailed synapse counts, so the full pipeline runs end to end.
"""

from __future__ import annotations

import numpy as np

from .schema import CORE, INPUT, OUTPUT, Connectome


def _edges(rng, src, dst, p, allow_self=False):
    mask = rng.random((len(src), len(dst))) < p
    i, j = np.nonzero(mask)
    pre, post = src[i], dst[j]
    if not allow_self:
        keep = pre != post
        pre, post = pre[keep], post[keep]
    return pre, post


def make_synthetic(
    n_input: int = 600,
    n_core: int = 2400,
    n_output: int = 100,
    density: float = 0.01,
    inhibitory_frac: float = 0.3,
    seed: int = 0,
) -> Connectome:
    rng = np.random.default_rng(seed)
    n = n_input + n_core + n_output
    group = np.array([INPUT] * n_input + [CORE] * n_core + [OUTPUT] * n_output, dtype=np.int8)
    idx_in, idx_core, idx_out = (np.flatnonzero(group == g) for g in (INPUT, CORE, OUTPUT))

    side = rng.choice([-1, 1], size=n).astype(np.int8)
    side[idx_core[: n_core // 10]] = 0  # some midline neurons

    # Feedforward input -> core, recurrent core, core -> output, some output feedback.
    parts = [
        _edges(rng, idx_in, idx_core, density * 2),
        _edges(rng, idx_core, idx_core, density),
        _edges(rng, idx_core, idx_out, density * 4),
        _edges(rng, idx_out, idx_core, density / 2),
    ]
    pre = np.concatenate([p for p, _ in parts])
    post = np.concatenate([q for _, q in parts])
    # Prefer ipsilateral wiring, as in a real brain: drop half the contralateral edges.
    contra = (side[pre] * side[post]) < 0
    keep = ~contra | (rng.random(len(pre)) < 0.5)
    pre, post = pre[keep], post[keep]
    count = np.ceil(rng.lognormal(mean=1.0, sigma=1.0, size=len(pre))).astype(np.float32)

    sign = np.where(rng.random(n) < inhibitory_frac, -1, 1).astype(np.int8)
    sign[idx_in] = 1  # sensory neurons are excitatory

    # Brain-shaped layout for visualization: optic lobes left/right, core in the
    # central brain, descending outputs dropping toward the nerve cord.
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[idx_in, 0] = side[idx_in] * rng.normal(3.2, 0.35, len(idx_in))
    pos[idx_in, 1] = rng.normal(0.0, 0.9, len(idx_in))
    pos[idx_core, 0] = rng.normal(0.0, 1.1, len(idx_core))
    pos[idx_core, 1] = rng.normal(-0.2, 0.7, len(idx_core))
    pos[idx_out, 0] = side[idx_out] * rng.normal(0.4, 0.15, len(idx_out))
    pos[idx_out, 1] = rng.normal(1.8, 0.4, len(idx_out))
    pos[:, 2] = rng.normal(0, 0.5, n)

    cell_type = np.array(
        [f"SYN_IN_{i % 20}" for i in range(n_input)]
        + [f"SYN_CORE_{i % 100}" for i in range(n_core)]
        + [f"SYN_DN_{i % 10}" for i in range(n_output)]
    )
    return Connectome(
        pre=pre.astype(np.int32), post=post.astype(np.int32), count=count, sign=sign,
        group=group, side=side, pos=pos, body_id=np.arange(n, dtype=np.int64),
        cell_type=cell_type,
        meta={"source": "synthetic", "dataset": f"seed{seed}", "density": density},
    )
