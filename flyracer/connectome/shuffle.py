"""Control connectomes that keep the statistics of a real one but destroy its specific wiring."""

from __future__ import annotations

import dataclasses

import numpy as np

from .schema import Connectome


def degree_preserving_shuffle(c: Connectome, seed: int = 0) -> Connectome:
    """Randomly rewire edges while keeping every neuron's in- and out-degree.

    Permuting the postsynaptic endpoints across edges leaves each neuron's
    out-degree (pre is untouched) and in-degree (same multiset of posts)
    exactly as before. Synapse counts travel with their presynaptic neuron and
    signs stay per neuron, so the E/I ratio and Dale's law are preserved.
    Duplicate edges created by the permutation are merged by summing counts.
    """
    rng = np.random.default_rng(seed)
    post = rng.permutation(c.post)
    keep = post != c.pre
    pre, post, count = c.pre[keep], post[keep], c.count[keep]
    key = pre.astype(np.int64) * c.n_neurons + post
    uniq, inv = np.unique(key, return_inverse=True)
    merged = np.bincount(inv, weights=count).astype(np.float32)
    meta = {**c.meta, "control": "degree_preserving_shuffle", "shuffle_seed": seed}
    return dataclasses.replace(
        c,
        pre=(uniq // c.n_neurons).astype(np.int32),
        post=(uniq % c.n_neurons).astype(np.int32),
        count=merged,
        meta=meta,
    )


def shuffle_signs(c: Connectome, seed: int = 0) -> Connectome:
    """Keep wiring, randomly reassign which neurons are inhibitory (same E/I ratio)."""
    rng = np.random.default_rng(seed)
    meta = {**c.meta, "control": "shuffled_signs", "shuffle_seed": seed}
    return dataclasses.replace(c, sign=rng.permutation(c.sign), meta=meta)
