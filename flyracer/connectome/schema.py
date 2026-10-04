"""On-disk connectome format shared by the neuPrint extractor and the synthetic generator.

A connectome is a directed graph of N neurons with E weighted edges:
  pre, post   int32[E]    presynaptic / postsynaptic neuron index
  count       float32[E]  synapse count of the connection
  sign        int8[N]     +1 excitatory, -1 inhibitory (Dale's law, per presynaptic neuron)
  group       int8[N]     0 = input (sensory), 1 = core, 2 = output (descending)
  side        int8[N]     -1 left, 0 midline/unknown, +1 right
  pos         float32[N,3] soma position (arbitrary units, used only for visualization)
  body_id     int64[N]    neuPrint bodyId (synthetic: 0..N-1)
  cell_type   str[N]
  meta        JSON string (source, dataset, selection config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INPUT, CORE, OUTPUT = 0, 1, 2
GROUP_NAMES = {INPUT: "input", CORE: "core", OUTPUT: "output"}


@dataclass
class Connectome:
    pre: np.ndarray
    post: np.ndarray
    count: np.ndarray
    sign: np.ndarray
    group: np.ndarray
    side: np.ndarray
    pos: np.ndarray
    body_id: np.ndarray
    cell_type: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.sign)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def indices(self, group: int) -> np.ndarray:
        return np.flatnonzero(self.group == group)

    def summary(self) -> str:
        counts = {GROUP_NAMES[g]: int((self.group == g).sum()) for g in GROUP_NAMES}
        inhib = float((self.sign < 0).mean())
        return (
            f"{self.meta.get('source', '?')} {self.meta.get('dataset', '')}: "
            f"{self.n_neurons} neurons {counts}, {self.n_edges} connections, "
            f"{int(self.count.sum())} synapses, {inhib:.0%} inhibitory"
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            pre=self.pre.astype(np.int32),
            post=self.post.astype(np.int32),
            count=self.count.astype(np.float32),
            sign=self.sign.astype(np.int8),
            group=self.group.astype(np.int8),
            side=self.side.astype(np.int8),
            pos=self.pos.astype(np.float32),
            body_id=self.body_id.astype(np.int64),
            cell_type=self.cell_type.astype(str),
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str | Path) -> Connectome:
        z = np.load(path, allow_pickle=False)
        return cls(
            pre=z["pre"], post=z["post"], count=z["count"], sign=z["sign"],
            group=z["group"], side=z["side"], pos=z["pos"], body_id=z["body_id"],
            cell_type=z["cell_type"], meta=json.loads(str(z["meta"])),
        )
