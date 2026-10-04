"""FlyRNN: a rate network whose wiring and signs come from a connectome.

Each neuron has a membrane state v and firing rate r = tanh(relu(v)) in [0, 1).
Per game frame the network settles for `steps` Euler steps from rest:

    v <- v + (dt / tau) * (-v + W r + b + I_in)

W is fixed in *topology* (only connectome edges exist) and in *sign* (Dale's
law, per presynaptic neuron). What learns: per-edge synaptic magnitude
(initialized from synapse counts), per-neuron time constants and biases, and
the projection from visual features onto the input neurons.

The network starts from rest every frame, so no backprop through time across
frames is needed; temporal context comes from frame stacking.
"""

from __future__ import annotations

import torch
from torch import nn

from ..connectome import INPUT, OUTPUT, Connectome


class FlyRNN(nn.Module):
    def __init__(
        self,
        c: Connectome,
        n_features: int,
        steps: int = 6,
        gain: float = 2.0,
        learn_weights: bool = True,
    ):
        super().__init__()
        self.n = c.n_neurons
        self.steps = steps
        pre = torch.as_tensor(c.pre, dtype=torch.long)
        post = torch.as_tensor(c.post, dtype=torch.long)
        count = torch.as_tensor(c.count, dtype=torch.float32)

        # Normalize each neuron's total input to 1 so activity neither dies nor explodes
        # regardless of how many synapses it receives; `gain` then sets the overall drive.
        in_strength = torch.zeros(self.n).index_add_(0, post, count).clamp_min(1.0)
        w0 = count / in_strength[post]

        self.register_buffer("pre", pre)
        self.register_buffer("post", post)
        self.register_buffer("sign", torch.as_tensor(c.sign, dtype=torch.float32))
        self.register_buffer("in_idx", torch.as_tensor(c.indices(INPUT), dtype=torch.long))
        self.register_buffer("out_idx", torch.as_tensor(c.indices(OUTPUT), dtype=torch.long))

        self.log_w = nn.Parameter(w0.log(), requires_grad=learn_weights)
        self.log_gain = nn.Parameter(torch.tensor(float(gain)).log())
        self.log_tau = nn.Parameter(torch.full((self.n,), 2.0).log())
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.inp = nn.Linear(n_features, len(self.in_idx))

    @property
    def n_out(self) -> int:
        return len(self.out_idx)

    def weight_matrix(self) -> torch.Tensor:
        """Dense (post, pre) matrix. Dense is fastest for Tier S sizes on CPU and MPS."""
        w = self.sign[self.pre] * self.log_w.exp() * self.log_gain.exp()
        W = torch.zeros(self.n, self.n, device=w.device, dtype=w.dtype)
        return W.index_put((self.post, self.pre), w)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features (B, F) -> (output-neuron rates (B, n_out), all rates (B, N))."""
        B = features.shape[0]
        Wt = self.weight_matrix().T
        drive = torch.zeros(B, self.n, device=features.device, dtype=features.dtype)
        drive = drive.index_copy(1, self.in_idx, self.inp(features)) + self.bias
        alpha = 1.0 / (1.0 + self.log_tau.exp())  # dt=1, keeps alpha in (0, 1)
        v = torch.zeros_like(drive)
        r = torch.zeros_like(drive)
        for _ in range(self.steps):
            v = v + alpha * (-v + r @ Wt + drive)
            r = torch.tanh(torch.relu(v))
        return r[:, self.out_idx], r
