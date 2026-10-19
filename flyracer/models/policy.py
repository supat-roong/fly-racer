"""Actor-critic for CarRacing: retina-like CNN encoder -> brain core -> motor readout.

Cores:
  fly     FlyRNN wired by the connectome
  random  FlyRNN wired by a degree-preserving shuffle of the same connectome
  signs   FlyRNN with the real wiring but shuffled inhibitory/excitatory labels
  mlp     plain MLP (no connectome), the standard deep-RL reference

The critic reads the encoder features directly: only the actor is "the fly".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

from ..connectome import OUTPUT, Connectome, degree_preserving_shuffle, shuffle_signs
from .flyrnn import FlyRNN

CORES = ("fly", "random", "signs", "mlp")
N_FEATURES = 256
ACTIONS = 3  # steer, gas, brake


def _init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    """Nature-DQN CNN over a stack of 4 grayscale 96x96 frames."""

    def __init__(self, in_frames: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            _init(nn.Conv2d(in_frames, 32, 8, stride=4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(64 * 8 * 8, N_FEATURES)), nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float() / 255.0)


class MLPCore(nn.Module):
    def __init__(self, n_features: int, hidden: int = 512, n_out: int = 128):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(
            _init(nn.Linear(n_features, hidden)), nn.Tanh(),
            _init(nn.Linear(hidden, n_out)), nn.Tanh(),
        )

    def forward(self, features):
        out = self.net(features)
        return out, out


def build_connectome(core: str, c: Connectome, seed: int = 0) -> Connectome:
    if core == "random":
        return degree_preserving_shuffle(c, seed)
    if core == "signs":
        return shuffle_signs(c, seed)
    return c


class Agent(nn.Module):
    """Each action u in [0, 1] comes from Beta(a, b) with a, b > 1. A bounded
    distribution wastes no probability on values the car would clip, which makes
    driving far more reliable than a clipped Gaussian. The env action is
    steer = 2u - 1, gas = u, brake = u.

    `forward` works in [0, 1]; use `env_action` before env.step.
    """

    def __init__(self, core: str, connectome: Connectome | None = None, steps: int = 6, seed: int = 0):
        super().__init__()
        if core not in CORES:
            raise ValueError(f"core must be one of {CORES}")
        self.core_name = core
        self.encoder = Encoder()
        if core == "mlp":
            self.core = MLPCore(N_FEATURES)
        else:
            if connectome is None:
                raise ValueError(f"core={core!r} needs a connectome")
            self.core = FlyRNN(build_connectome(core, connectome, seed), N_FEATURES, steps=steps)
        # Head outputs (alpha steer, gas, brake, beta steer, gas, brake).
        self.head = _init(nn.Linear(self.core.n_out, 2 * ACTIONS), std=0.01)
        with torch.no_grad():
            # Start with gas at mean ~0.58 and a light brake (mean ~0.2).
            self.head.bias.copy_(torch.tensor([0.0, 1.0, -3.0, 0.0, 0.0, 3.0]))
        if core != "mlp":
            self._steering_prior(connectome)
        self.critic = nn.Sequential(
            _init(nn.Linear(N_FEATURES, 256)), nn.Tanh(), _init(nn.Linear(256, 1), std=1.0)
        )

    @torch.no_grad()
    def _steering_prior(self, c: Connectome):
        """Right-side descending neurons steer right, left-side ones steer left,
        like a fly turning by driving one side harder. Learning can override it."""
        side = torch.as_tensor(c.side[c.group == OUTPUT], dtype=torch.float32)
        if side.abs().sum() > 0:
            w = 0.5 * side / side.abs().sum().clamp_min(1)
            self.head.weight[0] += w        # steer alpha: pushes u toward 1 (right)
            self.head.weight[ACTIONS] -= w  # steer beta

    @staticmethod
    def env_action(u: torch.Tensor) -> torch.Tensor:
        """[0, 1] actions -> CarRacing [steer, gas, brake]."""
        return torch.stack([2 * u[..., 0] - 1, u[..., 1], u[..., 2]], dim=-1)

    def features(self, obs):
        return self.encoder(obs)

    def value(self, obs):
        return self.critic(self.features(obs)).squeeze(-1)

    def dist_value(self, obs):
        """Beta parameters (alpha, beta) per action and the state value, one pass over obs."""
        f = self.features(obs)
        out, rates = self.core(f)
        a, b = (nn.functional.softplus(x) + 1.0 for x in self.head(out).split(ACTIONS, dim=-1))
        return a, b, self.critic(f).squeeze(-1), rates

    def forward(self, obs, action=None, deterministic=False):
        """Returns action in [0, 1], log_prob, entropy, value, all neuron rates."""
        a, b, value, rates = self.dist_value(obs)
        dist = Beta(a, b)
        if action is None:
            action = a / (a + b) if deterministic else dist.sample()
        logp = dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)
        return action, logp, dist.entropy().sum(-1), value, rates
