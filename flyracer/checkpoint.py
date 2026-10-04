"""Devices, connectome loading, and checkpoint save/load shared by training and tools."""

from __future__ import annotations

from pathlib import Path

import torch

from .connectome import Connectome, make_synthetic
from .models import Agent


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_connectome(path: str | None, core: str) -> Connectome | None:
    if core == "mlp":
        return None
    if path:
        return Connectome.load(path)
    print("no --connectome given: using a SYNTHETIC Tier S graph (not fly wiring)")
    return make_synthetic()


def save_checkpoint(path: Path, agent: Agent, args, step: int, connectome_path: str | None):
    torch.save(
        {"state_dict": agent.state_dict(), "args": vars(args), "step": step, "connectome": connectome_path},
        path,
    )


def load_agent(ckpt_path: str | Path, device="cpu") -> tuple[Agent, Connectome | None, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    conn = load_connectome(ckpt["connectome"], args["core"])
    agent = Agent(args["core"], conn, steps=args["rnn_steps"], seed=args["seed"]).to(device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent, conn, ckpt
