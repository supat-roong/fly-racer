"""Devices, connectome loading, and checkpoint save/load shared by training and tools.

    python -m flyracer.checkpoint strip runs/fly-1/best.pt weights/fly-1.pt

`strip` drops the connectome wiring (edge lists, signs, input/output neurons) from a
checkpoint so the learned weights can be shared without redistributing the connectome;
`load_agent` rebuilds the wiring from your own .npz and checks it matches.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from .connectome import Connectome, make_synthetic
from .models import Agent

WIRING = ("core.pre", "core.post", "core.sign", "core.in_idx", "core.out_idx")  # rebuilt from the connectome


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


def wiring_hash(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in WIRING:
        h.update(state_dict[k].cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, agent: Agent, args, step: int, connectome_path: str | None):
    torch.save(
        {"state_dict": agent.state_dict(), "args": vars(args), "step": step, "connectome": connectome_path},
        path,
    )


def strip_wiring(src: str | Path, dst: str | Path):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if all(k in sd for k in WIRING):  # an mlp core has no wiring to strip
        ckpt["wiring_sha256"] = wiring_hash(sd)
        ckpt["state_dict"] = {k: v for k, v in sd.items() if k not in WIRING}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)


def load_agent(ckpt_path: str | Path, device="cpu") -> tuple[Agent, Connectome | None, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    path = ckpt["connectome"]
    if path and not Path(path).exists() and Path("data", Path(path).name).exists():
        path = str(Path("data", Path(path).name))  # trained elsewhere, e.g. /kaggle/input/...
    conn = load_connectome(path, args["core"])
    agent = Agent(args["core"], conn, steps=args["rnn_steps"], seed=args["seed"]).to(device)
    if "wiring_sha256" in ckpt:  # stripped: keep the wiring just rebuilt from the connectome
        if wiring_hash(agent.state_dict()) != ckpt["wiring_sha256"]:
            raise ValueError(f"{ckpt_path} was trained on a different connectome than {path}")
        sd = {**ckpt["state_dict"], **{k: agent.state_dict()[k] for k in WIRING}}
    else:
        sd = ckpt["state_dict"]
    agent.load_state_dict(sd)
    agent.eval()
    return agent, conn, ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip", help="remove the connectome wiring from a checkpoint")
    s.add_argument("src")
    s.add_argument("dst")
    a = p.parse_args()
    strip_wiring(a.src, a.dst)
    print(f"{a.src} -> {a.dst} ({Path(a.dst).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
