"""Score a checkpoint on full CarRacing episodes (no early stop, deterministic actions).

    python -m flyracer.evaluate runs/fly-1/best.pt --episodes 24

CarRacing counts as solved at an average score of 900 over 100 episodes.
Tracks use seeds 10_000+ by default, so they differ from the training tracks.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .checkpoint import load_agent, pick_device
from .envs import make_vec_env


def evaluate(agent, episodes: int, seed: int = 10_000, action_repeat: int = 4, num_envs: int = 8) -> list[float]:
    device = next(agent.parameters()).device
    n = min(num_envs, episodes)
    envs = make_vec_env(n, seed, action_repeat, patience=None, normalize_reward=False)
    obs, _ = envs.reset(seed=seed)
    scores, running = [], np.zeros(n)
    started = np.ones(n, dtype=bool)  # envs past `episodes` stop counting
    launched = n
    while len(scores) < episodes:
        with torch.no_grad():
            action, *_ = agent(torch.as_tensor(obs, device=device), deterministic=True)
        obs, reward, term, trunc, _ = envs.step(agent.env_action(action).cpu().numpy())
        running += reward
        for i in np.flatnonzero(term | trunc):
            if started[i]:
                scores.append(float(running[i]))
                if launched < episodes:
                    launched += 1
                else:
                    started[i] = False
            running[i] = 0.0
    envs.close()
    return scores[:episodes]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", help="write scores to this file")
    args = p.parse_args(argv)
    agent, _, ckpt = load_agent(args.ckpt, pick_device(args.device))
    scores = evaluate(agent, args.episodes, args.seed, ckpt["args"]["action_repeat"])
    print(f"{args.ckpt}: core={ckpt['args']['core']} step={ckpt['step']:,} "
          f"score {np.mean(scores):.1f} +/- {np.std(scores):.1f} "
          f"(min {np.min(scores):.1f}, max {np.max(scores):.1f}, n={len(scores)})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ckpt": args.ckpt, "step": ckpt["step"], "scores": scores}, f)


if __name__ == "__main__":
    main()
