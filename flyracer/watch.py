"""Watch the fly brain drive, live and/or recorded as a 9:16 reel.

    python -m flyracer.watch --ckpt runs/fly-1/best.pt --live
    python -m flyracer.watch --ckpt runs/fly-1/best.pt --record fly.mp4
    # learning-progress montage: one clip per checkpoint, labeled with its training step
    python -m flyracer.watch --ckpt runs/fly-1/ckpt_*.pt --record progress.mp4 --max-seconds 8
    # untrained brain, no checkpoint needed
    python -m flyracer.watch --untrained --record untrained.mp4 --max-seconds 10
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .checkpoint import load_agent, load_connectome
from .connectome import Connectome
from .envs import make_env
from .models import Agent
from .models.policy import build_connectome

FPS = 50  # CarRacing physics rate


def describe(conn: Connectome | None, core: str) -> str:
    if conn is None:
        return "MLP baseline (no connectome)"
    src = conn.meta.get("dataset", "") if conn.meta.get("source") == "neuprint" else "synthetic connectome"
    label = {"random": "  |  RANDOM WIRING control", "signs": "  |  SHUFFLED SIGNS control"}.get(core, "")
    return f"{src}  |  {conn.n_neurons:,} neurons  |  {conn.n_edges:,} connections{label}"


def run_clip(agent, dash, env_seed, stage, max_seconds, writer, window, clock, speed, deterministic,
             action_repeat):
    import pygame

    env = make_env(env_seed, action_repeat, render_mode="rgb_array")()
    obs, _ = env.reset(seed=env_seed)
    score, t, done = 0.0, 0, False
    while not done and t < max_seconds * FPS:
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs)[None], device=next(agent.parameters()).device)
            action, *_, rates = agent(o, deterministic=deterministic)
        a = agent.env_action(action)[0].cpu().numpy()
        r = rates[0].cpu().numpy()
        obs, reward, term, trunc, _ = env.step(a)
        score += reward
        done = term or trunc
        for frame in env.get_wrapper_attr("frames"):
            t += 1
            img = dash.render(frame, r, np.clip(a, [-1, 0, 0], 1), {"score": score, "t": t, "stage": stage})
            if writer is not None:
                writer.append_data(img)
            if window is not None:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        env.close()
                        return score, True
                window.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                clock.tick(FPS * speed)
    env.close()
    return score, False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[], help="checkpoint(s); several = progress montage")
    p.add_argument("--untrained", action="store_true", help="random-init agent (use with --core/--connectome)")
    p.add_argument("--core", default="fly")
    p.add_argument("--connectome", help=".npz (default: synthetic) for --untrained")
    p.add_argument("--record", help="output .mp4 path")
    p.add_argument("--live", action="store_true", help="show a window")
    p.add_argument("--speed", type=float, default=1.0, help="live playback speed")
    p.add_argument("--episodes", type=int, default=1, help="episodes per checkpoint")
    p.add_argument("--max-seconds", type=float, default=20.0, help="cap per clip")
    p.add_argument("--seed", type=int, default=7, help="track seed")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--action-repeat", type=int, help="frames per decision (default: the checkpoint's; 2 for --untrained)")
    args = p.parse_args(argv)
    if not (args.record or args.live):
        p.error("use --record out.mp4 and/or --live")
    if not (args.ckpt or args.untrained):
        p.error("give --ckpt or --untrained")
    if not args.live:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import imageio.v2 as imageio
    import pygame

    from .viz.dashboard import H, W, Dashboard

    clips = []
    if args.untrained:
        conn = load_connectome(args.connectome, args.core)
        agent = Agent(args.core, conn, seed=0).eval()
        clips.append((agent, conn, args.core, 0, "UNTRAINED", args.action_repeat or 2))
    for path in args.ckpt:
        agent, conn, ckpt = load_agent(path)
        clips.append((agent, conn, ckpt["args"]["core"], ckpt["args"]["seed"], f"TRAINED {ckpt['step']:,} steps",
                      args.action_repeat or ckpt["args"].get("action_repeat", 4)))

    writer = imageio.get_writer(args.record, fps=FPS, codec="libx264", quality=7) if args.record else None
    window = None
    if args.live:
        pygame.init()
        window = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("fly-racer")
    clock = pygame.time.Clock()

    dash, key = None, None
    try:
        for agent, conn, core, seed, stage, repeat in clips:
            # Draw the wiring the agent actually uses (shuffled for control cores).
            shown = build_connectome(core, conn, seed) if conn is not None else None
            n_units = conn.n_neurons if conn is not None else agent.core.n_out
            if dash is None or key != (conn.meta.get("dataset") if conn else None, core, seed):
                dash = Dashboard(shown, n_units, describe(conn, core))
                key = (conn.meta.get("dataset") if conn else None, core, seed)
            for ep in range(args.episodes):
                score, quit_ = run_clip(agent, dash, args.seed + ep, stage, args.max_seconds, writer,
                                        window, clock, args.speed, not args.stochastic, repeat)
                print(f"{stage:>24}  episode {ep}  score {score:7.1f}")
                if quit_:
                    return
    finally:
        if writer is not None:
            writer.close()
            print(f"saved {args.record}")


if __name__ == "__main__":
    main()
