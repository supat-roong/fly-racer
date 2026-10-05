"""Race up to 4 checkpoints on the same track, side by side in a 2x2 grid video.

    python -m flyracer.race --ckpt runs/fly-1/best.pt runs/random-1/best.pt \
        runs/signs-1/best.pt runs/mlp-1/best.pt --seed 30037 --record race.mp4

Every car drives a full episode (deterministic actions). A car that finishes its
lap or runs out of time freezes on its last frame with its final score.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .checkpoint import load_agent
from .envs import make_env

FPS = 50  # CarRacing physics rate
W = H = 1080
CELL = W // 2
TITLE_H, LABEL_H, STATS_H = 90, 52, 44
GAME_W, GAME_H = CELL - 24, (CELL - 24) * 2 // 3

BG = (8, 8, 14)
TEXT = (235, 235, 245)
MUTED = (140, 140, 160)
CORE_STYLE = {
    "fly": ("FLY CONNECTOME", "real MaleCNS wiring", (255, 170, 40)),
    "random": ("RANDOM WIRING", "degree-preserving shuffle", (70, 200, 255)),
    "signs": ("SHUFFLED SIGNS", "real wiring, shuffled E/I", (255, 80, 200)),
    "mlp": ("PLAIN MLP", "no connectome", (90, 230, 120)),
}


def crop_bar(frame: np.ndarray) -> np.ndarray:
    """Drop CarRacing's indicator bar (bottom 1/8) and trim the sides back to 3:2."""
    h, w = frame.shape[:2]
    h = h * 7 // 8
    cw = min(w, h * 3 // 2)
    return frame[:h, (w - cw) // 2:(w + cw) // 2]


class Racer:
    def __init__(self, path: str, seed: int, action_repeat: int | None):
        self.agent, _, ckpt = load_agent(path)
        self.core = ckpt["args"]["core"]
        self.device = next(self.agent.parameters()).device
        self.env = make_env(seed, action_repeat or ckpt["args"].get("action_repeat", 2), render_mode="rgb_array")()
        self.obs, _ = self.env.reset(seed=seed)
        self.queue: list[np.ndarray] = []  # rendered frames not yet shown
        self.frame = crop_bar(self.env.render())
        self.score, self.t, self.done, self.finished = 0.0, 0, False, False

    def next_frame(self):
        """Advance one 50 fps frame, stepping the policy whenever the queued frames run out."""
        if self.done and not self.queue:
            return
        if not self.queue:
            with torch.no_grad():
                o = torch.as_tensor(np.asarray(self.obs)[None], device=self.device)
                action, *_ = self.agent(o, deterministic=True)
            self.obs, reward, term, trunc, _ = self.env.step(self.agent.env_action(action)[0].cpu().numpy())
            self.score += reward
            self.done, self.finished = term or trunc, term
            self.queue = list(self.env.get_wrapper_attr("frames"))
        self.frame = crop_bar(self.queue.pop(0))
        self.t += 1


def draw(surface, fonts, racers, seed):
    import pygame

    big, mid, small = fonts
    surface.fill(BG)
    title = big.render("SAME TRACK, FOUR BRAINS", True, TEXT)
    surface.blit(title, title.get_rect(midtop=(W // 2, 18)))
    sub = small.render(f"CarRacing-v3  |  held-out track {seed}  |  same PPO recipe, 1.5M decisions each", True, MUTED)
    surface.blit(sub, sub.get_rect(midtop=(W // 2, 62)))
    cell_h = (H - TITLE_H) // 2
    for i, r in enumerate(racers):
        x0, y0 = (i % 2) * CELL + 12, TITLE_H + (i // 2) * cell_h
        name, desc, color = CORE_STYLE.get(r.core, (r.core.upper(), "", TEXT))
        pygame.draw.circle(surface, color, (x0 + 8, y0 + 18), 7)
        surface.blit(mid.render(name, True, TEXT), (x0 + 24, y0 + 4))
        surface.blit(small.render(desc, True, MUTED), (x0 + 24, y0 + 30))
        game = pygame.surfarray.make_surface(np.ascontiguousarray(r.frame.swapaxes(0, 1)))
        surface.blit(pygame.transform.smoothscale(game, (GAME_W, GAME_H)), (x0, y0 + LABEL_H))
        pygame.draw.rect(surface, color, (x0, y0 + LABEL_H, GAME_W, GAME_H), 3)
        ys = y0 + LABEL_H + GAME_H + 10
        surface.blit(mid.render(f"score {r.score:6.1f}", True, TEXT), (x0, ys))
        clock = mid.render(f"t {r.t / FPS:4.1f}s", True, TEXT)
        surface.blit(clock, clock.get_rect(topright=(x0 + GAME_W, ys)))
        if r.done and not r.queue:
            badge = "LAP DONE" if r.finished else "TIME UP"
            text = mid.render(f"{badge}  {r.score:.1f}", True, BG)
            pad = text.get_rect(center=(x0 + GAME_W // 2, y0 + LABEL_H + GAME_H // 2)).inflate(28, 16)
            pygame.draw.rect(surface, color, pad, border_radius=10)
            surface.blit(text, text.get_rect(center=pad.center))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="up to 4 checkpoints, one per grid cell")
    p.add_argument("--seed", type=int, default=30037, help="track seed")
    p.add_argument("--record", required=True, help="output .mp4 path")
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold the final frame")
    p.add_argument("--action-repeat", type=int, help="frames per decision (default: each checkpoint's)")
    args = p.parse_args(argv)
    if len(args.ckpt) > 4:
        p.error("at most 4 checkpoints")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import imageio.v2 as imageio
    import pygame

    pygame.init()
    surface = pygame.Surface((W, H))
    fonts = tuple(pygame.font.SysFont("helvetica", s, bold=b) for s, b in ((38, True), (24, True), (17, False)))
    racers = [Racer(path, args.seed, args.action_repeat) for path in args.ckpt]
    writer = imageio.get_writer(args.record, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    try:
        while not all(r.done and not r.queue for r in racers):
            for r in racers:
                r.next_frame()
            draw(surface, fonts, racers, args.seed)
            writer.append_data(pygame.surfarray.array3d(surface).swapaxes(0, 1))
        for _ in range(int(args.hold * FPS)):
            writer.append_data(pygame.surfarray.array3d(surface).swapaxes(0, 1))
    finally:
        writer.close()
        for r in racers:
            r.env.close()
    for r in racers:
        print(f"{r.core:>7}  score {r.score:7.1f}  {r.t / FPS:5.1f}s  {'lap done' if r.finished else 'time up'}")
    print(f"saved {args.record}")


if __name__ == "__main__":
    main()
