"""Reel-style (9:16) dashboard: game on top, glowing fly brain in the middle, fly driver and motor HUD below."""

from __future__ import annotations

import numpy as np
import pygame

from ..connectome import CORE, INPUT, OUTPUT, Connectome
from .driver import Driver

W, H = 720, 1280
HEADER, GAME_H, BRAIN_H = 110, 480, 400
BRAIN_Y = HEADER + GAME_H
HUD_Y = BRAIN_Y + BRAIN_H
DRIVER_W = 340

BG = (8, 8, 14)
PANEL = (14, 14, 24)
TEXT = (235, 235, 245)
MUTED = (140, 140, 160)
GROUP_COLOR = {INPUT: (70, 200, 255), CORE: (255, 80, 200), OUTPUT: (255, 170, 40)}
GROUP_LABEL = {INPUT: "VISUAL INPUT", CORE: "CENTRAL BRAIN", OUTPUT: "DESCENDING (MOTOR)"}
EXC_EDGE, INH_EDGE = (120, 255, 160), (90, 120, 255)


class Dashboard:
    def __init__(self, connectome: Connectome | None, n_units: int, subtitle: str, max_edges: int = 160):
        pygame.init()
        self.surface = pygame.Surface((W, H))
        self.glow = pygame.Surface((W, BRAIN_H))
        self.font_big = pygame.font.SysFont("helvetica", 34, bold=True)
        self.font = pygame.font.SysFont("helvetica", 22)
        self.font_small = pygame.font.SysFont("helvetica", 17)
        self.subtitle = subtitle
        self.max_edges = max_edges
        self.smoothed = np.zeros(n_units, dtype=np.float32)
        self.driver = Driver((DRIVER_W, H - HUD_Y - 10))

        if connectome is not None:
            self.group = connectome.group.astype(int)
            self.xy = self._layout(connectome.pos[:, :2])
            # Candidate edges for the "activity flow" lines: strongest connections only.
            top = np.argsort(-connectome.count)[:4000]
            self.e_pre, self.e_post = connectome.pre[top], connectome.post[top]
            self.e_w = connectome.count[top] / connectome.count[top].max()
            self.e_sign = connectome.sign[self.e_pre]
        else:  # MLP baseline: hidden units on a grid, no wiring to draw
            self.group = np.full(n_units, CORE)
            side = int(np.ceil(np.sqrt(n_units)))
            g = np.stack(np.divmod(np.arange(n_units), side), 1)[:, ::-1].astype(np.float32)
            self.xy = self._layout(g)
            self.e_pre = np.zeros(0, dtype=int)
        self.radius = np.where(self.group == OUTPUT, 4, 2)

    @staticmethod
    def _layout(xy: np.ndarray) -> np.ndarray:
        """Fit soma positions into the brain panel (front view), robust to outliers."""
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        pad = 40
        scale = min((W - 2 * pad) / span[0], (BRAIN_H - 2 * pad - 20) / span[1])
        center = (lo + hi) / 2
        out = (xy - center) * scale + np.array([W / 2, BRAIN_H / 2 + 10])
        return np.clip(out, 4, [W - 4, BRAIN_H - 4]).astype(np.int32)

    def _text(self, s, font, color, pos, anchor="topleft"):
        img = font.render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        self.surface.blit(img, rect)

    def _bar(self, label, value, y, centered=False, color=(255, 255, 255)):
        x0, w, h = DRIVER_W + 120, W - DRIVER_W - 160, 22
        self._text(label, self.font, MUTED, (DRIVER_W + 20, y + 1))
        pygame.draw.rect(self.surface, PANEL, (x0, y, w, h), border_radius=6)
        if centered:
            mid = x0 + w // 2
            px = int(value * w / 2)
            rect = (mid, y, px, h) if px >= 0 else (mid + px, y, -px, h)
            pygame.draw.line(self.surface, MUTED, (mid, y - 3), (mid, y + h + 3), 2)
        else:
            rect = (x0, y, int(np.clip(value, 0, 1) * w), h)
        pygame.draw.rect(self.surface, color, rect, border_radius=6)

    def draw_brain(self, rates: np.ndarray):
        self.smoothed = 0.6 * self.smoothed + 0.4 * rates
        r = self.smoothed
        # Brightness per group: motor neurons fire far less than sensory ones and
        # would otherwise look dark next to them.
        level = np.zeros_like(r)
        for g in GROUP_COLOR:
            m = self.group == g
            if m.any():
                level[m] = np.clip(r[m] / max(float(np.percentile(r[m], 99)), 1e-3), 0, 1)

        brain = self.surface.subsurface((0, BRAIN_Y, W, BRAIN_H))
        brain.fill(BG)
        self.glow.fill((0, 0, 0))

        # Activity flow: synapses whose presynaptic neuron is firing hardest right now.
        if len(self.e_pre) and self.max_edges:
            flow = level[self.e_pre] * self.e_w
            for k in np.argsort(-flow)[: self.max_edges]:
                if flow[k] < 0.02:
                    break
                c = EXC_EDGE if self.e_sign[k] > 0 else INH_EDGE
                a = min(1.0, 0.25 + flow[k])
                pygame.draw.aaline(self.glow, [int(v * a * 0.5) for v in c],
                                   self.xy[self.e_pre[k]], self.xy[self.e_post[k]])

        for g, base in GROUP_COLOR.items():
            idx = np.flatnonzero(self.group == g)
            for i in idx:
                lv = level[i]
                col = [int(v * (0.18 + 0.82 * lv)) for v in base]
                pygame.draw.circle(brain, col, self.xy[i], self.radius[i])
                if lv > 0.35:
                    halo = [int(v * lv * 0.35) for v in base]
                    pygame.draw.circle(self.glow, halo, self.xy[i], self.radius[i] * 4)
        brain.blit(self.glow, (0, 0), special_flags=pygame.BLEND_ADD)

        x = 16
        for g in (INPUT, CORE, OUTPUT):
            if (self.group == g).any():
                pygame.draw.circle(brain, GROUP_COLOR[g], (x + 6, 18), 6)
                img = self.font_small.render(GROUP_LABEL[g], True, TEXT)
                brain.blit(img, (x + 18, 9))
                x += img.get_width() + 44

    def render(self, frame: np.ndarray, rates: np.ndarray, action: np.ndarray, info: dict) -> np.ndarray:
        s = self.surface
        s.fill(BG)
        self._text("FRUIT FLY BRAIN LEARNS TO DRIVE", self.font_big, TEXT, (W // 2, 22), "midtop")
        self._text(self.subtitle, self.font_small, MUTED, (W // 2, 70), "midtop")

        game = pygame.surfarray.make_surface(np.ascontiguousarray(frame.swapaxes(0, 1)))
        s.blit(pygame.transform.smoothscale(game, (W, GAME_H)), (0, HEADER))
        stage = info.get("stage")
        if stage:
            tag = self.font.render(stage, True, (0, 0, 0))
            pad = tag.get_rect().inflate(20, 10)
            pad.topleft = (14, HEADER + 14)
            pygame.draw.rect(s, (255, 210, 60), pad, border_radius=8)
            s.blit(tag, tag.get_rect(center=pad.center))

        self.draw_brain(rates)

        steer, gas, brake = (float(v) for v in action)
        s.blit(self.driver.render(steer, gas, brake, info.get("t", 0)), (10, HUD_Y + 5))
        y = HUD_Y + 40
        self._bar("STEER", np.clip(steer, -1, 1), y, centered=True, color=(255, 170, 40))
        self._bar("GAS", gas, y + 40, color=(90, 230, 120))
        self._bar("BRAKE", brake, y + 80, color=(255, 90, 90))
        x = DRIVER_W + 20
        self._text(f"score {info.get('score', 0):7.1f}", self.font, TEXT, (x, y + 130))
        self._text(f"t {info.get('t', 0) / 50:5.1f}s", self.font, TEXT, (W - 40, y + 130), "topright")
        active = int((rates > 0.1).sum())
        self._text(f"{active:,} neurons firing", self.font, TEXT, (x, y + 165))
        return pygame.surfarray.array3d(s).swapaxes(0, 1)
