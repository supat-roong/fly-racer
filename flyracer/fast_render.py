"""CarRacing-v3 with a faster frame renderer, same pictures.

Stock CarRacing spends most of a step drawing: every frame it transforms all
~800 grass and road polygons one point at a time in Python to cull the ones
off screen, and reloads the font from disk to print the score. Here the
transform and culling run vectorized in numpy and the font and score text are
cached; the visible polygons are drawn with the same pygame calls, in the same
order, at the same coordinates (up to float rounding in the last bit), so the
96x96 observations match the stock renderer's.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
from gymnasium.envs.box2d import car_racing as cr
from pygame import gfxdraw

_LO = -cr.MAX_SHAPE_DIM
_HI_X = cr.WINDOW_W + cr.MAX_SHAPE_DIM
_HI_Y = cr.WINDOW_H + cr.MAX_SHAPE_DIM


def _grass_polys() -> np.ndarray:
    g = cr.GRASS_DIM
    return np.array([[(g * x + g, g * y), (g * x, g * y), (g * x, g * y + g), (g * x + g, g * y + g)]
                     for x in range(-20, 20, 2) for y in range(-20, 20, 2)], dtype=np.float64)


class FastCarRacing(cr.CarRacing):
    """Set `skip_render` to step without drawing the observation (the previous
    one is returned): an action-repeat wrapper only shows the agent the last of
    its frames, and drawing has no effect on the physics or the score."""

    _GRASS = _grass_polys()
    _font = None
    skip_render = False

    def reset(self, *args, **kwargs):
        self._road_cache = None
        skip, self.skip_render = self.skip_render, False
        try:
            return super().reset(*args, **kwargs)
        finally:
            self.skip_render = skip

    def render_state(self) -> np.ndarray:
        """Draw the observation for the current state (after skipped frames)."""
        self.state = self._render("state_pixels")
        return self.state

    def _road_arrays(self):
        if self._road_cache is None or self._road_cache[0] is not self.road_poly:
            pts = np.array([[(p[0], p[1]) for p in poly] for poly, _ in self.road_poly], dtype=np.float64)
            # The live color arrays, not copies: a tile is recolored in place when the car visits it.
            colors = [color for _, color in self.road_poly]
            self._road_cache = (self.road_poly, pts, colors)
        return self._road_cache[1], self._road_cache[2]

    @staticmethod
    def _transform(polys: np.ndarray, zoom, translation, angle) -> np.ndarray:
        # pygame's Vector2.rotate_rad: angle wrapped to [0, 2pi), then the usual rotation.
        a = math.fmod(angle, 2 * math.pi)
        if a < 0:
            a += 2 * math.pi
        c, s = math.cos(a), math.sin(a)
        x, y = polys[..., 0], polys[..., 1]
        out = np.empty_like(polys)
        out[..., 0] = (c * x - s * y) * zoom + translation[0]
        out[..., 1] = (s * x + c * y) * zoom + translation[1]
        return out

    def _draw_batch(self, polys, colors, zoom, translation, angle):
        pts = self._transform(polys, zoom, translation, angle)
        inside = ((pts[..., 0] >= _LO) & (pts[..., 0] <= _HI_X) & (pts[..., 1] >= _LO) & (pts[..., 1] <= _HI_Y))
        for i in np.flatnonzero(inside.any(axis=1)):
            poly = pts[i].tolist()
            color = colors if isinstance(colors, tuple) else [int(c) for c in colors[i]]
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _render_road(self, zoom, translation, angle):
        b = cr.PLAYFIELD
        field = [(b, b), (b, -b), (-b, -b), (-b, b)]
        self._draw_colored_polygon(self.surf, field, self.bg_color, zoom, translation, angle, clip=False)
        grass_color = tuple(int(c) for c in self.grass_color)
        self._draw_batch(self._GRASS, grass_color, zoom, translation, angle)
        road, colors = self._road_arrays()
        if len(road):
            self._draw_batch(road, colors, zoom, translation, angle)

    def _render(self, mode: str):
        # Stock _render, except the font is loaded once instead of every frame.
        if mode != "state_pixels" or "t" not in self.__dict__:
            return super()._render(mode)
        if self.skip_render:
            return self.state
        pygame.font.init()
        self.surf = pygame.Surface((cr.WINDOW_W, cr.WINDOW_H))
        angle = -self.car.hull.angle
        zoom = 0.1 * cr.SCALE * max(1 - self.t, 0) + cr.ZOOM * cr.SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (cr.WINDOW_W / 2 + trans[0], cr.WINDOW_H / 4 + trans[1])
        self._render_road(zoom, trans, angle)
        self.car.draw(self.surf, zoom, trans, angle, False)
        self.surf = pygame.transform.flip(self.surf, False, True)
        self._render_indicators(cr.WINDOW_W, cr.WINDOW_H)
        if FastCarRacing._font is None:
            FastCarRacing._font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = FastCarRacing._font.render(f"{self.reward:04.0f}", True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, cr.WINDOW_H - cr.WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)
        return self._create_image_array(self.surf, (cr.STATE_W, cr.STATE_H))
