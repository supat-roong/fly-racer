"""Cartoon fruit fly at the controls: front legs on the wheel, middle and hind legs on the pedals."""

from __future__ import annotations

import numpy as np
import pygame

# Design box in local units; drawn at SS x supersampling, then smoothscaled down.
BOX_W, BOX_H = 340, 280
SS = 2
FLOOR = 262

CUTICLE = (196, 146, 62)
CUTICLE_DARK = (150, 106, 42)
STRIPE = (110, 72, 30)
LEG = (150, 105, 40)
LEG_FAR = (88, 64, 30)
EYE = (205, 30, 34)
WHEEL = (150, 150, 165)
GAS_ON, BRAKE_ON = (90, 230, 120), (255, 90, 90)

WHEEL_C, WHEEL_R = np.array([275.0, 118.0]), 42.0
WHEEL_TURN = np.radians(100)  # wheel rotation at full steer
GRIPS = np.radians([150.0, 210.0])  # where the front legs hold the rim at steer 0 (y-up angles)
GAS_PIVOT, BRAKE_PIVOT = np.array([258.0, FLOOR]), np.array([205.0, FLOOR])
PEDAL_ARM = 46.0


def _knee(hip, foot, l1, l2, up=True):
    """Two-bone IK: knee position for a leg of segment lengths l1, l2 reaching from hip to foot."""
    d_vec = foot - hip
    d = float(np.clip(np.linalg.norm(d_vec), abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3))
    u = d_vec / max(np.linalg.norm(d_vec), 1e-6)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))
    base = hip + a * u
    perp = np.array([u[1], -u[0]])  # points up on screen when the leg reaches rightward
    knee = base + h * perp if (perp[1] < 0) == up else base - h * perp
    return knee, hip + u * d  # foot clamped to reach


def _ellipse_pts(c, rx, ry, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)], 1)


class Driver:
    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.scale = min(size[0] / BOX_W, size[1] / BOX_H) * SS
        self.canvas = pygame.Surface((int(BOX_W * self.scale), int(BOX_H * self.scale)), pygame.SRCALPHA)
        self.wing = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)

    def _p(self, xy):
        return (float(xy[0]) * self.scale, float(xy[1]) * self.scale)

    def _w(self, w):
        return max(1, int(round(w * self.scale)))

    def _line(self, a, b, color, w):
        pygame.draw.line(self.canvas, color, self._p(a), self._p(b), self._w(w))
        pygame.draw.circle(self.canvas, color, self._p(b), self._w(w) / 2)

    def _leg(self, hip, foot, l1, l2, color, up=True):
        hip, foot = np.asarray(hip, float), np.asarray(foot, float)
        knee, foot = _knee(hip, foot, l1, l2, up)
        self._line(hip, knee, color, 6)
        self._line(knee, foot, color, 4.5)
        pygame.draw.circle(self.canvas, color, self._p(knee), self._w(4))
        pygame.draw.circle(self.canvas, (50, 36, 18), self._p(foot), self._w(3.5))

    def _pedal(self, pivot, press, color):
        ang = np.radians(18 + 30 * press)
        top = pivot + PEDAL_ARM * np.array([np.sin(ang), -np.cos(ang)])
        normal = np.array([np.cos(ang), np.sin(ang)])
        self._line(pivot, top, (70, 70, 82), 4)
        lit = [int(v * (0.45 + 0.55 * press)) for v in color]
        self._line(top - 13 * normal, top + 13 * normal, lit, 9)
        return top + np.array([-np.sin(ang), np.cos(ang)]) * -5  # foot rests just above the pad

    def render(self, steer: float, gas: float, brake: float, t: int) -> pygame.Surface:
        c = self.canvas
        c.fill((0, 0, 0, 0))
        bob = 1.2 * np.sin(t * 0.35)
        body = np.array([0.0, bob])

        pygame.draw.line(c, (40, 40, 52), self._p((10, FLOOR + 2)), self._p((BOX_W - 10, FLOOR + 2)), self._w(2))

        # Far-side middle and hind legs rest on the floor.
        self._leg(np.array([170, 140]) + body, (140, FLOOR), 62, 70, LEG_FAR)
        self._leg(np.array([152, 140]) + body, (88, FLOOR), 60, 68, LEG_FAR)

        # Steering column and pedals sit behind the near legs.
        self._line(WHEEL_C, (318, FLOOR), (60, 60, 72), 7)
        gas_foot = self._pedal(GAS_PIVOT, gas, GAS_ON)
        brake_foot = self._pedal(BRAKE_PIVOT, brake, BRAKE_ON)

        # Abdomen with dark tergite bands.
        ab_c, ab_rx, ab_ry = np.array([95, 118]) + body, 58, 36
        pygame.draw.polygon(c, CUTICLE, [self._p(q) for q in _ellipse_pts(ab_c, ab_rx, ab_ry)])
        for x0 in (-44, -22, 0, 22):
            xs = np.linspace(x0, x0 + 10, 6)
            hy = ab_ry * np.sqrt(np.clip(1 - (xs / ab_rx) ** 2, 0, 1))
            band = [(ab_c[0] + x, ab_c[1] - y) for x, y in zip(xs, hy)]
            band += [(ab_c[0] + x, ab_c[1] + y) for x, y in zip(xs[::-1], hy[::-1])]
            pygame.draw.polygon(c, STRIPE, [self._p(q) for q in band])

        # Wings, buzzing harder with more gas.
        self.wing.fill((0, 0, 0, 0))
        tilt = np.radians(-14 + (2 + 6 * gas) * np.sin(t * 2.7))
        root = np.array([150, 92]) + body
        pts = _ellipse_pts(np.array([-72, 0]), 78, 20)
        rot = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
        pts = pts @ rot.T + root
        pygame.draw.polygon(self.wing, (200, 225, 255, 90), [self._p(q) for q in pts])
        pygame.draw.polygon(self.wing, (220, 235, 255, 150), [self._p(q) for q in pts], self._w(1.5))
        c.blit(self.wing, (0, 0))

        # Thorax with bristles, then head, eye, antenna, proboscis.
        th_c = np.array([160, 108]) + body
        pygame.draw.polygon(c, CUTICLE_DARK, [self._p(q) for q in _ellipse_pts(th_c, 40, 35)])
        pygame.draw.polygon(c, CUTICLE, [self._p(q) for q in _ellipse_pts(th_c + (0, 3), 36, 30)])
        for k in range(6):
            a = np.radians(200 + 26 * k)
            base = th_c + 34 * np.array([np.cos(a), np.sin(a)])
            self._line(base, base + 9 * np.array([np.cos(a), np.sin(a)]), (40, 28, 14), 1.5)
        head = np.array([212, 96]) + body
        self._line(head + (-4, 18), head + (2, 36), CUTICLE_DARK, 4)
        pygame.draw.circle(c, CUTICLE, self._p(head), self._w(24))
        self._line(head + (12, -20), head + (22, -32), STRIPE, 2.5)
        eye = head + (6, -2)
        pygame.draw.circle(c, EYE, self._p(eye), self._w(19))
        pygame.draw.circle(c, (150, 15, 20), self._p(eye), self._w(19), self._w(2))
        pygame.draw.circle(c, (255, 170, 170), self._p(eye + (-6, -7)), self._w(4))

        # Steering wheel, rotating with the steer action (clockwise = right).
        turn = -float(np.clip(steer, -1, 1)) * WHEEL_TURN
        pygame.draw.circle(c, WHEEL, self._p(WHEEL_C), self._w(WHEEL_R), self._w(6))
        for a in np.radians([90, 210, 330]) + turn:
            self._line(WHEEL_C, WHEEL_C + WHEEL_R * np.array([np.cos(a), -np.sin(a)]), WHEEL, 4)
        pygame.draw.circle(c, (90, 90, 104), self._p(WHEEL_C), self._w(8))
        marker = WHEEL_C + WHEEL_R * np.array([np.cos(np.pi / 2 + turn), -np.sin(np.pi / 2 + turn)])
        pygame.draw.circle(c, (255, 170, 40), self._p(marker), self._w(5))

        # Near legs on top: front pair grips the rim, middle on gas, hind on brake.
        for g in GRIPS + turn:
            grip = WHEEL_C + WHEEL_R * np.array([np.cos(g), -np.sin(g)])
            self._leg(np.array([188, 132]) + body, grip, 62, 62, LEG, up=False)
        self._leg(np.array([166, 140]) + body, gas_foot, 88, 88, LEG)
        self._leg(np.array([150, 140]) + body, brake_foot, 76, 76, LEG)

        return pygame.transform.smoothscale(c, (c.get_width() // SS, c.get_height() // SS))
