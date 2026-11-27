"""Temporal smoothing for the camera path.

MediaPipe hands jitter frame to frame, and the jitter is what makes the skeleton
look like it is vibrating and makes a held letter flicker between two readings.
This is a one-euro filter (low lag at speed, heavy smoothing at rest) per
landmark, plus a jump clamp so a detection glitch cannot teleport the hand.

Deliberately light: a strong filter would blunt the fast strokes that J and Z
are made of, so nothing here is applied to the *recognition* window beyond what
the eye sees - the same filtered landmarks feed both the features and the
overlay, which is the only way the two can agree.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9


class LandmarkFilter:
    """One-euro filter over (21,3) landmarks, one filter per coordinate.

    `min_cutoff` sets how much a still hand is smoothed (lower = calmer),
    `beta` how quickly smoothing is released as the hand speeds up, and
    `max_step` caps per-frame travel in palm units so a glitch cannot jump.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.04,
                 d_cutoff: float = 1.0, max_step: float = 1.6):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_step = float(max_step)
        self.reset()

    def reset(self) -> None:
        self.value: np.ndarray | None = None
        self.vel: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, lm, now: float) -> np.ndarray:
        pts = np.asarray(lm, dtype=np.float64).copy()
        if pts.shape != (21, 3):
            raise ValueError(f"expected (21,3) landmarks, got {pts.shape}")
        if self.value is None or self.t is None:
            self.value = pts.copy()
            self.vel = np.zeros_like(pts)
            self.t = float(now)
            return pts
        dt = float(now) - self.t
        if dt <= 1e-6:
            return self.value.copy()
        self.t = float(now)

        # palm size from the *previous* filtered frame: the scale the clamp and
        # the speed threshold are expressed in
        palm = float(np.linalg.norm(self.value[MIDDLE_MCP, :2] - self.value[WRIST, :2]))
        palm = palm if palm > 1e-6 else 1e-6

        raw_vel = (pts - self.value) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.vel = a_d * raw_vel + (1.0 - a_d) * self.vel
        speed = np.linalg.norm(self.vel[:, :2], axis=1)
        cutoff = self.min_cutoff + self.beta * (speed / palm)
        a = np.array([self._alpha(float(c), dt) for c in cutoff])[:, None]
        out = a * pts + (1.0 - a) * self.value

        # Glitch clamp: no landmark may travel more than max_step palms in one
        # frame. A real stroke peaks well below this at camera frame rates.
        delta = out - self.value
        limit = self.max_step * palm
        dist = np.linalg.norm(delta[:, :2], axis=1)
        over = dist > limit
        if np.any(over):
            scale = np.ones_like(dist)
            scale[over] = limit / dist[over]
            out = self.value + delta * scale[:, None]

        out[:, 2] = pts[:, 2]          # depth is noisy already; smooth xy only
        self.value = out
        return out.copy()


def smoothing_enabled() -> bool:
    """`ASL_SMOOTH=off` disables the filter without touching code."""
    import os
    return os.environ.get("ASL_SMOOTH", "").strip().lower() not in (
        "off", "0", "false", "no")
