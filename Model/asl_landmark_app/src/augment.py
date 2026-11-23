"""Landmark-only augmentation of the prepared (N, 21, 3) hand: rotation, tilt,
scale, noise, landmark dropout and mirroring. No pixels, so it is cheap."""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize  # noqa: E402

DEFAULTS = {
    "rotate_deg": 15.0,
    "tilt_deg": 10.0,
    "scale_low": 0.9,
    "scale_high": 1.1,
    "noise_sigma": 0.01,
    "dropout_p": 0.30,        # fraction of examples that lose landmarks
    "dropout_k": 3,           # max landmarks dropped
    "mirror_p": 0.5,
}


def _rot_matrices(rng, n: int, max_deg: float) -> np.ndarray:
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    c, s = np.cos(ang), np.sin(ang)
    r = np.zeros((n, 3, 3), dtype=np.float64)
    r[:, 0, 0] = c
    r[:, 0, 1] = -s
    r[:, 1, 0] = s
    r[:, 1, 1] = c
    r[:, 2, 2] = 1.0
    return r


def _tilt_matrices(rng, n: int, max_deg: float) -> np.ndarray:
    ax = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    ay = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    rx = np.zeros((n, 3, 3))
    rx[:, 0, 0] = 1
    rx[:, 1, 1] = np.cos(ax); rx[:, 1, 2] = -np.sin(ax)
    rx[:, 2, 1] = np.sin(ax); rx[:, 2, 2] = np.cos(ax)
    ry = np.zeros((n, 3, 3))
    ry[:, 1, 1] = 1
    ry[:, 0, 0] = np.cos(ay); ry[:, 0, 2] = np.sin(ay)
    ry[:, 2, 0] = -np.sin(ay); ry[:, 2, 2] = np.cos(ay)
    return np.einsum("nij,njk->nik", rx, ry)


def augment_batch(lm: np.ndarray, rng: np.random.Generator,
                  strength: float = 1.0, cfg: dict | None = None) -> np.ndarray:
    """lm: (N,21,3) already prepared. Returns an augmented copy."""
    p = dict(DEFAULTS)
    if cfg:
        p.update(cfg)
    lm = np.asarray(lm, dtype=np.float64)
    n = lm.shape[0]
    if n == 0:
        return lm.copy()
    st = float(strength)

    r_in = _rot_matrices(rng, n, p["rotate_deg"] * st)
    r_tilt = _tilt_matrices(rng, n, p["tilt_deg"] * st)
    rot = np.einsum("nij,njk->nik", r_tilt, r_in)
    scale = rng.uniform(1 - (1 - p["scale_low"]) * st,
                        1 + (p["scale_high"] - 1) * st, size=(n, 1, 1))
    out = np.einsum("nij,nkj->nki", rot, lm) * scale
    out += rng.normal(0.0, p["noise_sigma"] * st, size=out.shape)

    # a dropped landmark is guessed from its own finger chain
    n_drop = int(round(p["dropout_p"] * st * n))
    if n_drop > 0:
        rows = rng.choice(n, size=n_drop, replace=False)
        for row in rows:
            k = int(rng.integers(1, p["dropout_k"] + 1))
            for idx in rng.choice(21, size=k, replace=False):
                chain = next((c for c in normalize.FINGER_CHAINS if idx in c), None)
                if chain is None:
                    out[row, idx] = out[row, 0]
                else:
                    out[row, idx] = out[row, list(chain)].mean(axis=0)
                out[row, idx] += rng.normal(0, 0.02, size=3)

    # x negated only: landmark ids are anatomical, see normalize.py
    n_mir = int(round(p["mirror_p"] * n))
    if n_mir > 0:
        mrows = rng.choice(n, size=n_mir, replace=False)
        mirrored = normalize.mirror_hand(out[mrows])
        out[mrows] = mirrored
    return out.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    base = np.tile(np.eye(3)[:3].astype(float), (256, 1, 1)).reshape(256, 21, 3)
    base[:] = rng.normal(0, 0.5, (256, 21, 3))
    aug = augment_batch(base, rng)
    print("in", base.shape, "-> out", aug.shape, "mean abs delta",
          float(np.abs(aug - base).mean()))
