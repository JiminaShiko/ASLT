"""J/Z window features - the one implementation shared by training and live.

A J or a Z is a stroke, not a shape, so no single frame can name it: the model
that knows them reads a *window* of about a second of canonicalised landmarks.
This module turns such a window into one vector. `train_jz.py` builds every
training example here and `live.py` builds every camera example here, so the
offline and live quantities cannot drift apart.

Everything is in palm units (wrist -> middle-MCP bone = 1), because that is the
only scale that survives a different camera distance.

Two frames of reference are kept on purpose, and v2 exists because of it: the
tip *relative to the wrist* names a stroke drawn with the finger alone, and the
tip *relative to where the wrist started* still names the same stroke when the
whole arm draws it. Measured on the MSL corpus, one signer draws Z by moving the
arm: with wrist-relative streams only, their Z looked motionless and their X
looked like a Z.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__:
    from . import normalize  # noqa: E402
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import normalize  # noqa: E402

# 1.2 s is the window the hand-written rule judged, kept so the two are
# comparable; STEPS resamples it onto a common time axis, which is what makes
# 30 fps recordings and a 15 fps webcam produce the same vector.
WINDOW_S = 1.20
STEPS = 24
FEATURE_VERSION = 2

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
INDEX_PIP = 6
INDEX_TIP = 8

# The resampled block: wrist position, tip relative to the wrist, tip relative
# to the window's starting wrist.
STREAMS = ("wrist_x", "wrist_y", "tip_x", "tip_y", "tip_x_from_start",
           "tip_y_from_start")

# Reversal counting: a direction change only counts once the stream has moved
# this far (palm units) away from its last turning point.
REV_SEGS = (0.05, 0.18)
REV_STREAMS = (0, 2, 3, 4)     # wrist x, tip x (rel), tip y (rel), tip x (abs)


def _series(lms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,21,3) landmarks -> (T,6) palm-unit streams, (T,) palm, (T,) idx_ext.

    Each frame is canonicalised on its own (mirrored hands are flipped, so the
    same physical stroke gives the same numbers from either hand), then scaled by
    that frame's palm size.
    """
    out = np.empty((len(lms), 6), dtype=np.float64)
    palms = np.empty(len(lms), dtype=np.float64)
    exts = np.empty(len(lms), dtype=np.float64)
    for i, lm in enumerate(lms):
        c = normalize.canonicalize(np.asarray(lm, dtype=np.float64))
        palm = float(np.linalg.norm(c[MIDDLE_MCP, :2] - c[WRIST, :2]))
        palm = palm if palm > 1e-6 else 1e-6
        palms[i] = palm
        out[i, 0] = c[WRIST, 0] / palm
        out[i, 1] = c[WRIST, 1] / palm
        out[i, 2] = (c[INDEX_TIP, 0] - c[WRIST, 0]) / palm
        out[i, 3] = (c[INDEX_TIP, 1] - c[WRIST, 1]) / palm
        out[i, 4] = c[INDEX_TIP, 0] / palm
        out[i, 5] = c[INDEX_TIP, 1] / palm
        a = float(np.linalg.norm(c[INDEX_PIP, :2] - c[INDEX_MCP, :2]))
        b = float(np.linalg.norm(c[INDEX_TIP, :2] - c[INDEX_MCP, :2]))
        exts[i] = b / max(a, 1e-6)
    # anchor the absolute streams to the window's first wrist, so a hand that
    # sits elsewhere in frame gives the same numbers
    out[:, 4] -= out[0, 0]
    out[:, 5] -= out[0, 1]
    return out, palms, exts


def _path(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x)).sum()) if len(x) > 1 else 0.0


def _net(x: np.ndarray) -> float:
    return abs(float(x[-1] - x[0])) if len(x) > 1 else 0.0


def _reversals(x: np.ndarray, seg: float) -> float:
    """Direction changes, ignoring movement smaller than `seg`."""
    n, last_dir, anchor = 0, 0, None
    for v in x:
        if anchor is None:
            anchor = v
            continue
        d = v - anchor
        if abs(d) < seg:
            continue
        direction = 1 if d > 0 else -1
        if last_dir and direction != last_dir:
            n += 1
        last_dir, anchor = direction, v
    return float(n)


def _resample(series: np.ndarray, times: np.ndarray, steps: int) -> np.ndarray:
    """Linear interpolation onto `steps` evenly spaced points in [0, 1] time."""
    t = np.asarray(times, dtype=np.float64)
    span = float(t[-1] - t[0])
    if span <= 1e-9 or len(t) < 2:
        return np.repeat(series[:1], steps, axis=0)
    u = (t - t[0]) / span
    grid = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(grid, u, series[:, j]) for j in range(series.shape[1])],
                    axis=1)


def features_from_frames(lms, times) -> np.ndarray:
    """(T,21,3) landmarks + (T,) timestamps -> (D,) float32 window vector.

    Order is chronological. The caller is responsible for passing a window of
    about WINDOW_S seconds; fewer than 4 frames raises, because a 2-frame
    "stroke" is noise.
    """
    lms = np.asarray(lms, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if lms.ndim != 3 or lms.shape[1:] != (21, 3):
        raise ValueError(f"expected (T,21,3) landmarks, got {lms.shape}")
    if len(lms) != len(times):
        raise ValueError("landmarks and timestamps disagree on frame count")
    if len(lms) < 4:
        raise ValueError(f"a J/Z window needs at least 4 frames, got {len(lms)}")
    order = np.argsort(times)
    lms, times = lms[order], times[order]

    series, palms, exts = _series(lms)
    res = _resample(series, times, STEPS)          # (STEPS, 6)

    dur = float(times[-1] - times[0])
    dur = dur if dur > 1e-6 else 1e-6
    diff = np.diff(series, axis=0)
    speed = np.linalg.norm(diff, axis=0) / max(len(diff), 1)   # per stream

    scal: list[float] = []
    for j in range(series.shape[1]):                # path, net, range per stream
        s = series[:, j]
        scal += [_path(s), _net(s), float(s.max() - s.min())]
    for j in REV_STREAMS:
        for seg in REV_SEGS:
            scal.append(_reversals(series[:, j], seg))
    scal += [float(speed.mean()), float(speed.max())]            # wrist speed
    scal += [float(speed[2:4].mean()), float(speed[2:4].max())]  # tip, wrist-relative
    scal += [float(speed[4:].mean()), float(speed[4:].max())]    # tip, window-anchored
    wrist_path, tip_path = _path(series[:, 0]), _path(series[:, 4])
    scal.append(tip_path / max(wrist_path, 1e-6))
    scal.append(dur)
    scal += [float(palms.mean()), float(palms.std())]
    up = np.asarray([e >= 1.6 for e in exts], dtype=np.float64)
    scal += [float(up.mean()), float(exts.mean()), float(exts.std())]
    # direction of the net stroke, which is what separates J's hook from a
    # sideways K sweep in the same place
    scal += [float(np.sign(_signed(series[:, 0]))),
             float(np.sign(_signed(series[:, 2]))),
             float(np.sign(_signed(series[:, 4])))]
    wx = series[:, 0]
    for b in (series[:, 2], series[:, 3], series[:, 4], series[:, 5]):
        if wx.std() > 1e-9 and b.std() > 1e-9:
            scal.append(float(np.corrcoef(wx, b)[0, 1]))
        else:
            scal.append(0.0)

    vec = np.concatenate([res.reshape(-1), np.asarray(scal, dtype=np.float64)])
    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec.astype(np.float32)


def _signed(x: np.ndarray) -> float:
    return float(x[-1] - x[0]) if len(x) > 1 else 0.0


def feature_dim() -> int:
    """Dimension of `features_from_frames`, derived rather than hard-coded."""
    t = np.linspace(0.0, WINDOW_S, 24)
    lm = np.zeros((len(t), 21, 3), dtype=np.float64)
    # a moving, non-degenerate hand so no constant-input shortcut hides a bug
    lm[:, :, 0] = np.linspace(0.0, 0.2, len(t))[:, None] + np.arange(21)[None, :] * 0.01
    lm[:, :, 1] = np.linspace(0.0, 0.1, len(t))[:, None] + np.arange(21)[None, :] * 0.005
    return int(features_from_frames(lm, t).shape[0])
