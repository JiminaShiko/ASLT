"""Stage 3 - the single feature function shared by training and live inference.

Packed landmarks -> [NUM_FRAMES, D] in shoulder widths, y up; z is ignored because pose z
(metres from the hip) and hand z (from the wrist) are incomparable. `train.py --parity` proves both paths agree.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import config

_POSE = {name: config.pose_row(i) for name, i in {
    "nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8,
    "l_mouth": 9, "r_mouth": 10, "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
    "l_hip": 25, "r_hip": 26}.items()}
POSE_ORDER = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_mouth",
              "r_mouth", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
              "l_wrist", "r_wrist", "l_hip", "r_hip"]
POSE_PAIRS = [("l_eye", "r_eye"), ("l_ear", "r_ear"), ("l_mouth", "r_mouth"),
              ("l_shoulder", "r_shoulder"), ("l_elbow", "r_elbow"),
              ("l_wrist", "r_wrist"), ("l_hip", "r_hip")]

FACE_COLS = {g: np.array([k for k, i in enumerate(config.FACE_ORDER)
                          if i in group], dtype=int)
             for g, group in config.FACE_GROUPS.items()}
FACE_ABS = {g: config.N_POSE + c for g, c in FACE_COLS.items()}

WRIST = 0
THUMB_TIP, INDEX_PIP, INDEX_TIP = 4, 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 19, 20
TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
CURLS = [(THUMB_TIP, 2), (INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]
SPREADS = [(INDEX_TIP, MIDDLE_TIP), (MIDDLE_TIP, RING_TIP),
           (RING_TIP, PINKY_TIP)]

BLOCK_WIDTHS = {
    "pose": 2 * len(POSE_ORDER),
    "hands": 4 * 21,
    "hand_shape": 2 * 9,
    "face": 12,
    "hand_to_body": 2 * 5,
    "mask": 4,
    "velocity": 12,
    "segment": 2 * 14 + 2,
}
BLOCK_NAMES = list(BLOCK_WIDTHS)
FEATURE_BLOCKS: dict[str, tuple[int, int]] = {}
FEATURE_DIM = 0
_start = 0
for _b in BLOCK_NAMES:
    FEATURE_BLOCKS[_b] = (_start, BLOCK_WIDTHS[_b])
    _start += BLOCK_WIDTHS[_b]
FEATURE_DIM = _start
FEATURE_VERSION = config.FEATURE_VERSION

# v2 is the older 170-dim vector (no segment block, no gap fill), kept for comparison
V2_BLOCK_NAMES = [b for b in BLOCK_NAMES if b != "segment"]
_LAYOUTS: dict[int, tuple[dict[str, tuple[int, int]], int]] = {
    FEATURE_VERSION: (FEATURE_BLOCKS, FEATURE_DIM),
}


def layout(version: int | None = None) -> tuple[dict[str, tuple[int, int]], int]:
    """(block -> (start, width), total dim) for a feature version."""
    v = FEATURE_VERSION if version is None else int(version)
    if v not in _LAYOUTS:
        if v == 2:
            blocks: dict[str, tuple[int, int]] = {}
            start = 0
            for b in V2_BLOCK_NAMES:
                blocks[b] = (start, BLOCK_WIDTHS[b])
                start += BLOCK_WIDTHS[b]
            _LAYOUTS[v] = (blocks, start)
        else:
            raise ValueError(f"no feature layout for version {v}; "
                             f"known: 2 and {FEATURE_VERSION}")
    return _LAYOUTS[v]


def feature_dim(version: int | None = None) -> int:
    return FEATURE_DIM if version is None else layout(version)[1]


def version_for_dim(dim: int) -> int | None:
    """Which feature version builds a vector of `dim` columns (None = none)."""
    for v in (2, FEATURE_VERSION):
        if feature_dim(v) == int(dim):
            return v
    return None


def block_of(index: int) -> str:
    for name, (start, width) in FEATURE_BLOCKS.items():
        if start <= index < start + width:
            return name
    raise IndexError(index)


def block_columns(selected: list[str] | None,
                  version: int | None = None) -> np.ndarray | slice:
    """Column selection for --blocks ablations; duplicates count once, unknown names raise."""
    blocks, dim = layout(version)
    if not selected or selected == ["all"]:
        return slice(0, dim)
    seen: list[str] = []
    for b in selected:
        if b not in blocks:
            raise KeyError(f"unknown feature block '{b}' (version "
                           f"{FEATURE_VERSION if version is None else version}); "
                           f"choose from {list(blocks)}")
        if b not in seen:
            seen.append(b)
    cols: list[int] = []
    for b in blocks:
        if b in seen:
            s, w = blocks[b]
            cols.extend(range(s, s + w))
    return np.array(cols, dtype=int)


def _interp(rows: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """NaN-aware linear resampling of [T, P, 2] at float positions `pos`."""
    T = rows.shape[0]
    if T == 1:
        return np.repeat(rows[:1], len(pos), axis=0)
    lo = np.clip(np.floor(pos).astype(int), 0, T - 1)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (pos - lo)[..., None, None].astype(np.float32)
    a, b = rows[lo], rows[hi]
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    both = a_nan & b_nan
    either = np.where(a_nan, b, np.where(b_nan, a,
                                         a * (1.0 - frac) + b * frac))
    return np.where(both, np.nan, either).astype(np.float32)


def _fill_rows(vec: np.ndarray) -> np.ndarray:
    """Replace NaN rows of [N, k] with the previous valid row (then backfill)."""
    out = vec.copy()
    bad = np.isnan(out).any(axis=1)
    if not bad.any():
        return out
    good = np.flatnonzero(~bad)
    if good.size == 0:
        return np.zeros_like(out)
    for j in range(out.shape[1]):
        out[:, j] = np.interp(np.arange(len(out)), good, out[good, j])
    return out.astype(np.float32)


def _reference(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shoulder midpoint [T,2] and shoulder width [T,1] per frame, filled."""
    l = pose[:, _POSE["l_shoulder"]]
    r = pose[:, _POSE["r_shoulder"]]
    both = ~np.isnan(l).any(1) & ~np.isnan(r).any(1)
    mid = np.full((len(pose), 2), np.nan, np.float32)
    width = np.full(len(pose), np.nan, np.float32)
    mid[both] = (l[both] + r[both]) / 2.0
    width[both] = np.linalg.norm(l[both] - r[both], axis=1)
    mid = _fill_rows(mid)
    width = _fill_rows(width[:, None])[:, 0]
    width = np.where(width <= 0, 0.30, width)
    return mid, np.clip(width, 0.08, 1.2)[:, None]


def _norm(points: np.ndarray, mid: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = (points - mid[:, None, :]) / w[:, None, :]
    out[..., 1] = -out[..., 1]
    return out


def _zero_missing(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T,k,2] with NaN -> ([T,k,2] zeros where absent, [T] 1.0 when complete)."""
    present = (~np.isnan(block).any(2)).all(1).astype(np.float32)
    return np.where(np.isnan(block), 0.0, block).astype(np.float32), present


def fill_gaps(rows: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Bridge detection dropouts of <= max_gap frames in a [T, P, 2] block.

    Longer absences are real, so no trajectory is invented; the mask comes from the
    un-filled rows, so a bridge never counts as a sighting.
    """
    n = config.GAP_FILL_FRAMES if max_gap is None else int(max_gap)
    out = np.array(rows, dtype=np.float32, copy=True)
    if n <= 0 or out.ndim != 3 or len(out) < 2:
        return out
    miss = np.isnan(out).any(axis=2).any(axis=1)
    if not miss.any() or miss.all():
        return out
    T, t = len(miss), 0
    while t < T:
        if not miss[t]:
            t += 1
            continue
        b = t
        while b + 1 < T and miss[b + 1]:
            b += 1
        if b - t + 1 <= n:
            a, c = t - 1, b + 1
            if a >= 0 and c < T:
                out[t:b + 1] = np.linspace(out[a], out[c], b - t + 3)[1:-1]
            elif a >= 0:
                out[t:b + 1] = out[a]
            elif c < T:
                out[t:b + 1] = out[c]
        t = b + 1
    return out


def seen_frames(rows: np.ndarray) -> np.ndarray:
    """[T, P, 2] -> [T] bool: the detector really returned this frame."""
    return ~(np.isnan(np.asarray(rows)).any(axis=2).any(axis=1))


def _resample_index(pos: np.ndarray, T: int) -> np.ndarray:
    """Which source frame each resampled position is nearest to."""
    return np.clip(np.round(pos).astype(int), 0, max(T - 1, 0))


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def hand_shape(hand: np.ndarray) -> np.ndarray:
    """[T,21,2] (zero-filled, shoulder units) -> [T,9]."""
    palm = _dist(hand[:, TIPS], hand[:, WRIST][:, None, :]).mean(1)
    scale = np.maximum(palm, 1e-3)
    curl = np.stack([_dist(hand[:, a], hand[:, b]) for a, b in CURLS], axis=1)
    spread = np.stack([_dist(hand[:, a], hand[:, b])
                       for a, b in SPREADS], axis=1).max(1)
    out = [palm, *(curl[:, k] / scale for k in range(len(CURLS))),
           spread / scale,
           _dist(hand[:, THUMB_TIP], hand[:, INDEX_TIP]) / scale,
           _dist(hand[:, PINKY_MCP], hand[:, WRIST]) / scale]
    return np.stack(out, axis=1).astype(np.float32)


def face_summary(face: np.ndarray) -> np.ndarray:
    """[T,N_FACE,2] -> [T,12] face summaries; empty slices are expected, so warn nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _face_summary(face)


def _face_summary(face: np.ndarray) -> np.ndarray:
    lips = face[:, FACE_COLS["lips"]]
    nose = face[:, FACE_COLS["nose"]]
    jaw = face[:, FACE_COLS["chin_jaw"]]
    brow = face[:, FACE_COLS["brows_forehead"]]
    xs, ys = lips[..., 0], lips[..., 1]
    mouth_c = np.nanmean(lips, axis=1)
    width = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
    open_ = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
    lo = np.nanmin(xs, axis=1)
    hi = np.nanmax(xs, axis=1)
    corner_y = (np.nanmin(np.where(xs <= (lo + 1e-6)[:, None], ys, np.nan), axis=1)
                + np.nanmin(np.where(xs >= (hi - 1e-6)[:, None], ys, np.nan),
                            axis=1)) / 2.0
    parts = [mouth_c[:, 0], mouth_c[:, 1], width, open_,
             corner_y - mouth_c[:, 1],
             np.nanmean(nose, axis=1)[:, 0], np.nanmean(nose, axis=1)[:, 1],
             np.nanmean(jaw, axis=1)[:, 0], np.nanmean(jaw, axis=1)[:, 1],
             np.nanmean(brow, axis=1)[:, 0], np.nanmean(brow, axis=1)[:, 1],
             np.nanmax(brow[..., 0], axis=1) - np.nanmin(brow[..., 0], axis=1)]
    return np.nan_to_num(np.stack(parts, axis=1), nan=0.0,
                         posinf=0.0, neginf=0.0).astype(np.float32)


def sequence(arr: np.ndarray, num_frames: int | None = None,
             augment: bool = False, rng: np.random.Generator | None = None,
             blocks: list[str] | None = None,
             version: int | None = None) -> np.ndarray:
    """[T, P_TOTAL, 3] -> [num_frames, D]; version picks the layout (v3 or v2).

    The checkpoint records its feature_version. augment=True is for the train split only.
    """
    v = FEATURE_VERSION if version is None else int(version)
    names, dim = layout(v)
    num_frames = num_frames or config.NUM_FRAMES
    if augment and rng is None:
        raise ValueError("augment=True needs a Generator so runs stay seedable")
    arr = np.asarray(arr, np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (config.P_TOTAL, 3):
        raise ValueError(
            f"expected [T, {config.P_TOTAL}, 3], got {arr.shape} - this cache "
            f"was written by a different landmark version; re-run "
            f"python -m src.extract --force")
    if len(arr) == 0:
        return np.zeros((num_frames, dim), np.float32)

    if augment:
        arr = _augment_landmarks(arr, rng)

    T = len(arr)
    if augment and T > 6:
        span = float(rng.uniform(*config.AUG["speed"]))
        keep = int(np.clip(round(T / span), max(6, T // 2), T))
        start = int(rng.integers(0, T - keep + 1)) if keep < T else 0
        arr = arr[start:start + keep]
        T = len(arr)

    xy = arr[..., :2]
    pos = np.linspace(0, T - 1, num_frames)
    src = _resample_index(pos, T)
    pose = _interp(xy[:, config.COLS_POSE], pos)
    face = _interp(xy[:, config.COLS_FACE], pos)
    l_raw, r_raw = xy[:, config.COLS_LHAND], xy[:, config.COLS_RHAND]
    l_seen, r_seen = seen_frames(l_raw), seen_frames(r_raw)
    if v >= 3:
        lh = _interp(fill_gaps(l_raw), pos)
        rh = _interp(fill_gaps(r_raw), pos)
    else:
        lh = _interp(l_raw, pos)
        rh = _interp(r_raw, pos)

    mid, w = _reference(pose)
    pose_n = _norm(pose, mid, w)
    face_n = _norm(face, mid, w)
    lh_z, _ = _zero_missing(_norm(lh, mid, w))
    rh_z, _ = _zero_missing(_norm(rh, mid, w))
    # mask from un-filled rows: a bridged frame is not a sighting
    lh_ok = l_seen[src].astype(np.float32)
    rh_ok = r_seen[src].astype(np.float32)

    pose_feats = pose_n[:, [_POSE[k] for k in POSE_ORDER]].reshape(num_frames, -1)
    hands_feats = np.concatenate(
        [lh_z.reshape(num_frames, -1), rh_z.reshape(num_frames, -1)], axis=1)
    shape_feats = np.concatenate([hand_shape(lh_z), hand_shape(rh_z)], axis=1)
    face_feats = face_summary(face_n)
    body_feats = _hand_to_body(pose_n, lh_z, rh_z)
    mask = np.stack([lh_ok, rh_ok,
                     (~np.isnan(pose).any(2)).all(1).astype(np.float32),
                     (~np.isnan(face).any(2)).all(1).astype(np.float32)], axis=1)
    vel = _velocity(pose_n, lh_z, rh_z, lh_ok, rh_ok)
    by_name: dict[str, np.ndarray] = {
        "pose": pose_feats, "hands": hands_feats, "hand_shape": shape_feats,
        "face": face_feats, "hand_to_body": body_feats, "mask": mask,
        "velocity": vel}
    if "segment" in names:
        by_name["segment"] = _segment(pose_n, lh_z, rh_z, lh_ok, rh_ok)

    full = np.concatenate([by_name[b] for b in names],
                          axis=1).astype(np.float32)
    want = (num_frames, dim)
    if full.shape != want:
        raise RuntimeError(f"feature layout drift: built {full.shape}, "
                           f"expected {want}; layout={names}")
    if augment:
        full = _augment_features(full, rng)
    cols = block_columns(blocks, version=v)
    out = full[:, cols]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _hand_to_body(pose_n: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Hand-to-face / chin / chest distances, in shoulder widths, per hand."""
    nose = pose_n[:, _POSE["nose"]]
    mouth = (pose_n[:, _POSE["l_mouth"]] + pose_n[:, _POSE["r_mouth"]]) / 2.0
    chin = mouth.copy()
    chin[:, 1] -= 0.35
    chest = np.zeros_like(nose)
    blocks = []
    for hand in (lh, rh):
        palm = hand[:, TIPS].mean(1)
        tips = hand[:, TIPS]
        tip_to_mouth = _dist(tips, mouth[:, None, :]).min(1)
        above = (palm[:, 1] > 0.0).astype(np.float32)
        blocks.append(np.stack([_dist(hand[:, WRIST], nose),
                                _dist(palm, chin),
                                _dist(palm, chest),
                                tip_to_mouth, above], axis=1))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _velocity(pose_n, lh, rh, lh_ok, rh_ok) -> np.ndarray:
    """[T,12] motion between resampled frames, in shoulder widths per frame."""
    def d(a: np.ndarray) -> np.ndarray:
        return np.diff(a, axis=0, prepend=a[:1])

    l_w, r_w = d(lh[:, WRIST]), d(rh[:, WRIST])
    l_p, r_p = d(lh[:, TIPS].mean(1)), d(rh[:, TIPS].mean(1))
    sh = d((pose_n[:, _POSE["l_shoulder"]] + pose_n[:, _POSE["r_shoulder"]]) / 2)
    mo = d((pose_n[:, _POSE["l_mouth"]] + pose_n[:, _POSE["r_mouth"]]) / 2)
    acc_l = np.concatenate([[0.0], np.diff(_dist(l_w, np.zeros_like(l_w)))])
    acc_r = np.concatenate([[0.0], np.diff(_dist(r_w, np.zeros_like(r_w)))])
    return np.stack([
        _dist(l_w, np.zeros_like(l_w)) * lh_ok,
        _dist(r_w, np.zeros_like(r_w)) * rh_ok,
        _dist(l_p, np.zeros_like(l_p)) * lh_ok,
        _dist(r_p, np.zeros_like(r_p)) * rh_ok,
        _dist(sh, np.zeros_like(sh)), _dist(mo, np.zeros_like(mo)),
        l_w[:, 1], r_w[:, 1], l_p[:, 0], r_p[:, 0], acc_l, acc_r,
    ], axis=1).astype(np.float32)


def _hand_segment(h: np.ndarray, ok: np.ndarray, mouth: np.ndarray,
                  chin: np.ndarray, chest: np.ndarray) -> np.ndarray:
    """One hand's whole-clip summary -> [14]; start/middle/end = first/middle/last detected frame.

    A never-detected hand returns zeros, not an invented trajectory.
    """
    out = np.zeros(14, np.float32)
    have = np.flatnonzero(ok > 0.5)
    if not len(have):
        return out
    palm, wrist = h[:, TIPS].mean(1), h[:, WRIST]
    d_mouth = _dist(palm, mouth)
    i0, im, i1 = have[0], have[len(have) // 2], have[-1]
    pick = np.array([i0, im, i1])
    out[0:3] = d_mouth[pick]
    out[3:6] = _dist(palm, chin)[pick]
    out[6:9] = _dist(palm, chest)[pick]
    out[9] = float(np.max(_dist(wrist[have], wrist[i0])))
    out[10] = float(_dist(wrist[i1], wrist[i0]))
    ic = have[int(np.argmin(d_mouth[have]))]
    out[11] = float(_dist(wrist[i1], wrist[ic]))
    out[12] = float(wrist[ic, 1])
    out[13] = float(len(have) / len(ok))
    return out


def _segment(pose_n: np.ndarray, lh: np.ndarray, rh: np.ndarray,
             lh_ok: np.ndarray, rh_ok: np.ndarray) -> np.ndarray:
    """[num_frames, 30]: whole-clip read-out repeated down the time axis.

    wrist-to-wrist needs both hands, so the presence fraction separates "0.6 apart" from "no data".
    """
    mouth = (pose_n[:, _POSE["l_mouth"]] + pose_n[:, _POSE["r_mouth"]]) / 2.0
    chin = mouth.copy()
    chin[:, 1] -= 0.35
    chest = np.zeros_like(mouth)
    both = (lh_ok > 0.5) & (rh_ok > 0.5)
    seen = np.flatnonzero(both)
    ww_end = float(_dist(lh[:, WRIST], rh[:, WRIST])[seen[-1]]) if len(seen) else 0.0
    vec = np.concatenate([
        _hand_segment(lh, lh_ok, mouth, chin, chest),
        _hand_segment(rh, rh_ok, mouth, chin, chest),
        [ww_end, float(both.mean())]]).astype(np.float32)
    return np.tile(vec, (len(pose_n), 1))


SEG_COLS = ["mouth_start", "mouth_mid", "mouth_end",
            "chin_start", "chin_mid", "chin_end",
            "chest_start", "chest_mid", "chest_end",
            "excursion", "travel", "retreat", "y_at_closest", "presence"]
SEG_NAMES = ([f"l_{c}" for c in SEG_COLS] + [f"r_{c}" for c in SEG_COLS]
             + ["wrist_to_wrist_end", "both_hands_present_frac"])


# one line per config.AUG knob; a test asserts every knob is named here
AUGMENT_DOC = """
landmark level:
  mirror         swap hands + anatomical pose pairs + flip x (handedness-agnostic)
  scale          positions x (1 + N(0, AUG['scale']))
  rotate         in-image rotation up to AUG['rotate'] degrees
  translate      shift up to AUG['translate'] shoulder widths
  jitter         gaussian noise, sigma AUG['jitter'] per landmark per frame
  sub-window     random slice of AUG['speed'] (0.80-1.20) of the clip
feature level:
  frame_dropout  AUG['frame_dropout'] of frames repeat the previous frame
  velocity scale velocity block x U(AUG['speed']) (0.8-1.2)
""".strip()


def _augment_landmarks(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    if rng.random() < config.AUG["mirror"]:
        out = mirror_pack(out)
    xy = out[..., :2].astype(np.float32)
    xy = xy * (1.0 + float(rng.normal(0.0, config.AUG["scale"])))
    a = float(rng.uniform(-config.AUG["rotate"], config.AUG["rotate"]))
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    xy = np.stack([c * xy[..., 0] - s * xy[..., 1],
                   s * xy[..., 0] + c * xy[..., 1]], axis=-1)
    xy = xy + np.array([rng.normal(0.0, config.AUG["translate"]),
                        rng.normal(0.0, config.AUG["translate"])], np.float32)
    noisy = xy + rng.normal(0.0, config.AUG["jitter"], xy.shape).astype(np.float32)
    out[..., :2] = np.where(np.isnan(xy), np.nan, noisy)
    return out


def mirror_pack(arr: np.ndarray) -> np.ndarray:
    """Left/right swap + x mirror; the face block needs no permutation (summaries are invariant)."""
    out = np.asarray(arr).copy()
    out[..., 0] = -out[..., 0]
    lh = out[:, config.COLS_LHAND].copy()
    out[:, config.COLS_LHAND] = out[:, config.COLS_RHAND]
    out[:, config.COLS_RHAND] = lh
    for a_name, b_name in POSE_PAIRS:
        ia, ib = _POSE[a_name], _POSE[b_name]
        tmp = out[:, ia].copy()
        out[:, ia] = out[:, ib]
        out[:, ib] = tmp
    return out


def _augment_features(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = feat.copy()
    n = len(out)
    k = int(round(config.AUG["frame_dropout"] * n))
    if k and n > 2:
        idx = rng.choice(np.arange(1, n), size=min(k, n - 1), replace=False)
        for i in idx:
            out[i] = out[i - 1]
    s, w = FEATURE_BLOCKS["velocity"]
    out[:, s:s + w] *= float(rng.uniform(*config.AUG["speed"]))
    return out


class RingBuffer:
    """Fixed-capacity landmark buffer for the live segmenter (memory only)."""

    def __init__(self, maxlen: int = 96) -> None:
        self.maxlen = int(maxlen)
        self._rows: list[np.ndarray] = []

    def append(self, row: np.ndarray) -> None:
        self._rows.append(np.asarray(row, np.float32))
        if len(self._rows) > self.maxlen:
            self._rows.pop(0)

    def clear(self) -> None:
        self._rows = []

    def array(self) -> np.ndarray:
        if not self._rows:
            return np.zeros((0, config.P_TOTAL, 3), np.float32)
        return np.stack(self._rows)

    def last(self) -> np.ndarray | None:
        """Newest row without stacking the whole buffer (overlay path)."""
        return self._rows[-1] if self._rows else None

    def __len__(self) -> int:
        return len(self._rows)


def from_live_buffer(rows, num_frames: int | None = None,
                     version: int | None = None) -> np.ndarray:
    """Exactly what training saw, from a live buffer of landmark rows."""
    arr = rows.array() if isinstance(rows, RingBuffer) else np.asarray(rows)
    return sequence(arr, num_frames=num_frames, version=version)


if __name__ == "__main__":
    import json
    print(json.dumps({
        "feature_dim": FEATURE_DIM,
        "feature_dim_v2": feature_dim(2),
        "v2_blocks": V2_BLOCK_NAMES,
        "num_frames": config.NUM_FRAMES,
        "blocks": FEATURE_BLOCKS,
        "landmark_version": config.LANDMARK_VERSION,
        "feature_version": config.FEATURE_VERSION,
        "packed_shape": [config.P_TOTAL, 3],
    }, indent=1))
