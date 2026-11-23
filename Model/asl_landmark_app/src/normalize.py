"""Step 3 - the single shared normalisation + feature function.

Used identically by training and the live app via `normalize.features(lm)`.
Handedness is decided geometrically, not from MediaPipe's `handedness` label: that
label assumes a mirrored selfie frame and is inverted for unmirrored webcam images.
"""
from __future__ import annotations

import os

import numpy as np

WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17
SCALE_EPS = 1e-6

FINGERTIPS = [4, 8, 12, 16, 20]
FINGER_CHAINS = [
    (1, 2, 3, 4),      # thumb  (angle at the IP joint)
    (5, 6, 7, 8),      # index
    (9, 10, 11, 12),   # middle
    (13, 14, 15, 16),  # ring
    (17, 18, 19, 20),  # pinky
]
# Landmark ids are anatomical, so a horizontal flip does not permute them.

# Which side of the geometric test gets mirrored; see docs/DETAILS.md.
MIRROR_WHEN_CROSS_POSITIVE = True

# Joints used by the optional "thumb" and "orient" blocks.
THUMB_TIP = 4
PALM_MCP = [0, 5, 9, 13, 17]          # wrist + the four MCPs -> palm centre
THUMB_TARGETS = [5, 6, 9, 10, 13, 14, 17]   # index MCP/PIP, middle MCP/PIP,
                                             # ring MCP/PIP, pinky MCP
FINGERS_ACROSS = [8, 12, 16, 20]      # thumb tip -> other fingertips
# Signed thumb-tip offset from two lines: 5->17 knuckles, 8->20 fingertips.
THUMB_LINE_A = (5, 17)
THUMB_LINE_B = (8, 20)

# Optional "cross" block: thumb tip offset/along each finger's MCP->PIP bone.
# Reference is the bone, not MCP->tip, whose length collapses on a folded finger.
# A 2-D crossing proxy only: z is not used, so this is not a depth test.
CROSS_FINGERS = [(5, 6), (9, 10), (13, 14), (17, 18)]   # (MCP, PIP) per finger

# Bone units; unclamped values reach ~800 on landmark failures and their tails
# would squash the standardised inputs. The cap keeps the sign (T vs L).
CROSS_CLAMP = 6.0

# Feature blocks, comma-separated via ASL_FEATURE_BLOCKS. Default xy (MediaPipe's
# z scale does not transfer between datasets). "cross" is built but loses cross-
# dataset, so it is not in the default; "orient" is kept off since the hand is
# deliberately not rotation-normalised.
_DEFAULT_BLOCKS = os.environ.get("ASL_FEATURE_BLOCKS", "xy,tip_dists,curl,thumb")
FEATURE_BLOCKS = tuple(b.strip() for b in _DEFAULT_BLOCKS.split(",") if b.strip())


def as_array(landmarks) -> np.ndarray:
    """Accept MediaPipe NormalizedLandmark objects or a (21,3) array."""
    arr = np.asarray(
        [[lm.x, lm.y, lm.z] if hasattr(lm, "x") else list(lm)[:3] for lm in landmarks],
        dtype=np.float64,
    )
    if arr.shape[0] != 21:
        raise ValueError(f"expected 21 landmarks, got {arr.shape[0]}")
    return arr.reshape(21, 3)


def palm_cross_sign(lm: np.ndarray) -> float:
    """2-D cross product of the two palm vectors; y-down flips its sign."""
    v1 = lm[INDEX_MCP, :2] - lm[WRIST, :2]
    v2 = lm[PINKY_MCP, :2] - lm[WRIST, :2]
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def should_mirror(lm: np.ndarray) -> bool:
    cross = palm_cross_sign(lm)
    if abs(cross) < 1e-9:          # degenerate / edge-on hand: leave as-is
        return False
    return (cross > 0) == MIRROR_WHEN_CROSS_POSITIVE


def chirality_family(lm: np.ndarray) -> str:
    """Image-plane chirality family: 'canonical', 'mirrored' or 'degenerate'.

    Not an anatomy label: a palm-forward right hand and a back-of-hand-forward
    left hand project identically. Diagnostic only; never feeds a feature.
    """
    lm = as_array(lm)
    cross = palm_cross_sign(lm)
    if abs(cross) < 1e-9:
        return "degenerate"
    return "mirrored" if should_mirror(lm) else "canonical"


def mirror_hand(lm: np.ndarray) -> np.ndarray:
    """Negate x only: landmark ids are anatomical, so a flip must not permute them."""
    out = np.asarray(lm, dtype=np.float64).copy()
    out[:, 0] *= -1.0
    return out


def canonicalize(lm: np.ndarray) -> np.ndarray:
    lm = as_array(lm)
    return mirror_hand(lm) if should_mirror(lm) else lm


def center_and_scale(lm: np.ndarray) -> np.ndarray:
    """Wrist to the origin, scaled by the wrist->middle-MCP bone length."""
    origin = lm[WRIST]
    ref = lm[MIDDLE_MCP]
    scale = float(np.linalg.norm(ref - origin))
    safe = scale if scale > SCALE_EPS else SCALE_EPS
    return (lm - origin) / safe


def features_from_normalized_batch(lms: np.ndarray, blocks=None) -> np.ndarray:
    """(N, 21, 3) prepared hands -> (N, D) features; the one implementation."""
    blocks = tuple(blocks or FEATURE_BLOCKS)
    lms = np.asarray(lms, dtype=np.float64)
    if lms.ndim == 2:
        lms = lms[None, :, :]
    n = lms.shape[0]
    # distances/angles follow whichever coordinate block is enabled
    ndim = 3 if "xyz" in blocks else 2
    pts = lms[:, :, :ndim]
    parts = []
    if "xyz" in blocks:
        parts.append(lms.reshape(n, -1))
    elif "xy" in blocks:
        parts.append(pts.reshape(n, -1))
    if "tip_dists" in blocks:
        tips = pts[:, FINGERTIPS, :]                       # (N,5,ndim)
        diff = tips[:, None, :, :] - tips[:, :, None, :]   # (N,5,5,ndim)
        dmat = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(len(FINGERTIPS), k=1)
        parts.append(dmat[:, iu[0], iu[1]])
    if "curl" in blocks:
        angles = np.zeros((n, len(FINGER_CHAINS)), dtype=np.float64)
        for f, (mcp, pip, dip, tip) in enumerate(FINGER_CHAINS):
            a = pts[:, pip, :] - pts[:, mcp, :]
            b = pts[:, tip, :] - pts[:, pip, :]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            safe = (na > SCALE_EPS) & (nb > SCALE_EPS)
            cosang = np.zeros(n, dtype=np.float64)
            cosang[safe] = np.einsum("ij,ij->i", a[safe], b[safe]) / (na[safe] * nb[safe])
            angles[:, f] = np.arccos(np.clip(cosang, -1.0, 1.0))
        parts.append(angles)
    if "thumb" in blocks:
        tt = pts[:, THUMB_TIP, :]                             # (N,ndim)
        tg = pts[:, THUMB_TARGETS, :]                         # (N,7,ndim)
        parts.append(np.linalg.norm(tg - tt[:, None, :], axis=-1))          # 7
        palm = pts[:, PALM_MCP, :].mean(axis=1)               # (N,ndim)
        parts.append(np.linalg.norm(palm - tt, axis=-1)[:, None])           # 1
        ft = pts[:, FINGERS_ACROSS, :]                        # (N,4,ndim)
        parts.append(np.linalg.norm(ft - tt[:, None, :], axis=-1))          # 4
        for a, b in (THUMB_LINE_A, THUMB_LINE_B):             # signed offsets
            p0, p1 = pts[:, a, :2], pts[:, b, :2]
            line = p1 - p0
            ln = np.maximum(np.linalg.norm(line, axis=1), SCALE_EPS)
            rel = pts[:, THUMB_TIP, :2] - p0
            cr = line[:, 0] * rel[:, 1] - line[:, 1] * rel[:, 0]
            parts.append((cr / ln)[:, None])                                # 2
    if "cross" in blocks:
        # (offset, along) in bone units; along > 1 = past the second joint
        tp = pts[:, THUMB_TIP, :2]
        cols = []
        for mcp, pip in CROSS_FINGERS:
            p0, p1 = pts[:, mcp, :2], pts[:, pip, :2]
            axis = p1 - p0
            ln = np.maximum(np.linalg.norm(axis, axis=1), SCALE_EPS)
            u = axis / ln[:, None]
            rel = tp - p0
            cols.append(((u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0]) / ln)[:, None])
            cols.append(((rel[:, 0] * u[:, 0] + rel[:, 1] * u[:, 1]) / ln)[:, None])
        parts.append(np.clip(np.concatenate(cols, axis=1),
                             -CROSS_CLAMP, CROSS_CLAMP))                     # 8
    if "orient" in blocks:
        d1 = _dir2(pts, WRIST, MIDDLE_MCP)                    # wrist -> middle
        d2 = _dir2(pts, INDEX_MCP, PINKY_MCP)                 # knuckle line
        parts.append(d1)                                                    # 2
        parts.append(d2)                                                    # 2
        dot = np.einsum("ij,ij->i", d1, d2)
        crs = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        parts.append(np.stack([dot, crs], axis=1))                          # 2
        wi = np.linalg.norm(pts[:, INDEX_MCP, :2] - pts[:, WRIST, :2], axis=1)
        wp = np.linalg.norm(pts[:, PINKY_MCP, :2] - pts[:, WRIST, :2], axis=1)
        parts.append((wp / np.maximum(wi, SCALE_EPS))[:, None])             # 1
        v1 = pts[:, INDEX_MCP, :2] - pts[:, WRIST, :2]
        v2 = pts[:, PINKY_MCP, :2] - pts[:, WRIST, :2]
        cz = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        den = np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1),
                         SCALE_EPS)
        parts.append((cz / den)[:, None])                                   # 1
        if ndim == 3:
            n3 = np.cross(pts[:, INDEX_MCP] - pts[:, WRIST],
                          pts[:, PINKY_MCP] - pts[:, WRIST])
            n3 = n3 / np.maximum(np.linalg.norm(n3, axis=1), SCALE_EPS)[:, None]
            parts.append(n3)                                                # +3
    if not parts:
        raise ValueError(f"no feature blocks selected: {blocks}")
    return np.concatenate(parts, axis=1).astype(np.float32)


def _dir2(pts: np.ndarray, i: int, j: int) -> np.ndarray:
    """Unit vector joint i -> j in the image plane; always 2-D, since z does not transfer."""
    v = pts[:, j, :2] - pts[:, i, :2]
    nrm = np.maximum(np.linalg.norm(v, axis=1), SCALE_EPS)[:, None]
    return v / nrm


def features_from_normalized(lm_scaled, blocks=None) -> np.ndarray:
    """Single (21,3) hand -> flat feature vector (delegates to the batch impl)."""
    return features_from_normalized_batch(lm_scaled, blocks)[0]


def prepare(landmarks) -> np.ndarray:
    """canonicalise + centre + scale -> (21,3) float64."""
    return center_and_scale(canonicalize(landmarks))


def prepare_batch(lms: np.ndarray) -> np.ndarray:
    """(N,21,3) raw -> (N,21,3) canonical/centred/scaled."""
    out = np.empty((len(lms), 21, 3), dtype=np.float64)
    for i in range(len(lms)):
        out[i] = prepare(lms[i])
    return out


def features(landmarks, blocks=None) -> np.ndarray:
    """(21,3) raw landmarks -> flat feature vector; the call live.py and training share.

    `blocks` is None by default (not FEATURE_BLOCKS): a default is frozen at
    import time, so ablation sweeps that set FEATURE_BLOCKS at runtime would
    otherwise be ignored here.
    """
    return features_from_normalized(prepare(landmarks), blocks)


def features_matrix(landmark_list, blocks=None) -> np.ndarray:
    return np.stack([features(lm, blocks) for lm in landmark_list])


_PROBE_HAND = (np.arange(63, dtype=np.float64).reshape(21, 3) - 31.0) / 63.0


def feature_dim(blocks=None) -> int:
    """Column count of features(); measured from the real function so it cannot drift."""
    return int(features(_PROBE_HAND, blocks).shape[0])


def normalize_for_drawing(landmarks) -> np.ndarray:
    """Canonicalised, centred, scaled 21x3; visualiser only."""
    return prepare(landmarks)


if __name__ == "__main__":
    dim = features(np.zeros((21, 3)) + np.arange(63).reshape(21, 3)).shape[0]
    print(f"blocks={FEATURE_BLOCKS} feature_dim={dim}")
    print(f"xyz=63 tip_dists={len(FINGERTIPS)*(len(FINGERTIPS)-1)//2} curl={len(FINGER_CHAINS)}")
