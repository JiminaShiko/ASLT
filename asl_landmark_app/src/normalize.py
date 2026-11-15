"""Step 3 - the single shared normalisation + feature function.

Used by BOTH training (src/train.py) and the live app (src/live.py) through the
identical call `normalize.features(lm)`. There is no second copy anywhere.

What it does, in order:
  1. canonical handedness - a geometric test mirrors "left" hands into one
     canonical hand shape, so the model only ever sees one handedness;
  2. translation  - wrist (landmark 0) becomes the origin;
  3. scale        - divided by the wrist(0) -> middle-finger-MCP(9) distance,
                    with an epsilon guard, so the hand size becomes irrelevant;
  4. features     - the normalised coordinates (42 in the image plane, or 63
                    with MediaPipe's z) + 10 fingertip-pair distances
                    + 5 finger-curl angles.

Why the handedness test is geometric and not MediaPipe's `handedness` label:
the Tasks API reports handedness assuming the frame has been horizontally
mirrored (a selfie camera). A plain OpenCV webcam frame is not mirrored, so the
label is inverted for live inference and is unreliable across datasets. The
palm cross product (wrist->index-MCP) x (wrist->pinky-MCP) is self-consistent
for a still image, a webcam frame and every dataset in this repo, which is what
we need for "the exact same normalize() in training and live inference".
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
# Landmark ids are anatomical, so a horizontal flip does not permute them;
# there is deliberately no chain-swap table here.

# Which side of the geometric test counts as "the other hand" and therefore
# gets mirrored. Fixed once, here, and reused everywhere; see docs/DETAILS.md.
MIRROR_WHEN_CROSS_POSITIVE = True

# --- joints used by the optional "thumb" and "orient" blocks (Stage D) ---
THUMB_TIP = 4
PALM_MCP = [0, 5, 9, 13, 17]          # wrist + the four MCPs -> palm centre
THUMB_TARGETS = [5, 6, 9, 10, 13, 14, 17]   # index MCP/PIP, middle MCP/PIP,
                                             # ring MCP/PIP, pinky MCP
FINGERS_ACROSS = [8, 12, 16, 20]      # thumb tip -> other fingertips
# The signed perpendicular offset of the thumb tip from these two lines is the
# "thumb across the palm / thumb over the fingers" relation: 5->17 is the knuckle
# line, 8->20 the fingertip line. Positive = thumb on the finger side.
THUMB_LINE_A = (5, 17)
THUMB_LINE_B = (8, 20)

# --- joints used by the optional "cross" block (round 4, T/L/M/N) ----------
# Round 3's Stage B found that T fails because its *within-dataset* nearest
# neighbour is L (grassknoted) and A (massey) - the letters it is actually
# answered as - and that nothing in the 71 features says "the thumb is pressed
# across the folded fingers". L holds the thumb out to the side; T folds the
# four fingers and lays the thumb over them, and M/N differ from each other in
# which finger the thumb crosses. So for each of the four fingers this block
# asks one question in two numbers: how far off that finger's own direction the
# thumb tip sits, signed to say which side, and how far along that direction it
# has travelled (1.0 = at the second joint, more = past it, negative = the thumb
# is behind the knuckle, i.e. over the palm).
#
# The reference direction is the MCP->PIP *bone*, not MCP->fingertip: a folded
# finger's MCP->tip vector collapses towards zero, so its direction becomes
# landmark noise and dividing by its length produced values up to +/-100 in the
# cache. A bone vector keeps its length (~0.4 palm units) whether the finger is
# straight or curled, and still rotates with it. Both numbers are divided by
# that bone length, so the block is invariant to translation, in-plane rotation
# and hand size - the two things MediaPipe's z is not.
#
# Honest limitation: occlusion is a DEPTH relation, and this project feeds z to
# no feature because its scale does not transfer between datasets (see above).
# This is therefore a 2-D *crossing* proxy for "thumb over the fingers", not a
# depth test. Stage B's thumb audit says the thumb landmarks are well behaved
# (jitter 0.22-0.28 palm units), which is what makes the proxy worth measuring.
CROSS_FINGERS = [(5, 6), (9, 10), (13, 14), (17, 18)]   # (MCP, PIP) per finger

# Both numbers below are in units of ONE MCP->PIP bone, so a |value| > 6 means
# the thumb tip sits more than six bones away from that finger's knuckle line -
# geometrically off the hand, which is a landmark failure rather than sign
# information. Measured over all 122,359 cached hands, unclamped columns reach
# 91/68/113/111/481/194/798/616 while the p99.9 of the first two is 6.5, so the
# raw ratio has a tail ~100x longer than the signal. The MLP's inputs are
# standardised by cached mean/std, and those tails drag the std down and squash
# every real value; clamping is what makes the block usable. Cost: 2.0% of
# cells (1.3% of the 8 columns per hand on average) are capped, and the cap
# keeps the sign, which is the part that separates T from L.
CROSS_CLAMP = 6.0

# "xy"   = 42 image-plane coordinates (robust, dataset-independent)
# "xyz"  = 63 including MediaPipe's z
#          MediaPipe documents z as being on an ARBITRARY scale, and measured
#          across our datasets it is not comparable: the palm-width/wrist-MCP
#          ratio is 0.67 for grassknoted and danrasband but 1.05 for Massey.
#          Default is therefore xy; set ASL_FEATURE_BLOCKS=xyz,... to compare.
# "thumb"  = 14 thumb-relative features (see THUMB_TARGETS); the thumb is the
#          only digit whose *relationship to the other fingers* carries the
#          letter (A/S/U, P/Q/R, F), and plain tip distances do not capture it.
#          Stage D measured it: 71 dims is the best pooled cross-dataset set of
#          the five tried, and it is the only one that does not cost any letter
#          (K 43.7 -> 58.9). The gain over the 57-dim baseline is +0.2 points,
#          which is inside the run-to-run noise band, so it is promoted on the
#          "never worse on a watched letter" rule rather than on the headline.
# "cross"  = 8 thumb-crossing features (see CROSS_FINGERS): where the thumb tip
#          sits relative to each finger's MCP->PIP bone, in bone units, clamped
#          to +-CROSS_CLAMP. Built in round 4 for T/L and M/N and MEASURED:
#          geometrically it does separate them (T vs L in massey reaches Cohen d
#          4.5 on the first column), but as a model input it loses. 3 seeds on
#          the usual leave-one-dataset-out folds: pooled 66.1 -> 64.1, and every
#          watched letter went down, including T itself (13.5 -> 7.3) and K
#          (56.6 -> 46.5). reports/ablation_thumb_cross.md. NOT in the shipped
#          default, so nothing else in the project pays for it; kept here
#          because the geometry is correct and the negative result is the reason
#          not to spend another round on thumb-crossing features.
# "orient" = 8 image-plane orientation features (+3 if xyz is also on). The hand
#          is deliberately NOT rotation-normalised: P and K, and Q and G, are
#          the same finger pattern at different wrist angles, so the angle is
#          signal. These features just make it linearly available to the MLP
#          instead of forcing it to re-derive sin/cos from 42 raw coordinates.
_DEFAULT_BLOCKS = os.environ.get("ASL_FEATURE_BLOCKS", "xy,tip_dists,curl,thumb")
FEATURE_BLOCKS = tuple(b.strip() for b in _DEFAULT_BLOCKS.split(",") if b.strip())


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
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
    """Signed z of (wrist->index MCP) x (wrist->pinky MCP) in image coords.

    Image y grows downwards, so the sign flips relative to maths convention;
    that is handled by MIRROR_WHEN_CROSS_POSITIVE.
    """
    v1 = lm[INDEX_MCP, :2] - lm[WRIST, :2]
    v2 = lm[PINKY_MCP, :2] - lm[WRIST, :2]
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def should_mirror(lm: np.ndarray) -> bool:
    cross = palm_cross_sign(lm)
    if abs(cross) < 1e-9:          # degenerate / edge-on hand: leave as-is
        return False
    return (cross > 0) == MIRROR_WHEN_CROSS_POSITIVE


def chirality_family(lm: np.ndarray) -> str:
    """Which of the two 2-D chirality families this hand belongs to, named as
    the pipeline uses it: 'canonical' (features taken as-is) or 'mirrored'
    (flipped by `canonicalize()` first). 'degenerate' = an edge-on hand whose
    cross product is ~0, which `canonicalize()` deliberately leaves alone.

    This is an IMAGE-PLANE label, not an anatomy label. `palm_cross_sign`
    reads only x and y, so a right hand shown palm-forward and a left hand
    shown back-of-hand-forward land in the SAME family - they project
    identically in 2-D. Anything that calls this result "Left"/"Right" is
    therefore over-claiming, including MediaPipe.

    Measured against MediaPipe's own `handedness` field over all 127,791
    cached hands, the two tests sort hands into the same pair of groups on
    98.4 % of danrasband, 91.1 % of grassknoted and 87.0 % of massey rows -
    but MediaPipe names each group by the opposite word on these frames,
    because its label is defined for a horizontally mirrored (selfie) image
    and the cache holds unmirrored ones. Both fields are diagnostics; neither
    reaches a feature vector.
    """
    lm = as_array(lm)
    cross = palm_cross_sign(lm)
    if abs(cross) < 1e-9:
        return "degenerate"
    return "mirrored" if should_mirror(lm) else "canonical"


def mirror_hand(lm: np.ndarray) -> np.ndarray:
    """Horizontal flip of a hand.

    Landmark indices are ANATOMICAL (1-4 is always that hand's thumb, 17-20
    always its pinky), so flipping the image horizontally must NOT permute
    them - it only negates x. Negating x turns a left hand into a right hand
    shape with the same numbering. (The commonly copy-pasted chain-reversal
    index map is a different operation and breaks the chirality test below.)
    """
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


# ---------------------------------------------------------------------------
# the one public entry point
# ---------------------------------------------------------------------------
def features_from_normalized_batch(lms: np.ndarray, blocks=None) -> np.ndarray:
    """(N, 21, 3) prepared hands -> (N, D) features. Vectorised reference impl.

    Every feature in this project is computed HERE, once. The single-hand path
    and the training path both call this, so live inference cannot differ from
    training by even a rounding order.
    """
    blocks = tuple(blocks or FEATURE_BLOCKS)
    lms = np.asarray(lms, dtype=np.float64)
    if lms.ndim == 2:
        lms = lms[None, :, :]
    n = lms.shape[0]
    # MediaPipe's z is documented to be on an arbitrary scale and does not
    # transfer between datasets, so distances/angles follow whichever
    # coordinate block is enabled.
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
        # (offset, along) of the thumb tip against each finger's MCP->PIP bone,
        # both in units of that bone. Positive offset is the side the thumb
        # normally sits on for a canonicalised hand; along > 1 means the thumb
        # tip has travelled past that finger's second joint.
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
    """Unit vector from joint i to joint j in the IMAGE PLANE, batched.

    Always 2-D even when the xyz block is on: the in-plane heading is what
    separates P from K and Q from G, and MediaPipe's z scale does not transfer
    between datasets, so a 3-D heading would be dataset-dependent.
    """
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
    """(21,3) raw or MediaPipe landmarks -> flat feature vector.

    THIS is what src/live.py calls, and what training uses for un-augmented
    samples, so the two paths cannot drift.

    `blocks` defaults to None, not to FEATURE_BLOCKS, on purpose: a default
    argument is frozen at import time, so anything that sets
    normalize.FEATURE_BLOCKS at runtime (the ablation sweeps do) would silently
    keep getting the import-time feature set here while training got the new
    one. Resolving None inside the batch impl makes both paths read the same
    live value.
    """
    return features_from_normalized(prepare(landmarks), blocks)


def features_matrix(landmark_list, blocks=None) -> np.ndarray:
    return np.stack([features(lm, blocks) for lm in landmark_list])


_PROBE_HAND = (np.arange(63, dtype=np.float64).reshape(21, 3) - 31.0) / 63.0


def feature_dim(blocks=None) -> int:
    """How many columns features() returns for the current feature blocks.

    Measured by calling the real function on a throwaway hand rather than from a
    table of per-block widths, so it cannot drift out of sync with the code.
    Used by live.py to refuse a model that was trained on a different set.
    """
    return int(features(_PROBE_HAND, blocks).shape[0])


def normalize_for_drawing(landmarks) -> np.ndarray:
    """Canonicalised, centred, scaled 21x3 - only used by the visualiser in
    src/inspect_data.py, never by the model."""
    return prepare(landmarks)


if __name__ == "__main__":
    dim = features(np.zeros((21, 3)) + np.arange(63).reshape(21, 3)).shape[0]
    print(f"blocks={FEATURE_BLOCKS} feature_dim={dim}")
    print(f"xyz=63 tip_dists={len(FINGERTIPS)*(len(FINGERTIPS)-1)//2} curl={len(FINGER_CHAINS)}")
