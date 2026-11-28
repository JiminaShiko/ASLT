"""Step 5 - the live webcam recogniser (MediaPipe -> normalize -> MLP, in memory).

Smoothing: confidence gate, top1-top2 margin gate (per-pair margins for measured
pairs), majority vote, hold-to-commit, no-repeat. Keys: SPACE commit, BACKSPACE,
C clear, Q quit. J and Z come from MotionDecider, not from a single frame: the
trained stroke head names them when it is installed, the trajectory rule is the
fallback. --selftest is the guided drill; nothing is ever written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np  # noqa: E402

if __package__:
    from . import config, hand_backend as hb, normalize  # noqa: E402
    from . import jz_features, jz_model, smoothing  # noqa: E402
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # noqa: E402
    import hand_backend as hb  # noqa: E402
    import normalize  # noqa: E402
    import jz_features  # noqa: E402
    import jz_model  # noqa: E402
    import smoothing  # noqa: E402


# One process, one clock: all three loops share one VIDEO detector, and two
# epochs make MediaPipe reject the second loop's first timestamp.
_TS_EPOCH = time.monotonic()


def stamp_ms() -> int:
    """Milliseconds since this process started - the only VIDEO timestamp."""
    return int((time.monotonic() - _TS_EPOCH) * 1000)


class TorchBackend:
    name = "torch"

    def __init__(self, weights: Path, profile: str | None = None):
        import torch

        if __package__:
            from . import model as M
        else:
            import model as M

        meta = json.loads(config.meta_path(profile).read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        self.net = M.load_model(weights, meta["model"]["in_dim"], len(self.labels),
                                tuple(meta["model"]["hidden"]), meta["model"]["dropout"])
        self.torch = torch

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(feats[None, :]))
            p = self.torch.softmax(logits, 1).numpy()[0]
        return p


class OnnxBackend:
    name = "onnx"

    def __init__(self, path: Path, profile: str | None = None):
        import onnxruntime as ort

        meta = json.loads(config.meta_path(profile).read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        logits = self.sess.run(None, {self.input_name: feats[None, :].astype(np.float32)})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()


def build_backend(prefer: str = "torch", profile: str | None = None):
    """Load one trained profile; the one place every front end resolves the same file."""
    profile = config.profile(profile)
    meta = json.loads(config.meta_path(profile).read_text(encoding="utf-8"))
    # Fail before the first frame if the model's feature set differs from normalize.py.
    want, have = normalize.feature_dim(), int(meta["model"]["in_dim"])
    if want != have:
        raise SystemExit(
            f"Feature mismatch: profile '{profile}' takes {have} inputs, but "
            f"normalize.features() now returns {want} "
            f"({'+'.join(normalize.FEATURE_BLOCKS)}).\n"
            f"Retrain to match it  ->  python src/train.py --profile {profile}\n"
            "or pick the old set ->  set ASL_FEATURE_BLOCKS to the blocks that "
            "model was trained with.")
    if prefer == "onnx" and config.onnx_path(profile).exists():
        return OnnxBackend(config.onnx_path(profile), profile), meta["labels"]
    path = config.model_path(profile, must_exist=True)
    return TorchBackend(path, profile), meta["labels"]


class Smoother:
    """Confidence gate -> majority vote -> hold-to-commit -> no-repeat.

    The hold timer runs from the first frame the hand looked like that letter, not
    from when the vote locked in.
    """

    def __init__(self, conf: float = 0.7, vote: int = 7, hold: float = 0.5,
                 min_votes: int | None = None):
        self.conf = conf
        self.vote = vote
        self.hold = hold
        self.min_votes = min_votes or max(2, (vote + 1) // 2)
        self.buf: deque[tuple[str, float]] = deque(maxlen=vote)
        self.run_letter: str | None = None   # current confident single-frame guess
        self.run_since: float = 0.0
        self.pending: str | None = None      # current majority winner
        self.pending_since: float = 0.0
        self.last_committed: str | None = None
        self.empty_streak = 0

    def step(self, letter: str | None, confidence: float, now: float,
             force: bool = False, hand_present: bool = True,
             gate: float | None = None
             ) -> tuple[str | None, str | None, float]:
        """Return (held_letter, letter_committed_now, mean_confidence_of_held).

        `gate` overrides this instance's confidence floor for one call. The web
        typer passes 0.0 so the highest-probability letter is always eligible
        to vote; the vote window and hold timer, not the confidence gate, then
        decide what is typed. Omit it for the original gate-at-`conf` behaviour.
        """
        floor = self.conf if gate is None else float(gate)
        confident = letter if (hand_present and letter is not None
                               and confidence >= floor) else None
        self.buf.append((confident or "", confidence))

        # streak of the same confident letter, for the hold timer
        if confident != self.run_letter:
            self.run_letter = confident
            self.run_since = now

        # a hand that leaves the frame resets the no-repeat lock
        if not hand_present:
            self.empty_streak += 1
            if self.empty_streak >= 2:
                self.last_committed = None
                self.pending = None
                self.pending_since = now
        else:
            self.empty_streak = 0

        votes = Counter(l for l, _c in self.buf if l)
        held, held_conf = None, 0.0
        if votes:
            top, top_n = votes.most_common(1)[0]
            if top_n >= self.min_votes:
                held = top
                confs = [c for l, c in self.buf if l == top]
                held_conf = float(np.mean(confs)) if confs else 0.0
        if held != self.pending:
            self.pending = held
            self.pending_since = now

        committed = None
        if force:
            # SPACE bypasses the vote window and timer; no-repeat still applies
            if self.run_letter is not None and self.run_letter != self.last_committed:
                committed = self.run_letter
        elif held is not None and (now - self.run_since) >= self.hold - 1e-6 \
                and held != self.last_committed:
            committed = held
        if committed is not None:
            self.last_committed = committed
            self.run_since = now
            self.pending_since = now
        return held, committed, (held_conf if held else 0.0)

    def hold_progress(self, now: float) -> float:
        """0..1 - how far through the hold-to-commit timer the current letter is."""
        if self.run_letter is None or self.hold <= 0:
            return 0.0
        return max(0.0, min(1.0, (now - self.run_since) / self.hold))


def top3_of(probs: np.ndarray, labels: list[str]) -> list[tuple[str, float]]:
    """The three most likely letters with their probabilities, best first."""
    order = np.argsort(probs)[::-1][:3]
    return [(labels[int(i)], float(probs[int(i)])) for i in order]


# Extra top1-top2 gap required for measured pairs, chosen by replaying the live
# vote window over reports/oof_all.csv (held letters: 13,094 wrong -> 10,905).
# frame_call takes max() with --margin, so this can only ever refuse more.
PAIR_MARGINS: dict[str, float] = {"PQ": 0.50, "AS": 0.50, "MN": 0.50, "GH": 0.50}


def pair_key(a: str, b: str) -> str:
    """Order-insensitive pair key: the rule covers the shape, not the direction."""
    return "".join(sorted((a, b)))


def frame_call(probs: np.ndarray, labels: list[str], conf: float,
               margin: float, pairs: dict[str, float] | None = None) -> dict:
    """Decide what one frame is allowed to say.

    `letter` is None for two independent reasons: `low_conf` (under the gate) or
    `close` (gap under the margin this pair needs). `need` is the margin actually
    applied and `collide`/`pair` name a binding pair rule.
    """
    pairs = PAIR_MARGINS if pairs is None else pairs
    order = np.argsort(probs)[::-1]
    i1 = int(order[0])
    i2 = int(order[1]) if len(order) > 1 else -1
    p1 = float(probs[i1])
    p2 = float(probs[i2]) if i2 >= 0 else 0.0
    m = p1 - p2
    key = pair_key(labels[i1], labels[i2]) if i2 >= 0 else ""
    raise_to = float((pairs or {}).get(key, 0.0))
    need = max(margin, raise_to)
    low_conf, close = p1 < conf, m < need
    return {"letter": None if (low_conf or close) else labels[i1],
            "conf": p1, "margin": m, "need": need,
            "low_conf": low_conf, "close": close,
            "collide": raise_to > margin and m < raise_to, "pair": key,
            "uncertain": close, "top3": top3_of(probs, labels)}


def blank_call() -> dict:
    """frame_call's result for a frame with no hand; must carry exactly the same keys."""
    return {"letter": None, "conf": 0.0, "margin": 0.0, "need": 0.0,
            "low_conf": False, "close": False, "collide": False, "pair": "",
            "uncertain": False, "top3": []}


def format_pair_margins(pairs: dict[str, float]) -> str:
    """`{'PQ': 0.5}` -> `'PQ=0.50'` - what `--pair-margins` and the app print."""
    return ",".join(f"{k}={pairs[k]:.2f}" for k in sorted(pairs)) or "none"


def resolve_pair_margins(spec: str | None) -> dict[str, float]:
    """`--pair-margins` -> table: None keeps PAIR_MARGINS, 'none' clears it, else a
    `PQ=0.4,AS=0.3` list that replaces it. Bad input raises."""
    
    if spec is None:
        return dict(PAIR_MARGINS)
    if spec.strip().lower() in ("", "none", "off", "flat"):
        return {}
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        k = "".join(sorted(k.strip().upper()))
        if not sep:
            raise ValueError(f"{part!r} is not PAIR=VALUE, e.g. PQ=0.40")
        if len(k) != 2 or not set(k) <= set(config.LABELS):
            raise ValueError(
                f"pair {k!r} must be two letters of the 24-class alphabet "
                f"({''.join(config.LABELS)}); J and Z come from the motion rule, "
                f"not the classifier")
        try:
            val = float(v)
        except ValueError:
            raise ValueError(f"{k}={v!r} is not a number")
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{k}={val} is outside 0..1")
        out[k] = val
    if not out:
        raise ValueError(f"--pair-margins {spec!r} parsed to nothing")
    return out


class MotionDecider:
    """Name J and Z from movement, never from a single frame.

    The per-frame classifier only opens the check: an I opens J, an X opens Z.
    When the trained stroke head is installed it decides both letters over the
    window. Measured on the MSL corpus it fired J on 102/102 real J strokes and
    never on K, X or N-tilde, where the old wrist-relative rule fired on none of
    the 102. A held hand is still never a J: the head's J only counts once the
    pinky tip has travelled J_MOTION across the frame, which a still hand
    measures at 0.00. Without the head, the corrected trajectory rule decides.
    """

    WINDOW = 1.20        # s of trajectory kept
    MIN_HOLD = 0.40      # s the J/Z handshape must be held before judging
    COOLDOWN = 1.10      # s of silence after a letter fires
    SEG = 0.18           # palm units of real movement needed to count a reversal
    # J. Measured on the pinky tip's own track across the frame, because a J is
    # drawn by the hand travelling: demanding a still wrist while the fingertip
    # moves is the Z stroke, and it fired on 0 of 102 real J clips.
    J_NET = 0.80         # pinky-tip net travel that counts as a whole stroke
    # The stroke head judges the *shape* of a window, so it calls a motionless I
    # a J with probability 1.00. This is the separate "the hand moved at all"
    # bar: a held hand measures <= 0.05 here after smoothing, a signed J more.
    J_MOTION = 0.30      # pinky-tip travel before the head's J is believed
    # Z
    Z_PATH = 0.90        # total sideways path of the fingertip
    Z_DROP = 0.30        # the stroke must also go down/up, not just wobble
    Z_WRIST_MAX = 0.45   # the hand itself must be roughly still

    # Shipped defaults; instances copy these and calibration only ever loosens them.
    DEFAULTS = {"WINDOW": WINDOW, "MIN_HOLD": MIN_HOLD, "COOLDOWN": COOLDOWN,
                "SEG": SEG, "J_NET": J_NET, "J_MOTION": J_MOTION,
                "Z_PATH": Z_PATH, "Z_DROP": Z_DROP, "Z_WRIST_MAX": Z_WRIST_MAX}
    # Permissive limit per key, so a sloppy calibration cannot make J/Z fire on
    # everything. COOLDOWN equals its default: it is never calibrated.
    FLOORS = {"WINDOW": 2.00, "MIN_HOLD": 0.25, "COOLDOWN": COOLDOWN,
              "SEG": 0.05, "J_NET": 0.12, "J_MOTION": 0.10, "Z_PATH": 0.45,
              "Z_DROP": 0.12, "Z_WRIST_MAX": 0.90}

    # Display precondition: which classifier reading opens the watch. Not in
    # DEFAULTS on purpose, so --calibrate-jz can never move them.
    # I and X are static predictions that can open their corresponding motion
    # checks. The trajectory decides whether either one was actually moving.
    DISPLAY_FOR = {"I": "J", "X": "Z"}
    DISPLAY_P1_MAX = 0.90      # retained for compatibility with old reports
    DISPLAY_MIN_SHARE = 0.60   # of the judged window, how many frames must agree
    # No confidence ceiling: a confident I or X is still only a static frame,
    # while the stroke head judges whether its motion is J/Z or an ordinary hold.
    DISPLAY_P1_MAX_FOR = {"J": None, "Z": None}
    STATIC_FOR = {"J": "I", "Z": "X"}
    AUTO_MOTION_LETTERS = ("J", "Z")

    def __init__(self, use_model: bool = True):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)      # per-instance copy; the class stays clean
        self.threshold_source = ""   # "defaults" / "--jz ..." / "calibrated"
        self.strokes: list[dict] = []    # recorded stroke series (in memory only)
        self.intended: str | None = None  # which letter the next stroke is for
        self._seg: list[tuple[float, float, float, float, float, float, str | None]] = []
        self.holding = False
        self.last_flags = (False, False)
        # The trained head reads raw landmarks, so the ring keeps them as well as
        # the derived streams the fallback rule needs.
        self.jz = jz_model.JZModel.load() if use_model else None
        self.lms: deque = deque(maxlen=400)        # (t, (21,3) landmarks)
        self.last_probs: dict[str, float] = {}
        self.t: deque[float] = deque(maxlen=400)
        self.wx: deque[float] = deque(maxlen=400)   # wrist x, palm units
        # Two tips, because the letters use different fingers: J is drawn with
        # the pinky, Z with the index. Both are tracked every frame, so a
        # classifier reading can never swap one finger's path for the other's.
        self.ix: deque[float] = deque(maxlen=400)   # index tip x relative to wrist
        self.iy: deque[float] = deque(maxlen=400)   # index tip y
        self.px: deque[float] = deque(maxlen=400)   # pinky tip x across the frame
        self.py: deque[float] = deque(maxlen=400)   # pinky tip y across the frame
        # per-frame classifier candidate ("J", "Z" or None), aligned with wx/ix
        self.dq: deque[str | None] = deque(maxlen=400)
        self.hold_since: float | None = None
        self.miss = 0                # consecutive frames that failed both gates
        self.fired_at: float = -1e9
        self.last_now: float | None = None   # the timestamp `hint()` reads
        self.last: str | None = None        # what fired most recently (for overlay)
        self.last_top1: str | None = None   # the classifier reading of the frame
        self.last_p1: float = 0.0
        self.candidates: tuple[str, ...] = ()   # what the gates allow right now
        self.debug: dict = {}
        self.last_debug: dict = {}          # the measurements that produced `last`

    def reset(self):
        """Forget the trajectory; the segment that ended is archived for calibration."""
        self._close_stroke()
        self.t.clear(); self.wx.clear(); self.dq.clear(); self.lms.clear()
        self.ix.clear(); self.iy.clear(); self.px.clear(); self.py.clear()
        self.hold_since = None
        self.miss = 0
        self.holding = False
        self.candidates = ()

    def blocks_static(self, letter: str | None, now: float) -> bool:
        """Hold I/X votes while their matching J/Z stroke is still being judged.

        A still I or X is released after one motion window, so a later deliberate
        stroke can still be recognized. A fired J/Z masks its paired static letter during
        the motion cooldown, preventing one gesture from typing both.
        """
        if letter not in self.STATIC_FOR.values():
            return False
        if (self.last in self.STATIC_FOR
                and self.STATIC_FOR[self.last] == letter
                and now - self.fired_at < self.COOLDOWN):
            return True
        if not self.holding:
            return False
        candidate = self.display_candidate(self.last_top1, self.last_p1)
        relevant = (letter == "I" and (candidate == "J" or
                                         self.debug.get("gate") == "shape")
                    or letter == "X" and candidate == "Z")
        if not relevant:
            return False
        if self.hold_since is not None and now - self.hold_since < self.WINDOW:
            return True
        return False

    MIN_STROKE_S = 0.20       # shorter than this is a twitch, not a stroke
    MIN_STROKE_FRAMES = 6
    MAX_STROKES = 40          # a calibration needs 6; this is only a memory guard

    def _close_stroke(self) -> None:
        """Archive the segment that just ended, for --calibrate-jz.

        Gated on `self.intended` (only the calibrator sets it) and trimmed to the
        trailing WINDOW, the exact span update() judges.
        """
        seg, self._seg = self._seg, []
        if self.intended is None:
            return
        if len(seg) < self.MIN_STROKE_FRAMES:
            return
        held_full = seg[-1][0] - seg[0][0]
        if held_full < self.MIN_STROKE_S:
            return
        t0 = seg[-1][0] - self.WINDOW
        seg = [s for s in seg if s[0] >= t0]
        if len(seg) < self.MIN_STROKE_FRAMES:
            return
        wx = [s[1] for s in seg]
        ix = [s[2] for s in seg]
        iy = [s[3] for s in seg]
        px = [s[4] for s in seg]
        py = [s[5] for s in seg]
        # share of frames the classifier reading supported; reported, never a threshold
        cand = [s[6] for s in seg]
        share = sum(1 for v in cand if v == self.intended) / max(len(cand), 1)
        # the same smoothed track `_criteria` judges, so calibration derives
        # thresholds from the quantity that is actually compared
        pxs, pys = self._smooth(px, py)
        self.strokes.append({
            "intended": self.intended,
            "held": seg[-1][0] - seg[0][0], "held_full": held_full,
            "n": len(seg), "wx": wx,
            "ix": ix, "iy": iy, "px": px, "py": py,
            "disp_share": share,
            "wrist_net": abs(wx[-1] - wx[0]), "wrist_path": self._path(wx),
            "wrist_range": max(wx) - min(wx),
            # the index tip draws Z, the pinky tip draws J
            "tip_net": abs(ix[-1] - ix[0]), "tip_path": self._path(ix),
            "tip_net_2d": float(np.hypot(ix[-1] - ix[0], iy[-1] - iy[0])),
            "tip_path_2d": self._path2d(ix, iy),
            "tip_range": max(ix) - min(ix), "drop": abs(iy[-1] - iy[0]),
            "pinky_net_2d": float(np.hypot(pxs[-1] - pxs[0], pys[-1] - pys[0])),
            "pinky_path_2d": self._path2d(pxs, pys),
            "pinky_range": max(pxs) - min(pxs), "pinky_drop": abs(pys[-1] - pys[0]),
        })
        if len(self.strokes) > self.MAX_STROKES:
            del self.strokes[0]

    @staticmethod
    def _path(xs) -> float:
        a = np.asarray(xs, dtype=np.float64)
        return float(np.abs(np.diff(a)).sum()) if len(a) > 1 else 0.0

    @staticmethod
    def _path2d(xs, ys) -> float:
        a = np.column_stack((xs, ys)).astype(np.float64)
        return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) if len(a) > 1 else 0.0

    @staticmethod
    def _smooth(xs, ys, k: int = 5):
        """(x, y) averaged over k frames: keeps the stroke, drops the jitter."""
        a = np.column_stack((np.asarray(xs, dtype=np.float64),
                             np.asarray(ys, dtype=np.float64)))
        if len(a) < k:
            return a[:, 0], a[:, 1]
        ker = np.ones(k) / k
        return (np.convolve(a[:, 0], ker, mode="valid"),
                np.convolve(a[:, 1], ker, mode="valid"))

    @staticmethod
    def _travel(xs, ys, k: int = 5) -> float:
        """Pinky-tip displacement between the window's ends, jitter averaged out.

        Comparing two raw endpoints measures the tracker's noise as well as the
        stroke, so a still hand reads as moving.
        """
        sx, sy = MotionDecider._smooth(xs, ys, k)
        if len(sx) < 1:
            return 0.0
        return float(np.hypot(sx[-1] - sx[0], sy[-1] - sy[0]))

    @staticmethod
    def _reversals(xs, seg: float) -> int:
        """Count direction changes, ignoring wiggles smaller than `seg`."""
        n, last_dir, anchor = 0, 0, None
        for x in xs:
            if anchor is None:
                anchor = x
                continue
            d = x - anchor
            if abs(d) < seg:
                continue
            direction = 1 if d > 0 else -1
            if last_dir and direction != last_dir:
                n += 1
            last_dir, anchor = direction, x
        return n

    @staticmethod
    def handshape(lm: np.ndarray) -> tuple[bool, float]:
        """(index-up or pinky-up with other fingers curled, index ratio)."""
        chains = config.FINGER_CHAINS          # [(mcp, pip, dip, tip), ...]
        ratios = []
        for mcp, pip, _dip, tip in chains:
            a = float(np.linalg.norm(lm[pip, :2] - lm[mcp, :2]))
            b = float(np.linalg.norm(lm[tip, :2] - lm[mcp, :2]))
            ratios.append(b / max(a, 1e-6))
        idx_ext, pinky_ext = ratios[1], ratios[4]
        index_shape = idx_ext >= 1.6 and all(r <= 1.20 for r in ratios[2:])
        pinky_shape = pinky_ext >= 1.6 and all(r <= 1.20 for r in ratios[1:4])
        up = index_shape or pinky_shape
        return bool(up), idx_ext

    @staticmethod
    def _pinky_extended(lm: np.ndarray) -> bool:
        mcp, pip, _dip, tip = config.FINGER_CHAINS[4]
        base = float(np.linalg.norm(lm[pip, :2] - lm[mcp, :2]))
        length = float(np.linalg.norm(lm[tip, :2] - lm[mcp, :2]))
        return length / max(base, 1e-6) >= 1.6

    @classmethod
    def display_candidate(cls, top1, p1) -> str | None:
        """The J/Z candidate the classifier's own reading names, or None.

        Static I/X readings only open the motion check; the trajectory decides
        whether the movement is J/Z. None when there is no usable reading, so
        landmark-only callers keep the shape-gate behaviour.
        """
        if top1 is None or p1 is None:
            return None
        want = cls.DISPLAY_FOR.get(str(top1))
        if want is None:
            return None
        try:
            p1 = float(p1)
        except (TypeError, ValueError):
            return None
        limit = cls.DISPLAY_P1_MAX_FOR.get(want, cls.DISPLAY_P1_MAX)
        return want if limit is None or p1 < limit else None

    def _allowed(self, up: bool, dq) -> tuple[str, ...]:
        """Letters the preconditions permit over the judged window.

        The shape gate names no letter, so both stay possible; the display gate
        needs a majority reading (DISPLAY_MIN_SHARE), so one flicker to I cannot
        let a hand travelling between letters be judged as a J.
        """
        allowed = set()
        if up:
            allowed.update(self.AUTO_MOTION_LETTERS)
        n = max(len(dq), 1)
        for letter in self.AUTO_MOTION_LETTERS:
            if sum(1 for v in dq if v == letter) / n >= self.DISPLAY_MIN_SHARE:
                allowed.add(letter)
        return tuple(sorted(allowed))

    @staticmethod
    def _criteria(wx, ix, iy, px, py, th: dict) -> dict:
        """(name, value, need, ok) per criterion; shared by firing and the hint text.

        J is measured on the pinky tip's own track across the frame and Z on the
        index tip relative to the wrist: a J moves the whole hand, a Z is drawn
        with the fingertip. Judging one on the other's path is exactly how a Z
        used to be read as a J, and how a real J read as no movement at all.
        """
        S = th["SEG"]
        # The pinky's own track is smoothed before it is measured: the endpoint
        # difference is the stroke, while a raw path length over a second is
        # mostly tracker jitter (a perfectly still hand measures more than a palm
        # of "path" at 5% noise, which is why a path threshold only ever fires
        # on noise or does nothing).
        pxs, pys = MotionDecider._smooth(px, py)
        wr = MotionDecider._reversals(wx, S)
        ir = MotionDecider._reversals(ix, S)
        # the hook of a J can turn in either axis: hook down then back, or across
        pr = max(MotionDecider._reversals(pxs, S), MotionDecider._reversals(pys, S))
        wnet = abs(wx[-1] - wx[0])
        inet = float(np.hypot(ix[-1] - ix[0], iy[-1] - iy[0]))
        ipath = MotionDecider._path2d(ix, iy)
        pnet = float(np.hypot(pxs[-1] - pxs[0], pys[-1] - pys[0]))
        drop = abs(iy[-1] - iy[0])
        return {
            "J": [("pinky hook", pr, 1, pr >= 1),
                  ("pinky net travel", pnet, th["J_NET"], pnet >= th["J_NET"])],
            "Z": [("tip reversals", ir, 2, ir >= 2),
                  ("tip path", ipath, th["Z_PATH"], ipath >= th["Z_PATH"]),
                  ("vertical drop", drop, th["Z_DROP"], drop >= th["Z_DROP"]),
                  ("hand still", wnet, th["Z_WRIST_MAX"], wnet <= th["Z_WRIST_MAX"]),
                  ("no hand hook", wr, 1, wr <= 1)],
        }

    @classmethod
    def _judge(cls, wx, ix, iy, px, py, th: dict) -> tuple[bool, bool]:
        """The two boolean rules; shared by update() and the calibrator replay."""
        crit = cls._criteria(wx, ix, iy, px, py, th)
        return (all(c[3] for c in crit["J"]), all(c[3] for c in crit["Z"]))

    def thresholds(self) -> dict:
        return {k: getattr(self, k) for k in self.DEFAULTS}

    def apply_thresholds(self, th: dict, source: str = "") -> dict:
        """Set this session's J/Z thresholds in memory; bad keys raise, never ignored."""
        bad = [k for k in th if k not in self.DEFAULTS]
        if bad:
            raise ValueError(
                f"unknown J/Z threshold key(s) {sorted(bad)}; the names are "
                f"{', '.join(sorted(self.DEFAULTS))}")
        for k, v in th.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"--jz {k}={v!r} is not a number")
            if not v >= 0.0 or v != v:          # catches NaN too
                raise ValueError(f"--jz {k}={v!r} must be a finite number >= 0")
            setattr(self, k, v)
        self.threshold_source = source or self.threshold_source
        return self.thresholds()

    def criteria(self) -> dict:
        """Per-criterion detail for the current trajectory window."""
        if len(self.wx) < 2:
            return {}
        return self._criteria(list(self.wx), list(self.ix), list(self.iy),
                              list(self.px), list(self.py), self.thresholds())

    def hint(self, width: int | None = None) -> str:
        """One sentence naming the criterion closest to firing, or "" if nothing to say."""
        if not self.holding:
            return ""
        if self.last_now is not None and self.last_now - self.fired_at < self.COOLDOWN:
            return ""          # still in the anti-double-typing gap
        held = self.debug.get("held", 0.0)
        if held < self.MIN_HOLD:
            if self.debug.get("cand") or self.debug.get("gate") == "display":
                return (f"the model reads this frame as "
                        f"{self.debug.get('top1') or '-'}, so J/Z is watching the "
                        f"movement - hold {self.MIN_HOLD - held:.2f}s longer "
                        f"before it is judged")
            return (f"hold the J/Z handshape {self.MIN_HOLD - held:.2f}s longer "
                    f"before J/Z is judged")
        if len(self.wx) < 2:
            return ""
        crit = self.criteria()
        if not crit:
            return ""
        # only explain letters the preconditions permit
        letters = self.candidates or ("J", "Z")
        for letter in ("J", "Z"):
            if letter not in letters:
                continue
            if all(c[3] for c in crit[letter]):
                return ""            # a rule is met; the fire needs no apology

        def binding(items):
            """Ratio of the criterion furthest from being met."""
            vals = [v / need if need else (1.0 if v > 0 else 0.0)
                    for _n, v, need, ok in items if not ok]
            return min(vals) if vals else 1.0

        def misses(items):
            return sorted((c for c in items if not c[3]),
                          key=lambda c: c[1] / c[2] if c[2] else 1.0)

        letter = max(letters, key=lambda L: binding(crit[L]))
        txt = f"{letter} held but not fired: " + ", ".join(
            f"{n} {v:.2f} needs {need:.2f}" for n, v, need, _ok in misses(crit[letter]))
        if width:
            txt = txt[:width]
        return txt

    def update(self, now: float, lm: np.ndarray, top1: str | None = None,
               p1: float | None = None) -> str | None:
        """Feed one frame; returns "J"/"Z" when a precondition holds and the path matches."""
        c = normalize.canonicalize(lm)
        palm = float(np.linalg.norm(c[config.MIDDLE_MCP_INDEX] - c[config.WRIST_INDEX]))
        if palm <= 1e-6:
            self.reset()
            return None
        up, idx_ext = self.handshape(c)
        cand = self.display_candidate(top1, p1)
        self.last_top1 = None if top1 is None else str(top1)
        self.last_p1 = 0.0 if p1 is None else float(p1)
        if now - self.fired_at < self.COOLDOWN:
            # Silent gap: the hand still relaxing out of a fired stroke is not a
            # second letter, so nothing from this span reaches the window.
            if self.t:
                self.t.clear(); self.wx.clear(); self.dq.clear()
                self.lms.clear(); self.ix.clear(); self.iy.clear()
                self.px.clear(); self.py.clear(); self._seg.clear()
                self.hold_since = None
                self.holding = False
                self.candidates = ()
            return None
        if not up and cand is None:
            # MediaPipe drops fingertips now and then; two bad frames end the gesture
            self.miss += 1
            if self.miss > 2:
                self.reset()
            return None
        self.miss = 0
        if self.hold_since is None:
            self.hold_since = now
        w = c[config.WRIST_INDEX, 0] / palm
        # ASL J is drawn with the pinky and Z with the index finger; both tips are
        # followed, so the verdict never depends on which finger the reading implies.
        wrist = c[config.WRIST_INDEX]
        itip = (c[config.FINGERTIPS[1]] - wrist) / palm
        # The pinky is tracked in frame coordinates: a real J travels through
        # space, and a wrist-relative track of it reads as "no movement at all".
        ptip = c[config.FINGERTIPS[4]] / palm
        self.t.append(now); self.wx.append(w)
        self.ix.append(float(itip[0])); self.iy.append(float(itip[1]))
        self.px.append(float(ptip[0])); self.py.append(float(ptip[1]))
        self.dq.append(cand)
        self.lms.append((now, np.asarray(lm, dtype=np.float64)))
        self._seg.append((now, w, float(itip[0]), float(itip[1]),
                          float(ptip[0]), float(ptip[1]), cand))
        self.holding = True          # a precondition met, whatever the path says
        self.last_now = now
        while len(self.t) > 2 and now - self.t[0] > self.WINDOW:
            self.t.popleft(); self.wx.popleft(); self.dq.popleft()
            self.ix.popleft(); self.iy.popleft()
            self.px.popleft(); self.py.popleft()
        while len(self.lms) > 2 and now - self.lms[0][0] > self.WINDOW:
            self.lms.popleft()

        wx_now = list(self.wx)
        ix_now, iy_now = list(self.ix), list(self.iy)
        px_now, py_now = list(self.px), list(self.py)
        self.debug = {"held": now - self.hold_since, "idx_ext": idx_ext,
                      "top1": self.last_top1, "p1": self.last_p1,
                      "gate": "shape" if up else "display",
                      "cand": cand,
                      "wrist_path": self._path(wx_now),
                      "wrist_net": abs(wx_now[-1] - wx_now[0]),
                      # "tip_*" is the index tip (Z); "pinky_*" is the pinky (J)
                      "tip_path": self._path2d(ix_now, iy_now),
                      "tip_net": float(np.hypot(ix_now[-1] - ix_now[0],
                                                iy_now[-1] - iy_now[0])),
                      "pinky_path": self._path2d(px_now, py_now),
                      "pinky_net": float(np.hypot(px_now[-1] - px_now[0],
                                                  py_now[-1] - py_now[0]))}
        if now - self.hold_since < self.MIN_HOLD or now - self.fired_at < self.COOLDOWN:
            return None

        allowed = self._allowed(up, self.dq)
        self.candidates = allowed
        wr = self._reversals(wx_now, self.SEG)
        tr = self._reversals(ix_now, self.SEG)
        pr = self._reversals(px_now, self.SEG)
        drop = abs(iy_now[-1] - iy_now[0])
        self.debug.update({"wrist_rev": wr, "tip_rev": tr, "pinky_rev": pr,
                           "drop": drop, "allowed": list(allowed)})
        hit_j, hit_z = self._judge(wx_now, ix_now, iy_now, px_now, py_now,
                                   self.thresholds())
        self.last_flags = (hit_j, hit_z)
        # How far the pinky tip itself travelled across the frame. A J is a stroke
        # through space; a held hand measures 0.00 here, whatever the head thinks
        # of its shape (it reads a still I as J with probability 1.00).
        travel = self._travel(px_now, py_now)
        self.debug["pinky_travel"] = travel
        out = None
        if self.jz is not None:
            self.debug["decider"] = "stroke head (J needs pinky travel)"
            span = self.lms[-1][0] - self.lms[0][0] if len(self.lms) > 1 else 0.0
            if allowed and len(self.lms) >= 6 and span >= 0.5:
                feats = jz_features.features_from_frames(
                    [frame for _t, frame in self.lms],
                    [t for t, _f in self.lms])
                letter, probs = self.jz.decide(feats, allowed)
                self.last_probs = {k: round(v, 4) for k, v in probs.items()}
                self.debug["jz_probs"] = self.last_probs
                if letter == "J" and travel < self.J_MOTION:
                    self.debug["blocked"] = (
                        f"the head reads J, but the pinky moved {travel:.2f} "
                        f"< {self.J_MOTION:.2f}: holding a shape is not a J")
                    letter = None
                out = letter
        else:
            self.debug["decider"] = "trajectory J/Z"

        # Without the head the hand-calibratable rule decides, and only then: the
        # rule names a J on 64 of 104 moving X strokes where the head names none,
        # so letting it overrule the head would trade a rare miss for a common
        # false letter.
        if out is None and self.jz is None:
            if "J" in allowed and hit_j:
                out = "J"
            elif "Z" in allowed and hit_z:
                out = "Z"

        if (hit_j or hit_z) and not allowed:
            # path matched but no precondition survived the window
            self.debug["blocked"] = "path matched, precondition did not hold"
        if out:
            self.last_debug = dict(self.debug)
            self.fired_at = now
            self.last = out
            self.reset()
        return out


CALIBRATION_SAFETY = 0.72
# every stroke measurement a threshold can be derived from
STROKE_KEYS = ("MIN_HOLD", "SEG", "WINDOW", "J_NET", "J_MOTION",
               "Z_PATH", "Z_DROP", "Z_WRIST_MAX")


def _stroke_measures(strokes: list[dict], letter: str | None = None) -> list[dict]:
    return [s for s in strokes
            if s.get("intended") == letter and len(s.get("wx", [])) > 2]


def derive_thresholds(strokes: list[dict],
                      safety: float = CALIBRATION_SAFETY) -> tuple[dict, dict]:
    """Recorded strokes -> (thresholds, notes). Pure, so the logic test can check it.

    Thresholds only ever move in the permissive direction, capped by
    MotionDecider.FLOORS; `safety` leaves headroom; SEG is searched, WINDOW only
    raised, COOLDOWN never derived. Letters with no strokes keep the defaults.
    """
    D, F = MotionDecider.DEFAULTS, MotionDecider.FLOORS
    th = dict(D)
    notes: dict[str, str] = {}

    def loosen_below(key: str, value: float, why: str) -> None:
        new = max(F[key], min(D[key], value))
        th[key] = round(new, 4)
        notes[key] = (f"{new:.2f} <- {why}" if new < D[key]
                      else f"{new:.2f} (default {D[key]:.2f}; {why})")

    def widen_above(key: str, value: float, why: str) -> None:
        new = min(F[key], max(D[key], value))
        th[key] = round(new, 4)
        notes[key] = (f"{new:.2f} <- {why}" if new > D[key]
                      else f"{new:.2f} (default {D[key]:.2f}; {why})")

    js, zs = _stroke_measures(strokes, "J"), _stroke_measures(strokes, "Z")
    usable = js + zs
    if not usable:
        return dict(D), {"note": "no strokes recorded - all defaults kept"}

    hold = min(s["held_full"] for s in usable)
    loosen_below("MIN_HOLD", safety * hold, f"shortest hold {hold:.2f}s x {safety}")
    longest = max(s["held_full"] for s in usable)
    if longest > th["WINDOW"]:
        new = min(F["WINDOW"], longest + 0.15)
        th["WINDOW"] = round(new, 4)
        notes["WINDOW"] = (f"{new:.2f} <- longest stroke {longest:.2f}s "
                           f"(default {D['WINDOW']:.2f}s)")
    else:
        notes["WINDOW"] = f"{th['WINDOW']:.2f} (default; fits every stroke)"
    notes["COOLDOWN"] = f"{th['COOLDOWN']:.2f} (default; never calibrated)"

    if js:
        smallest = min(s["pinky_net_2d"] for s in js)
        loosen_below("J_NET", safety * smallest,
                     f"smallest pinky travel {smallest:.2f}")
        loosen_below("J_MOTION", safety * smallest,
                     f"smallest pinky travel {smallest:.2f}")
    else:
        for k in ("J_NET", "J_MOTION"):
            notes[k] = f"{th[k]:.2f} (default; no J strokes recorded)"
    if zs:
        loosen_below("Z_PATH", safety * min(s["tip_path"] for s in zs),
                     f"shortest zig-zag {min(s['tip_path'] for s in zs):.2f}")
        loosen_below("Z_DROP", safety * min(s["drop"] for s in zs),
                     f"least vertical travel {min(s['drop'] for s in zs):.2f}")
        widen_above("Z_WRIST_MAX", max(s["wrist_net"] for s in zs) / safety,
                    f"most hand drift {max(s['wrist_net'] for s in zs):.2f}")
    else:
        for k in ("Z_PATH", "Z_DROP", "Z_WRIST_MAX"):
            notes[k] = f"{th[k]:.2f} (default; no Z strokes recorded)"

    # reversal hysteresis: largest SEG that still counts every stroke's turns
    grid = [round(x, 4) for x in
            np.linspace(F["SEG"], D["SEG"], 13)][::-1]
    best_seg, best_hits = D["SEG"], -1
    for seg in grid:
        cand = dict(th)
        cand["SEG"] = seg
        hits = sum(1 for s in usable
                   if _stroke_fires(s, cand) == s["intended"])
        if hits > best_hits:
            best_seg, best_hits = seg, hits
    th["SEG"] = best_seg
    notes["SEG"] = (f"{best_seg:.2f} <- counts the reversals of "
                    f"{best_hits}/{len(usable)} strokes "
                    f"(default {D['SEG']:.2f})")
    return th, notes


def _stroke_fires(stroke: dict, th: dict) -> str | None:
    """Letter a recorded stroke produces under `th` (Z wins a tie, as live)."""
    hit_j, hit_z = MotionDecider._judge(stroke["wx"], stroke["ix"], stroke["iy"],
                                        stroke["px"], stroke["py"], th)
    return "Z" if hit_z else ("J" if hit_j else None)


def self_consistency(strokes: list[dict], th: dict) -> dict:
    """Replay recorded strokes through _judge(), the function the live path uses."""
    out: dict[str, object] = {}
    per = {}
    for letter in ("J", "Z"):
        ss = _stroke_measures(strokes, letter)
        hits = sum(1 for s in ss if _stroke_fires(s, th) == letter)
        per[letter] = (hits, len(ss))
    tot = sum(v[1] for v in per.values())
    hit = sum(v[0] for v in per.values())
    false = sum(1 for s in strokes
                if _stroke_fires(s, th) not in (None, s.get("intended")))
    out["per_letter"], out["fired"], out["n"] = per, hit, tot
    out["rate"] = hit / tot if tot else float("nan")
    out["wrong_letter"] = false
    return out


def format_spec(th: dict) -> str:
    """A pasteable `--jz` override: only the keys that differ from the default."""
    diff = {k: v for k, v in th.items() if abs(v - MotionDecider.DEFAULTS[k]) > 1e-9}
    return ",".join(f"{k}={diff[k]:.4f}" for k in sorted(diff)) or ""


def parse_spec(spec: str) -> dict:
    """`--jz KEY=VALUE,...` -> a validated threshold dict (inverse of format_spec)."""
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{part!r} is not KEY=VALUE")
        k, _, v = part.partition("=")
        k = k.strip().upper()
        if k not in MotionDecider.DEFAULTS:
            raise ValueError(
                f"unknown key {k!r}; the names are {', '.join(sorted(MotionDecider.DEFAULTS))}")
        try:
            out[k] = float(v)
        except ValueError:
            raise ValueError(f"{k}={v!r} is not a number")
    return out


class JZCalibrator:
    """Guided J/Z calibration session, in memory only; shared by both front ends."""

    LETTERS = ("J", "Z")
    INSTRUCTIONS = {
        "J": "keep your hand steady, extend your pinky, and draw a J with the "
             "pinky tip",
        "Z": "same shape; keep the HAND still and draw the zig-zag with the "
             "FINGERTIP",
    }

    def __init__(self, motion: "MotionDecider", reps: int = 3,
                 safety: float = CALIBRATION_SAFETY):
        self.motion = motion
        self.reps = max(1, int(reps))
        self.safety = float(safety)
        self.letter = self.LETTERS[0]
        self.index = 0
        self.done = False
        self.await_release = False
        self.result: dict | None = None
        self._seen = 0
        motion.strokes.clear()
        motion.intended = self.letter
        motion.reset()

    # -- progress ----------------------------------------------------
    def collected(self) -> int:
        return len([s for s in self.motion.strokes if s["intended"] == self.letter])

    def progress(self) -> float:
        want = self.reps * len(self.LETTERS)
        return min(1.0, len(self.motion.strokes) / want) if want else 1.0

    def prompt(self) -> str:
        if self.done:
            return "calibration finished"
        if self.await_release:
            return (f"relax your hand, then sign {self.letter} "
                    f"({self.collected()}/{self.reps} recorded)")
        return (f"Calibrating {self.letter}  {self.collected()}/{self.reps}: "
                f"{self.INSTRUCTIONS[self.letter]}")

    # -- per frame ---------------------------------------------------
    def step(self, now: float, lm, top1: str | None = None,
              p1: float | None = None) -> str | None:
        """Feed one frame; returns the letter the live rule fired, if any.

        `top1`/`p1` matter: most hand-made Z strokes never pass the shape gate, so
        without the display reading too few Z strokes would be archived.
        """
        if self.done:
            return None
        fired = self.motion.update(now, lm, top1, p1)
        if self.await_release:
            # nothing is recorded until the handshape has really been released, or
            # a held J tail would be filed as a Z stroke
            if not self.motion.holding:
                self.await_release = False
                self.motion.intended = self.letter
            return fired
        strokes = self.motion.strokes
        if len(strokes) > self._seen:
            self._seen = len(strokes)
            if self.collected() >= self.reps:
                self._advance()
        return fired

    def _advance(self) -> None:
        self.motion.reset()
        self.index += 1
        if self.index >= len(self.LETTERS):
            self.finish()
        else:
            self.letter = self.LETTERS[self.index]
            self.motion.intended = None
            self.await_release = True

    def finish(self) -> dict:
        th, notes = derive_thresholds(self.motion.strokes, self.safety)
        cons = self_consistency(self.motion.strokes, th)
        base = self_consistency(self.motion.strokes, dict(MotionDecider.DEFAULTS))
        self.result = {
            "thresholds": th, "notes": notes, "consistency": cons,
            "baseline_consistency": base, "spec": format_spec(th),
            "n_strokes": len(self.motion.strokes), "safety": self.safety,
            "strokes": [{k: v for k, v in s.items() if k in
                         ("intended", "held", "held_full", "wrist_net", "wrist_path",
                          "tip_net", "tip_path", "drop")}
                        for s in self.motion.strokes],
        }
        self.done = True
        self.motion.intended = None
        self.motion.apply_thresholds(th, source="calibrated this session")
        return self.result

    def cancel(self) -> None:
        self.motion.intended = None
        self.motion.strokes.clear()
        self.done = True
        self.result = None


def calibrate_camera(cap, det, recovery, args, motion: MotionDecider,
                     backend=None, labels=None) -> dict | None:
    """Run JZCalibrator against the webcam; ESC skips a letter, Q keeps the defaults.

    Optional backend/labels let each stroke frame carry the classifier reading.
    """
    import cv2

    cal = JZCalibrator(motion, reps=args.calibrate_reps)
    print("\n  J/Z calibration - nothing is saved, and the numbers live in RAM only")
    print(f"  {cal.reps} strokes per letter (change with --calibrate-reps)")
    print(f"  J: {JZCalibrator.INSTRUCTIONS['J']}")
    print(f"  Z: {JZCalibrator.INSTRUCTIONS['Z']}")
    print("  press ESC to skip a letter, Q to stop")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("  camera stream ended")
            break
        now = time.time()
        res, _how = hb.detect_frame(det, frame, stamp_ms(),
                                    recovery, empty_streak=1)
        top1 = p1 = None
        if res.landmarks is not None:
            hb.draw_landmarks(frame, res.landmarks[:, :2])
            if backend is not None and labels is not None:
                probs = backend(normalize.features(res.landmarks))
                k1 = int(np.argsort(probs)[-1])
                top1, p1 = str(labels[k1]), float(probs[k1])
            cal.step(now, res.landmarks, top1, p1)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 96), (0, 0, 0), -1)
        cv2.putText(frame, cal.prompt(), (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 220, 255), 2, cv2.LINE_AA)
        hint = motion.hint()
        cv2.putText(frame, (hint if res.landmarks is not None else "show your hand"),
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 255) if res.landmarks is not None else (0, 0, 255),
                    1, cv2.LINE_AA)
        said = "" if top1 is None else f"model: {top1} {p1:.2f}   "
        cv2.putText(frame, f"strokes so far: {len(motion.strokes)}   {said}"
                           f"ESC = skip letter   Q = stop", (14, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("J/Z calibration (nothing is saved)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            cal.cancel()
            print("  calibration cancelled - shipped defaults kept")
            return None
        if cal.done:
            break
    cv2.destroyAllWindows()
    if cal.result is None:
        return None
    report_calibration(cal.result)
    return cal.result


def report_calibration(res: dict) -> None:
    """Print a calibration result: strokes, numbers and the pasteable spec."""
    print("\n  your strokes:")
    for i, s in enumerate(res["strokes"], 1):
        print(f"    {i}. {s['intended']}  held {s['held_full']:.2f}s  "
              f"hand travel {s['wrist_net']:.2f}/{s['wrist_path']:.2f}  "
              f"pinky path {s['pinky_path_2d']:.2f}  "
              f"index path {s['tip_path_2d']:.2f}  drop {s['drop']:.2f}")
    print("\n  thresholds now in force (this session only):")
    for k in sorted(MotionDecider.DEFAULTS):
        mark = "*" if abs(res["thresholds"][k] - MotionDecider.DEFAULTS[k]) > 1e-9 else " "
        print(f"    {mark} {k:<12} {res['thresholds'][k]:>6.2f}   "
              f"{res['notes'].get(k, '')}")
    c, b = res["consistency"], res["baseline_consistency"]
    for letter in ("J", "Z"):
        got, n = c["per_letter"][letter]
        bgot, bn = b["per_letter"][letter]
        print(f"    replayed {letter}: {got}/{n} strokes fire"
              f" (shipped defaults: {bgot}/{bn})")
    if c["wrong_letter"]:
        print(f"    NOTE: {c['wrong_letter']} stroke(s) would produce the "
              f"other letter - sign them more distinctly")
    if res["spec"]:
        print(f"\n  to keep these numbers next time without recalibrating:\n"
              f"    python src/live.py --jz {res['spec']}")
    else:
        print("\n  your strokes already suit the shipped defaults; nothing changed")


# Open right hand in image coords, MediaPipe layout; used by the benchmark/selftest.
SYNTH_HAND = np.array(
    [[0.5, 0.90, 0], [.45, .75, 0], [.43, .68, 0], [.42, .63, 0], [.41, .58, 0],
     [.47, .68, 0], [.46, .58, 0], [.455, .52, 0], [.45, .47, 0],
     [.50, .67, 0], [.50, .56, 0], [.50, .50, 0], [.50, .45, 0],
     [.53, .68, 0], [.54, .58, 0], [.545, .52, 0], [.55, .47, 0],
     [.56, .69, 0], [.575, .60, 0], [.585, .54, 0], [.59, .49, 0]])


def synth_jshape_hand(curl_others: bool = True) -> np.ndarray:
    """SYNTH_HAND with the index straight up; other fingers optionally curled."""
    lm = SYNTH_HAND.copy()
    lm[config.FINGERTIPS[1]] = [0.45, 0.41, 0.0]
    if curl_others:
        for mcp, pip, _dip, tip in config.FINGER_CHAINS[2:]:
            lm[tip] = lm[mcp] + 0.55 * (lm[pip] - lm[mcp])
    return lm


def synth_ishape_hand() -> np.ndarray:
    """ASL I starting shape: pinky extended, other fingers curled."""
    lm = SYNTH_HAND.copy()
    for mcp, pip, _dip, tip in config.FINGER_CHAINS[1:4]:
        lm[tip] = lm[mcp] + 0.55 * (lm[pip] - lm[mcp])
    return lm


def render(frame: np.ndarray, held: str | None, conf: float, committed: str | None,
           text: str, fps: float, hand_present: bool, pending_s: float, hold: float,
           backend_name: str, top3: list[tuple[str, float]] | None = None,
           uncertain: bool = False, margin: float = 0.10,
           need: float | None = None) -> np.ndarray:
    """Draw the per-frame state. `margin` is the gap got, `need` the gap required."""
    import cv2

    h, w = frame.shape[:2]
    if uncertain:
        big = "UNCERTAIN"
        color = (0, 200, 255)
    else:
        big = held or ("NO HAND" if not hand_present else "...")
        color = (0, 220, 0) if held else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w, 64), (0, 0, 0), -1)
    cv2.putText(frame, big, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.7, color, 3, cv2.LINE_AA)
    right = f"{conf * 100:4.0f}%" if (held or uncertain) else ""
    cv2.putText(frame, right, (w - 110, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)

    # top-3 probabilities, best first
    if top3:
        bw = 118
        for i, (lab, p) in enumerate(top3):
            y = 96 + i * 26
            cv2.putText(frame, lab, (w - 152, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (w - 128, y), (w - 128 + int(bw * min(p, 1.0)),
                                                y + 16), (60, 200, 90) if i == 0
                          else (90, 90, 90), -1)
            cv2.rectangle(frame, (w - 128, y), (w - 128 + bw, y + 16),
                          (200, 200, 200), 1)
            cv2.putText(frame, f"{p * 100:.0f}%", (w - 60, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        if uncertain:
            need = margin if need is None else need
            txt = (f"gap {margin * 100:.0f} pts" if need <= margin else
                   f"gap {margin * 100:.0f} < needs {need * 100:.0f}")
            cv2.putText(frame, txt, (w - 152, 96 + 3 * 26 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

    # hold-to-commit progress bar
    if held:
        prog = min(1.0, pending_s / hold)
        cv2.rectangle(frame, (14, 72), (14 + int(180 * prog), 84), (0, 220, 0), -1)
        cv2.rectangle(frame, (14, 72), (194, 84), (255, 255, 255), 1)
    cv2.putText(frame, f"{fps:4.1f} FPS  [{backend_name}]", (w - 190, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # text buffer
    cv2.rectangle(frame, (0, h - 96), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "typed:", (10, h - 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)
    shown = text[-90:]
    import textwrap

    for i, line in enumerate(textwrap.wrap(shown, width=42)[-2:] or [""]):
        cv2.putText(frame, line, (10, h - 48 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SPACE=commit  BS=del  C=clear  Q=quit", (10, 20 + h - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    if committed:
        cv2.putText(frame, f"+{committed}", (w // 2, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 255, 0), 3, cv2.LINE_AA)
    return frame


def selftest() -> int:
    print("== live.py selftest (smoothing logic, no camera, nothing saved) ==")
    checks = []
    # Everything up to the head's own block pins the *fallback rule*, whose
    # semantics are not the head's; switch the head off for that stretch so a
    # trained model cannot mask a rule regression (and vice versa).
    _jz_env = os.environ.get(jz_model.ENV)
    os.environ[jz_model.ENV] = "off"

    def _raises(fn) -> bool:
        """True when `fn` raises."""
        try:
            fn()
        except Exception:
            return True
        return False

    def run_seq(s, letters, conf, dt=0.05, t0=0.0, force_last=False, gate=None):
        """Feed a scripted sequence, return (commits, last_held, last_hconf)."""
        t, commits, held, hc = t0, [], None, 0.0
        for i, l in enumerate(letters):
            t += dt
            held, c, hc = s.step(l, conf, t, force=(force_last and i == len(letters) - 1),
                                 hand_present=(l is not None), gate=gate)
            if c:
                commits.append(c)
        return commits, held, hc

    s = Smoother(conf=0.7, vote=7, hold=0.5)
    commits, held, _ = run_seq(s, ["H"] * 12, 0.9)
    checks.append(("H held once 7 frames agree", held == "H"))
    checks.append(("H committed exactly once, never repeated", commits == ["H"]))

    commits2, _, _ = run_seq(s, ["I"] * 12, 0.9, t0=10.0)
    checks.append(("switching to I commits I", commits2 == ["I"]))
    commits3, _, _ = run_seq(s, ["H"] * 12, 0.9, t0=20.0)
    checks.append(("H is allowed again after another letter", commits3 == ["H"]))

    out, held_low, _ = run_seq(Smoother(conf=0.7, vote=7, hold=0.5), ["A"] * 20, 0.4)
    checks.append(("sub-threshold frames never commit or hold", out == [] and held_low is None))

    # The web typer's write rule: no confidence floor, so the top guess votes
    # even on a frame the display calls UNCERTAIN; vote+hold still govern it.
    out, held_open, _ = run_seq(Smoother(conf=0.7, vote=7, hold=0.5),
                                ["A"] * 20, 0.4, gate=0.0)
    checks.append(("gate=0.0 lets the top guess commit after the vote/hold",
                   out == ["A"] and held_open == "A"))

    s3 = Smoother(conf=0.7, vote=7, hold=0.5)
    c1, _, _ = run_seq(s3, ["S"] * 12, 0.9)
    c2, _, _ = run_seq(s3, [None] * 4, 0.9, t0=10.0)
    c3, _, _ = run_seq(s3, ["S"] * 12, 0.9, t0=20.0)
    checks.append(("hand leaving the frame re-arms the same letter",
                   c1 == ["S"] and c2 == [] and c3 == ["S"]))

    cc1 = run_seq(Smoother(conf=0.7, vote=1, min_votes=1, hold=0.0), ["T"], 0.70)[0]
    cc2 = run_seq(Smoother(conf=0.7, vote=1, min_votes=1, hold=0.0), ["T"], 0.69)[0]
    checks.append(("conf exactly at threshold votes; just below does not",
                   cc1 == ["T"] and cc2 == []))

    s5 = Smoother(conf=0.7, vote=7, hold=0.5)
    _, _, hc = run_seq(s5, ["M"] * 7, 0.85)
    checks.append(("held confidence reported for the overlay", abs(hc - 0.85) < 1e-6))

    early = run_seq(Smoother(conf=0.7, vote=7, hold=0.5), ["B"] * 4, 0.9)[0]
    late = run_seq(Smoother(conf=0.7, vote=7, hold=0.5), ["B"] * 11, 0.9)[0]
    checks.append(("hold-to-commit fires only after ~0.5 s", early == [] and late == ["B"]))

    # passing through a letter on the way to another must not type it
    s7 = Smoother(conf=0.7, vote=7, hold=0.5)
    c8, _, _ = run_seq(s7, ["E"] * 6 + ["F"] * 20, 0.9)
    checks.append(("a letter held for <0.5 s is skipped (E->F types only F)", c8 == ["F"]))

    # SPACE force-commits without waiting, but never repeats
    s8 = Smoother(conf=0.7, vote=7, hold=0.5)
    f1, _, _ = run_seq(s8, ["K"] * 3, 0.9, force_last=True)
    f2, _, _ = run_seq(s8, ["K"] * 3, 0.9, t0=9.0, force_last=True)
    checks.append(("SPACE commits immediately, then blocks the repeat",
                   f1 == ["K"] and f2 == []))

    # J / Z trajectory rules
    def run_motion(lms, dt=0.05, top1=None):
        """Feed a scripted trajectory; return (letters, fire times, decider).

        The rule is what these checks pin, so the trained head is switched off:
        synthetic geometry is not a human stroke. The head has its own checks.
        """
        md = MotionDecider(use_model=False)
        out, times, t = [], [], 0.0
        for lm in lms:
            t += dt
            r = md.update(t, lm, top1, 0.80 if top1 else None)
            if r:
                out.append(r)
                times.append(t)
        return out, times, md

    def seq_offset(base, dxs, dys=None):
        """Translate the whole hand: one frame per (dx, dy) in image units."""
        dys = dys or [0.0] * len(dxs)
        return [base + np.array([dx, dy, 0.0]) for dx, dy in zip(dxs, dys)]

    def hook_xs(n, span, hook):
        xs = []
        for i in range(n):
            f = i / (n - 1)
            x = span * min(f / 0.7, 1.0)
            if f > 0.7:
                x -= hook * (f - 0.7) / 0.3
            xs.append(x)
        return xs

    def straight_xs(n, span):
        return [span * i / (n - 1) for i in range(n)]

    def seq_zigzag(base, n=22, scale=1.0):
        """Index tip traces a Z while the hand stays put; `scale` shrinks it."""
        way = [(0.0, 0.0), (0.11, 0.0), (-0.11, 0.05), (0.11, 0.10)]
        way = [(x * scale, y * scale) for x, y in way]
        out = []
        for i in range(n):
            f = i / (n - 1) * (len(way) - 1)
            k = min(int(f), len(way) - 2)
            u = f - k
            dx = way[k][0] + (way[k + 1][0] - way[k][0]) * u
            dy = way[k][1] + (way[k + 1][1] - way[k][1]) * u
            lm = base.copy()
            lm[config.FINGERTIPS[1]] += np.array([dx, dy, 0.0])
            out.append(lm)
        return out

    def seq_pinky_j(base, n=30, scale=1.0):
        """Pinky draws a curved J while the palm stays in place."""
        way = [(0.0, 0.0), (0.0, 0.30), (0.15, 0.40), (0.36, 0.20)]
        way = [(x * scale, y * scale) for x, y in way]
        start = base[config.FINGERTIPS[4]].copy()
        out = []
        for i in range(n):
            f = i / (n - 1) * (len(way) - 1)
            k = min(int(f), len(way) - 2)
            u = f - k
            dx = way[k][0] + (way[k + 1][0] - way[k][0]) * u
            dy = way[k][1] + (way[k + 1][1] - way[k][1]) * u
            lm = base.copy()
            lm[config.FINGERTIPS[4]] = start + np.array([dx, dy, 0.0])
            out.append(lm)
        return out

    jshape = synth_jshape_hand()
    ishape = synth_ishape_hand()
    checks.append(("the J/Z handshape is detected as index-up, others curled",
                   MotionDecider.handshape(normalize.canonicalize(jshape))[0]))
    open_hand = synth_jshape_hand(curl_others=False)
    checks.append(("an open hand is NOT the J/Z handshape",
                   not MotionDecider.handshape(normalize.canonicalize(open_hand))[0]))

    m1, _t1, _ = run_motion(seq_offset(jshape, [0.0] * 30))
    checks.append(("a still J-shape fires nothing", m1 == []))

    m2, _t2, _ = run_motion(seq_offset(jshape, straight_xs(30, 0.22)))
    checks.append(("a straight hand transit is not a J", m2 == []))

    m3, _t3, md3 = run_motion(seq_pinky_j(ishape), top1="I")
    checks.append(("the pinky-drawn J is recognized while the palm stays still",
                   m3 == ["J"]))

    m4, _t4, md4 = run_motion(seq_zigzag(jshape))
    checks.append((f"the fingertip zig-zag commits Z (at fire: {md4.last_debug})",
                   m4 == ["Z"]))

    m5, _t5, _ = run_motion(seq_offset(open_hand, hook_xs(30, 0.22, 0.075)))
    checks.append(("the same hook with an open hand fires nothing", m5 == []))

    rng_j = np.random.default_rng(3)
    jitter = [ishape + rng_j.normal(0, 0.006, (21, 3)) for _ in range(70)]
    m6, _t6, _ = run_motion(jitter, top1="I")
    checks.append(("landmark jitter alone does not fire J or Z", m6 == []))

    # repeated gestures must be spaced by the cooldown, never double-typed
    triple = seq_pinky_j(ishape) * 3
    m7, t7, _ = run_motion(triple, top1="I")
    gaps = [b - a for a, b in zip(t7, t7[1:])]
    checks.append(("repeated J strokes respect the motion cooldown",
                   all(b - a >= MotionDecider.COOLDOWN for a, b in zip(t7, t7[1:]))))

    m8, _t8, _ = run_motion(seq_pinky_j(ishape) + [ishape] * 40
                            + seq_pinky_j(ishape), top1="I")
    checks.append(("a second pinky J can be recognized after the cooldown",
                   len(m8) == 2))

    # The trained stroke head: it must load, agree with the feature builder, and
    # reproduce its own training windows - that last one is the parity check
    # between what train_jz.py computed and what the camera path computes.
    # (the head is switched off for the rule checks above; build it directly here
    # so the two cannot interfere)
    if not jz_model.weights_path().is_file():
        checks.append(("J/Z head: no weights, the rule is in charge (not a failure)",
                       True))
    else:
        import glob as _glob
        head = jz_model.JZModel()
        checks.append(("the J/Z head loads and matches jz_features' dimension",
                       head is not None
                       and int(head.meta["feature_dim"]) == jz_features.feature_dim()))
        checks.append(("the J/Z head carries a threshold per letter",
                       head is not None and {"j", "z"} <= set(head.thresholds)
                       and "other" in head.labels))
        shards = sorted(_glob.glob(str(config.CACHE_DIR / "msl_landmarks" / "*.npz")))
        want = {"Z": "Z", "K": None}
        seen = {}
        for path in shards:
            z = np.load(path, allow_pickle=True)
            letter = str(z["letter"])
            if letter in want and letter not in seen:
                lm = np.asarray(z["lm"], dtype=np.float64)
                fps = float(z["fps"]) if float(z["fps"]) > 1 else 30.0
                idx = np.asarray(z["idx"], dtype=np.float64) if "idx" in z else None
                t = (idx if idx is not None and len(idx) == len(lm)
                     else np.arange(len(lm), dtype=np.float64)) / fps
                keep = t >= max(float(t[0]), float(t[-1]) - jz_features.WINDOW_S)
                feats = jz_features.features_from_frames(lm[keep], t[keep])
                seen[letter] = head.decide(feats, ("J", "Z"))[0]
            if len(seen) == len(want):
                break
        checks.append((f"a real Z stroke is read as Z by the shipped head "
                       f"(got {seen.get('Z')!r})", seen.get("Z") == "Z"))
        checks.append((f"a real K stroke is not read as J or Z "
                       f"(got {seen.get('K')!r})", seen.get("K") is None))

    # Landmark smoothing: a glitch is clamped, a real move still tracks.
    smf = smoothing.LandmarkFilter()
    base = synth_jshape_hand()
    smf(base, 0.0)
    jumped = base + np.array([0.45, 0.0, 0.0])      # half a palm in one frame
    out = smf(jumped, 0.05)
    moved = float(np.linalg.norm(out[0, :2] - base[0, :2]))
    checks.append((f"a one-frame glitch moves the wrist {moved:.2f} palms, not 0.45",
                   moved < 0.44))
    smf.reset()
    smf(base, 0.0)
    track_err = []
    for i in range(1, 25):                      # a steady 0.02 palm/frame slide
        raw = base + np.array([0.02 * i, 0.0, 0.0])
        out = smf(raw, 0.033 * i)
        track_err.append(float(np.linalg.norm(out[0, :2] - raw[0, :2])))
    checks.append((f"a steady move is tracked to within {max(track_err):.3f} palms",
                   max(track_err) < 0.10))

    # top-3 / uncertain gate
    L = config.LABELS
    a, b, c = L[0], L[1], L[2]

    def probs_of(pairs):
        p = np.zeros(len(L), dtype=np.float64)
        for lab, v in pairs:
            p[L.index(lab)] = v
        return p

    pc1 = probs_of([(a, 0.85), (b, 0.06), (c, 0.05)])
    r1 = frame_call(pc1, L, 0.7, 0.10)
    checks.append(("a clear winner votes", r1["letter"] == a and not r1["uncertain"]))
    checks.append(("top-3 is sorted and complete",
                   [x[0] for x in r1["top3"]] == [a, b, c]
                   and abs(sum(x[1] for x in r1["top3"]) - 0.96) < 1e-9))

    pc2 = probs_of([(a, 0.45), (b, 0.40), (c, 0.05)])
    r2 = frame_call(pc2, L, 0.7, 0.10)
    checks.append(("a 5-point tie is UNCERTAIN and casts no vote",
                   r2["letter"] is None and r2["uncertain"]))
    checks.append(("a tie is reported even though it is also below the gate",
                   r2["low_conf"] and r2["close"]))

    pc3 = probs_of([(a, 0.55), (b, 0.05), (c, 0.05)])
    r3 = frame_call(pc3, L, 0.7, 0.10)
    checks.append(("below the confidence gate alone is not called uncertain",
                   r3["letter"] is None and not r3["uncertain"] and r3["low_conf"]))

    r4 = frame_call(pc2, L, 0.7, 0.04)
    checks.append(("a smaller --margin stops calling the same frame a tie",
                   not r4["uncertain"] and r4["letter"] is None))
    r4b = frame_call(pc2, L, 0.40, 0.04)
    checks.append(("with both gates satisfied the same frame does vote",
                   r4b["letter"] == a))

    pc5 = probs_of([(a, 0.72), (b, 0.20), (c, 0.05)])
    r5 = frame_call(pc5, L, 0.7, 0.10)
    checks.append(("confident enough AND far enough apart -> votes",
                   r5["letter"] == a and not r5["uncertain"]))

    pc6 = probs_of([(a, 0.78), (b, 0.70), (c, 0.02)])
    r6 = frame_call(pc6, L, 0.7, 0.10)
    checks.append(("a high-confidence 8-point tie still casts no vote",
                   r6["letter"] is None and r6["uncertain"] and not r6["low_conf"]))

    # pair-specific refusal margins: A/S is measured, A/B is not
    pcA = probs_of([("A", 0.72), ("S", 0.25), ("B", 0.03)])
    prA = frame_call(pcA, L, 0.7, 0.10)
    checks.append(("a measured pair's 47-pt gap is refused even above --margin",
                   prA["letter"] is None and prA["uncertain"] and not prA["low_conf"]))
    checks.append(("the refusal names the pair and the margin it needed",
                   prA["collide"] and prA["pair"] == "AS"
                   and abs(prA["margin"] - 0.47) < 1e-9 and abs(prA["need"] - 0.50) < 1e-9))
    checks.append(("the same frame votes when the pair table is off",
                   frame_call(pcA, L, 0.7, 0.10, pairs={})["letter"] == "A"))
    pcB = probs_of([("A", 0.72), ("B", 0.25), ("S", 0.03)])
    checks.append(("a pair with no measured rule keeps using the flat --margin",
                   frame_call(pcB, L, 0.7, 0.10)["letter"] == "A"))
    pcC = probs_of([("S", 0.72), ("A", 0.25), ("B", 0.03)])
    checks.append(("the rule is about the pair, not the direction",
                   frame_call(pcC, L, 0.7, 0.10)["collide"]))
    pcD = probs_of([("A", 0.88), ("S", 0.09), ("B", 0.03)])
    checks.append(("a measured pair still votes when the gap clears its margin",
                   frame_call(pcD, L, 0.7, 0.10)["letter"] == "A"))
    checks.append(("--margin above a pair entry wins: max(), never min()",
                   frame_call(pcA, L, 0.7, 0.80)["need"] == 0.80))
    checks.append(("every shipped pair key is two real labels with a sane margin",
                   all(len(k) == 2 and set(k) <= set(config.LABELS)
                       and 0.0 < v <= 1.0 for k, v in PAIR_MARGINS.items())))
    checks.append(("a pair spec round-trips through format/resolve",
                   resolve_pair_margins(format_pair_margins(PAIR_MARGINS))
                   == dict(PAIR_MARGINS)))
    checks.append(("a spec key is order-insensitive, like pair_key",
                   resolve_pair_margins("QP=0.4") == {"PQ": 0.4}))
    checks.append(("--pair-margins none means the flat margin for every pair",
                   resolve_pair_margins("none") == {}))
    checks.append(("leaving --pair-margins out keeps the measured table",
                   resolve_pair_margins(None) == dict(PAIR_MARGINS)))
    for bad_spec, why in [("PZ=0.4", "Z is a motion letter, not a class"),
                          ("PQJ=0.4", "three letters"),
                          ("PQ=1.4", "a gap over 1.0"),
                          ("PQ=wide", "not a number"),
                          ("PQ", "no = at all")]:
        try:
            resolve_pair_margins(bad_spec)
            ok = False
        except ValueError:
            ok = True
        checks.append((f"--pair-margins rejects {bad_spec!r} ({why})", ok))

    # no-hand frame contract: blank_call must carry exactly frame_call's keys
    b, r_real = blank_call(), frame_call(pc5, L, 0.7, 0.10)
    checks.append(("blank_call has exactly frame_call's keys - never a subset",
                   set(b) == set(r_real)))
    checks.append(("a frame with no hand carries no opinion in any of them",
                   b["letter"] is None and b["conf"] == 0.0 and b["margin"] == 0.0
                   and b["need"] == 0.0 and b["top3"] == [] and b["pair"] == ""
                   and not any(b[k] for k in ("low_conf", "close", "collide",
                                              "uncertain"))))
    try:
        _f = np.zeros((320, 240, 3), np.uint8)
        _o = render(_f, None, 0.0, None, "", 0.0, False, 0.0, 0.8, "test",
                    top3=b["top3"], uncertain=b["uncertain"],
                    margin=b["margin"], need=b["need"])
        drew = _o.shape == _f.shape
    except Exception:                                       # noqa: BLE001
        drew = False
    checks.append(("render() survives the no-hand frame the way main() calls it",
                   drew))
    try:
        _f2 = np.zeros((320, 240, 3), np.uint8)
        drew2 = render(_f2, "P", 0.9, "P", "P", 30.0, True, 0.4, 0.8, "test",
                       top3=r_real["top3"], uncertain=r_real["uncertain"],
                       margin=r_real["margin"], need=r_real["need"]).shape == _f2.shape
    except Exception:                                       # noqa: BLE001
        drew2 = False
    checks.append(("render() still survives a confident hand frame", drew2))

    # one clock per detector: a second loop must not rewind the timestamps
    class StrictDetector:
        """Non-increasing stamps raise, like the real VIDEO API."""

        def __init__(self):
            self.seen = []

        def detect_for_video(self, image, ts):
            if self.seen and ts <= self.seen[-1]:
                raise ValueError(
                    "Input timestamp must be monotonically increasing.")
            self.seen.append(ts)
            return type("R", (), {"hand_landmarks": [], "handedness": []})()

    tiny = np.zeros((24, 32, 3), np.uint8)
    sd = StrictDetector()
    try:
        for _ in range(2):           # two loops, each starting its own clock at 0
            for i in range(30):
                hb.detect_video(sd, tiny, int(i * 33))
        rewound = False
    except ValueError:
        rewound = True
    checks.append(("a second loop cannot rewind a live detector's clock",
                   not rewound and len(sd.seen) == 60
                   and all(y > x for x, y in zip(sd.seen, sd.seen[1:]))))
    checks.append(("an honest increasing run is passed through untouched",
                   sd.seen[:30] == [int(i * 33) for i in range(30)]))
    checks.append(("the clamp counts the calls it had to rescue",
                   getattr(sd, "_hb_ts_rewound", 0) == 30))
    # AST, not text, so this check's own source cannot satisfy it.
    import ast

    _tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def _owns_clock(fn) -> bool:
        for n in ast.walk(fn):
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
                    and isinstance(n.left, ast.Call)
                    and ast.unparse(n.left.func).endswith("monotonic")):
                return True
        return False

    owners = sorted({f.name for f in ast.walk(_tree)
                     if isinstance(f, ast.FunctionDef) and _owns_clock(f)})
    callers = sum(1 for n in ast.walk(_tree)
                  if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "stamp_ms")
    checks.append(("only stamp_ms() turns the clock into a timestamp - no loop "
                   f"keeps its own epoch (owners: {owners})",
                   owners == ["stamp_ms"]))
    checks.append((f"every camera loop takes its timestamp from stamp_ms() "
                   f"({callers} call sites)", callers >= 3))

    # chirality labels: a flip_0.3 hit is mapped back (x -> 1-x), so its label must swap too
    checks.append(("the mirror label swap matches the extractor's rule and its "
                   "own inverse",
                   all(hb.swap_label(v) == {"Left": "Right", "Right": "Left"}[v]
                       for v in ("Left", "Right"))
                   and hb.swap_label(None) is None
                   and all(hb.swap_label(hb.swap_label(v)) == v
                           for v in ("Left", "Right", None, ""))))
    hand = normalize.as_array(np.concatenate(
        [np.array([[0.50, 0.90, 0.0]]),                      # wrist, low centre
         np.linspace([0.35, 0.15, 0.0], [0.65, 0.15, 0.0], 20).reshape(20, 3)]))
    flipped = hand.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    fam, fam_f = normalize.chirality_family(hand), normalize.chirality_family(flipped)
    checks.append((f"chirality family flips with x -> 1-x ({fam} / {fam_f}) and "
                   "canonicalize() lands on the canonical one",
                   fam != fam_f and fam in ("canonical", "mirrored")
                   and normalize.chirality_family(
                       normalize.canonicalize(hand)) == "canonical"
                   and normalize.chirality_family(
                       normalize.canonicalize(flipped)) == "canonical"))

    # calibration: smaller strokes miss at the defaults and must be fixed by it
    def record(lms, letter, reps=3, dt=0.05):
        """Feed a trajectory `reps` times with the calibrator's label set."""
        md = MotionDecider()
        md.intended = letter
        t = 0.0
        for _ in range(reps):
            for lm in lms:
                t += dt
                md.update(t, lm, "I" if letter == "J" else "X", 0.80)
            md.reset()
        md.intended = None
        return md.strokes

    def refeed(lms, th, reps=3, dt=0.05, top1=None):
        """Re-run a trajectory live through a decider carrying `th`."""
        md = MotionDecider()
        md.apply_thresholds(th)
        out, t = [], 0.0
        for _ in range(reps):
            for lm in lms:
                t += dt
                r = md.update(t, lm, top1, 0.80 if top1 else None)
                if r:
                    out.append(r)
            md.reset()
        return out

    small_j = seq_pinky_j(ishape, n=24, scale=0.35)
    small_z = seq_zigzag(jshape, n=22, scale=0.5)
    sj, sz = record(small_j, "J"), record(small_z, "Z")
    both = sj + sz
    D = MotionDecider.DEFAULTS
    checks.append((f"the small strokes really do miss at the shipped defaults "
                   f"(replay {self_consistency(both, D)['rate'] * 100:.0f}%)",
                   self_consistency(both, D)["rate"] == 0.0
                   and refeed(small_j, D, top1="I") == []
                   and refeed(small_z, D, top1="X") == []))
    th, notes = derive_thresholds(both)
    checks.append((f"derive_thresholds makes every recorded stroke fire in "
                   f"replay (was {self_consistency(both, D)['fired']}/{len(both)}, "
                   f"now {self_consistency(both, th)['fired']}/{len(both)})",
                   self_consistency(both, th)["rate"] == 1.0
                   and self_consistency(both, th)["wrong_letter"] == 0))
    checks.append((f"calibration supports both strokes live "
                   f"(J {refeed(small_j, th, top1='I')}, "
                   f"Z {refeed(small_z, th, top1='X')})",
                   len(refeed(small_j, th, top1="I")) >= 2
                   and len(refeed(small_z, th, top1="X")) >= 2))
    below = ("MIN_HOLD", "SEG", "J_NET", "J_MOTION", "Z_PATH", "Z_DROP")
    above = ("Z_WRIST_MAX", "WINDOW")
    checks.append(("calibration only loosens: never tighter than shipped, never "
                   "past its own floor",
                   all(MotionDecider.FLOORS[k] <= th[k] <= D[k] for k in below)
                   and all(D[k] <= th[k] <= MotionDecider.FLOORS[k] for k in above)
                   and th["COOLDOWN"] == D["COOLDOWN"]))
    checks.append(("deriving from no strokes returns the shipped defaults",
                   derive_thresholds([])[0] == dict(D)))
    spec = format_spec(th)
    checks.append((f"--jz spec round-trips and rejects junk ({spec})",
                   parse_spec(spec) == {k: v for k, v in th.items()
                                        if abs(v - D[k]) > 1e-9}
                   and format_spec(D) == ""
                   and parse_spec(" j_net = 0.5 ") == {"J_NET": 0.5}
                   and all(_raises(lambda: parse_spec(bad)) for bad in
                           ("SEG=nope", "NOPE=1", "SEG"))))
    md_none = MotionDecider()
    refeed(small_j, D, reps=3, top1="I")
    checks.append(("an ordinary session records nothing (no unbounded stroke list)",
                   len(md_none.strokes) == 0))
    checks.append(("strokes are only archived while a letter is intended",
                   len(sj) == 3 and all(s["intended"] == "J" for s in sj)))

    # the hint: a held shape with no trajectory must say what is missing
    md_h = MotionDecider()
    t = 0.0
    for lm in small_j[:16]:
        t += 0.05
        md_h.update(t, lm, "I", 0.80)
    h = md_h.hint()
    checks.append((f"a held shape that does not fire explains itself ({h!r})",
                   "not fired" in h and "needs" in h))
    md_h2 = MotionDecider()
    md_h2.update(0.05, small_j[0], "I", 0.80)
    checks.append(("before MIN_HOLD the message is to keep holding",
                   "reads this frame as I" in md_h2.hint()))
    fired, _tt, md_ok = run_motion(seq_zigzag(jshape))
    checks.append(("and while the Z rule is genuinely met nothing is complained about",
                   fired == ["Z"] and md_ok.hint() == ""))
    # the explanation may never contradict the rule it explains
    md_x, agree = MotionDecider(), True
    tt = 0.0
    for lm in seq_offset(jshape, hook_xs(30, 0.22, 0.075))[:22]:
        tt += 0.05
        md_x.update(tt, lm)
        met = any(all(c[3] for c in v) for v in md_x.criteria().values())
        if md_x.debug.get("held", 0) >= md_x.MIN_HOLD and met == bool(md_x.hint()):
            agree = False
    checks.append(("hint() is empty exactly when a rule is satisfied", agree))

    # the guided session itself, with no camera
    md_c = MotionDecider()
    cal = JZCalibrator(md_c, reps=2)
    open_hand_c = synth_jshape_hand(curl_others=False)
    ishape_c = synth_ishape_hand()
    tt, guard = 0.0, 0
    while not cal.done and guard < 4000:
        for lm in (small_j if cal.letter == "J" else small_z):
            tt += 0.05
            guard += 1
            # the shipped model reads J as I and Z as X
            cal.step(tt, lm, "I" if cal.letter == "J" else "X", 0.55)
        for _ in range(10):          # the hand must actually be let go
            tt += 0.05
            guard += 1
            cal.step(tt, open_hand_c, None, None)
    r = cal.result or {}
    checks.append((f"the guided calibrator finishes 2+2 strokes and reports a "
                   f"spec ({r.get('spec')})",
                   cal.done and r.get("n_strokes") == 4
                   and r["consistency"]["rate"] == 1.0
                   and r["consistency"]["wrong_letter"] == 0
                   and r["baseline_consistency"]["rate"] == 0.0))
    checks.append(("finishing applies the numbers to the live decider and stops "
                   "recording",
                   md_c.thresholds() == r["thresholds"] and md_c.intended is None
                   and "calibrated" in md_c.threshold_source))
    checks.append(("nothing about a calibration is written to disk",
                   not any(isinstance(v, (bytes,)) for v in r["thresholds"].values())
                   and all(isinstance(v, float) for v in r["thresholds"].values())))

    # display precondition: the second way into the rule, and its limits
    def feed(lms, top1=None, p1=None, dt=0.05):
        md = MotionDecider()
        out, t = [], 0.0
        for lm in lms:
            t += dt
            r = md.update(t, lm, top1, p1)
            if r:
                out.append(r)
        return out, md

    hook_open = seq_pinky_j(ishape_c)
    # An open hand fails the shape gate, so only the display gate can open it.
    hook_noshape = seq_pinky_j(open_hand_c)
    ramp_open = []
    pinky_start = ishape_c[config.FINGERTIPS[4]].copy()
    for dx in np.linspace(0.0, 0.36, 30):
        lm = ishape_c.copy()
        lm[config.FINGERTIPS[4]] = pinky_start + np.array([dx, 0.0, 0.0])
        ramp_open.append(lm)
    zig_open = seq_zigzag(open_hand_c, n=22, scale=1.0)
    checks.append(("the open hand really is rejected by the shape gate, so every "
                   "check below runs on the display path alone",
                   not MotionDecider.handshape(
                       normalize.canonicalize(open_hand_c))[0]))
    checks.append(("an I reading opens the J motion check at any confidence",
                   MotionDecider.display_candidate("I", 0.60) == "J"
                   and MotionDecider.display_candidate("I", 0.99) == "J"))
    checks.append((f"an unsure X reading turns the zig-zag into a Z "
                   f"(no reading: {feed(zig_open)[0]}, with one: "
                   f"{feed(zig_open, 'X', 0.55)[0]})",
                   feed(zig_open)[0] == []
                   and feed(zig_open, "X", 0.55)[0] == ["Z"]))
    checks.append(("a confident X still opens the Z stroke head",
                   feed(zig_open, "X", 0.95)[0] == ["Z"]
                   and MotionDecider.display_candidate("X", 0.99) == "Z"))
    held_i = seq_offset(ishape_c, [0.0] * 40)
    checks.append(("a still I never fires J, even though I opens the motion check",
                   feed(held_i, "I", 0.99)[0] == []))
    checks.append(("J and Z have automatic motion routes and calibration cannot alter them",
                   MotionDecider.AUTO_MOTION_LETTERS == ("J", "Z")
                   and MotionDecider.DISPLAY_P1_MAX_FOR["J"] is None
                   and MotionDecider.DISPLAY_P1_MAX_FOR["Z"] is None
                   and not ({"DISPLAY_FOR", "DISPLAY_P1_MAX", "DISPLAY_P1_MAX_FOR",
                             "DISPLAY_MIN_SHARE"} & set(MotionDecider.DEFAULTS))))
    checks.append(("a reading that names neither I nor X is no candidate",
                   feed(hook_noshape, "S", 0.55)[0] == []
                   and feed(hook_noshape, None, None)[0] == []))
    # A mixed I/X window cannot satisfy either display gate's majority.
    alt, blocked = MotionDecider(), 0
    tt_alt = 0.0
    for i, lm in enumerate(hook_noshape):
        tt_alt += 0.05
        alt.update(tt_alt, lm, "I" if i % 2 == 0 else "X", 0.5)
        blocked += 1 if alt.debug.get("blocked") else 0
    checks.append(("a mixed I/X reading cannot create a motion letter",
                   alt.last is None))
    def feed_by(lms, readings, dt=0.05):
        """Like `feed`, with a per-frame reading (a real model flickers)."""
        md = MotionDecider()
        out, t = [], 0.0
        for lm, (top1, p1) in zip(lms, readings):
            t += dt
            r = md.update(t, lm, top1, p1)
            if r:
                out.append(r)
        return out

    # A majority I reading is allowed to identify a real J hook.
    three_of_four = [("I", .5) if i % 4 else ("X", .5) for i in range(len(hook_open))]
    checks.append(("an I-majority window can identify a moving J hook",
                   feed(hook_open, "I", 0.5)[0] == ["J"]
                   and feed_by(hook_open, three_of_four) == ["J"]))
    checks.append(("a straight transit remains silent",
                   feed(ramp_open, "I", 0.5)[0] == []))
    md_dz = MotionDecider()
    md_dz.intended = "Z"
    tt_dz = 0.0
    for lm in zig_open:
        tt_dz += 0.05
        md_dz.update(tt_dz, lm, "X", 0.5)
    n_with = len(md_dz.strokes)
    md_dn = MotionDecider()
    md_dn.intended = "Z"
    tt_dn = 0.0
    for lm in zig_open:
        tt_dn += 0.05
        md_dn.update(tt_dn, lm)
    checks.append((f"the calibrator can now see a Z at all ({n_with} stroke "
                   f"recorded with a reading, {len(md_dn.strokes)} without)",
                   n_with == 1 and len(md_dn.strokes) == 0))
    checks.append((f"every recorded stroke reports how much of it the model "
                   f"supported ({md_dz.strokes[0]['disp_share']:.2f})",
                   0.0 <= md_dz.strokes[0]["disp_share"] <= 1.0))
    def call_sites(path):
        """(total, with_reading) for motion.update(...) / cal.step(...) calls."""
        src = Path(path).read_text(encoding="utf-8")
        tot = ok_ = 0
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if "motion.update" not in seg and "cal.step" not in seg:
                continue
            tot += 1
            ok_ += 1 if len(node.args) >= 4 else 0
        return tot, ok_

    n_live = call_sites(Path(__file__).resolve())
    n_app = call_sites(config.ROOT / "app.py")
    checks.append((f"both front ends hand the decider the frame's own reading "
                   f"(live.py {n_live[1]}/{n_live[0]}, app.py {n_app[1]}/{n_app[0]})",
                   n_live[0] >= 2 and n_live[1] == n_live[0]
                   and n_app[0] >= 2 and n_app[1] == n_app[0]))

    ok = all(p for _, p in checks)
    for label, passed in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {label}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _fit_size(img, wh):
    """Scale `img` up/down so it fits inside wh, keeping its aspect ratio."""
    import cv2

    h0, w0 = img.shape[:2]
    s = min(wh[0] / w0, wh[1] / h0)
    return cv2.resize(img, (max(16, int(round(w0 * s))), max(16, int(round(h0 * s)))))


def _real_frames(n_images: int, wh) -> list:
    """Frames from public dataset images on disk, so the benchmark is reproducible.

    A drawn stick figure finds no hand and measures the cheap no-detection path.
    """
    import cv2

    if not config.LANDMARKS_PARQUET.exists():
        return []
    import landmark_store as LS

    df = LS.load_raw()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(len(df), n_images * 4), replace=False)
    frames = []
    for p in df["image"].astype(str).to_numpy()[idx]:
        img = cv2.imread(p)
        if img is None or img.ndim != 3:
            continue
        frames.append(_fit_size(img, wh))
        if len(frames) >= n_images:
            break
    return frames


def _drawn_frame(rng, wh) -> np.ndarray:
    """Fallback frame when no dataset images are on disk: a drawn hand."""
    import cv2

    frame = rng.integers(30, 90, (wh[1], wh[0], 3), dtype=np.uint8)
    pts = SYNTH_HAND + rng.normal(0, 0.004, SYNTH_HAND.shape)
    for a, b in hb.HAND_CONNECTIONS:
        p1 = (int(pts[a, 0] * wh[0]), int(pts[a, 1] * wh[1]))
        p2 = (int(pts[b, 0] * wh[0]), int(pts[b, 1] * wh[1]))
        cv2.line(frame, p1, p2, (170, 190, 210), 9)
    return frame


def benchmark(n_frames: int, size=(640, 480)) -> float:
    """Time MediaPipe + normalise + classifier on in-memory frames (no camera)."""
    det = hb.create_detector(static=False)
    backend, labels = build_backend("torch")
    rng = np.random.default_rng(0)
    frames = _real_frames(min(60, n_frames), size)
    if frames:
        hs = sorted({f.shape[0] for f in frames})
        ws = sorted({f.shape[1] for f in frames})
        print(f"== FPS benchmark: {n_frames} frames from {len(frames)} distinct real "
              f"public-dataset images, {ws[0]}-{ws[-1]}x{hs[0]}-{hs[-1]} px (no camera) ==")
    else:
        print(f"== FPS benchmark: {n_frames} drawn synthetic frames at {size} "
              f"(no camera, no dataset) ==")
        print("  WARNING: with no dataset images MediaPipe finds no hand, so this "
              "only times the no-detection path and is far too optimistic. "
              "Run src/download_data.py + src/extract_landmarks.py first.")

    t_total = t_det = t_cls = 0.0
    n_ok = 0
    for i in range(n_frames):
        frame = frames[i % len(frames)].copy() if frames else _drawn_frame(rng, size)
        t0 = time.perf_counter()
        res = hb.detect_video(det, frame, int(i * 33))
        t1 = time.perf_counter()
        if res.landmarks is not None:
            feats = normalize.features(res.landmarks)
            backend(feats)
            n_ok += 1
        t_total += time.perf_counter() - t0
        t_det += t1 - t0
        t_cls += time.perf_counter() - t1

    fps = n_frames / max(t_total, 1e-9)
    print(f"  pipeline fps        = {fps:.1f}  ({1000 * t_total / n_frames:.1f} ms/frame)")
    print(f"    MediaPipe hands   = {1000 * t_det / n_frames:.1f} ms/frame")
    print(f"    normalise + MLP   = {1000 * t_cls / n_frames:.1f} ms/frame")
    print(f"  hands found         = {n_ok}/{n_frames} "
          f"({100 * n_ok / max(n_frames, 1):.0f}%)")
    if n_ok < 0.5 * n_frames:
        print("  most frames had no hand - the FPS above is NOT representative")
    print(f"  target >= 20 FPS    : {'PASS' if fps >= 20 else 'BELOW TARGET'}")
    print("  NOTE: a real webcam adds capture + overlay cost; the live app prints "
          "the end-to-end number when you quit with Q.")
    det.close()
    return fps


def _profiles(spec: str | None) -> list[str]:
    """`--profile all` -> ['all']; validates each name (only `all` ships now)."""
    names = [x.strip() for x in (spec or "").split(",") if x.strip()]
    if not names:
        return [config.PROFILE_DEFAULT]
    for n in names:
        config.profile(n)          # raises SystemExit listing the valid names
    return names


def drill_overlay(frame, target: str, mode: str, secs_left: float, st: dict,
                  prof: str, idx: int, total: int, labels: list[str]) -> np.ndarray:
    """The one-screen display for the drill: what to sign, and how it is going."""
    import cv2

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 118), (0, 0, 0), -1)
    if mode == "count":
        cv2.putText(frame, f"SHOW  {target}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{secs_left:.1f}", (w - 120, 96), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, "get your hand ready - nothing is counted yet",
                    (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1,
                    cv2.LINE_AA)
    else:
        hit = st["top1"].get(target, 0)
        rate = 100 * hit / max(st["hand"], 1)
        cv2.putText(frame, f"HOLD  {target}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 220, 0), 3, cv2.LINE_AA)
        prog = 1.0 - max(secs_left, 0.0) / max(st["budget"], 1e-6)
        cv2.rectangle(frame, (16, 66), (16 + int((w - 32) * min(max(prog, 0), 1)), 78),
                      (0, 220, 0), -1)
        cv2.rectangle(frame, (16, 66), (w - 16, 78), (255, 255, 255), 1)
        cv2.putText(frame, f"{target} {rate:.0f}% of {st['hand']} frames   "
                           f"no hand {st['frames'] - st['hand']}   "
                           f"gate {st['below']}   uncertain {st['uncertain']}",
                    (16, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)
    top = ", ".join(f"{l} {n}" for l, n in st["top1"].most_common(3)) or "-"
    cv2.rectangle(frame, (0, h - 78), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"profile {prof}  letter {idx}/{total}  "
                       f"Q=stop+report  S=skip  SPACE=redo this letter",
                (10, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"top-1 so far: {top}", (10, h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def drill_phase(cap, det, recovery, backend, labels, args, target: str, prof: str,
                idx: int, total: int, mode: str, secs: float,
                streak: list, st: dict) -> str:
    """One countdown or capture phase; returns quit/skip/redo/done.

    'hold' counts the raw argmax, not the smoothed letter: gate, vote and hold
    each answer a different question and are reported as separate columns.
    """
    import cv2
    from collections import Counter

    if mode == "hold":
        st.update(frames=0, hand=0, below=0, uncertain=0, top1=Counter(),
                  retry=0, budget=secs)
    t_end = time.time() + secs
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            print("  camera stream ended")
            return "quit"
        ts_ms = stamp_ms()
        res, how = hb.detect_frame(det, frame, ts_ms, recovery,
                                   empty_streak=streak[0] + 1)
        streak[0] = streak[0] + 1 if res.landmarks is None else 0
        call = blank_call()
        probs = None
        if res.landmarks is not None:
            feats = normalize.features(res.landmarks)
            probs = np.asarray(backend(feats))
            call = frame_call(probs, labels, args.conf, args.margin, pairs=args.pairs)
        if mode == "hold":
            st["frames"] += 1
            if how in ("direct_0.3", "flip_0.3"):
                st["retry"] += 1
            if res.landmarks is not None:
                st["hand"] += 1
                # raw argmax: the drill measures the opinion, not the gate's verdict
                st["top1"][labels[int(probs.argmax())]] += 1
                st["below"] += int(bool(call.get("low_conf")))
                st["uncertain"] += int(call["uncertain"])
        if not args.no_landmarks and res.landmarks is not None:
            hb.draw_landmarks(frame, res.landmarks[:, :2])
        drill_overlay(frame, target, mode, t_end - time.time(), st, prof, idx,
                      total, labels)
        cv2.imshow("ASL live self-test (nothing is saved)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return "quit"
        if k == ord("s"):
            return "skip"
        if k == ord(" "):
            return "redo"
    return "done"


def drill(args) -> int:
    """Guided A-Z drill against the webcam; prints a table, saves nothing."""
    import cv2
    from collections import Counter

    profs = _profiles(args.profile)
    engines = []
    for p in profs:
        backend, labels = build_backend(args.backend, profile=p)
        engines.append((p, backend, labels))
    labels = engines[0][2]
    if args.letters.strip():
        letters = [x.strip().upper() for x in args.letters.split(",") if x.strip()]
        bad = [l for l in letters if l not in labels]
        if bad:
            raise SystemExit(f"--letters has names the model has no class for: "
                             f"{', '.join(bad)} (J and Z are motion letters)")
    else:
        letters = list(labels)

    print("== live self-test (guided drill) ==")
    print("  privacy: frames stay in memory; nothing is written to disk - not the "
          "table, which is printed only.")
    print(f"  profiles: {', '.join(profs)}   letters: {''.join(letters)}")
    print(f"  {args.drill_countdown}s to get ready, then hold each letter for "
          f"{args.drill_hold}s. Q stops and reports, S skips, SPACE redoes.")

    det = hb.create_detector(static=False)
    recovery = hb.Recovery(conf=args.recover_conf, after=args.recover,
                           every=args.recover_every,
                           flip=not args.no_recover_flip) if args.recover else None
    cap, _be = open_camera(args.camera, args.width, args.height)
    streak = [0]
    results: dict[str, dict[str, dict]] = {p: {} for p in profs}
    stopped = False
    try:
        for p, backend, labs in engines:
            print(f"\n  -- profile {p} --")
            for i, target in enumerate(letters, 1):
                st = {"frames": 0, "hand": 0, "below": 0, "uncertain": 0,
                      "top1": Counter(), "retry": 0, "budget": args.drill_hold}
                note = (f"profile {p}: {i}/{len(letters)}" if len(profs) > 1 else
                        f"{i}/{len(letters)}")
                print(f"  [{p}] {note}  show {target}", end="", flush=True)
                while True:
                    r = drill_phase(cap, det, recovery, backend, labs, args, target,
                                    p, i, len(letters), "count",
                                    args.drill_countdown, streak, st)
                    if r != "done":
                        break
                    r = drill_phase(cap, det, recovery, backend, labs, args, target,
                                    p, i, len(letters), "hold", args.drill_hold,
                                    streak, st)
                    if r == "redo":
                        continue
                    break
                results[p][target] = st
                hit = st["top1"].get(target, 0)
                print(f"\r  [{p}] {note}  {target}: {st['hand']} frames with a "
                      f"hand, {100 * hit / max(st['hand'], 1):.0f}% answered {target}"
                      + ("" if r == "done" else f"   ({r})"), flush=True)
                if r == "quit":
                    stopped = True
                    break
            if stopped:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if recovery is not None:
            print(f"  {recovery.summary()}")
            recovery.close()
        det.close()

    print("\n== self-test result (nothing saved to disk) ==")
    if stopped:
        print("  stopped early with Q - the table covers the letters attempted.\n")
    for p in profs:
        print_drill_table(p, results[p], letters)
    if len(profs) > 1:
        print_drill_compare(profs, results, letters)
    print("  This is YOUR hand and YOUR camera, so it is the only live accuracy "
          "number in this project. No model here was trained on it - nothing "
          "from this window can reach a training set - so unlike every other "
          "number in the reports it is a genuine out-of-sample test, and it is "
          "the one that says whether the profile fits how you sign.")
    return 0


def print_drill_table(prof: str, res: dict, letters: list[str]) -> None:
    tot = sum(s["frames"] for s in res.values())
    if not tot:
        print(f"\n  profile {prof}: nothing captured.")
        return
    hand = sum(s["hand"] for s in res.values())
    hit = sum(s["top1"].get(l, 0) for l, s in res.items())
    print(f"\n  profile {prof}  ({len(res)} letters, {tot} frames)")
    print("  letter  frames  hand%   hit%   below-gate%  uncertain%  "
          "most common wrong (share of frames with a hand)")
    print("  " + "-" * 96)
    for lab in letters:
        s = res.get(lab)
        if not s or not s["frames"]:
            continue
        n, nh = s["frames"], max(s["hand"], 1)
        hits = s["top1"].get(lab, 0)
        wrong = [(l, c) for l, c in s["top1"].most_common() if l != lab][:2]
        wtxt = ", ".join(f"{l} {100 * c / nh:.0f}%" for l, c in wrong) or "-"
        print(f"  {lab:<7} {s['frames']:>6}  {100 * s['hand'] / n:>5.1f}  "
              f"{100 * hits / nh:>5.1f}  {100 * s['below'] / nh:>10.1f}  "
              f"{100 * s['uncertain'] / nh:>10.1f}   {wtxt}")
    print(f"  {'ALL':<7} {tot:>6}  {100 * hand / max(tot, 1):>5.1f}  "
          f"{100 * hit / max(hand, 1):>5.1f}  "
          f"{100 * sum(s['below'] for s in res.values()) / max(hand, 1):>10.1f}  "
          f"{100 * sum(s['uncertain'] for s in res.values()) / max(hand, 1):>10.1f}")


def print_drill_compare(profs: list[str], results: dict, letters: list[str]) -> None:
    """Side-by-side hit rates, because the whole point is picking a profile."""
    print("\n  hit rate by profile (share of frames with a hand answered right)")
    print("  letter  " + "".join(f"{p:>14}" for p in profs) + "     better")
    wins = {p: 0 for p in profs}
    for lab in letters:
        if not any(results[p].get(lab, {}).get("frames") for p in profs):
            continue
        rates = {}
        for p in profs:
            s = results[p].get(lab) or {}
            nh = s.get("hand", 0)
            rates[p] = 100 * s.get("top1", {}).get(lab, 0) / nh if nh else float("nan")
        best = max((p for p in profs if not np.isnan(rates[p])),
                   key=lambda p: rates[p], default=None)
        if best and not np.isnan(rates[best]):
            wins[best] += 1
        print(f"  {lab:<7} " + "".join(
            f"{rates[p]:>13.1f}%" if not np.isnan(rates[p]) else f"{'-':>14}"
            for p in profs) + f"     {best or '-'}")
    for p in profs:
        cells = [(l, s) for l, s in results[p].items() if s.get("hand")]
        if cells:
            nh = sum(s["hand"] for _l, s in cells)
            hit = sum(s["top1"].get(l, 0) for l, s in cells)
            print(f"  overall {p}: {100 * hit / max(nh, 1):.1f}%   "
                  f"letters won: {wins[p]}")
    print("\n  Pick the profile with the better hit rate for how YOU sign - that "
          "is what\n  `--profile` in live.py and the selector in app.py should "
          "default to.")


def open_camera(index: int, width: int, height: int):
    """Open a webcam proven to deliver frames; tries DirectShow, MSMF, then any.

    Some Windows drivers open fine but never deliver a frame, so is_opened() is
    not enough - one real read is tested per backend.
    """
    import cv2

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    last_err = None
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            last_err = f"backend {be}: could not open"
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _frame = cap.read()
        if ok:
            return cap, be
        last_err = f"backend {be}: opens but will not deliver frames"
        cap.release()
    raise SystemExit(
        f"Could not get a working video stream from camera index {index}.\n"
        f"  last error: {last_err}\n"
        "  - close other apps using the webcam (Teams/Zoom/browser)\n"
        "  - try --camera 1\n"
        "  - Windows Settings > Privacy & security > Camera > allow desktop apps")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.7, help="confidence threshold")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="minimum top1-top2 probability gap; below it the frame "
                         "is shown as UNCERTAIN and casts no vote")
    ap.add_argument("--pair-margins", default=None, metavar="PAIR=VALUE,...",
                    help=f"override the measured per-pair refusal margins "
                         f"(default {format_pair_margins(PAIR_MARGINS)}; "
                         f"'none' = use the flat --margin for every pair). "
                         f"Keys are two letters in either order, e.g. PQ=0.40")
    ap.add_argument("--vote", type=int, default=7, help="frames in the majority vote")
    ap.add_argument("--hold", type=float, default=0.5, help="seconds to hold before commit")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--profile", default=None,
                    help="which trained model to load. Only `all` exists now "
                         "(every signer weighted equally; the old `standard` "
                         "and `round2` models were deleted) "
                         f"(default: {config.PROFILE_DEFAULT}).")
    ap.add_argument("--selftest", "--drill", dest="selftest", action="store_true",
                    help="guided live self-test: show each letter, hold it, "
                         "count the per-frame top-1 in memory and print a hit "
                         "rate table. Needs a camera; saves nothing. "
                         "Takes --letters and a comma list in --profile.")
    ap.add_argument("--letters", default="",
                    help="with --selftest: which letters to test "
                         "(default: every class the model has)")
    ap.add_argument("--drill-hold", type=float, default=3.0,
                    help="with --selftest: seconds of counting per letter")
    ap.add_argument("--drill-countdown", type=float, default=2.0,
                    help="with --selftest: seconds to get the hand into shape "
                         "before counting starts")
    ap.add_argument("--recover", type=int, default=1,
                    help="after N consecutive frames with no hand at all, retry "
                         "that frame once in IMAGE mode at --recover-conf and then "
                         "on a mirrored copy; 0 disables it (round-2 behaviour). "
                         "Tracking successes never pay for it")
    ap.add_argument("--recover-every", type=int, default=3,
                    help="while the hand stays missing, retry every K-th frame "
                         "instead of every frame (1 = always, costs ~12%% "
                         "throughput)")
    ap.add_argument("--recover-conf", type=float, default=0.3,
                    help="confidence used by the recovery retry (default 0.3)")
    ap.add_argument("--no-recover-flip", action="store_true",
                    help="skip the mirrored retry rung")
    ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--no-motion", action="store_true",
                    help="disable the J/Z trajectory rules")
    ap.add_argument("--motion-debug", action="store_true",
                    help="show the J/Z measurements on the overlay")
    ap.add_argument("--calibrate-jz", action="store_true",
                    help="guided J/Z calibration before the typer starts: sign "
                         "each letter a few times and this session's thresholds "
                         "are measured from your strokes. In memory only - "
                         "nothing is written to disk.")
    ap.add_argument("--calibrate-reps", type=int, default=3,
                    help="strokes per letter for --calibrate-jz (default 3)")
    ap.add_argument("--jz", default="",
                    help="override J/Z thresholds for this session, e.g. "
                         "--jz J_NET=0.22,SEG=0.09 (the string --calibrate-jz "
                         "prints). RAM only; keys: "
                         + ", ".join(sorted(MotionDecider.DEFAULTS)))
    ap.add_argument("--logic-test", action="store_true",
                    help="run the smoothing/normalisation unit checks against synthetic landmark sequences: no camera, nothing saved.")
    ap.add_argument("--benchmark", type=int, default=0, help="FPS test with N synthetic frames")
    ap.add_argument("--max-frames", type=int, default=0, help="auto-quit after N frames")
    args = ap.parse_args()
    try:
        args.pairs = resolve_pair_margins(args.pair_margins)
    except ValueError as exc:
        ap.error(f"--pair-margins: {exc}")
    print(f"  refusal margins : flat {args.margin:.2f}"
          + (", pairs " + format_pair_margins(args.pairs) if args.pairs
             else " (no pair rules - --pair-margins none)"))

    if args.selftest:
        return drill(args)
    if args.logic_test:
        return selftest()
    if args.benchmark:
        benchmark(args.benchmark, (args.width, args.height))
        return 0

    profs = _profiles(args.profile)
    if len(profs) > 1:
        raise SystemExit("--profile takes several names only with --selftest "
                         "(the typer runs one model at a time).")
    prof = profs[0]

    import cv2

    print("== live ASL typer ==")
    print("  privacy: frames are processed in memory only; nothing is saved.")
    det = hb.create_detector(static=False)
    recovery = None
    if args.recover:
        recovery = hb.Recovery(conf=args.recover_conf, after=args.recover,
                               every=args.recover_every,
                               flip=not args.no_recover_flip)
        print(f"  detection recovery: retry at {args.recover_conf} confidence "
              f"after {args.recover} empty frame(s), then every "
              f"{args.recover_every}th while the hand stays missing"
              f"{'' if recovery.flip else ' (mirror retry off)'}")
    backend, labels = build_backend(args.backend, profile=prof)
    print(f"  backend={backend.name}  profile={prof}  labels={''.join(labels)}"
          f"{'  (J/Z by trajectory rule)' if not args.no_motion else '  (J/Z disabled)'}")

    cap, be_used = open_camera(args.camera, args.width, args.height)
    print(f"  camera {args.camera} streaming (backend "
          f"{ {cv2.CAP_DSHOW: 'DirectShow', cv2.CAP_MSMF: 'MSMF', cv2.CAP_ANY: 'auto'}.get(be_used, be_used) })")

    sm = Smoother(conf=args.conf, vote=args.vote, hold=args.hold)
    motion = MotionDecider()
    lfilter = smoothing.LandmarkFilter() if smoothing.smoothing_enabled() else None
    if args.jz:
        try:
            th = parse_spec(args.jz)
        except ValueError as exc:
            raise SystemExit(f"--jz: {exc}")
        motion.apply_thresholds(th, source="--jz " + args.jz)
        print("  J/Z thresholds overridden this session: " +
              ", ".join(f"{k}={v:.4f}" for k, v in sorted(th.items())) +
              f"  (defaults were {', '.join(f'{k}={MotionDecider.DEFAULTS[k]:.2f}' for k in sorted(th))})")
    if args.calibrate_jz:
        if args.no_motion:
            print("  (--no-motion also disables --calibrate-jz; skipping)")
        else:
            calibrate_camera(cap, det, recovery, args, motion, backend, labels)
    text = ""
    frames = 0
    streak = 0            # consecutive frames with no hand at all
    t_start = time.time()
    fpsShown = 0.0
    print("  press Q to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("  camera stream ended")
                break
            frames += 1
            now = time.time()
            ts_ms = stamp_ms()
            res, how = hb.detect_frame(det, frame, ts_ms, recovery,
                                       empty_streak=streak + 1)
            streak = streak + 1 if res.landmarks is None else 0
            # Recognition uses the raw landmarks the models were trained on; the
            # filter is only what gets drawn, so the picture is calm and the
            # numbers the models see are unchanged.
            lms = res.landmarks
            if lms is None:
                if lfilter is not None:
                    lfilter.reset()
                drawn = None
            else:
                drawn = lfilter(lms, ts_ms / 1000.0) if lfilter is not None else lms
            call = blank_call()
            if lms is not None:
                if not args.no_landmarks:
                    hb.draw_landmarks(frame, drawn[:, :2])
                feats = normalize.features(lms)
                probs = backend(feats)
                call = frame_call(probs, labels, args.conf, args.margin, pairs=args.pairs)
            held = call["letter"]
            conf = call["conf"]
            k = cv2.waitKey(1) & 0xFF
            force = k == ord(" ")
            # J/Z come from the trained stroke head; the classifier reading picks
            # which frames it is asked about.
            motion_letter = None
            if not args.no_motion:
                if lms is None:
                    motion.reset()
                else:
                    shown = call["top3"][0][0] if call["top3"] else None
                    motion_letter = motion.update(now, lms, shown,
                                                  call["conf"])
                if motion_letter:
                    text += motion_letter
                    committed = motion_letter
            if not args.no_motion and motion.blocks_static(held, now):
                held = None
            held_now, committed, held_conf = sm.step(
                held, conf, now, force=force, hand_present=lms is not None,
            )
            if committed:
                text += committed
            if k == 8:            # backspace
                text = text[:-1]
            elif k == ord("c"):
                text = ""
            elif k in (ord("q"), 27):
                break

            elapsed = now - t_start
            fpsShown = frames / elapsed if elapsed > 0 else 0.0
            pending_s = sm.hold_progress(now)
            render(frame, held_now, held_conf or conf, committed, text, fpsShown,
                   lms is not None, pending_s, args.hold, backend.name,
                   top3=call["top3"], uncertain=call["uncertain"],
                   margin=call["margin"], need=call["need"])
            if call["top3"] and (call["low_conf"] or call["close"]):
                lab, p = call["top3"][0]
                why = (f"below the {args.conf:.2f} confidence gate"
                       if call["low_conf"] else
                       (f"only {call['margin'] * 100:.0f} pts ahead of "
                        f"{call['top3'][1][0]}" +
                        (f", {call['pair'][0]}/{call['pair'][1]} needs "
                         f"{call['need'] * 100:.0f}" if call["collide"] else "")))
                cv2.putText(frame, f"no vote: {lab} {p * 100:.0f}%, {why}",
                            (14, 108), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 200, 255), 1, cv2.LINE_AA)
            if how in ("direct_0.3", "flip_0.3"):
                cv2.putText(frame, f"hand recovered ({how})", (14, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1,
                            cv2.LINE_AA)
            jz_hint = "" if args.no_motion else motion.hint()
            if args.motion_debug:
                d = motion.debug or {}
                info = (f"J/Z held={d.get('held', 0):.2f}s gate={d.get('gate', '?')} "
                        f"model={d.get('top1') or '-'} {d.get('p1', 0):.2f} "
                        f"idx={d.get('idx_ext', 0):.2f} "
                        f"wrist(path={d.get('wrist_path', 0):.2f} rev={d.get('wrist_rev', 0)}) "
                        f"tip(path={d.get('tip_path', 0):.2f} rev={d.get('tip_rev', 0)})"
                        ) if d else "J/Z: neither gate met (index-up shape, or a " \
                                    "reading that was not I or X)"
                cv2.putText(frame, info, (14, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"last motion letter: {motion.last or '-'}", (14, 148),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
            if jz_hint:
                # say why the rule did not fire instead of silently voting I
                cv2.putText(frame, motion.hint(width=62) + "  [--calibrate-jz]",
                            (14, 166 if args.motion_debug else 126),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1,
                            cv2.LINE_AA)
            cv2.imshow("ASL live typer (nothing is saved)", frame)
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if recovery is not None:
            print(f"  {recovery.summary()}")
            recovery.close()
        det.close()
        avg = frames / max(time.time() - t_start, 1e-6)
        print(f"\n  frames processed: {frames}")
        print(f"  average FPS     : {avg:.1f}")
        print(f"  final text      : {text!r}")
        if not args.no_motion:
            spec = format_spec(motion.thresholds())
            print(f"  J/Z rule        : fired {motion.last or 'nothing'} this session; "
                  f"thresholds {motion.threshold_source or 'shipped defaults'}"
                  + (f"\n  to reuse them   : python src/live.py --jz {spec}"
                     if spec else ""))
        if avg < 20:
            print("  below the 20 FPS target - see README.md (the full table is in docs/DETAILS.md)")
        print("  nothing was written to disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
