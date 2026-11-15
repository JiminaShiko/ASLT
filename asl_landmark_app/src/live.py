"""Step 5 - the live webcam recogniser.

Pipeline per frame, entirely in memory:
    webcam (BGR) -> cv2.cvtColor BGR2RGB -> MediaPipe Hands -> 21 landmarks
    -> normalize.features() (the SAME function training uses) -> MLP -> letter

Smoothing, top to bottom:
  1. frames whose top-1 confidence is below --conf (default 0.7) cast no vote;
  1b. frames whose top1-top2 gap is below --margin (default 0.10) are shown as
      UNCERTAIN and also cast no vote - two nearly tied letters means one
      finger is being read differently frame to frame; for a handful of letter
      pairs the measured out-of-fold gaps say 0.10 is not enough, and those get
      their own number (PAIR_MARGINS; `--pair-margins none` falls back to the
      flat --margin for every pair);
  2. a majority vote over the last --vote frames (default 7) picks the letter
     currently shown as "held";
  3. hold-to-commit: a letter is appended to the text only after it has been
     the voted letter continuously for --hold seconds (default 0.5);
  4. a letter can never be committed twice in a row - the hand must leave the
     frame or move to a different letter first.

The overlay always shows the top-3 letters with their probabilities, so you can
see what the model thought even when it refused to commit.

NOTHING is written to disk: no frames, no video, no landmarks, no logs. If you
want a record of what it typed, read the on-screen buffer.

Keys: SPACE force-commit · BACKSPACE delete · C clear · Q quit

`--selftest` (also spelled `--drill`) is the guided live self-test: it shows one
letter, gives you two seconds to get into shape, counts the model's raw
per-frame top-1 for three seconds, then moves on, and prints a hit-rate table
when the alphabet is done. It is the only measurement in this project of how the
pipeline performs on your hand, and `--profile all,standard` runs the whole test
once per model so the two can be compared letter by letter. Everything is
counted in memory; the table is printed, never saved.

J and Z are not in the 24-class model (they are moving letters). They are added
back by `MotionDecider`, which reads the *trajectory* of a frame it already has
a reason to suspect: a hand either holds the index-up shape, or the 24-class
model reads that same frame as I (which is what real J looks like to it, at any
confidence) or as an unsure X (which is what real Z looks like). A sideways
hook is J, a fingertip zig-zag is Z. Disable with --no-motion, watch the numbers
with --motion-debug.

Run:  python src/live.py                     the typer, on the default profile
      python src/live.py --selftest          guided self-test on YOUR camera,
                                             saves nothing (alias: --drill)
      python src/live.py --selftest --letters M,N,T --profile all,standard
      python src/live.py --profile standard  typer on one specific model
      python src/live.py --logic-test        smoothing/normalisation checks, no camera
      python src/live.py --benchmark 200     FPS test, no camera
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import config  # noqa: E402
import hand_backend as hb  # noqa: E402
import normalize  # noqa: E402


# ---------------------------------------------------------------------------
# the one clock
# ---------------------------------------------------------------------------
# Every MediaPipe VIDEO timestamp in this file comes from here. It used to be
# `time.monotonic() - <local epoch>` computed inside each loop, and this program
# has three loops that share one VIDEO detector - the typing loop, the J/Z
# calibration and the A-Z drill. Two epochs on one detector means the second
# loop's clock starts at 0 while the detector has already seen thousands of
# milliseconds, and MediaPipe answers with
# `ValueError: Input timestamp must be monotonically increasing.`, on the first
# frame, permanently. That is why `--calibrate-jz` then typing crashed every
# time; `hb.detect_video` now clamps as a backstop, but one process, one clock
# is the actual rule.
_TS_EPOCH = time.monotonic()


def stamp_ms() -> int:
    """Milliseconds since this process started - the only VIDEO timestamp."""
    return int((time.monotonic() - _TS_EPOCH) * 1000)


# ---------------------------------------------------------------------------
# model backends
# ---------------------------------------------------------------------------
class TorchBackend:
    name = "torch"

    def __init__(self, weights: Path, profile: str | None = None):
        import torch

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
    """Load one trained profile. `profile=None` means config.PROFILE_DEFAULT,
    so every front end (live.py, app.py, the self-test) resolves the same file
    and the two of them can never disagree about what is deployed."""
    profile = config.profile(profile)
    meta = json.loads(config.meta_path(profile).read_text(encoding="utf-8"))
    # Fail loudly, and before the first frame, if the model on disk was trained
    # on a different feature set than normalize.py now produces. Without this a
    # 57-dim model fed 71 features dies inside a matmul with a shape error that
    # says nothing about the real cause.
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


# ---------------------------------------------------------------------------
# smoothing state machine
# ---------------------------------------------------------------------------
class Smoother:
    """Confidence gate -> majority vote -> hold-to-commit -> no-repeat.

    Three separate mechanisms, because they fix three different problems:
      * the confidence gate stops the model talking when it is unsure;
      * the majority vote kills single-frame landmark jitter;
      * hold-to-commit stops a letter being typed just because you passed
        through it on the way to the next one;
      * the no-repeat lock stops one long-held letter filling the buffer.

    The hold timer counts from the FIRST frame the hand has looked like that
    letter (not from when the vote locked in), so at 20 FPS a commit takes
    ~0.5 s + one frame, which is what the brief asks for.
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
             force: bool = False, hand_present: bool = True
             ) -> tuple[str | None, str | None, float]:
        """Return (held_letter, letter_committed_now, mean_confidence_of_held)."""
        confident = letter if (hand_present and letter is not None
                               and confidence >= self.conf) else None
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
            # SPACE commits the letter you are holding right now, bypassing the
            # vote window and the timer; the no-repeat rule still applies.
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


# ---------------------------------------------------------------------------
# what to show for one frame: top-3 with probabilities + an uncertain state
# ---------------------------------------------------------------------------
def top3_of(probs: np.ndarray, labels: list[str]) -> list[tuple[str, float]]:
    """The three most likely letters with their probabilities, best first."""
    order = np.argsort(probs)[::-1][:3]
    return [(labels[int(i)], float(probs[int(i)])) for i in order]


# ---------------------------------------------------------------------------
# pair-specific refusal margins
# ---------------------------------------------------------------------------
# The global --margin (0.10) treats every near-tie as equally suspicious, but the
# out-of-fold data says the pairs differ enormously in how much a small gap is
# worth: for P/Q, A/S, M/N and G/H a close second letter almost always means the
# frame is wrong, while for most other pairs a 0.12 gap still means the model is
# right. Each entry below raises the top1-top2 gap a frame needs before it may
# vote, for that pair only. `frame_call` takes the max of this and --margin, so
# an entry can only ever refuse MORE - it can never let a marginal frame through.
#
# MEASURED, not hand-picked. `python src/train.py --profile all --folds-only
# --dump-preds reports/oof_all.csv` writes the top score, the runner-up score and
# the runner-up letter for all 122,359 held-out images, and the table was chosen
# by replaying the live vote window (4-of-7 frames, session-contiguous, never
# mixed across letters) over every one of them:
#
#                                wrong held letters   correct held letters
#   shipped flat 0.10                     13,094                40,134
#   this table                            10,905 (-17%)          39,786 (-0.9%)
#
# 6.3 wrong letters deleted per correct one lost. Getting the same 3.1-point drop
# in the wrong-window rate by raising the flat margin costs 3.7% of correct
# letters, so a flat --margin 0.20 is about four times less efficient - which is
# the whole argument for the table existing. Per letter the gain sits where the
# pairs are: P 54.1% -> 36.8% wrong, Q 30.3% -> 16.4%, H 39.4% -> 33.3%,
# S 36.7% -> 31.7%, A 44.3% -> 39.6%. It holds inside each fold separately
# (grassknoted 11.6% -> 8.1% wrong, massey 39.1% -> 35.6%), so it is not one
# dataset's quirk. `--pair-margins none` turns the table off to A/B it on camera.
#
# What this deliberately does NOT cover is P-as-N, Q-as-N and M-as-S - the three
# collisions that looked like the targets when this was scoped. They are not
# near-ties: the true letter is not even the runner-up in ~90% of those frames
# (only 9.9% of true-P frames have P second, and the median gap on the wrong ones
# is 0.4+), so a top1-top2 rule cannot see them at all. Raising the NP/NQ/MS
# margins to 0.25-0.30 moves the global wrong-window rate by 0.1 points, which is
# why they are absent. The same dump is also why M and N are the two letters to
# re-check on your own camera: when the model says M or N on an unseen hand it is
# right 9.7% and 0.5% of the time (top-1 precision - their per-letter accuracy is
# 3.2% and 0.4%, a different denominator), and still only 14.9% and 0.0% at
# p1 >= 0.85, so confidence does not rescue them.
PAIR_MARGINS: dict[str, float] = {"PQ": 0.50, "AS": 0.50, "MN": 0.50, "GH": 0.50}


def pair_key(a: str, b: str) -> str:
    """The lookup key for a top1 / runner-up pair, order-insensitive.

    Two letters sorted and joined ("NP" for N-vs-P either way), because the
    refusal rule cares about the *pair*, not about which one happened to win:
    the model that confuses P with N produces both directions, and a margin has
    to cover the shape, not the arrow.
    """
    return "".join(sorted((a, b)))


def frame_call(probs: np.ndarray, labels: list[str], conf: float,
               margin: float, pairs: dict[str, float] | None = None) -> dict:
    """Decide what one frame is allowed to say.

    Returns {"letter", "conf", "margin", "need", "low_conf", "close", "collide",
    "pair", "uncertain", "top3"}.

    `letter` is None - the frame casts no vote - for two independent reasons,
    and the overlay names whichever one it is:
      * `low_conf`: top1 is under the confidence gate, i.e. the model is not
        sure it is looking at a letter at all; and
      * `close`: the top1-top2 gap is under the margin this pair needs. With 24
        classes a real near-tie (N 0.45 / S 0.40) is *also* under a 0.7 gate, so
        the two conditions are kept separate and `uncertain` reports the tie,
        not the gate - otherwise the state we were asked to add would never fire.

    The margin is per-pair: `margin` is the floor for every pair, and
    `PAIR_MARGINS` raises it for the four letter pairs whose close second letter
    the out-of-fold data shows is nearly always a wrong answer. `need` is the
    margin actually applied and `collide`/`pair` say whether a named pair rule
    was the binding one, so the overlay can print "P/Q needs 50, got 31" instead
    of refusing silently.

    Marking a tie UNCERTAIN instead of letting it vote is what stops the vote
    window from quietly majority-ruling the wrong letter into the text buffer.
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
    """What `frame_call` would return for a frame with no hand in it at all.

    Every caller that has to render or count a frame before it knows whether
    MediaPipe found anything needs this dict, and it has to carry exactly the
    keys `frame_call` produces: `live.py`'s camera loop reads six of them
    unconditionally, so a hand-written short list used to raise
    `KeyError: 'need'` on the first frame with no hand and take the whole
    webcam session down with it. `--logic-test` now compares this key set
    against a real `frame_call` result, so the two cannot drift apart again.
    """
    return {"letter": None, "conf": 0.0, "margin": 0.0, "need": 0.0,
            "low_conf": False, "close": False, "collide": False, "pair": "",
            "uncertain": False, "top3": []}


def format_pair_margins(pairs: dict[str, float]) -> str:
    """`{'PQ': 0.5}` -> `'PQ=0.50'` - what `--pair-margins` and the app print."""
    return ",".join(f"{k}={pairs[k]:.2f}" for k in sorted(pairs)) or "none"


def resolve_pair_margins(spec: str | None) -> dict[str, float]:
    """Turn `--pair-margins` into the table a session actually uses.

    `None` keeps the measured `PAIR_MARGINS`; `none`/`off`/`flat` drops back to
    the single flat --margin for every pair, which is what you want when A/B
    testing the table on camera; anything else is a `PQ=0.4,AS=0.3` list that
    REPLACES the table (keys are the two letters in either order, because
    `pair_key` sorts them). Bad input raises rather than being ignored - a typo
    here silently changes what the app refuses to say.
    """
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


# ---------------------------------------------------------------------------
# J and Z - trajectory rules (step 7)
# ---------------------------------------------------------------------------
class MotionDecider:
    """Recognise J and Z from how the hand MOVES, never from one frame.

    Why a rule and not a trained model: in a single frame J is "index up,
    everything else curled", which is indistinguishable from a stiff I/L/D, and
    Z is the same shape mid-stroke. The information is entirely in the path, so
    a frame classifier (the 24-class MLP) cannot supply it. Training a motion
    network instead would need recorded fingerspelling video of J and Z, and the
    brief forbids using anything from your webcam - so the rule below is written
    from the shape of the letters and is deliberately conservative.

    All distances are in palm units (the wrist -> middle-MCP bone, the same
    scale the classifier normalises by) so sitting closer or further from the
    camera does not change the thresholds.

      shape gate : index extended, middle/ring/pinky curled, held >= MIN_HOLD s
      display    : or the 24-class model reads the frame as I (-> J candidate,
      gate         any confidence) or as an unsure X below the ceiling
                   (-> Z candidate), named by >= DISPLAY_MIN_SHARE of the
                   frames actually judged
      J          : the hand hooks - it sweeps sideways and comes back once
                   (>= 1 direction reversal of the wrist, net travel, finger
                   still relative to the palm)
      Z          : the hand stays put and the index TIP draws the zig-zag
                   (>= 2 reversals of the tip, plus vertical progress)

    A straight transit between two letters has no reversal, and hand jitter is
    smaller than the hysteresis, so neither fires by accident. When in doubt
    they do not fire: a missing J is better than a J that was never signed.

    The shape gate alone was measured to be **unreachable**, which is the whole
    reason J and Z never appeared: it opens on 0 % of local J, I, S, A, N and M
    frames and 25 % of Z, because ASL Z is a "2" handshape and the
    `others <= 1.20` clause forbids that letter by construction (numbers in
    `reports/jz_trigger.md`). The 24-class MLP reads the same frame far better:
    over 1,200 cached frames per letter it shows true **J as I on 92 %** of
    frames and true **Z as X on 86 %**, and a frame whose top-1 is X with
    `p1 < 0.90` is **93 % truly Z** (96 % below 0.70). So the classifier's own
    reading is a second, independent way *into* the rule - it never replaces
    the trajectory test, it only decides which frames the trajectory is
    measured over.
    """

    WINDOW = 1.20        # s of trajectory kept
    MIN_HOLD = 0.40      # s the J/Z handshape must be held before judging
    COOLDOWN = 1.10      # s of silence after a letter fires
    SEG = 0.18           # palm units of real movement needed to count a reversal
    # J
    J_NET = 0.30         # net sideways travel of the hand
    J_PATH = 0.75        # total path length of the hand
    J_TIP_STILL = 0.40   # the index must stay put relative to the palm
    # Z
    Z_PATH = 0.90        # total sideways path of the fingertip
    Z_DROP = 0.30        # the stroke must also go down/up, not just wobble
    Z_WRIST_MAX = 0.45   # the hand itself must be roughly still

    # The same ten numbers as a dict, because `--calibrate-jz` and `--jz` need
    # to override them for one session and the stroke replayer needs to judge a
    # recorded series under a candidate set. These class attributes stay at the
    # shipped defaults so the logic tests can quote them; a live instance copies
    # them into instance attributes in `__init__` and only those are mutated.
    # Every one of them was chosen from scripted synthetic strokes, which is
    # exactly why a real hand can miss them: `--calibrate-jz` measures YOUR
    # strokes and loosens (never tightens past the default) what is too strict.
    DEFAULTS = {"WINDOW": WINDOW, "MIN_HOLD": MIN_HOLD, "COOLDOWN": COOLDOWN,
                "SEG": SEG, "J_NET": J_NET, "J_PATH": J_PATH,
                "J_TIP_STILL": J_TIP_STILL, "Z_PATH": Z_PATH, "Z_DROP": Z_DROP,
                "Z_WRIST_MAX": Z_WRIST_MAX}
    # The permissive limit of each threshold, in whichever direction that key is
    # permissive (down for the "must travel at least this" keys, up for the
    # "may wander this much" keys). Calibration may never move a number past
    # its own default in the strict direction and never past this in the
    # permissive one, so three tiny or sloppy strokes cannot turn J/Z into a
    # fire-on-everything rule. COOLDOWN's limit equals its default because it is
    # deliberately not calibrated: it protects against double-typing.
    FLOORS = {"WINDOW": 2.00, "MIN_HOLD": 0.25, "COOLDOWN": COOLDOWN,
              "SEG": 0.05, "J_NET": 0.12, "J_PATH": 0.35,
              "J_TIP_STILL": 0.85, "Z_PATH": 0.45, "Z_DROP": 0.12,
              "Z_WRIST_MAX": 0.90}

    # --- the display precondition (rounds 6 and 7) --------------------------
    # Which letter the 24-class MLP has to be showing for a frame to count as a
    # J or Z candidate, and how unsure it has to look. Measured, not guessed:
    # `top-1 == I` with p1 < 0.90 is 81 % truly J (ungated it is a coin flip
    # against real I), and `top-1 == X` with p1 < 0.90 is 93 % truly Z.
    # These are class constants and deliberately NOT in DEFAULTS: they are not
    # trajectory thresholds, so `--calibrate-jz` must never move them, and a
    # threshold key the calibrator cannot reach is a key nobody maintains.
    DISPLAY_FOR = {"I": "J", "X": "Z"}
    DISPLAY_P1_MAX = 0.90      # default ceiling: a confident X is an X, not a Z
    DISPLAY_MIN_SHARE = 0.60   # of the judged window, how many frames must agree
    # Round 7: the ceiling is per candidate, because the two routes are not
    # alike. `None` means no ceiling - and for J that is the honest setting. A
    # J frame *is* an index-up-other-curled frame, which is precisely what a
    # confident I means, so confidence is not evidence against J; only the
    # trajectory rule is. Measured on 400 sampled local J frames: 94 % are read
    # as I at median p1 0.95, and 81 % of those sat above the flat 0.90, so the
    # shipped ceiling refused the precondition on 76 % of real J frames and
    # opened it on 18.2 % - the exemption takes that to 94.2 %. Z keeps 0.90
    # (unchanged, 84.0 % -> 85.0 %) because a real Z is read as X at median p1
    # 0.50: there, unsure really is the signal.
    DISPLAY_P1_MAX_FOR = {"J": None, "Z": DISPLAY_P1_MAX}

    def __init__(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)      # per-instance copy; the class stays clean
        self.threshold_source = ""   # "defaults" / "--jz ..." / "calibrated"
        self.strokes: list[dict] = []    # recorded stroke series (in memory only)
        self.intended: str | None = None  # which letter the next stroke is for
        self._seg: list[tuple[float, float, float, float]] = []
        self.holding = False
        self.last_flags = (False, False)
        self.t: deque[float] = deque(maxlen=400)
        self.wx: deque[float] = deque(maxlen=400)   # wrist x, palm units
        self.tx: deque[float] = deque(maxlen=400)   # index tip x relative to wrist
        self.ty: deque[float] = deque(maxlen=400)   # index tip y relative to wrist
        # which letter each kept frame's classifier reading pointed at ("J",
        # "Z" or None) - same length and same popleft as the series above
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
        """Forget the trajectory so far (hand left, shape lost, or after a fire).

        The segment that just ended is first archived into `self.strokes` if it
        lasted long enough to be a stroke, because that list is what
        `--calibrate-jz` measures. Nothing leaves the process: the archive is a
        few hundred floats per stroke and it is dropped when this object dies.
        """
        self._close_stroke()
        self.t.clear(); self.wx.clear(); self.tx.clear(); self.ty.clear()
        self.dq.clear()
        self.hold_since = None
        self.miss = 0
        self.holding = False
        self.candidates = ()

    MIN_STROKE_S = 0.20       # shorter than this is a twitch, not a stroke
    MIN_STROKE_FRAMES = 6
    MAX_STROKES = 40          # a calibration needs 6; this is only a memory guard

    def _close_stroke(self) -> None:
        """Archive the segment that just ended, for `--calibrate-jz`.

        Recording is gated on `self.intended`, which only the calibrator ever
        sets: an ordinary long typing session must not grow this list. The
        series is trimmed to the trailing WINDOW first, because that is exactly
        the span `update()` judges - measuring a 3 s hold when the rule only
        ever sees the last 1.2 s would derive thresholds that cannot fire.
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
        tx = [s[2] for s in seg]
        ty = [s[3] for s in seg]
        # how much of this stroke the classifier reading supported, and whether
        # it supported the letter the calibrator asked for. Reported, never used
        # to derive a threshold: a stroke the model never saw as I or Z is still
        # the user's real hand, but it is worth showing them the mismatch.
        cand = [s[4] for s in seg]
        share = sum(1 for v in cand if v == self.intended) / max(len(cand), 1)
        self.strokes.append({
            "intended": self.intended,
            "held": seg[-1][0] - seg[0][0], "held_full": held_full,
            "n": len(seg), "wx": wx, "tx": tx, "ty": ty,
            "disp_share": share,
            "wrist_net": abs(wx[-1] - wx[0]), "wrist_path": self._path(wx),
            "wrist_range": max(wx) - min(wx),
            "tip_net": abs(tx[-1] - tx[0]), "tip_path": self._path(tx),
            "tip_range": max(tx) - min(tx), "drop": abs(ty[-1] - ty[0]),
        })
        if len(self.strokes) > self.MAX_STROKES:
            del self.strokes[0]

    @staticmethod
    def _path(xs) -> float:
        a = np.asarray(xs, dtype=np.float64)
        return float(np.abs(np.diff(a)).sum()) if len(a) > 1 else 0.0

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
        """(index-up-and-rest-curled, index extension ratio)."""
        chains = config.FINGER_CHAINS          # [(mcp, pip, dip, tip), ...]
        ratios = []
        for mcp, pip, _dip, tip in chains:
            a = float(np.linalg.norm(lm[pip, :2] - lm[mcp, :2]))
            b = float(np.linalg.norm(lm[tip, :2] - lm[mcp, :2]))
            ratios.append(b / max(a, 1e-6))
        idx_ext, others = ratios[1], ratios[2:]
        up = idx_ext >= 1.6 and all(r <= 1.20 for r in others)
        return bool(up), idx_ext

    @classmethod
    def display_candidate(cls, top1, p1) -> str | None:
        """The J/Z letter the classifier's *own* reading of this frame points at.

        True J is shown as I and true Z as X by the shipped 24-class model (it
        has no J or Z to show), so those two readings are the usable signal. A
        confident X is an X - hence the `p1 < ceiling` clause, which is what
        turns "displays X" from 86 % Z into 93 % Z. Round 7 made the ceiling
        per candidate (`DISPLAY_P1_MAX_FOR`): a confident I is exactly what a J
        looks like, so the J route has no ceiling and rests entirely on the
        trajectory rule, while Z keeps 0.90 because there the unsure reading
        really is the evidence.

        Returns None when there is no reading to use, including when `p1` was
        never supplied, so callers that only know the landmarks keep exactly the
        old shape-gate behaviour.
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
        """Which letters the two preconditions currently permit, judged over
        exactly the window the trajectory rules are about to use.

        The shape gate names no letter, so when it holds both stay possible -
        that is the shipped behaviour and it is not narrowed. The display gate
        does name one, and it has to be the *majority* reading over the window
        (`DISPLAY_MIN_SHARE`), because a single flicker to I in the middle of an
        A would otherwise open a gate and let a hand *travelling between* two
        letters be judged as a J.
        """
        allowed = set()
        if up:
            allowed.update(("J", "Z"))
        n = max(len(dq), 1)
        for letter in ("J", "Z"):
            if sum(1 for v in dq if v == letter) / n >= self.DISPLAY_MIN_SHARE:
                allowed.add(letter)
        return tuple(sorted(allowed))

    @staticmethod
    def _criteria(wx, tx, ty, th: dict) -> dict:
        """Every rule criterion with the number it saw, as (name, value, need, ok).

        This is the single source of truth for both the firing decision and the
        "why didn't that fire" message, so the explanation can never describe a
        rule that is different from the one that ran.
        """
        wr = MotionDecider._reversals(wx, th["SEG"])
        tr = MotionDecider._reversals(tx, th["SEG"])
        wnet = abs(wx[-1] - wx[0])
        wpath = MotionDecider._path(wx)
        tnet = abs(tx[-1] - tx[0])
        tpath = MotionDecider._path(tx)
        drop = abs(ty[-1] - ty[0])
        return {
            "J": [("reversals", wr, 1, wr >= 1),
                  ("net travel", wnet, th["J_NET"], wnet >= th["J_NET"]),
                  ("path", wpath, th["J_PATH"], wpath >= th["J_PATH"]),
                  ("tip wander", tnet, th["J_TIP_STILL"], tnet <= th["J_TIP_STILL"])],
            "Z": [("tip reversals", tr, 2, tr >= 2),
                  ("tip path", tpath, th["Z_PATH"], tpath >= th["Z_PATH"]),
                  ("vertical drop", drop, th["Z_DROP"], drop >= th["Z_DROP"]),
                  ("hand still", wnet, th["Z_WRIST_MAX"], wnet <= th["Z_WRIST_MAX"]),
                  ("no hand hook", wr, 1, wr <= 1)],
        }

    @classmethod
    def _judge(cls, wx, tx, ty, th: dict) -> tuple[bool, bool]:
        """The two boolean rules, given matched series and a threshold dict.

        Shared by `update()` (live, on the windowed deques) and by the
        calibrator (replaying a recorded stroke under a candidate threshold set)
        so the two can never drift apart - the same trap the extractor and the
        live ladder used to be.
        """
        crit = cls._criteria(wx, tx, ty, th)
        return (all(c[3] for c in crit["J"]), all(c[3] for c in crit["Z"]))

    def thresholds(self) -> dict:
        return {k: getattr(self, k) for k in self.DEFAULTS}

    def apply_thresholds(self, th: dict, source: str = "") -> dict:
        """Set this session's J/Z thresholds (in memory only, never on disk).

        Unknown keys and non-numeric values are rejected loudly rather than
        ignored, because a silently dropped override is the kind of thing that
        makes a calibrated session look broken.
        """
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
        return self._criteria(list(self.wx), list(self.tx), list(self.ty),
                              self.thresholds())

    def hint(self, width: int | None = None) -> str:
        """Say why J/Z have not fired, instead of leaving the static model to vote.

        A held J/Z handshape that produces no trajectory is the exact moment the
        console app used to go silent: the 24-class MLP does not contain J or Z,
        so it answers with the closest letter it does have - usually I - and the
        overlay shows a confident-looking wrong letter with no hint that a motion
        rule was waiting to fire. This returns the one sentence that names the
        criterion closest to being met, at the thresholds actually in force.

        Returns "" when there is nothing to explain (no shape held, or a letter
        has just fired), and `explain`-style detail through `criteria()`.
        """
        if not self.holding:
            return ""
        if self.last_now is not None and self.last_now - self.fired_at < self.COOLDOWN:
            return ""          # still in the anti-double-typing gap
        held = self.debug.get("held", 0.0)
        if held < self.MIN_HOLD:
            if self.debug.get("gate") == "display":
                return (f"the model reads this frame as "
                        f"{self.debug.get('top1') or '-'}, so J/Z is watching the "
                        f"movement - hold {self.MIN_HOLD - held:.2f}s longer "
                        f"before it is judged")
            return (f"hold the index-up shape {self.MIN_HOLD - held:.2f}s longer "
                    f"before J/Z is judged")
        if len(self.wx) < 2:
            return ""
        crit = self.criteria()
        if not crit:
            return ""
        # once the window has spoken, only the letters the preconditions permit
        # are worth explaining - "Z held but not fired" is noise when the model
        # has been reading the frame as I
        letters = self.candidates or ("J", "Z")
        for letter in ("J", "Z"):
            if letter not in letters:
                continue
            if all(c[3] for c in crit[letter]):
                return ""            # a rule is met; the fire needs no apology

        def binding(items):
            """The ratio of the criterion furthest from being met - the one that
            has to change before this letter can fire."""
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
        """Feed one frame's raw 21x3 landmarks plus, when the caller has it, what
        the classifier said about the same frame (`top1`, `p1`).

        Returns "J"/"Z" when a precondition held and the trajectory matches,
        otherwise None. Without `top1`/`p1` this is exactly the shipped
        shape-gate rule, which is what the logic tests drive.
        """
        c = normalize.canonicalize(lm)
        palm = float(np.linalg.norm(c[config.MIDDLE_MCP_INDEX] - c[config.WRIST_INDEX]))
        if palm <= 1e-6:
            self.reset()
            return None
        up, idx_ext = self.handshape(c)
        cand = self.display_candidate(top1, p1)
        self.last_top1 = None if top1 is None else str(top1)
        self.last_p1 = 0.0 if p1 is None else float(p1)
        if not up and cand is None:
            # MediaPipe drops the finger tips for a frame now and then; two
            # bad frames in a row and the gesture is over.
            self.miss += 1
            if self.miss > 2:
                self.reset()
            return None
        self.miss = 0
        if self.hold_since is None:
            self.hold_since = now
        w = c[config.WRIST_INDEX, 0] / palm
        tip = c[config.FINGERTIPS[1]] - c[config.WRIST_INDEX]
        self.t.append(now); self.wx.append(w)
        self.tx.append(float(tip[0]) / palm); self.ty.append(float(tip[1]) / palm)
        self.dq.append(cand)
        self._seg.append((now, w, float(tip[0]) / palm, float(tip[1]) / palm, cand))
        self.holding = True          # a precondition met, whatever the path says
        self.last_now = now
        while len(self.t) > 2 and now - self.t[0] > self.WINDOW:
            self.t.popleft(); self.wx.popleft(); self.tx.popleft()
            self.ty.popleft(); self.dq.popleft()

        wx_now, tx_now = list(self.wx), list(self.tx)
        self.debug = {"held": now - self.hold_since, "idx_ext": idx_ext,
                      "top1": self.last_top1, "p1": self.last_p1,
                      "gate": "shape" if up else "display",
                      "cand": cand,
                      "wrist_path": self._path(wx_now), "tip_path": self._path(tx_now),
                      "wrist_net": abs(wx_now[-1] - wx_now[0]),
                      "tip_net": abs(tx_now[-1] - tx_now[0])}
        if now - self.hold_since < self.MIN_HOLD or now - self.fired_at < self.COOLDOWN:
            return None

        allowed = self._allowed(up, self.dq)
        self.candidates = allowed
        wx, tx = wx_now, tx_now
        hit_j, hit_z = self._judge(wx, tx, list(self.ty), self.thresholds())
        self.last_flags = (hit_j, hit_z)
        wr = self._reversals(wx, self.SEG)
        tr = self._reversals(tx, self.SEG)
        drop = abs(self.ty[-1] - self.ty[0])
        self.debug.update({"wrist_rev": wr, "tip_rev": tr, "drop": drop,
                           "allowed": list(allowed)})
        out = None
        if hit_z and "Z" in allowed:
            out = "Z"
        elif hit_j and "J" in allowed:
            out = "J"
        elif (hit_j or hit_z) and not allowed:
            # the path matched but no precondition survived the window - worth
            # saying out loud, because this is the case that used to be invisible
            self.debug["blocked"] = "path matched, precondition did not hold"
        if out:
            self.last_debug = dict(self.debug)
            self.fired_at = now
            self.last = out
            self.reset()
        return out


# ---------------------------------------------------------------------------
# J / Z calibration (in memory only)
# ---------------------------------------------------------------------------
CALIBRATION_SAFETY = 0.72
# every stroke measurement a threshold can be derived from
STROKE_KEYS = ("MIN_HOLD", "SEG", "WINDOW", "J_NET", "J_PATH", "J_TIP_STILL",
               "Z_PATH", "Z_DROP", "Z_WRIST_MAX")


def _stroke_measures(strokes: list[dict], letter: str | None = None) -> list[dict]:
    return [s for s in strokes
            if s.get("intended") == letter and len(s.get("wx", [])) > 2]


def derive_thresholds(strokes: list[dict],
                      safety: float = CALIBRATION_SAFETY) -> tuple[dict, dict]:
    """Turn recorded strokes into a threshold set, and say where each number came from.

    Pure function over the dicts `MotionDecider._close_stroke` archives - no
    camera, no files, so the logic test can feed it scripted strokes and check
    the arithmetic. Returns `(thresholds, notes)` where `notes[key]` explains
    the choice in words.

    Rules:

    * a threshold is only ever moved in the permissive direction, and only as
      far as `MotionDecider.FLOORS[key]` allows. `J_PATH` may come down because
      your hook is shorter than the synthetic template the defaults were written
      against; it may never go up, so a session can never become more
      trigger-happy than shipped;
    * `safety` (0.72) leaves headroom, so a stroke measured at 0.60 of travel
      fires at 0.43 rather than exactly at 0.60;
    * permissive-upper thresholds (`J_TIP_STILL`, `Z_WRIST_MAX`) get the safety
      factor as a divisor, which widens them instead of narrowing them;
    * `SEG`, the reversal hysteresis, is searched rather than scaled: the
      largest value that still counts every recorded stroke's reversals is the
      most noise-resistant one that keeps YOUR strokes working;
    * `WINDOW` is raised (never lowered) to the longest stroke recorded, capped
      at `FLOORS["WINDOW"]`, because a slow hand would otherwise be judged on
      the tail of its own gesture;
    * `COOLDOWN` is never derived from three strokes - it protects against
      double-typing, which no amount of personal style justifies shortening.

    Letters with no strokes keep every one of their thresholds at the default,
    and the notes say so.
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
        loosen_below("J_NET", safety * min(s["wrist_net"] for s in js),
                     f"smallest hook travel {min(s['wrist_net'] for s in js):.2f}")
        loosen_below("J_PATH", safety * min(s["wrist_path"] for s in js),
                     f"shortest hook path {min(s['wrist_path'] for s in js):.2f}")
        widen_above("J_TIP_STILL", max(s["tip_net"] for s in js) / safety,
                    f"most finger wander {max(s['tip_net'] for s in js):.2f}")
    else:
        for k in ("J_NET", "J_PATH", "J_TIP_STILL"):
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
    """Which letter a recorded stroke would produce under `th` (J, Z, both -> Z
    as the live rule prefers it, or None)."""
    hit_j, hit_z = MotionDecider._judge(stroke["wx"], stroke["tx"], stroke["ty"], th)
    return "Z" if hit_z else ("J" if hit_j else None)


def self_consistency(strokes: list[dict], th: dict) -> dict:
    """Replay recorded strokes through the real rule under a candidate set.

    This is the only honest way to report a calibration: the numbers are read
    back through `_judge()`, the same function the live path uses, rather than
    compared with the thresholds by hand.
    """
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
    """Guided "sign J three times, sign Z three times" session, in memory only.

    Both front-ends drive one of these from their own frame loop - the console
    typer in `src/live.py` and the browser app in `app.py` - so the prompts and
    the arithmetic live in exactly one place. Nothing is written to disk: the
    strokes are lists of floats inside `MotionDecider.strokes`, the derived
    numbers go onto that object's instance attributes, and both disappear when
    the process ends. Re-calibrating in the same session simply starts a new
    calibrator.
    """

    LETTERS = ("J", "Z")
    INSTRUCTIONS = {
        "J": "index straight up, the rest curled; sweep the HAND sideways and "
             "hook it back",
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
        """Feed one frame. Returns the letter the live rule fired (if any), so
        typing keeps working while the calibrator watches.

        `top1`/`p1` are the classifier's reading of the same frame. They matter
        here: most of the strokes a user makes as Z never pass the shape gate,
        so without the display precondition the calibrator would archive too few
        Z strokes to measure anything - which is exactly the failure that made
        `--calibrate-jz` look broken from the very first session.
        """
        if self.done:
            return None
        fired = self.motion.update(now, lm, top1, p1)
        if self.await_release:
            # A stroke has to be filed under the letter it was asked for. The
            # third J is often still being held when the Z prompt appears, and
            # recording that tail as "Z" would teach the thresholds a J shape
            # with a Z label. So: nothing is recorded until the handshape has
            # genuinely been let go.
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
    """Run `JZCalibrator` against the webcam until both letters are measured.

    Frames are read and discarded; only the small float series inside the
    MotionDecider survives, and only for this process. ESC skips the rest of the
    current letter, Q abandons calibration and keeps the shipped defaults.

    `backend`/`labels` are optional: with them every stroke frame also carries
    the classifier's own reading, so the display precondition is live and a Z
    that never passes the shape gate is still recorded. Without them this is the
    old shape-gate-only calibration.
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
    """Print a calibration result: measurements, the numbers, and the pasteable spec."""
    print("\n  your strokes:")
    for i, s in enumerate(res["strokes"], 1):
        print(f"    {i}. {s['intended']}  held {s['held_full']:.2f}s  "
              f"hand travel {s['wrist_net']:.2f}/{s['wrist_path']:.2f}  "
              f"finger path {s['tip_path']:.2f}  drop {s['drop']:.2f}")
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


# A plausible open right hand in image coordinates (0..1), used by --benchmark
# and by the motion selftest. Order follows the MediaPipe 21-point layout.
SYNTH_HAND = np.array(
    [[0.5, 0.90, 0], [.45, .75, 0], [.43, .68, 0], [.42, .63, 0], [.41, .58, 0],
     [.47, .68, 0], [.46, .58, 0], [.455, .52, 0], [.45, .47, 0],
     [.50, .67, 0], [.50, .56, 0], [.50, .50, 0], [.50, .45, 0],
     [.53, .68, 0], [.54, .58, 0], [.545, .52, 0], [.55, .47, 0],
     [.56, .69, 0], [.575, .60, 0], [.585, .54, 0], [.59, .49, 0]])


def synth_jshape_hand(curl_others: bool = True) -> np.ndarray:
    """SYNTH_HAND with the index up and (optionally) the other fingers curled.

    The index gets a little longer than the open-hand template because a real
    pointing index is straight, and the shape gate below measures extension as
    tip-to-MCP over PIP-to-MCP.
    """
    lm = SYNTH_HAND.copy()
    lm[config.FINGERTIPS[1]] = [0.45, 0.41, 0.0]
    if curl_others:
        for mcp, pip, _dip, tip in config.FINGER_CHAINS[2:]:
            lm[tip] = lm[mcp] + 0.55 * (lm[pip] - lm[mcp])
    return lm


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------
def render(frame: np.ndarray, held: str | None, conf: float, committed: str | None,
           text: str, fps: float, hand_present: bool, pending_s: float, hold: float,
           backend_name: str, top3: list[tuple[str, float]] | None = None,
           uncertain: bool = False, margin: float = 0.10,
           need: float | None = None) -> np.ndarray:
    """Draw the per-frame state.

    `margin` is the gap the frame actually got (top1-top2) and `need` the margin
    it was required to clear - the two are only different when `PAIR_MARGINS`
    raised the bar for that pair, and printing both is what makes a refusal
    explainable on screen instead of looking like a dropout.
    """
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

    # top-3 probabilities, best first - the model's actual opinion, not just
    # the letter that happened to win
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


# ---------------------------------------------------------------------------
# selftest / benchmark (no camera, nothing read from disk)
# ---------------------------------------------------------------------------
def selftest() -> int:
    print("== live.py selftest (smoothing logic, no camera, nothing saved) ==")
    checks = []

    def _raises(fn) -> bool:
        """True when `fn` raises - the shape an argument parser check needs."""
        try:
            fn()
        except Exception:
            return True
        return False

    def run_seq(s, letters, conf, dt=0.05, t0=0.0, force_last=False):
        """Feed a scripted sequence, return (commits, last_held, last_hconf)."""
        t, commits, held, hc = t0, [], None, 0.0
        for i, l in enumerate(letters):
            t += dt
            held, c, hc = s.step(l, conf, t, force=(force_last and i == len(letters) - 1),
                                 hand_present=(l is not None))
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

    # ---- J / Z trajectory rules -------------------------------------------
    def run_motion(lms, dt=0.05):
        """Feed a scripted trajectory; return (letters, fire times, decider)."""
        md = MotionDecider()
        out, times, t = [], [], 0.0
        for lm in lms:
            t += dt
            r = md.update(t, lm)
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
        """Index tip traces a Z (3 segments) while the hand stays put.

        `scale` shrinks the whole zig-zag, which is how the calibration checks
        below imitate a signer whose Z is smaller than the template the shipped
        thresholds were written against.
        """
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

    jshape = synth_jshape_hand()
    checks.append(("the J/Z handshape is detected as index-up, others curled",
                   MotionDecider.handshape(normalize.canonicalize(jshape))[0]))
    open_hand = synth_jshape_hand(curl_others=False)
    checks.append(("an open hand is NOT the J/Z handshape",
                   not MotionDecider.handshape(normalize.canonicalize(open_hand))[0]))

    m1, _t1, _ = run_motion(seq_offset(jshape, [0.0] * 30))
    checks.append(("a still J-shape fires nothing", m1 == []))

    m2, _t2, _ = run_motion(seq_offset(jshape, straight_xs(30, 0.22)))
    checks.append(("a straight hand transit is not a J", m2 == []))

    m3, _t3, md3 = run_motion(seq_offset(jshape, hook_xs(30, 0.22, 0.075)))
    checks.append((f"the sideways hook commits J (at fire: {md3.last_debug})",
                   m3 == ["J"]))

    m4, _t4, md4 = run_motion(seq_zigzag(jshape))
    checks.append((f"the fingertip zig-zag commits Z (at fire: {md4.last_debug})",
                   m4 == ["Z"]))

    m5, _t5, _ = run_motion(seq_offset(open_hand, hook_xs(30, 0.22, 0.075)))
    checks.append(("the same hook with an open hand fires nothing", m5 == []))

    rng_j = np.random.default_rng(3)
    jitter = [jshape + rng_j.normal(0, 0.006, (21, 3)) for _ in range(70)]
    m6, _t6, _ = run_motion(jitter)
    checks.append(("landmark jitter alone does not fire J or Z", m6 == []))

    # repeated gestures must be spaced by the cooldown, never double-typed
    triple = seq_offset(jshape, hook_xs(30, 0.22, 0.075)) * 3
    m7, t7, _ = run_motion(triple)
    gaps = [b - a for a, b in zip(t7, t7[1:])]
    checks.append((f"three hooks type {len(m7)} letters, min gap "
                   f"{(min(gaps) if gaps else float('nan')):.2f}s >= cooldown "
                   f"{MotionDecider.COOLDOWN}s",
                   len(m7) >= 1 and all(g >= MotionDecider.COOLDOWN - 1e-9 for g in gaps)))

    m8, _t8, _ = run_motion(seq_offset(jshape, hook_xs(30, 0.22, 0.075) + [0.0] * 40
                                       + hook_xs(30, 0.22, 0.075)))
    checks.append(("a second hook after the cooldown does fire again", len(m8) == 2))

    # --- top-3 / uncertain gate (Stage F) ---
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

    # --- pair-specific refusal margins (round 5, item 3) ---
    # A/S is in PAIR_MARGINS at 0.50; A/B is not. So a 30-point gap that the flat
    # 0.10 margin happily waves through must be refused for A/S and allowed for
    # A/B, and the frame must SAY which rule bound it.
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

    # ---- the no-hand frame contract (round 5b regression guard) -----------
    # The camera loop must draw a frame it got no landmarks from, and it did
    # that from a hand-written placeholder dict. When `frame_call` grew `need`,
    # `close`, `collide` and `pair` the placeholder did not, so the FIRST frame
    # with no hand raised `KeyError: 'need'` out of `main()` and killed the
    # session - which looks exactly like "the webcam does not work". These three
    # checks make that impossible to repeat: same keys, no opinion, and the
    # render call built the way `main` builds it.
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

    # ---- one clock per detector (the `--calibrate-jz` crash) --------------
    # `--calibrate-jz` was followed by the typing loop on the SAME VIDEO
    # detector, and each loop computed its timestamp from its own
    # `time.monotonic()` epoch. The second loop therefore started at 0 ms while
    # the detector had already seen thousands, and MediaPipe raised
    # `ValueError: Input timestamp must be monotonically increasing.` on its
    # first frame - every single time, which is why the feature as shipped could
    # never actually be used. Two guards: `hb.detect_video` clamps per detector,
    # and this file is only allowed one epoch subtraction, inside `stamp_ms()`.
    class StrictDetector:
        """MediaPipe VIDEO as far as timestamps go: a non-increasing stamp
        raises, and nothing short of a new detector recovers."""

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
    # AST, not text: a comment - or this check's own source - must not be able
    # to satisfy or break it. `stamp_ms` is the only place allowed to turn
    # `monotonic() - epoch` into a timestamp, and the camera loops must call it.
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

    # --- chirality / handedness labelling (round 4 fix) ---
    # A `flip_0.3` hit is mapped back into the original frame (x -> 1-x), so its
    # MediaPipe label has to be mapped back too. The live rung and the offline
    # ladder must agree, and neither may be presented as anatomy.
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

    # ---- J / Z calibration + fail-loud messaging (round 5) ---------------
    # The shipped thresholds were written against the synthetic templates
    # above, which is precisely why a real hand with a smaller gesture can hold
    # the shape forever and type nothing. These checks drive the calibration
    # with those smaller strokes and assert it fixes them - and nothing else.
    def record(lms, letter, reps=3, dt=0.05):
        """Feed a trajectory `reps` times with the calibrator's label set."""
        md = MotionDecider()
        md.intended = letter
        t = 0.0
        for _ in range(reps):
            for lm in lms:
                t += dt
                md.update(t, lm)
            md.reset()
        md.intended = None
        return md.strokes

    def refeed(lms, th, reps=3, dt=0.05):
        """Re-run a trajectory live through a decider carrying `th`."""
        md = MotionDecider()
        md.apply_thresholds(th)
        out, t = [], 0.0
        for _ in range(reps):
            for lm in lms:
                t += dt
                r = md.update(t, lm)
                if r:
                    out.append(r)
            md.reset()
        return out

    small_j = seq_offset(jshape, hook_xs(24, 0.11, 0.04))
    small_z = seq_zigzag(jshape, n=22, scale=0.5)
    sj, sz = record(small_j, "J"), record(small_z, "Z")
    both = sj + sz
    D = MotionDecider.DEFAULTS
    checks.append((f"the small strokes really do miss at the shipped defaults "
                   f"(replay {self_consistency(both, D)['rate'] * 100:.0f}%)",
                   self_consistency(both, D)["rate"] == 0.0
                   and refeed(small_j, D) == [] and refeed(small_z, D) == []))
    th, notes = derive_thresholds(both)
    checks.append((f"derive_thresholds makes every recorded stroke fire in "
                   f"replay (was {self_consistency(both, D)['fired']}/{len(both)}, "
                   f"now {self_consistency(both, th)['fired']}/{len(both)})",
                   self_consistency(both, th)["rate"] == 1.0
                   and self_consistency(both, th)["wrong_letter"] == 0))
    checks.append((f"and the calibrated decider fires them live "
                   f"(J {refeed(small_j, th)}, Z {refeed(small_z, th)})",
                   len(refeed(small_j, th)) >= 2 and len(refeed(small_z, th)) >= 2))
    below = ("MIN_HOLD", "SEG", "J_NET", "J_PATH", "Z_PATH", "Z_DROP")
    above = ("J_TIP_STILL", "Z_WRIST_MAX", "WINDOW")
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
                   and parse_spec(" j_path = 0.5 ") == {"J_PATH": 0.5}
                   and all(_raises(lambda: parse_spec(bad)) for bad in
                           ("SEG=nope", "NOPE=1", "SEG"))))
    md_none = MotionDecider()
    refeed(small_j, D, reps=3)
    checks.append(("an ordinary session records nothing (no unbounded stroke list)",
                   len(md_none.strokes) == 0))
    checks.append(("strokes are only archived while a letter is intended",
                   len(sj) == 3 and all(s["intended"] == "J" for s in sj)))

    # the hint: a held shape with no trajectory must say what is missing
    md_h = MotionDecider()
    t = 0.0
    for lm in small_j[:16]:
        t += 0.05
        md_h.update(t, lm)
    h = md_h.hint()
    checks.append((f"a held shape that does not fire explains itself ({h!r})",
                   "not fired" in h and "reversals" in h and "needs" in h))
    md_h2 = MotionDecider()
    md_h2.update(0.05, small_j[0])
    checks.append(("before MIN_HOLD the message is to keep holding",
                   "hold the index-up shape" in md_h2.hint()))
    fired, _tt, md_ok = run_motion(seq_offset(jshape, hook_xs(30, 0.22, 0.075)))
    checks.append(("and while the rule is genuinely met nothing is complained about",
                   fired == ["J"] and md_ok.hint() == ""))
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
    tt, guard = 0.0, 0
    while not cal.done and guard < 4000:
        for lm in (small_j if cal.letter == "J" else small_z):
            tt += 0.05
            guard += 1
            # what the shipped 24-class model says about these frames: J is
            # shown as I, Z as X (`reports/jz_trigger.md`)
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

    # ---- the display precondition (round 6) --------------------------------
    # The shape gate was measured to open on 0 % of local J frames and 25 % of
    # local Z frames, so a rule that needs it can never fire however good the
    # trajectory is. These checks pin the second way in - and its limits.
    def feed(lms, top1=None, p1=None, dt=0.05):
        md = MotionDecider()
        out, t = [], 0.0
        for lm in lms:
            t += dt
            r = md.update(t, lm, top1, p1)
            if r:
                out.append(r)
        return out, md

    hook_open = seq_offset(open_hand_c, hook_xs(30, 0.22, 0.075))
    ramp_open = seq_offset(open_hand_c, straight_xs(30, 0.25))
    zig_open = seq_zigzag(open_hand_c, n=22, scale=1.0)
    checks.append(("the open hand really is rejected by the shape gate, so every "
                   "check below runs on the display path alone",
                   not MotionDecider.handshape(
                       normalize.canonicalize(open_hand_c))[0]))
    checks.append((f"an unsure I reading turns that hook into a J "
                   f"(no reading: {feed(hook_open)[0]}, with one: "
                   f"{feed(hook_open, 'I', 0.60)[0]})",
                   feed(hook_open)[0] == []
                   and feed(hook_open, "I", 0.60)[0] == ["J"]))
    checks.append((f"an unsure X reading turns the zig-zag into a Z "
                   f"(no reading: {feed(zig_open)[0]}, with one: "
                   f"{feed(zig_open, 'X', 0.55)[0]})",
                   feed(zig_open)[0] == []
                   and feed(zig_open, "X", 0.55)[0] == ["Z"]))
    checks.append(("a confident X is still an X, not a Z: no candidate, so no fire",
                   feed(zig_open, "X", 0.95)[0] == []
                   and MotionDecider.display_candidate("X", 0.90) is None
                   and MotionDecider.display_candidate("X", 0.89) == "Z"))
    # Round 7: the ceiling is per candidate, and J has none. What makes that
    # safe is not the reading - it is that a still I has no reversal, no net
    # travel and no path, so the trajectory rule refuses it by itself.
    checks.append(("a confident I is what a J looks like: the same hook fires "
                   "J at p1 0.99",
                   feed(hook_open, "I", 0.99)[0] == ["J"]
                   and MotionDecider.display_candidate("I", 0.99) == "J"
                   and MotionDecider.display_candidate("I", 0.50) == "J"))
    checks.append(("and an exempt I still needs a J: the same confident reading, "
                   "held still, types nothing "
                   f"({feed(seq_offset(jshape, [0.0] * 30), 'I', 0.99)[0]})",
                   feed(seq_offset(jshape, [0.0] * 30), "I", 0.99)[0] == []
                   and feed(seq_offset(jshape, straight_xs(30, 0.25)), "I", 0.99)[0] == []))
    checks.append(("the exemption table covers J only, Z keeps the shipped "
                   "ceiling, and neither is reachable by --calibrate-jz",
                   MotionDecider.DISPLAY_P1_MAX_FOR["J"] is None
                   and MotionDecider.DISPLAY_P1_MAX_FOR["Z"] == MotionDecider.DISPLAY_P1_MAX
                   and not ({"DISPLAY_FOR", "DISPLAY_P1_MAX", "DISPLAY_P1_MAX_FOR",
                             "DISPLAY_MIN_SHARE"} & set(MotionDecider.DEFAULTS))))
    checks.append(("a reading that names neither I nor X is no candidate",
                   feed(hook_open, "S", 0.55)[0] == []
                   and feed(hook_open, None, None)[0] == []))
    # Both readings open the gate (I -> J, X -> Z), so the trajectory is the
    # same one that just fired - only the *share* over the judged window differs.
    alt, blocked = MotionDecider(), 0
    tt_alt = 0.0
    for i, lm in enumerate(hook_open):
        tt_alt += 0.05
        alt.update(tt_alt, lm, "I" if i % 2 == 0 else "X", 0.5)
        blocked += 1 if alt.debug.get("blocked") else 0
    checks.append((f"a reading that flickers between the two candidates is "
                   f"neither of them ({blocked} frames matched the path and were "
                   f"refused)",
                   alt.last is None and blocked > 0))
    def feed_by(lms, readings, dt=0.05):
        """Same as `feed`, with a per-frame reading (a real model flickers)."""
        md = MotionDecider()
        out, t = [], 0.0
        for lm, (top1, p1) in zip(lms, readings):
            t += dt
            r = md.update(t, lm, top1, p1)
            if r:
                out.append(r)
        return out

    # 3 of 4 frames read I, 1 reads X: the window majority names J, so this
    # fires - the share rule only refuses ties, not imperfect readings. (Both
    # readings open the gate, so the trajectory itself is untouched.)
    three_of_four = [("I", .5) if i % 4 else ("X", .5) for i in range(len(hook_open))]
    checks.append((f"a reading that agrees on three frames in four is a J "
                   f"candidate ({feed_by(hook_open, three_of_four)})",
                   feed(hook_open, "I", 0.5)[0] == ["J"]
                   and feed_by(hook_open, three_of_four) == ["J"]
                   and MotionDecider.DISPLAY_MIN_SHARE > 0.5))
    checks.append((f"a straight transit still types nothing under the new gate "
                   f"({feed(ramp_open, 'I', 0.5)[0]})",
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
    """Pre-load frames from the PUBLIC dataset images already on disk, into RAM.

    Deliberately not your webcam: the brief forbids touching your images, and a
    benchmark has to be reproducible. These are the same photographs the model
    trained on, so MediaPipe actually finds a hand and the timing covers the
    real detect -> normalise -> classify path instead of the cheap
    no-detection path (which is what a drawn stick figure measures, at ~8x the
    wrong speed).
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


# ---------------------------------------------------------------------------
# Stage D: guided live self-test -- the camera, your own hand, nothing saved
# ---------------------------------------------------------------------------
def _profiles(spec: str | None) -> list[str]:
    """`--profile all,standard` -> ['all', 'standard']; validates each name."""
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
    """One countdown or capture phase for one letter. Returns quit/skip/redo/done.

    `mode='count'` reads and draws frames but records nothing, so the hand can
    settle; `mode='hold'` counts every frame's raw top-1 in memory. Counting the
    raw argmax (not the smoothed letter on screen) is deliberate: the drill asks
    "does the model recognise this shape from this hand", and the gate, the vote
    and the hold timer each answer a different question - their effect is
    reported separately as the gate/uncertain columns.
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
                # the RAW argmax, not call["letter"]: a frame the gate refused
                # still has an opinion, and the drill is measuring the opinion.
                # What the gate did with it is the below-gate/uncertain columns.
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


# ---------------------------------------------------------------------------
def open_camera(index: int, width: int, height: int):
    """Open a webcam and return a capture that is PROVEN to deliver frames.

    On some Windows camera drivers (this machine's USB2.0 HD UVC WebCam among
    them) the default MSMF backend opens fine but every cap.read() fails with
    'can't grab frame', so is_opened() alone is not enough. We therefore test
    one real frame per backend and fall through DirectShow -> MSMF -> any
    until one actually streams. Nothing is read from or written to disk.
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


# ---------------------------------------------------------------------------
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
                    help="which trained model to load: all = every style "
                         "equally, standard = minority-style cells removed from "
                         "training, round2 = the round-2 shipped model "
                         f"(default: {config.PROFILE_DEFAULT}). The guided "
                         "self-test also takes a comma list, e.g. "
                         "`--profile all,standard`, and runs both back to back.")
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
                         "--jz J_PATH=0.52,SEG=0.09 (the string --calibrate-jz "
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
    streak = 0            # consecutive frames where no hand was found at all
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
            call = blank_call()
            if res.landmarks is not None:
                if not args.no_landmarks:
                    hb.draw_landmarks(frame, res.landmarks[:, :2])
                feats = normalize.features(res.landmarks)
                probs = backend(feats)
                call = frame_call(probs, labels, args.conf, args.margin, pairs=args.pairs)
            held = call["letter"]
            conf = call["conf"]
            k = cv2.waitKey(1) & 0xFF
            force = k == ord(" ")
            held_now, committed, held_conf = sm.step(
                held, conf, now, force=force, hand_present=res.landmarks is not None)
            if committed:
                text += committed

            # J and Z come from the trajectory, not from the classifier - but
            # the classifier's reading of the frame says which frames to measure
            motion_letter = None
            if not args.no_motion:
                if res.landmarks is None:
                    motion.reset()
                else:
                    shown = call["top3"][0][0] if call["top3"] else None
                    motion_letter = motion.update(now, res.landmarks, shown,
                                                  call["conf"])
                if motion_letter:
                    text += motion_letter
                    committed = motion_letter
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
                   res.landmarks is not None, pending_s, args.hold, backend.name,
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
                                    "reading that was not I / not an unsure X)"
                cv2.putText(frame, info, (14, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"last motion letter: {motion.last or '-'}", (14, 148),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
            if jz_hint:
                # The old behaviour here was silence: the shape gate was held,
                # the trajectory did not clear a threshold, the rule returned
                # nothing, and the 24-class model voted the closest letter it
                # actually has - usually I - so a missed J looked like a wrong
                # answer instead of a rule that did not fire.
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
