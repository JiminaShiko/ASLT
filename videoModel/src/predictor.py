"""
Stage 6a - the importable API.

    from src.predictor import WordPredictor
    p = WordPredictor()
    for frame in camera:                     # RGB uint8 [H, W, 3]
        r = p.update(frame)
        # r = {"state": "SIGNING", "top3": [("hello", 0.71), ...],
        #      "committed_word": "", "text": "hello good friend", ...}

What it does, in one pass per call:
  * keeps a rolling buffer of landmark rows sized by BUFFER_SECONDS,
  * decides SIGNING vs REST from the hands rising above the shoulder line
    (the same test src/extract.py used to trim the dataset clips, so the
    segmentation cannot depend on a different rule in the two paths),
  * while signing, runs the one shared feature function and the model over the
    buffer, and reports top-3 with a state of UNCERTAIN when the reading does
    not clear the confidence and margin bars. The gate is CONFIDENCE_MIN 0.85
    by default (measured on the test readings: 75% of real words still typed,
    96% of those right) and both bars are flags, not edits.
  * commits a word once, at the end of the segment, and hands it to
    src/phrases.py. REST and OTHER are never typed, and a hedged pair
    (config.HEDGE_PAIRS, e.g. good/thanks) types neither word - the demo
    shows the hint instead.
  * watches the hand share of the buffer and returns it with a framing hint,
    so "I cannot see your hands" is never displayed as a word.

Everything happens in memory. No frame, no landmark and no text is ever
written to disk by this module, and it is never fitted to webcam data - the
weights come from ASL Citizen only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, features, landmarks, model, phrases, vocab


def _pair_key(key) -> frozenset:
    """Normalise a PAIR_MARGINS key to a set of two lowercase words.

    Both shapes are accepted because the table is hand-edited from a CSV:
    ("please", "sorry") and "please|sorry". Anything else raises - the
    failure mode of a mistyped key is a refusal margin that quietly does
    nothing, which is the last thing anyone should discover after training.
    """
    if isinstance(key, str):
        parts = [p.strip().lower() for p in re.split(r"[|,/]|->", key)
                 if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in key]
    if len(parts) != 2 or parts[0] == parts[1]:
        raise ValueError(f"PAIR_MARGINS key {key!r} is not two different "
                         "words, e.g. ('please', 'sorry')")
    return frozenset(parts)


def required_margin(a: str, b: str, pair_margins=None,
                    floor: float | None = None) -> float:
    """The top1 minus top2 gap needed to trust this reading.

    config.PAIR_MARGINS overrides the global floor for pairs the trained model
    actually swaps (filled from reports/confused_pairs.csv). Order does not
    matter: a pair is a pair. `floor` replaces config.MARGIN_MIN for callers
    that pass it (the --margin flag), so the demo can be loosened or tightened
    without editing config.py.
    """
    pm = config.PAIR_MARGINS if pair_margins is None else pair_margins
    want = frozenset({str(a).lower(), str(b).lower()})
    for key, v in pm.items():
        if _pair_key(key) == want:
            return float(v)
    return float(config.MARGIN_MIN if floor is None else floor)


def hedge_hint(a: str, b: str, hedge_pairs=None) -> str:
    """The hint to show when a reading hits a hedged pair, or "".

    config.HEDGE_PAIRS maps a pair to the text the demo shows instead of a
    commit. decide() refuses the reading no matter how confident it is, so
    neither word of a hedged pair is ever typed. Order does not matter, and
    the same key shapes as PAIR_MARGINS raise on a mistyped key.
    """
    hp = config.HEDGE_PAIRS if hedge_pairs is None else hedge_pairs
    if not hp:
        return ""
    want = frozenset({str(a).lower(), str(b).lower()})
    for key, v in hp.items():
        if _pair_key(key) == want:
            return str(v)
    return ""


def commitable(r: "Reading | None") -> bool:
    """The one commit rule: certain, and not a background class.

    Kept apart from update() so the logic test can drive it with made-up
    readings and still prove REST/OTHER can never reach the phrase buffer.
    """
    return (r is not None and r.certain
            and r.word not in (config.REST_LABEL, config.OTHER_LABEL))


@dataclass
class Reading:
    """One decision over a buffer of frames."""
    word: str
    p1: float
    p2: float
    margin: float
    top3: list[tuple[str, float]]
    certain: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {"word": self.word, "p1": round(self.p1, 4),
                "p2": round(self.p2, 4), "margin": round(self.margin, 4),
                "top3": [[w, round(p, 4)] for w, p in self.top3],
                "certain": self.certain, "reason": self.reason}


def decide(probs: np.ndarray, labels: list[str], pair_margins=None,
           hedge_pairs=None,
           min_conf: float | None = None,
           min_margin: float | None = None) -> Reading:
    """Pure decision function: probability row -> Reading. Nothing else calls
    the thresholds, so the logic test can drive it with made-up rows.

    At the shipped floor the margin cannot bite: p1 >= 0.85 forces
    p1 - p2 >= 2*p1 - 1 = 0.70, which is above any margin a demo would set.
    The margin is the lever for a lowered --conf, not for the default gate.

    A reading that passes both bars still comes back uncertain when the top
    two are a hedged pair (config.HEDGE_PAIRS): the model swaps those words
    confidently, so the demo types neither and shows the hint instead.
    """
    min_conf = config.CONFIDENCE_MIN if min_conf is None else float(min_conf)
    order = np.argsort(-probs)[:3]
    top = [(labels[i], float(probs[i])) for i in order]
    a, b = labels[order[0]], labels[order[1]]
    margin = float(probs[order[0]] - probs[order[1]])
    need = required_margin(a, b, pair_margins, min_margin)
    if float(probs[order[0]]) < min_conf:
        return Reading(a, float(probs[order[0]]), float(probs[order[1]]),
                       margin, top, False,
                       f"confidence {probs[order[0]]:.2f} < {min_conf:.2f}")
    if margin < need:
        return Reading(a, float(probs[order[0]]), float(probs[order[1]]),
                       margin, top, False,
                       f"margin {margin:.2f} < {need:.2f} against '{b}'")
    hint = hedge_hint(a, b, hedge_pairs)
    if hint:
        return Reading(a, float(probs[order[0]]), float(probs[order[1]]),
                       margin, top, False, hint)
    return Reading(a, float(probs[order[0]]), float(probs[order[1]]),
                   margin, top, True, "")


@dataclass
class Segmenter:
    """SIGNING / REST state machine over per-frame hand-above-shoulder flags.

    A segment opens after `hold` consecutive signing frames and closes after
    `rest` consecutive non-signing frames, so one noisy frame cannot start or
    end a word.
    """
    hold: int = 3
    rest: int = 5
    signing: bool = False
    run_sign: int = 0
    run_rest: int = 0
    frames: int = 0

    def update(self, is_signing: bool) -> str:
        self.frames += 1
        self.run_sign = self.run_sign + 1 if is_signing else 0
        self.run_rest = 0 if is_signing else self.run_rest + 1
        if not self.signing and self.run_sign >= self.hold:
            self.signing = True
        elif self.signing and self.run_rest >= self.rest:
            self.signing = False
            self.frames = 0
            return "END"
        return "SIGNING" if self.signing else "REST"

    def reset(self) -> None:
        self.signing = False
        self.run_sign = self.run_rest = 0
        self.frames = 0


def reading_due(frame_i: int, stride: int) -> bool:
    """True on the frames that get a model pass. The geometry test runs on
    every frame; only the expensive part is strided."""
    return frame_i % max(1, int(stride)) == 0


class WordPredictor:
    """The whole recogniser behind one method: update()."""

    def __init__(self, weights: Path | None = None,
                 phrases_path: Path | None = None,
                 fps: float = 30.0,
                 stride: int = config.LIVE_STRIDE,
                 buffer_seconds: float | None = None,
                 min_conf: float | None = None,
                 min_margin: float | None = None,
                 pair_margins=None,
                 hedge_pairs=None,
                 net: "model.WordNet | None" = None,
                 meta: dict | None = None,
                 log=None) -> None:
        self.log = log or config.logger_for("predictor")
        weights = weights or config.FOR_RELEASE
        loaded_from_file = net is None
        if net is None:
            if not weights.is_file():
                raise FileNotFoundError(
                    f"{weights} is missing. Train it first:\n"
                    f'  cd "{config.ROOT}"\n'
                    "  python -m src.extract --splits train,val,test --workers 4\n"
                    "  python -m src.train")
            net, meta = model.load(weights)
        self.net, self.meta = net, dict(meta or {})
        self.weights = Path(weights)
        # where the weights came from: "in memory" only when a caller injected
        # a net, so a log line can never claim a file was read that was not
        self.source = "from file" if loaded_from_file else "in memory"
        self.labels: list[str] = self.meta.get("labels") or vocab.load_vocab().labels
        self.words_only = [i for i, l in enumerate(self.labels)
                           if l not in (config.REST_LABEL, config.OTHER_LABEL)]
        # which feature layout these weights were trained on: the checkpoint
        # meta carries feature_version; otherwise the input width identifies
        # it (170 -> the v2 fair-comparison vector, 200 -> v3). Anything else
        # is stale and refused below rather than guessed at.
        fver = self.meta.get("feature_version")
        self.fver = (int(fver) if fver is not None
                     else features.version_for_dim(self.net.arch.in_dim))
        if self.fver is None or features.feature_dim(self.fver) != \
                self.net.arch.in_dim:
            self.fver = None
            self.stale = (f"the weights expect {self.net.arch.in_dim} "
                          f"features, which matches no layout this code "
                          f"builds (v2={features.feature_dim(2)}, "
                          f"v{config.FEATURE_VERSION}="
                          f"{features.FEATURE_DIM}) - retrain with "
                          f"python -m src.train")
        else:
            self.stale = ""
        self.fps = float(fps)
        self.stride = max(1, int(stride))
        self.min_conf = config.CONFIDENCE_MIN if min_conf is None else min_conf
        self.min_margin = (config.MARGIN_MIN if min_margin is None
                           else float(min_margin))
        self.pair_margins = pair_margins
        self.hedge_pairs = hedge_pairs
        self.buf_s = float(buffer_seconds or config.BUFFER_SECONDS)
        self.buffer = features.RingBuffer(maxlen=max(8, int(self.buf_s * self.fps)))
        # this segment's signing frames only; the decision is built from these
        self.seg_rows: list[np.ndarray] = []
        self.seg = Segmenter()
        self.n_reads = 0
        self.n_decisions = 0
        self._decision_frames = 0
        self._frame_i = 0
        self.book = phrases.load_book(phrases_path)
        self.phrases = phrases.PhraseBuilder(self.book)
        self.tracker: landmarks.Tracker | None = None
        self._last_ts = -1
        self._n_in = 0
        self.last: Reading | None = None
        self.state = "REST"
        self.log.info("loaded %s (%s, %d classes, feature v%s, %d features, "
                      "%d frames, stride %d, buffer %.1f s, gate p1 >= %.2f / "
                      "margin >= %.2f)", self.weights.name,
                      self.source, len(self.labels), self.fver,
                      self.net.arch.in_dim,
                      config.NUM_FRAMES, self.stride, self.buf_s,
                      float(self.min_conf), float(self.min_margin))
        if self.stale:
            self.log.warning(self.stale)

    # --- input: either an RGB frame or an already-extracted landmark row ---
    def _row(self, frame_or_landmarks: np.ndarray,
             ts_ms: int | None = None) -> np.ndarray | None:
        a = np.asarray(frame_or_landmarks)
        if a.ndim == 3 and a.shape[-1] in (3, 4) and a.dtype == np.uint8:
            # a camera frame: run MediaPipe on it, in memory
            if self.tracker is None:
                self.tracker = landmarks.Tracker()
            ts = self._n_in * (1000.0 / max(self.fps, 1.0)) if ts_ms is None \
                else float(ts_ms)
            self._n_in += 1
            ts = int(max(ts, self._last_ts + 1))
            self._last_ts = ts
            row = self.tracker.detect(a[..., :3], ts)
            return row
        if a.shape == (config.P_TOTAL, 3):
            return a.astype(np.float32)
        if a.ndim == 3 and a.shape[1:] == (config.P_TOTAL, 3):
            return a[-1].astype(np.float32)
        raise ValueError(f"update() needs an RGB frame [H,W,3] uint8 or a "
                         f"landmark row [{config.P_TOTAL}, 3], got {a.shape}")

    def _blocks(self) -> list[str] | None:
        b = self.meta.get("feature_blocks")
        return None if not b or b == "all" else list(b)

    def _decision_rows(self) -> list[np.ndarray]:
        """The frames the reading is built from: this segment's signing frames,
        capped to the rolling window.

        The five rest frames that close a segment are deliberately not in it -
        training windows came from src/landmarks.signing_window(), which trims
        exactly those idle frames, so including them would feed the model a
        distribution it never saw.
        """
        return self.seg_rows[-self.buffer.maxlen:]

    def update(self, frame_or_landmarks: np.ndarray,
               ts_ms: int | None = None) -> dict:
        row = self._row(frame_or_landmarks, ts_ms)
        committed = ""
        if row is None:
            return self.result()
        self._frame_i += 1
        self.buffer.append(row)
        flag = landmarks.hands_above_shoulders(row)
        if flag:
            self.seg_rows.append(row)
        step = self.seg.update(flag)
        if step == "END":
            # the segment just closed: decide once, on the whole segment
            reading = self._reading()
            if commitable(reading):
                committed = reading.word
                self.phrases.commit(committed)
            self.buffer.clear()
            self.seg_rows = []
            self.state = "REST"
            self.last = reading
        elif self.seg.signing:
            self.state = "SIGNING"
            # stride = run the model on every Nth frame; the geometry test
            # above still runs on all of them, because it costs nothing
            if reading_due(self._frame_i, self.stride) or self.last is None:
                self.last = self._reading()
                self.n_reads += 1
            if self.last is not None and not self.last.certain:
                self.state = "UNCERTAIN"
        else:
            self.state = "REST"
        out = self.result()
        out["committed_word"] = committed
        out["segment_cut"] = step == "END"
        out["text"] = self.phrases.text()
        return out

    def _reading(self) -> Reading | None:
        rows = self._decision_rows()
        self.n_decisions += 1
        if len(rows) < 4:
            return None
        feats = features.sequence(np.stack(rows), blocks=self._blocks(),
                                  version=self.fver)
        self._decision_frames = len(rows)
        if self.stale:
            # The weights were built for a vector this code cannot build (or
            # a version it does not know). Feeding a wrong-width matrix would
            # raise inside torch, and silently trimming the vector would
            # invent readings. Refuse instead, once per frame, with the
            # command that fixes it.
            return Reading("", 0.0, 0.0, 0.0, [], False, self.stale)
        probs = self.net.predict(feats[None])[0]
        return decide(probs, self.labels, self.pair_margins, self.hedge_pairs,
                      self.min_conf, self.min_margin)

    def framing(self) -> tuple[float | None, str]:
        """(hand share of the live buffer, the framing hint or "").

        A reading built on frames where MediaPipe never saw a hand is not a
        wrong word, it is no information, so the demo has to say so instead of
        showing the argmax. Same measurement as hand_rate_seg in the manifest
        (both wrists, same definition), over the last BUFFER_SECONDS only.
        """
        arr = self.buffer.array()
        if len(arr) < config.FRAMING_MIN_FRAMES:
            return None, ""
        rate = landmarks.hand_share(arr)
        return round(rate, 3), ("" if rate >= config.FRAMING_HAND_RATE_MIN
                                else config.FRAMING_HINT)

    def result(self) -> dict:
        r = self.last
        rate, hint = self.framing()
        return {"state": self.state,
                "top3": [(w, round(p, 4)) for w, p in r.top3] if r else [],
                "committed_word": "",
                "segment_cut": False,
                "text": self.phrases.text(),
                "p1": round(r.p1, 4) if r else 0.0,
                "margin": round(r.margin, 4) if r else 0.0,
                "reason": r.reason if r else "",
                # what the screen shows for the current reading: the word only
                # when the gate is met, otherwise the honest "uncertain"
                "display": ((r.word if r.certain else "uncertain")
                            if r is not None else ""),
                "hand_rate": rate,
                "hint": hint,
                "buffer_frames": len(self.buffer),
                "decision_frames": self._decision_frames,
                "segment_frames": self.seg.frames,
                "hand_in_frame": (landmarks.hand_present(
                    self.buffer.array()[-1])
                    if len(self.buffer) else (False, False))}

    def reset(self) -> None:
        self.buffer.clear()
        self.seg_rows = []
        self.seg.reset()
        self.phrases.reset()
        self.last = None
        self.state = "REST"
        self._frame_i = 0
        self._decision_frames = 0
        if self.tracker is not None:
            self.tracker.restart()
        self._last_ts, self._n_in = -1, 0

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None


# ---------------------------------------------------------------------------
# offline demonstration of the API: no camera, nothing written
# ---------------------------------------------------------------------------
def demo(n: int = 4, weights: Path | None = None) -> None:
    """Drive update() with landmark rows taken from the cache, and print the
    state machine's answers. This is the usage example made runnable."""
    from . import extract
    man = extract.read_manifest()
    man = man[man["status"].astype(str) == "ok"]
    if man.empty:
        raise SystemExit("the landmark cache is empty - run "
                         "python -m src.extract --splits train --limit 3")
    p = WordPredictor(weights=weights)
    rows = []
    for _, r in man.head(n).iterrows():
        f = extract.landmark_file(r["video"])
        if f.is_file():
            arr = np.load(f).astype(np.float32)
            rows.append((r["video"], r["gloss"], arr))
    for video, gloss, arr in rows:
        p.reset()
        out, seen, closed, best = {}, {}, 0, None
        for i, row in enumerate(arr):
            out = p.update(row, ts_ms=int(i * 1000 / p.fps))
            seen[out["state"]] = seen.get(out["state"], 0) + 1
            if out["segment_cut"]:
                closed += 1
                best = out                                   # reading at the cut
        # the final REST line proves nothing on its own: what matters is that
        # the segment actually opened, was read while signing, and was closed.
        print(json.dumps({"video": video, "gloss": gloss, "frames": len(arr),
                          "states_seen": seen, "segments_closed": closed,
                          "at_last_cut": {k: best[k] for k in
                                          ("state", "p1", "margin", "reason")}
                          if best else None,
                          "top3": [[w, round(q, 3)] for w, q in out["top3"]],
                          "committed": best["committed_word"] if best else "",
                          "text": out["text"]}, indent=1))
    p.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WordPredictor demo (no camera).")
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    demo(a.videos, Path(a.weights) if a.weights else None)
