"""Stage 6a - the importable API: `WordPredictor().update(rgb_frame)` -> reading dict.

Buffers landmarks, splits SIGNING/REST by the same hand-above-shoulder test the
extractor used, gates readings on confidence/margin, commits one word per
segment and shows a framing hint when no hand is visible. All in memory; the
weights come from ASL Citizen only.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, features, landmarks, model, phrases, vocab


def _pair_key(key) -> frozenset:
    """Normalise a pair key (tuple or "a|b") to a frozenset; a mistyped key must raise, not no-op."""
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
    """Top1 - top2 gap needed to trust a reading; PAIR_MARGINS overrides the floor, order-free."""
    pm = config.PAIR_MARGINS if pair_margins is None else pair_margins
    want = frozenset({str(a).lower(), str(b).lower()})
    for key, v in pm.items():
        if _pair_key(key) == want:
            return float(v)
    return float(config.MARGIN_MIN if floor is None else floor)


def hedge_hint(a: str, b: str, hedge_pairs=None) -> str:
    """Hint for a hedged top-2 pair, or ""; neither word is ever typed."""
    hp = config.HEDGE_PAIRS if hedge_pairs is None else hedge_pairs
    if not hp:
        return ""
    want = frozenset({str(a).lower(), str(b).lower()})
    for key, v in hp.items():
        if _pair_key(key) == want:
            return str(v)
    return ""


def commitable(r: "Reading | None") -> bool:
    """The one commit rule: certain, and not a background class (kept pure for tests)."""
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
    """Pure decision: probability row -> Reading.

    At the shipped floor the margin cannot bite (p1 >= 0.85 implies a gap of
    0.70), so it only matters for a lowered --conf. Hedged top-2 pairs stay
    uncertain even when both bars pass.
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
    """SIGNING / REST state machine; `hold`/`rest` debounce out single noisy frames."""
    hold: int = 3
    rest: int = 5
    signing: bool = False
    run_sign: int = 0
    run_rest: int = 0
    frames: int = 0

    def update(self, is_signing: bool) -> str:
        self.run_sign = self.run_sign + 1 if is_signing else 0
        self.run_rest = 0 if is_signing else self.run_rest + 1
        self.frames = self.frames + 1 if self.signing else self.run_sign
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
    """True on frames that get a model pass; the geometry test runs every frame."""
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
                 log=None,
                 manual: bool = False) -> None:
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
        self.source = "from file" if loaded_from_file else "in memory"
        labels = self.meta.get("labels")
        self.labels: list[str] = list(vocab.load_vocab().labels if labels is None
                                     else labels)
        classes = getattr(self.net.arch, "num_classes", None)
        if classes is not None and len(self.labels) != classes:
            raise ValueError(f"checkpoint labels ({len(self.labels)}) do not match "
                             f"net.arch.num_classes ({classes})")
        if len(self.labels) < 2:
            raise ValueError("at least two checkpoint labels are required")
        self.words_only = [i for i, l in enumerate(self.labels)
                           if l not in (config.REST_LABEL, config.OTHER_LABEL)]
        # layout from checkpoint meta, else inferred from the input width; never guessed
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
        # rest tail kept apart so closing debounce cannot evict the last signing frames
        self.seg_rows: list[np.ndarray] = []
        self._onset_rows: list[np.ndarray] = []
        self._rest_rows: list[np.ndarray] = []
        self._previous_row: np.ndarray | None = None
        self.seg = Segmenter()
        self.manual = bool(manual)
        self.manual_open = False
        self._decision_hand_rate: float | None = None
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

    def _row(self, frame_or_landmarks: np.ndarray,
             ts_ms: int | None = None) -> np.ndarray | None:
        a = np.asarray(frame_or_landmarks)
        if a.ndim == 3 and a.shape[-1] in (3, 4) and a.dtype == np.uint8:
            if self.tracker is None:
                self.tracker = landmarks.Tracker()
            ts = time.monotonic() * 1000.0 if ts_ms is None else float(ts_ms)
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
        """Contiguous window plus one end-context frame; debounce frames are not included."""
        return (self.seg_rows + self._rest_rows[:1])[-self.buffer.maxlen:]

    def _append_segment(self, rows: list[np.ndarray]) -> None:
        self.seg_rows.extend(rows)
        del self.seg_rows[:-self.buffer.maxlen]

    def _clear_segment(self) -> None:
        self.seg_rows.clear()
        self._onset_rows.clear()
        self._rest_rows.clear()
        self._previous_row = None
        self.seg.reset()
        self.manual_open = False

    def begin_segment(self) -> dict:
        """Open a manual segment; repeat calls keep it, switching from auto discards partials."""
        if self.manual_open:
            return self.result()
        self._clear_segment()
        self.buffer.clear()
        self.manual = self.manual_open = True
        self.seg.signing = True
        self.last = None
        self._decision_frames = 0
        self._decision_hand_rate = None
        self.state = "SIGNING"
        return self.result()

    def end_segment(self) -> dict:
        """Cut a manual segment once, including an explicit too-short refusal."""
        if not self.manual_open:
            out = self.result()
            out["reason"] = "no open manual segment"
            return out
        return self._finish_segment()

    def _finish_segment(self) -> dict:
        self.last = self._reading()  # read at the cut, never peak-pick
        committed = ""
        if commitable(self.last):
            committed = self.last.word
            self.phrases.commit(committed)
        rate = self._decision_hand_rate
        self._clear_segment()
        self.state = "REST"
        out = self.result()
        out.update(committed_word=committed, segment_cut=True,
                   hand_rate=rate,
                   hint=(config.FRAMING_HINT if rate is not None and
                         rate < config.FRAMING_HAND_RATE_MIN else ""))
        return out

    def _refresh_reading(self) -> None:
        self.state = "SIGNING"
        if reading_due(self._frame_i, self.stride) or self.last is None:
            self.last = self._reading()
            self.n_reads += 1
        if self.last is not None and not self.last.certain:
            self.state = "UNCERTAIN"

    def update(self, frame_or_landmarks: np.ndarray,
               ts_ms: int | None = None) -> dict:
        row = self._row(frame_or_landmarks, ts_ms)
        if row is None:
            return self.result()
        self._frame_i += 1
        self.buffer.append(row)
        if self.manual:
            if self.manual_open:
                self._append_segment([row])
                self.seg.frames += 1
                self._refresh_reading()
            return self.result()

        flag = landmarks.hands_above_shoulders(row)
        was_signing = self.seg.signing
        onset_length = self.seg.run_sign
        step = self.seg.update(flag)
        if was_signing:
            if flag:
                self._append_segment(self._rest_rows + [row])
                self._rest_rows.clear()
            else:
                self._rest_rows.append(row)
            if step == "END":
                out = self._finish_segment()
                self._previous_row = row
                return out
        elif flag:
            if onset_length == 0:
                self._onset_rows = ([] if self._previous_row is None else
                                    [self._previous_row])
                self.last = None
                self._decision_frames = 0
                self._decision_hand_rate = None
            self._onset_rows.append(row)
            if self.seg.signing:
                self._append_segment(self._onset_rows)
                self._onset_rows.clear()
        else:
            if onset_length:
                self.last = Reading("", 0.0, 0.0, 0.0, [], False,
                                    f"too short: onset has {onset_length} active "
                                    f"frames; needs {self.seg.hold} consecutive")
            self._onset_rows.clear()
        self._previous_row = row
        if self.seg.signing:
            self._refresh_reading()
        else:
            self.state = "REST"
        return self.result()

    def _reading(self) -> Reading:
        rows = self._decision_rows()
        self.n_decisions += 1
        self._decision_frames = len(rows)
        arr = np.stack(rows) if rows else None
        self._decision_hand_rate = (landmarks.hand_share(arr)
                                    if arr is not None else None)
        # an accepted onset is readable even at stream start; manual uses the same min
        if len(rows) < self.seg.hold:
            return Reading("", 0.0, 0.0, 0.0, [], False,
                           f"too short: {len(rows)} frames; needs {self.seg.hold}")
        if self._decision_hand_rate == 0.0:
            return Reading("", 0.0, 0.0, 0.0, [], False,
                           "no hands detected in segment")
        if self.stale:
            return Reading("", 0.0, 0.0, 0.0, [], False, self.stale)
        feats = features.sequence(arr, blocks=self._blocks(), version=self.fver)
        probs = self.net.predict(feats[None])[0]
        return decide(probs, self.labels, self.pair_margins, self.hedge_pairs,
                      self.min_conf, self.min_margin)

    def framing(self) -> tuple[float | None, str]:
        """(hand share of the live buffer, framing hint or ""); no hand = no info, not a guess."""
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
        self._clear_segment()
        self.phrases.reset()
        self.last = None
        self.state = "REST"
        self._frame_i = 0
        self._decision_frames = 0
        self._decision_hand_rate = None
        if self.tracker is not None:
            self.tracker.restart()
        self._last_ts, self._n_in = -1, 0

    def close(self) -> None:
        try:
            if self.tracker is not None:
                self.tracker.close()
        finally:
            self.tracker = None
            self.reset()


def demo(n: int = 4, weights: Path | None = None) -> None:
    """Drive update() with cached landmark rows and print the readings (no camera)."""
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
                best = out
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
