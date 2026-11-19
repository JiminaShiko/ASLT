"""
Stage 6 - the live demo.

    python -m src.live                      webcam (q or ESC quits, r resets)
    python -m src.live --selftest           guided drill over the core words
    python -m src.live --selftest hello yes no
    python -m src.live --benchmark          CPU fps against the 15 fps target
    python -m src.live --render-test        a dataset clip through the live
                                            code path, no camera at all
    python -m src.live --logic-test         the decision rules, no camera and
                                            no trained weights

PRIVACY, enforced here rather than promised:
  * every frame is a local variable. This module contains no VideoWriter and
    no image-save call; --render-test proves it by snapshotting the folder
    tree (contents and mtimes) before and after a run that streams real uint8
    frames through update() and the drawing code.
  * --selftest prints its table to the terminal only. Keeping a record means
    copying it yourself.
  * the weights come from ASL Citizen alone; nothing here is fitted to webcam
    data.

WHAT THE WINDOW SHOWS: mirrored video, the skeleton overlay (upper body, both
hands, and exactly the face subset features.py reads), the state
(REST / SIGNING / UNCERTAIN), the top-3 words with probabilities, and the
sentence built from the committed words.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config, features, landmarks, phrases, predictor, vocab

# ---------------------------------------------------------------------------
# skeleton: MediaPipe's own indices, mapped into the packed layout by
# src/landmarks.py, so the overlay can never drift from what features reads
# ---------------------------------------------------------------------------
POSE_CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (11, 23), (12, 24), (23, 24),
             (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)]
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
             (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
             (0, 17)]
LIPS_OUTER = config.FACE_GROUPS["lips"][:20]
LIPS_INNER = config.FACE_GROUPS["lips"][20:]
JAW = config.FACE_GROUPS["chin_jaw"]
BROWS = (config.FACE_GROUPS["brows_forehead"][0:5],
         config.FACE_GROUPS["brows_forehead"][5:10])
NOSE = config.FACE_GROUPS["nose"]

# BGR, because the overlay is drawn on the cv2 image
GREEN = (90, 220, 90)
CYAN = (230, 200, 90)
YELLOW = (70, 215, 235)
VIOLET = (225, 130, 60)
ORANGE = (60, 140, 240)
GREY = (160, 160, 160)
RED = (70, 70, 235)
DARK = (30, 30, 30)
STATE_COLOUR = {"REST": GREY, "SIGNING": GREEN, "UNCERTAIN": RED}


def _xy(row: np.ndarray, r: int, w: int, h: int, mirror: bool):
    """Packed row -> pixel xy, or None when that landmark is missing."""
    x, y = float(row[r, 0]), float(row[r, 1])
    if np.isnan(x) or np.isnan(y):
        return None
    return (int((1.0 - x if mirror else x) * w), int(y * h))


def _draw(img, row, pairs, colour, mirror, dots=False, rows=()):
    """Draw line segments (and optional dots) for one landmark group."""
    want = set(rows) | {r for pr in pairs for r in pr}
    pos = {r: _xy(row, r, img.shape[1], img.shape[0], mirror) for r in want}
    for a, b in pairs:
        if pos.get(a) and pos.get(b):
            cv2.line(img, pos[a], pos[b], colour, 1, cv2.LINE_AA)
    if dots:
        for r in want:
            if pos.get(r):
                cv2.circle(img, pos[r], 2, colour, -1, cv2.LINE_AA)


def _ring(mesh_indices):
    """Face-mesh indices -> packed rows, plus the closed polyline between them."""
    rows = [landmarks.face_row(i) for i in mesh_indices]
    return rows, list(zip(rows, rows[1:])) + [(rows[-1], rows[0])]


def _chain(mesh_indices):
    rows = [landmarks.face_row(i) for i in mesh_indices]
    return rows, list(zip(rows, rows[1:]))


def overlay(rgb: np.ndarray, row: np.ndarray, mirror: bool = True) -> np.ndarray:
    """Draw the skeleton on the frame (BGR, ready for cv2.imshow).

    A frame with no landmarks at all draws nothing and cannot crash - every
    group is skipped while its points are NaN.
    """
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    pose_rows = [config.pose_row(i) for i in range(27)]
    _draw(img, row, [(config.pose_row(a), config.pose_row(b))
                     for a, b in POSE_CONN], GREEN, mirror, True, pose_rows)
    for hand, colour in (("left", CYAN), ("right", YELLOW)):
        rows = [landmarks.hand_row(hand, k) for k in range(config.N_HAND)]
        _draw(img, row, [(rows[a], rows[b]) for a, b in HAND_CONN],
              colour, mirror, True, rows)
    for grp, colour in ((LIPS_OUTER, VIOLET), (LIPS_INNER, VIOLET),
                        (JAW, ORANGE)):
        rows, pairs = _ring(grp) if grp is not JAW else _chain(grp)
        _draw(img, row, pairs, colour, mirror, True, rows)
    for grp in BROWS:
        rows, pairs = _chain(grp)
        _draw(img, row, pairs, GREY, mirror, True, rows)
    rows = [landmarks.face_row(i) for i in NOSE]
    _draw(img, row, [], GREY, mirror, True, rows)
    return img


def hud(img: np.ndarray, r: dict, fps: float, hint: str = "") -> np.ndarray:
    """State, what it heard, sentence and top-3. Only numbers and labels are
    drawn - never a pixel of the frame is kept."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), DARK, -1)
    cv2.putText(img, r["state"], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                STATE_COLOUR.get(r["state"], GREY), 2, cv2.LINE_AA)
    # the word it is prepared to stand behind, or the word "uncertain". A
    # near-tie is shown as uncertain rather than as its argmax: the viewer has
    # to be able to tell a refusal from a reading.
    disp = str(r.get("display") or "")
    if disp:
        cv2.putText(img, disp.upper()[:14], (130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    GREEN if disp != "uncertain" else RED, 1, cv2.LINE_AA)
    cv2.putText(img, f"{fps:4.1f} fps", (w - 96, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(img, "q quit   r reset   nothing is saved", (280, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    if r["text"]:
        cv2.rectangle(img, (0, 30), (w, 66), DARK, -1)
        cv2.putText(img, r["text"][:64], (8, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
    y = h - 108
    for word, p in list(r["top3"])[:3]:
        cv2.rectangle(img, (0, y - 18), (200, y + 6), DARK, -1)
        cv2.putText(img, f"{word:<10s} {p:5.2f}", (8, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    GREEN if p >= config.CONFIDENCE_MIN else RED, 1,
                    cv2.LINE_AA)
        y += 27
    if r.get("reason"):
        cv2.putText(img, r["reason"][:46], (8, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
    if hint:
        cv2.putText(img, hint[:60], (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def build_predictor(args, log=None) -> predictor.WordPredictor:
    return predictor.WordPredictor(
        weights=Path(args.weights) if args.weights else None,
        phrases_path=Path(args.phrases) if args.phrases else None,
        fps=args.fps, stride=args.stride, buffer_seconds=args.buffer,
        min_conf=None if args.conf < 0 else args.conf,
        min_margin=None if getattr(args, "margin", -1.0) < 0 else args.margin,
        log=log)


def open_camera(index: int, want_fps: float = 30.0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        raise SystemExit(
            f"camera {index} did not open. Close any other app using it and "
            "retry with --camera 1; Windows Settings > Privacy & security > "
            "Camera must allow desktop apps.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, want_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ---------------------------------------------------------------------------
# webcam
# ---------------------------------------------------------------------------
def live(args) -> int:
    log = config.logger_for("live", Path(args.log) if args.log else None)
    p = build_predictor(args, log)
    cap = open_camera(args.camera, args.fps)
    ema, last = 0.0, time.perf_counter()
    log.info("window open: %d classes, commit needs p1 >= %.2f and margin >= "
             "%.2f; q/ESC quit, r reset", len(p.labels), p.min_conf,
             p.min_margin)
    if config.HEDGE_PAIRS:
        log.info("hedged pairs (type neither, show the hint instead): %s",
                 ", ".join(str(v) for v in config.HEDGE_PAIRS.values()))
    hinted = False
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                log.warning("the camera stopped delivering frames")
                break
            rgb = landmarks.bgr_to_rgb(bgr)
            r = p.update(rgb)
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            ema = inst if ema == 0 else 0.9 * ema + 0.1 * inst
            buf = p.buffer.array()
            frame = overlay(rgb, buf[-1] if len(buf) else
                            np.full((config.P_TOTAL, 3), np.nan, np.float32),
                            mirror=not args.no_mirror)
            if bool(r["hint"]) != hinted:
                hinted = bool(r["hint"])
                log.info("framing %s: hand in %.0f%% of the last %.1f s",
                         "hint on" if hinted else "ok",
                         100 * (r["hand_rate"] or 0.0), p.buf_s)
            if args.headless:
                hud(frame, r, ema, r["hint"])
            else:
                cv2.imshow("ASL word recognizer - in memory only",
                           hud(frame, r, ema, r["hint"]))
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("r"):
                    p.reset()
            if r["committed_word"]:
                log.info("committed %s (p1 %.2f)", r["committed_word"], r["p1"])
    finally:
        cap.release()
        p.close()
        if not args.headless:
            cv2.destroyAllWindows()
    log.info("closed after %d frames; sentence was '%s'", p._n_in,
             p.phrases.text())
    return 0


# ---------------------------------------------------------------------------
# --selftest: guided drill, one terminal table, nothing written
# ---------------------------------------------------------------------------
def selftest(args) -> int:
    spec = vocab.load_vocab()
    active = [w.word for w in spec.active]
    words = [w.lower() for w in args.words] or active
    unknown = [w for w in words if w not in active]
    if unknown:
        raise SystemExit(f"not in the current vocabulary: {unknown}\n"
                         f"active words: {active}")
    p = build_predictor(args)
    cap = open_camera(args.camera, args.fps)
    rows: list[dict] = []
    print(f"\n{len(words)} words. For each: raise your hands, sign the word "
          f"once, then drop them and hold - the segment ends when the hands "
          f"rest. {args.seconds:.0f} s per word. Press q in the window to "
          "stop early.\n")
    quit_early = False
    try:
        for want in words:
            if quit_early:
                break
            print(f"--- sign: {want.upper()}", flush=True)
            deadline = time.time() + args.seconds
            cut, best = None, None
            seen: list[float] = []
            while time.time() < deadline:
                ok, bgr = cap.read()
                if not ok:
                    print("camera dropped out", flush=True)
                    deadline = 0.0
                    break
                rgb = landmarks.bgr_to_rgb(bgr)
                r = p.update(rgb)
                if r["top3"] and (best is None or r["p1"] > best["p1"]):
                    best = r
                if r["hand_rate"] is not None:
                    seen.append(r["hand_rate"])
                buf = p.buffer.array()
                frame = overlay(rgb, buf[-1] if len(buf) else
                                np.full((config.P_TOTAL, 3), np.nan,
                                        np.float32))
                cap_txt = f"sign: {want}"
                if r["hint"]:
                    cap_txt += f"  |  {r['hint']}"
                if args.headless:
                    hud(frame, r, 0.0, cap_txt)
                else:
                    cv2.imshow("ASL self-test - in memory only",
                               hud(frame, r, 0.0, cap_txt))
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        quit_early = True
                        deadline = 0.0
                if r["segment_cut"]:
                    cut = r
                    break
            hand = round(float(np.mean(seen)), 3) if seen else None
            notes = []
            if hand is not None and hand < config.FRAMING_HAND_RATE_MIN:
                notes.append(f"hands in only {100 * hand:.0f}% of frames, "
                             f"{config.FRAMING_HINT}")
            if (cut or best or {}).get("reason"):
                notes.append((cut or best)["reason"])
            if not cut:
                notes.append("no segment - the hands never rested")
            rows.append({
                "wanted": want,
                "segment": bool(cut),
                "committed": (cut or {}).get("committed_word", ""),
                "top1": (best or {}).get("top3", [["_", 0.0]])[0][0],
                "p1": round((best or {}).get("p1", 0.0), 3),
                "margin": round((best or {}).get("margin", 0.0), 3),
                # the framing check for this word: MediaPipe had a hand in this
                # fraction of the buffered frames while you signed it
                "hand": hand,
                "reason": "; ".join(notes),
            })
            p.reset()
    finally:
        cap.release()
        p.close()
        if not args.headless:
            cv2.destroyAllWindows()

    hdr = (f"{'word':<10s} {'seg':>3s} {'heard':<10s} {'top1':<10s} "
           f"{'p1':>5s} {'margin':>7s} {'hand':>5s}  note")
    print("\n" + hdr)
    print("-" * (len(hdr) + 8))
    n_ok = 0
    for r in rows:
        ok = r["committed"] == r["wanted"]
        n_ok += ok
        hr = "-" if r["hand"] is None else f"{r['hand']:.2f}"
        print(f"{r['wanted']:<10s} {'y' if r['segment'] else 'n':>3s} "
              f"{r['committed'] or '-':<10s} {r['top1'] or '-':<10s} "
              f"{r['p1']:5.2f} {r['margin']:7.2f} {hr:>5s}  "
              f"{'correct' if ok else r['reason'] or 'wrong word'}")
    print("-" * (len(hdr) + 8))
    print(f"{n_ok}/{len(rows)} signed correctly in this session. This is not a "
          "model score: one signer, one room, one take, and it can be worse "
          "than the trained number or better than it by luck. The trained "
          "numbers are in reports/words_summary.json.")
    low = [r["wanted"] for r in rows
           if r["hand"] is not None and r["hand"] < config.FRAMING_HAND_RATE_MIN]
    if low:
        print("the hand was barely in frame for: " + ", ".join(low) +
              f" - {config.FRAMING_HINT}. No gate can read a sign it cannot "
              "see, so those words are not a model result.")
    if args.json:
        print(json.dumps(rows, indent=1))
    return 0


# ---------------------------------------------------------------------------
# --benchmark: CPU frames per second on real dataset frames, nothing saved
# ---------------------------------------------------------------------------
def cached_videos(n: int) -> list[Path]:
    vids: list[Path] = []
    if config.MANIFEST_CSV.is_file():
        import pandas as pd
        man = pd.read_csv(config.MANIFEST_CSV)
        vids = [config.video_path(v) for v in man["video"].head(n)]
    vids = [v for v in vids if v.is_file()]
    if not vids:
        raise SystemExit("no dataset clip to measure. Run first: "
                         "python -m src.extract --splits train --limit 3")
    return vids


def benchmark(args) -> int:
    p = build_predictor(args)
    vids = cached_videos(args.videos)
    full, land, feat, inf = [], [], [], []
    frames = 0
    tracker = landmarks.Tracker()
    buf = features.RingBuffer(maxlen=400)
    for v in vids:
        p.reset()
        buf.clear()
        for idx, ts, rgb in landmarks.frames_of(v, max_frames=args.frames):
            if idx % args.stride:
                continue
            t0 = time.perf_counter()
            r = p.update(rgb)
            t1 = time.perf_counter()
            full.append((t1 - t0) * 1000.0)
            if frames >= 5:                       # ignore the warm-up frames
                # the same frame, measured in three pieces
                t2 = time.perf_counter()
                row = tracker.detect(rgb, ts)
                t3 = time.perf_counter()
                buf.append(row)
                f = features.sequence(buf.array()[-64:] if len(buf) >= 2
                                      else row[None],
                                      version=getattr(p, "fver", None))
                t4 = time.perf_counter()
                p.net.predict(f[None])
                t5 = time.perf_counter()
                land.append((t3 - t2) * 1000.0)
                feat.append((t4 - t3) * 1000.0)
                inf.append((t5 - t4) * 1000.0)
            frames += 1
    tracker.close()
    p.close()

    def med(xs):
        return round(float(np.median(xs)), 2) if xs else None

    total = med(full) or 0.0
    fps = 1000.0 / total if total else 0.0
    out = {
        "clips": len(vids), "frames": frames,
        "median_ms": {"full_update": total, "landmarks_only": med(land),
                      "features_only": med(feat), "model_only": med(inf),
                      "sum_of_parts": round(sum(x for x in
                                                [med(land), med(feat),
                                                 med(inf)] if x), 2)},
        "effective_fps": round(fps, 1),
        "target_fps": config.TARGET_LIVE_FPS,
        "meets_target": fps >= config.TARGET_LIVE_FPS,
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cpus": config.cpu_count(),
        "note": "full_update is landmarks + features + model + the "
                "segmenter, one call per frame; parts are the same frame "
                "measured separately and are not additive",
    }
    print(json.dumps(out, indent=1))
    return 0 if out["meets_target"] else 1


# ---------------------------------------------------------------------------
# --render-test: the live code path on dataset frames, no camera, no writes
# ---------------------------------------------------------------------------
def snapshot() -> dict[str, float]:
    """Every file this project may write, with its mtime. Never walks the
    dataset - that folder is read-only and enormous."""
    out = {}
    for d in ("src", "cache", "models", "reports", "logs"):
        root = config.ROOT / d
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                out[str(p)] = round(p.stat().st_mtime, 6)
    return out


def render_test(args) -> int:
    """Feeds real uint8 frames from the dataset through update() and overlay().

    Proves three things at once: camera-shaped input works, the drawing code
    survives a frame with no landmarks, and nothing is written to disk.
    """
    p = build_predictor(args)
    before = snapshot()
    states: dict[str, int] = {}
    drawn = empty_rows = 0
    try:
        for v in cached_videos(args.videos):
            p.reset()
            for idx, ts, rgb in landmarks.frames_of(v, max_frames=args.frames):
                if idx % args.stride:
                    continue
                r = p.update(rgb, ts_ms=ts)
                states[r["state"]] = states.get(r["state"], 0) + 1
                buf = p.buffer.array()
                row = buf[-1] if len(buf) else np.full((config.P_TOTAL, 3),
                                                       np.nan, np.float32)
                if np.isnan(row[:, :2]).any():
                    empty_rows += 1
                img = hud(overlay(rgb, row), r, 30.0)
                assert img.shape == rgb.shape and img.dtype == np.uint8
                drawn += 1
    finally:
        p.close()
    after = snapshot()
    touched = {k for k in after if k not in before} | {
        k for k in after if k in before and after[k] != before[k]}
    # this module may only ever write the log file the operator asked for
    allowed = {str(config.LOGS_DIR / "live.log"),
               str(config.LOGS_DIR / "render.log"),
               str(config.MODELS_DIR / "holistic_landmarker.task")}
    unexpected = sorted(x for x in touched if x not in allowed)
    print(json.dumps({"clips": args.videos, "frames_drawn": drawn,
                      "frames_with_missing_landmarks": empty_rows,
                      "states": states,
                      "files_touched": sorted(touched),
                      "unexpected_writes": unexpected,
                      "nothing_saved": not unexpected}, indent=1))
    if unexpected:
        print("FAIL: the live path wrote to disk", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# --logic-test: the decision rules, driven with made-up numbers
# ---------------------------------------------------------------------------
class _ScriptedNet:
    """A stand-in whose probabilities are supplied.

    Lets the whole live path (segmenter -> features -> decide -> phrase
    buffer -> commit) be tested without trained weights and without a camera.
    """

    def __init__(self, probs) -> None:
        self.probs = np.asarray(probs, np.float32)
        self.arch = type("Arch", (), {"in_dim": features.FEATURE_DIM})()

    def predict(self, x):
        a = np.asarray(x)
        n = 1 if a.ndim == 2 else a.shape[0]
        return np.tile(self.probs, (n, 1))


def fake_row(signing: bool, drop: float = 0.25) -> np.ndarray:
    """A landmark row with both hands above (signing) or below (rest) the
    shoulder line - the only geometry the segmenter looks at."""
    row = np.full((config.P_TOTAL, 3), np.nan, np.float32)
    sh_y = 0.40
    row[config.POSE_L_SHOULDER, :2] = (0.42, sh_y)
    row[config.POSE_R_SHOULDER, :2] = (0.58, sh_y)
    wrist_y = sh_y - drop if signing else sh_y + drop
    for r, x in ((config.POSE_L_WRIST, 0.45), (config.POSE_R_WRIST, 0.55)):
        row[r, :2] = (x, wrist_y)
    for hand in ("left", "right"):
        for k in range(config.N_HAND):
            row[landmarks.hand_row(hand, k), :2] = (0.45 + 0.004 * k, wrist_y)
    return row


def nohand_row(signing: bool = True) -> np.ndarray:
    """Pose wrists up, hand landmarks absent: the framing failure the hint is
    for. The segmenter opens (it reads the pose wrists), the model gets a
    buffer with no hand in it, and the demo has to say so."""
    row = fake_row(signing)
    for hand in ("left", "right"):
        for k in range(config.N_HAND):
            row[landmarks.hand_row(hand, k)] = np.nan
    return row


def logic_test(args) -> int:
    labels = vocab.load_vocab().labels
    K = len(labels)
    hi = labels.index("hello")
    gi = labels.index("good")
    rest_i = labels.index(config.REST_LABEL)
    other_i = labels.index(config.OTHER_LABEL)
    checks: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), str(detail)))

    def floor_row(p1, second=0.0):
        r = np.full(K, 0.0, np.float32)
        r[hi] = p1
        r[gi] = second
        return r

    # 1 - the confidence bar
    d = predictor.decide(floor_row(config.CONFIDENCE_MIN, 0.05), labels)
    check(f"p1 exactly at the floor ({config.CONFIDENCE_MIN:.2f}) is accepted",
          d.certain, d.reason)
    d = predictor.decide(floor_row(config.CONFIDENCE_MIN - 0.05, 0.01), labels)
    check("p1 below the floor is refused, with the reason",
          not d.certain and "confidence" in d.reason, d.reason)

    # 2 - the margin bar, and it names the runner-up. At the shipped 0.85 floor
    # a near-tie is already refused by confidence, so the margin is shown at a
    # lowered floor - which is exactly what --conf/--margin are for.
    d = predictor.decide(floor_row(0.70, 0.60), labels, min_conf=0.60)
    check("0.10 gap under the margin floor is refused against 'good'",
          not d.certain and "'good'" in d.reason, d.reason)
    check("the same mass split over three classes is refused",
          not predictor.decide(np.full(K, 1.0 / K, np.float32), labels).certain)
    d = predictor.decide(floor_row(0.85, 0.10), labels)
    check("confident and well separated is certain", d.certain, d.reason)
    check("top-3 is ordered and capped at three",
          len(d.top3) == 3 and d.top3[0][1] >= d.top3[1][1] >= d.top3[2][1],
          str(d.top3))

    # 2b - the shipped gate, and what the margin does next to it
    check(f"the demo commits at p1 >= {config.CONFIDENCE_MIN:.2f} by default",
          config.CONFIDENCE_MIN >= 0.85, str(config.CONFIDENCE_MIN))
    check("0.80 is refused at that default",
          not predictor.decide(floor_row(0.80, 0.05), labels).certain)
    # p1 - p2 >= 2*p1 - 1, so a 0.85 floor already implies a 0.70 margin: the
    # margin is the lever for a lowered --conf, never for the default gate
    check("a 0.85/0.14 reading clears any margin up to 0.70",
          predictor.decide(floor_row(0.85, 0.14), labels,
                           min_margin=0.70).certain)
    check("a raised --margin refuses it above 0.70",
          not predictor.decide(floor_row(0.85, 0.14), labels,
                               min_margin=0.90).certain)
    check("a lowered --margin accepts what the default refused at 0.70",
          predictor.decide(floor_row(0.70, 0.60), labels, min_conf=0.60,
                           min_margin=0.05).certain)

    # 3 - per-pair overrides, in either order
    pm = {"hello|good": 0.30}
    check("a pair override is symmetric and only for that pair",
          predictor.required_margin("good", "hello", pm) == 0.30
          and predictor.required_margin("hello", "good", pm) == 0.30
          and predictor.required_margin("hello", "sad", pm)
          == config.MARGIN_MIN)
    check("the override actually refuses a 0.20 gap (at a lowered floor, so "
          "it is the margin refusing)",
          not predictor.decide(floor_row(0.70, 0.50), labels, pm,
                               min_conf=0.60).certain
          and predictor.decide(floor_row(0.70, 0.50), labels,
                               min_conf=0.60).certain)
    check("the tuple key shape means the same thing as the string one",
          predictor.required_margin("good", "hello",
                                    {("hello", "good"): 0.30}) == 0.30)
    for bad_key in [("hello",), "hello", "hello|hello",
                    ("hello", "good", "sad"), ("hello", "HELLO")]:
        try:
            predictor.required_margin("hello", "good", {bad_key: 0.3})
            detail = f"{bad_key!r} was accepted"
            raised = False
        except ValueError as exc:
            detail, raised = str(exc), True
        check(f"a malformed pair key {bad_key!r} fails loud", raised, detail)

    # 3b - the live hedge: a hedged pair's top-2 types neither word, at any
    # confidence, in either order; everything else is untouched
    def pair_row(w1, p1, w2, p2=0.0):
        r = np.full(K, 0.0, np.float32)
        r[labels.index(w1)] = p1
        r[labels.index(w2)] = p2
        return r

    hedge = pair_row("good", 0.92, "thanks", 0.05)
    check("confident good with thanks second is hedged, not typed",
          not predictor.decide(hedge, labels).certain
          and "good/thanks?" in predictor.decide(hedge, labels).reason,
          predictor.decide(hedge, labels).reason)
    d = predictor.decide(pair_row("thanks", 0.55, "good", 0.40), labels,
                         min_conf=0.50)
    check("the hedge fires in either order and even at a lowered floor",
          not d.certain and "good/thanks?" in d.reason, d.reason)
    d = predictor.decide(hedge, labels, hedge_pairs={})
    check("hedge_pairs={} turns the hedge off for callers that opt out",
          d.certain, d.reason)
    check("a hedged reading is never commitable",
          not predictor.commitable(predictor.decide(hedge, labels)))
    check("good with a non-thanks runner-up is still certain",
          predictor.decide(pair_row("good", 0.92, "hello", 0.05),
                           labels).certain)
    check("a hedged pair below the confidence floor refuses for confidence, "
          "not for the hedge",
          "confidence" in predictor.decide(pair_row("good", 0.50, "thanks",
                                                    0.45), labels).reason)
    for bad_key in [("good",), "good", "good|good"]:
        try:
            predictor.hedge_hint("good", "thanks", {bad_key: "x?"})
            raised = False
        except ValueError:
            raised = True
        check(f"a malformed hedge key {bad_key!r} fails loud", raised)

    # 4 - background classes can never be a commit
    rb = np.full(K, 0.0, np.float32)
    rb[rest_i], rb[hi] = 0.95, 0.03
    check("REST as top-1 is refused as a commit",
          not predictor.commitable(predictor.decide(rb, labels)))
    ro = np.full(K, 0.0, np.float32)
    ro[other_i] = 0.95
    check("OTHER as top-1 is refused as a commit",
          not predictor.commitable(predictor.decide(ro, labels)))
    check("a refused reading is never commitable",
          not predictor.commitable(predictor.decide(
              floor_row(0.30, 0.20), labels)))

    # 5 - the segmenter, frame by frame
    s = predictor.Segmenter()
    seq = [False, True, False, False, True, True, True, False, True,
           False, False, False, False, False, False, True]
    trace = [(s.update(f), s.frames) for f in seq]
    steps = [t[0] for t in trace]
    check("a single signing frame does not open a segment",
          set(steps[:3]) == {"REST"}, str(steps[:3]))
    check("three signing frames open it", steps[6] == "SIGNING", str(steps))
    check("a stray rest frame inside the word does not close it",
          steps[7] == "SIGNING" and steps[8] == "SIGNING", str(steps[7:9]))
    check("five rest frames close it, and END is reported once",
          steps[13] == "END" and steps.count("END") == 1, str(steps))
    check("the segment counter restarts at the cut", trace[13][1] == 0,
          str(trace[10:]))
    s2 = predictor.Segmenter()
    open2 = [s2.update(True) for _ in range(6)]
    rest4 = [s2.update(False) for _ in range(4)]
    check("four rest frames are not enough to close", "END" not in rest4,
          str(open2 + rest4))
    s3 = predictor.Segmenter()
    eight = [s3.update(True) for _ in range(8)]
    check("the first SIGNING step lands on the third frame, then holds",
          eight[:2] == ["REST", "REST"]
          and set(eight[2:]) == {"SIGNING"} and eight[2] == "SIGNING",
          str(eight))

    # 6 - the phrase buffer
    b = phrases.PhraseBuilder()
    for w in ("hello", "REST", "OTHER", "good"):
        b.commit(w)
    check("REST/OTHER are neither typed nor trimmed",
          b.text() == "hello good" and b.words == ["hello", "good"], b.text())
    b = phrases.PhraseBuilder()
    b.commit("my")
    b.commit("name")
    check("a two-word rule renders and is reported as the active rule",
          b.text() == "my name" and b.last_rule is not None
          and b.last_rule.pattern == ("my", "name"), b.text())
    b2 = phrases.PhraseBuilder()
    for w in ["good", "friend", "please", "help", "sad", "school", "home",
              "yes", "no"]:
        b2.commit(w)
    check(f"the word buffer caps at {b2.book.max_buffer_words}",
          len(b2.words) == b2.book.max_buffer_words
          and b2.words[0] == "friend", str(b2.words))
    check("ME is written as I", phrases.sentence(["me", "sad"]) == "I sad",
          phrases.sentence(["me", "sad"]))

    # 7 - the whole live path, with no weights and no camera
    def run(probs, flags):
        p = predictor.WordPredictor(net=_ScriptedNet(probs),
                                    meta={"labels": labels}, fps=30.0,
                                    log=config.logger_for("logic"))
        out = [p.update(fake_row(f)) for f in flags]
        p.close()
        return out

    hl = np.full(K, 0.0, np.float32)
    hl[hi] = 0.92
    hl[other_i] = 0.08
    rise_rest = [False] * 3 + [True] * 12 + [False] * 8
    got = run(hl, rise_rest)
    cuts = [g for g in got if g["segment_cut"]]
    check("a scripted rise-and-rest commits exactly one word",
          len(cuts) == 1 and cuts[0]["committed_word"] == "hello",
          [(g["state"], g["committed_word"]) for g in got])
    check("the committed word is the only thing in the sentence",
          cuts[0]["text"] == "hello", cuts[0]["text"])
    check("the states walk REST -> SIGNING -> REST",
          got[0]["state"] == "REST" and got[6]["state"] == "SIGNING"
          and cuts[0]["state"] == "REST",
          str([g["state"] for g in got]))
    check("the decision uses the signing frames of the segment, not the "
          "rest frames that closed it",
          cuts[0]["decision_frames"] == 12,
          str(cuts[0]["decision_frames"]))
    got2 = run(ro, rise_rest)          # OTHER on top
    cuts2 = [g for g in got2 if g["segment_cut"]]
    check("an OTHER-only stream commits nothing and types nothing",
          len(cuts2) == 1 and cuts2[0]["committed_word"] == ""
          and cuts2[0]["text"] == "", str(cuts2))
    weak = run(np.full(K, 1.0 / K, np.float32), rise_rest)
    cuts3 = [g for g in weak if g["segment_cut"]]
    check("an unsure stream commits nothing but still says why",
          cuts3[0]["committed_word"] == ""
          and "confidence" in cuts3[0]["reason"], str(cuts3[0]))
    two = run(hl, rise_rest + [False] * 4 + [True] * 12 + [False] * 8)
    cut2 = [g for g in two if g["segment_cut"]]
    check("two segments commit twice and the sentence grows",
          [g["committed_word"] for g in cut2] == ["hello", "hello"]
          and cut2[-1]["text"] == "hello hello",
          str([g["text"] for g in cut2]))
    hedge_run = run(pair_row("good", 0.92, "thanks", 0.05), rise_rest)
    cuts4 = [g for g in hedge_run if g["segment_cut"]]
    check("a good/thanks stream types nothing and says why",
          len(cuts4) == 1 and cuts4[0]["committed_word"] == ""
          and cuts4[0]["text"] == ""
          and "good/thanks?" in cuts4[0]["reason"], str(cuts4[0]))

    # 7b - what the screen says, and the framing hint
    def run_rows(rows, probs=None):
        pt = predictor.WordPredictor(
            net=_ScriptedNet(hl if probs is None else probs),
            meta={"labels": labels}, fps=30.0,
            log=config.logger_for("logic"))
        out = [pt.update(rw) for rw in rows]
        pt.close()
        return out

    vis = run_rows([fake_row(True)] * 40)
    check("a hand in every buffered frame means no hint and rate 1.00",
          vis[-1]["hint"] == "" and vis[-1]["hand_rate"] == 1.0,
          str((vis[-1]["hand_rate"], vis[-1]["hint"])))
    blind = run_rows([nohand_row()] * 40)
    check("with no hand in the buffer the hint names the framing, not a word",
          blind[-1]["hand_rate"] == 0.0
          and blind[-1]["hint"] == config.FRAMING_HINT,
          str((blind[-1]["hand_rate"], blind[-1]["hint"])))
    check("the hint waits for FRAMING_MIN_FRAMES before it speaks",
          run_rows([nohand_row()]
                   * (config.FRAMING_MIN_FRAMES - 1))[-1]["hand_rate"] is None)
    arr40 = np.stack([nohand_row()] * 20 + [fake_row(True)] * 20)
    check("hand_share() and hand_detection_rate() are the same measurement",
          abs(landmarks.hand_share(arr40)
              - landmarks.hand_detection_rate(arr40)) < 1e-9,
          f"{landmarks.hand_share(arr40):.3f} vs "
          f"{landmarks.hand_detection_rate(arr40):.3f}")
    check("a certain reading is displayed as its word",
          vis[-1]["display"] == "hello", vis[-1]["display"])
    check("an unsure reading is displayed as uncertain, never as its argmax",
          run_rows([fake_row(True)] * 40, np.full(K, 1.0 / K,
                                                  np.float32))[-1]["display"]
          == "uncertain")

    # weights that expect a vector no feature version builds must refuse,
    # and the v2 fair-comparison loser (170 dims) must stay loadable
    wrong = _ScriptedNet(hl)
    wrong.arch = type("Arch", (), {"in_dim": features.FEATURE_DIM + 3})()
    pstale = predictor.WordPredictor(net=wrong, meta={"labels": labels},
                                     fps=30.0, log=config.logger_for("logic"))
    res_last = None
    for rw in [fake_row(False)] * 3 + [fake_row(True)] * 20:
        res_last = pstale.update(rw)
    check("weights that expect an unknown vector refuse to read it",
          res_last["display"] == "uncertain" and "retrain" in res_last["reason"],
          res_last["reason"])
    pstale.close()
    v2net = _ScriptedNet(hl)
    v2net.arch = type("Arch", (), {"in_dim": features.feature_dim(2)})()
    pv2 = predictor.WordPredictor(net=v2net, meta={"labels": labels},
                                  fps=30.0, log=config.logger_for("logic"))
    res_v2 = None
    for rw in [fake_row(False)] * 3 + [fake_row(True)] * 20:
        res_v2 = pv2.update(rw)
    check("v2 weights (170 features) load and read, version inferred",
          res_v2["display"] == "hello" and pv2.fver == 2 and not pv2.stale,
          f"fver={pv2.fver} reason={res_v2['reason']}")
    pv2.close()

    # 8 - the API contract
    keys = {"state", "top3", "committed_word", "text", "p1", "margin",
            "reason", "buffer_frames", "decision_frames", "segment_frames",
            "hand_in_frame", "segment_cut", "display", "hand_rate", "hint"}
    check("update() returns exactly the documented keys",
          set(got[-1]) == keys, sorted(set(got[-1]) ^ keys))
    check("top3 is three (label, prob) pairs",
          len(got[-1]["top3"]) == 3
          and all(isinstance(w, str) and isinstance(q, float)
                  for w, q in got[-1]["top3"]))
    check("the same reading is reported while a segment is open",
          got[6]["top3"][0][0] == "hello", str(got[6]["top3"]))
    check("hand_in_frame is the two-hand pair the overlay draws",
          tuple(got[6]["hand_in_frame"]) == (True, True),
          str(got[6]["hand_in_frame"]))
    px = predictor.WordPredictor(net=_ScriptedNet(hl), meta={"labels": labels},
                                 fps=30.0, log=config.logger_for("logic"))
    try:
        detail = "no error raised"
        try:
            px.update(np.zeros((5, 5), np.uint8))
            raised = False
        except ValueError as exc:
            raised, detail = True, str(exc)
    finally:
        px.close()
    check("a wrong-shaped input raises ValueError, not a silent skip",
          raised, detail)
    stride_runs = [predictor.reading_due(i, 1) for i in range(1, 5)]
    check("stride 1 reads every frame, stride 2 every second",
          all(stride_runs) and [predictor.reading_due(i, 2)
                                for i in range(1, 5)] == [False, True, False,
                                                          True])

    width = max(len(n) for n, _, _ in checks) + 2
    ok = 0
    for name, passed, detail in checks:
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}s} "
              f"{'' if passed else detail}")
    print(f"\n{ok}/{len(checks)} live-path logic checks passed")
    return 0 if ok == len(checks) else 1


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live word recognizer.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=config.LIVE_STRIDE,
                    help="run the model on every Nth camera frame")
    ap.add_argument("--buffer", type=float, default=None,
                    help=f"rolling seconds (default {config.BUFFER_SECONDS})")
    ap.add_argument("--conf", type=float, default=-1.0,
                    help="top-1 floor to commit (default config.CONFIDENCE_MIN"
                         " = 0.85)")
    ap.add_argument("--margin", type=float, default=-1.0,
                    help="top-1 minus top-2 floor to commit (default "
                         "config.MARGIN_MIN = 0.15). By arithmetic it can "
                         "only bite when --conf is below 0.58")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--phrases", default=None)
    ap.add_argument("--no-mirror", action="store_true",
                    help="show the frame un-mirrored (the overlay follows)")
    ap.add_argument("--headless", action="store_true",
                    help="no window; used by the automated checks")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="--selftest: time allowed per word")
    ap.add_argument("--frames", type=int, default=60,
                    help="--benchmark/--render-test: frames per clip")
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--json", action="store_true", help="--selftest: dump rows")
    ap.add_argument("--log", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--render-test", action="store_true")
    g.add_argument("--logic-test", action="store_true")
    g.add_argument("--demo", action="store_true",
                   help="the API demo on cached landmark rows (no camera)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a)
    if a.benchmark:
        return benchmark(a)
    if a.render_test:
        return render_test(a)
    if a.logic_test:
        return logic_test(a)
    if a.demo:
        predictor.demo(a.videos, Path(a.weights) if a.weights else None)
        return 0
    return live(a)


if __name__ == "__main__":
    sys.exit(main())
