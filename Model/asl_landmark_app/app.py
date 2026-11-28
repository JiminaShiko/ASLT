"""Streamlit frontend for the ASL landmark recogniser (photo / snapshot / live webcam).

Reuses live.Smoother, live.MotionDecider and live.build_backend so this page and the
console app cannot disagree. PRIVACY: nothing is ever written to disk.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_PROPAGATE_CXX_STD", "1")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import av                      # noqa: E402
import cv2                     # noqa: E402
import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402
import streamlit as st         # noqa: E402
from PIL import Image          # noqa: E402
from streamlit_webrtc import (  # noqa: E402
    RTCConfiguration, VideoProcessorBase, webrtc_streamer,
)

import config                  # noqa: E402
import hand_backend as hb      # noqa: E402
import normalize               # noqa: E402
import live                    # noqa: E402  (Smoother, MotionDecider, build_backend)

MIN_CONFIDENCE = 0.70      # frames below this cast no vote
UNCERTAIN_MARGIN = 0.10    # top1-top2 gap below this -> UNCERTAIN, no vote
VOTE_FRAMES = 7            # majority vote window
HOLD_SECONDS = 0.5         # how long a letter must be held before it is typed
MAX_SIDE = 640             # frames wider/taller than this are shrunk first
STRIDE_DEFAULT = 2         # a processed frame costs ~50 ms here; 2 fits 24 fps.
                           # Halves the J/Z rule's temporal resolution, so drop
                           # to 1 if J or Z stops firing.
MEANINGS = {
    "J": "J - moving letter: the stroke head plus real pinky travel, never one frame",
    "Z": "Z - moving letter: drawn by the fingertip zig-zag rule, not by the MLP",
}

# Timestamps are measured against the cached `_ts_clock()`, never a module global:
# MediaPipe rejects any timestamp not strictly greater than its last, and a rerun
# that re-created the epoch would rewind the clock and kill the camera.

# ICE servers default to empty: an unanswered STUN query stalls ICE gathering past
# streamlit_webrtc's 10 s offer timeout, which closes the camera. The library only
# fills in Google's server when the field is absent (None), so browser and aiortc
# must both be told - hence rtc_pair().
GOOGLE_STUN = "stun:stun.l.google.com:19302"
RTC_CONFIGURATION = RTCConfiguration({"iceServers": []})
FRONTEND_RTC_CONFIGURATION: dict = {"iceServers": []}
RTC_CONFIGURATION_STUN = RTCConfiguration(
    {"iceServers": [{"urls": [GOOGLE_STUN]}]}
)
FRONTEND_RTC_CONFIGURATION_STUN: dict = {"iceServers": [{"urls": [GOOGLE_STUN]}]}


def rtc_pair(use_stun: bool):
    """(server, browser) ICE configs; they must be swapped together or the stream never starts."""
    if use_stun:
        return RTC_CONFIGURATION_STUN, FRONTEND_RTC_CONFIGURATION_STUN
    return RTC_CONFIGURATION, FRONTEND_RTC_CONFIGURATION


# Ask for 24 fps so the browser does not send frames we would only drop.
VIDEO_CONSTRAINTS = {
    "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
              "frameRate": {"ideal": 24}},
    "audio": False,
}

st.set_page_config(page_title="ASL Landmark Recogniser", page_icon="🤟",
                   layout="centered")


# Cached once per server, not per rerun.
@st.cache_resource
def get_backend(name: str, prof: str):
    """(backend, labels) exactly as src/live.py builds them for the webcam."""
    return live.build_backend(name, profile=prof)


@st.cache_resource
def get_detector(static: bool):
    return hb.create_detector(static=static)


@st.cache_resource
def _ts_clock() -> float:
    """The one epoch for every VIDEO timestamp; cached for the detector's lifetime."""
    return time.monotonic()


def trained_profiles() -> list[str]:
    """Profiles whose weights exist on disk."""
    return [p for p in sorted(config.PROFILES) if config.model_path(p).is_file()]


def require_model(prof: str):
    """Stop with a clear message if the pipeline has not been trained yet."""
    if not (config.model_path(prof).is_file() and config.meta_path(prof).is_file()):
        st.error(f"No trained model for profile `{prof}`. Run this first:\n\n"
                 f"`python src/train.py --profile {prof}`"
                 f"`python src/export.py --profile {prof}`")
        st.stop()


def onnx_staleness(prof: str) -> str | None:
    """A warning when the exported ONNX is older than the .pt it mirrors."""
    on, pt = config.onnx_path(prof), config.model_path(prof)
    if not (on.is_file() and pt.is_file()) or on.stat().st_mtime + 1.0 >= pt.stat().st_mtime:
        return None
    return (f"`{on.name}` is from "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(on.stat().st_mtime))}"
            f" but `{pt.name}` is from "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(pt.stat().st_mtime))}"
            f" - the exported copy is an older model. Refresh it with "
            f"`python src/export.py --profile {prof}`.")


def fit_square(points, size=240, pad=0.12):
    """Map normalized (x, y) points into a square image, aspect preserved."""
    p = np.asarray(points, dtype=np.float64)[:, :2]
    s = max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min(), 1e-6)
    cx = 0.5 * (p[:, 0].max() + p[:, 0].min())
    cy = 0.5 * (p[:, 1].max() + p[:, 1].min())
    k = (1.0 - 2.0 * pad) / s
    q = np.stack([(p[:, 0] - cx) * k + 0.5, (p[:, 1] - cy) * k + 0.5], axis=1)
    return np.clip(q, 0.02, 0.98), size


def feature_view(lm: np.ndarray) -> np.ndarray:
    """What the MLP actually sees: the canonicalised, mirrored skeleton."""
    prepared = normalize.prepare(lm)
    q, size = fit_square(prepared[:, :2])
    canvas = np.zeros((size, size, 3), np.uint8)
    hb.draw_landmarks(canvas, [(float(a), float(b)) for a, b in q])
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def annotated(bgr: np.ndarray, lm) -> np.ndarray:
    out = bgr.copy()
    if lm is not None:
        hb.draw_landmarks(out, lm[:, :2])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def downscale(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= MAX_SIDE:
        return bgr
    k = MAX_SIDE / longest
    return cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def analyse_rgb(rgb: np.ndarray, backend_name: str, conf_min: float,
                prof: str) -> dict:
    """Colour array -> prediction dict. Nothing is stored anywhere."""
    backend, labels = get_backend(backend_name, prof)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hand = hb.detect_image(get_detector(True), downscale(bgr))
    out = {"hand": hand.landmarks is not None, "letter": None, "conf": 0.0,
           "probs": None, "labels": labels, "lm": hand.landmarks,
           "handedness": hand.handedness, "mirrored": None, "family": None,
           "feats": None}
    if hand.landmarks is None:
        return out
    out["mirrored"] = bool(normalize.should_mirror(hand.landmarks))
    out["family"] = normalize.chirality_family(hand.landmarks)
    feats = normalize.features(hand.landmarks)
    probs = backend(feats)
    out["feats"] = feats
    out["probs"] = probs
    call = live.frame_call(probs, labels, conf_min, UNCERTAIN_MARGIN)
    top = int(probs.argmax())
    out["letter"] = labels[top]
    out["conf"] = float(probs[top])
    out["confident"] = call["letter"] is not None
    out["uncertain"] = call["uncertain"]
    out["low_conf"] = call["low_conf"]
    out["margin"] = call["margin"]
    out["need"] = call["need"]
    out["collide"] = call["pair"] if call["collide"] else ""
    out["top3"] = call["top3"]
    out["conf_min"] = conf_min
    return out


def show_prediction(res: dict, key: str, show_feats: bool):
    """The result block for the photo / snapshot tabs."""
    if not res["hand"]:
        st.warning("MediaPipe found no hand in this image, so there is no "
                   "prediction. Use good light and keep the whole hand in view.")
        return
    label, conf = res["letter"], res["conf"]
    conf_min = res.get("conf_min", MIN_CONFIDENCE)
    if res.get("uncertain"):
        t3 = res.get("top3") or [(label, conf)]
        second = t3[1][0] if len(t3) > 1 else "-"
        pair = (f" - {res['collide'][0]}/{res['collide'][1]} is a pair this "
                f"model is measured to get wrong, so it needs "
                f"{res['need'] * 100:.0f} points there"
                if res.get("collide") else "")
        st.warning(f"**UNCERTAIN** - best is **{label}** at {conf * 100:.1f} % "
                   f"but only {res['margin'] * 100:.1f} points ahead of "
                   f"**{second}**{pair}, so the live app would cast no vote for "
                   "this frame.")
    elif res.get("confident", True):
        st.success(f"Predicted sign: **{label}**  ({conf * 100:.1f} %)")
    else:
        st.info(f"Best guess **{label}** at {conf * 100:.1f} %, under the "
                f"{conf_min * 100:.0f} % confidence gate - the live app would "
                "cast no vote for this frame.")
    if label in MEANINGS and res.get("confident", True):
        st.info(MEANINGS[label])

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**What the model sees** (canonicalised skeleton)")
        st.image(feature_view(res["lm"]))
        st.caption(
            f"Chirality family the model used: **{res.get('family') or 'unknown'}** "
            f"({'mirrored to the canonical shape first' if res['mirrored'] else 'taken as-is'}).")
        st.caption(
            "This 2-D test is the only handedness decision the pipeline makes; "
            "features never read a label. It cannot tell a right hand seen "
            "palm-first from a left hand seen back-of-hand-first - those "
            "project identically - so it is reported as a shape family, not as "
            "anatomy. "
            f"MediaPipe's own label for this hand: {res['handedness'] or 'unknown'} "
            "(audit only - it assumes a mirrored selfie frame, so on unmirrored "
            "frames it splits hands the way the geometry does on 87-98 % of "
            "this corpus while naming each group the opposite word).")
    with right:
        st.markdown("**Top 3**")
        for name, pr in res.get("top3") or []:
            st.progress(min(1.0, float(pr)), text=f"{name}: {pr * 100:.1f} %")
        with st.expander("Full distribution (top 5)"):
            order = np.argsort(res["probs"])[::-1][:5]
            for i in order:
                st.write(f"{res['labels'][i]}: {res['probs'][i] * 100:.1f} %")

    if show_feats:
        with st.expander(f"The {len(res['feats'])} feature values "
                         f"({' + '.join(config.FEATURE_BLOCKS)})"):
            st.dataframe(pd.DataFrame({
                "block": ["xy"] * 42 + ["tip_dists"] * 10 + ["curl"] * 5,
                "value": np.round(res["feats"], 4)}), height=320)


class LandmarkProcessor(VideoProcessorBase):
    """One instance per START; recv() runs on the webrtc thread only."""

    def __init__(self):
        self.backend = None
        self.labels = []
        self.detector = None
        self.lock = threading.Lock()
        self.smoother = live.Smoother(conf=MIN_CONFIDENCE, vote=VOTE_FRAMES,
                                      hold=HOLD_SECONDS)
        self.motion = live.MotionDecider()
        self.conf_min = MIN_CONFIDENCE
        self.margin = UNCERTAIN_MARGIN
        self.pairs = dict(live.PAIR_MARGINS)
        self.mirror = True
        self.overlay = True
        self.motion_on = True
        self.stride = int(STRIDE_DEFAULT)
        self.force_next = False
        self.text = ""
        self.held = None
        self.held_conf = 0.0
        self.raw_letter = None
        self.raw_conf = 0.0
        self.top_names = []
        self.top_probs = []
        self.uncertain = False
        self.low_conf = False
        self.margin_now = 0.0
        self.need_now = 0.0     # the gap this frame had to clear
        self.collide = ""       # pair key when PAIR_MARGINS was binding
        self.progress = 0.0
        self.lm = None
        self.motion_debug = {}
        self.last_motion = None
        self.motion_hint = ""      # why J/Z did not fire
        self.cal = None            # a live.JZCalibrator while running
        self.cal_result = None
        self.commits = []
        self.n_frames = 0
        self.n_hands = 0
        self.fps = 0.0
        self.ms = 0.0
        self._last_tick = None
        self._t0 = time.monotonic()
        self.n_frame_errors = 0
        self.consec_errors = 0
        self.frame_error = ""
        self._last_ts: int | None = None   # strictly increasing per processor
        self.det_private = None            # a healed detector, see _heal
        self._healed_at = 0.0

    def recv(self, frame):
        """No exception may escape: anything raised here closes the camera."""
        try:
            out = self._recv(frame)
            self.consec_errors = 0
            return out
        except Exception as exc:                                  # noqa: BLE001
            self.n_frame_errors += 1
            self.consec_errors += 1
            self.frame_error = f"{type(exc).__name__}: {str(exc)[:150]}"
            now = time.monotonic()
            if self.consec_errors >= 30 and now - self._healed_at > 60.0:
                self._heal(now)
            return frame

    def _heal(self, now: float) -> None:
        """Swap in a fresh landmarker after ~1 s of consecutive frame failures."""
        self._healed_at = now
        self.consec_errors = 0
        if not config.HAND_MODEL_PATH.is_file():
            # a frame handler must never trigger a model download
            return
        old, self.det_private = (self.det_private,
                                 hb.create_detector(static=False,
                                                    model_path=config.HAND_MODEL_PATH))
        if old is not None:
            try:
                old.close()
            except Exception:                                     # noqa: BLE001
                pass

    def _recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        started = time.perf_counter()
        self.n_frames += 1

        if self.stride > 1 and self.n_frames % self.stride != 1:
            return self._emit(img, started)

        detector = self.det_private or self.detector
        if detector is None or self.backend is None:
            return self._emit(img, started)

        ts_ms = int((time.monotonic() - _ts_clock()) * 1000)
        if self._last_ts is not None and ts_ms <= self._last_ts:
            # MediaPipe rejects a repeat as hard as a rewind
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        res = hb.detect_video(detector, downscale(img), ts_ms)
        now = time.monotonic()

        with self.lock:
            self.lm = res.landmarks
            if res.landmarks is None:
                self.raw_letter, self.raw_conf = None, 0.0
                self.top_names, self.top_probs = [], []
                self.uncertain = self.low_conf = False
                self.margin_now = 0.0
                self.need_now, self.collide = 0.0, ""
                call_letter = None
                if self.motion_on:
                    self.motion.reset()
                    self.motion_hint = ""
            else:
                self.n_hands += 1
                feats = normalize.features(res.landmarks)
                probs = self.backend(feats)
                call = live.frame_call(probs, self.labels, self.conf_min,
                                       self.margin, pairs=self.pairs)
                self.raw_letter = call["top3"][0][0]
                self.raw_conf = call["conf"]
                self.margin_now = call["margin"]
                self.need_now = call["need"]
                self.collide = call["pair"] if call["collide"] else ""
                self.uncertain, self.low_conf = call["close"], call["low_conf"]
                self.top_names = [l for l, _p in call["top3"]]
                self.top_probs = [p for _l, p in call["top3"]]
                call_letter = call["letter"]
                if self.motion_on:
                    cal = self.cal
                    # raw top-1, not the smoothed letter: the J/Z precondition is
                    # about this frame, and the smoother lags by construction
                    shown, shown_p = self.raw_letter, self.raw_conf
                    if cal is not None and not cal.done:
                        fired = cal.step(now, res.landmarks, shown, shown_p)
                        if cal.done:
                            self.cal_result = cal.result
                            self.cal = None
                    else:
                        fired = self.motion.update(now, res.landmarks, shown,
                                                   shown_p)
                    self.motion_debug = dict(self.motion.debug)
                    self.motion_hint = self.motion.hint()
                    if fired:
                        self.text += fired
                        self.last_motion = fired
                        self.commits.append((fired, now))
                if self.motion_on and self.motion.blocks_static(self.raw_letter, now):
                    call_letter = None
            force, self.force_next = self.force_next, False
            held, committed, held_conf = self.smoother.step(
                call_letter, self.raw_conf, now, force=force,
                hand_present=res.landmarks is not None)
            if committed:
                self.text += committed
                self.commits.append((committed, now))
            self.held, self.held_conf = held, held_conf or self.raw_conf
            self.progress = self.smoother.hold_progress(now)
            self.ms = 1000.0 * (time.perf_counter() - started)
            el = now - self._t0
            self.fps = self.n_frames / el if el > 0 else 0.0
        return self._emit(img, started)

    def _emit(self, img, started):
        with self.lock:
            lm = self.lm
            held, conf = self.held, self.held_conf
            progress = self.progress
            fps = self.fps
            top3 = list(zip(self.top_names, self.top_probs))
            uncertain, low_conf = self.uncertain, self.low_conf
            margin = self.margin_now
            need, collide = self.need_now, self.collide
        disp = img.copy()
        if self.overlay and lm is not None:
            hb.draw_landmarks(disp, lm[:, :2])
        if self.mirror:
            disp = cv2.flip(disp, 1)          # flip AFTER drawing, or the text mirrors
        h, w = disp.shape[:2]
        colour = (0, 200, 0) if held and conf >= self.conf_min else (0, 165, 255)
        if held:
            tag = f"{held}  {conf * 100:.0f}%"
        elif lm is None:
            tag = "no hand"
        elif uncertain:
            tag = f"UNCERTAIN  {conf * 100:.0f}%"
            colour = (0, 200, 255)
        else:
            tag = f"holding ... {progress * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.rectangle(disp, (12, 12), (12 + tw + 26, 12 + th + 26), (0, 0, 0), -1)
        cv2.rectangle(disp, (12, 12), (12 + tw + 26, 12 + th + 26), colour, 2)
        cv2.putText(disp, tag, (25, 12 + th + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, colour, 2, cv2.LINE_AA)

        if top3:
            bx = 24
            by = 12 + th + 40
            for i, (lab, p) in enumerate(top3):
                y = by + i * 24
                cv2.putText(disp, lab, (bx, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(disp, (bx + 26, y + 3),
                              (bx + 26 + int(110 * min(1.0, p)), y + 17),
                              (60, 200, 90) if i == 0 else (90, 90, 90), -1)
                cv2.rectangle(disp, (bx + 26, y + 3), (bx + 26 + 110, y + 17),
                              (200, 200, 200), 1)
                cv2.putText(disp, f"{p * 100:.0f}%", (bx + 142, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                            cv2.LINE_AA)
            if uncertain or low_conf:
                why = (f"only {margin * 100:.0f} pts ahead of "
                       f"{top3[1][0]}" if uncertain else
                       f"under the {self.conf_min:.2f} gate")
                if collide and uncertain:
                    why += (f" - {collide[0]}/{collide[1]} needs "
                            f"{need * 100:.0f}")
                cv2.putText(disp, f"no vote: {why}", (bx, by + 3 * 24 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
                            cv2.LINE_AA)

        cal = self.cal
        if self.motion_on and self.motion_hint:
            cv2.putText(disp, self.motion_hint[:80], (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1,
                        cv2.LINE_AA)
        if cal is not None and not cal.done:
            cv2.rectangle(disp, (0, h - 124), (w, h - 92), (0, 0, 0), -1)
            cv2.putText(disp, cal.prompt()[:104], (12, h - 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2,
                        cv2.LINE_AA)
        cv2.putText(disp, f"{fps:4.1f} fps   {1000 / max(fps, 1e-6):.0f} ms/frame"
                    f"   backend={getattr(self.backend, 'name', '?')}",
                    (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        if self.motion_on and self.motion_debug:
            d = self.motion_debug
            cv2.putText(disp, f"J/Z  idx={d.get('idx_ext', 0):.2f} "
                        f"wrist(path={d.get('wrist_path', 0):.2f},"
                        f"rev={d.get('wrist_rev', '-')}) "
                        f"tip(path={d.get('tip_path', 0):.2f},"
                        f"rev={d.get('tip_rev', '-')})",
                        (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 220, 255), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(disp, format="bgr24")


def md_table(md: str, heading: str) -> pd.DataFrame | None:
    """Grab the first markdown table under a '## heading' section."""
    try:
        block = md.split("## " + heading, 1)[1]
    except IndexError:
        return None
    block = block.split("\n## ", 1)[0]          # stop at the next section
    rows = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 3:
        return None
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    header, body = cells[0], [c for c in cells[2:] if c != cells[1]]
    if any(len(c) != len(header) for c in body):
        return None
    return pd.DataFrame(body, columns=header)


PROFILES = trained_profiles()
if not PROFILES:
    st.error("No trained model found. Run this first:\n\n"
             "`python run_all.py --skip-download --skip-extract`")
    st.stop()

with st.sidebar:
    st.header("Model")
    # One shipped model; the old `standard` / `round2` weights were deleted.
    prof = PROFILES[0]
    st.caption(f"`{prof}` — {config.PROFILES[prof]['note']}")

require_model(prof)
meta = json.loads(config.meta_path(prof).read_text(encoding="utf-8"))

st.title("ASL Landmark Recogniser")
st.write(
    "Webcam or photo in, 21 hand landmarks out, one small MLP in between. "
    "The letter you see is produced by the same `normalize()` and the same "
    "smoother as the console app (`python src/live.py`)."
)
st.caption(
    "Privacy: frames are analysed in memory and never saved. Nothing typed "
    "here, and no image you show it, can end up in the training set."
)

with st.sidebar:
    backend_name = st.radio("Runtime", ["torch", "onnx"], horizontal=True,
                            help="ONNX is the exported copy; src/export.py "
                                 "proves they agree to 1e-4 on real hands.")
    st.write(f"weights: `{config.model_path(prof).name}`")
    if backend_name == "onnx":
        stale = onnx_staleness(prof)
        if stale:
            st.warning(stale)
    st.write(f"**{meta['model']['type']}** "
             f"{meta['model']['in_dim']}→"
             f"{'→'.join(str(h) for h in meta['model']['hidden'])}→"
             f"{meta['model']['n_classes']}")
    st.write(f"labels: `{''.join(meta['labels'])}` (J and Z come from the "
             "trained stroke head, not the MLP)")
    st.write(f"trained: {meta.get('trained_at', 'unknown')}")

    st.header("Live settings")
    conf_min = st.slider("Minimum confidence to vote", 0.40, 0.95,
                         float(MIN_CONFIDENCE), 0.05)
    margin_min = st.slider("Minimum top1-top2 gap to vote", 0.00, 0.40,
                           float(UNCERTAIN_MARGIN), 0.02,
                           help="below this the frame is shown as UNCERTAIN and "
                                "casts no vote - it is the two-letters-apart-by-"
                                "one-finger case (N/S, K/V, T/L)")
    pairs_on = st.checkbox(
        f"Raised gaps for the pairs this model cannot separate "
        f"(`{live.format_pair_margins(live.PAIR_MARGINS)}`)", value=True,
        help="Four pairs get a bigger gap than the slider: the model is wrong "
             "far more often than its numbers suggest there. Uncheck to use the "
             "flat gap everywhere and compare.")
    with st.expander("Where those four pairs came from", expanded=False):
        st.markdown(
            "Measured on the 122,359 leave-one-dataset-out images in "
            "`reports/oof_all.csv` (each scored by a model that never saw its "
            "own dataset), replaying this page's own vote rule - 4 of the last "
            "7 frames, per capture session.\n\n"
            "| rule | wrong held letters | correct held letters |\n"
            "|---|---|---|\n"
            f"| flat gap only (as before) | 13,094 | 40,134 |\n"
            f"| with `{live.format_pair_margins(live.PAIR_MARGINS)}` | "
            "10,905 (-17%) | 39,786 (-0.9%) |\n\n"
            "6.3 wrong letters deleted per correct one lost; buying the same "
            "3.1-point drop by raising the flat slider instead costs 3.7% of the "
            "correct letters. Per letter: P 54.1% -> 36.8% wrong, Q 30.3% -> "
            "16.4%, H 39.4% -> 33.3%, S 36.7% -> 31.7%, A 44.3% -> 39.6%.\n\n"
            "**P-as-N, Q-as-N and M-as-S are deliberately NOT in the table.** "
            "They are not near-ties: the letter you actually signed is not even "
            "the runner-up in ~90% of those frames, and the gap is often over "
            "0.4 - the model is confidently wrong, and no top1-top2 rule can see "
            "that. Giving those three pairs a raised gap moved the overall wrong "
            "rate by 0.1 points, so they would have cost letters for nothing. "
            "The fix for them is data this model has never seen (more "
            "independent signers), not a gate.")
    vote_n = st.slider("Vote window (frames)", 3, 15, int(VOTE_FRAMES))
    hold_s = st.slider("Hold before typing (seconds)", 0.0, 2.0,
                       float(HOLD_SECONDS), 0.1)
    motion_on = st.checkbox("J / Z stroke head", value=True)
    overlay_on = st.checkbox("Draw the 21 landmarks", value=True)
    mirror_on = st.checkbox("Mirror the webcam (selfie view)", value=True)
    stride = st.select_slider("Work every Nth frame", options=[1, 2, 3, 4],
                              value=int(STRIDE_DEFAULT),
                              help="Raise this on a slow machine; the video "
                                   "stays smooth but the letter updates less "
                                   "often. Measured here: a processed frame "
                                   "costs ~50 ms, so 2 is what fits the frames "
                                   "this camera sends. Set it back to 1 if J or "
                                   "Z stops firing - the stroke head judges a "
                                   "window of frames, and 2 halves how many "
                                   "positions it can see.")
    show_feats = st.checkbox(f"Show the {meta['features']['dim']} feature values",
                             value=False)
    stun_on = st.checkbox("Use a public STUN server", value=False,
                          help="Leave this off unless the browser is on a "
                               "different network from this server - a phone, "
                               "another machine across the internet. On the "
                               "network this app was tested on the Google STUN "
                               "server never answers, and every START then "
                               "dies 10 seconds later with the camera closing "
                               "on its own. Off means the two ends pair "
                               "directly, which is what a local webcam session "
                               "needs.")
    st.header("Keys / buttons")
    st.markdown("- **Commit now** types the letter you are holding.\n"
                "- **Backspace** and **Clear** edit the buffer.\n"
                "- In the console app: SPACE, BACKSPACE, C, Q.")

backend, labels = get_backend(backend_name, prof)

tab_live, tab_photo, tab_snap, tab_model = st.tabs(
    ["Live Webcam", "Upload Photo", "Camera Snapshot", "Model & Results"])

# Tab 1 - Live Webcam
with tab_live:
    st.write(
        "Press **START**. Hold the sign still for about half a second and the "
        "letter is typed below. To type the same letter twice, take your hand "
        "out of view and bring it back. Turn the palm or the back of the hand "
        "to the camera - a hand seen exactly from the side is the case this "
        "pipeline finds hardest."
    )
    srv_rtc, fe_rtc = rtc_pair(stun_on)
    live_stream = webrtc_streamer(
        key="asl-landmark-live",
        rtc_configuration=srv_rtc,
        frontend_rtc_configuration=fe_rtc,
        media_stream_constraints=VIDEO_CONSTRAINTS,
        video_processor_factory=LandmarkProcessor,
        async_processing=True,
    )

    ctrl = st.columns([1, 1, 1, 2])
    if ctrl[0].button("Commit now", use_container_width=True) \
            and live_stream.video_processor:
        live_stream.video_processor.force_next = True
    if ctrl[1].button("Backspace", use_container_width=True) \
            and live_stream.video_processor:
        p = live_stream.video_processor
        with p.lock:
            p.text = p.text[:-1]
    if ctrl[2].button("Clear", use_container_width=True) \
            and live_stream.video_processor:
        p = live_stream.video_processor
        with p.lock:
            p.text = ""
            p.commits = []
    ctrl[3].write("")

    jz_btns = st.columns([1.1, 1.1, 2.0])
    if jz_btns[0].button("Calibrate J / Z", use_container_width=True,
                         help="sign J three times, then Z three times; this "
                              "session's motion thresholds are measured from "
                              "your strokes. In memory only - nothing is "
                              "saved, and the numbers are gone when the page "
                              "closes.") and live_stream.video_processor:
        p = live_stream.video_processor
        with p.lock:
            p.cal = live.JZCalibrator(p.motion, reps=3)
            p.cal_result = None
    if jz_btns[1].button("Cancel calibration", use_container_width=True,
                         help="keep the shipped thresholds") \
            and live_stream.video_processor:
        p = live_stream.video_processor
        with p.lock:
            if p.cal is not None:
                p.cal.cancel()
            p.motion.intended = None
            p.cal = None
    jz_btns[2].caption("J and Z are typed by a motion rule, not by the "
                       "classifier. The rule watches a frame when either the "
                       "index-up shape is held **or** the classifier reads that "
                       "frame as I (which is what a real J looks like to a "
                       "24-class model with no J - at any confidence) or as an "
                       "X handshape (the motion head checks for Z). "
                       "If they never fire for your hand, calibrate once per "
                       "session.")

    def live_panel():
        """Refresh the read-out while the stream runs."""
        p = live_stream.video_processor
        # A heartbeat measured between panel runs: unlike p.fps (lifetime average)
        # and state.playing (a browser flag that goes stale), it proves frames are
        # still reaching Python.
        now = time.monotonic()
        prev_n, prev_t = st.session_state.get("_hb", (None, None))
        in_fps = None
        if p is not None and prev_n is not None and now > prev_t:
            in_fps = max(0.0, (p.n_frames - prev_n) / (now - prev_t))
        st.session_state["_hb"] = (None if p is None else p.n_frames, now)
        alive = in_fps is not None and in_fps > 0.5

        if p is None:
            st.info("The webcam component is not attached to this page. If you "
                    "had already pressed START, the browser reloaded the "
                    "component and dropped the stream with it - press START "
                    "again, and reload the tab (Ctrl+Shift+R) if it keeps "
                    "happening.")
            return
        if not live_stream.state.playing and not alive:
            st.info("The webcam is not running. Press START above.")
            return
        if not live_stream.state.playing:
            st.warning(f"The page thinks the stream is stopped, but "
                       f"**{in_fps:.0f} frames/s are still arriving** - the "
                       "browser component lost its state flag, not the camera. "
                       "Press STOP then START to make the two agree.")
        p.backend, p.labels = backend, labels
        if p.det_private is None:
            # re-pointing a healed detector here would undo the repair
            p.detector = get_detector(False)
        p.conf_min, p.motion_on = conf_min, motion_on
        p.margin = float(margin_min)
        p.pairs = dict(live.PAIR_MARGINS) if pairs_on else {}
        p.overlay, p.mirror, p.stride = overlay_on, mirror_on, int(stride)
        p.smoother.conf, p.smoother.vote = conf_min, int(vote_n)
        p.smoother.hold = hold_s
        p.smoother.min_votes = max(2, (int(vote_n) + 1) // 2)

        with p.lock:
            snap = dict(held=p.held, conf=p.held_conf, raw=p.raw_letter,
                        raw_conf=p.raw_conf, text=p.text, fps=p.fps,
                        n=p.n_frames, hands=p.n_hands, ms=p.ms,
                        progress=p.progress, top=list(zip(p.top_names,
                                                          p.top_probs)),
                        uncertain=p.uncertain, low_conf=p.low_conf,
                        margin=p.margin_now, margin_min=float(margin_min),
                        need=p.need_now, collide=p.collide,
                        lm=None if p.lm is None else p.lm.copy(),
                        motion=p.motion_debug, last_motion=p.last_motion,
                        hint=p.motion_hint,
                        cal=(None if p.cal is None or p.cal.done else {
                            "prompt": p.cal.prompt(),
                            "progress": p.cal.progress()}),
                        cal_result=p.cal_result,
                        thresholds=p.motion.thresholds(),
                        th_source=p.motion.threshold_source,
                        n_err=p.n_frame_errors, err=p.frame_error,
                        healed=p.det_private is not None,
                        commits=list(p.commits[-8:]))

        if snap["err"]:
            st.error(
                f"**The webcam is still running.** {snap['n_err']} frames have "
                f"failed and been skipped; the last one raised "
                f"`{snap['err']}`" +
                (", so a fresh landmark detector was swapped in."
                 if snap["healed"] else ".") +
                " If the overlay stops updating, press STOP and START.")

        a, b = st.columns([3, 2])
        with a:
            if snap["held"]:
                st.success(f"**{snap['held']}**  ({snap['conf'] * 100:.0f} %)")
            elif snap["uncertain"]:
                # use `need`, not the slider value: a raised pair rule may be the
                # reason this frame is silent
                pair = (f" - {snap['collide'][0]}/{snap['collide'][1]} is one of "
                        f"the pairs this model cannot separate"
                        if snap["collide"] else "")
                st.warning(f"**UNCERTAIN** - {snap['raw']} "
                           f"{snap['raw_conf'] * 100:.0f} % is only "
                           f"{snap['margin'] * 100:.0f} points ahead of "
                           f"{snap['top'][1][0] if len(snap['top']) > 1 else '-'} "
                           f"(needs {snap['need'] * 100:.0f}){pair}, so it casts "
                           "no vote")
            else:
                st.info("waiting for a sign")
            if snap["raw"] and snap["low_conf"] and not snap["uncertain"]:
                st.caption(f"raw best: {snap['raw']} "
                           f"{snap['raw_conf'] * 100:.0f} % - below the "
                           f"{conf_min * 100:.0f} % gate, so it casts no vote")
            st.progress(min(1.0, float(snap["progress"])),
                        text=f"hold-to-commit {snap['progress'] * 100:.0f} %")
            st.code(snap["text"] or " ", language=None)
        with b:
            st.metric("pipeline", f"{snap['fps']:.1f} fps",
                      help="frames the browser actually ran through the model")
            st.metric("hand found",
                      f"{100 * snap['hands'] / max(snap['n'], 1):.0f} %",
                      help=f"{snap['hands']} of {snap['n']} frames - "
                           "MediaPipe found no hand in the rest")
            st.write(f"{snap['ms']:.1f} ms per processed frame")
            if in_fps is not None:
                st.write(f"**{in_fps:.1f} frames/s** reaching Python right now "
                         "(this is the number that says the webcam is really "
                         "being read)")
            if snap["top"]:
                st.markdown("**Top 3 right now**")
                for name, pr in snap["top"]:
                    st.progress(min(1.0, float(pr)), text=f"{name}: {pr * 100:.1f} %")
        if snap["lm"] is not None:
            with st.expander("What the model sees (canonicalised skeleton)",
                             expanded=False):
                c1, c2 = st.columns([1, 2])
                c1.image(feature_view(snap["lm"]))
                c2.markdown("Hands that fall in the other 2-D chirality "
                            "family are mirrored to the canonical shape before "
                            "the features are computed - the test is the sign "
                            "of the palm cross product, not MediaPipe's "
                            "`handedness` field, which assumes a mirrored "
                            "selfie frame. Because the test is 2-D it can "
                            "mistake a right hand seen from the back for a "
                            "left hand seen from the palm; if this skeleton "
                            "looks flipped compared with your real hand, that "
                            "mirror decision is why the letter is wrong.")
                if show_feats:
                    st.dataframe(pd.DataFrame({
                        "feature": [f"f{i}" for i in range(
                            len(normalize.features(snap["lm"])))],
                        "value": np.round(normalize.features(snap["lm"]), 4),
                    }), height=240)
        if snap["cal"] is not None:
            st.info(snap["cal"]["prompt"])
            st.progress(min(1.0, float(snap["cal"]["progress"])),
                        text=f"calibration {snap['cal']['progress'] * 100:.0f} % "
                             "- sign the letter, then let your hand relax")
        elif snap["cal_result"]:
            r = snap["cal_result"]
            c, b = r["consistency"], r["baseline_consistency"]
            st.success(
                f"Calibrated this session: {r['n_strokes']} strokes measured, "
                f"{c['fired']}/{c['n']} now fire their own letter "
                f"(shipped thresholds: {b['fired']}/{b['n']}). Nothing was "
                "written to disk - these numbers live in this page only.")
            with st.expander("What the calibration changed, and why",
                             expanded=False):
                D = live.MotionDecider.DEFAULTS
                st.dataframe(pd.DataFrame([{
                    "threshold": k,
                    "shipped": D[k],
                    "this session": r["thresholds"][k],
                    "how it was derived": r["notes"].get(k, ""),
                } for k in sorted(D)]), hide_index=True, height=340)
                st.dataframe(pd.DataFrame([{
                    "stroke": i, "letter": s["intended"],
                    "held (s)": round(float(s["held_full"]), 2),
                    "hand travel": round(float(s["wrist_net"]), 2),
                    "hand path": round(float(s["wrist_path"]), 2),
                    "pinky path": round(float(s["pinky_path_2d"]), 2),
                    "index path": round(float(s["tip_path_2d"]), 2),
                    "vertical": round(float(s["drop"]), 2),
                } for i, s in enumerate(r["strokes"], 1)]), hide_index=True)
                if r["spec"]:
                    st.write("To reuse these numbers in the console app, run:")
                    st.code(f"python src/live.py --jz {r['spec']}")
                else:
                    st.write("Your strokes already suit the shipped "
                             "thresholds, so nothing needed changing.")
        if motion_on and snap["hint"]:
            st.warning(f"{snap['hint']} - if that is your natural gesture, "
                       "press **Calibrate J / Z** above.")
        if motion_on and snap["motion"]:
            m = snap["motion"]
            # report the gate that actually opened the watch
            why = (f"shape gate: index extension {m.get('idx_ext', 0):.2f} "
                   f"(needs >= 1.6)" if m.get("gate") == "shape" else
                   f"display gate: the model reads this frame as "
                   f"{m.get('top1') or '-'} at {m.get('p1', 0):.2f} "
                   f"(I and X frames open J/Z motion checks; the stroke decides)")
            st.caption(
                f"J/Z watch: held {m.get('held', 0):.2f} s, {why}; wrist path "
                f"{m.get('wrist_path', 0):.2f}, reversals "
                f"{m.get('wrist_rev', 0)}, fingertip path "
                f"{m.get('tip_path', 0):.2f}, reversals "
                f"{m.get('tip_rev', 0)}, drop "
                f"{m.get('drop', 0):.2f}. Last motion letter: "
                f"{snap['last_motion'] or '-'} - thresholds "
                f"{snap['th_source'] or 'shipped defaults'}")
        if snap["commits"]:
            st.markdown("**Last commits**")
            rows = [{"letter": letter, "t (s)": round(at - _ts_clock(), 2)}
                    for letter, at in snap["commits"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    if hasattr(st, "fragment"):
        # a timed fragment keeps the sliders and buttons responsive
        live_panel = st.fragment(run_every=0.2)(live_panel)
    live_panel()

# Tab 2 - Upload a photo
with tab_photo:
    st.write("Any hand-sign photo. It is decoded, measured and forgotten - "
             "never written back to disk, never added to training data.")
    up = st.file_uploader("Upload a hand sign image",
                          type=["png", "jpg", "jpeg", "bmp", "webp"],
                          key="upload")
    if up is not None:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="your photo")
        show_prediction(analyse_rgb(np.asarray(img), backend_name, conf_min, prof),
                        "upload", show_feats)

# Tab 3 - Camera snapshot
with tab_snap:
    st.write("Take one still and predict it. Useful for checking a single "
             "handshape without the smoother in the way.")
    mirror_snap = st.checkbox("This camera saves a mirrored (selfie) image",
                              key="mirror_snap")
    snap = st.camera_input("Take a photo of your hand sign")
    if snap is not None:
        img = Image.open(snap).convert("RGB")
        if mirror_snap:
            from PIL import ImageOps
            img = ImageOps.mirror(img)
        st.image(img, caption="snapshot (shown back to you, not saved)")
        show_prediction(analyse_rgb(np.asarray(img), backend_name, conf_min, prof),
                        "snap", show_feats)

# Tab 4 - Model & results
with tab_model:
    md = ""
    rp = config.profile_report(prof)
    if not rp.is_file():
        rp = config.REPORT_MD          # not trained yet: show the summary
    if rp.is_file():
        md = rp.read_text(encoding="utf-8")
        st.caption(f"results read from `{rp.name}`")
    headline = re.search(
        r"\*\*Cross-dataset accuracy, POOLED \(headline\):\s*([\d.]+)%", md)
    pooled = meta.get("cross_dataset_pooled_accuracy")
    if pooled is None and headline:
        pooled = float(headline.group(1)) / 100.0
    macro = meta.get("cross_dataset_macro_accuracy")
    small = meta.get("cross_dataset_small_holdouts") or []
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("cross-dataset, POOLED (headline)",
              f"{100 * pooled:.1f} %" if pooled is not None else "run training",
              help="every held-out image counted once, scored by a model that "
                   "never saw its dataset")
    c2.metric("cross-dataset, macro",
              f"{100 * macro:.1f} %" if macro is not None else "run training",
              help="unweighted mean of the per-dataset accuracies; higher than "
                   "pooled when a small set scores well"
                   + (f" (small-n holdouts: {', '.join(small)})" if small else ""))
    c3.metric("letters covered", f"{len(meta['labels'])} static + J, Z")
    c4.metric("feature dim", meta["features"]["dim"],
              help=" + ".join(meta["features"]["blocks"]))

    st.subheader("How the numbers were produced")
    st.markdown(
        "- **Cross-dataset / leave-one-dataset-out** is the headline: a model "
        "trained on the other datasets predicts this one, so it has never seen "
        "those hands, cameras or rooms.\n"
        "- **Held-out capture group** is the closest thing to your webcam: "
        "Massey groups A+B train, C+D+E test - different signers, same set.\n"
        "- **Block splits** are optimistic and are never quoted as accuracy: "
        "adjacent frames of one clip share both sides.\n"
        "- A random image split is not used anywhere as a headline number."
    )
    for title in ("Held-out capture group (unseen hands, same dataset)",
                  "Pairwise cross-dataset (direction matters)",
                  "Per-class accuracy (pooled cross-dataset predictions)"):
        t = md_table(md, title)
        if t is None:
            continue
        with st.expander(title, expanded=(title.startswith("Per-class"))):
            st.dataframe(t, hide_index=True, height=min(420, 34 + 26 * len(t)))
            if title.startswith("Per-class"):
                acc = pd.to_numeric(t["accuracy"].str.rstrip("%"), errors="coerce")
                chart = pd.DataFrame({"letter": t["letter"], "acc": acc}).set_index(
                    "letter").sort_values("acc")
                st.bar_chart(chart, height=240)

    st.subheader("Run the checks (no camera, nothing saved)")
    b1, b2, b3 = st.columns(3)
    if b1.button("Logic selftest", use_container_width=True,
                 help="Unit checks on the smoother, the gate and the shared "
                      "features, against synthetic landmark sequences. No "
                      "camera, nothing saved."):
        r = subprocess.run([sys.executable, "-u", str(SRC / "live.py"), "--logic-test",
                            "--profile", prof],
                           capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        st.code((r.stdout or "") + (r.stderr or "")[-2000:], language="text")
    if b2.button("FPS benchmark", use_container_width=True):
        r = subprocess.run([sys.executable, "-u", str(SRC / "live.py"),
                            "--benchmark", "150", "--profile", prof],
                           capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        st.code((r.stdout or "") + (r.stderr or "")[-2000:], language="text")
    if b3.button("Live-path A–Z", use_container_width=True,
                 help="Feeds real dataset photographs through the exact webcam "
                      "code path — colour order, MediaPipe video mode, the "
                      "shared features, the gate, the smoother — without "
                      "opening a camera. Reads the report it already wrote; "
                      "re-run it in a terminal to refresh that file. In-sample: "
                      "it proves the wiring, not the accuracy."):
        p = config.REPORTS_DIR / "live_path.md"
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8"))
        else:
            st.info("Not run yet — run "
                    "`python src/verify_livepath.py --per 20 --backend onnx "
                    f"--profile {prof}`.")

    st.markdown("**Live drill** (your webcam, scores in memory only — nothing "
                "is saved). It shows each letter, you hold it ~3 s, and it "
                "counts the per-frame top-1. This is the only measurement of "
                "*your* signing, and it is the one that picks the profile:")
    st.code("cd \"" + str(ROOT) + "\"\n"
            "python src/live.py --selftest --profile all,standard",
            language="powershell")

    st.subheader("Retrain")
    st.code("cd \"" + str(ROOT) + "\"\n"
            "python run_all.py --skip-download --skip-extract\n"
            "python src/train.py && python src/export.py",
            language="powershell")
    st.caption("Full pipeline: `python run_all.py`. Everything is idempotent; "
               "`--quick` is a small smoke run. One model ships (`all`, every "
               "signer weighted equally) — see docs/DETAILS.md.")
