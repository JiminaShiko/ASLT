# --------------------------------------------------
# Streamlit frontend for the ASL LANDMARK recogniser.
#
# Same three ways of using it as the photo model's app (photo / snapshot /
# live webcam), but the model behind it is the landmark pipeline:
#
#   frame (BGR) -> RGB -> MediaPipe Hands 21 landmarks
#       -> src/normalize.py (the exact function training used)
#       -> feature vector (71 dims by default) -> MLP -> Smoother -> letter
#
# It imports the live app's own classes (live.Smoother, live.MotionDecider,
# live.build_backend) instead of duplicating them, so the browser version and
# `python src/live.py` can never disagree.
#
# PRIVACY: this page never writes anything. No frame, photo, landmark or text is
# saved to disk, and nothing you sign can reach the training set. There is no
# calibration tab on purpose - the photo model's calibrate_camera.py is NOT part
# of this project.
#
# Run:  streamlit run app.py
# --------------------------------------------------
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

# --------------------------------------------------
# Settings
# --------------------------------------------------
MIN_CONFIDENCE = 0.70      # frames below this cast no vote
UNCERTAIN_MARGIN = 0.10    # frames whose top1-top2 gap is below this are shown
                           # as UNCERTAIN and cast no vote (Stage F)
VOTE_FRAMES = 7            # majority vote window
HOLD_SECONDS = 0.5         # how long a letter must be held before it is typed
MAX_SIDE = 640             # frames wider/taller than this are shrunk first
STRIDE_DEFAULT = 2         # the app's cheap mode; see "Work every Nth frame".
                           # Measured here: one processed frame costs ~50 ms
                           # (MediaPipe VIDEO 45-52 ms whatever the resolution,
                           # features + MLP 1.4 ms, overlay + re-encode 1.2 ms
                           # at 640x480), so a 30 fps camera feeds 33 ms of
                           # budget with 50 ms of work. Every second frame is
                           # 26 ms, which fits. The cost is temporal resolution:
                           # the J/Z rule sees 12 hand positions a second
                           # instead of 24, and its binding criterion on real
                           # movement is the reversal count - so if J or Z
                           # stops firing, drop this back to 1 (or calibrate).
MEANINGS = {
    "J": "J - moving letter: drawn by the sideways hook rule, not by the MLP",
    "Z": "Z - moving letter: drawn by the fingertip zig-zag rule, not by the MLP",
}

# Every MediaPipe VIDEO timestamp is measured against `_ts_clock()` below, NOT
# against a module global here. The Tasks API rejects any timestamp that is not
# strictly greater than the one its graph last saw, and the landmarker is cached
# for the life of the server - so an epoch that a rerun re-assigned rewound the
# clock underneath a detector that remembered a larger timestamp, the very next
# frame raised, and streamlit-webrtc closed the camera over it.

# ------------------------------------------------- the WebRTC connection
# This app had pointed both ends of the peer connection at
# `stun:stun.l.google.com:19302`. STUN only matters when the two ends are on
# different networks: it asks a public server "what address do you see me
# from?". Here the browser and the server are the same machine, so the local
# (host) candidates ICE already gathers are enough - and the external query
# was worse than useless on the network this app is used on. Measured, from
# the app's own log after one press of START:
#
#   20:57:29.5  create a new worker          (the click)
#   20:57:30.9  setRemoteDescription(offer)  (the camera arrived fine)
#   20:57:36.1 / 37.2 / 38.6  STUN BINDING -> 74.125.250.129:19302, retransmitted,
#                              never answered
#   20:57:39.8  Stopping WebRTC worker       (10.3 s after the click)
#
# The unanswered STUN query stalls ICE gathering, so the SDP answer is never
# finished; `streamlit_webrtc` calls `process_offer(..., timeout=10)`, and at
# ten seconds it stops the worker - releasing the camera and flipping STOP back
# to START - and raises `SignallingTimeoutError`, which is the error on the
# page and the half-drawn white screen behind it.
#
# An empty list is honoured rather than replaced, on both sides. The library
# only fills in Google's server when the field is *absent* (`None`), so the
# browser has to be told as well as aiortc - hence `rtc_pair`. Reproduced
# in-process without a browser, two aiortc peers in one script:
#
#   iceServers []                      gathering 5 ms,  SDP answer 9 ms
#   iceServers [Google]  (the old app) gathering 5 009 ms, SDP answer 10 035 ms
#
# 10 035 ms is over the 10 000 ms budget, which is the whole bug in two lines:
# aioice waits five seconds per host candidate for a STUN reply that this
# network never sends.
GOOGLE_STUN = "stun:stun.l.google.com:19302"
RTC_CONFIGURATION = RTCConfiguration({"iceServers": []})
FRONTEND_RTC_CONFIGURATION: dict = {"iceServers": []}
RTC_CONFIGURATION_STUN = RTCConfiguration(
    {"iceServers": [{"urls": [GOOGLE_STUN]}]}
)
FRONTEND_RTC_CONFIGURATION_STUN: dict = {"iceServers": [{"urls": [GOOGLE_STUN]}]}


def rtc_pair(use_stun: bool):
    """(server, browser) ICE configs that agree with each other.

    They have to be swapped together. A browser told to use STUN while the
    server is not, or the other way round, gives up on the candidate pair and
    the stream never starts - which is exactly the failure this section is
    about, so it is not left to whoever edits one of the two constants.
    """
    if use_stun:
        return RTC_CONFIGURATION_STUN, FRONTEND_RTC_CONFIGURATION_STUN
    return RTC_CONFIGURATION, FRONTEND_RTC_CONFIGURATION


# A 30 fps camera asks for 33 ms a frame; one processed frame measures ~50 ms
# here (see STRIDE_DEFAULT). Asking the browser for 24 fps does not make the
# model faster, it stops the pipeline being handed frames it will throw away
# before MediaPipe ever sees them - the decode and re-encode of a dropped
# frame is pure lost time, and lost time is what the page shows as lag.
VIDEO_CONSTRAINTS = {
    "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
              "frameRate": {"ideal": 24}},
    "audio": False,
}

st.set_page_config(page_title="ASL Landmark Recogniser", page_icon="🤟",
                   layout="centered")


# --------------------------------------------------
# Cached model / detectors (loaded once per server, not per rerun)
# --------------------------------------------------
@st.cache_resource
def get_backend(name: str, prof: str):
    """(backend, labels) exactly as src/live.py builds them for the webcam."""
    return live.build_backend(name, profile=prof)


@st.cache_resource
def get_detector(static: bool):
    return hb.create_detector(static=static)


@st.cache_resource
def _ts_clock() -> float:
    """The one epoch for every VIDEO timestamp, alive as long as the detector is.

    Two lifetimes have to match. `get_detector(False)` is cached, so a rerun -
    a button press, a slider, a resized window, a reconnecting browser - hands
    the *same* MediaPipe graph another frame, and that graph refuses a
    timestamp it has already passed (`ValueError: Input timestamp must be
    monotonically increasing.`, measured). A module-level `time.monotonic()`
    restarts on every rerun, so the longer the camera had been running the
    sooner the next rerun poisoned it. `recv()` adds a per-processor guard on
    top of this, so even a cleared cache cannot make a timestamp go backwards.
    """
    return time.monotonic()


def trained_profiles() -> list[str]:
    """Profiles with weights on disk, so the selector can never offer a model
    that has not been trained yet."""
    return [p for p in sorted(config.PROFILES) if config.model_path(p).is_file()]


def require_model(prof: str):
    """Stop with a clear message if the pipeline has not been trained yet."""
    if not (config.model_path(prof).is_file() and config.meta_path(prof).is_file()):
        st.error(f"No trained model for profile `{prof}`. Run this first:\n\n"
                 f"`python src/train.py --profile {prof}`"
                 f"`python src/export.py --profile {prof}`")
        st.stop()


def onnx_staleness(prof: str) -> str | None:
    """A sentence when the exported copy is older than the weights it mirrors.

    `src/train.py` writes the .pt and nothing else, so after a retrain the ONNX
    quietly becomes the previous model - and this page offers a Runtime switch
    whose entire promise is that the two agree. Cheaper to say so than to be
    wrong silently.
    """
    on, pt = config.onnx_path(prof), config.model_path(prof)
    if not (on.is_file() and pt.is_file()) or on.stat().st_mtime + 1.0 >= pt.stat().st_mtime:
        return None
    return (f"`{on.name}` is from "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(on.stat().st_mtime))}"
            f" but `{pt.name}` is from "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(pt.stat().st_mtime))}"
            f" - the exported copy is an older model. Refresh it with "
            f"`python src/export.py --profile {prof}`.")


# --------------------------------------------------
# Small drawing helpers (in memory only)
# --------------------------------------------------
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
    """What the MLP actually sees: canonicalised, wrist-centred, palm-scaled.

    The photo model's app shows a 96x96 gray silhouette. This pipeline has no
    pixels at all, so the honest equivalent is the normalised skeleton: it
    already includes the left/right mirroring, which is where the model's
    mistakes come from.
    """
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


# --------------------------------------------------
# One still image -> everything the UI shows
# --------------------------------------------------
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
    """The photo tabs' result block, in the same shape as the photo app's."""
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


# --------------------------------------------------
# Live webcam processor: detection + MLP + smoothing, all in the browser tab
# --------------------------------------------------
class LandmarkProcessor(VideoProcessorBase):
    """One instance per START. recv() runs on the webrtc thread only, so the
    VIDEO-mode detector it uses is never touched by two threads at once."""

    def __init__(self):
        self.backend = None
        self.labels = []
        self.detector = None
        self.lock = threading.Lock()
        self.smoother = live.Smoother(conf=MIN_CONFIDENCE, vote=VOTE_FRAMES,
                                      hold=HOLD_SECONDS)
        self.motion = live.MotionDecider()
        # settings pushed from the page
        self.conf_min = MIN_CONFIDENCE
        self.margin = UNCERTAIN_MARGIN
        # the measured per-pair refusal margins from src/live.py; the sidebar
        # checkbox can drop them back to the flat margin above
        self.pairs = dict(live.PAIR_MARGINS)
        self.mirror = True
        self.overlay = True
        self.motion_on = True
        self.stride = int(STRIDE_DEFAULT)
        self.force_next = False
        # results
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
        self.need_now = 0.0     # the gap this frame had to clear, pair rule included
        self.collide = ""       # pair key when PAIR_MARGINS was the binding rule
        self.progress = 0.0
        self.lm = None
        self.motion_debug = {}
        self.last_motion = None
        self.motion_hint = ""      # why J/Z did not fire, when that is the case
        self.cal = None            # a live.JZCalibrator while one is running
        self.cal_result = None     # its finished report, kept for the page
        self.commits = []
        # stats
        self.n_frames = 0
        self.n_hands = 0
        self.fps = 0.0
        self.ms = 0.0
        self._last_tick = None
        self._t0 = time.monotonic()
        # frame-failure bookkeeping: nothing that happens inside a frame is
        # allowed to reach streamlit-webrtc (see recv()), but it must not be
        # invisible either.
        self.n_frame_errors = 0
        self.consec_errors = 0
        self.frame_error = ""
        self._last_ts: int | None = None   # strictly increasing per processor
        self.det_private = None            # a healed detector, see _heal
        self._healed_at = 0.0

    # -- per frame -----------------------------------------------------
    def recv(self, frame):
        """The one thing that keeps the camera open.

        This runs on streamlit-webrtc's worker thread, and `process.py` wraps
        that thread in a try/except which stores whatever escapes and re-raises
        it from the track's own `recv()`. The track then ends, the library calls
        `stop()`, and the browser closes the video element - so a single raise
        here is indistinguishable from "the camera switched itself off". No
        frame failure may leave this method: it is counted, named in the
        read-out panel, and the untouched frame is passed on instead.
        """
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
        """Swap in a private landmarker when every frame has failed for a second.

        A VIDEO graph that has been handed a timestamp it will not accept stays
        unfixed for the rest of its life, and the epoch guard makes that nearly
        impossible - but a corrupted graph would otherwise mean a page that
        shows the camera and nothing else. `live_panel` re-points `self.detector`
        at the cached one every 0.2 s, so the replacement has to live in its own
        field; it is closed on the way out so the old graph is not left running.
        """
        self._healed_at = now
        self.consec_errors = 0
        if not config.HAND_MODEL_PATH.is_file():
            # never let a frame handler start a model download; the stream
            # shows the red box and waits for STOP/START instead
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
            # cheap mode for slow machines: keep the last result, redraw it
            return self._emit(img, started)

        detector = self.det_private or self.detector
        if detector is None or self.backend is None:
            return self._emit(img, started)

        ts_ms = int((time.monotonic() - _ts_clock()) * 1000)
        if self._last_ts is not None and ts_ms <= self._last_ts:
            # cannot happen while `_ts_clock` outruns the detector it belongs
            # to; kept because MediaPipe rejects a repeat as hard as a rewind
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
                # the SAME decision function src/live.py uses, so the browser
                # app cannot disagree with the console app about a frame
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
                    # the raw top-1 of this same frame, NOT the smoothed letter:
                    # the J/Z precondition is about what the model sees right now
                    # (true J reads as I, true Z reads as X), and the smoother is
                    # one decision behind by construction
                    shown, shown_p = self.raw_letter, self.raw_conf
                    if cal is not None and not cal.done:
                        # the guided calibration watches the same frames the
                        # typer uses; nothing is stored except the float
                        # series inside p.motion
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

    # -- drawing the frame the user actually sees ----------------------
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

        # top-3 with probabilities, so a near miss is visible instead of silent
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
                    # name the rule, or a measured refusal looks like a dropout
                    why += (f" - {collide[0]}/{collide[1]} needs "
                            f"{need * 100:.0f}")
                cv2.putText(disp, f"no vote: {why}", (bx, by + 3 * 24 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
                            cv2.LINE_AA)

        cal = self.cal
        if self.motion_on and self.motion_hint:
            # J and Z are not classes, so a missed stroke used to leave the page
            # showing the MLP's nearest neighbour (usually I) with no hint that
            # a motion rule had been holding and not firing.
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


# --------------------------------------------------
# Report parsing for the Model tab (read-only)
# --------------------------------------------------
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


# --------------------------------------------------
# Page
# --------------------------------------------------
PROFILES = trained_profiles()
if not PROFILES:
    st.error("No trained model found. Run this first:\n\n"
             "`python run_all.py --skip-download --skip-extract`")
    st.stop()

with st.sidebar:
    st.header("Model")
    prof = st.selectbox(
        "Profile", PROFILES,
        index=PROFILES.index(config.PROFILE_DEFAULT)
        if config.PROFILE_DEFAULT in PROFILES else 0,
        help="One trained model per profile. 'all' trains on every signer "
             "equally; 'standard' sets the flagged minority-style cells to "
             "weight 0; 'round2' is the shipped halved-weights model. Pick "
             "the one that wins the live drill: "
             "python src/live.py --selftest --profile all,standard")
    st.caption(config.PROFILES[prof]["note"])

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
             "motion rule, not the MLP)")
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
    motion_on = st.checkbox("J / Z motion rule", value=True)
    overlay_on = st.checkbox("Draw the 21 landmarks", value=True)
    mirror_on = st.checkbox("Mirror the webcam (selfie view)", value=True)
    stride = st.select_slider("Work every Nth frame", options=[1, 2, 3, 4],
                              value=int(STRIDE_DEFAULT),
                              help="Raise this on a slow machine; the video "
                                   "stays smooth but the letter updates less "
                                   "often. Measured here: a processed frame "
                                   "costs ~50 ms, so 2 is what fits the frames "
                                   "this camera sends. Set it back to 1 if J or "
                                   "Z stops firing - the motion rule counts "
                                   "direction changes, and 2 halves how many "
                                   "it can see.")
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

# --------------------------------------------------
# Tab 1 - Live Webcam
# --------------------------------------------------
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
                       "unsure X (a real Z). "
                       "If they never fire for your hand, calibrate once per "
                       "session.")

    def live_panel():
        """Refresh the read-out while the stream runs."""
        p = live_stream.video_processor
        # A heartbeat measured between panel runs, independent of both
        # `p.fps` (which averages over the processor's whole life) and
        # `live_stream.state.playing` (which is a flag the browser component
        # sends us, and is the first thing to go stale when that component
        # reloads). It is the only number here that answers "are frames still
        # reaching Python", which is the question "is the webcam working"
        # actually asks - the picture in the box proves the camera is open, not
        # that the model is being fed.
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
            # the flag is wrong but the pipeline is not: keep rendering, and
            # say which of the two broke, because "webcam is not working" and
            # "the page lost track of the webcam" look identical from here
            st.warning(f"The page thinks the stream is stopped, but "
                       f"**{in_fps:.0f} frames/s are still arriving** - the "
                       "browser component lost its state flag, not the camera. "
                       "Press STOP then START to make the two agree.")
        p.backend, p.labels = backend, labels
        if p.det_private is None:
            # a healed detector (_heal) stays in front until STOP/START;
            # re-pointing it at the cached one here would undo the repair
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
            # The stream is still up - say so, and say what broke, instead of
            # letting a skipped frame look like a dead model or a closed camera.
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
                # `need`, not the slider value: a measured pair rule can be the
                # reason this frame is silent, and saying "needs 10" while
                # refusing a 47-point gap would be a lie the user cannot debug.
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
                    "finger path": round(float(s["tip_path"]), 2),
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
            # which of the two preconditions is holding this frame up - saying
            # "needs >= 1.6" while the display gate is what opened the watch
            # would describe a rule that is not the one running
            why = (f"shape gate: index extension {m.get('idx_ext', 0):.2f} "
                   f"(needs >= 1.6)" if m.get("gate") == "shape" else
                   f"display gate: the model reads this frame as "
                   f"{m.get('top1') or '-'} at {m.get('p1', 0):.2f} "
                   f"(I points at J at any confidence; an unsure X points at Z)")
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
        # A timed fragment refreshes only this panel, so the sliders and
        # buttons above stay responsive while the webcam runs.
        live_panel = st.fragment(run_every=0.2)(live_panel)
    live_panel()

# --------------------------------------------------
# Tab 2 - Upload a photo
# --------------------------------------------------
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

# --------------------------------------------------
# Tab 3 - Camera snapshot
# --------------------------------------------------
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

# --------------------------------------------------
# Tab 4 - Model & results
# --------------------------------------------------
with tab_model:
    md = ""
    rp = config.profile_report(prof)
    if not rp.is_file():
        rp = config.REPORT_MD          # profile not trained yet: show the summary
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
            "python src/train.py --profile standard && python src/export.py --profile standard",
            language="powershell")
    st.caption("Full pipeline: `python run_all.py`. Everything is idempotent; "
               "`--quick` is a small smoke run. Profiles: "
               "`all` (everything equal), `standard` (minority-style cells at "
               "weight 0), `round2` (those cells halved) — see docs/DETAILS.md.")
