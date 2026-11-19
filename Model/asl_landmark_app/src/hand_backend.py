"""MediaPipe Hands wrapper shared by the extractor and the live app.

mediapipe>=1.0 removed `mp.solutions.hands`, so this uses the Tasks API
(`HandLandmarker`). The brief's settings map onto it exactly:

    static_image_mode=True            -> vision.RunningMode.IMAGE
    max_num_hands=1                   -> num_hands=1
    min_detection_confidence=0.5      -> min_hand_detection_confidence=0.5

Input is always COLOR: OpenCV hands us BGR, we convert to RGB and only then
hand the frame to MediaPipe. Grayscale is never used.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_PROPAGATE_CXX_STD", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

RunningMode = vision.RunningMode

# 21-landmark skeleton, needed because mp.solutions.drawing_utils is gone in
# mediapipe>=1.0. Used only for drawing; the model never sees it.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(bgr: np.ndarray, lm_px) -> np.ndarray:
    """Overlay the hand skeleton on a copy-free in-memory frame (BGR, in place)."""
    h, w = bgr.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in lm_px]
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(bgr, (x, y), 4, (0, 128, 255) if i in (4, 8, 12, 16, 20) else (255, 255, 255),
                   -1, cv2.LINE_AA)
    return bgr


@dataclass
class HandResult:
    landmarks: np.ndarray | None      # (21,3) float32, normalized image coords
    handedness: str | None            # MediaPipe's raw label, audit only.
    #   It is defined for a horizontally MIRRORED (selfie) frame, so on the
    #   unmirrored frames this project feeds it names each chirality family by
    #   the opposite word; measured over the 127,791 cached hands the two tests
    #   sort hands into the same groups 98.4 / 91.1 / 87.0 % of the time
    #   (danrasband / grassknoted / massey). Nothing here is used for features
    #   or for the on-screen "which hand" line - see
    #   `normalize.chirality_family`, which is the image-plane test.
    cross_sign: float                 # geometric handedness test (see normalize)


def swap_label(label: str | None) -> str | None:
    """The MediaPipe label for the same hand in a horizontally flipped image.

    One definition, used by the live recovery rung and (inline) by the offline
    extraction ladder, so a `flip_0.3` row cannot mean two different things in
    the two paths.
    """
    if label in ("Left", "Right"):
        return "Right" if label == "Left" else "Left"
    return label


def _base_options(mode: RunningMode, model_path: Path,
                  confidence: float | None = None):
    return vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_hands=config.NUM_HANDS,
        min_hand_detection_confidence=(config.MIN_HAND_DETECTION_CONFIDENCE
                                       if confidence is None else float(confidence)),
        min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=0.5,
    )


def create_detector(model_path: Path | None = None, static: bool = True,
                    confidence: float | None = None):
    """Build a landmarker. static=True for still images, False for the webcam.

    confidence=None keeps the brief's spec value (0.5). The extractor passes
    0.3 only for its RETRY detectors, never for the primary one.
    """
    model_path = Path(model_path or config.HAND_MODEL_PATH)
    if not model_path.is_file():
        from download_data import ensure_hand_model

        model_path = ensure_hand_model(quiet=True)
    mode = RunningMode.IMAGE if static else RunningMode.VIDEO
    return vision.HandLandmarker.create_from_options(
        _base_options(mode, model_path, confidence))


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """The one place colour conversion happens. Never grayscale."""
    if bgr.ndim == 2:
        raise ValueError("got a grayscale frame; the pipeline requires colour")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_mp_image(rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb, dtype=np.uint8))


def _unpack(result) -> HandResult:
    if not result.hand_landmarks:
        return HandResult(None, None, 0.0)
    lms = result.hand_landmarks[0]
    arr = np.asarray([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
    label = None
    if result.handedness and result.handedness[0]:
        label = result.handedness[0][0].category_name
    from normalize import palm_cross_sign

    return HandResult(arr, label, palm_cross_sign(arr))


def detect_image(detector, bgr: np.ndarray) -> HandResult:
    """Static-image detection (Step 2)."""
    return _unpack(detector.detect(_to_mp_image(bgr_to_rgb(bgr))))


def _stamp(detector, timestamp_ms: int) -> int:
    """The stamp this detector is allowed to see - never a rewind.

    MediaPipe's VIDEO mode keeps a temporal state machine keyed on the caller's
    clock. Anything at or behind the last stamp raises
    `ValueError: Input timestamp must be monotonically increasing.`, and the
    detector stays poisoned until wall-clock time passes it, which from outside
    looks like "the webcam died". So the rule *one clock per detector, moving
    only forward* belongs with the detector, not with whichever loop happens to
    be reading the camera: `live.py` runs three loops over one detector (typing,
    J/Z calibration, the A-Z drill), `app.py` re-runs its page on every rerun,
    and each of those used to invent its own epoch - a calibration followed by
    typing rewound the clock by however long the calibration took and killed the
    session on its first frame.

    The high-water mark is stored on the detector object itself, so it is born
    with the detector, dies with it, and cannot be forgotten by a caller. A
    caller that keeps one clock, as they all should, never trips the clamp.
    """
    ts = int(timestamp_ms)
    last = getattr(detector, "_hb_last_ts", None)
    if last is not None and ts <= last:
        ts = last + 1
        detector._hb_ts_rewound = getattr(detector, "_hb_ts_rewound", 0) + 1
    detector._hb_last_ts = ts
    return ts


def detect_video(detector, bgr: np.ndarray, timestamp_ms: int) -> HandResult:
    """Webcam detection (Step 5). `timestamp_ms` may come from any clock: see
    `_stamp`, which makes the monotonicity MediaPipe demands a property of the
    detector rather than a promise from the caller."""
    return _unpack(detector.detect_for_video(_to_mp_image(bgr_to_rgb(bgr)),
                                             _stamp(detector, timestamp_ms)))


def load_image_rgb(path: str | Path) -> np.ndarray | None:
    """Read a file the same way the webcam path does: BGR -> RGB, colour."""
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr_to_rgb(bgr)


class Recovery:
    """The cheap version of the extractor's ladder, for live frames.

    The extractor can afford five detections per image because it runs once,
    offline, over 122k files. A webcam cannot. This keeps the one rung that is
    both cheap and safe to repeat: an IMAGE-mode detection at a lower
    confidence, then the same on a mirrored copy.

    Two knobs decide when it runs. `after` is how many consecutive frames the
    tracker has to miss before the first retry - a single dropped frame costs
    the smoother nothing, so retrying it would be pure waste. `every` is the
    cadence after that: once the streak is established, the retry repeats every
    `every`-th frame rather than every frame, because a hand that is genuinely
    out of shot only has to be probed to know it is gone. Tracking successes
    never touch it, so a clean session pays nothing.

    A mirrored hit is mapped back to the original frame (x -> 1-x) so the
    feature vector always describes the pixels the user is looking at;
    `normalize.canonicalize` still does the chirality work afterwards.
    """

    def __init__(self, conf: float = 0.3, after: int = 1, every: int = 3,
                 flip: bool = True, model_path: Path | None = None):
        self.after = max(0, int(after))
        self.every = max(1, int(every))
        self.conf = float(conf)
        self.flip = bool(flip)
        self.enabled = self.after > 0
        self._det = None
        self.model_path = model_path
        self.frames = 0          # frames seen by this helper
        self.tries = 0           # how many got the expensive retry
        self.hits = 0            # ... of which found a hand
        self.flip_hits = 0
        self.cost = 0.0          # seconds spent inside recovery
        self.detects = 0         # detections per tried frame, for the cost note

    def _detector(self):
        if self._det is None:
            self._det = create_detector(model_path=self.model_path, static=True,
                                        confidence=self.conf)
        return self._det

    def close(self):
        if self._det is not None:
            self._det.close()
            self._det = None

    def attempt(self, bgr: np.ndarray) -> tuple[HandResult, str]:
        """One rung at a time; stop at the first hand. `how` matches the
        extractor's vocabulary so live and offline numbers can be compared."""
        det = self._detector()
        t0 = time.perf_counter()
        self.tries += 1
        self.detects += 1
        res = detect_image(det, bgr)
        if res.landmarks is not None:
            self.hits += 1
            self.cost += time.perf_counter() - t0
            return res, "direct_0.3"
        if self.flip:
            self.detects += 1
            fres = detect_image(det, cv2.flip(bgr, 1))
            if fres.landmarks is not None:
                lm = fres.landmarks.copy()
                lm[:, 0] = 1.0 - lm[:, 0]
                self.hits += 1
                self.flip_hits += 1
                self.cost += time.perf_counter() - t0
                # The landmarks are mapped back into the original frame, so the
                # label has to be mapped back too, or the live path would report
                # the opposite hand from the extractor's flip rung
                # (extract_landmarks.py does exactly this swap).
                return (HandResult(lm, swap_label(fres.handedness),
                                   -fres.cross_sign), "flip_0.3")
        self.cost += time.perf_counter() - t0
        return HandResult(None, None, 0.0), "none"

    def summary(self) -> str:
        """`frames` counts the frames the VIDEO tracker returned nothing for,
        which is the only population recovery can act on. The retry rate is
        therefore 'tried / empty', not 'tried / all'."""
        pct = 100 * self.tries / self.frames if self.frames else 0.0
        per = self.cost / self.tries if self.tries else 0.0
        avg = self.cost / self.frames if self.frames else 0.0
        return (f"recovery(conf={self.conf}, after={self.after}, "
                f"every={self.every}, flip={'on' if self.flip else 'off'}): "
                f"{self.frames} empty frames, retried on {self.tries} "
                f"({pct:.1f}% of them), found {self.hits} hands "
                f"({self.flip_hits} via mirror), "
                f"{1000 * per:.1f} ms per try, {1000 * avg:.1f} ms amortised")


def detect_frame(detector, bgr: np.ndarray, timestamp_ms: int,
                 recovery: Recovery | None = None,
                 empty_streak: int = 0) -> tuple[HandResult, str]:
    """THE live detection path, shared by live.py and verify_livepath.py.

    Returns (result, how) with how in {video, direct_0.3, flip_0.3, none}.
    `empty_streak` is how many frames including this one the tracker has
    returned nothing for; recovery only starts once it reaches the threshold,
    so a session of clean tracking pays nothing for it.
    """
    res = detect_video(detector, bgr, timestamp_ms)
    if res.landmarks is not None:
        return res, "video"
    if recovery is None or not recovery.enabled:
        return res, "none"
    recovery.frames += 1
    if empty_streak < recovery.after:
        return res, "none"
    # Cadence. Once the threshold is crossed, retry every `every`-th frame
    # instead of every frame: a hand that is genuinely out of shot needs one
    # retry to prove it, not thirty, and an unbounded retry is what pushes the
    # loop under the 20 FPS target during a long pause.
    if (empty_streak - recovery.after) % recovery.every:
        return res, "none"
    return recovery.attempt(bgr)


if __name__ == "__main__":
    det = create_detector(static=True)
    print("HandLandmarker (static) created OK")
    dv = create_detector(static=False)
    print("HandLandmarker (video) created OK")
