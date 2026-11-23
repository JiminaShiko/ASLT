"""MediaPipe Tasks `HandLandmarker` wrapper shared by extractor and live app.

mediapipe>=1.0 removed `mp.solutions.hands`. Input is always colour (BGR->RGB).
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

if __package__:
    from . import config  # noqa: E402
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

RunningMode = vision.RunningMode

# drawing only (drawing_utils is gone in mediapipe>=1.0)
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
    handedness: str | None            # raw label, audit only (selfie-frame convention)
    cross_sign: float                 # geometric test, see normalize.chirality_family


def swap_label(label: str | None) -> str | None:
    """Label of the same hand in a flipped image; shared by both flip rungs."""
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
    """Build a landmarker; static=True for stills. 0.3 is for retry detectors only."""
    model_path = Path(model_path or config.HAND_MODEL_PATH)
    if not model_path.is_file():
        if __package__:
            from .download_data import ensure_hand_model
        else:
            from download_data import ensure_hand_model

        model_path = ensure_hand_model(quiet=True)
    mode = RunningMode.IMAGE if static else RunningMode.VIDEO
    return vision.HandLandmarker.create_from_options(
        _base_options(mode, model_path, confidence))


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """The one colour conversion. Never grayscale."""
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
    if __package__:
        from .normalize import palm_cross_sign
    else:
        from normalize import palm_cross_sign

    return HandResult(arr, label, palm_cross_sign(arr))


def detect_image(detector, bgr: np.ndarray) -> HandResult:
    """Static-image detection (Step 2)."""
    return _unpack(detector.detect(_to_mp_image(bgr_to_rgb(bgr))))


def _stamp(detector, timestamp_ms: int) -> int:
    """Stamp to feed this detector, clamped forward.

    VIDEO mode rejects stamps <= the last one ("Input timestamp must be
    monotonically increasing") and stays poisoned until wall clock passes it,
    so the high-water mark lives on the detector: several loops and reruns
    share one detector, each with its own epoch.
    """
    ts = int(timestamp_ms)
    last = getattr(detector, "_hb_last_ts", None)
    if last is not None and ts <= last:
        ts = last + 1
        detector._hb_ts_rewound = getattr(detector, "_hb_ts_rewound", 0) + 1
    detector._hb_last_ts = ts
    return ts


def detect_video(detector, bgr: np.ndarray, timestamp_ms: int) -> HandResult:
    """Webcam detection. Any clock is fine: `_stamp` enforces monotonicity."""
    return _unpack(detector.detect_for_video(_to_mp_image(bgr_to_rgb(bgr)),
                                             _stamp(detector, timestamp_ms)))


def load_image_rgb(path: str | Path) -> np.ndarray | None:
    """Read a file the same way the webcam path does: BGR -> RGB, colour."""
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr_to_rgb(bgr)


class Recovery:
    """Cheap two-rung retry (low-conf IMAGE, then mirrored), for live frames.

    Runs after `after` consecutive misses, then every `every`-th frame. A
    mirrored hit is mapped back (x -> 1-x) so features describe the seen pixels.
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
        self.frames = 0          # empty frames seen
        self.tries = 0
        self.hits = 0
        self.flip_hits = 0
        self.cost = 0.0          # seconds inside recovery
        self.detects = 0

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
        """One rung at a time; `how` uses the extractor's vocabulary."""
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
                # label must be swapped with the coordinates (see extract_landmarks)
                return (HandResult(lm, swap_label(fres.handedness),
                                   -fres.cross_sign), "flip_0.3")
        self.cost += time.perf_counter() - t0
        return HandResult(None, None, 0.0), "none"

    def summary(self) -> str:
        """Retry rate is tried/empty, not tried/all."""
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
    """The live detection path shared by live.py and verify_livepath.py.

    Returns (result, how) with how in {video, direct_0.3, flip_0.3, none}.
    """
    res = detect_video(detector, bgr, timestamp_ms)
    if res.landmarks is not None:
        return res, "video"
    if recovery is None or not recovery.enabled:
        return res, "none"
    recovery.frames += 1
    if empty_streak < recovery.after:
        return res, "none"
    # retry at `every` cadence so a long pause stays above 20 FPS
    if (empty_streak - recovery.after) % recovery.every:
        return res, "none"
    return recovery.attempt(bgr)


if __name__ == "__main__":
    det = create_detector(static=True)
    print("HandLandmarker (static) created OK")
    dv = create_detector(static=False)
    print("HandLandmarker (video) created OK")
