"""Landmark payloads for the browser skeleton overlay.

Every payload is a list of normalised (x, y) points (or a chain of such lists
for the face); the page maps them onto the object-fit: cover <video>. Keeping
the geometry here means the Flask bridge and the learning test share one copy,
and the browser never receives an annotated JPEG.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache

import cv2
import numpy as np

from videoModel.src import config as words_config
from videoModel.src import landmarks as word_landmarks

log = logging.getLogger(__name__)

FACE_LIPS_OUTER = tuple(words_config.FACE_GROUPS["lips"][:20])
FACE_LIPS_INNER = tuple(words_config.FACE_GROUPS["lips"][20:])
FACE_JAW = tuple(words_config.FACE_GROUPS["chin_jaw"])
FACE_BROWS = (
    tuple(words_config.FACE_GROUPS["brows_forehead"][:5]),
    tuple(words_config.FACE_GROUPS["brows_forehead"][5:10]),
)
FACE_NOSE = tuple(words_config.FACE_GROUPS["nose"])


@lru_cache(maxsize=None)
def _face_row(mesh_index: int) -> int:
    return word_landmarks.face_row(mesh_index)


def coords(points) -> list:
    """(n, 2) normalised coords -> [[x, y] | None]; NaN stays None."""
    return [
        [round(float(x), 4), round(float(y), 4)]
        if np.isfinite(x) and np.isfinite(y) else None
        for x, y in points
    ]


def hand_points(landmarks) -> list:
    """Single-hand detector output -> one 21-point chain."""
    if landmarks is None:
        return []
    return coords(np.asarray(landmarks)[:, :2])


def packed_hand_points(row) -> list:
    """Both hands out of one packed videoModel row (NaN = not seen)."""
    if row is None:
        return []
    arr = np.asarray(row)
    try:
        present = word_landmarks.hand_present(arr)
    except Exception:
        return []
    hands = []
    for is_present, cols in zip(
            present, (words_config.COLS_LHAND, words_config.COLS_RHAND)):
        if not is_present:
            continue
        points = arr[cols, :2]
        if np.isnan(points).any():
            continue
        hands.append(coords(points))
    return hands


def body_points(row) -> list:
    """Upper-body pose rows (0-26) as one point list."""
    if row is None:
        return []
    return coords(np.asarray(row)[words_config.COLS_POSE, :2])


def head_payload(row) -> dict:
    """Face subset as polylines plus nose dots (same chains live.py strokes)."""
    if row is None:
        return {"lines": [], "dots": []}
    arr = np.asarray(row)

    def chain(indices, close=False):
        rows = [_face_row(index) for index in indices]
        if close:
            rows.append(rows[0])
        return arr[rows, :2]

    lines = []
    for indices, close in (
            (FACE_LIPS_OUTER, True), (FACE_LIPS_INNER, True),
            (FACE_JAW, False), (FACE_BROWS[0], False), (FACE_BROWS[1], False)):
        line = coords(chain(indices, close))
        if any(point is not None for point in line):
            lines.append(line)
    dots = [point for point in coords(chain(FACE_NOSE))
            if point is not None]
    return {"lines": lines, "dots": dots}


EMPTY_POSE = {"head": {"lines": [], "dots": []}, "body": []}

# VIDEO-mode Holistic dies if two frames differ in size, and building it costs
# seconds while its close() can block for ~20-40 s (it hung camera stops), so
# one tracker serves the whole process and gets a fixed-size frame. Normalised
# coordinates do not change under axis-aligned scaling, so the payload still
# maps onto the original video.
DETECT_SIZE = (640, 480)  # (width, height)
_MAX_BUILDS = 3

_tracker = None
_tracker_failed = False
_tracker_builds = 0
_tracker_ready = threading.Event()
_build_lock = threading.Lock()
_detect_lock = threading.Lock()


def _build_tracker() -> None:
    global _tracker, _tracker_failed, _tracker_builds
    with _build_lock:
        if _tracker is not None or _tracker_failed:
            _tracker_ready.set()
            return
        if _tracker_builds >= _MAX_BUILDS:
            _tracker_failed = True
            _tracker_ready.set()
            return
        _tracker_builds += 1
        try:
            _tracker = word_landmarks.Tracker()
        except Exception as error:
            _tracker = None
            if _tracker_builds >= _MAX_BUILDS:
                _tracker_failed = True
            log.warning("Holistic tracker build %d/%d failed: %s",
                        _tracker_builds, _MAX_BUILDS, error)
        finally:
            _tracker_ready.set()


def prewarm() -> None:
    """Start building the tracker now so the first frame does not wait."""
    if _tracker_ready.is_set() and (_tracker is not None or _tracker_failed):
        return
    threading.Thread(target=_build_tracker, daemon=True).start()


def _invalidate_tracker() -> None:
    """A failed MediaPipe graph cannot recover; drop it and rebuild once."""
    global _tracker
    with _build_lock:
        _tracker = None
        _tracker_ready.clear()
    prewarm()


def _canonical_frame(rgb: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    if (rgb.shape[1], rgb.shape[0]) == DETECT_SIZE:
        return rgb
    shrinking = rgb.shape[1] > DETECT_SIZE[0] or rgb.shape[0] > DETECT_SIZE[1]
    return np.ascontiguousarray(cv2.resize(
        rgb, DETECT_SIZE,
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR))


class PoseOverlay:
    """Body + face trace from the Holistic tracker, throttled for the web loop.

    One model pass per `interval` seconds keeps the letter pipeline close to its
    old per-frame budget; between passes the previous payload is reused, so the
    browser still gets a payload with every frame it sends. Detection runs on
    the shared tracker and simply skips until it is ready.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = max(0.0, float(interval))
        self._last_at = float("-inf")
        self._payload = EMPTY_POSE
        prewarm()

    def _detect(self, rgb: np.ndarray, now: float) -> dict:
        frame = _canonical_frame(rgb)
        with _detect_lock:
            tracker = _tracker
            if tracker is None:
                return EMPTY_POSE
            try:
                row = tracker.detect(frame, int(now * 1000.0))
            except Exception as error:
                log.warning("Holistic detect failed, rebuilding tracker: %s", error)
                _invalidate_tracker()
                return EMPTY_POSE
        return {"head": head_payload(row), "body": body_points(row)}

    def update(self, rgb: np.ndarray, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        if now - self._last_at < self.interval:
            return self._payload
        self._last_at = now
        try:
            self._payload = self._detect(rgb, now)
        except Exception:
            # The trace is cosmetic; a Holistic failure must never stop recognition.
            self._payload = EMPTY_POSE
        return self._payload

    def close(self) -> None:
        """Nothing to release: the tracker is shared and lives with the process."""
