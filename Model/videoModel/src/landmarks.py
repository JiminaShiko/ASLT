"""
MediaPipe Holistic wrapper - the only place that talks to MediaPipe.

Training extraction (src/extract.py) and the live demo (src/live.py) both go
through this module, so the landmarks a model was trained on and the
landmarks it sees at runtime come from the same code path, the same model
asset and the same options. RGB in, never grayscale.

Packed per-frame layout (config.P_TOTAL columns, x/y/z float32, NaN when a
group was not detected):

    [ 0 : 27 )                pose, upper body only (indices 0..26)
    [ 27 : 27+N_FACE )        face subset: lips, nose, chin/jaw, brows/forehead
    [ ..  +21 )               left hand, 21 landmarks
    [ ..  +21 )               right hand, 21 landmarks

Landmark z is stored but deliberately NOT used by src/features.py: MediaPipe
reports pose z in metres from the hip while hand z is relative to the wrist,
so mixing them across groups is meaningless. Depth enters the features as the
2D geometry (hand-to-face/chin/chest) plus the missing-hand mask.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config

Missing = type("Missing", (), {})()          # sentinel: this group was absent


# ---------------------------------------------------------------------------
# model asset
# ---------------------------------------------------------------------------
def model_path() -> Path:
    override = __import__("os").environ.get(config.HOLISTIC_ENV_VAR)
    if override:
        return Path(override)
    return config.HOLISTIC_TASK


def ensure_model(log=None, quiet: bool = False) -> Path:
    """Return the .task asset, downloading it once (13.7 MB) if needed."""
    path = model_path()
    if path.is_file() and path.stat().st_size == config.HOLISTIC_BYTES:
        return path
    if path.is_file():
        path.unlink()                      # truncated download - start again
    path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        msg = (f"downloading MediaPipe Holistic ({config.HOLISTIC_BYTES/1e6:.1f}"
               f" MB)\n  {config.HOLISTIC_URL}\n  -> {path}")
        print(msg)
        if log:
            log.info(msg.replace("\n", " "))
    tmp = path.with_suffix(".task.part")
    try:
        with urllib.request.urlopen(config.HOLISTIC_URL, timeout=120) as r:
            total = int(r.headers.get("Content-Length") or config.HOLISTIC_BYTES)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if not quiet:
                        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB",
                              end="", flush=True)
        if not quiet:
            print()
    except Exception as exc:                        # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the Holistic model ({type(exc).__name__}: {exc})\n"
            f"Put the file at {path} yourself, or set {config.HOLISTIC_ENV_VAR}\n"
            f"to a copy of holistic_landmarker.task from\n"
            f"  {config.HOLISTIC_URL}"
        ) from exc
    if tmp.stat().st_size != config.HOLISTIC_BYTES:
        raise RuntimeError(
            f"downloaded model is {tmp.stat().st_size} B, expected "
            f"{config.HOLISTIC_BYTES} B - re-run or place the file manually")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# frame decoding
# ---------------------------------------------------------------------------
def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """cv2 gives BGR; MediaPipe wants contiguous RGB uint8. Never grayscale."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def video_info(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def frames_of(path: Path, max_frames: int | None = None,
              stride: int = 1):
    """Yield (index, timestamp_ms, rgb_uint8) for one video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step_ms = 1000.0 * stride / fps
    count = 0
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, int(round(count * step_ms)), bgr_to_rgb(bgr)
                count += 1
                if max_frames and count >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# tracker
# ---------------------------------------------------------------------------
def _flat(seq) -> list:
    """Accept both layouts: a flat list of landmarks or one nested per person."""
    out = list(seq or ())
    if out and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return out


def _landmark_row(lm) -> tuple[float, float, float]:
    return (float(lm.x), float(lm.y), float(lm.z))


def _pack_single(group, indices: list[int]) -> np.ndarray:
    out = np.full((len(indices), 3), np.nan, dtype=np.float32)
    for i, idx in enumerate(indices):
        if idx < len(group):
            out[i] = _landmark_row(group[idx])
    return out


def pack_result(result) -> np.ndarray:
    """HolisticLandmarkerResult -> [P_TOTAL, 3] float32 with NaN rows.

    Measured on mediapipe 1.2.2: the Holistic result lists are FLAT
    (pose_landmarks is a list of 33 NormalizedLandmark, face_landmarks 478,
    left/right_hand_landmarks 21 or empty) - unlike PoseLandmarker, which
    nests them per detection. Getting this wrong reads len() of a landmark.
    """
    row = np.full((config.P_TOTAL, 3), np.nan, dtype=np.float32)
    pose = _flat(result.pose_landmarks)
    if pose:
        row[config.COLS_POSE] = _pack_single(pose, config.POSE_UPPER)
    face = _flat(result.face_landmarks)
    if face:
        row[config.COLS_FACE] = _pack_single(face, config.FACE_ORDER)
    for hands, cols in ((result.left_hand_landmarks, config.COLS_LHAND),
                        (result.right_hand_landmarks, config.COLS_RHAND)):
        hand = _flat(hands)
        if hand:
            row[cols] = _pack_single(hand, list(range(config.N_HAND)))
    return row


@dataclass
class Tracker:
    """VIDEO-mode Holistic landmarker with a monotonic clock it cannot break.

    MediaPipe raises "Input timestamp must be monotonically increasing" if a
    VIDEO-mode detector is fed a smaller timestamp than before (learned the
    hard way in the alphabet project). Two guards: a clamp, and an automatic
    reset when the caller's clock rewinds (new segment, new video, rerun).

    HolisticLandmarkerOptions exposes no num_threads / num_hands: one pose and
    two hands is fixed, and threading is inside the TFLite delegate.
    """
    min_face_conf: float = 0.30
    min_pose_conf: float = 0.30
    min_landmarks_conf: float = 0.30
    _det: object = field(default=None, repr=False)
    _last_ts: int = field(default=-1, repr=False)
    n_frames: int = 0
    n_resets: int = 0

    def __post_init__(self) -> None:
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        opts = mp_vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model(quiet=True))),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=self.min_face_conf,
            min_face_landmarks_confidence=self.min_landmarks_conf,
            min_pose_detection_confidence=self.min_pose_conf,
            min_pose_landmarks_confidence=self.min_landmarks_conf,
            min_hand_landmarks_confidence=self.min_landmarks_conf,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self._det = mp_vision.HolisticLandmarker.create_from_options(opts)

    # -- lifecycle ---------------------------------------------------------
    def restart(self) -> None:
        """Rebuild the detector: the only way to give a VIDEO-mode landmarker a
        fresh clock. Used between videos and when a webcam stream restarts."""
        self.close()
        self.__post_init__()
        self._last_ts = -1
        self.n_resets += 1

    reset = restart                       # explicit name for the same thing

    def close(self) -> None:
        try:
            self._det.close()
        except Exception:                          # noqa: BLE001
            pass

    # -- one frame ---------------------------------------------------------
    def detect(self, rgb: np.ndarray, ts_ms: int | None = None) -> np.ndarray:
        """rgb uint8 [H,W,3] -> packed [P_TOTAL,3] float32 (NaN rows if absent).

        The timestamp handed to MediaPipe is NEVER allowed to go backwards:
        the clamp is absolute, so a caller that restarts its own clock (new
        video, new segment) simply gets a monotonic continuation instead of
        the "Input timestamp must be monotonically increasing" ValueError
        that killed the alphabet project's live app. Call restart() when you
        genuinely want a new stream rather than a continuation.
        """
        import mediapipe as mp
        ts = int(ts_ms if ts_ms is not None else self.n_frames * 33)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._det.detect_for_video(img, ts)
        self.n_frames += 1
        return pack_result(res)


# ---------------------------------------------------------------------------
# geometry helpers shared by extraction and the live segmenter
# ---------------------------------------------------------------------------
def _xy(arr: np.ndarray) -> np.ndarray:
    return arr[:, :2]


def face_row(original_index: int) -> int:
    """Row of a face-mesh landmark inside the packed face block.

    The block is FACE_ORDER (a sorted union), not the group order, so the
    overlay has to go through this map rather than assume a slice.
    """
    return config.COLS_FACE.start + config.FACE_ORDER.index(original_index)


def hand_row(hand: str, landmark: int) -> int:
    """Row of one of the 21 hand landmarks ('left' or 'right')."""
    start = (config.COLS_LHAND if hand == "left"
             else config.COLS_RHAND).start
    return start + landmark


def shoulder_mid(arr: np.ndarray) -> np.ndarray:
    """[2] shoulder midpoint in normalised image coords; NaN if no shoulders."""
    l = arr[config.POSE_L_SHOULDER, :2]
    r = arr[config.POSE_R_SHOULDER, :2]
    if np.isnan(l).any() or np.isnan(r).any():
        return np.full(2, np.nan)
    return (l + r) / 2.0


def shoulder_width(arr: np.ndarray) -> float:
    l = arr[config.POSE_L_SHOULDER, :2]
    r = arr[config.POSE_R_SHOULDER, :2]
    if np.isnan(l).any() or np.isnan(r).any():
        return float("nan")
    return float(np.linalg.norm(l - r))


def hand_present(arr: np.ndarray) -> tuple[bool, bool]:
    """Per-hand presence, keyed on the wrist (landmark 0 of each block)."""
    return tuple(not np.isnan(arr[c, :2]).any()
                 for c in config.HAND_WRIST_COLS)


def hands_above_shoulders(arr: np.ndarray, margin: float = 0.02) -> bool:
    """True when a detected wrist is at or above the shoulder line.

    Image y grows downwards, so 'above' means a smaller y. Used to tell a
    resting hand (in the lap / out of frame) from a signing hand.
    """
    sh = shoulder_mid(arr)
    if np.isnan(sh).any():
        return False
    for idx in (config.POSE_L_WRIST, config.POSE_R_WRIST):
        w = arr[idx, 1]
        if not np.isnan(w) and w <= sh[1] - margin:
            return True
    for cols in (config.COLS_LHAND, config.COLS_RHAND):
        block = _xy(arr[cols])
        if np.isnan(block).all():                  # this hand was not detected
            continue
        if np.nanmin(block[:, 1]) <= sh[1] - margin:
            return True
    return False


def hand_detection_rate(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    ok = [any(hand_present(f)) for f in arr]
    return float(sum(ok) / len(ok))


def hand_share(arr: np.ndarray) -> float:
    """The same number as hand_detection_rate(), vectorised for the live loop.

    It is called on every webcam frame over the rolling buffer, so it must not
    loop in Python. Both key on the two wrists, which is what hand_present()
    keys on, so the live rate and the manifest's hand_rate_seg are the same
    measurement on different windows - the one number the framing hint can
    honestly compare against.
    """
    a = np.asarray(arr)
    if a.ndim != 3 or len(a) == 0:
        return 0.0
    l = a[:, config.HAND_WRIST_COLS[0], :2]
    r = a[:, config.HAND_WRIST_COLS[1], :2]
    return float(np.mean(~(np.isnan(l).any(axis=1) & np.isnan(r).any(axis=1))))


def hand_rate_window(arr: np.ndarray, start: float, end: float) -> float:
    """Detection rate over the signing window only - the number that matters.

    hand_detection_rate() counts the whole clip, and a clip is a sentence: the
    hands drop to the speaker's side between signs, so a clip-wide rate reads
    low even when the hand is perfectly visible for every frame of the sign
    itself. Measured on the val/test cache the two numbers differ enough to
    change conclusions (good reads 0.38 over the clip and 0.92 inside its
    signing window, and no core word is below 0.69), which is why the manifest
    carries both and why the clip-wide rate must not be read as a detection
    ceiling.
    """
    T = len(arr)
    if T == 0:
        return 0.0
    a, b = int(max(0, float(start))), int(min(T, float(end)))
    if b <= a:
        a, b = 0, T
    return hand_detection_rate(arr[a:b])


def signing_window(arr: np.ndarray, min_rest: int = 3,
                   hold: int = 2) -> tuple[int, int]:
    """Trim the idle head/tail of a clip: (sign_start, sign_end), end exclusive.

    The trimmed frames are what feeds the REST class, so the rest of the clip
    must stay in the array - this only reports indices, it never deletes.
    """
    T = len(arr)
    if T == 0:
        return (0, 0)
    flags = np.array([hands_above_shoulders(f) for f in arr])
    if not flags.any():
        return (0, T)                              # nothing detected: all REST
    start = int(np.argmax(flags))
    end = int(T - np.argmax(flags[::-1]))
    # require a stable run, otherwise a single noisy frame would open the window
    for i in range(start, min(end, start + 2 * hold)):
        if flags[i:i + hold].all():
            start = i
            break
    for i in range(end, max(start, end - 2 * hold), -1):
        if flags[max(i - hold, 0):i].all():
            end = i
            break
    if end - start < 2:                            # degenerate: keep everything
        return (0, T)
    if start < min_rest and end > T - min_rest:
        pass                                       # symmetrical idle on both sides
    return (max(0, start - 1), min(T, end + 1))    # one frame of context


# ---------------------------------------------------------------------------
# whole-video convenience (used by extract.py and by the offline verifiers)
# ---------------------------------------------------------------------------
@dataclass
class Extraction:
    video: str
    landmarks: np.ndarray            # [T, P_TOTAL, 3] float16
    fps: float
    stride: int
    window: tuple[int, int]
    hand_rate: float
    ms: float                        # decode + MediaPipe for this clip
    n_frames: int
    restart_ms: float = 0.0          # per-video stream rebuild overhead

    @property
    def rest_frames(self) -> int:
        s, e = self.window
        return s + (self.n_frames - e)


def extract_video(path: Path, tracker: Tracker | None = None,
                  restart: bool = True) -> Extraction:
    """Run one whole video file through Holistic in VIDEO mode.

    Two clock facts, both measured on mediapipe 1.2.2:
      * frames_of() numbers each clip from 0 (clip-relative time), and
      * Tracker.detect() clamps to a globally increasing timestamp, so a new
        clip never raises "Input timestamp must be monotonically increasing".
    restart=True additionally rebuilds the detector, so clip N+1 is not
    temporal-smoothed with the tail of clip N. The rebuild cost is timed and
    stored, because it is real per-video overhead in the ETA.
    """
    own = tracker is None
    tracker = tracker or Tracker()
    t_restart = 0.0
    if restart:
        t0 = time.perf_counter()
        tracker.restart()
        t_restart = (time.perf_counter() - t0) * 1000
    n, fps, w, h = video_info(path)
    stride = 1
    if n > config.MAX_VIDEO_FRAMES:
        stride = min(config.FRAME_STRIDE_MAX,
                     int(np.ceil(n / config.MAX_VIDEO_FRAMES)))
    rows = []
    t0 = time.perf_counter()
    for idx, ts, rgb in frames_of(path, stride=stride):
        rows.append(tracker.detect(rgb, ts))
    arr = (np.stack(rows) if rows
           else np.zeros((0, config.P_TOTAL, 3), np.float32))
    ms = (time.perf_counter() - t0) * 1000
    win = signing_window(arr)
    if own:
        tracker.close()
    return Extraction(video=path.name, landmarks=arr.astype(np.float16),
                      fps=fps, stride=stride, window=win,
                      hand_rate=hand_detection_rate(arr), ms=ms,
                      n_frames=len(arr), restart_ms=t_restart)


# ---------------------------------------------------------------------------
# sanity: prove the face subset indices really are lips / nose / jaw / brows
# ---------------------------------------------------------------------------
def sanity(path: Path | None = None, n_frames: int = 20) -> dict:
    """Mean coordinates of every face group on real frames, top-to-bottom.

    The MediaPipe face mesh has 478 points and no readable names, so the
    subset in config.FACE_GROUPS is only trustworthy if a geometric check
    says so: eyebrows must sit above the eyes, the nose between them and the
    lips, the chin below the lips. This prints that ordering.
    """
    if path is None:
        cands = sorted(config.VIDEOS_DIR.glob("*-HELLO.mp4"))[:1]
        if not cands:
            raise SystemExit("no video to sanity-check; pass a path")
        path = cands[0]
    tracker = Tracker()
    rows = []
    for _i, _t, rgb in frames_of(path, max_frames=n_frames):
        rows.append(tracker.detect(rgb))
    tracker.close()
    arr = np.stack(rows).astype(np.float32)
    face = arr[:, config.COLS_FACE, :2]
    # eyes are NOT in our subset; use the pose eye rows as the reference band
    eye_y = np.nanmean(arr[:, [config.pose_row(2), config.pose_row(5)], 1])
    groups = {g: [k for k, i in enumerate(config.FACE_ORDER) if i in ids]
              for g, ids in config.FACE_GROUPS.items()}
    out = {"video": path.name, "frames": len(arr), "eye_y": round(float(eye_y), 3),
           "groups": {}}
    for g, cols in groups.items():
        y = float(np.nanmean(face[:, cols, 1]))
        x = float(np.nanmean(face[:, cols, 0]))
        out["groups"][g] = {"x": round(x, 3), "y": round(y, 3),
                            "n": len(cols)}
    order = sorted(out["groups"], key=lambda g: out["groups"][g]["y"])
    out["top_to_bottom"] = order
    gy = {g: out["groups"][g]["y"] for g in out["groups"]}
    ok = (gy["brows_forehead"] < eye_y < gy["nose"] < gy["lips"] < gy["chin_jaw"])
    out["expected"] = "brows_forehead < eyes < nose < lips < chin_jaw"
    out["verdict"] = "PASS" if ok else "CHECK THE INDICES"
    return out


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanity", nargs="?", const="", default=None,
                    help="video file name (default: any HELLO) to verify the "
                         "face/pose subset with a geometric check")
    ap.add_argument("--video", default=None,
                    help="full path override for --sanity")
    args = ap.parse_args()
    config.ensure_dirs()
    p = None
    if args.video:
        p = Path(args.video)
    elif args.sanity is not None:
        p = (config.VIDEOS_DIR / args.sanity) if args.sanity else None
    if p is None:
        print(json.dumps(sanity(), indent=1))
        sys.exit(0)
    print(json.dumps(sanity(p), indent=1))
