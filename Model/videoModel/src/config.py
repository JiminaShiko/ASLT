"""
videoModel word recognizer - one place for every path, constant and pin.

Nothing in this file touches the alphabet project (../asl_landmark_app) and
nothing writes inside ASL Citizen/. The dataset is read-only by design: the
only files this module creates are under cache/, models/, reports/, logs/.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # .../videoModel

# Git moved code into ASLT/Model/, but intentionally left the large local
# data/cache at ASLT/videoModel/. Read them in place; a fresh clone uses ROOT.
# An explicit dataset override is read-only and never changes download paths.
LEGACY_ROOT = (ROOT.parent.parent / "videoModel"
               if ROOT.parent.name == "Model" else ROOT)


def existing_local(relative: str, marker: str = "") -> Path:
    """Prefer this project, otherwise reuse a pre-move local resource."""
    current = ROOT / relative
    previous = LEGACY_ROOT / relative
    if not (current / marker).exists() and (previous / marker).exists():
        return previous
    return current


DATASET_DIR = Path(os.environ.get("ASL_WORDS_DATASET_DIR") or
                   existing_local("ASL_Citizen", "splits/train.csv"))
SPLITS_DIR = DATASET_DIR / "splits"
VIDEOS_DIR = DATASET_DIR / "videos"
LICENSE_FILE = DATASET_DIR / "use.txt"
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

CACHE_DIR = existing_local("cache", "manifest.csv")  # generated, gitignored
LANDMARK_DIR = CACHE_DIR / "landmarks"              # one .npy per video
MANIFEST_CSV = CACHE_DIR / "manifest.csv"           # one row per processed video
ERRORS_CSV = CACHE_DIR / "extract_errors.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

VOCAB_JSON = ROOT / "vocab.json"
PHRASES_JSON = ROOT / "phrases.json"

# PopSign v2.1 subset, fetched by download_popsign_targets.py (per-clip
# branch). Clips land under EXTERNAL_DIR/<dataset>/<split>/<gloss>/<file>.mp4
# and job ids carry the same path with a "ps:" prefix, so they can never
# collide with ASL Citizen file names.
EXTERNAL_DIR = ROOT / "datasets" / "external"  # all NEW downloads go here
EXTERNAL_READ_DIRS = tuple(dict.fromkeys(
    (EXTERNAL_DIR, LEGACY_ROOT / "datasets" / "external")))

FOR_RELEASE = MODELS_DIR / "words_core20.pt"       # shipped weights
META_JSON = MODELS_DIR / "meta.json"
CKPT_DIR = MODELS_DIR / "checkpoints"
ONNX_PATH = MODELS_DIR / "words_core20.onnx"
HOLISTIC_TASK = existing_local("models/holistic_landmarker.task")

# ---------------------------------------------------------------------------
# MediaPipe Holistic asset (13.7 MB, fetched once by src/landmarks.py)
# ---------------------------------------------------------------------------
HOLISTIC_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
HOLISTIC_BYTES = 13_683_609          # verified with a HEAD request, 2026-11-16
# an already-downloaded copy can be pointed at with this env var
HOLISTIC_ENV_VAR = "ASL_WORDS_HOLISTIC_MODEL"

# ---------------------------------------------------------------------------
# landmark / feature geometry
# ---------------------------------------------------------------------------
LANDMARK_VERSION = 2                 # bump if the packed .npy layout changes
FEATURE_VERSION = 3                  # bump if features.py changes its vector
# v3 = short hand dropouts are bridged by interpolation, plus the segment
# read-out block (hand-to-mouth/chin/chest at start/middle/end, excursion,
# travel, retreat after the closest approach, presence fractions).

NUM_FRAMES = 64                      # every sequence is resampled to this
GAP_FILL_FRAMES = 5                  # bridge a hand dropout this long (source
                                     # frames, not resampled ones: 5 frames is
                                     # 0.17 s of a 30 fps clip, 0.33 s live)
MAX_VIDEO_FRAMES = 240               # safety cap for an unexpectedly long clip
FRAME_STRIDE_MAX = 4                 # skip frames only if a clip is huge

# packed landmark array: [T, P, 3] float16, NaN where a group was missing
POSE_UPPER = list(range(27))         # 0..26 = nose..hips; legs are cropped out
N_POSE = len(POSE_UPPER)
N_HAND = 21
HANDS_TOTAL = 2 * N_HAND
FACE_GROUPS: dict[str, list[int]] = {
    # Every index below was measured, not recalled: python -m src.landmarks
    # --sanity prints the mean normalised position of each group on real
    # frames and checks brow < eye < nose < lip < chin. (2026-11-16, 25
    # frames of a HELLO clip; y grows downwards.)
    "lips": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321,
             405, 314, 17, 84, 181, 91, 146,                    # outer ring
             13, 14, 78, 80, 81, 312, 311, 308, 415, 310],      # inner ring
    "nose": [168, 6, 197, 5, 195, 4, 1, 2, 98, 327],            # y .276-.351
    "chin_jaw": [152, 148, 176, 149, 377, 400, 378, 396, 150,
                 395, 136, 397, 172, 58],                       # chin + jaw line
    "brows_forehead": [70, 63, 105, 66, 107,                   # subject-left brow
                       300, 293, 334, 296, 336,                 # subject-right brow
                       10, 151, 108],                           # forehead
}
# fixed concatenation order inside the packed array
FACE_ORDER = sorted({i for group in FACE_GROUPS.values() for i in group})
N_FACE = len(FACE_ORDER)
P_TOTAL = N_POSE + N_FACE + HANDS_TOTAL          # columns of the packed array

COLS_POSE = slice(0, N_POSE)
COLS_FACE = slice(N_POSE, N_POSE + N_FACE)
COLS_LHAND = slice(N_POSE + N_FACE, N_POSE + N_FACE + N_HAND)
COLS_RHAND = slice(N_POSE + N_FACE + N_HAND, P_TOTAL)

# column of each hand's wrist (landmark 0 of its 21-point block); slices above
# are slice objects, so the first row of a block is taken from .start
HAND_WRIST_COLS = (COLS_LHAND.start, COLS_RHAND.start)

# pose indices we care about, expressed in the *cropped* POSE_UPPER space
def pose_row(original_index: int) -> int:
    """Row of a MediaPipe pose landmark inside the cropped upper-body block."""
    return POSE_UPPER.index(original_index)


POSE_L_SHOULDER = pose_row(11)
POSE_R_SHOULDER = pose_row(12)
POSE_L_ELBOW = pose_row(13)
POSE_R_ELBOW = pose_row(14)
POSE_L_WRIST = pose_row(15)
POSE_R_WRIST = pose_row(16)
POSE_NOSE = pose_row(0)
POSE_L_HIP = pose_row(25)
POSE_R_HIP = pose_row(26)
POSE_MOUTH_L = pose_row(9)
POSE_MOUTH_R = pose_row(10)

# ---------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------
REST_LABEL = "REST"
OTHER_LABEL = "OTHER"

# ---------------------------------------------------------------------------
# training defaults (overridable on the command line)
# ---------------------------------------------------------------------------
SEED = 20261116
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
PATIENCE = 8                       # early stopping on val top-1
GRAD_CLIP = 1.0
AUG = {
    "scale": 0.10,                 # +-10 % shoulder-width jitter
    "translate": 0.06,             # shoulder-width units
    "rotate": 8.0,                 # degrees, in the image plane
    "jitter": 0.015,               # per-landmark gaussian, scaled units
    "speed": (0.80, 1.20),         # velocity block scale + sub-window length
    "frame_dropout": 0.05,         # repeat-last-frame dropout
    "mirror": 0.50,                # p(hand/left-right swap + x mirror)
}

# live inference
CONFIDENCE_MIN = 0.85              # flag: commit threshold
# 0.85 is measured, not guessed: on the signer-disjoint test readings
# (reports/predictions.csv) committing only at p1 >= 0.85 still types 75% of
# the real words, and 96% of what it types is the right word, while OTHER slips
# through 12% and REST 2%. At the old 0.60 it typed almost everything and was
# right about 58% of the time. Lower it only if you want a livelier demo.
MARGIN_MIN = 0.15                  # flag: top1 - top2 gap required to commit
# Framing. When the live buffer has barely any hand in it, no threshold can
# help: say so instead of showing a guess. The rate is measured over the
# buffered frames exactly like hand_rate_seg in the manifest.
FRAMING_HAND_RATE_MIN = 0.35       # below this fraction of frames with a hand
FRAMING_MIN_FRAMES = 12            # ...and the buffer must hold at least this
                                   # many frames before the hint is trusted
FRAMING_HINT = ("keep both hands and your face in frame, chest up")
PAIR_MARGINS: dict = {
    # filled from reports/confused_pairs.csv after the first real training run:
    # pairs the model actually swaps get a bigger required gap. A key is a
    # pair of words in either shape - ("please", "sorry") or "please|sorry" -
    # and a key that is not exactly two different words raises at read time.
    #   ("please", "sorry"): 0.30,
}

# Live hedge: when these two words are the top-2, type neither and show the
# hint instead - at ANY confidence. Unlike PAIR_MARGINS this is not a
# threshold: the good/thanks swaps were measured (2026-11-18, test readings)
# with margins 0.36..0.92 overlapping the correct rows' 0.38..0.95, so the
# model is confidently wrong and no floor separates them. The value is what
# the window prints under the reading.
HEDGE_PAIRS: dict = {("good", "thanks"): "good/thanks?"}
BUFFER_SECONDS = 2.5               # rolling window for the live segmenter
LIVE_STRIDE = 1                    # process every Nth webcam frame
TARGET_LIVE_FPS = 15.0

MIN_FREE_BYTES_FOR_CACHE = 2_000_000_000      # ~2 GB is enough for the caps here

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def use_candidate(name: str) -> Path:
    """Route outputs to one named candidate; never change input data or weights
    in the deployed models/ directory. Calling this function writes nothing.
    """
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("candidate name must contain only letters, digits, _ or -")
    base = ROOT / "candidates" / name
    if not base.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("candidate directory must stay inside videoModel")
    global MODELS_DIR, FOR_RELEASE, META_JSON, ONNX_PATH, CKPT_DIR
    global REPORTS_DIR, LOGS_DIR
    MODELS_DIR = base / "models"
    FOR_RELEASE = MODELS_DIR / "words_core20.pt"
    META_JSON = MODELS_DIR / "meta.json"
    ONNX_PATH = MODELS_DIR / "words_core20.onnx"
    CKPT_DIR = MODELS_DIR / "checkpoints"
    REPORTS_DIR = base / "reports"
    LOGS_DIR = base / "logs"
    return base


def ensure_dirs() -> None:
    for d in (CACHE_DIR, LANDMARK_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
              CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def dataset_available() -> bool:
    return VIDEOS_DIR.is_dir() and all(
        (SPLITS_DIR / f).is_file() for f in SPLIT_FILES.values())


def require_dataset() -> None:
    if not dataset_available():
        sys.exit(
            f"ASL Citizen is not where this module expects it:\n"
            f"  videos : {VIDEOS_DIR}\n"
            f"  splits : {SPLITS_DIR}\\train.csv|val.csv|test.csv\n\n"
            "ASL Citizen is a separate ~70 GB download with Microsoft Research\n"
            "terms. Existing data is reused without moving it. Set\n"
            "ASL_WORDS_DATASET_DIR to the existing ASL_Citizen folder if needed.\n"
            "New PopSign downloads are handled by download_popsign_targets.py."
        )


def free_bytes(path: Path) -> int:
    import shutil
    return shutil.disk_usage(str(path)).free


def safe_stem(video_file: str) -> str:
    """cache-safe name for a video file (some glosses contain spaces)."""
    stem = Path(video_file).stem
    return (stem.replace(" ", "_").replace("/", "-").replace("\\", "-")
            .replace(":", "-"))


def video_path(video_file: str) -> Path:
    """Video on disk, honouring the source prefix of a job id.

    ASL Citizen rows are plain file names and always resolve under VIDEOS_DIR
    (read-only). PopSign rows start with "ps:" and carry their path relative
    to EXTERNAL_DIR, e.g. ps:popsign_v2_1/train/good/122a-good-....mp4.
    """
    if video_file.startswith("ps:"):
        relative = Path(video_file[3:])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("external video id must be a relative dataset path")
        for root in EXTERNAL_READ_DIRS:
            path = root / relative
            if path.is_file():
                return path
        return EXTERNAL_DIR / relative
    return VIDEOS_DIR / video_file


def source_of(video_file: str) -> str:
    """The corpus a job id comes from: 'ps' (PopSign) or 'ac' (ASL Citizen)."""
    return "ps" if video_file.startswith("ps:") else "ac"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def logger_for(name: str, log_path: Path | None = None) -> logging.Logger:
    """Console + (optionally) file logger, utf-8 safe on Windows consoles."""
    ensure_dirs()
    log = logging.getLogger(f"words.{name}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(fh)
    return log


def env_summary() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("mediapipe", "numpy", "torch", "cv2", "pandas", "pyarrow",
                "sklearn", "onnx", "onnxruntime", "tqdm"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception as exc:                       # noqa: BLE001
            out[mod] = f"MISSING ({type(exc).__name__})"
    return out


def cpu_count() -> int:
    return os.cpu_count() or 4
