"""All paths, constants and pins for the word recognizer; the dataset is read-only."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = Path(os.environ.get("ASL_WORDS_DATASET_DIR") or
                   ROOT / "ASL_Citizen")
SPLITS_DIR = DATASET_DIR / "splits"
VIDEOS_DIR = DATASET_DIR / "videos"
LICENSE_FILE = DATASET_DIR / "use.txt"
SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

CACHE_DIR = ROOT / "cache"
LANDMARK_DIR = CACHE_DIR / "landmarks"
MANIFEST_CSV = CACHE_DIR / "manifest.csv"
ERRORS_CSV = CACHE_DIR / "extract_errors.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

VOCAB_JSON = ROOT / "vocab.json"
PHRASES_JSON = ROOT / "phrases.json"

# "ps:" ids cannot collide with ASL Citizen names
EXTERNAL_DIR = ROOT / "datasets" / "external"
EXTERNAL_READ_DIRS = (EXTERNAL_DIR,)

FOR_RELEASE = MODELS_DIR / "words_core20.pt"
META_JSON = MODELS_DIR / "meta.json"
CKPT_DIR = MODELS_DIR / "checkpoints"
ONNX_PATH = MODELS_DIR / "words_core20.onnx"
HOLISTIC_TASK = MODELS_DIR / "holistic_landmarker.task"

HOLISTIC_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
HOLISTIC_BYTES = 13_683_609
HOLISTIC_ENV_VAR = "ASL_WORDS_HOLISTIC_MODEL"

# bump both whenever the packed layout or feature vector changes
LANDMARK_VERSION = 2
FEATURE_VERSION = 3

NUM_FRAMES = 64
# bridge a hand dropout this long (source frames, so 0.33 s at 30 fps)
GAP_FILL_FRAMES = 5
MAX_VIDEO_FRAMES = 240
FRAME_STRIDE_MAX = 4

# packed [T, P, 3] float16 with NaN for a missing group
POSE_UPPER = list(range(27))
N_POSE = len(POSE_UPPER)
N_HAND = 21
HANDS_TOTAL = 2 * N_HAND
FACE_GROUPS: dict[str, list[int]] = {
    # y grows downwards; `python -m src.landmarks --sanity` re-checks the indices
    "lips": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321,
             405, 314, 17, 84, 181, 91, 146,
             13, 14, 78, 80, 81, 312, 311, 308, 415, 310],
    "nose": [168, 6, 197, 5, 195, 4, 1, 2, 98, 327],
    "chin_jaw": [152, 148, 176, 149, 377, 400, 378, 396, 150,
                 395, 136, 397, 172, 58],
    "brows_forehead": [70, 63, 105, 66, 107,
                       300, 293, 334, 296, 336,
                       10, 151, 108],
}
FACE_ORDER = sorted({i for group in FACE_GROUPS.values() for i in group})
N_FACE = len(FACE_ORDER)
P_TOTAL = N_POSE + N_FACE + HANDS_TOTAL

COLS_POSE = slice(0, N_POSE)
COLS_FACE = slice(N_POSE, N_POSE + N_FACE)
COLS_LHAND = slice(N_POSE + N_FACE, N_POSE + N_FACE + N_HAND)
COLS_RHAND = slice(N_POSE + N_FACE + N_HAND, P_TOTAL)

# row 0 of each hand block (wrist)
HAND_WRIST_COLS = (COLS_LHAND.start, COLS_RHAND.start)


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

REST_LABEL = "REST"
OTHER_LABEL = "OTHER"

SEED = 20261116
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
PATIENCE = 8
GRAD_CLIP = 1.0

# beyond ~3x the strong classes start to suffer; judge on the 3-seed VAL mean
WEAK_WORD_BOOST: dict = {
    "thanks": 2.0,
    "happy": 1.5,
    "please": 1.5,
    "hungry": 1.5,
    "thirsty": 1.5,
    "doctor": 1.5,
    "when": 1.5,
    "now": 1.5,
}
AUG = {
    "scale": 0.10,
    "translate": 0.06,
    "rotate": 8.0,
    "jitter": 0.015,
    "speed": (0.80, 1.20),
    "frame_dropout": 0.05,
    "mirror": 0.50,
}

CONFIDENCE_MIN = 0.85
# 75% of real words at 96% precision on test (the old 0.60 typed ~58%)
MARGIN_MIN = 0.15
# below this a threshold cannot help; show the framing hint instead
FRAMING_HAND_RATE_MIN = 0.35
FRAMING_MIN_FRAMES = 12
FRAMING_HINT = ("keep both hands and your face in frame, chest up")
PAIR_MARGINS: dict = {
    # ("a", "b") or "a|b"; malformed keys raise at read time
}

# these swaps survive the confidence floor, so no threshold separates them
HEDGE_PAIRS: dict = {
    # only pairs whose *both* words are in the shipped 40-word vocabulary: a
    # pair naming a word the model cannot say can never be produced
    ("good", "thanks"): "good/thanks?",
    ("now", "today"): "now/today?",
    ("sorry", "me"): "sorry/me?",
}
BUFFER_SECONDS = 2.5
LIVE_STRIDE = 1
TARGET_LIVE_FPS = 15.0

MIN_FREE_BYTES_FOR_CACHE = 2_000_000_000


def use_candidate(name: str) -> Path:
    """Route outputs to candidates/<name>; writes nothing, never touches inputs."""
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
    """Cache-safe stem (some glosses contain spaces)."""
    stem = Path(video_file).stem
    return (stem.replace(" ", "_").replace("/", "-").replace("\\", "-")
            .replace(":", "-"))


def video_path(video_file: str) -> Path:
    """Video on disk; "ps:" ids resolve under EXTERNAL_DIR, the rest under VIDEOS_DIR."""
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
    """'ps' (PopSign) or 'ac' (ASL Citizen)."""
    return "ps" if video_file.startswith("ps:") else "ac"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def logger_for(name: str, log_path: Path | None = None) -> logging.Logger:
    """Console + optional file logger, utf-8 safe on Windows."""
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
