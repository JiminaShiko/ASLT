"""Shared paths and label definitions, so training and the live app cannot drift apart."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # asl_landmark_app/
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
KAGGLEHUB_CACHE_DIR = ROOT / "datasets" / "kagglehub"
# Keep downloads inside the project, never a machine-global Kaggle cache.
os.environ["KAGGLEHUB_CACHE"] = str(KAGGLEHUB_CACHE_DIR)

LANDMARKS_PARQUET = CACHE_DIR / "landmarks.parquet"
# Cache version; the loader refuses a cache written by a different extractor.
CACHE_VERSION = 2
EXTRACT_META_JSON = CACHE_DIR / "extract_meta.json"
DATASET_INDEX_PARQUET = CACHE_DIR / "dataset_index.parquet"
REPORT_MD = ROOT / "REPORT.md"

# One shipped profile. The superseded `standard` (massey M/N/P down-weighted to
# zero, 66.0 % pooled) and `round2` (65.9 %) models were deleted on 2026-11-27
# along with their weights: `all` is the model that won cross-dataset (68.7 %)
# and keeping a second one only invited the wrong file to be loaded.
PROFILES: dict[str, dict] = {
    "all": {
        "prefix": "asl_all",
        "style_weight": 1.0,
        "drop_cells": False,
        "also_drop": "",
        "note": "every signer trained on equally, nothing down-weighted",
    },
}
# Override with ASL_PROFILE or --profile (only `all` exists now).
PROFILE_DEFAULT = os.environ.get("ASL_PROFILE", "all")


def profile(name: str | None = None) -> str:
    """Validate a profile name; falls back to PROFILE_DEFAULT."""
    name = name or PROFILE_DEFAULT
    if name not in PROFILES:
        raise SystemExit(
            f"unknown --profile {name!r}; choose from "
            f"{', '.join(sorted(PROFILES))}")
    return name


def _prefix(name: str | None = None) -> str:
    return PROFILES[profile(name)]["prefix"]


def model_path(name: str | None = None, must_exist: bool = False):
    p = MODELS_DIR / f"{_prefix(name)}.pt"
    if must_exist and not p.is_file():
        raise SystemExit(
            f"{p.name} is not trained yet. Build it with:\n"
            f"  python src/train.py --profile {profile(name)}\n"
            f"  python src/export.py --profile {profile(name)}\n"
            f"Available: {', '.join(sorted(x.name for x in MODELS_DIR.glob('*.pt'))) or 'none'}")
    return p


def onnx_path(name: str | None = None):
    return MODELS_DIR / f"{_prefix(name)}.onnx"


def meta_path(name: str | None = None):
    return MODELS_DIR / f"{_prefix(name)}.meta.json"


def profile_report(name: str | None = None):
    """Where a profile run writes its report; never REPORT.md (hand-written)."""
    return REPORTS_DIR / f"report_{profile(name)}.md"


# MediaPipe hand model from Google's official storage bucket.
HAND_MODEL_NAME = "hand_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / HAND_MODEL_NAME
HAND_MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
]

ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Folders we must never treat as letters.
IGNORED_LABELS = {"nothing", "del", "space", "background", "background_traffic"}
# J/Z need motion, not a static shape: excluded here, handled by MotionDecider.
MOTION_LETTERS = {"J", "Z"}
LABELS = [c for c in ALL_LETTERS if c not in MOTION_LETTERS]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}

# Holdouts smaller than this are reported but flagged, not used as a headline.
HOLDOUT_MIN_FOR_HEADLINE = 2000

NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
# The Tasks API has no tracking-duration knob; each frame is decided on its own.

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
SCALE_EPS = 1e-6
# Opposite-chirality hands are mirrored (x negated); landmark ids are anatomical,
# so mirroring does not permute them. See docs/DETAILS.md.
CANONICAL_HANDEDNESS = "Right"
FINGERTIPS = [4, 8, 12, 16, 20]
# (mcp, pip, dip, tip) chains used for the per-finger curl angle feature.
FINGER_CHAINS = [
    (1, 2, 3, 4),      # thumb
    (5, 6, 7, 8),      # index
    (9, 10, 11, 12),   # middle
    (13, 14, 15, 16),  # ring
    (17, 18, 19, 20),  # pinky
]
# Single source of truth is src/normalize.py; mirrored here for reporting.
if __package__:
    from . import normalize as _nz  # noqa: E402
else:
    import normalize as _nz  # noqa: E402

FEATURE_BLOCKS = _nz.FEATURE_BLOCKS


def feature_dim() -> int:
    """Computed from normalize itself, so the two cannot disagree."""
    if __package__:
        from . import normalize as normalizer
    else:
        import normalize as normalizer

    return int(normalizer.features_from_normalized_batch(np.zeros((1, 21, 3))).shape[1])



# Where images live, how folders map to labels, and how they group into sessions.
DATASETS = [
    {
        "id": "grassknoted",
        "kind": "kaggle",
        "ref": "grassknoted/asl-alphabet",
        "role": "primary",
        "license": "GPL 2.0 (as listed on the Kaggle dataset page)",
        "why": (
            "26 real photographs per letter taken by the dataset author plus "
            "3000 synthetic MNIST-style renders per letter; it is the de-facto "
            "standard static ASL alphabet set and the one the brief asked for."
        ),
        "roots": ["asl_alphabet_train", "asl_alphabet_test"],
        "letter_from": "top_folder",
        "session_from": "letter_folder",
    },
    {
        "id": "massey",
        "kind": "kaggle",
        "ref": "mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out",
        "role": "cross_dataset",
        "license": "Unknown (no license stated on the Kaggle page) - "
                   "research use; original source is Massey University, "
                   "published under an academic non-commercial understanding.",
        "why": (
            "Genuinely independent multi-signer fingerspelling set: cropped RGB "
            "(and depth) frames from a Kinect, 5 different non-native signers, "
            "different hands, backgrounds and lighting from grassknoted. This is "
            "what makes an honest cross-dataset accuracy number possible."
        ),
        # dataset5/<GROUP>/<letter>/color_<n>_<frame>.png - the outer folder is a
        # capture group (A-E), not the letter.
        "roots": ["dataset5"],
        "letter_from": "sub_folder",
        "session_from": "top_folder",
        "image_prefix": "color_",
    },
    {
        "id": "danrasband",
        "kind": "kaggle",
        "ref": "danrasband/asl-alphabet-test",
        "role": "holdout",
        "license": "CC0 - Public Domain",
        "why": (
            "870 extra 200x200 real-world photos, 30 per letter, CC0 licensed. "
            "Caveat recorded in the report: the photos are taken by the same "
            "author as grassknoted, so it is a weaker independence check than "
            "Massey. We keep it as a third evaluation set."
        ),
        "roots": [""],
        "letter_from": "top_folder",
        "session_from": "letter_folder",
    },
]

# A fourth, independent set of photographs: another camera, crop and signer.
DATASETS.append({
    "id": "rushil",
    "kind": "kaggle",
    "ref": "officialrushil/asl-alphabet-dataset",
    "role": "cross_dataset",
    "license": "Unknown (no licence stated on the Kaggle page) - research use",
    "why": ("about 7,800 independently shot letter photographs (~300 per "
            "letter, 26 folders) that share neither camera, crop nor signer "
            "with the other three sources."),
    # "asl data" holds the letter folders; "" is the fallback root for repacks.
    "roots": ["asl data", ""],
    "letter_from": "top_folder",
    "session_from": "letter_folder",
})


# J and Z are strokes, not shapes, so no photo dataset can add them. msl_dynamic is
# Mexican SL (trajectory stage only, never the classifier); chicagofswild has
# sequence-level labels only.
JZ_DIR = ROOT / "datasets" / "jz"
JZ_CLIP_DIR = JZ_DIR / "clips"
JZ_LANDMARK_DIR = CACHE_DIR / "jz_landmarks"

MOTION_DATASETS = [
    {
        "id": "msl_dynamic",
        "kind": "zenodo",
        "record": "14689869",
        "files": ["MSL-dynamic-signs-frontal-view.7z"],
        "license": "CC-BY-4.0",
        "size_gb": 2.2,
        # ASCII on purpose: cp1252 consoles mangle the N-tilde.
        "letters": "J K (N-tilde) Q X Z",
        "why": ("real fingerspelling strokes for J/Z; Mexican SL, so its "
                "labels train the trajectory stage only"),
    },
    {
        "id": "chicagofswild",
        "kind": "url",
        "url": "https://home.ttic.edu/~klivescu/ChicagoFSWild.tgz",
        "license": "research use (see the TTIC ChicagoFSWild page)",
        "size_gb": 14.0,
        "letters": "A-Z (sequence-level labels only)",
        "why": ("real ASL fingerspelling from 160 signers; needs alignment "
                "before it can train anything"),
    },
]


META_DEFAULT = {
    "labels": LABELS,
    "normalization": {
        "origin_landmark": WRIST_INDEX,
        "scale_landmarks": [WRIST_INDEX, MIDDLE_MCP_INDEX],
        "scale_eps": SCALE_EPS,
        "canonical_handedness": CANONICAL_HANDEDNESS,
        "mirror_left_to_right": True,
    },
    "features": {
        "blocks": list(FEATURE_BLOCKS),
        "dim": feature_dim(),
        "coord_space": "2d_image_plane" if "xyz" not in FEATURE_BLOCKS else "3d",
        "fingertips": FINGERTIPS,
        "finger_chains": FINGER_CHAINS,
    },
    "mediapipe": {
        "num_hands": NUM_HANDS,
        "min_hand_detection_confidence": MIN_HAND_DETECTION_CONFIDENCE,
        "min_hand_presence_confidence": MIN_HAND_PRESENCE_CONFIDENCE,
        "running_mode_static": "vision.RunningMode.IMAGE",
        "running_mode_live": "vision.RunningMode.VIDEO",
        "api": "mediapipe.tasks.python.vision.HandLandmarker",
    },
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
