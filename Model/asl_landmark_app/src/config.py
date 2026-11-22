"""Shared configuration for the ASL landmark recognizer.

Every script in this package imports paths and label definitions from here so
that training and the live app can never drift apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # asl_landmark_app/
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
KAGGLEHUB_CACHE_DIR = ROOT / "datasets" / "kagglehub"
# Keep dataset acquisition inside this project folder so extraction/training
# never silently depends on a machine-global Kaggle cache.
os.environ["KAGGLEHUB_CACHE"] = str(KAGGLEHUB_CACHE_DIR)

LANDMARKS_PARQUET = CACHE_DIR / "landmarks.parquet"
# Sidecar that records WHICH extractor produced the cache. Round 1 (v1) used a
# single detection attempt at confidence 0.5 and dropped ~37k images; v2 adds
# the recovery ladder (lower confidence / upscale+pad / flipped retry). The
# loader refuses to train on a cache whose version does not match the code, so
# an old cache can never silently mix with new rows.
CACHE_VERSION = 2
EXTRACT_META_JSON = CACHE_DIR / "extract_meta.json"
DATASET_INDEX_PARQUET = CACHE_DIR / "dataset_index.parquet"
# Legacy round-2 paths. The shipped profiles are named per profile
# (`models/asl_all.*`, `models/asl_standard.*`); these bare names are what the
# `round2` profile still writes, and the bare `reports/confusion_matrix.png` is
# what `train.py` used to write for the default profile. Those five files were
# pruned in the 2026-11-15 folder cleanup - `python src/train.py --profile
# round2` regenerates the model files, and nothing else reads them. Since that
# cleanup `write_report` names its figure after the profile too, so no run
# recreates the bare PNG; the constant stays only so old notes that quote
# `reports/confusion_matrix.png` still resolve to a path.
MODEL_PATH = MODELS_DIR / "asl_mlp.pt"
META_PATH = MODELS_DIR / "meta.json"
ONNX_PATH = MODELS_DIR / "asl_mlp.onnx"
REPORT_MD = ROOT / "REPORT.md"
CONFUSION_PNG = REPORTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------
# The same features, the same MLP and the same folds, differing only in how
# much training weight the minority-style cells get (Stage B: massey signs M, N
# and P differently from grassknoted + danrasband). Two profiles ship; `round2`
# is the incumbent that every round-2 number came from. Its weights, sidecar and
# report were pruned in the 2026-11-15 folder cleanup, so it is defined here but
# not loadable: `python src/train.py --profile round2 && python src/export.py
# --profile round2` brings it back if a self-test needs all three on one camera.
#   style_weight : weight for the flagged (dataset, letter) cells
#   drop_cells   : flagged cells get weight 0, i.e. they never train the model
PROFILES: dict[str, dict] = {
    "all": {
        "prefix": "asl_all",
        "style_weight": 1.0,
        "drop_cells": False,
        "also_drop": "",
        "note": "every signer trained on equally, nothing down-weighted",
    },
    "standard": {
        "prefix": "asl_standard",
        "style_weight": 1.0,
        "drop_cells": True,
        # divergence.csv flags M/N/P in massey on its own. T was tested for
        # inclusion in round-3 Stage B and rejected: its cross-dataset
        # mean-shape gap is 4.3 bone units, inside the range of letters that
        # work fine (A 3.6, Y 2.8, V 5.3) and a fifth of M's 25.9, and BOTH
        # large datasets fail on it, which is not what one odd cell does.
        "also_drop": "",
        "note": ("the minority-style cells (massey M/N/P) get weight 0: the "
                 "model only ever sees the majority signing of those letters"),
    },
    "round2": {
        "prefix": "asl_mlp",
        "style_weight": 0.5,
        "drop_cells": False,
        "also_drop": "",
        "note": ("round 2's shipped model: minority cells halved, not removed "
                 "(weights pruned 2026-11-15; retrain to load)"),
    },
}
# Which one `live.py` / `app.py` load when nobody says.
#
# Round 3 measured all three on the same held-out images: `all` 67.0 % pooled,
# `standard` 66.0 %, `round2` 65.9 % (reports/report_*.md, and the paired table
# in `python src/profile_compare.py`). Down-weighting the massey M/N/P cells did
# not rescue those letters in the other datasets - grassknoted M went 3.3 % ->
# 1.7 % and T 31.9 % -> 17.0 % - so the profile that treats every signer alike
# is also the one that scores best, and it is the default. `ASL_PROFILE` or
# `--profile` override it per run; the live self-test
# (`python src/live.py --selftest --profile all,standard`) is still the only
# evidence about a particular hand, and it outranks this table for one person's
# webcam.
#
# The 44 MB of out-of-fold dumps that `src/profile_compare.py` and
# `src/diagnose_letter.py` read by default (reports/oof_all.csv,
# reports/oof_standard.csv) were pruned in the 2026-11-15 cleanup; both scripts
# print the exact `--dump-preds` command that rebuilds them.
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
    # round 2 predates profiles, so its sidecar kept the original bare name.
    if profile(name) == "round2":
        return META_PATH
    return MODELS_DIR / f"{_prefix(name)}.meta.json"


def profile_report(name: str | None = None):
    """Where a profile run puts its full generated report.

    Never REPORT.md: that file is the hand-written round-3 summary, and a
    re-run of training must not overwrite it. Every profile, including the
    default, writes reports/report_<profile>.md."""
    return REPORTS_DIR / f"report_{profile(name)}.md"


# MediaPipe hand model. Reused from the existing photo model if present,
# otherwise downloaded from Google's official storage bucket.
HAND_MODEL_NAME = "hand_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / HAND_MODEL_NAME
HAND_MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
]
LEGACY_HAND_MODEL_CANDIDATES = [
    ROOT.parent / "sign language photo model" / HAND_MODEL_NAME,
]

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Folders we must never treat as letters.
IGNORED_LABELS = {"nothing", "del", "space", "background", "background_traffic"}
# J and Z are produced by motion, not by a static shape: a single frame is not
# enough to recognise them, so they are excluded from the static model and
# handled by the optional motion stage (`MotionDecider` in src/live.py).
MOTION_LETTERS = {"J", "Z"}
# X is a static shape but is missing from the Massey set; it is kept in the
# label space and we simply have less cross-dataset evidence for it.
LABELS = [c for c in ALL_LETTERS if c not in MOTION_LETTERS]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}

# ---------------------------------------------------------------------------
# Evaluation policy
# ---------------------------------------------------------------------------
# A held-out dataset smaller than this cannot carry a headline number: with
# danrasband's 683 images (~28 per letter) one letter swings the figure by
# several points, so it is reported but flagged, while still being USED IN
# TRAINING for every other fold and for the final model.
HOLDOUT_MIN_FOR_HEADLINE = 2000

# ---------------------------------------------------------------------------
# MediaPipe Hands settings (spec'd values, expressed in the Tasks API)
# ---------------------------------------------------------------------------
# spec: static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
# NOTE: the Tasks API has no `max_tracking_duration_frames` knob, so nothing in
# this project relies on temporal tracking: each frame is decided on its own.

# ---------------------------------------------------------------------------
# Normalisation / feature settings (must match between train and live)
# ---------------------------------------------------------------------------
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
SCALE_EPS = 1e-6
# Canonical hand: samples whose palm cross product indicates the opposite
# chirality are mirrored (x negated) so the model only ever sees one handedness.
# Landmark ids are anatomical, so mirroring does NOT permute them.
# See docs/DETAILS.md.
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
# The real definition lives in src/normalize.py (single source of truth);
# mirror it here for reporting. Override with ASL_FEATURE_BLOCKS=xyz,tip_dists,curl
if __package__:
    from . import normalize as _nz  # noqa: E402
else:
    import normalize as _nz  # noqa: E402

FEATURE_BLOCKS = _nz.FEATURE_BLOCKS


def feature_dim() -> int:
    """Computed from normalize itself, so the two can never disagree."""
    if __package__:
        from . import normalize as normalizer
    else:
        import normalize as normalizer

    return int(normalizer.features_from_normalized_batch(np.zeros((1, 21, 3))).shape[1])



# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry describes where the images live, how a folder name maps to a
# letter label, and how images group into "sessions" (used for the honest
# block-wise splits when only one dataset is available).
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
        # folder layout: asl_alphabet_train/<LETTER>/<LETTER>1.jpg
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
        # folder layout: dataset5/<GROUP>/<letter>/color_<n>_<frame>.png
        # The OUTER folder is NOT the letter - it is one of five capture groups
        # (A-E, i.e. different recording sessions/signers). The INNER folder is
        # the letter, and it is confirmed by the filename: color_<n>_<f>.png
        # where n == ord(letter) - ord('a') for every file we checked.
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

# A fourth source of *photographs*, added 2026-11-20. The three original
# sources are saturated for static letters: grassknoted is 3,000 renders plus
# 26 photographs per letter from one person, danrasband is that same author's
# 30 test photos per letter, and massey is five signers on a Kinect. None of
# them is a modern webcam, which is why letters like N (0.4 % cross-dataset)
# have almost no evidence behind them. This set is 26 letter folders of about
# 300 independently shot photographs, i.e. a fourth camera, crop and signer,
# and it is the cheapest honest way to widen the training mix.
#
# Dropped on purpose: ayuraj/asl-dataset (36 classes x 70 images, CC0) is too
# small to move a 122k-row model, and "sign language mnist" is 28x28 greyscale,
# where MediaPipe cannot even find the hand.
DATASETS.append({
    "id": "rushil",
    "kind": "kaggle",
    "ref": "officialrushil/asl-alphabet-dataset",
    "role": "cross_dataset",
    "license": "Unknown (no licence stated on the Kaggle page) - research use",
    "why": ("about 7,800 independently shot letter photographs (~300 per "
            "letter, 26 folders) that share neither camera, crop nor signer "
            "with the other three sources."),
    # The download ships the letter folders inside an "asl data" directory; the
    # empty entry is the fallback for a repack without it. locate_roots drops
    # the parent when both resolve, so the images are never counted twice.
    "roots": ["asl data", ""],
    "letter_from": "top_folder",
    "session_from": "letter_folder",
})


# ---------------------------------------------------------------------------
# Motion data - the only thing that can ever add J and Z
# ---------------------------------------------------------------------------
# J and Z are not shapes. A single frame of either is a legal static letter (a
# J reads as I, a Z as X at 96.3 % / 94.2 %), which is why the 24-class model
# excludes them. Adding them is therefore a question about STROKES, and no
# photo dataset can answer it:
#
#   msl_dynamic   Zenodo 10.5281/zenodo.14689869, CC-BY-4.0. 1,200 clips of the
#                 six dynamic letters of the *Mexican* manual alphabet
#                 (J K Ñ Q X Z) from 20 signers, split 18 train / 2 test BY
#                 SIGNER. It is the only public, directly downloadable corpus of
#                 real fingerspelling strokes, and it is not ASL: it may fit and
#                 validate the trajectory stage, never the 24-class classifier,
#                 and the ASL claim still needs your own hand -
#                 `python src/live.py --calibrate-jz`.
#   chicagofswild TTIC, 14 GB, 7,304 ASL fingerspelling sequences from 160
#                 signers recorded in the wild. Sequence-level transcriptions
#                 only (no per-letter timing), so nothing in this project
#                 consumes it yet; it is what a CTC / forced-alignment step
#                 would need, and it downloads only when asked for by name.
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
        # ASCII on purpose: this string is printed by --list, and a Windows
        # console in cp1252 turns the Spanish N-tilde into "?".
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


def write_default_meta() -> dict:
    ensure_dirs()
    META_PATH.write_text(json.dumps(META_DEFAULT, indent=2), encoding="utf-8")
    return META_DEFAULT


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
