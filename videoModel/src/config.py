"""Shared paths and vocabulary for the video dataset workflow."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"

WLASL_DIR = DATASETS_DIR / "WLASL"
WLASL_METADATA = WLASL_DIR / "WLASL_v0.3.json"
WLASL_VIDEOS = WLASL_DIR / "videos"

ASL_CITIZEN_DIR = DATASETS_DIR / "ASL_Citizen"
ASL_CITIZEN_ARCHIVE = ASL_CITIZEN_DIR / "ASL_Citizen.zip"
ASL_CITIZEN_METADATA_DIR = ASL_CITIZEN_DIR / "metadata"
ASL_CITIZEN_METADATA = ASL_CITIZEN_METADATA_DIR / "selected_videos.csv"
ASL_CITIZEN_VIDEOS = ASL_CITIZEN_DIR / "videos"

PROCESSED_DIR = DATASETS_DIR / "processed"
NUM_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
ANNOTATION_FPS = 25.0
MIN_ASL_CITIZEN_FREE_BYTES = 35_000_000_000

TARGET_WORDS = (
    "hello",
    "thanks",
    "yes",
    "no",
    "please",
    "help",
    "name",
    "love",
    "sorry",
    "good",
    "bad",
    "eat",
    "drink",
    "home",
    "school",
    "friend",
    "family",
    "happy",
    "sad",
    "bye",
)

WORD_ALIASES = {
    "thank you": "thanks",
}


def canonicalize_gloss(gloss: object) -> str:
    """Normalize a gloss and map known spelling variants to project labels."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(gloss).casefold()).strip()
    return WORD_ALIASES.get(normalized, normalized)


def safe_filename(value: object) -> str:
    """Return a path-safe filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot create a safe filename from {value!r}.")
    return cleaned
