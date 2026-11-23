"""
Text -> sign plan for the "Text to Sign" panel.

The panel needs, for every word the user types, either the sign of that word
(the word model's vocabulary, Model/videoModel) or the letter-by-letter
fingerspelling (the alphabet model, Model/asl_landmark_app). This module is the
lookup that decides which one, and it hands the browser a media URL for each
item. Nothing here runs a model: no torch, no MediaPipe, no TensorFlow, so the
route works even when the weights are not loaded.

Where the picture comes from, in this order:
  1. a file the project owns, Front-end/text_to_sign_demo/word/<word>.*
     (mp4 / webm / gif / png / jpg / webp) and, for letters,
     Front-end/text_to_sign_demo/alphabet/<A-Z>.*;
  2. for a trained word with no owned file, the shortest cached ASL Citizen
     clip for that word (Model/videoModel/cache/manifest.csv), read-only from
     Model/videoModel/ASL_Citizen/videos.
A word with neither is returned as fingerspelling, so the answer is never an
invented sign.

Speech is not implemented here: the panel calls the existing /api/tts route
(Backend/user/api.py), which reads Google's voice.
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading
import urllib.parse
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
VIDEO_MODEL_DIR = PROJECT_ROOT / "Model" / "videoModel"
VOCAB_JSON = VIDEO_MODEL_DIR / "vocab.json"
MANIFEST_CSV = VIDEO_MODEL_DIR / "cache" / "manifest.csv"
SIGNS_DIR = PROJECT_ROOT / "Front-end" / "text_to_sign_demo" / "alphabet"
WORD_SIGNS_DIR = PROJECT_ROOT / "Front-end" / "text_to_sign_demo" / "word"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MEDIA_EXTS = (".mp4", ".webm", ".gif", ".webp", ".png", ".jpg", ".jpeg")
VIDEO_EXTS = {".mp4", ".webm"}

# Browsers decode these inside MP4; older clips in the dataset that are
# MPEG-4 Part 2 (mp4v) fail in Chrome/Edge and would show as a text card.
PLAYABLE_VIDEO_CODECS = {"avc1", "avc3", "vp08", "vp09", "av01"}

# English written forms that are the same sign in the word model's vocabulary.
# Only consulted when the target word really is in vocab.json.
ALIASES = {
    "i": "me",
    "im": "me",
    "thank": "thanks",
    "thankyou": "thanks",
    "okay": "ok",
    "alright": "ok",
}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

text_to_sign_bp = Blueprint("text_to_sign_api", __name__)

_LOCK = threading.Lock()
_CATALOG = None
_CATALOG_KEY = None


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def _videos_dir() -> Path:
    """Same folder contract as Model/videoModel/src/config.py."""
    custom = os.environ.get("ASL_WORDS_DATASET_DIR", "").strip()
    return Path(custom) / "videos" if custom else VIDEO_MODEL_DIR / "ASL_Citizen" / "videos"


def _vocab_words() -> list[str]:
    if not VOCAB_JSON.is_file():
        return []
    try:
        raw = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    words = []
    for item in raw.get("words", []):
        word = str(item.get("word", "")).strip().lower()
        if word:
            words.append(word)
    return words


def _boxes(handle, start: int, end: int):
    """Yield (type, body_start, limit) for the MP4 boxes between start and end."""
    pos = start
    while pos + 8 <= end:
        handle.seek(pos)
        header = handle.read(8)
        if len(header) < 8:
            return
        size = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        body = pos + 8
        if size == 1:
            extended = handle.read(8)
            if len(extended) < 8:
                return
            size = int.from_bytes(extended, "big")
            body = pos + 16
        elif size == 0:
            size = end - pos
        if size < 8:
            return
        yield kind, body, min(pos + size, end)
        pos += size


def _find_box(handle, start: int, end: int, path: list[bytes], depth: int = 0):
    for kind, body, limit in _boxes(handle, start, end):
        if kind != path[0]:
            continue
        if len(path) == 1:
            return body, limit
        if depth < 6:
            found = _find_box(handle, body, limit, path[1:], depth + 1)
            if found is not None:
                return found
    return None


_CODECS: dict[str, str] = {}


def _video_codec(path: Path) -> str:
    """Video codec tag ('avc1', 'mp4v', ...) from the MP4 sample description."""
    key = str(path)
    if key in _CODECS:
        return _CODECS[key]
    codec = ""
    try:
        with path.open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            found = _find_box(
                handle, 0, end, [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"]
            )
            if found is not None:
                body, limit = found
                handle.seek(body)
                data = handle.read(min(16, max(limit - body, 0)))
                if len(data) >= 16:
                    codec = data[12:16].decode("latin-1").strip("\x00 ").lower()
    except OSError:
        codec = ""
    _CODECS[key] = codec
    return codec


def _browser_playable(path: Path) -> bool:
    """True when the browser can decode this file as-is."""
    suffix = path.suffix.lower()
    if suffix in {".webm", ".gif"}:
        return True
    if suffix not in {".mp4", ".mov", ".m4v"}:
        return True
    return _video_codec(path) in PLAYABLE_VIDEO_CODECS


def _word_clips() -> dict[str, list[tuple[int, Path]]]:
    """word -> cached clips of that word, shortest first."""
    if not MANIFEST_CSV.is_file():
        return {}
    videos = _videos_dir()
    found: dict[str, dict[Path, int]] = {}
    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).strip() != "ok":
                continue
            word = str(row.get("word", "")).strip().lower()
            name = str(row.get("video", "")).strip()
            if not word or not name:
                continue
            path = videos / name
            if not path.is_file():
                continue
            try:
                frames = int(float(row.get("frames") or 0))
            except ValueError:
                frames = 0
            per_word = found.setdefault(word, {})
            if path not in per_word or frames < per_word[path]:
                per_word[path] = frames
    return {
        word: sorted(((frames, path) for path, frames in clips.items()), key=lambda item: (item[0], item[1].name))
        for word, clips in found.items()
    }


def _asset(directory: Path, stem: str) -> Path | None:
    for candidate in dict.fromkeys((stem, stem.lower(), stem.upper())):
        for ext in MEDIA_EXTS:
            path = directory / f"{candidate}{ext}"
            if path.is_file():
                return path
    return None


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _catalog_key():
    return (
        _mtime(VOCAB_JSON),
        _mtime(MANIFEST_CSV),
        _mtime(SIGNS_DIR),
        _mtime(WORD_SIGNS_DIR),
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
def _build_catalog() -> dict:
    vocab = _vocab_words()
    clips = _word_clips()
    words: dict[str, dict] = {}
    for word in vocab:
        path = _asset(WORD_SIGNS_DIR, word)
        source = "asset"
        if path is None:
            path = None
            source = "clip"
            for _frames, candidate in clips.get(word, ()):
                # Skip clips the browser cannot decode (e.g. MPEG-4 Part 2).
                if _browser_playable(candidate):
                    path = candidate
                    break
        if path is not None:
            words[word] = {"path": path, "source": source}
    letters = {}
    for letter in ALPHABET:
        path = _asset(SIGNS_DIR, letter)
        if path is not None:
            letters[letter] = {"path": path, "source": "asset"}
    return {"words": words, "letters": letters, "vocab": vocab}


def catalog(force: bool = False) -> dict:
    global _CATALOG, _CATALOG_KEY
    with _LOCK:
        key = _catalog_key()
        if force or _CATALOG is None or key != _CATALOG_KEY:
            _CATALOG = _build_catalog()
            _CATALOG_KEY = key
        return _CATALOG


def _media_url(kind: str, name: str) -> str:
    query = urllib.parse.urlencode({"kind": kind, "name": name})
    return f"/api/text-to-sign/media?{query}"


def _media(entry: dict | None, kind: str, name: str) -> dict | None:
    if entry is None:
        return None
    return {
        "type": "video" if entry["path"].suffix.lower() in VIDEO_EXTS else "image",
        "url": _media_url(kind, name),
        "source": entry["source"],
    }


def _letters(token: str, cat: dict) -> list[dict]:
    return [
        {
            "letter": letter,
            "media": _media(cat["letters"].get(letter), "letter", letter),
        }
        for letter in token.upper()
        if letter in ALPHABET
    ]


def plan(text: str) -> dict:
    cat = catalog()
    items: list[dict] = []
    notes: list[str] = []
    words_signed = 0
    words_spelled = 0
    tokens = list(TOKEN_RE.finditer(text))
    skipped = len(re.sub(r"[A-Za-z\s']", "", text))

    for index, match in enumerate(tokens):
        token = match.group(0)
        key = token.lower().replace("'", "")
        word = key if key in cat["words"] or key in cat["vocab"] else ALIASES.get(key, key)
        entry = cat["words"].get(word)
        if entry is not None:
            words_signed += 1
            items.append(
                {
                    "type": "word",
                    "text": token,
                    "word": word,
                    "sign": "word",
                    "media": _media(entry, "word", word),
                    "letters": [],
                }
            )
        else:
            words_spelled += 1
            if word in cat["vocab"]:
                notes.append(
                    f"No browser-playable sign clip is available for '{token}', so it is fingerspelled."
                )
            items.append(
                {
                    "type": "word",
                    "text": token,
                    "word": word,
                    "sign": "fingerspell",
                    "media": None,
                    "letters": _letters(word, cat),
                }
            )
        if index < len(tokens) - 1:
            items.append({"type": "space"})

    return {
        "text": text,
        "items": items,
        "summary": {
            "words_signed": words_signed,
            "words_fingerspelled": words_spelled,
            "letters": sum(len(item.get("letters") or []) for item in items),
            "words_with_media": len(cat["words"]),
            "vocabulary": len(cat["vocab"]),
            "letter_images": len(cat["letters"]),
            "skipped_characters": skipped,
        },
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@text_to_sign_bp.post("/api/text-to-sign")
def text_to_sign():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 200:
        return jsonify({"error": "Text is too long (maximum 200 characters)."}), 400
    try:
        return jsonify(plan(text))
    except Exception as error:  # noqa: BLE001 - reported to the panel
        return jsonify({"error": f"Could not build the sign plan: {error}"}), 500


@text_to_sign_bp.get("/api/text-to-sign/catalog")
def text_to_sign_catalog():
    cat = catalog(force=request.args.get("refresh") == "1")
    return jsonify(
        {
            "vocabulary": len(cat["vocab"]),
            "words": sorted(cat["words"]),
            "words_with_media": len(cat["words"]),
            "letter_images": sorted(cat["letters"]),
            "sources": {
                "word_media": str(WORD_SIGNS_DIR),
                "letter_media": str(SIGNS_DIR),
                "clips": str(_videos_dir()),
            },
        }
    )


@text_to_sign_bp.get("/api/text-to-sign/media")
def text_to_sign_media():
    kind = str(request.args.get("kind") or "word").strip().lower()
    name = str(request.args.get("name") or "").strip()
    cat = catalog()
    if kind == "word":
        entry = cat["words"].get(name.lower())
    elif kind == "letter":
        entry = cat["letters"].get(name.upper())
    else:
        entry = None
    if entry is None:
        return jsonify({"error": "No sign media for that item."}), 404
    return send_file(entry["path"], conditional=True, max_age=0)
