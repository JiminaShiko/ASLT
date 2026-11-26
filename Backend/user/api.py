"""
Flask bridge from the HTML translator to the ASL landmark console pipeline.

Shares the detector, backend, frame gate, smoother, motion rules, and overlay
renderer with asl_landmark_app/src/live.py. Registered on the main app in
Backend/app.py.
"""

import base64
import io
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_PROPAGATE_CXX_STD", "1")

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, send_file, session as flask_session
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "Model"
LANDMARK_APP = MODEL_DIR / "asl_landmark_app"
sys.path.insert(0, str(MODEL_DIR))

from asl_landmark_app.src import config as landmark_config  # noqa: E402
from asl_landmark_app.src import hand_backend, live as landmark_live  # noqa: E402
from asl_landmark_app.src import normalize  # noqa: E402
from videoModel.src import config as words_config  # noqa: E402
from .tracking import (  # noqa: E402
    PoseOverlay, body_points, hand_points, head_payload, packed_hand_points,
)

MAX_SIDE = 640
TTS_MAX_CHARS = 500
TTS_CHUNK_CHARS = 200
TTS_TIMEOUT = 15
TTS_LANG_PATTERN = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,4})?$")
DEFAULT_SETTINGS = {
    "profile": landmark_config.PROFILE_DEFAULT,
    "backend": "torch",
    "conf": 0.70,
    "margin": 0.10,
    "pair_margins": None,
    "vote": 7,
    "hold": 0.5,
    "motion": True,
    "landmarks": True,
    "motion_debug": False,
    "mirror": False,
    "recover": 2,
    "recover_every": 5,
    "recover_conf": 0.3,
    "recover_flip": True,
    "jz": "",
    "calibrate_reps": 3,
}
WORD_DEFAULT_SETTINGS = {
    "fps": 20.0,
    "stride": max(1, int(words_config.LIVE_STRIDE)),
    "buffer": float(words_config.BUFFER_SECONDS),
    "conf": float(words_config.CONFIDENCE_MIN),
    "margin": float(words_config.MARGIN_MIN),
}

api_bp = Blueprint("asl_translation_api", __name__)

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
WORD_SESSIONS = {}
WORD_SESSIONS_LOCK = threading.Lock()
STATIC_DETECTOR_LOCK = threading.Lock()
STATIC_DETECTOR = None
STATIC_RECOVERY = None


@lru_cache(maxsize=8)
def get_backend(name, profile):
    return landmark_live.build_backend(name, profile=profile)


def available_profiles():
    return [
        profile for profile in sorted(landmark_config.PROFILES)
        if landmark_config.model_path(profile).is_file()
        and landmark_config.meta_path(profile).is_file()
    ]


def validate_settings(payload):
    settings = dict(DEFAULT_SETTINGS)
    if payload:
        settings.update(payload)
    if settings["profile"] not in available_profiles():
        raise ValueError(f"Unknown or unavailable model profile: {settings['profile']}")
    if settings["backend"] not in ("torch", "onnx"):
        raise ValueError("backend must be 'torch' or 'onnx'")
    for key, low, high in (
        ("conf", 0.0, 1.0), ("margin", 0.0, 1.0),
        ("recover_conf", 0.0, 1.0),
    ):
        settings[key] = float(settings[key])
        if not low <= settings[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    settings["vote"] = int(settings["vote"])
    settings["hold"] = float(settings["hold"])
    if not 1 <= settings["vote"] <= 60:
        raise ValueError("vote must be between 1 and 60")
    if not 0.0 <= settings["hold"] <= 10.0:
        raise ValueError("hold must be between 0 and 10 seconds")
    settings["recover"] = int(settings["recover"])
    settings["recover_every"] = int(settings["recover_every"])
    settings["calibrate_reps"] = int(settings["calibrate_reps"])
    if settings["recover"] < 0 or settings["recover_every"] < 1:
        raise ValueError("recovery frame counts are invalid")
    if settings["calibrate_reps"] < 1:
        raise ValueError("calibrate_reps must be at least 1")
    pair_spec = settings["pair_margins"]
    settings["pairs"] = (
        landmark_live.resolve_pair_margins(pair_spec)
        if pair_spec is not None else dict(landmark_live.PAIR_MARGINS)
    )
    settings["jz"] = str(settings["jz"] or "")
    settings["jz_thresholds"] = (
        landmark_live.parse_spec(settings["jz"]) if settings["jz"] else {}
    )
    return settings


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def decode_image(data_url_or_b64):
    b64 = data_url_or_b64
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def reduce_image(image):
    width, height = image.size
    longest = max(width, height)
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / longest
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def image_to_bgr(image):
    return cv2.cvtColor(np.asarray(image.convert("RGB"), dtype=np.uint8),
                        cv2.COLOR_RGB2BGR)


def image_to_rgb(image):
    return np.ascontiguousarray(np.asarray(image.convert("RGB"),
                                           dtype=np.uint8))


class LiveSession:
    def __init__(self, raw_settings):
        self.lock = threading.RLock()
        self.owner = None
        self.settings = validate_settings(raw_settings)
        self.backend, self.labels = get_backend(
            self.settings["backend"], self.settings["profile"])
        # Built on the first image frame: landmark frames from the page never need it.
        self.detector = None
        self.recovery = (
            hand_backend.Recovery(
                conf=self.settings["recover_conf"],
                after=self.settings["recover"],
                every=self.settings["recover_every"],
                flip=self.settings["recover_flip"],
            ) if self.settings["recover"] else None
        )
        self.smoother = landmark_live.Smoother(
            conf=self.settings["conf"],
            vote=self.settings["vote"],
            hold=self.settings["hold"],
        )
        self.motion = landmark_live.MotionDecider()
        if self.settings["jz_thresholds"]:
            self.motion.apply_thresholds(
                self.settings["jz_thresholds"],
                source="web --jz " + self.settings["jz"],
            )
        self.text = ""
        self.landmarks = None
        self.held = None
        self.held_conf = 0.0
        self.call = landmark_live.blank_call()
        self.motion_letter = None
        self.fps = 0.0
        self.frames = 0
        self.hands = 0
        self.started_at = time.monotonic()
        self.last_activity = self.started_at
        self.last_frame_ms = 0.0
        # EMA of arrival intervals, so the readout follows the live rate instead
        # of averaging in a slow start for the rest of the session.
        self._last_frame_mono = 0.0
        self.streak = 0
        self.cal = None
        self.cal_result = None
        self.force_next = False
        self.recent_commits = []
        # One note per typed letter: the top-3 the model saw and whether the
        # letter came out of an UNCERTAIN / low-confidence stretch. Returned
        # when the session stops so the page can attach it to history.
        self.commit_notes = []
        self.uncertain_at = {}
        self.started_wall = time.time()
        # Body/face trace for the overlay only; recognition still runs on the
        # hand detector, so enabling it can never change a reading. The pass
        # runs on a worker thread so the answer never waits for it.
        self.pose = PoseOverlay() if self.settings["landmarks"] else None
        self._pose_lock = threading.Lock()
        self._pose_busy = False
        self._pose_payload = {"head": {"lines": [], "dots": []}, "body": []}
        self.overlay = {"hands": [], "head": {"lines": [], "dots": []},
                        "body": []}
        self.last_recovery = ""

    def process_frame(self, image=None, landmarks=None, browser=False):
        with self.lock:
            started = time.perf_counter()
            now_mono = time.monotonic()
            self.last_activity = now_mono
            self.frames += 1
            if self._last_frame_mono:
                interval = now_mono - self._last_frame_mono
                if interval > 0:
                    instant = 1.0 / interval
                    self.fps = (instant if self.fps <= 0 else
                                self.fps * 0.85 + instant * 0.15)
            self._last_frame_mono = now_mono
            rgb = None
            if browser:
                # The page's hand tracker already found the landmarks, so this
                # frame skips decoding, detection and recovery entirely.
                result = hand_backend.HandResult(landmarks, None, 0.0)
                self.streak = 0 if landmarks is not None else self.streak + 1
                self.last_recovery = ""
                # An optional image still feeds the body/face trace only.
                if image is not None and self.pose is not None:
                    rgb = image_to_rgb(reduce_image(image))
            else:
                if self.detector is None:
                    self.detector = hand_backend.create_detector(static=False)
                image = reduce_image(image)
                # Only the overlay needs RGB; recognition converts BGR itself, so
                # skip the extra round trip when the trace is off.
                if self.pose is not None:
                    rgb = image_to_rgb(image)
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    frame = image_to_bgr(image)
                timestamp_ms = int((now_mono - self.started_at) * 1000)
                result, how = hand_backend.detect_frame(
                    self.detector, frame, timestamp_ms, self.recovery,
                    empty_streak=self.streak + 1,
                )
                self.last_recovery = how if how in ("direct_0.3", "flip_0.3") else ""
                self.streak = self.streak + 1 if result.landmarks is None else 0
            self.landmarks = result.landmarks
            call = landmark_live.blank_call()
            now = time.time()
            if result.landmarks is not None:
                self.hands += 1
                probabilities = self.backend(normalize.features(result.landmarks))
                call = landmark_live.frame_call(
                    probabilities, self.labels, self.settings["conf"],
                    self.settings["margin"], pairs=self.settings["pairs"],
                )
            if self.settings["motion"]:
                if result.landmarks is None:
                    self.motion.reset()
                else:
                    shown = call["top3"][0][0] if call["top3"] else None
                    if self.cal is not None and not self.cal.done:
                        motion_letter = self.cal.step(
                            now, result.landmarks, shown, call["conf"])
                        if self.cal.done:
                            self.cal_result = self.cal.result
                            self.cal = None
                    else:
                        motion_letter = self.motion.update(
                            now, result.landmarks, shown, call["conf"])
                    if motion_letter:
                        self.text += motion_letter
                        self.motion_letter = motion_letter
                        self.recent_commits.append(motion_letter)
                        self._note_commit(motion_letter, now, source="motion")
            # The display gates still label a frame UNCERTAIN; they no longer
            # block typing. The highest-probability letter always votes, and the
            # smoother's vote window + hold timer decide when it is written.
            top_letter = call["top3"][0][0] if call["top3"] else None
            if call["top3"] and (call["uncertain"] or call["low_conf"]):
                self.uncertain_at[top_letter] = now
            force, self.force_next = self.force_next, False
            held, typed, held_conf = self.smoother.step(
                top_letter, call["conf"], now, force=force,
                hand_present=result.landmarks is not None, gate=0.0,
            )
            if typed:
                self.text += typed
                self.recent_commits.append(typed)
                self._note_commit(typed, now, call,
                                  confidence=held_conf or call["conf"])
            self.held = held
            self.held_conf = held_conf or call["conf"]
            self.call = call
            self.last_frame_ms = (time.perf_counter() - started) * 1000.0
            self.recent_commits = self.recent_commits[-8:]
            overlay = {
                "hands": ([hand_points(result.landmarks)]
                          if result.landmarks is not None else []),
                "head": {"lines": [], "dots": []},
                "body": [],
            }
            if self.pose is not None and rgb is not None:
                with self._pose_lock:
                    payload = self._pose_payload
                overlay["head"] = payload["head"]
                overlay["body"] = payload["body"]
                self._pose_tick(rgb)
            self.overlay = overlay
            return self.response()

    def _pose_tick(self, rgb):
        """One throttled Holistic pass on a worker thread; the frame never waits."""
        with self._pose_lock:
            if self._pose_busy:
                return
            self._pose_busy = True

        def run():
            try:
                payload = self.pose.update(rgb)
            except Exception:
                payload = None
            with self._pose_lock:
                if payload is not None:
                    self._pose_payload = payload
                self._pose_busy = False

        threading.Thread(target=run, name="pose-overlay", daemon=True).start()

    def _note_commit(self, letter, now, call=None, confidence=0.0,
                     source="classifier"):
        """Record one typed letter: what it was, and what else was possible."""
        call = call or {}
        top = [[str(label), float(probability)]
               for label, probability in (call.get("top3") or [])][:3]
        reason = ""
        if call.get("uncertain"):
            reason = "margin"
        elif call.get("low_conf"):
            reason = "confidence"
        recent_uncertain = self.uncertain_at.get(letter, 0.0) >= now - 2.0
        self.commit_notes.append({
            "letter": str(letter),
            "at": round(now - self.started_wall, 2),
            "guess": bool(source == "classifier"
                          and (reason or recent_uncertain)),
            "source": source,
            "reason": reason,
            "confidence": float(confidence),
            "top": top,
        })
        if len(self.commit_notes) > 120:
            del self.commit_notes[:-120]

    def response(self):
        call = self.call
        return {
            "letter": call["top3"][0][0] if call["top3"] else "-",
            "confidence": float(call["conf"]),
            "hand_found": self.landmarks is not None,
            "held": self.held,
            "held_confidence": float(self.held_conf),
            "text": self.text,
            "top_predictions": [
                {"text": label, "confidence": float(probability)}
                for label, probability in call["top3"]
            ],
            "uncertain": bool(call["uncertain"]),
            "low_conf": bool(call["low_conf"]),
            "margin": float(call["margin"]),
            "required_margin": float(call["need"]),
            "pair": call["pair"] if call["collide"] else "",
            "progress": float(self.smoother.hold_progress(time.time())),
            "fps": float(self.fps),
            "frame_ms": float(self.last_frame_ms),
            "frames": self.frames,
            "hands": self.hands,
            "backend": getattr(self.backend, "name", self.settings["backend"]),
            "model_profile": self.settings["profile"],
            "motion_hint": self.motion.hint() if self.settings["motion"] else "",
            "motion_debug": dict(self.motion.debug),
            "last_motion": self.motion.last,
            "motion_letter": self.motion_letter,
            "recovery": self.recovery.summary() if self.recovery else "disabled",
            "recovered_by": self.last_recovery,
            "thresholds": self.motion.thresholds(),
            "threshold_source": self.motion.threshold_source or "shipped defaults",
            "calibration": (
                {"prompt": self.cal.prompt(), "progress": self.cal.progress()}
                if self.cal is not None and not self.cal.done else None
            ),
            "calibration_result": json_safe(
                dict(self.cal_result, defaults=landmark_live.MotionDecider.DEFAULTS)
                if self.cal_result else None
            ),
            "commits": list(self.recent_commits),
            "overlay": self.overlay,
        }

    def action(self, name):
        with self.lock:
            if name == "commit":
                self.force_next = True
            elif name == "backspace":
                self.text = self.text[:-1]
                if self.recent_commits:
                    self.recent_commits.pop()
                if self.commit_notes:
                    self.commit_notes.pop()
            elif name == "clear":
                self.text = ""
                self.recent_commits.clear()
                self.commit_notes.clear()
            elif name == "calibrate":
                self.cal = landmark_live.JZCalibrator(
                    self.motion, reps=self.settings["calibrate_reps"])
                self.cal_result = None
            elif name == "cancel_calibration":
                if self.cal is not None:
                    self.cal.cancel()
                self.motion.intended = None
                self.cal = None
            else:
                raise ValueError(f"Unknown live action: {name}")
            self.last_activity = time.monotonic()
            return self.response()

    def close(self):
        with self.lock:
            if self.cal is not None and not self.cal.done:
                self.cal.cancel()
            if self.detector is not None:
                self.detector.close()
            if self.recovery is not None:
                self.recovery.close()
            if self.pose is not None:
                self.pose.close()


def _static_recovery():
    """One shared retry detector.

    Building a Recovery per photo loaded a whole MediaPipe model for every
    frame without a hand; that is what made the camera pages lag and heat.
    """
    global STATIC_RECOVERY
    if STATIC_RECOVERY is None:
        STATIC_RECOVERY = hand_backend.Recovery(after=1, every=1)
    return STATIC_RECOVERY


def predict_photo(image, settings):
    global STATIC_DETECTOR
    image = reduce_image(image)
    frame = image_to_bgr(image)
    with STATIC_DETECTOR_LOCK:
        if STATIC_DETECTOR is None:
            STATIC_DETECTOR = hand_backend.create_detector(static=True)
        hand = hand_backend.detect_image(STATIC_DETECTOR, frame)
        if hand.landmarks is None:
            hand, _ = _static_recovery().attempt(frame)
    if hand.landmarks is None:
        return {
            "text": "nothing", "confidence": 1.0, "hand_found": False,
            "top_predictions": [], "uncertain": False, "low_conf": False,
        }
    backend, labels = get_backend(settings["backend"], settings["profile"])
    probabilities = backend(normalize.features(hand.landmarks))
    call = landmark_live.frame_call(
        probabilities, labels, settings["conf"], settings["margin"],
        pairs=settings["pairs"],
    )
    return {
        "text": call["top3"][0][0], "confidence": call["conf"],
        "hand_found": True,
        "top_predictions": [
            {"text": label, "confidence": probability}
            for label, probability in call["top3"]
        ],
        "uncertain": call["uncertain"], "low_conf": call["low_conf"],
        "margin": call["margin"], "required_margin": call["need"],
        "pair": call["pair"] if call["collide"] else "",
        "features": normalize.features(hand.landmarks).tolist(),
    }


def validate_word_settings(payload):
    settings = dict(WORD_DEFAULT_SETTINGS)
    if payload:
        for key in settings:
            if payload.get(key) is not None:
                settings[key] = payload[key]
    settings["fps"] = float(settings["fps"])
    settings["stride"] = int(settings["stride"])
    settings["buffer"] = float(settings["buffer"])
    settings["conf"] = float(settings["conf"])
    settings["margin"] = float(settings["margin"])
    if not 5.0 <= settings["fps"] <= 60.0:
        raise ValueError("fps must be between 5 and 60")
    if not 1 <= settings["stride"] <= 8:
        raise ValueError("stride must be between 1 and 8")
    if not 0.5 <= settings["buffer"] <= 10.0:
        raise ValueError("buffer must be between 0.5 and 10 seconds")
    for key in ("conf", "margin"):
        if not 0.0 <= settings[key] <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    return settings


def word_model_info():
    weights = Path(words_config.FOR_RELEASE)
    return {
        "available": weights.is_file(),
        "weights": weights.name,
        "default_settings": WORD_DEFAULT_SETTINGS,
    }


class WordLiveSession:
    """One live word-recognition session over the shared camera frames."""

    def __init__(self, raw_settings):
        from videoModel.src import predictor as word_predictor

        self.lock = threading.RLock()
        self.owner = None
        self.settings = validate_word_settings(raw_settings)
        self.predictor = word_predictor.WordPredictor(
            weights=Path(words_config.FOR_RELEASE),
            fps=self.settings["fps"],
            stride=self.settings["stride"],
            buffer_seconds=self.settings["buffer"],
            min_conf=self.settings["conf"],
            min_margin=self.settings["margin"],
        )
        self.started_at = time.monotonic()
        self.last_activity = self.started_at
        self.frames = 0
        self.hands = 0
        self.last_frame_ms = 0.0
        self.last_result = self.predictor.result()
        self.overlay = {"hands": [], "head": {"lines": [], "dots": []},
                        "body": []}

    def process_frame(self, image):
        with self.lock:
            started = time.perf_counter()
            self.last_activity = time.monotonic()
            rgb = image_to_rgb(image)
            result = self.predictor.update(rgb)
            self.frames += 1
            if any(result.get("hand_in_frame") or ()):
                self.hands += 1
            self.last_frame_ms = (time.perf_counter() - started) * 1000.0
            self.last_result = result
            row = self.predictor.latest_row()
            self.overlay = {
                "hands": packed_hand_points(row),
                "head": head_payload(row),
                "body": body_points(row),
            }
            return self.response()

    def response(self):
        elapsed = time.monotonic() - self.started_at
        out = dict(self.last_result)
        out.update({
            "model": "words",
            "fps": float(self.frames / elapsed) if elapsed > 0 else 0.0,
            "frame_ms": float(self.last_frame_ms),
            "frames": self.frames,
            "hands": self.hands,
            "backend": "torch",
            "model_profile": "words",
            "overlay": self.overlay,
        })
        return out

    def action(self, name):
        with self.lock:
            if name != "clear":
                raise ValueError(f"Unknown word live action: {name}")
            self.predictor.reset()
            self.last_activity = time.monotonic()
            self.last_result = self.predictor.result()
            return self.response()

    def close(self):
        # Dropping the session first keeps stop instant: MediaPipe's native
        # close can block for tens of seconds, so it finishes in the background.
        with self.lock:
            predictor, self.predictor = self.predictor, None
        if predictor is not None:
            threading.Thread(target=predictor.close, daemon=True).start()


def tts_chunks(text, limit=TTS_CHUNK_CHARS):
    chunks = []
    current = ""
    for word in text.split():
        if len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                word[index:index + limit]
                for index in range(0, len(word), limit)
            )
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current += " " + word
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


@lru_cache(maxsize=16)
def synthesize_speech(text, lang, slow):
    audio = b""
    for chunk in tts_chunks(text):
        query = urllib.parse.urlencode({
            "ie": "UTF-8", "q": chunk, "tl": lang, "client": "tw-ob",
            "ttsspeed": "0.3" if slow else "1",
        })
        request_url = "https://translate.google.com/translate_tts?" + query
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://translate.google.com/",
        }
        with urllib.request.urlopen(
            urllib.request.Request(request_url, headers=headers),
            timeout=TTS_TIMEOUT,
        ) as response:
            audio += response.read()
    if not audio:
        raise RuntimeError("Google returned an empty audio response.")
    return audio


@api_bp.post("/api/tts")
def text_to_speech():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > TTS_MAX_CHARS:
        return jsonify({
            "error": f"Text is too long (maximum {TTS_MAX_CHARS} characters)."
        }), 400
    lang = str(data.get("lang") or "en").strip().lower()
    if not TTS_LANG_PATTERN.match(lang):
        return jsonify({
            "error": "lang must be a language code such as 'en' or 'en-gb'."
        }), 400
    slow = bool(data.get("slow", False))
    try:
        audio = synthesize_speech(text, lang, slow)
    except urllib.error.HTTPError as error:
        return jsonify({
            "error": f"Google TTS refused the request (HTTP {error.code})."
        }), 502
    except Exception as error:
        return jsonify({"error": f"Google TTS failed: {error}"}), 502
    return send_file(
        io.BytesIO(audio), mimetype="audio/mpeg",
        download_name="speech.mp3", max_age=0,
    )


@api_bp.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "profiles": available_profiles(),
        "default_profile": landmark_config.PROFILE_DEFAULT,
        "labels": landmark_config.LABELS,
        "pair_margins": landmark_live.PAIR_MARGINS,
        "default_settings": DEFAULT_SETTINGS,
        "word_model": word_model_info(),
    })


@api_bp.post("/api/predict-photo")
def predict_photo_route():
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "No image provided."}), 400
        settings = validate_settings(data.get("settings"))
        result = predict_photo(decode_image(data["image"]), settings)
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@api_bp.post("/api/live/start")
def live_start():
    try:
        data = request.get_json(silent=True) or {}
        session = LiveSession(data.get("settings"))
        session.owner = flask_session.get("user_id")
        session_id = uuid.uuid4().hex
        with SESSIONS_LOCK:
            now = time.monotonic()
            expired = [
                key for key, value in SESSIONS.items()
                if now - value.last_activity > 600
            ]
            for key in expired:
                SESSIONS.pop(key).close()
            SESSIONS[session_id] = session
        return jsonify({"session_id": session_id, "state": session.response()})
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@api_bp.post("/api/live/frame")
def live_frame():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "Session expired. Start live translation again."}), 404
    if session.owner != flask_session.get("user_id"):
        return jsonify({"error": "This live session belongs to another user."}), 403
    if "landmarks" in data:
        landmarks = data.get("landmarks")
        if landmarks is None:
            pass  # no hand in this frame
        else:
            try:
                landmarks = np.asarray(landmarks, dtype=np.float32)
            except (TypeError, ValueError):
                landmarks = None
            if (landmarks is None or landmarks.shape != (21, 3)
                    or not np.isfinite(landmarks).all()):
                return jsonify({"error": "landmarks must be 21 [x, y, z] points"}), 400
        image = decode_image(data["image"]) if "image" in data else None
        try:
            return jsonify(session.process_frame(
                image=image, landmarks=landmarks, browser=True))
        except Exception as error:
            return jsonify({"error": str(error)}), 500
    if "image" not in data:
        return jsonify({"error": "image or landmarks required"}), 400
    try:
        return jsonify(session.process_frame(decode_image(data["image"])))
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@api_bp.post("/api/live/action")
def live_action():
    data = request.get_json(silent=True) or {}
    with SESSIONS_LOCK:
        session = SESSIONS.get(data.get("session_id"))
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    try:
        return jsonify(session.action(data.get("action", "")))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@api_bp.post("/api/live/clear")
def live_clear():
    data = request.get_json(silent=True) or {}
    with SESSIONS_LOCK:
        session = SESSIONS.get(data.get("session_id"))
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session.action("clear"))


@api_bp.post("/api/live/stop")
def live_stop():
    data = request.get_json(silent=True) or {}
    with SESSIONS_LOCK:
        session = SESSIONS.pop(data.get("session_id"), None)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    text = session.text
    notes = list(session.commit_notes)
    session.close()
    return jsonify({"text": text, "notes": notes})


@api_bp.post("/api/words/live/start")
def words_live_start():
    try:
        data = request.get_json(silent=True) or {}
        session = WordLiveSession(data.get("settings"))
        session.owner = flask_session.get("user_id")
        session_id = uuid.uuid4().hex
        with WORD_SESSIONS_LOCK:
            now = time.monotonic()
            expired = [
                key for key, value in WORD_SESSIONS.items()
                if now - value.last_activity > 600
            ]
            for key in expired:
                WORD_SESSIONS.pop(key).close()
            WORD_SESSIONS[session_id] = session
        return jsonify({"session_id": session_id, "state": session.response()})
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@api_bp.post("/api/words/live/frame")
def words_live_frame():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    with WORD_SESSIONS_LOCK:
        session = WORD_SESSIONS.get(session_id)
    if session is None:
        return jsonify(
            {"error": "Word session expired. Start live translation again."}
        ), 404
    if session.owner != flask_session.get("user_id"):
        return jsonify({"error": "This live session belongs to another user."}), 403
    if "image" not in data:
        return jsonify({"error": "image required"}), 400
    try:
        return jsonify(session.process_frame(decode_image(data["image"])))
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@api_bp.post("/api/words/live/action")
def words_live_action():
    data = request.get_json(silent=True) or {}
    with WORD_SESSIONS_LOCK:
        session = WORD_SESSIONS.get(data.get("session_id"))
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    try:
        return jsonify(session.action(data.get("action", "")))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@api_bp.post("/api/words/live/stop")
def words_live_stop():
    data = request.get_json(silent=True) or {}
    with WORD_SESSIONS_LOCK:
        session = WORD_SESSIONS.pop(data.get("session_id"), None)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    text = session.last_result.get("text", "")
    session.close()
    return jsonify({"text": text})
