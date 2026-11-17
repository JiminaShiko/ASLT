"""
Flask bridge from the HTML translator to the ASL landmark console pipeline.

Run from this folder:
    python api.py

The live API uses the same detector, model backend, frame gate, smoother,
motion rules, and overlay renderer as asl_landmark_app/src/live.py.
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
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

LANDMARK_APP = Path(__file__).resolve().parents[1] / "asl_landmark_app"
LANDMARK_SRC = LANDMARK_APP / "src"
sys.path.insert(0, str(LANDMARK_SRC))

import config as landmark_config  # type: ignore[reportMissingImports]  # noqa: E402
import hand_backend  # type: ignore[reportMissingImports]  # noqa: E402
import live as landmark_live  # type: ignore[reportMissingImports]  # noqa: E402
import normalize  # type: ignore[reportMissingImports]  # noqa: E402

PORT = 5000
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
    "recover": 1,
    "recover_every": 3,
    "recover_conf": 0.3,
    "recover_flip": True,
    "jz": "",
    "calibrate_reps": 3,
}

app = Flask(__name__)
CORS(app, origins=[
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.1:5500", "http://localhost:5500", "null",
])

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
STATIC_DETECTOR_LOCK = threading.Lock()
STATIC_DETECTOR = None


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


def render_frame(frame, session, call, held, held_conf, committed, now, fps, how):
    display = frame.copy()
    if session.settings["landmarks"] and session.landmarks is not None:
        hand_backend.draw_landmarks(display, session.landmarks[:, :2])
    if session.settings["mirror"]:
        display = cv2.flip(display, 1)
    pending = session.smoother.hold_progress(now)
    landmark_live.render(
        display, held, held_conf or call["conf"], committed, session.text,
        fps, session.landmarks is not None, pending, session.settings["hold"],
        getattr(session.backend, "name", session.settings["backend"]),
        top3=call["top3"], uncertain=call["uncertain"],
        margin=call["margin"], need=call["need"],
    )
    if call["top3"] and (call["low_conf"] or call["close"]):
        label, probability = call["top3"][0]
        reason = (
            f"below the {session.settings['conf']:.2f} confidence gate"
            if call["low_conf"] else
            f"only {call['margin'] * 100:.0f} pts ahead of "
            f"{call['top3'][1][0]}" +
            (f", {call['pair'][0]}/{call['pair'][1]} needs "
             f"{call['need'] * 100:.0f}" if call["collide"] else "")
        )
        cv2.putText(display, f"no vote: {label} {probability * 100:.0f}%, {reason}",
                    (14, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
    if how in ("direct_0.3", "flip_0.3"):
        cv2.putText(display, f"hand recovered ({how})", (14, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1,
                    cv2.LINE_AA)
    if session.settings["motion"]:
        if session.settings["motion_debug"]:
            debug = session.motion.debug or {}
            line = (
                f"J/Z held={debug.get('held', 0):.2f}s gate={debug.get('gate', '?')} "
                f"model={debug.get('top1') or '-'} {debug.get('p1', 0):.2f} "
                f"idx={debug.get('idx_ext', 0):.2f} "
                f"wrist(path={debug.get('wrist_path', 0):.2f} "
                f"rev={debug.get('wrist_rev', 0)}) "
                f"tip(path={debug.get('tip_path', 0):.2f} "
                f"rev={debug.get('tip_rev', 0)})"
            ) if debug else (
                "J/Z: neither gate met (index-up shape, or a reading that was "
                "not I / not an unsure X)"
            )
            cv2.putText(display, line, (14, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(display, f"last motion letter: {session.motion.last or '-'}",
                        (14, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 200, 255), 1, cv2.LINE_AA)
        hint = session.motion.hint()
        if hint:
            y = 166 if session.settings["motion_debug"] else 126
            cv2.putText(display, session.motion.hint(width=62) + "  [--calibrate-jz]", (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 165, 255), 1, cv2.LINE_AA)
    if session.cal is not None and not session.cal.done:
        height, width = display.shape[:2]
        cv2.rectangle(display, (0, height - 124), (width, height - 92),
                      (0, 0, 0), -1)
        cv2.putText(display, session.cal.prompt()[:104], (12, height - 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2,
                    cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Could not encode the rendered camera frame.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


class LiveSession:
    def __init__(self, raw_settings):
        self.lock = threading.RLock()
        self.settings = validate_settings(raw_settings)
        self.backend, self.labels = get_backend(
            self.settings["backend"], self.settings["profile"])
        self.detector = hand_backend.create_detector(static=False)
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
        self.streak = 0
        self.cal = None
        self.cal_result = None
        self.force_next = False
        self.recent_commits = []
        self.frame_image = None
        self.last_recovery = ""

    def process_frame(self, image):
        with self.lock:
            started = time.perf_counter()
            self.last_activity = time.monotonic()
            self.frames += 1
            image = reduce_image(image)
            frame = image_to_bgr(image)
            timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
            result, how = hand_backend.detect_frame(
                self.detector, frame, timestamp_ms, self.recovery,
                empty_streak=self.streak + 1,
            )
            self.last_recovery = how if how in ("direct_0.3", "flip_0.3") else ""
            self.streak = self.streak + 1 if result.landmarks is None else 0
            self.landmarks = result.landmarks
            call = landmark_live.blank_call()
            now = time.time()
            committed = None
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
                        committed = motion_letter
                        self.motion_letter = motion_letter
                        self.recent_commits.append(motion_letter)
            force, self.force_next = self.force_next, False
            held, typed, held_conf = self.smoother.step(
                call["letter"], call["conf"], now, force=force,
                hand_present=result.landmarks is not None,
            )
            if typed:
                self.text += typed
                committed = typed
                self.recent_commits.append(typed)
            self.held = held
            self.held_conf = held_conf or call["conf"]
            self.call = call
            elapsed = time.monotonic() - self.started_at
            self.fps = self.frames / elapsed if elapsed > 0 else 0.0
            self.last_frame_ms = (time.perf_counter() - started) * 1000.0
            self.recent_commits = self.recent_commits[-8:]
            self.frame_image = render_frame(
                frame, self, call, held, self.held_conf, committed, now,
                self.fps, how,
            )
            return self.response()

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
            "frame_image": self.frame_image,
        }

    def action(self, name):
        with self.lock:
            if name == "commit":
                self.force_next = True
            elif name == "backspace":
                self.text = self.text[:-1]
                if self.recent_commits:
                    self.recent_commits.pop()
            elif name == "clear":
                self.text = ""
                self.recent_commits.clear()
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
            self.detector.close()
            if self.recovery is not None:
                self.recovery.close()


def predict_photo(image, settings):
    global STATIC_DETECTOR
    image = reduce_image(image)
    frame = image_to_bgr(image)
    with STATIC_DETECTOR_LOCK:
        if STATIC_DETECTOR is None:
            STATIC_DETECTOR = hand_backend.create_detector(static=True)
        hand = hand_backend.detect_image(STATIC_DETECTOR, frame)
        if hand.landmarks is None:
            recovery = hand_backend.Recovery(after=1, every=1)
            try:
                hand, _ = recovery.attempt(frame)
            finally:
                recovery.close()
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


@app.post("/api/tts")
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


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "profiles": available_profiles(),
        "default_profile": landmark_config.PROFILE_DEFAULT,
        "labels": landmark_config.LABELS,
        "pair_margins": landmark_live.PAIR_MARGINS,
        "default_settings": DEFAULT_SETTINGS,
    })


@app.post("/api/predict-photo")
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


@app.post("/api/live/start")
def live_start():
    try:
        data = request.get_json(silent=True) or {}
        session = LiveSession(data.get("settings"))
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


@app.post("/api/live/frame")
def live_frame():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "Session expired. Start live translation again."}), 404
    if "image" not in data:
        return jsonify({"error": "image required"}), 400
    try:
        return jsonify(session.process_frame(decode_image(data["image"])))
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/live/action")
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


@app.post("/api/live/clear")
def live_clear():
    data = request.get_json(silent=True) or {}
    with SESSIONS_LOCK:
        session = SESSIONS.get(data.get("session_id"))
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session.action("clear"))


@app.post("/api/live/stop")
def live_stop():
    data = request.get_json(silent=True) or {}
    with SESSIONS_LOCK:
        session = SESSIONS.pop(data.get("session_id"), None)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    text = session.text
    session.close()
    return jsonify({"text": text})


if __name__ == "__main__":
    print(f"\nASL landmark API running on http://127.0.0.1:{PORT}")
    print("Serving the same live.py frame gate, smoother, J/Z rules, and overlay.")
    print("Keep this terminal open while using the website.\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
