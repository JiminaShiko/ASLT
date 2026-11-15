"""
Flask API for the ASL landmark model used by HTML/translation.html.

Endpoints
---------
GET  /api/health
POST /api/predict-photo          single image
POST /api/live/start             start a live session
POST /api/live/frame             send a frame
POST /api/live/clear             clear typed text
POST /api/live/stop              stop session and return final text

Run (from this folder):
    pip install -r requirements.txt
    python api.py
"""

import base64
import io
import json
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps

LANDMARK_APP = Path(__file__).resolve().parents[1] / "asl_landmark_app"
LANDMARK_SRC = LANDMARK_APP / "src"
sys.path.insert(0, str(LANDMARK_SRC))

import config as landmark_config  # noqa: E402
import hand_backend  # noqa: E402
import model as landmark_model  # noqa: E402
import normalize  # noqa: E402

# ---------------------------------------------------------------------------
# Model and detector
# ---------------------------------------------------------------------------
PROFILE = landmark_config.profile()
META = json.loads(landmark_config.meta_path(PROFILE).read_text(encoding="utf-8"))
LABELS = META["labels"]
MODEL_META = META["model"]
feature_dim = normalize.feature_dim()
if feature_dim != int(MODEL_META["in_dim"]):
    raise RuntimeError(
        f"Landmark feature mismatch: model expects {MODEL_META['in_dim']} "
        f"features, but normalize.features() produces {feature_dim}."
    )

MODEL_PATH = landmark_config.model_path(PROFILE, must_exist=True)
print(f"Loading landmark model ({PROFILE}) from: {MODEL_PATH}")
model = landmark_model.load_model(
    MODEL_PATH,
    int(MODEL_META["in_dim"]),
    len(LABELS),
    tuple(MODEL_META["hidden"]),
    float(MODEL_META["dropout"]),
)
model.eval()
print(f"Landmark model loaded. Classes: {len(LABELS)}")

MIN_CONFIDENCE = 0.60
STABLE_FRAMES = 4
DETECT_MAX_SIDE = 640
NOTHING_TICKS = 2
PORT = 5000

app = Flask(__name__)
CORS(app, origins=[
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.1:5500", "http://localhost:5500", "null",
])

MODEL_LOCK = threading.Lock()
STATIC_DETECTOR_LOCK = threading.Lock()
STATIC_DETECTOR = hand_backend.create_detector(static=True)
STATIC_RECOVERY = hand_backend.Recovery(after=1, every=1)
LIVE_SESSIONS = {}
LIVE_SESSIONS_LOCK = threading.Lock()


def predict_probabilities(features):
    batch = np.ascontiguousarray(features[None, :], dtype=np.float32)
    with MODEL_LOCK, torch.inference_mode():
        logits = model(torch.from_numpy(batch))
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def apply_to_text(text, name):
    return text + name


def decode_image(data_url_or_b64):
    b64 = data_url_or_b64
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def detect_static(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with STATIC_DETECTOR_LOCK:
        result = hand_backend.detect_image(STATIC_DETECTOR, bgr)
        if result.landmarks is None:
            result, _ = STATIC_RECOVERY.attempt(bgr)
    return result.landmarks


# ---------------------------------------------------------------------------
# Live sessions
# ---------------------------------------------------------------------------
class LiveSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.detector = hand_backend.create_detector(static=False)
        self.recovery = hand_backend.Recovery(after=1, every=3)
        self.letter = "-"
        self.confidence = 0.0
        self.hand_found = False
        self.top_predictions = []
        self.recent = deque(maxlen=STABLE_FRAMES)
        self.last_added = None
        self.misses = 0
        self.text = ""
        self.prob_ema = None
        self.last_activity = time.monotonic()
        self.started_at = time.monotonic()

    def _response(self):
        return {
            "letter": self.letter,
            "confidence": float(self.confidence),
            "hand_found": bool(self.hand_found),
            "text": self.text,
            "top_predictions": self.top_predictions,
            "processed_image": None,
        }

    def process_frame(self, image, left_hand):
        with self.lock:
            self.last_activity = time.monotonic()
            if left_hand:
                image = ImageOps.mirror(image)

            width, height = image.size
            longest = max(width, height)
            if longest > DETECT_MAX_SIDE:
                scale = DETECT_MAX_SIDE / longest
                image = image.resize(
                    (int(width * scale), int(height * scale)),
                    Image.Resampling.LANCZOS,
                )

            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
            hand, _ = hand_backend.detect_frame(
                self.detector,
                bgr,
                timestamp_ms,
                recovery=self.recovery,
                empty_streak=self.misses + 1,
            )

            if hand.landmarks is None:
                self.misses += 1
                self.letter = "nothing"
                self.confidence = 1.0
                self.hand_found = False
                self.top_predictions = []
                self.recent.append("?")
                if self.misses >= NOTHING_TICKS:
                    self.last_added = None
                    self.prob_ema = None
                return self._response()

            self.misses = 0
            self.hand_found = True
            probabilities = predict_probabilities(normalize.features(hand.landmarks))
            if self.prob_ema is None:
                self.prob_ema = probabilities
            else:
                old_idx = int(np.argmax(self.prob_ema))
                new_idx = int(np.argmax(probabilities))
                if new_idx != old_idx and probabilities[new_idx] >= MIN_CONFIDENCE:
                    self.prob_ema = probabilities
                else:
                    self.prob_ema = 0.6 * self.prob_ema + 0.4 * probabilities

            probabilities = self.prob_ema
            idx = int(np.argmax(probabilities))
            self.letter = LABELS[idx]
            self.confidence = float(probabilities[idx])
            self.top_predictions = [
                {"text": LABELS[i], "confidence": float(probabilities[i])}
                for i in np.argsort(probabilities)[::-1][:5]
            ]
            self.recent.append(self.letter if self.confidence >= MIN_CONFIDENCE else "?")

            stable = (
                self.recent[0]
                if len(self.recent) == STABLE_FRAMES
                and len(set(self.recent)) == 1
                else None
            )
            if stable is not None and stable != "?" and stable != self.last_added:
                self.text = apply_to_text(self.text, stable)
                self.last_added = stable

            return self._response()

    def close(self):
        with self.lock:
            self.detector.close()
            self.recovery.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model": f"asl_landmark_{PROFILE}",
        "classes": len(LABELS),
        "labels": LABELS,
    })


@app.route("/api/predict-photo", methods=["POST"])
def predict_photo():
    """Single-image prediction (Upload Media tab)."""
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "No image provided."}), 400
        image = decode_image(data["image"])
        landmarks = detect_static(image)
        if landmarks is None:
            return jsonify({"text": "nothing", "confidence": 1.0, "hand_found": False})
        probabilities = predict_probabilities(normalize.features(landmarks))
        idx = int(np.argmax(probabilities))
        return jsonify({
            "text": LABELS[idx],
            "confidence": float(probabilities[idx]),
            "hand_found": True,
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/api/live/start", methods=["POST"])
def live_start():
    now = time.monotonic()
    with LIVE_SESSIONS_LOCK:
        expired = [
            session_id
            for session_id, session in LIVE_SESSIONS.items()
            if now - session.last_activity > 600
        ]
        for session_id in expired:
            LIVE_SESSIONS.pop(session_id).close()
        session_id = uuid.uuid4().hex
        LIVE_SESSIONS[session_id] = LiveSession()
    return jsonify({"session_id": session_id})


@app.route("/api/live/frame", methods=["POST"])
def live_frame():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    with LIVE_SESSIONS_LOCK:
        session = LIVE_SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "Session expired. Start live translation again."}), 404
    if "image" not in data:
        return jsonify({"error": "image required"}), 400
    try:
        image = decode_image(data["image"])
        result = session.process_frame(image, bool(data.get("left_hand", False)))
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/api/live/clear", methods=["POST"])
def live_clear():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    with LIVE_SESSIONS_LOCK:
        session = LIVE_SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    with session.lock:
        session.text = ""
        session.last_added = None
        session.recent.clear()
        return jsonify({"text": ""})


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    with LIVE_SESSIONS_LOCK:
        session = LIVE_SESSIONS.pop(session_id, None)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    with session.lock:
        text = session.text
    session.close()
    return jsonify({"text": text})


if __name__ == "__main__":
    print(f"\nASL landmark API running on http://127.0.0.1:{PORT}")
    print("Endpoints: /api/health  /api/predict-photo  /api/live/*")
    print("This model recognizes static letters A-I and K-Y; J, Z, space, and delete are unsupported.")
    print("Keep this terminal open while using the website.\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
