"""
ASL Photo Model API (Flask)
---------------------------
Backend required by the teacher (Flask) + live translation features.

Endpoints
---------
GET  /api/health
POST /api/predict-photo          single image
POST /api/live/start             start a live session
POST /api/live/frame             send a frame
POST /api/live/clear             clear typed text
POST /api/live/stop              stop session and return final text

Run (from this folder):
    pip install flask flask-cors
    python api.py
"""

import os
import io
import base64
import uuid
import time
import threading
from collections import deque

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps

import preprocess

# ---------------------------------------------------------------------------
# Settings (aligned with Streamlit app.py)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign_photo_model.keras")
IMG_SIZE = preprocess.IMG_SIZE
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]

MIN_CONFIDENCE = 0.60
STABLE_FRAMES = 4
DETECT_MAX_SIDE = 640
NOTHING_TICKS = 2
PORT = 5000

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:8000", "http://localhost:8000",
                   "http://127.0.0.1:5500", "http://localhost:5500",
                   "null"])

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Classes:", len(CLASS_NAMES))

PREDICTION_LOCK = threading.Lock()
LIVE_SESSIONS = {}
LIVE_SESSIONS_LOCK = threading.Lock()


def predict_probabilities(processed):
    batch = np.asarray(processed, dtype="float32").reshape(1, IMG_SIZE, IMG_SIZE, 1)
    with PREDICTION_LOCK:
        return model(batch, training=False).numpy()[0]


def apply_to_text(text, name):
    if name == "space":
        return text + " "
    if name == "del":
        return text[:-1]
    if name == "nothing":
        return text
    return text + name


def decode_image(data_url_or_b64):
    b64 = data_url_or_b64
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encoded_processed_image(processed):
    buf = io.BytesIO()
    Image.fromarray(processed).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Live session (same logic as Streamlit / attached web_api.py)
# ---------------------------------------------------------------------------
class LiveSession:
    def __init__(self):
        self.lock = threading.RLock()
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

    def _response(self, processed):
        return {
            "letter": self.letter,
            "confidence": float(self.confidence),
            "hand_found": bool(self.hand_found),
            "text": self.text,
            "top_predictions": self.top_predictions,
            "processed_image": encoded_processed_image(processed) if processed is not None else None,
        }

    def process_frame(self, image, left_hand):
        with self.lock:
            self.last_activity = time.monotonic()
            if left_hand:
                image = ImageOps.mirror(image)

            w, h = image.size
            longest = max(w, h)
            if longest > DETECT_MAX_SIDE:
                scale = DETECT_MAX_SIDE / longest
                image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            processed, hand_found = preprocess.preprocess_image(image, video=True)

            if hand_found:
                self.misses = 0
                prediction = predict_probabilities(processed)
                if self.prob_ema is None:
                    self.prob_ema = prediction
                else:
                    old_idx = int(np.argmax(self.prob_ema))
                    new_idx = int(np.argmax(prediction))
                    if new_idx != old_idx and prediction[new_idx] >= MIN_CONFIDENCE:
                        self.prob_ema = prediction
                    else:
                        self.prob_ema = 0.6 * self.prob_ema + 0.4 * prediction
                prediction = self.prob_ema
                idx = int(np.argmax(prediction))
                name = CLASS_NAMES[idx]
                confidence = float(prediction[idx])
            else:
                self.misses += 1
                if self.misses < NOTHING_TICKS:
                    return self._response(processed)
                self.prob_ema = None
                prediction = np.zeros(len(CLASS_NAMES), dtype="float32")
                prediction[CLASS_NAMES.index("nothing")] = 1.0
                name = "nothing"
                confidence = 1.0

            self.letter = name
            self.confidence = confidence
            self.hand_found = hand_found
            top_five = np.argsort(prediction)[::-1][:5]
            self.top_predictions = [
                {"text": CLASS_NAMES[i], "confidence": float(prediction[i])}
                for i in top_five
            ]
            self.recent.append(name if confidence >= MIN_CONFIDENCE else "?")

            if name == "nothing":
                self.last_added = None
            else:
                stable = (
                    self.recent[0]
                    if len(self.recent) == STABLE_FRAMES and len(set(self.recent)) == 1
                    else None
                )
                if stable is not None and stable != "?" and stable != self.last_added:
                    self.text = apply_to_text(self.text, stable)
                    self.last_added = stable

            return self._response(processed)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "classes": len(CLASS_NAMES)})


@app.route("/api/predict-photo", methods=["POST"])
def predict_photo():
    """Single-image prediction (Upload Media tab)."""
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "No image provided."}), 400
        image = decode_image(data["image"])
        processed, hand_found = preprocess.preprocess_image(image)
        if not hand_found:
            return jsonify({"text": "nothing", "confidence": 1.0, "hand_found": False})
        probs = predict_probabilities(processed)
        idx = int(np.argmax(probs))
        return jsonify({
            "text": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "hand_found": True,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/start", methods=["POST"])
def live_start():
    now = time.monotonic()
    with LIVE_SESSIONS_LOCK:
        expired = [sid for sid, s in LIVE_SESSIONS.items() if now - s.last_activity > 600]
        for sid in expired:
            del LIVE_SESSIONS[sid]
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
        left_hand = bool(data.get("left_hand", False))
        result = session.process_frame(image, left_hand)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        return jsonify({"text": session.text})


if __name__ == "__main__":
    print(f"\nASL Photo API (Flask) running on http://127.0.0.1:{PORT}")
    print("Endpoints: /api/health  /api/predict-photo  /api/live/*")
    print("Keep this terminal open while using the website.\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
