import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, jsonify, request, session
from config.database.db_connection import get_db_connection
from .api import (
    WordLiveSession,
    decode_image,
    get_backend,
    hand_points,
    image_to_bgr,
    image_to_rgb,
    reduce_image,
    validate_settings,
)
from .tracking import PoseOverlay  # noqa: E402

# api.py puts Model/ on sys.path; this is the same pipeline live.py runs.
from asl_landmark_app.src import hand_backend, live as landmark_live  # noqa: E402
from asl_landmark_app.src import normalize  # noqa: E402


learning_bp = Blueprint("learning_api", __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT_DIR = PROJECT_ROOT / "Front-end" / "text_to_sign_demo"
ALPHABET_DIR = DEMO_ROOT_DIR / "alphabet"
WORD_DIR = DEMO_ROOT_DIR / "word"

ASL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
ASL_VIDEO_LETTERS = {"J", "Z"}  # J and Z are .mp4 videos; all other letters are .png

# A camera must read the exact sign at this confidence before progress is saved.
LETTER_PASS_CONF = 0.60
LETTER_VOTE_FRAMES = 5      # size of the live letter vote window
LETTER_VOTE_CONF = 0.70     # a frame below this never votes (live.py's gate)
PRACTICE_TTL_SECONDS = 600

WLASL_WORDS = [
    "eat", "family", "friend", "good", "hello", "help", "name", "sad",
    "school", "drink", "thanks", "love", "bye", "sorry", "yes", "no",
    "happy", "please", "bad", "home", "me", "my", "you", "your",
    "what", "where", "who", "want", "need", "like", "go", "come",
    "stop", "more", "water", "understand", "now", "today", "work", "wait",
    "finish", "sleep", "bathroom", "hungry", "thirsty", "when", "why", "how",
    "apple", "bread", "chicken", "tomorrow", "sick", "hurt", "doctor", "medicine",
]


def ensure_learning_progress_table() -> None:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS `user_learning_progress` (
              `progress_id` INT NOT NULL AUTO_INCREMENT,
              `user_id` INT NOT NULL,
              `category` VARCHAR(20) NOT NULL,
              `item_key` VARCHAR(100) NOT NULL,
              `completed_at` DATETIME DEFAULT NULL,
              PRIMARY KEY (`progress_id`),
              UNIQUE KEY `uniq_user_cat_item` (`user_id`, `category`, `item_key`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
            """
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def _resolve_user_id(cur) -> int | None:
    user_id = session.get("user_id")
    if user_id:
        return int(user_id)
    user_email = session.get("user_email")
    if not user_email:
        return None
    cur.execute("SELECT user_id FROM users WHERE email = %s", (user_email,))
    row = cur.fetchone()
    if row:
        session["user_id"] = row[0]
        return int(row[0])
    return None


def _save_progress(user_id: int, category: str, item_key: str) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_learning_progress (user_id, category, item_key, completed_at)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE completed_at = VALUES(completed_at)
        """,
        (user_id, category, item_key, datetime.now(timezone.utc)),
    )
    conn.commit()
    cur.close()
    conn.close()


def _resolve_alphabet_url(letter: str) -> tuple[str, str]:
    if letter in ASL_VIDEO_LETTERS:
        for name in (letter, letter.lower()):
            if (ALPHABET_DIR / f"{name}.mp4").is_file():
                return f"/text_to_sign_demo/alphabet/{name}.mp4", "video"
        return f"/text_to_sign_demo/alphabet/{letter}.mp4", "video"

    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for name in (letter, letter.lower()):
            if (ALPHABET_DIR / f"{name}{ext}").is_file():
                return f"/text_to_sign_demo/alphabet/{name}{ext}", "image"
    return f"/text_to_sign_demo/alphabet/{letter}.png", "image"


def _resolve_word_url(word: str) -> str:
    for name in (word, word.lower(), word.capitalize()):
        if (WORD_DIR / f"{name}.mp4").is_file():
            return f"/text_to_sign_demo/word/{name}.mp4"
    return f"/text_to_sign_demo/word/{word}.mp4"


@learning_bp.route("/api/learning/curriculum", methods=["GET"])
def get_learning_curriculum():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    ensure_learning_progress_table()
    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)

    asl_completed = []
    wlasl_completed = []
    if user_id:
        cur.execute(
            "SELECT category, item_key FROM user_learning_progress WHERE user_id = %s",
            (user_id,),
        )
        for cat, item_key in cur.fetchall():
            if cat == "asl":
                asl_completed.append(item_key)
            elif cat == "wlasl":
                wlasl_completed.append(item_key)

    cur.close()
    conn.close()

    asl_items = []
    for letter in ASL_LETTERS:
        url, mtype = _resolve_alphabet_url(letter)
        asl_items.append(
            {
                "key": letter,
                "label": letter,
                "media_url": url,
                "media_type": mtype,
            }
        )

    wlasl_items = [
        {
            "key": word,
            "label": word.capitalize(),
            "media_url": _resolve_word_url(word),
            "media_type": "video",
        }
        for word in WLASL_WORDS
    ]

    response = jsonify(
        {
            "asl": {
                "total": len(asl_items),
                "completed": asl_completed,
                "items": asl_items,
            },
            "wlasl": {
                "total": len(wlasl_items),
                "completed": wlasl_completed,
                "items": wlasl_items,
            },
        }
    )
    # Progress must never be replayed from a browser cache after a page reload.
    response.headers["Cache-Control"] = "no-store"
    return response, 200


@learning_bp.route("/api/quizzes", methods=["GET"])
def list_active_quizzes():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)

    cur.execute(
        """
        SELECT q.quiz_id, q.title, q.description, COUNT(qq.question_id) AS question_count
        FROM quizzes q
        LEFT JOIN quiz_questions qq ON qq.quiz_id = q.quiz_id
        WHERE q.status = 'active'
        GROUP BY q.quiz_id, q.title, q.description
        ORDER BY q.quiz_id DESC
        """
    )
    rows = cur.fetchall()

    best_scores = {}
    if user_id:
        cur.execute(
            """
            SELECT quiz_id, MAX(score) AS best_score, MAX(total_questions) AS total_q
            FROM quiz_results
            WHERE user_id = %s
            GROUP BY quiz_id
            """,
            (user_id,),
        )
        for qid, bscore, tq in cur.fetchall():
            best_scores[qid] = {"best_score": bscore, "total_questions": tq}

    cur.close()
    conn.close()

    quizzes = []
    for qid, title, desc, qcount in rows:
        quizzes.append(
            {
                "quiz_id": qid,
                "title": title,
                "description": desc or "",
                "question_count": int(qcount or 0),
                "best_result": best_scores.get(qid),
            }
        )

    return jsonify({"quizzes": quizzes}), 200


@learning_bp.route("/api/quizzes/<int:quiz_id>", methods=["GET"])
def get_quiz_detail(quiz_id: int):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT quiz_id, title, description FROM quizzes WHERE quiz_id = %s AND status = 'active'",
        (quiz_id,),
    )
    qrow = cur.fetchone()
    if not qrow:
        cur.close()
        conn.close()
        return jsonify({"error": "Quiz not found."}), 404

    cur.execute(
        """
        SELECT question_id, question_text, image_path, options
        FROM quiz_questions
        WHERE quiz_id = %s
        ORDER BY question_id ASC
        """,
        (quiz_id,),
    )
    questions = []
    for qid, qtext, img_path, raw_opts in cur.fetchall():
        parsed_opts = raw_opts
        if isinstance(raw_opts, str):
            try:
                parsed_opts = json.loads(raw_opts)
            except Exception:
                parsed_opts = [o.strip() for o in raw_opts.split(",") if o.strip()]
        if not isinstance(parsed_opts, list):
            parsed_opts = []

        questions.append(
            {
                "question_id": qid,
                "question_text": qtext or "",
                "image_path": img_path or "",
                "options": parsed_opts,
            }
        )

    cur.close()
    conn.close()

    return jsonify(
        {
            "quiz_id": qrow[0],
            "title": qrow[1],
            "description": qrow[2] or "",
            "questions": questions,
        }
    ), 200


@learning_bp.route("/api/quizzes/<int:quiz_id>/submit", methods=["POST"])
def submit_quiz_attempt(quiz_id: int):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    answers = data.get("answers") or {}

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    if not user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found."}), 404

    cur.execute(
        "SELECT question_id, correct_answer FROM quiz_questions WHERE quiz_id = %s ORDER BY question_id ASC",
        (quiz_id,),
    )
    qrows = cur.fetchall()
    if not qrows:
        cur.close()
        conn.close()
        return jsonify({"error": "Quiz has no questions."}), 400

    total_questions = len(qrows)
    score = 0
    breakdown = []

    for qid, correct_ans in qrows:
        user_choice = str(answers.get(str(qid), "")).strip()
        is_correct = user_choice.lower() == str(correct_ans).strip().lower()
        if is_correct:
            score += 1
        breakdown.append(
            {
                "question_id": qid,
                "user_answer": user_choice,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
            }
        )

    cur.execute(
        """
        INSERT INTO quiz_results (quiz_id, user_id, score, total_questions, completed_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (quiz_id, user_id, score, total_questions, datetime.now(timezone.utc)),
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify(
        {
            "quiz_id": quiz_id,
            "score": score,
            "total_questions": total_questions,
            "breakdown": breakdown,
        }
    ), 200


# ---------------------------------------------------------------------------
# Camera sign test
#
# Opening a lesson no longer grants progress. The user shows the sign to the
# camera first; the frame goes through the same alphabet / word model the
# translator uses, and progress is written only when the model reads the sign
# the lesson asked for.
# ---------------------------------------------------------------------------

PRACTICE_SESSIONS = {}
PRACTICE_LOCK = threading.Lock()


class LetterPractice:
    """One alphabet handshape, read the way live.py reads a webcam.

    One VIDEO detector lives for the whole session (the old still-image path
    built a MediaPipe model per frame without a hand, which is what dropped
    frames and heated the CPU), and the letter only counts once the vote
    window agrees, so a random handshape cannot pass on a single lucky frame.
    """

    def __init__(self, target: str):
        self.target = target.strip().upper()
        self.settings = validate_settings(None)
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
            conf=LETTER_VOTE_CONF, vote=LETTER_VOTE_FRAMES, hold=0.0)
        # J and Z are not among the classifier's 24 static classes, so without the
        # trained stroke head this camera test could never pass them - it read an
        # I-shaped hand as "I" forever. Same decider live.py runs.
        self.motion = landmark_live.MotionDecider()
        # Body/face trace for the overlay only; the letter reading is unchanged.
        self.pose = PoseOverlay()
        # The pose pass is heavy next to the hand model and the answer never needs
        # it, so it runs on a worker thread and the last payload is shown meanwhile.
        self._pose_lock = threading.Lock()
        self._pose_busy = False
        self._pose_payload = {"head": {"lines": [], "dots": []}, "body": []}
        self.started_at = time.monotonic()
        self.streak = 0
        self.frames = 0
        self.last_frame_ms = 0.0

    def frame(self, image):
        started = time.perf_counter()
        self.frames += 1
        image = reduce_image(image)
        bgr = image_to_bgr(image)
        overlay = {"hands": [], "head": {"lines": [], "dots": []}, "body": []}
        with self._pose_lock:
            payload = self._pose_payload
        overlay["head"] = payload["head"]
        overlay["body"] = payload["body"]
        self._pose_tick(image_to_rgb(image))
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
        result, _how = hand_backend.detect_frame(
            self.detector, bgr, timestamp_ms, self.recovery,
            empty_streak=self.streak + 1,
        )
        self.streak = self.streak + 1 if result.landmarks is None else 0
        now = time.time()
        self.last_frame_ms = (time.perf_counter() - started) * 1000.0

        if result.landmarks is None:
            self.motion.reset()
            self.smoother.step(None, 0.0, now, hand_present=False)
            return {
                "matched": False,
                "predicted": "",
                "confidence": 0.0,
                "held": "",
                "hands": [],
                "head": overlay["head"],
                "body": overlay["body"],
                "frame_ms": self.last_frame_ms,
                "hint": "No hand detected. Move your hand into the camera view.",
            }

        probabilities = self.backend(normalize.features(result.landmarks))
        call = landmark_live.frame_call(
            probabilities, self.labels, self.settings["conf"],
            self.settings["margin"], pairs=self.settings["pairs"],
        )
        held, _committed, held_conf = self.smoother.step(
            call["letter"], call["conf"], now, hand_present=True)
        top1 = call["top3"][0][0] if call["top3"] else ""
        # A stroke the J/Z head names overrides this frame's static reading: the
        # classifier has no J or Z to offer, so the hand-written letter is all it
        # can say about a hook or a zigzag.
        motion_letter = self.motion.update(
            now, result.landmarks, top1, call["conf"])
        if motion_letter:
            held = motion_letter
            held_conf = float(self.motion.last_probs.get(motion_letter, 1.0))
        confidence = float(held_conf if held else call["conf"])
        # A fired stroke already cleared the head's own threshold, so the static
        # pass-confidence gate must not reject it afterwards.
        matched = bool(
            held
            and held.upper() == self.target
            and (motion_letter is not None or confidence >= LETTER_PASS_CONF)
        )
        return {
            "matched": matched,
            "predicted": held or top1,
            "confidence": confidence,
            "held": held or "",
            "hands": [hand_points(result.landmarks)],
            "head": overlay["head"],
            "body": overlay["body"],
            "frame_ms": self.last_frame_ms,
            "hint": "" if held else "Hold the sign steady in the middle of the frame.",
        }

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

        threading.Thread(target=run, name="practice-pose", daemon=True).start()

    def close(self):
        try:
            self.detector.close()
        finally:
            self.pose.close()
            if self.recovery is not None:
                self.recovery.close()


class WordPractice:
    """One word checked with the live word model, which reads a whole sign."""

    def __init__(self, target: str):
        self.target = target.strip().lower()
        self.session = WordLiveSession(None)

    def frame(self, image):
        state = self.session.process_frame(image)
        said = re.findall(r"[a-z']+", str(state.get("text") or "").lower())
        top3 = state.get("top3") or []
        predicted, confidence = (
            (top3[0][0], float(top3[0][1] or 0.0)) if top3 else ("", 0.0)
        )
        hands = state.get("hand_in_frame") or (False, False)
        overlay = state.get("overlay") or {}
        return {
            "matched": self.target in said,
            "predicted": predicted,
            "confidence": confidence,
            "hands": overlay.get("hands") or [],
            "head": overlay.get("head") or {"lines": [], "dots": []},
            "body": overlay.get("body") or [],
            "frame_ms": float(state.get("frame_ms") or 0.0),
            "hint": "" if any(hands) else "No hand detected. Sign the word facing the camera.",
        }

    def close(self):
        self.session.close()


def _practice_caller_id() -> int | None:
    if not session.get("user_email") and not session.get("user_id"):
        return None
    try:
        raw = session.get("user_id")
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _close_practice(entry) -> None:
    try:
        entry["engine"].close()
    except Exception:
        pass


def _sweep_practice_sessions(now: float) -> None:
    expired = [
        key for key, entry in PRACTICE_SESSIONS.items()
        if now - entry["started"] > PRACTICE_TTL_SECONDS
    ]
    for key in expired:
        _close_practice(PRACTICE_SESSIONS.pop(key))


@learning_bp.route("/api/learning/practice/start", methods=["POST"])
def start_learning_practice():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    category = (data.get("category") or "").strip().lower()
    item_key = (data.get("item_key") or "").strip()

    if category == "asl":
        item_key = item_key.upper()
        if item_key not in ASL_LETTERS:
            return jsonify({"error": "Unknown alphabet sign."}), 400
    elif category == "wlasl":
        item_key = item_key.lower()
        if item_key not in WLASL_WORDS:
            return jsonify({"error": "Unknown word sign."}), 400
    else:
        return jsonify({"error": "Invalid category or item_key."}), 400

    ensure_learning_progress_table()
    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    cur.close()
    conn.close()
    if not user_id:
        return jsonify({"error": "User not found."}), 404

    try:
        engine = LetterPractice(item_key) if category == "asl" else WordPractice(item_key)
    except Exception as exc:
        return jsonify({"error": f"Could not load the sign model: {exc}"}), 503

    session_id = uuid.uuid4().hex
    now = time.monotonic()
    with PRACTICE_LOCK:
        _sweep_practice_sessions(now)
        PRACTICE_SESSIONS[session_id] = {
            "engine": engine,
            "user_id": user_id,
            "category": category,
            "item_key": item_key,
            "recorded": False,
            "started": now,
        }

    return jsonify(
        {"session_id": session_id, "category": category, "item_key": item_key}
    ), 200


@learning_bp.route("/api/learning/practice/frame", methods=["POST"])
def learning_practice_frame():
    caller_id = _practice_caller_id()
    if caller_id is None:
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    image_data = data.get("image")
    if not session_id or not image_data:
        return jsonify({"error": "session_id and image are required."}), 400

    with PRACTICE_LOCK:
        entry = PRACTICE_SESSIONS.get(session_id)
    if entry is None:
        return jsonify({"error": "This test session expired. Open the test again."}), 404
    if entry["user_id"] != caller_id:
        return jsonify({"error": "This test session belongs to another user."}), 403

    try:
        state = entry["engine"].frame(decode_image(image_data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if state.get("matched") and not entry["recorded"]:
        _save_progress(entry["user_id"], entry["category"], entry["item_key"])
        entry["recorded"] = True
        state["saved"] = True

    state["passed"] = bool(state.get("matched"))
    state["category"] = entry["category"]
    state["item_key"] = entry["item_key"]
    return jsonify(state), 200


@learning_bp.route("/api/learning/practice/stop", methods=["POST"])
def stop_learning_practice():
    caller_id = _practice_caller_id()
    if caller_id is None:
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    with PRACTICE_LOCK:
        entry = PRACTICE_SESSIONS.get(data.get("session_id"))
        if entry is not None and entry["user_id"] != caller_id:
            return jsonify({"error": "This test session belongs to another user."}), 403
        PRACTICE_SESSIONS.pop(data.get("session_id"), None)

    if entry is None:
        return jsonify({"stopped": False}), 200
    _close_practice(entry)
    return jsonify({"stopped": True}), 200