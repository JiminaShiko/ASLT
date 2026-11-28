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

# The daily-communication vocabulary the shipped word model is trained on
# (Model/videoModel/vocab.json -> enabled_words). Keep the two in step: a lesson
# for a word the recogniser cannot say can never be passed.
WLASL_WORDS = [
    "eat", "family", "friend", "good", "hello", "help", "name", "school",
    "drink", "thanks", "love", "bye", "sorry", "yes", "no", "please",
    "bad", "home", "me", "you", "what", "where", "want", "need", "go",
    "come", "stop", "more", "water", "understand", "now", "today",
    "work", "finish", "sleep", "bathroom", "hungry", "when", "why", "how",
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


@learning_bp.route("/api/learning/rank", methods=["GET"])
def learning_rank():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401
    ensure_learning_progress_table()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        user_id = _resolve_user_id(cur)
        if not user_id:
            return jsonify({"error": "User not found."}), 404
        cur.execute("SELECT user_id FROM users WHERE role = 'user' AND status = 'active'")
        user_ids = [row[0] for row in cur.fetchall()]
        if user_id not in user_ids:
            return jsonify({"error": "Ranking is available for active learners."}), 403
        progress = {uid: {"asl": set(), "wlasl": set()} for uid in user_ids}
        cur.execute("SELECT user_id, category, item_key FROM user_learning_progress")
        for uid, category, key in cur.fetchall():
            if uid in progress and ((category == "asl" and key in ASL_LETTERS) or
                                    (category == "wlasl" and key in WLASL_WORDS)):
                progress[uid][category].add(key)
        cur.execute("SELECT quiz_id FROM quizzes WHERE status = 'active'")
        quiz_ids = {row[0] for row in cur.fetchall()}
        best = {uid: {} for uid in user_ids}
        cur.execute("SELECT user_id, quiz_id, score, total_questions FROM quiz_results")
        for uid, quiz_id, score, total in cur.fetchall():
            if uid in best and quiz_id in quiz_ids and score is not None and total and total > 0:
                percent = max(0.0, min(1.0, score / total))
                best[uid][quiz_id] = max(best[uid].get(quiz_id, 0.0), percent)
        scores = {}
        for uid in user_ids:
            learning_score = (len(progress[uid]["asl"]) / len(ASL_LETTERS) +
                              len(progress[uid]["wlasl"]) / len(WLASL_WORDS)) / 2
            quiz_score = sum(best[uid].values()) / len(quiz_ids) if quiz_ids else 0.0
            scores[uid] = (learning_score, quiz_score, 50 * (learning_score + quiz_score))
        learning_score, quiz_score, score = scores[user_id]
        rank = 1 + sum(1 for _learn, _quiz, other in scores.values() if other > score + 1e-9)
        return jsonify({"rank": rank, "learner_count": len(user_ids), "score": round(score, 2),
                        "learning_percent": round(learning_score * 100, 2),
                        "quiz_percent": round(quiz_score * 100, 2),
                        "published_quizzes": len(quiz_ids), "donations_included": False})
    finally:
        cur.close()
        conn.close()


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
    attempt_counts = {}
    attempt_history = {}
    in_progress = {}
    if user_id:
        cur.execute(
            """
            SELECT quiz_id, score, total_questions, completed_at
            FROM quiz_results
            WHERE user_id = %s
            ORDER BY completed_at DESC, result_id DESC
            """,
            (user_id,),
        )
        for qid, score, total, completed_at in cur.fetchall():
            attempt_counts[qid] = attempt_counts.get(qid, 0) + 1
            attempt_history.setdefault(qid, []).append({
                "score": score, "total_questions": total,
                "completed_at": completed_at.isoformat() if hasattr(completed_at, "isoformat") else str(completed_at or ""),
            })
            if score is None or not total:
                continue
            current = best_scores.get(qid)
            if current is None or score / total > current["best_score"] / current["total_questions"]:
                best_scores[qid] = {"best_score": score, "total_questions": total}
        # Completed runs also have a quiz_results row; count them only once.
        cur.execute("SELECT quiz_id, status, answer_state, question_snapshot FROM quiz_attempts WHERE user_id = %s", (user_id,))
        for qid, status, raw_answers, raw_snapshot in cur.fetchall():
            if status == "in_progress":
                in_progress[qid] = {"answered": len(_quiz_json(raw_answers, {})), "total": len(_quiz_json(raw_snapshot, []))}
                attempt_counts[qid] = attempt_counts.get(qid, 0) + 1

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
                "attempts_used": attempt_counts.get(qid, 0),
                "attempts_remaining": max(0, 3 - attempt_counts.get(qid, 0)),
                "in_progress": qid in in_progress,
                "progress": in_progress.get(qid),
                "results": attempt_history.get(qid, []),
            }
        )

    return jsonify({"quizzes": quizzes}), 200


def _quiz_json(value, default):
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return value if value is not None else default


def _quiz_attempt_payload(attempt_id, quiz_id, title, snapshot, answers, attempt_number):
    # Correct answers never leave the server while the quiz is in progress.
    return {"attempt_id": attempt_id, "quiz_id": quiz_id, "title": title,
            "attempt_number": attempt_number, "answers": answers,
            "questions": [{key: value for key, value in question.items() if key != "correct_answer"}
                          for question in snapshot]}


@learning_bp.route("/api/quizzes/<int:quiz_id>/attempts", methods=["POST"])
def start_or_resume_quiz(quiz_id):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        user_id = _resolve_user_id(cur)
        if not user_id:
            return jsonify({"error": "User not found."}), 404
        cur.execute("SELECT user_id FROM users WHERE user_id=%s FOR UPDATE", (user_id,))
        if not cur.fetchone():
            return jsonify({"error": "User not found."}), 404
        cur.execute("SELECT title, status FROM quizzes WHERE quiz_id=%s FOR UPDATE", (quiz_id,))
        quiz = cur.fetchone()
        if not quiz or quiz[1] != "active":
            return jsonify({"error": "This quiz is not available."}), 404
        cur.execute("""SELECT attempt_id, question_snapshot, answer_state, attempt_number
                       FROM quiz_attempts WHERE quiz_id=%s AND user_id=%s AND status='in_progress'
                       ORDER BY attempt_number DESC LIMIT 1 FOR UPDATE""", (quiz_id, user_id))
        existing = cur.fetchone()
        if existing:
            snapshot, answers = _quiz_json(existing[1], []), _quiz_json(existing[2], {})
            conn.commit()
            return jsonify(_quiz_attempt_payload(existing[0], quiz_id, quiz[0], snapshot, answers, existing[3]))
        cur.execute("SELECT COUNT(*) FROM quiz_results WHERE quiz_id=%s AND user_id=%s", (quiz_id, user_id))
        completed = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(MAX(attempt_number),0) FROM quiz_attempts WHERE quiz_id=%s AND user_id=%s", (quiz_id, user_id))
        attempt_number = max(completed, cur.fetchone()[0]) + 1
        if attempt_number > 3:
            return jsonify({"error": "You have used all three attempts for this quiz."}), 409
        cur.execute("""SELECT question_id, question_text, image_path, options, question_kind,
                       prompt_media_type, correct_answer FROM quiz_questions WHERE quiz_id=%s
                       ORDER BY display_order, question_id""", (quiz_id,))
        rows = cur.fetchall()
        if not rows or any(row[4] != "multiple_choice" for row in rows):
            return jsonify({"error": "This quiz is not available for multiple-choice play."}), 409
        snapshot = [{"question_id": row[0], "question_text": row[1] or "", "image_path": row[2] or "",
                     "options": _quiz_json(row[3], []), "question_kind": row[4],
                     "prompt_media_type": row[5] or "text", "correct_answer": row[6]} for row in rows]
        if any(not isinstance(q["options"], list) or len(q["options"]) != 4 for q in snapshot):
            return jsonify({"error": "This quiz has invalid answer options."}), 409
        cur.execute("""INSERT INTO quiz_attempts (quiz_id,user_id,attempt_number,status,question_snapshot,answer_state,started_at)
                       VALUES (%s,%s,%s,'in_progress',%s,%s,%s)""",
                    (quiz_id, user_id, attempt_number, json.dumps(snapshot), "{}", datetime.now(timezone.utc)))
        attempt_id = cur.lastrowid
        conn.commit()
        return jsonify(_quiz_attempt_payload(attempt_id, quiz_id, quiz[0], snapshot, {}, attempt_number)), 201
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@learning_bp.route("/api/quiz-attempts/<int:attempt_id>/answer", methods=["POST"])
def save_quiz_answer(attempt_id):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401
    data = request.get_json(silent=True) or {}
    question_id, answer = data.get("question_id"), data.get("answer")
    if type(question_id) is not int or not isinstance(answer, str):
        return jsonify({"error": "Invalid answer."}), 400
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        user_id = _resolve_user_id(cur)
        cur.execute("""SELECT status, question_snapshot, answer_state FROM quiz_attempts
                       WHERE attempt_id=%s AND user_id=%s FOR UPDATE""", (attempt_id, user_id))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Attempt not found."}), 404
        if row[0] != "in_progress":
            return jsonify({"error": "This attempt has already finished."}), 409
        snapshot, answers = _quiz_json(row[1], []), _quiz_json(row[2], {})
        question = next((q for q in snapshot if q["question_id"] == question_id), None)
        if not question:
            return jsonify({"error": "Question not found in this attempt."}), 400
        options = question["options"]
        allowed = [str(i) for i in range(len(options))] if options and all(isinstance(o, dict) for o in options) else [str(o) for o in options]
        if answer not in allowed:
            return jsonify({"error": "Invalid answer option."}), 400
        answers[str(question_id)] = answer
        cur.execute("UPDATE quiz_attempts SET answer_state=%s WHERE attempt_id=%s", (json.dumps(answers), attempt_id))
        conn.commit()
        return jsonify({"answered": len(answers), "total_questions": len(snapshot)})
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@learning_bp.route("/api/quiz-attempts/<int:attempt_id>/finish", methods=["POST"])
def finish_quiz_attempt(attempt_id):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        user_id = _resolve_user_id(cur)
        cur.execute("""SELECT quiz_id, status, question_snapshot, answer_state FROM quiz_attempts
                       WHERE attempt_id=%s AND user_id=%s FOR UPDATE""", (attempt_id, user_id))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Attempt not found."}), 404
        snapshot, answers = _quiz_json(row[2], []), _quiz_json(row[3], {})
        if not snapshot or any(str(q["question_id"]) not in answers for q in snapshot):
            return jsonify({"error": "Please answer every question before submitting."}), 400
        score = sum(1 for q in snapshot if str(answers[str(q["question_id"])]).strip().lower() == str(q["correct_answer"]).strip().lower())
        if row[1] == "in_progress":
            now = datetime.now(timezone.utc)
            cur.execute("INSERT INTO quiz_results (quiz_id,user_id,score,total_questions,completed_at) VALUES (%s,%s,%s,%s,%s)",
                        (row[0], user_id, score, len(snapshot), now))
            cur.execute("UPDATE quiz_attempts SET status='completed', completed_at=%s WHERE attempt_id=%s", (now, attempt_id))
            conn.commit()
        elif row[1] != "completed":
            return jsonify({"error": "This attempt cannot be submitted."}), 409
        return jsonify({"quiz_id": row[0], "score": score, "total_questions": len(snapshot)})
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@learning_bp.route("/api/quizzes/<int:quiz_id>", methods=["GET"])
def get_quiz_detail(quiz_id: int):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    conn = get_db_connection()
    cur = conn.cursor()

    user_id = _resolve_user_id(cur)
    cur.execute("SELECT attempt_id FROM quiz_attempts WHERE quiz_id=%s AND user_id=%s AND status='in_progress' LIMIT 1", (quiz_id, user_id))
    if not cur.fetchone():
        cur.close()
        conn.close()
        return jsonify({"error": "Start a quiz attempt to view its questions."}), 409

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
        SELECT question_id, question_text, image_path, options, question_kind, prompt_media_type
        FROM quiz_questions
        WHERE quiz_id = %s
        ORDER BY display_order ASC, question_id ASC
        """,
        (quiz_id,),
    )
    questions = []
    for qid, qtext, img_path, raw_opts, question_kind, prompt_media_type in cur.fetchall():
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
                "question_kind": question_kind or "multiple_choice",
                "prompt_media_type": prompt_media_type or "text",
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
    if not isinstance(answers, dict):
        return jsonify({"error": "Answers must be provided for each question."}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    if not user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found."}), 404

    # Serialize attempts for a user so concurrent submissions cannot exceed
    # the three-attempt limit. The lock is released on commit or rollback.
    cur.execute("SELECT user_id FROM users WHERE user_id = %s FOR UPDATE", (user_id,))
    if not cur.fetchone():
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "User not found."}), 404
    cur.execute("SELECT status FROM quizzes WHERE quiz_id = %s FOR UPDATE", (quiz_id,))
    quiz = cur.fetchone()
    if not quiz or quiz[0] != "active":
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "This quiz is not available."}), 404
    cur.execute("SELECT COUNT(*) FROM quiz_results WHERE quiz_id = %s AND user_id = %s", (quiz_id, user_id))
    if cur.fetchone()[0] >= 3:
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "You have used all three attempts for this quiz."}), 409
    cur.execute("SELECT COUNT(*) FROM quiz_attempts WHERE quiz_id=%s AND user_id=%s AND status='in_progress'", (quiz_id, user_id))
    if cur.fetchone()[0]:
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "Resume your in-progress quiz instead."}), 409

    cur.execute(
        "SELECT question_id, correct_answer, question_kind, options FROM quiz_questions WHERE quiz_id = %s ORDER BY display_order ASC, question_id ASC",
        (quiz_id,),
    )
    qrows = cur.fetchall()
    if not qrows:
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "Quiz has no questions."}), 400

    if any(kind != "multiple_choice" for _qid, _correct, kind, _options in qrows):
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "Camera quiz questions are not available yet."}), 409

    if any(isinstance(_quiz_json(raw, None), list) and _quiz_json(raw, None) and
           all(isinstance(option, dict) for option in _quiz_json(raw, None))
           for _qid, _correct, _kind, raw in qrows):
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "Please use the saved quiz attempt to submit this quiz."}), 409

    if any(not isinstance(answers.get(str(qid)), str) or not answers[str(qid)].strip()
           for qid, _correct, _kind, _options in qrows):
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({"error": "Please answer every question before submitting."}), 400

    total_questions = len(qrows)
    score = 0
    breakdown = []

    for qid, correct_ans, _kind, raw_options in qrows:
        user_choice = str(answers.get(str(qid), "")).strip()
        try:
            options = json.loads(raw_options) if isinstance(raw_options, str) else raw_options
        except (ValueError, TypeError):
            options = []
        if isinstance(options, list) and options and all(isinstance(option, dict) for option in options):
            if user_choice not in {str(index) for index in range(len(options))}:
                conn.rollback()
                cur.close()
                conn.close()
                return jsonify({"error": "Invalid answer option."}), 400
            is_correct = user_choice == str(correct_ans)
        else:
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
        # J and Z are not among the classifier's 24 static classes. The shared
        # motion decider reads the pinky trajectory for J and the index trajectory
        # for Z, so this practice camera follows the same live recognition path.
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
        top1 = call["top3"][0][0] if call["top3"] else ""
        # A stroke the J/Z head names overrides this frame's static reading: the
        # classifier has no J or Z to offer, so the hand-written letter is all it
        # can say about a hook or a zigzag.
        motion_letter = self.motion.update(
            now, result.landmarks, top1, call["conf"])
        motion_blocked = self.motion.blocks_static(top1, now)
        static_vote = None if motion_blocked else call["letter"]
        held, _committed, held_conf = self.smoother.step(
            static_vote, call["conf"], now, hand_present=True)
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
            "predicted": held or ("" if motion_blocked else top1),
            "confidence": confidence,
            "held": held or "",
            "hands": [hand_points(result.landmarks)],
            "head": overlay["head"],
            "body": overlay["body"],
            "frame_ms": self.last_frame_ms,
            "hint": (
                (f"Trace the {self.target} stroke in one clear motion."
                 if self.target in ("J", "Z") else
                 "Checking the hand movement before choosing I/J or X/Z.")
                if motion_blocked else
                ("" if held else "Hold the sign steady in the middle of the frame.")
            ),
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
