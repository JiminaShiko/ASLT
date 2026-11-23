import json
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, jsonify, request, session
from config.database.db_connection import get_db_connection

learning_bp = Blueprint("learning_api", __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT_DIR = PROJECT_ROOT / "Front-end" / "text_to_sign_demo"
ALPHABET_DIR = DEMO_ROOT_DIR / "alphabet"
WORD_DIR = DEMO_ROOT_DIR / "word"

ASL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
ASL_VIDEO_LETTERS = {"J", "Z"}  # J and Z are .mp4 videos; all other letters are .png

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

    return jsonify(
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
    ), 200


@learning_bp.route("/api/learning/progress", methods=["POST"])
def record_learning_progress():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    category = (data.get("category") or "").strip().lower()
    item_key = (data.get("item_key") or "").strip()

    if category not in {"asl", "wlasl"} or not item_key:
        return jsonify({"error": "Invalid category or item_key."}), 400

    ensure_learning_progress_table()
    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    if not user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "User not found."}), 404

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

    return jsonify({"saved": True, "category": category, "item_key": item_key}), 200


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