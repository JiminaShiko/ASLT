"""Admin-only quiz manager API and storage."""

import os

from flask import Blueprint, g, jsonify, request, session

from config.database import get_db_connection
from .user_management import admin_required

admin_quiz_bp = Blueprint("admin_quiz", __name__, url_prefix="/api/admin/quiz")
TYPES = {"sign_to_text", "text_to_sign"}
STATUSES = {"active", "draft"}
OPTION_COUNT = 4

_QUESTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_quiz_questions (
    question_id INT AUTO_INCREMENT PRIMARY KEY,
    question_type VARCHAR(30) NOT NULL,
    prompt_material_id INT NOT NULL,
    prompt_text VARCHAR(120) NULL,
    category VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    correct_material_id INT NOT NULL,
    created_by INT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_admin_quiz_status_created (status, created_at),
    CONSTRAINT fk_admin_quiz_prompt_material
        FOREIGN KEY (prompt_material_id) REFERENCES sign_materials(material_id),
    CONSTRAINT fk_admin_quiz_correct_material
        FOREIGN KEY (correct_material_id) REFERENCES sign_materials(material_id),
    CONSTRAINT fk_admin_quiz_created_by
        FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB
"""

_OPTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_quiz_options (
    question_id INT NOT NULL,
    material_id INT NOT NULL,
    position TINYINT NOT NULL,
    PRIMARY KEY (question_id, position),
    UNIQUE KEY uq_admin_quiz_question_material (question_id, material_id),
    CONSTRAINT fk_admin_quiz_option_question
        FOREIGN KEY (question_id) REFERENCES admin_quiz_questions(question_id) ON DELETE CASCADE,
    CONSTRAINT fk_admin_quiz_option_material
        FOREIGN KEY (material_id) REFERENCES sign_materials(material_id)
) ENGINE=InnoDB
"""


class ApiError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@admin_quiz_bp.errorhandler(ApiError)
def handle_api_error(error):
    return jsonify({"detail": error.detail}), error.status


def ensure_admin_quiz_schema():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(_QUESTION_SCHEMA)
            cursor.execute(_OPTION_SCHEMA)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _write_origin_allowed():
    origin = request.headers.get("Origin")
    if not origin:
        return True
    allowed = {
        value.strip().rstrip("/")
        for value in os.getenv("SIGNREST_ALLOWED_ORIGINS", "").split(",")
        if value.strip()
    }
    allowed.add(request.host_url.rstrip("/"))
    return origin.rstrip("/") in allowed


def _table_exists(cursor, name):
    cursor.execute(
        "SELECT COUNT(*) AS n FROM information_schema.tables "
        "WHERE table_schema = DATABASE() AND table_name = %s",
        (name,),
    )
    return bool(cursor.fetchone()["n"])


def _media_url(path):
    if not path:
        return None
    path = str(path)
    if path.startswith("https://") or (path.startswith("/") and not path.startswith("//")):
        return path
    return None


def _admin_id():
    admin = g.get("signrest_account") or {}
    value = admin.get("user_id") or session.get("user_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ApiError(401, "Your administrator session is not valid")


def _validate_question(cursor, data):
    if not isinstance(data, dict):
        raise ApiError(400, "A JSON object is required")

    qtype = data.get("type")
    status = data.get("status", "active")
    category = data.get("category")
    prompt_text = data.get("prompt_text", "")
    if not isinstance(qtype, str) or qtype not in TYPES:
        raise ApiError(422, "Choose a valid question type")
    if not isinstance(status, str) or status not in STATUSES:
        raise ApiError(422, "Choose a valid status")
    if not isinstance(category, str) or not category.strip() or len(category.strip()) > 30:
        raise ApiError(422, "Category is required (max 30 characters)")
    if not isinstance(prompt_text, str) or len(prompt_text.strip()) > 120:
        raise ApiError(422, "Text is too long (max 120 characters)")
    prompt_text = prompt_text.strip()
    if qtype == "text_to_sign" and not prompt_text:
        raise ApiError(422, "Enter the text shown to the learner")

    try:
        if isinstance(data.get("prompt_material_id"), bool):
            raise ValueError
        prompt_id = int(data.get("prompt_material_id"))
        raw_options = data.get("options")
        if not isinstance(raw_options, list) or any(
            isinstance(value, bool) for value in raw_options
        ):
            raise ValueError
        option_ids = [int(value) for value in raw_options]
        if isinstance(data.get("correct"), bool):
            raise ValueError
        correct_id = int(data.get("correct"))
    except (TypeError, ValueError):
        raise ApiError(422, "Signs must be valid material IDs")

    if (
        prompt_id < 1
        or len(option_ids) != OPTION_COUNT
        or any(value < 1 for value in option_ids)
        or len(set(option_ids)) != OPTION_COUNT
        or prompt_id not in option_ids
        or correct_id != prompt_id
    ):
        raise ApiError(422, "Provide four different signs and mark the shown sign as correct")

    placeholders = ", ".join(["%s"] * len(option_ids))
    cursor.execute(
        "SELECT material_id, sign_label FROM sign_materials "
        f"WHERE material_id IN ({placeholders})",
        tuple(option_ids),
    )
    signs = {row["material_id"]: row for row in cursor.fetchall()}
    if len(signs) != OPTION_COUNT:
        raise ApiError(422, "One or more signs no longer exist")

    return {
        "type": qtype,
        "prompt_id": prompt_id,
        "prompt_text": prompt_text if qtype == "text_to_sign" else None,
        "category": category.strip(),
        "status": status,
        "options": option_ids,
        "correct": correct_id,
    }


def _write_options(cursor, question_id, options):
    cursor.execute(
        "DELETE FROM admin_quiz_options WHERE question_id = %s",
        (question_id,),
    )
    cursor.executemany(
        "INSERT INTO admin_quiz_options (question_id, material_id, position) "
        "VALUES (%s, %s, %s)",
        [(question_id, material_id, index) for index, material_id in enumerate(options)],
    )


def _load_questions(cursor):
    cursor.execute(
        """
        SELECT question_id, question_type, prompt_material_id, prompt_text,
               category, status, correct_material_id
        FROM admin_quiz_questions
        ORDER BY created_at DESC, question_id DESC
        """
    )
    questions = cursor.fetchall()
    if not questions:
        return []

    ids = [question["question_id"] for question in questions]
    placeholders = ", ".join(["%s"] * len(ids))
    cursor.execute(
        "SELECT question_id, material_id FROM admin_quiz_options "
        f"WHERE question_id IN ({placeholders}) ORDER BY question_id, position",
        tuple(ids),
    )
    options = {}
    for row in cursor.fetchall():
        options.setdefault(row["question_id"], []).append(row["material_id"])

    return [
        {
            "id": question["question_id"],
            "type": question["question_type"],
            "prompt_material_id": question["prompt_material_id"],
            "prompt_text": question["prompt_text"] or "",
            "category": question["category"],
            "status": question["status"],
            "options": options.get(question["question_id"], []),
            "correct": question["correct_material_id"],
        }
        for question in questions
    ]


def _require_question(cursor, question_id):
    cursor.execute(
        "SELECT question_id FROM admin_quiz_questions WHERE question_id = %s",
        (question_id,),
    )
    if not cursor.fetchone():
        raise ApiError(404, "Question not found")


@admin_quiz_bp.get("/signs")
@admin_required
def list_signs():
    connection = get_db_connection()
    try:
        with connection.cursor(dictionary=True) as cursor:
            has_stats = _table_exists(cursor, "sign_model_stats")
            cursor.execute(
                "SELECT m.material_id, m.sign_label, m.category, m.image_path"
                + (", s.confidence" if has_stats else ", NULL AS confidence")
                + " FROM sign_materials m"
                + (" LEFT JOIN sign_model_stats s ON s.material_id = m.material_id" if has_stats else "")
                + " ORDER BY m.category, m.sign_label LIMIT 2000"
            )
            items = [
                {
                    "material_id": row["material_id"],
                    "sign_label": row["sign_label"],
                    "category": row["category"],
                    "media_url": _media_url(row["image_path"]),
                    "model_confidence": (
                        float(row["confidence"]) if row["confidence"] is not None else None
                    ),
                }
                for row in cursor.fetchall()
            ]
    finally:
        connection.close()
    return jsonify({"items": items})


@admin_quiz_bp.get("/questions")
@admin_required
def list_questions():
    connection = get_db_connection()
    try:
        with connection.cursor(dictionary=True) as cursor:
            items = _load_questions(cursor)
    finally:
        connection.close()
    return jsonify({"items": items})


@admin_quiz_bp.post("/questions")
@admin_required
def create_question():
    if not _write_origin_allowed():
        raise ApiError(403, "This origin is not allowed")
    connection = get_db_connection()
    try:
        with connection.cursor(dictionary=True) as cursor:
            question = _validate_question(cursor, request.get_json(silent=True))
            cursor.execute(
                "INSERT INTO admin_quiz_questions "
                "(question_type, prompt_material_id, prompt_text, category, status, "
                "correct_material_id, created_by) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    question["type"],
                    question["prompt_id"],
                    question["prompt_text"],
                    question["category"],
                    question["status"],
                    question["correct"],
                    _admin_id(),
                ),
            )
            question_id = cursor.lastrowid
            _write_options(cursor, question_id, question["options"])
            items = _load_questions(cursor)
            item = next(row for row in items if row["id"] == question_id)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return jsonify({"item": item}), 201


@admin_quiz_bp.patch("/questions/<int:question_id>")
@admin_required
def update_question(question_id):
    if not _write_origin_allowed():
        raise ApiError(403, "This origin is not allowed")
    connection = get_db_connection()
    try:
        with connection.cursor(dictionary=True) as cursor:
            _require_question(cursor, question_id)
            question = _validate_question(cursor, request.get_json(silent=True))
            cursor.execute(
                "UPDATE admin_quiz_questions SET question_type = %s, "
                "prompt_material_id = %s, prompt_text = %s, category = %s, "
                "status = %s, correct_material_id = %s WHERE question_id = %s",
                (
                    question["type"],
                    question["prompt_id"],
                    question["prompt_text"],
                    question["category"],
                    question["status"],
                    question["correct"],
                    question_id,
                ),
            )
            _write_options(cursor, question_id, question["options"])
            items = _load_questions(cursor)
            item = next(row for row in items if row["id"] == question_id)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return jsonify({"item": item})


@admin_quiz_bp.delete("/questions/<int:question_id>")
@admin_required
def delete_question(question_id):
    if not _write_origin_allowed():
        raise ApiError(403, "This origin is not allowed")
    connection = get_db_connection()
    try:
        with connection.cursor(dictionary=True) as cursor:
            _require_question(cursor, question_id)
            cursor.execute(
                "DELETE FROM admin_quiz_questions WHERE question_id = %s",
                (question_id,),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return "", 204


@admin_quiz_bp.get("/distractors")
@admin_required
def distractors():
    try:
        material_id = int(request.args.get("material_id", ""))
    except ValueError:
        material_id = 0
    if material_id < 1:
        raise ApiError(422, "material_id is required")

    connection = get_db_connection()
    try:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(
                "SELECT category FROM sign_materials WHERE material_id = %s",
                (material_id,),
            )
            base = cursor.fetchone()
            if not base:
                raise ApiError(404, "Sign not found")
            picks = []
            if _table_exists(cursor, "sign_confusions"):
                cursor.execute(
                    "SELECT confused_with_id AS id FROM sign_confusions "
                    "WHERE material_id = %s AND confused_with_id <> %s "
                    "ORDER BY score DESC LIMIT 3",
                    (material_id, material_id),
                )
                picks = [row["id"] for row in cursor.fetchall()]
            if len(picks) < 3:
                cursor.execute(
                    "SELECT material_id AS id FROM sign_materials "
                    "WHERE material_id <> %s "
                    "ORDER BY (category = %s) DESC, RAND() LIMIT 20",
                    (material_id, base["category"]),
                )
                for row in cursor.fetchall():
                    if row["id"] not in picks and len(picks) < 3:
                        picks.append(row["id"])
    finally:
        connection.close()
    return jsonify({"items": picks})
