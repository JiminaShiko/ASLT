from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file, session
from werkzeug.utils import secure_filename

from config.database.db_connection import get_db_connection

history_bp = Blueprint("search_history_api", __name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
MEDIA_DIR = BACKEND_DIR / "uploads" / "search_history"
MAX_MEDIA_BYTES = 25 * 1024 * 1024
MAX_RESULT_CHARS = 500_000
HISTORY_TYPES = {"webcam_letters", "webcam_words", "text", "upload"}
MEDIA_EXTENSIONS = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


def _resolve_user_id(cursor) -> int | None:
    raw_user_id = session.get("user_id")
    if raw_user_id:
        try:
            return int(raw_user_id)
        except (TypeError, ValueError):
            return None

    email = session.get("user_email")
    if not email:
        return None
    cursor.execute("SELECT user_id FROM users WHERE email = %s LIMIT 1", (email,))
    row = cursor.fetchone()
    if not row:
        return None
    session["user_id"] = int(row[0])
    return int(row[0])


def _session_identity_present() -> bool:
    return bool(session.get("user_id") or session.get("user_email"))


def _history_row(row) -> dict:
    searched_at = row[5]
    return {
        "search_id": row[0],
        "search_keyword": row[1],
        "search_type": row[2],
        "search_result": row[3],
        "media_type": row[4],
        "searched_at": (
            searched_at.isoformat() if hasattr(searched_at, "isoformat")
            else str(searched_at) if searched_at else None
        ),
        "has_media": bool(row[6]),
    }


@history_bp.get("/api/search-history")
def list_search_history():
    if not _session_identity_present():
        return jsonify({"error": "Authentication required."}), 401

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = _resolve_user_id(cursor)
        if user_id is None:
            return jsonify({"error": "Could not resolve the signed-in user."}), 401

        cursor.execute(
            """
            SELECT search_id, search_keyword, search_type, search_result,
                   media_type, searched_at, media_path
            FROM search_history
            WHERE user_id = %s
            ORDER BY searched_at DESC, search_id DESC
            LIMIT 100
            """,
            (user_id,),
        )
        return jsonify({"items": [_history_row(row) for row in cursor.fetchall()]})
    except Exception:
        current_app.logger.exception("Failed to load translation search history")
        return jsonify({"error": "Could not load translation history."}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@history_bp.post("/api/search-history")
def create_search_history():
    if not _session_identity_present():
        return jsonify({"error": "Authentication required."}), 401

    if request.content_length and request.content_length > MAX_MEDIA_BYTES + 1024 * 1024:
        return jsonify({"error": "Media must be smaller than 25 MB."}), 413

    is_multipart = request.mimetype == "multipart/form-data"
    payload = request.form if is_multipart else (request.get_json(silent=True) or {})
    search_type = str(payload.get("search_type") or "").strip().lower()
    search_keyword = str(payload.get("search_keyword") or "").strip()
    search_result = payload.get("search_result")
    if not isinstance(search_result, str):
        return jsonify({"error": "search_result must be a string."}), 400
    search_result = search_result.strip()
    if search_type not in HISTORY_TYPES:
        return jsonify({"error": "Invalid search_type."}), 400
    if not search_result:
        return jsonify({"error": "search_result cannot be empty."}), 400
    if len(search_keyword) > 4000 or len(search_result) > MAX_RESULT_CHARS:
        return jsonify({"error": "History content is too long."}), 413

    media = request.files.get("media") if is_multipart else None
    if search_type == "upload" and media is None:
        return jsonify({"error": "Upload history requires the original media file."}), 400
    if search_type != "upload" and media is not None:
        return jsonify({"error": "Media files are only accepted for upload history."}), 400

    media_path = None
    media_type = None
    saved_path = None
    media_extension = None
    if media is not None:
        original_name = secure_filename(media.filename or "")
        extension = Path(original_name).suffix.lower()
        content_type = MEDIA_EXTENSIONS.get(extension)
        if not content_type:
            return jsonify({"error": "Unsupported image or video file type."}), 400
        media_type = "video" if content_type.startswith("video/") else "image"
        media.stream.seek(0, 2)
        media_size = media.stream.tell()
        media.stream.seek(0)
        if media_size <= 0 or media_size > MAX_MEDIA_BYTES:
            return jsonify({"error": "Media must be smaller than 25 MB."}), 413

        media_extension = extension

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = _resolve_user_id(cursor)
        if user_id is None:
            return jsonify({"error": "Could not resolve the signed-in user."}), 401

        if media is not None and media_extension:
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            media_path = f"{uuid.uuid4().hex}{media_extension}"
            saved_path = MEDIA_DIR / media_path
            media.save(saved_path)

        cursor.execute(
            """
            INSERT INTO search_history
                (user_id, search_keyword, search_type, search_result,
                 media_path, media_type, searched_at)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                user_id,
                search_keyword or None,
                search_type,
                search_result,
                media_path,
                media_type,
            ),
        )
        conn.commit()
        return jsonify({"search_id": cursor.lastrowid}), 201
    except Exception:
        if conn:
            conn.rollback()
        if saved_path:
            saved_path.unlink(missing_ok=True)
        current_app.logger.exception("Failed to save translation search history")
        return jsonify({"error": "Could not save translation history."}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@history_bp.get("/api/search-history/<int:search_id>/media")
def get_search_history_media(search_id: int):
    if not _session_identity_present():
        return jsonify({"error": "Authentication required."}), 401

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = _resolve_user_id(cursor)
        if user_id is None:
            return jsonify({"error": "Could not resolve the signed-in user."}), 401

        cursor.execute(
            """
            SELECT media_path, media_type
            FROM search_history
            WHERE search_id = %s AND user_id = %s
            LIMIT 1
            """,
            (search_id, user_id),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return jsonify({"error": "History media not found."}), 404

        media_path = (MEDIA_DIR / Path(row[0]).name).resolve()
        if media_path.parent != MEDIA_DIR.resolve() or not media_path.is_file():
            return jsonify({"error": "History media not found."}), 404
        mimetype = MEDIA_EXTENSIONS.get(media_path.suffix.lower())
        if not mimetype:
            return jsonify({"error": "History media type is invalid."}), 415
        return send_file(media_path, mimetype=mimetype, conditional=True, max_age=0)
    except Exception:
        current_app.logger.exception("Failed to load translation history media")
        return jsonify({"error": "Could not load history media."}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
