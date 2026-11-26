from datetime import datetime, timezone
from flask import Blueprint, current_app, jsonify, request, session
from config.cloudinary import upload_to_cloudinary
from config.database.db_connection import get_db_connection

cursor_bp = Blueprint("cursor_effects", __name__)

ALLOWED_EXTS = {".png", ".webp", ".gif"}


def _ensure_seen_table(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS cursor_effect_selections (
        user_id INT NOT NULL, effect_id INT NOT NULL,
        PRIMARY KEY (user_id, effect_id)
    )""")


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


@cursor_bp.route("/api/cursor-effects", methods=["GET"])
def list_cursor_effects():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    if not user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "User record not found."}), 404
    _ensure_seen_table(cur)

    # Fetch global effects + this user's own custom stickers, newest first (`ORDER BY effect_id DESC`)
    cur.execute(
        """
        SELECT e.effect_id, e.effect_name, e.effect_path, e.effect_type, e.created_by_user_id,
               s.effect_id IS NOT NULL
        FROM cursor_effects e
        LEFT JOIN cursor_effect_selections s ON s.effect_id = e.effect_id AND s.user_id = %s
        WHERE e.status = 'active'
          AND (e.created_by_user_id IS NULL OR e.created_by_user_id = %s)
        ORDER BY e.effect_id DESC
        """,
        (user_id, user_id),
    )
    rows = cur.fetchall()
    effects = [
        {
            "effect_id": r[0],
            "effect_name": r[1],
            "effect_path": r[2],
            "effect_type": r[3] or "both",
            "is_custom": r[4] is not None,
            "is_new": r[4] is None and not bool(r[5]),
        }
        for r in rows
    ]

    selected_trail_id = None
    selected_click_id = None
    trail_size = 24
    click_size = 26

    if user_id:
        cur.execute(
            """
            SELECT trail_effect_id, click_effect_id, trail_effect_size, click_effect_size
            FROM users
            WHERE user_id = %s
            """,
            (user_id,),
        )
        urow = cur.fetchone()
        if urow:
            selected_trail_id = urow[0]
            selected_click_id = urow[1]
            trail_size = urow[2] or 24
            click_size = urow[3] or 26

    for selected_id in (selected_trail_id, selected_click_id):
        if selected_id:
            cur.execute("INSERT IGNORE INTO cursor_effect_selections (user_id, effect_id) VALUES (%s, %s)", (user_id, selected_id))
    conn.commit()

    cur.close()
    conn.close()

    effect_map = {e["effect_id"]: e for e in effects}
    for selected_id in (selected_trail_id, selected_click_id):
        if selected_id in effect_map:
            effect_map[selected_id]["is_new"] = False
    trail_obj = effect_map.get(selected_trail_id)
    click_obj = effect_map.get(selected_click_id)
    if trail_obj:
        trail_obj = {**trail_obj, "size": trail_size}
    if click_obj:
        click_obj = {**click_obj, "size": click_size}

    return jsonify(
        {
            "effects": effects,
            "trail_effect_id": selected_trail_id,
            "click_effect_id": selected_click_id,
            "trail_effect_size": trail_size,
            "click_effect_size": click_size,
            "trail_effect": trail_obj,
            "click_effect": click_obj,
        }
    ), 200


@cursor_bp.route("/api/cursor-effects/<int:effect_id>/select", methods=["POST"])
def mark_effect_selected(effect_id):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        user_id = _resolve_user_id(cur)
        if not user_id:
            return jsonify({"error": "User record not found."}), 404
        _ensure_seen_table(cur)
        cur.execute("""SELECT effect_id FROM cursor_effects WHERE effect_id = %s AND status = 'active'
                       AND (created_by_user_id IS NULL OR created_by_user_id = %s)""", (effect_id, user_id))
        if not cur.fetchone():
            return jsonify({"error": "Effect not found."}), 404
        cur.execute("INSERT IGNORE INTO cursor_effect_selections (user_id, effect_id) VALUES (%s, %s)", (user_id, effect_id))
        conn.commit()
        return jsonify({"message": "Selection recorded."})
    finally:
        cur.close()
        conn.close()


@cursor_bp.route("/api/cursor-effects/<int:effect_id>", methods=["DELETE"])
def delete_user_cursor_effect(effect_id):
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        user_id = _resolve_user_id(cur)
        if not user_id:
            return jsonify({"error": "User record not found."}), 404
        cur.execute("UPDATE cursor_effects SET status = 'inactive' WHERE effect_id = %s AND created_by_user_id = %s AND status = 'active'", (effect_id, user_id))
        if cur.rowcount == 0:
            return jsonify({"error": "Effect not found."}), 404
        cur.execute("""UPDATE users SET trail_effect_id = CASE WHEN trail_effect_id = %s THEN NULL ELSE trail_effect_id END,
                       click_effect_id = CASE WHEN click_effect_id = %s THEN NULL ELSE click_effect_id END
                       WHERE user_id = %s""", (effect_id, effect_id, user_id))
        conn.commit()
        return jsonify({"message": "Effect deleted."})
    except Exception:
        conn.rollback()
        current_app.logger.exception("Could not delete cursor effect")
        return jsonify({"error": "Could not delete the effect. Please try again."}), 500
    finally:
        cur.close()
        conn.close()


@cursor_bp.route("/api/cursor-effects/upload", methods=["POST"])
def upload_user_custom_sticker():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    file = request.files.get("sticker")
    if not file or not file.filename:
        return jsonify({"error": "Please select a background-removed PNG, WebP, or GIF file."}), 400

    filename = file.filename.strip()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": "Only PNG, WebP, or GIF stickers are supported."}), 400

    effect_name = (request.form.get("effect_name") or filename.rsplit(".", 1)[0]).strip()[:80] or "Custom Sticker"

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    if not user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "User record not found."}), 404

    secure_url = upload_to_cloudinary(
        file_source=file.stream,
        folder="cursor_effects",
        filename=filename,
        resource_type="image",
    )

    cur.execute(
        """
        INSERT INTO cursor_effects (effect_name, effect_path, effect_type, created_by_user_id, status, created_at)
        VALUES (%s, %s, 'both', %s, 'active', %s)
        """,
        (effect_name, secure_url, user_id, datetime.now(timezone.utc)),
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()

    return jsonify(
        {
            "effect": {
                "effect_id": new_id,
                "effect_name": effect_name,
                "effect_path": secure_url,
                "effect_type": "both",
                "is_custom": True,
            }
        }
    ), 201


@cursor_bp.route("/api/user/cursor-effects", methods=["POST"])
def save_user_cursor_effects():
    if not session.get("user_email") and not session.get("user_id"):
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    trail_id = data.get("trail_effect_id") or None
    click_id = data.get("click_effect_id") or None
    trail_size = max(12, min(64, int(data.get("trail_effect_size") or 24)))
    click_size = max(14, min(72, int(data.get("click_effect_size") or 26)))

    conn = get_db_connection()
    cur = conn.cursor()
    user_id = _resolve_user_id(cur)
    if not user_id:
        cur.close()
        conn.close()
        return jsonify({"error": "User record not found."}), 404

    for effect_id, effect_type in ((trail_id, "trail"), (click_id, "click")):
        if effect_id:
            cur.execute("""SELECT effect_id FROM cursor_effects WHERE effect_id = %s AND status = 'active'
                           AND (created_by_user_id IS NULL OR created_by_user_id = %s)
                           AND (effect_type = %s OR effect_type = 'both')""", (effect_id, user_id, effect_type))
            if not cur.fetchone():
                cur.close()
                conn.close()
                return jsonify({"error": "A selected cursor effect is no longer available."}), 400

    cur.execute(
        """
        UPDATE users
        SET trail_effect_id = %s,
            click_effect_id = %s,
            trail_effect_size = %s,
            click_effect_size = %s
        WHERE user_id = %s
        """,
        (trail_id, click_id, trail_size, click_size, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify(
        {
            "message": "Cursor effects saved.",
            "trail_effect_id": trail_id,
            "click_effect_id": click_id,
            "trail_effect_size": trail_size,
            "click_effect_size": click_size,
        }
    ), 200
