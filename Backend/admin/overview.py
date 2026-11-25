from datetime import date, datetime, timedelta

from flask import Blueprint, current_app, jsonify, request, session

from config.database import get_db_connection

admin_overview_bp = Blueprint("admin_overview", __name__)

SIGN_TO_TEXT = {"sign_to_text", "Sign to Text"}
TEXT_TO_SIGN = {"text_to_sign", "Text to Sign"}
PENDING_FEEDBACK = ("pending", "new")


def _iso(value):
    return value.isoformat() if isinstance(value, (date, datetime)) else None


def _display_mode(value):
    if value in SIGN_TO_TEXT:
        return "Sign to Text"
    if value in TEXT_TO_SIGN:
        return "Text to Sign"
    return value or "Unknown"


@admin_overview_bp.get("/api/admin/overview")
def admin_overview():
    if not session.get("user_email"):
        return jsonify({"message": "Authentication required."}), 401
    if session.get("user_role") != "admin":
        return jsonify({"message": "Administrator access required."}), 403

    raw_days = request.args.get("days", "7")
    try:
        days = int(raw_days)
    except ValueError:
        return jsonify({"message": "days must be 7 or 30"}), 422
    if days not in (7, 30):
        return jsonify({"message": "days must be 7 or 30"}), 422

    start = date.today() - timedelta(days=days - 1)
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        # One round trip for all four totals: each statement to the hosted
        # database costs ~80 ms, so four separate COUNTs added a quarter second.
        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM users) AS total_users,
                (SELECT COUNT(*) FROM translations) AS total_translations,
                (SELECT COUNT(*) FROM user_sessions
                  WHERE revoked_at IS NULL AND expires_at > NOW()) AS active_sessions,
                (SELECT COUNT(*) FROM feedbacks WHERE status IN (%s, %s)) AS pending_feedback
            """,
            PENDING_FEEDBACK,
        )
        users, translations, active_sessions, pending_feedback = cursor.fetchone()

        cursor.execute(
            """
            SELECT model_name, model_version, total_samples, accuracy_percentage, evaluated_at
            FROM model_accuracy
            ORDER BY evaluated_at DESC, accuracy_id DESC
            LIMIT 1
            """
        )
        model_row = cursor.fetchone()
        model = None
        if model_row:
            model = {
                "model_name": model_row[0],
                "model_version": model_row[1],
                "total_samples": model_row[2],
                "accuracy_percentage": (
                    float(model_row[3]) if model_row[3] is not None else None
                ),
                "evaluated_at": _iso(model_row[4]),
            }

        cursor.execute(
            """
            SELECT DATE(created_at), mode, COUNT(*)
            FROM translations
            WHERE created_at >= %s
            GROUP BY DATE(created_at), mode
            """,
            (start,),
        )
        buckets = {start + timedelta(days=index): [0, 0] for index in range(days)}
        for activity_date, mode, count in cursor.fetchall():
            if activity_date not in buckets:
                continue
            display_mode = _display_mode(mode)
            if display_mode == "Sign to Text":
                buckets[activity_date][0] += count
            elif display_mode == "Text to Sign":
                buckets[activity_date][1] += count
        activity = [
            {
                "date": activity_date.isoformat(),
                "sign_to_text": values[0],
                "text_to_sign": values[1],
            }
            for activity_date, values in sorted(buckets.items())
        ]

        cursor.execute(
            """
            SELECT t.translation_id, u.username, t.mode,
                   t.input_type, t.predicted_text, t.confidence_score, t.created_at
            FROM translations t
            LEFT JOIN users u ON u.user_id = t.user_id
            WHERE t.confidence_score < 75
            ORDER BY t.created_at DESC, t.translation_id DESC
            LIMIT 5
            """
        )
        low_confidence = [
            {
                "translation_id": row[0],
                "user_name": row[1],
                "mode": _display_mode(row[2]),
                "input_type": row[3],
                "predicted_text": row[4],
                "confidence_score": float(row[5]) if row[5] is not None else None,
                "created_at": _iso(row[6]),
            }
            for row in cursor.fetchall()
        ]

        cursor.execute(
            """
            SELECT f.feedback_id, u.username, f.rating_score,
                   f.message, f.status, f.created_at
            FROM feedbacks f
            LEFT JOIN users u ON u.user_id = f.user_id
            ORDER BY f.created_at DESC, f.feedback_id DESC
            LIMIT 5
            """
        )
        recent_feedback = [
            {
                "feedback_id": row[0],
                "user_name": row[1],
                "rating_score": row[2],
                "message": row[3],
                "status": row[4],
                "created_at": _iso(row[5]),
            }
            for row in cursor.fetchall()
        ]
    except Exception:
        current_app.logger.exception("Failed to load admin overview")
        return jsonify({"message": "Could not load overview"}), 500
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    email = session["user_email"]
    admin_name = (
        session.get("display_name")
        or session.get("full_name")
        or session.get("username")
        or email.partition("@")[0]
    )
    return jsonify(
        {
            "admin_name": admin_name,
            "totals": {
                "users": users,
                "translations": translations,
                "active_sessions": active_sessions,
                "pending_feedback": pending_feedback,
            },
            "model": model,
            "activity": activity,
            "low_confidence": low_confidence,
            "recent_feedback": recent_feedback,
        }
    )
