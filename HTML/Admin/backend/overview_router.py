"""Signrest admin Overview endpoint (FastAPI + MySQL, existing tables only).

Plug in your own `get_db` (returns a DB-API cursor-capable connection, e.g. PyMySQL)
and `require_admin` (validates the session cookie and users.role == 'admin',
raising 401 / 403). Then: app.include_router(router)
"""
from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query

from .dependencies import get_db, require_admin  # <- your existing helpers

router = APIRouter(prefix="/api/admin", tags=["admin-overview"])

# Map your stored translations.mode values to the two display names.
SIGN_TO_TEXT = ("sign_to_text", "Sign to Text")
TEXT_TO_SIGN = ("text_to_sign", "Text to Sign")
PENDING_FEEDBACK = ("pending", "new")


def iso(value):
    # MySQL DATETIME has no timezone; set your server's real zone here.
    return value.isoformat() if isinstance(value, (datetime, date)) else None


def display_mode(raw):
    if raw in SIGN_TO_TEXT:
        return "Sign to Text"
    if raw in TEXT_TO_SIGN:
        return "Text to Sign"
    return raw or "Unknown"


@router.get("/overview")
def overview(days: int = Query(7, ge=7, le=30), admin=Depends(require_admin), db=Depends(get_db)):
    if days not in (7, 30):
        raise HTTPException(422, "days must be 7 or 30")
    start = date.today() - timedelta(days=days - 1)
    try:
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            users = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM translations")
            translations = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM user_sessions WHERE revoked_at IS NULL AND expires_at > NOW()")
            active_sessions = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM feedbacks WHERE status IN (%s, %s)", PENDING_FEEDBACK)
            pending = cur.fetchone()[0]

            cur.execute("""SELECT model_name, model_version, total_samples, accuracy_percentage, evaluated_at
                           FROM model_accuracy ORDER BY evaluated_at DESC, accuracy_id DESC LIMIT 1""")
            m = cur.fetchone()
            model = None if not m else {
                "model_name": m[0], "model_version": m[1], "total_samples": m[2],
                "accuracy_percentage": float(m[3]) if m[3] is not None else None, "evaluated_at": iso(m[4]),
            }

            cur.execute("""SELECT DATE(created_at) d, mode, COUNT(*) FROM translations
                           WHERE created_at >= %s GROUP BY d, mode""", (start,))
            buckets = {start + timedelta(days=i): [0, 0] for i in range(days)}
            for d, mode, n in cur.fetchall():
                if d in buckets:
                    name = display_mode(mode)
                    if name == "Sign to Text":
                        buckets[d][0] += n
                    elif name == "Text to Sign":
                        buckets[d][1] += n
            activity = [{"date": d.isoformat(), "sign_to_text": v[0], "text_to_sign": v[1]} for d, v in sorted(buckets.items())]

            cur.execute("""SELECT t.translation_id, COALESCE(u.full_name, u.username), t.mode, t.input_type,
                                  t.predicted_text, t.confidence_score, t.created_at
                           FROM translations t LEFT JOIN users u ON u.user_id = t.user_id
                           WHERE t.confidence_score < 75
                           ORDER BY t.created_at DESC, t.translation_id DESC LIMIT 5""")
            low = [{"translation_id": r[0], "user_name": r[1], "mode": display_mode(r[2]), "input_type": r[3],
                    "predicted_text": r[4], "confidence_score": float(r[5]) if r[5] is not None else None,
                    "created_at": iso(r[6])} for r in cur.fetchall()]

            cur.execute("""SELECT f.feedback_id, COALESCE(u.full_name, u.username), f.rating_score, f.message, f.status, f.created_at
                           FROM feedbacks f LEFT JOIN users u ON u.user_id = f.user_id
                           ORDER BY f.created_at DESC, f.feedback_id DESC LIMIT 5""")
            feedback = [{"feedback_id": r[0], "user_name": r[1], "rating_score": r[2], "message": r[3],
                         "status": r[4], "created_at": iso(r[5])} for r in cur.fetchall()]
    except Exception:
        # Log privately; never return SQL or credentials to the browser.
        raise HTTPException(500, "Could not load overview")

    return {
        "admin_name": admin.get("full_name") or admin.get("username") or "",
        "totals": {"users": users, "translations": translations,
                   "active_sessions": active_sessions, "pending_feedback": pending},
        "model": model, "activity": activity,
        "low_confidence": low, "recent_feedback": feedback,
    }
