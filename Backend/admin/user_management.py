"""
Admin User Management + Session Management for Signrest (Flask blueprint).

Location in the project:  Backend/admin/user_management.py

What this module does
---------------------
* Admin API under /api/admin/users and /api/admin/sessions (see the routes at the bottom).
* Server-side web sessions on top of the existing `user_sessions` table, so an admin can
  really revoke a login. Flask's signed cookie alone can't be revoked, so:
    - start_user_session()   is called by the login flows after a successful login,
    - session_guard()        runs in app.py's before_request on every request,
    - end_current_session()  is called by /api/logout.
  Only a SHA-256 hash of the session token is stored (column `refresh_token_hash`).
* ensure_user_management_schema() adds the few missing columns/tables at startup, the same
  way login_process.ensure_login_columns() does.

Existing tables used: users, user_sessions, login_history, translations, feedbacks.
Nothing here ever returns password hashes or session tokens.

All datetimes in the database are treated as naive UTC (this is what login_process.py and
profile.py write) and are sent to the browser as ISO strings ending in "Z".
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Blueprint, current_app, jsonify, redirect, request, session

from config.database import get_db_connection

admin_users_bp = Blueprint("admin_users", __name__)

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
SESSION_LIFETIME = timedelta(hours=24)   # how long a web login stays valid server-side
LAST_SEEN_WRITE_SECONDS = 60             # write last_seen_at at most once a minute per session
STATIC_PREFIXES = ("/css/", "/js/", "/assets/")

STATUS_FILTERS = {"all", "active", "inactive", "banned", "pending"}
SESSION_STATUS_FILTERS = {"all", "active", "expired", "revoked", "signed_out"}
HISTORY_EVENTS = {"all", "login", "logout", "expired", "revoked", "failed_login"}

# action -> (new status, statuses it may start from, past-tense word for messages)
ACCOUNT_ACTIONS = {
    "ban": ("banned", {"active", "inactive", "pending"}, "banned"),
    "deactivate": ("inactive", {"active"}, "deactivated"),
    "reactivate": ("active", {"inactive", "banned"}, "reactivated"),
}

SIGN_TO_TEXT = ("sign_to_text", "Sign to Text")
TEXT_TO_SIGN = ("text_to_sign", "Text to Sign")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(value):
    """Naive-UTC datetime -> ISO string with a Z suffix (so browsers convert it correctly)."""
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat() + "Z"
    return None


def _err(message: str, status: int):
    return jsonify({"success": False, "message": message}), status


def _display_name(username, email) -> str:
    # The users table has no full_name column, so the username (or the e-mail prefix) is the name.
    return (username or "").strip() or (email or "").partition("@")[0] or "Unknown user"


def _like(value: str) -> str:
    """Escape LIKE wildcards in user input, then wrap it for a contains-search."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _paging(default_size: int = 12, maximum: int = 50):
    try:
        page = int(request.args.get("page", 1))
        size = int(request.args.get("page_size", default_size))
    except ValueError:
        return None
    if page < 1 or size < 1 or size > maximum:
        return None
    return page, size


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must use the YYYY-MM-DD format.")


@contextmanager
def _db():
    """Yield (connection, dict cursor); roll back on error and always close."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        yield conn, cursor
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001 - connection may already be gone
            pass
        raise
    finally:
        cursor.close()
        conn.close()


def parse_user_agent(user_agent: str | None):
    """Return (browser, operating system, device type) from a User-Agent string."""
    ua = user_agent or ""
    low = ua.lower()

    if "edg/" in low or "edge/" in low:
        browser = "Edge"
    elif "opr/" in low or "opera" in low:
        browser = "Opera"
    elif "firefox/" in low or "fxios" in low:
        browser = "Firefox"
    elif "chrome/" in low or "crios" in low:
        browser = "Chrome"
    elif "safari/" in low:
        browser = "Safari"
    else:
        browser = "Unknown browser"

    if "windows" in low:
        os_name = "Windows"
    elif "android" in low:
        os_name = "Android"
    elif "iphone" in low or "ipad" in low:
        os_name = "iOS"
    elif "mac os x" in low or "macintosh" in low:
        os_name = "macOS"
    elif "cros" in low:
        os_name = "ChromeOS"
    elif "linux" in low:
        os_name = "Linux"
    else:
        os_name = "Unknown OS"

    if "ipad" in low or "tablet" in low or ("android" in low and "mobile" not in low):
        device_type = "tablet"
    elif "mobile" in low or "iphone" in low or "android" in low:
        device_type = "mobile"
    elif ua:
        device_type = "desktop"
    else:
        device_type = "other"
    return browser, os_name, device_type


def _device_type_for(row) -> str:
    """Stored device_type, or a guess from the platform for rows created by other clients."""
    stored = (row.get("device_type") or "").lower()
    if stored in {"desktop", "mobile", "tablet"}:
        return stored
    platform = (row.get("platform") or "").lower()
    if platform in {"android", "ios"}:
        return "mobile"
    if platform in {"windows", "macos", "linux", "chromeos", "web"}:
        return "desktop"
    return "other"


def _device_label(raw: str | None) -> str:
    """Readable 'Chrome · Windows' from either a raw User-Agent or an already parsed label."""
    if not raw:
        return "Unknown device"
    if "Mozilla/" in raw:
        browser, os_name, _ = parse_user_agent(raw)
        return f"{browser} · {os_name}"
    return raw


def _session_status(row, now: datetime) -> str:
    if row.get("revoked_at") is not None:
        return "revoked" if row.get("end_reason") == "admin" else "signed_out"
    if row.get("expires_at") is not None and row["expires_at"] <= now:
        return "expired"
    return "active"


def _user_summary(row) -> dict:
    return {
        "user_id": row["user_id"],
        "username": row.get("username"),
        "display_name": _display_name(row.get("username"), row.get("email")),
        "email": row.get("email"),
        "profile_photo": row.get("profile_photo"),
        "role": (row.get("role") or "user").lower(),
        "status": (row.get("status") or "").lower(),
        "created_at": _iso(row.get("created_at")),
    }


# --------------------------------------------------------------------------------------
# Schema (runs once at startup, like ensure_login_columns)
# --------------------------------------------------------------------------------------
def ensure_user_management_schema() -> None:
    """
    Adds what this module needs and the project doesn't have yet:
      * user_sessions.ip_address / device_type / revoked_by / end_reason
      * account_activity_log table (admin actions on accounts)
      * 'inactive' in users.status if that column is an ENUM without it
    Safe to run on every start.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        def has_column(table: str, column: str) -> bool:
            cursor.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (column,))
            return cursor.fetchone() is not None

        for column, ddl in (
            ("ip_address", "VARCHAR(45) NULL"),
            ("device_type", "VARCHAR(20) NULL"),
            ("revoked_by", "INT NULL"),
            ("end_reason", "VARCHAR(20) NULL"),
        ):
            if not has_column("user_sessions", column):
                cursor.execute(f"ALTER TABLE user_sessions ADD COLUMN {column} {ddl}")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS account_activity_log (
                activity_id   INT AUTO_INCREMENT PRIMARY KEY,
                user_id       INT NOT NULL,
                admin_id      INT NULL,
                activity_type VARCHAR(30) NOT NULL,
                description   VARCHAR(255) NULL,
                created_at    DATETIME NOT NULL,
                INDEX idx_activity_user (user_id, created_at)
            )
            """
        )

        # users.status must be able to hold 'inactive'.
        cursor.execute("SHOW COLUMNS FROM users LIKE 'status'")
        status_col = cursor.fetchone()  # (Field, Type, Null, Key, Default, Extra)
        if status_col and str(status_col[1]).lower().startswith("enum("):
            values = re.findall(r"'([^']*)'", str(status_col[1]))
            if "inactive" not in values:
                values.append("inactive")
                enum = ",".join("'" + v.replace("'", "''") + "'" for v in values)
                nullable = " NULL" if status_col[2] == "YES" else " NOT NULL"
                default = ""
                if status_col[4] is not None and status_col[4] in values:
                    default = " DEFAULT '" + str(status_col[4]).replace("'", "''") + "'"
                cursor.execute(f"ALTER TABLE users MODIFY status ENUM({enum}){nullable}{default}")

        conn.commit()
        cursor.close()
    except Exception as err:  # noqa: BLE001 - mirrors ensure_login_columns: warn, don't crash the app
        print(f"[User Management Schema Setup Warning]: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()


# --------------------------------------------------------------------------------------
# Server-side web sessions (used by the login flows and app.py)
# --------------------------------------------------------------------------------------
def start_user_session(user_id: int) -> None:
    """
    Create a `user_sessions` row for this login and remember its id + a secret token in the
    Flask cookie. Call right after the login code has filled flask.session.
    Raises on database problems so the caller's existing error handling reports it.
    """
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    browser, os_name, device_type = parse_user_agent(request.headers.get("User-Agent"))
    now = _now()

    with _db() as (conn, cursor):
        cursor.execute(
            """
            INSERT INTO user_sessions
                (user_id, refresh_token_hash, device_name, platform, device_type,
                 ip_address, created_at, expires_at, last_seen_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id, token_hash, browser, os_name, device_type,
                (request.remote_addr or "")[:45], now, now + SESSION_LIFETIME, now,
            ),
        )
        session_id = cursor.lastrowid
        conn.commit()

    if not session_id:
        raise RuntimeError("user_sessions.session_id is not AUTO_INCREMENT; cannot track the session.")
    session["session_id"] = int(session_id)
    session["session_token"] = token
    session["session_seen_at"] = int(time.time())


def end_current_session(reason: str = "logout") -> None:
    """Mark the caller's own session as ended (used by /api/logout). Never raises."""
    session_id = session.get("session_id")
    user_id = session.get("user_id")
    if not session_id or not user_id:
        return
    try:
        with _db() as (conn, cursor):
            cursor.execute(
                """
                UPDATE user_sessions SET revoked_at = %s, end_reason = %s
                WHERE session_id = %s AND user_id = %s AND revoked_at IS NULL
                """,
                (_now(), reason, session_id, user_id),
            )
            conn.commit()
    except Exception:  # noqa: BLE001 - logging out must still work
        current_app.logger.exception("Could not close the server-side session")


def _validate_server_session():
    """None when the cookie's session is valid, otherwise a short reason string."""
    user_id = session.get("user_id")
    session_id = session.get("session_id")
    token = session.get("session_token")
    if not user_id:
        return None  # not logged in: nothing to validate
    if not session_id or not token:
        return "missing"  # signed in before server-side sessions existed

    try:
        with _db() as (conn, cursor):
            cursor.execute(
                """
                SELECT s.refresh_token_hash, s.revoked_at, s.expires_at, u.status
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.session_id = %s AND s.user_id = %s
                """,
                (session_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return "unknown"

            supplied = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
            if not hmac.compare_digest(str(row["refresh_token_hash"] or ""), supplied):
                return "invalid"
            if row["revoked_at"] is not None:
                return "revoked"
            now = _now()
            if row["expires_at"] is None or row["expires_at"] <= now:
                return "expired"
            if (row["status"] or "").lower() not in {"active", "pending"}:
                return "account_disabled"

            if time.time() - session.get("session_seen_at", 0) > LAST_SEEN_WRITE_SECONDS:
                cursor.execute(
                    "UPDATE user_sessions SET last_seen_at = %s WHERE session_id = %s",
                    (now, session_id),
                )
                conn.commit()
                session["session_seen_at"] = int(time.time())
        return None
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Session validation failed")
        return "unavailable"


def session_guard():
    """
    Call from app.py's before_request. Ends the cookie session when its server-side record
    is revoked/expired or the account was banned/deactivated. Returns a response or None.
    """
    if request.method == "OPTIONS" or request.path.startswith(STATIC_PREFIXES):
        return None
    reason = _validate_server_session()
    if reason == "missing":
        try:
            start_user_session(session["user_id"])
        except Exception:  # noqa: BLE001
            current_app.logger.exception("Could not upgrade the existing user session")
            return jsonify({
                "success": False,
                "message": "Service temporarily unavailable."
            }), 503
        reason = _validate_server_session()
    if reason is None:
        return None
    if reason == "unavailable":
        return jsonify({"success": False, "message": "Service temporarily unavailable."}), 503
    session.clear()
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "message": "Your session has ended. Please sign in again."}), 401
    return redirect("/login")


# --------------------------------------------------------------------------------------
# Admin authorization
# --------------------------------------------------------------------------------------
def admin_required(view):
    """
    401 when not signed in, 403 when not an active admin. The role is re-checked in the
    database on every call (the cookie alone is never trusted), and state-changing requests
    must carry the same X-Signrest-Request header the other admin pages send.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id") or not session.get("user_email"):
            return _err("Authentication required.", 401)
        if (session.get("user_role") or "").lower() != "admin":
            return _err("Administrator access required.", 403)
        try:
            with _db() as (_conn, cursor):
                cursor.execute(
                    "SELECT role, status FROM users WHERE user_id = %s", (session["user_id"],)
                )
                row = cursor.fetchone()
        except Exception:  # noqa: BLE001
            current_app.logger.exception("Could not verify administrator")
            return _err("Could not verify administrator access.", 500)
        if (
            not row
            or (row["role"] or "").lower() != "admin"
            or (row["status"] or "").lower() != "active"
        ):
            return _err("Administrator access required.", 403)
        if request.method not in ("GET", "HEAD") and request.headers.get("X-Signrest-Request") != "admin":
            return _err("Missing or invalid request header.", 403)
        return view(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------------
# Login / session event history (shared by the drawer timeline and Session History)
# --------------------------------------------------------------------------------------
# login_history -> login + failed_login; user_sessions -> logout / revoked / expired.
_EVENTS_SQL = """(
    SELECT 'login' AS event, lh.user_id AS user_id, lh.login_time AS at,
           lh.device_info AS device, lh.ip_address AS ip
      FROM login_history lh WHERE lh.login_status = 'success'
    UNION ALL
    SELECT 'failed_login', lh.user_id, lh.login_time, lh.device_info, lh.ip_address
      FROM login_history lh WHERE lh.login_status IN ('login_fail', 'otp_fail')
    UNION ALL
    SELECT IF(s.end_reason = 'admin', 'revoked', 'logout'), s.user_id, s.revoked_at,
           CONCAT_WS(' · ', s.device_name, s.platform), s.ip_address
      FROM user_sessions s WHERE s.revoked_at IS NOT NULL
    UNION ALL
    SELECT 'expired', s.user_id, s.expires_at,
           CONCAT_WS(' · ', s.device_name, s.platform), s.ip_address
      FROM user_sessions s WHERE s.revoked_at IS NULL AND s.expires_at <= %s
) e"""


def _history(cursor, where: list[str], params: list, limit: int, offset: int):
    """Return (rows, total) of events joined with their user, newest first."""
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    base = f"FROM {_EVENTS_SQL} JOIN users u ON u.user_id = e.user_id{clause}"
    now = _now()

    cursor.execute(f"SELECT COUNT(*) AS n {base}", [now, *params])
    total = cursor.fetchone()["n"]

    cursor.execute(
        f"""
        SELECT e.event, e.user_id, e.at, e.device, e.ip, u.username, u.email, u.profile_photo
        {base}
        ORDER BY e.at DESC
        LIMIT %s OFFSET %s
        """,
        [now, *params, limit, offset],
    )
    items = [
        {
            "event": row["event"],
            "at": _iso(row["at"]),
            "device": _device_label(row["device"]),
            "ip_address": row["ip"] or None,
            "user": {
                "user_id": row["user_id"],
                "username": row["username"],
                "display_name": _display_name(row["username"], row["email"]),
                "email": row["email"],
                "profile_photo": row["profile_photo"],
            },
        }
        for row in cursor.fetchall()
    ]
    return items, total


# --------------------------------------------------------------------------------------
# User Management routes
# --------------------------------------------------------------------------------------
@admin_users_bp.get("/api/admin/users")
@admin_required
def list_users():
    q = (request.args.get("q") or "").strip()[:100]
    status = (request.args.get("status") or "all").lower()
    sort = (request.args.get("sort") or "newest").lower()
    paging = _paging(default_size=12, maximum=48)
    if status not in STATUS_FILTERS or sort not in {"newest", "oldest"} or paging is None:
        return _err("Invalid filter, sort or paging value.", 422)
    page, size = paging

    where, params = [], []
    if status != "all":
        where.append("LOWER(u.status) = %s")
        params.append(status)
    if q:
        where.append("(u.username LIKE %s OR u.email LIKE %s)")
        params += [_like(q), _like(q)]
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    direction = "DESC" if sort == "newest" else "ASC"

    try:
        with _db() as (_conn, cursor):
            cursor.execute(f"SELECT COUNT(*) AS n FROM users u{clause}", params)
            total = cursor.fetchone()["n"]
            cursor.execute(
                f"""
                SELECT u.user_id, u.username, u.email, u.profile_photo, u.role, u.status, u.created_at
                FROM users u{clause}
                ORDER BY u.created_at {direction}, u.user_id {direction}
                LIMIT %s OFFSET %s
                """,
                [*params, size, (page - 1) * size],
            )
            items = [_user_summary(row) for row in cursor.fetchall()]

            cursor.execute("SELECT LOWER(status) AS s, COUNT(*) AS n FROM users GROUP BY LOWER(status)")
            counts = {name: 0 for name in STATUS_FILTERS}
            for row in cursor.fetchall():
                if row["s"] in counts:
                    counts[row["s"]] = row["n"]
            counts["all"] = sum(v for k, v in counts.items() if k != "all")
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to list users")
        return _err("Could not load users.", 500)

    return jsonify({
        "success": True, "items": items, "total": total,
        "page": page, "page_size": size, "counts": counts,
    })


@admin_users_bp.get("/api/admin/users/<int:user_id>")
@admin_required
def get_user_details(user_id: int):
    now = _now()
    try:
        with _db() as (_conn, cursor):
            cursor.execute(
                """
                SELECT user_id, username, email, profile_photo, role, status, created_at,
                       updated_at, failed_login_attempts, locked_until
                FROM users WHERE user_id = %s
                """,
                (user_id,),
            )
            user = cursor.fetchone()
            if not user:
                return _err("User not found.", 404)

            cursor.execute(
                """
                SELECT COUNT(*) AS n FROM user_sessions
                WHERE user_id = %s AND revoked_at IS NULL AND expires_at > %s
                """,
                (user_id, now),
            )
            active_sessions = cursor.fetchone()["n"]

            cursor.execute(
                """
                SELECT login_time, device_info, ip_address FROM login_history
                WHERE user_id = %s AND login_status = 'success'
                ORDER BY login_time DESC LIMIT 5
                """,
                (user_id,),
            )
            logins = []
            for row in cursor.fetchall():
                browser, os_name, device_type = parse_user_agent(row["device_info"])
                logins.append({
                    "at": _iso(row["login_time"]), "browser": browser, "os": os_name,
                    "device_type": device_type, "ip_address": row["ip_address"] or None,
                })

            cursor.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(mode IN (%s, %s)) AS sign_to_text,
                       SUM(mode IN (%s, %s)) AS text_to_sign,
                       MAX(created_at) AS last_at
                FROM translations WHERE user_id = %s
                """,
                (*SIGN_TO_TEXT, *TEXT_TO_SIGN, user_id),
            )
            tr = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) AS n FROM feedbacks WHERE user_id = %s", (user_id,))
            feedback_count = cursor.fetchone()["n"]

            # Timeline: registration + login/session events + admin actions, newest first.
            events, _ = _history(cursor, ["e.user_id = %s"], [user_id], 20, 0)
            timeline = [
                {"type": ev["event"], "at": ev["at"], "detail": ev["device"]} for ev in events
            ]
            cursor.execute(
                """
                SELECT a.activity_type, a.description, a.created_at, adm.username AS admin_name,
                       adm.email AS admin_email
                FROM account_activity_log a
                LEFT JOIN users adm ON adm.user_id = a.admin_id
                WHERE a.user_id = %s ORDER BY a.created_at DESC LIMIT 20
                """,
                (user_id,),
            )
            for row in cursor.fetchall():
                by = _display_name(row["admin_name"], row["admin_email"]) if row["admin_email"] else "an administrator"
                detail = f"By {by}" + (f" · {row['description']}" if row["description"] else "")
                timeline.append({"type": row["activity_type"], "at": _iso(row["created_at"]), "detail": detail})
            timeline.append({"type": "registered", "at": _iso(user["created_at"]), "detail": None})
            timeline.sort(key=lambda item: item["at"] or "", reverse=True)
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to load user details")
        return _err("Could not load user details.", 500)

    status = (user["status"] or "").lower()
    locked = user["locked_until"] is not None and user["locked_until"] > now
    item = _user_summary(user)
    item.update({
        "updated_at": _iso(user["updated_at"]),
        # Accounts stay 'pending' until the signup OTP is confirmed.
        "email_verified": status != "pending",
        "security": {
            "failed_login_attempts": int(user["failed_login_attempts"] or 0),
            "locked_until": _iso(user["locked_until"]) if locked else None,
        },
        "sessions": {
            "active_count": active_sessions,
            "last_login": logins[0] if logins else None,
            "recent_logins": logins,
        },
        "learning": {
            "translations_total": int(tr["total"] or 0),
            "sign_to_text": int(tr["sign_to_text"] or 0),
            "text_to_sign": int(tr["text_to_sign"] or 0),
            "last_translation_at": _iso(tr["last_at"]),
            "feedback_count": feedback_count,
        },
        "activity": timeline[:25],
    })
    return jsonify({"success": True, "item": item})


@admin_users_bp.post("/api/admin/users/<int:user_id>/<any(ban,deactivate,reactivate):action>")
@admin_required
def change_account_status(user_id: int, action: str):
    new_status, allowed_from, verb = ACCOUNT_ACTIONS[action]
    admin_id = session["user_id"]
    if user_id == admin_id:
        return _err("You can't change your own account.", 409)

    data = request.get_json(silent=True) or {}
    reason = data.get("reason")
    reason = reason.strip()[:200] if isinstance(reason, str) and reason.strip() else None
    now = _now()

    try:
        with _db() as (conn, cursor):
            cursor.execute(
                "SELECT user_id, role, status FROM users WHERE user_id = %s FOR UPDATE", (user_id,)
            )
            target = cursor.fetchone()
            if not target:
                return _err("User not found.", 404)
            if (target["role"] or "").lower() == "admin":
                return _err("Administrator accounts can't be changed here.", 403)

            current = (target["status"] or "").lower()
            if current == new_status:
                return _err(f"This account is already {new_status}.", 409)
            if current not in allowed_from:
                return _err(f"A {current} account can't be {verb}.", 409)

            if action == "reactivate":
                cursor.execute(
                    """
                    UPDATE users SET status = %s, failed_login_attempts = 0, locked_until = NULL,
                                     updated_at = %s WHERE user_id = %s
                    """,
                    (new_status, now, user_id),
                )
            else:
                cursor.execute(
                    "UPDATE users SET status = %s, updated_at = %s WHERE user_id = %s",
                    (new_status, now, user_id),
                )

            revoked = 0
            if action in ("ban", "deactivate"):
                # A disabled account must not keep working logins.
                cursor.execute(
                    """
                    UPDATE user_sessions SET revoked_at = %s, revoked_by = %s, end_reason = 'admin'
                    WHERE user_id = %s AND revoked_at IS NULL AND expires_at > %s
                    """,
                    (now, admin_id, user_id, now),
                )
                revoked = cursor.rowcount

            cursor.execute(
                """
                INSERT INTO account_activity_log (user_id, admin_id, activity_type, description, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, admin_id, verb, reason, now),
            )
            conn.commit()
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to change account status")
        return _err("The account could not be updated. Nothing was changed.", 500)

    return jsonify({
        "success": True,
        "message": f"Account {verb}." + (f" {revoked} session(s) ended." if revoked else ""),
        "item": {"user_id": user_id, "status": new_status},
        "sessions_revoked": revoked,
    })


# --------------------------------------------------------------------------------------
# Session Management routes
# --------------------------------------------------------------------------------------
@admin_users_bp.get("/api/admin/sessions/stats")
@admin_required
def session_stats():
    now = _now()
    try:
        with _db() as (_conn, cursor):
            cursor.execute(
                """
                SELECT
                  SUM(revoked_at IS NULL AND expires_at > %s)                AS active,
                  SUM(revoked_at IS NULL AND expires_at <= %s)               AS expired,
                  SUM(revoked_at IS NOT NULL AND end_reason = 'admin')       AS revoked,
                  COUNT(DISTINCT CASE WHEN revoked_at IS NULL AND expires_at > %s
                                      THEN user_id END)                      AS unique_users
                FROM user_sessions
                """,
                (now, now, now),
            )
            totals = cursor.fetchone()
            cursor.execute(
                """
                SELECT device_type, platform, device_name FROM user_sessions
                WHERE revoked_at IS NULL AND expires_at > %s LIMIT 5000
                """,
                (now,),
            )
            devices = {"desktop": 0, "mobile": 0, "tablet": 0, "other": 0}
            browsers: dict[str, int] = {}
            for row in cursor.fetchall():
                devices[_device_type_for(row)] += 1
                name = row["device_name"] or "Unknown"
                browsers[name] = browsers.get(name, 0) + 1
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to load session stats")
        return _err("Could not load session statistics.", 500)

    return jsonify({
        "success": True,
        "totals": {
            "active": int(totals["active"] or 0),
            "expired": int(totals["expired"] or 0),
            "revoked": int(totals["revoked"] or 0),
            "unique_active_users": int(totals["unique_users"] or 0),
        },
        "devices": devices,
        "browsers": [
            {"name": name, "count": count}
            for name, count in sorted(browsers.items(), key=lambda kv: kv[1], reverse=True)[:6]
        ],
    })


@admin_users_bp.get("/api/admin/sessions")
@admin_required
def list_sessions():
    status = (request.args.get("status") or "active").lower()
    q = (request.args.get("q") or "").strip()[:100]
    paging = _paging(default_size=12, maximum=48)
    if status not in SESSION_STATUS_FILTERS or paging is None:
        return _err("Invalid status or paging value.", 422)
    page, size = paging
    now = _now()

    where, params = [], []
    if status == "active":
        where.append("s.revoked_at IS NULL AND s.expires_at > %s")
        params.append(now)
    elif status == "expired":
        where.append("s.revoked_at IS NULL AND s.expires_at <= %s")
        params.append(now)
    elif status == "revoked":
        where.append("s.revoked_at IS NOT NULL AND s.end_reason = 'admin'")
    elif status == "signed_out":
        where.append("s.revoked_at IS NOT NULL AND (s.end_reason IS NULL OR s.end_reason <> 'admin')")
    if q:
        where.append("(u.username LIKE %s OR u.email LIKE %s)")
        params += [_like(q), _like(q)]
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    base = f"FROM user_sessions s JOIN users u ON u.user_id = s.user_id{clause}"

    try:
        with _db() as (_conn, cursor):
            cursor.execute(f"SELECT COUNT(*) AS n {base}", params)
            total = cursor.fetchone()["n"]
            # refresh_token_hash is deliberately never selected.
            cursor.execute(
                f"""
                SELECT s.session_id, s.user_id, s.device_name, s.platform, s.device_type, s.ip_address,
                       s.created_at, s.last_seen_at, s.expires_at, s.revoked_at, s.end_reason,
                       u.username, u.email, u.profile_photo
                {base}
                ORDER BY COALESCE(s.last_seen_at, s.created_at) DESC, s.session_id DESC
                LIMIT %s OFFSET %s
                """,
                [*params, size, (page - 1) * size],
            )
            rows = cursor.fetchall()
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to list sessions")
        return _err("Could not load sessions.", 500)

    current_id = session.get("session_id")
    items = [
        {
            "session_id": row["session_id"],
            "user": {
                "user_id": row["user_id"],
                "username": row["username"],
                "display_name": _display_name(row["username"], row["email"]),
                "email": row["email"],
                "profile_photo": row["profile_photo"],
            },
            "device_type": _device_type_for(row),
            "browser": row["device_name"] or "Unknown browser",
            "os": row["platform"] or "Unknown OS",
            "ip_address": row["ip_address"] or None,
            "login_at": _iso(row["created_at"]),
            "last_active_at": _iso(row["last_seen_at"] or row["created_at"]),
            "expires_at": _iso(row["expires_at"]),
            "revoked_at": _iso(row["revoked_at"]),
            "status": _session_status(row, now),
            "is_current": row["session_id"] == current_id,
        }
        for row in rows
    ]
    return jsonify({"success": True, "items": items, "total": total, "page": page, "page_size": size})


@admin_users_bp.get("/api/admin/sessions/history")
@admin_required
def session_history():
    event = (request.args.get("event") or "all").lower()
    q = (request.args.get("q") or "").strip()[:100]
    paging = _paging(default_size=15, maximum=50)
    if event not in HISTORY_EVENTS or paging is None:
        return _err("Invalid event or paging value.", 422)
    page, size = paging
    try:
        start = _parse_date(request.args.get("from"))
        end = _parse_date(request.args.get("to"))
    except ValueError as error:
        return _err(str(error), 422)

    where, params = [], []
    if event != "all":
        where.append("e.event = %s")
        params.append(event)
    if start:
        where.append("e.at >= %s")
        params.append(start)
    if end:
        where.append("e.at < %s")
        params.append(end + timedelta(days=1))
    if q:
        where.append("(u.username LIKE %s OR u.email LIKE %s)")
        params += [_like(q), _like(q)]

    try:
        with _db() as (_conn, cursor):
            items, total = _history(cursor, where, params, size, (page - 1) * size)
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to load session history")
        return _err("Could not load session history.", 500)
    return jsonify({"success": True, "items": items, "total": total, "page": page, "page_size": size})


@admin_users_bp.post("/api/admin/sessions/<int:session_id>/revoke")
@admin_required
def revoke_session(session_id: int):
    if session_id == session.get("session_id"):
        return _err("This is your current session. Use Log out instead.", 409)
    now = _now()
    try:
        with _db() as (conn, cursor):
            cursor.execute(
                "SELECT session_id, revoked_at, expires_at FROM user_sessions WHERE session_id = %s FOR UPDATE",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return _err("Session not found.", 404)
            if row["revoked_at"] is not None or row["expires_at"] <= now:
                return _err("This session has already ended.", 409)
            cursor.execute(
                """
                UPDATE user_sessions SET revoked_at = %s, revoked_by = %s, end_reason = 'admin'
                WHERE session_id = %s AND revoked_at IS NULL
                """,
                (now, session["user_id"], session_id),
            )
            conn.commit()
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to revoke session")
        return _err("The session could not be revoked.", 500)
    return jsonify({"success": True, "message": "Session revoked.", "item": {"session_id": session_id, "status": "revoked"}})


@admin_users_bp.post("/api/admin/users/<int:user_id>/sessions/revoke-all")
@admin_required
def revoke_all_user_sessions(user_id: int):
    if user_id == session["user_id"]:
        return _err("You can't revoke your own sessions here. Use Log out instead.", 409)
    now = _now()
    try:
        with _db() as (conn, cursor):
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            if not cursor.fetchone():
                return _err("User not found.", 404)
            cursor.execute(
                """
                UPDATE user_sessions SET revoked_at = %s, revoked_by = %s, end_reason = 'admin'
                WHERE user_id = %s AND revoked_at IS NULL AND expires_at > %s
                """,
                (now, session["user_id"], user_id, now),
            )
            revoked = cursor.rowcount
            if revoked:
                cursor.execute(
                    """
                    INSERT INTO account_activity_log (user_id, admin_id, activity_type, description, created_at)
                    VALUES (%s, %s, 'sessions_revoked', %s, %s)
                    """,
                    (user_id, session["user_id"], f"{revoked} session(s) revoked", now),
                )
            conn.commit()
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to revoke all sessions")
        return _err("The sessions could not be revoked.", 500)
    return jsonify({"success": True, "message": f"{revoked} session(s) revoked.", "revoked": revoked})
