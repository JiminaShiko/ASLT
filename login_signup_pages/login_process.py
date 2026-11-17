from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash

from config.database import get_db_connection

login_bp = Blueprint("login_bp", __name__)

MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_HOURS = 24


def ensure_login_columns() -> None:
    """
    Safely adds `failed_login_attempts` and `locked_until` columns to `users`
    if they do not already exist.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SHOW COLUMNS FROM users LIKE 'failed_login_attempts'")
        if not cursor.fetchone():
            cursor.execute(
                "ALTER TABLE users ADD COLUMN failed_login_attempts INT NOT NULL DEFAULT 0"
            )

        cursor.execute("SHOW COLUMNS FROM users LIKE 'locked_until'")
        if not cursor.fetchone():
            cursor.execute(
                "ALTER TABLE users ADD COLUMN locked_until DATETIME NULL DEFAULT NULL"
            )

        conn.commit()
        cursor.close()
    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Login Schema Setup Warning]: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()


@login_bp.route("/api/login", methods=["POST"])
def process_login():
    """
    Handles email/password login:
    1. If email does not exist OR status == 'pending' -> "Invalid Email!"
    2. If status == 'banned' -> "Sorry, you're banned from our community. For more concern, contact to admin via mail."
    3. If locked_until > now -> 24-hour lockout notification
    4. Checks password_hash; up to 3 wrong attempts allowed before 24-hour lock ("Invalid Password")
    5. On success: resets attempts, records login_history, routes 'admin' -> /admin-dashboard, 'user' -> /home
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"success": False, "message": "Invalid Email!"}), 400

    if not password:
        return jsonify({"success": False, "message": "Invalid Password"}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT user_id, username, full_name, email, password_hash, profile_photo,
                   role, status, failed_login_attempts, locked_until
            FROM users
            WHERE email = %s
            LIMIT 1
            """,
            (email,),
        )
        user = cursor.fetchone()

        # 1. Non-existent or 'pending' (unverified) account -> treat as Invalid Email!
        if not user or (user["status"] or "").lower() == "pending":
            return jsonify({"success": False, "message": "Invalid Email!"}), 401

        status = (user["status"] or "").lower()

        # 2. Banned account check
        if status == "banned":
            return jsonify(
                {
                    "success": False,
                    "message": "Sorry, you're banned from our community. For more concern, contact to admin via mail.",
                }
            ), 403

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        locked_until = user.get("locked_until")
        attempts = int(user.get("failed_login_attempts") or 0)

        # 3. Check active 24-hour lockout
        if locked_until:
            if now_utc < locked_until:
                remaining = locked_until - now_utc
                hours_left = max(1, int((remaining.total_seconds() + 3599) // 3600))
                return jsonify(
                    {
                        "success": False,
                        "message": f"Account locked due to 3 failed attempts. Try again in {hours_left}h.",
                    }
                ), 423
            else:
                # 24-hour lock expired -> reset counter automatically
                attempts = 0
                cursor.execute(
                    "UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE user_id = %s",
                    (user["user_id"],),
                )
                conn.commit()

        # 4. Verify password
        if not check_password_hash(user["password_hash"], password):
            attempts += 1

            if attempts >= MAX_LOGIN_ATTEMPTS:
                lock_expiry = now_utc + timedelta(hours=LOCKOUT_HOURS)
                cursor.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = %s, locked_until = %s, updated_at = %s
                    WHERE user_id = %s
                    """,
                    (attempts, lock_expiry, now_utc, user["user_id"]),
                )
                conn.commit()
                return jsonify(
                    {
                        "success": False,
                        "message": "Invalid Password. Too many failed attempts—account locked for 24 hours.",
                    }
                ), 423
            else:
                cursor.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = %s, updated_at = %s
                    WHERE user_id = %s
                    """,
                    (attempts, now_utc, user["user_id"]),
                )
                conn.commit()
                return jsonify({"success": False, "message": "Invalid Password"}), 401

        # 5. Password matched -> reset failed attempts & lockout
        cursor.execute(
            """
            UPDATE users
            SET failed_login_attempts = 0, locked_until = NULL, updated_at = %s
            WHERE user_id = %s
            """,
            (now_utc, user["user_id"]),
        )

        # Optional: record login in `login_history` if table exists
        try:
            ip_addr = request.headers.get("X-Forwarded-For", request.remote_addr or "")[:45]
            user_agent = (request.headers.get("User-Agent") or "")[:255]
            cursor.execute(
                """
                INSERT INTO login_history (user_id, ip_address, device_info, login_time)
                VALUES (%s, %s, %s, %s)
                """,
                (user["user_id"], ip_addr, user_agent, now_utc),
            )
        except Exception:
            pass

        conn.commit()
        cursor.close()

        role = (user.get("role") or "user").lower()
        has_complete_profile = bool(user.get("username") and user.get("profile_photo"))

        session.clear()
        session["user_id"] = user["user_id"]
        session["user_email"] = user["email"]
        session["user_role"] = role
        session["username"] = user.get("username")
        session["full_name"] = user.get("full_name")
        session["display_name"] = (
            user.get("full_name") or user.get("username") or user["email"].partition("@")[0]
        )
        session["profile_photo"] = user.get("profile_photo")
        session["profile_completed"] = has_complete_profile

        redirect_url = "/admin-dashboard" if role == "admin" else "/home"

        return jsonify(
            {
                "success": True,
                "message": "Login successful!",
                "redirect_url": redirect_url,
                "profile_completed": has_complete_profile,
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Login Error]: {err}")
        return jsonify({"success": False, "message": "Server error during login. Please try again."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@login_bp.route("/api/logout", methods=["POST", "GET"])
def logout_user():
    session.clear()
    return jsonify({"success": True, "redirect_url": "/login"}), 200