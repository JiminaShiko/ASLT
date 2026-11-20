import random
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash

from config.database import get_db_connection
from config.mailer import send_otp_email

login_bp = Blueprint("login_bp", __name__)

MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_HOURS = 24
LOGIN_OTP_EXPIRY_MINUTES = 10


def _generate_otp(length: int = 6) -> str:
    return "".join(str(random.SystemRandom().randint(0, 9)) for _ in range(length))


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
    Step 1 of Login:
    1. If email does not exist OR status == 'pending' -> "Invalid Email!"
    2. If status == 'banned' -> Banned community notification
    3. If locked_until > now -> 24-hour lockout notification
    4. Checks password_hash (max 3 attempts before 24-hour lock -> "Invalid Password")
    5. On valid password: resets attempts, sends 6-digit Login OTP (`purpose='login'`),
       and redirects to `/verify-otp?email=...&purpose=login`.
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

        # 1. Non-existent or 'pending' account -> treat as Invalid Email!
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

        # 5. Password matched -> reset failed attempts & generate 6-digit Login OTP
        user_id = user["user_id"]
        otp_code = _generate_otp(6)
        expires_at = now_utc + timedelta(minutes=LOGIN_OTP_EXPIRY_MINUTES)

        cursor.execute(
            """
            UPDATE users
            SET failed_login_attempts = 0, locked_until = NULL, updated_at = %s
            WHERE user_id = %s
            """,
            (now_utc, user_id),
        )
        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'login'",
            (user_id,),
        )
        cursor.execute(
            """
            INSERT INTO otp (user_id, otp_code, purpose, expires_at, is_verified, created_at)
            VALUES (%s, %s, 'login', %s, 0, %s)
            """,
            (user_id, otp_code, expires_at, now_utc),
        )
        conn.commit()
        cursor.close()

        mail_ok, mail_msg = send_otp_email(
            to_email=email,
            otp_code=otp_code,
            purpose="login",
            recipient_name=user.get("username"),
        )
        if not mail_ok:
            return jsonify(
                {"success": False, "message": f"Could not send Login OTP email: {mail_msg}"}
            ), 500

        session.pop("user_id", None)
        session.pop("user_email", None)
        session["pending_login_user_id"] = user_id
        session["pending_login_email"] = email

        return jsonify(
            {
                "success": True,
                "message": "Login OTP sent! Check your inbox.",
                "redirect_url": f"/verify-otp?email={email}&purpose=login",
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Login Error]: {err}")
        return jsonify(
            {"success": False, "message": "Server error during login. Please try again."}
        ), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@login_bp.route("/api/login/verify-otp", methods=["POST"])
def verify_login_otp():
    """
    Step 2 of Login:
    Verifies the 6-digit Login OTP in `otp`, removes the used row,
    records `login_history`, establishes the user/admin session,
    and routes to `/admin-dashboard` or `/home`.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or session.get("pending_login_email") or "").strip().lower()
    otp_input = (data.get("otp") or "").strip()

    if not email or not otp_input:
        return jsonify({"success": False, "message": "Email and 6-digit OTP are required."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT o.otp_id, o.user_id, o.otp_code, o.expires_at,
                   u.username, u.email, u.profile_photo, u.role, u.status
            FROM otp o
            INNER JOIN users u ON o.user_id = u.user_id
            WHERE u.email = %s AND o.purpose = 'login'
            ORDER BY o.otp_id DESC
            LIMIT 1
            """,
            (email,),
        )
        record = cursor.fetchone()

        if not record:
            return jsonify(
                {"success": False, "message": "No login verification found. Please log in again."}
            ), 404

        if (record.get("status") or "").lower() == "banned":
            return jsonify(
                {
                    "success": False,
                    "message": "Sorry, you're banned from our community. For more concern, contact to admin via mail.",
                }
            ), 403

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        if now_utc > record["expires_at"]:
            cursor.execute("DELETE FROM otp WHERE otp_id = %s", (record["otp_id"],))
            conn.commit()
            return jsonify(
                {"success": False, "message": "Your Login OTP has expired. Please resend a new code."}
            ), 410

        if otp_input != record["otp_code"]:
            return jsonify({"success": False, "message": "Invalid OTP code. Please try again."}), 401

        user_id = record["user_id"]

        # Remove used login OTP
        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'login'",
            (user_id,),
        )

        # Record login history (best-effort)
        try:
            ip_addr = request.headers.get("X-Forwarded-For", request.remote_addr or "")[:45]
            user_agent = (request.headers.get("User-Agent") or "")[:255]
            cursor.execute(
                """
                INSERT INTO login_history (user_id, ip_address, device_info, login_time)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, ip_addr, user_agent, now_utc),
            )
        except Exception:
            pass

        conn.commit()
        cursor.close()

        role = (record.get("role") or "user").lower()
        has_complete_profile = bool(record.get("username") and record.get("profile_photo"))

        session.clear()
        session["user_id"] = user_id
        session["user_email"] = record["email"]
        session["user_role"] = role
        session["username"] = record.get("username")
        session["profile_photo"] = record.get("profile_photo")
        session["profile_completed"] = has_complete_profile

        if role == "admin":
            redirect_url = "/admin-dashboard"
        elif not has_complete_profile:
            redirect_url = "/profile-setup"
        else:
            redirect_url = "/home"

        return jsonify(
            {
                "success": True,
                "message": "Login verified! Redirecting...",
                "redirect_url": redirect_url,
                "profile_completed": has_complete_profile,
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Login OTP Verify Error]: {err}")
        return jsonify({"success": False, "message": "Server error verifying Login OTP."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@login_bp.route("/api/login/resend-otp", methods=["POST"])
def resend_login_otp():
    """Resends a fresh 6-digit Login OTP."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or session.get("pending_login_email") or "").strip().lower()

    if not email:
        return jsonify({"success": False, "message": "Email is required to resend OTP."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT user_id, username, status FROM users WHERE email = %s LIMIT 1",
            (email,),
        )
        user = cursor.fetchone()
        if not user or (user["status"] or "").lower() != "active":
            return jsonify({"success": False, "message": "Invalid Email!"}), 404

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        new_otp = _generate_otp(6)
        new_expiry = now_utc + timedelta(minutes=LOGIN_OTP_EXPIRY_MINUTES)

        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'login'",
            (user["user_id"],),
        )
        cursor.execute(
            """
            INSERT INTO otp (user_id, otp_code, purpose, expires_at, is_verified, created_at)
            VALUES (%s, %s, 'login', %s, 0, %s)
            """,
            (user["user_id"], new_otp, new_expiry, now_utc),
        )
        conn.commit()
        cursor.close()

        mail_ok, mail_msg = send_otp_email(
            to_email=email,
            otp_code=new_otp,
            purpose="login",
            recipient_name=user.get("username"),
        )
        if not mail_ok:
            return jsonify({"success": False, "message": f"Failed to resend email: {mail_msg}"}), 500

        return jsonify({"success": True, "message": "A new Login OTP has been sent to your email."}), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Resend Login OTP Error]: {err}")
        return jsonify({"success": False, "message": "Could not resend OTP right now."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@login_bp.route("/api/logout", methods=["POST", "GET"])
def logout_user():
    session.clear()
    return jsonify({"success": True, "redirect_url": "/login"}), 200