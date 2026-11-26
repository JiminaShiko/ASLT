import random
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, session
from werkzeug.security import generate_password_hash

from config import realtime
from config.database import get_db_connection
from config.mailer import send_otp_email

signup_bp = Blueprint("signup_bp", __name__)

OTP_EXPIRY_MINUTES = 10


def ensure_auth_tables() -> None:
    """Allow NULL `users.username` and drop the legacy `signup_otps` table."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "ALTER TABLE users MODIFY COLUMN username VARCHAR(100) NULL DEFAULT NULL"
            )
        except Exception:
            pass

        try:
            cursor.execute("DROP TABLE IF EXISTS signup_otps")
        except Exception:
            pass

        conn.commit()
        cursor.close()
    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Signup Schema Setup Warning]: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()


def _generate_otp(length: int = 6) -> str:
    return "".join(str(random.SystemRandom().randint(0, 9)) for _ in range(length))


@signup_bp.route("/api/signup", methods=["POST"])
def initiate_signup():
    """Step 1: validate, upsert the user as 'pending', store an OTP, send the email."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"success": False, "message": "Please enter a valid email address."}), 400

    if len(password) < 8:
        return jsonify({"success": False, "message": "Password must be at least 8 characters."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT user_id, status FROM users WHERE email = %s LIMIT 1",
            (email,),
        )
        existing_user = cursor.fetchone()

        password_hash = generate_password_hash(password)
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        if existing_user:
            if existing_user["status"] == "active":
                return jsonify(
                    {"success": False, "message": "This email is already in use! Just Login to your acc!"}
                ), 409

            user_id = existing_user["user_id"]
            cursor.execute(
                """
                UPDATE users
                SET password_hash = %s, updated_at = %s
                WHERE user_id = %s
                """,
                (password_hash, now_utc, user_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, role, status, created_at, updated_at)
                VALUES (NULL, %s, %s, 'user', 'pending', %s, %s)
                """,
                (email, password_hash, now_utc, now_utc),
            )
            user_id = cursor.lastrowid

        otp_code = _generate_otp(6)
        expires_at = now_utc + timedelta(minutes=OTP_EXPIRY_MINUTES)

        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'signup'",
            (user_id,),
        )

        cursor.execute(
            """
            INSERT INTO otp (user_id, otp_code, purpose, expires_at, is_verified, created_at)
            VALUES (%s, %s, 'signup', %s, 0, %s)
            """,
            (user_id, otp_code, expires_at, now_utc),
        )
        conn.commit()
        cursor.close()

        mail_ok, mail_msg = send_otp_email(to_email=email, otp_code=otp_code, purpose="signup")
        if not mail_ok:
            return jsonify(
                {"success": False, "message": f"Could not send verification email: {mail_msg}"}
            ), 500

        session["pending_signup_email"] = email
        realtime.publish_sessions_changed("account_created")

        return jsonify(
            {
                "success": True,
                "message": "OTP sent! Check your inbox to complete registration.",
                "redirect_url": f"/verify-otp?email={email}&purpose=signup",
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Signup Error]: {err}")
        return jsonify({"success": False, "message": "Server error during signup. Please try again."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@signup_bp.route("/api/signup/verify-otp", methods=["POST"])
def verify_signup_otp():
    """Step 2: consume the OTP, activate the account, open the session."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or session.get("pending_signup_email") or "").strip().lower()
    otp_input = (data.get("otp") or "").strip()

    if not email or not otp_input:
        return jsonify({"success": False, "message": "Email and 6-digit OTP are required."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT o.otp_id, o.user_id, o.otp_code, o.expires_at
            FROM otp o
            INNER JOIN users u ON o.user_id = u.user_id
            WHERE u.email = %s AND o.purpose = 'signup'
            ORDER BY o.otp_id DESC
            LIMIT 1
            """,
            (email,),
        )
        record = cursor.fetchone()

        if not record:
            return jsonify(
                {"success": False, "message": "No pending verification found. Please sign up again."}
            ), 404

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        if now_utc > record["expires_at"]:
            cursor.execute("DELETE FROM otp WHERE otp_id = %s", (record["otp_id"],))
            conn.commit()
            return jsonify(
                {"success": False, "message": "Your OTP has expired. Please request a new code."}
            ), 410

        if otp_input != record["otp_code"]:
            return jsonify({"success": False, "message": "Invalid OTP code. Please try again."}), 401

        cursor.execute(
            """
            UPDATE users
            SET status = 'active', updated_at = %s
            WHERE user_id = %s
            """,
            (now_utc, record["user_id"]),
        )

        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'signup'",
            (record["user_id"],),
        )
        conn.commit()
        cursor.close()

        realtime.publish_sessions_changed("account_updated")
        session.pop("pending_signup_email", None)
        session["user_id"] = record["user_id"]
        session["user_email"] = email
        session["user_role"] = "user"

        return jsonify(
            {
                "success": True,
                "message": "Email verified! Let's personalize your profile.",
                "redirect_url": "/profile-setup",
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[OTP Verify Error]: {err}")
        return jsonify({"success": False, "message": "Server error verifying OTP."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@signup_bp.route("/api/signup/resend-otp", methods=["POST"])
def resend_signup_otp():
    """Generates and emails a fresh 6-digit OTP in the `otp` table for a pending signup."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or session.get("pending_signup_email") or "").strip().lower()

    if not email:
        return jsonify({"success": False, "message": "Email is required to resend OTP."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT user_id, status FROM users WHERE email = %s LIMIT 1",
            (email,),
        )
        user = cursor.fetchone()
        if not user or user["status"] == "active":
            return jsonify(
                {"success": False, "message": "No pending signup found for this email."}
            ), 404

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        new_otp = _generate_otp(6)
        new_expiry = now_utc + timedelta(minutes=OTP_EXPIRY_MINUTES)

        cursor.execute("DELETE FROM otp WHERE user_id = %s AND purpose = 'signup'", (user["user_id"],))
        cursor.execute(
            """
            INSERT INTO otp (user_id, otp_code, purpose, expires_at, is_verified, created_at)
            VALUES (%s, %s, 'signup', %s, 0, %s)
            """,
            (user["user_id"], new_otp, new_expiry, now_utc),
        )
        conn.commit()
        cursor.close()

        mail_ok, mail_msg = send_otp_email(to_email=email, otp_code=new_otp, purpose="resend")
        if not mail_ok:
            return jsonify({"success": False, "message": f"Failed to resend email: {mail_msg}"}), 500

        return jsonify({"success": True, "message": "A new OTP has been sent to your email."}), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Resend OTP Error]: {err}")
        return jsonify({"success": False, "message": "Could not resend OTP right now."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()