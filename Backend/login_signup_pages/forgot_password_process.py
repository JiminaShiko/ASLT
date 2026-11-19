import random
import re
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, session
from werkzeug.security import generate_password_hash

from config.database import get_db_connection
from config.mailer import send_otp_email

forgot_password_bp = Blueprint("forgot_password_bp", __name__)

OTP_EXPIRY_MINUTES = 10


def _generate_otp(length: int = 6) -> str:
    return "".join(str(random.SystemRandom().randint(0, 9)) for _ in range(length))


def _is_strong_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "At least 8 characters required."
    if not re.search(r"[A-Z]", password):
        return False, "Include at least one capital letter (A-Z)."
    if not re.search(r"[0-9]", password):
        return False, "Include at least one number (0-9)."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Include at least one special character (!@#$...)."
    return True, ""


@forgot_password_bp.route("/api/forgot-password", methods=["POST"])
def initiate_forgot_password():
    """
    Step 1: Validates email exists and is active, generates a 6-digit OTP
    with purpose='forgot_password' in `otp`, emails it, and redirects to /verify-otp.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"success": False, "message": "Invalid Email!"}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT user_id, status FROM users WHERE email = %s LIMIT 1",
            (email,),
        )
        user = cursor.fetchone()

        if not user or (user["status"] or "").lower() == "pending":
            return jsonify({"success": False, "message": "Invalid Email!"}), 404

        if (user["status"] or "").lower() == "banned":
            return jsonify(
                {
                    "success": False,
                    "message": "Sorry, you're banned from our community. For more concern, contact to admin via mail.",
                }
            ), 403

        user_id = user["user_id"]
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        otp_code = _generate_otp(6)
        expires_at = now_utc + timedelta(minutes=OTP_EXPIRY_MINUTES)

        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'forgot_password'",
            (user_id,),
        )
        cursor.execute(
            """
            INSERT INTO otp (user_id, otp_code, purpose, expires_at, is_verified, created_at)
            VALUES (%s, %s, 'forgot_password', %s, 0, %s)
            """,
            (user_id, otp_code, expires_at, now_utc),
        )
        conn.commit()
        cursor.close()

        mail_ok, mail_msg = send_otp_email(
            to_email=email, otp_code=otp_code, purpose="forgot_password"
        )
        if not mail_ok:
            return jsonify(
                {"success": False, "message": f"Could not send OTP email: {mail_msg}"}
            ), 500

        session["pending_reset_email"] = email
        session.pop("password_reset_verified_email", None)

        return jsonify(
            {
                "success": True,
                "message": "OTP sent! Check your inbox.",
                "redirect_url": f"/verify-otp?email={email}&purpose=forgot_password",
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Forgot Password Error]: {err}")
        return jsonify({"success": False, "message": "Server error. Please try again."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@forgot_password_bp.route("/api/forgot-password/verify-otp", methods=["POST"])
def verify_forgot_password_otp():
    """
    Step 2: Verifies the 6-digit reset OTP in `otp`, deletes the used row,
    marks the email as verified in session, and redirects to /reset-password.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or session.get("pending_reset_email") or "").strip().lower()
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
            WHERE u.email = %s AND o.purpose = 'forgot_password'
            ORDER BY o.otp_id DESC
            LIMIT 1
            """,
            (email,),
        )
        record = cursor.fetchone()

        if not record:
            return jsonify(
                {"success": False, "message": "No reset request found. Please request a new OTP."}
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
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'forgot_password'",
            (record["user_id"],),
        )
        conn.commit()
        cursor.close()

        session.pop("pending_reset_email", None)
        session["password_reset_verified_email"] = email

        return jsonify(
            {
                "success": True,
                "message": "OTP verified! Set your new password.",
                "redirect_url": "/reset-password",
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Forgot Password OTP Verify Error]: {err}")
        return jsonify({"success": False, "message": "Server error verifying OTP."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@forgot_password_bp.route("/api/forgot-password/resend-otp", methods=["POST"])
def resend_forgot_password_otp():
    """Resends a fresh 6-digit password reset OTP."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or session.get("pending_reset_email") or "").strip().lower()

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
        if not user or (user["status"] or "").lower() != "active":
            return jsonify({"success": False, "message": "Invalid Email!"}), 404

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        new_otp = _generate_otp(6)
        new_expiry = now_utc + timedelta(minutes=OTP_EXPIRY_MINUTES)

        cursor.execute(
            "DELETE FROM otp WHERE user_id = %s AND purpose = 'forgot_password'",
            (user["user_id"],),
        )
        cursor.execute(
            """
            INSERT INTO otp (user_id, otp_code, purpose, expires_at, is_verified, created_at)
            VALUES (%s, %s, 'forgot_password', %s, 0, %s)
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
        print(f"[Resend Reset OTP Error]: {err}")
        return jsonify({"success": False, "message": "Could not resend OTP right now."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@forgot_password_bp.route("/api/reset-password", methods=["POST"])
def complete_password_reset():
    """
    Step 3: Updates `users.password_hash` for the OTP-verified email,
    clears failed login attempts/locks, and redirects back to /login.
    """
    email = (session.get("password_reset_verified_email") or "").strip().lower()
    if not email:
        return jsonify(
            {"success": False, "message": "Reset session expired. Please verify your email again."}
        ), 401

    data = request.get_json(silent=True) or {}
    new_password = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    is_valid, msg = _is_strong_password(new_password)
    if not is_valid:
        return jsonify({"success": False, "message": msg}), 400

    if new_password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match."}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        new_hash = generate_password_hash(new_password)

        cursor.execute(
            """
            UPDATE users
            SET password_hash = %s,
                failed_login_attempts = 0,
                locked_until = NULL,
                updated_at = %s
            WHERE email = %s
            """,
            (new_hash, now_utc, email),
        )
        conn.commit()
        cursor.close()

        session.pop("password_reset_verified_email", None)

        return jsonify(
            {
                "success": True,
                "message": "Password reset successful! Please login.",
                "redirect_url": "/login",
            }
        ), 200

    except Exception as err:
        if conn:
            conn.rollback()
        print(f"[Reset Password Error]: {err}")
        return jsonify({"success": False, "message": "Server error resetting password."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()