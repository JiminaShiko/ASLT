import json
import os
import secrets
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from flask import Blueprint, redirect, request, session, url_for
from werkzeug.security import generate_password_hash

from config.environment import load_environment
from config.database import get_db_connection

load_environment()

google_bp = Blueprint("google_bp", __name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def _record_login_history(cursor, user_id: int, login_status: str, login_time: datetime) -> None:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    ip_addr = (forwarded_for.split(",")[0].strip() or request.remote_addr or "")[:45]
    user_agent = (request.headers.get("User-Agent") or "Google OAuth")[:255]
    cursor.execute(
        """
        INSERT INTO login_history (
            user_id, login_time, login_status, device_info, ip_address
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (user_id, login_time, login_status, user_agent, ip_addr),
    )


def _get_google_credentials() -> tuple[str, str]:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError("Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in environment.")
    return client_id, client_secret


def _build_callback_uri() -> str:
    """ProxyFix in app.py makes url_for emit the right scheme/host per environment."""
    return url_for("google_bp.google_callback", _external=True)


def _redirect_with_oauth_error(message: str, mode: str = "login"):
    session["oauth_error_message"] = message
    session["oauth_error_mode"] = mode if mode in ("signup", "login") else "login"
    return redirect("/login")


@google_bp.route("/api/auth/google", methods=["GET"])
def google_login_start():
    """Step 1: shared entry point for Google signup and login."""
    mode = (request.args.get("mode") or "login").strip().lower()
    if mode not in ("signup", "login"):
        mode = "login"

    try:
        client_id, _ = _get_google_credentials()
    except ValueError as err:
        return _redirect_with_oauth_error(str(err), mode)

    state_token = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state_token
    session["google_oauth_mode"] = mode

    params = {
        "client_id": client_id,
        "redirect_uri": _build_callback_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state_token,
        "access_type": "online",
        "prompt": "select_account",
    }
    auth_redirect_url = f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return redirect(auth_redirect_url)


@google_bp.route("/api/auth/google/callback", methods=["GET"])
def google_callback():
    """Step 2: callback; upserts the user by status and routes to setup, home, or admin."""
    mode = session.pop("google_oauth_mode", "login")
    expected_state = session.pop("google_oauth_state", None)
    returned_state = request.args.get("state")

    if request.args.get("error"):
        return _redirect_with_oauth_error("Google sign-in was cancelled or denied.", mode)

    if not expected_state or returned_state != expected_state:
        return _redirect_with_oauth_error("Invalid OAuth session state. Please try again.", mode)

    code = request.args.get("code")
    if not code:
        return _redirect_with_oauth_error("Missing authorization code from Google.", mode)

    try:
        client_id, client_secret = _get_google_credentials()

        token_payload = urllib.parse.urlencode(
            {
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": _build_callback_uri(),
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")

        token_req = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=token_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(token_req, timeout=15) as token_resp:
            token_data = json.loads(token_resp.read().decode("utf-8"))

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError(token_data.get("error_description") or "Failed to obtain Google access token.")

        userinfo_req = urllib.request.Request(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        with urllib.request.urlopen(userinfo_req, timeout=15) as info_resp:
            google_user = json.loads(info_resp.read().decode("utf-8"))

        email = (google_user.get("email") or "").strip().lower()
        if not email:
            raise ValueError("Google account did not provide an email address.")

    except Exception as oauth_err:
        print(f"[Google OAuth Error]: {oauth_err}")
        return _redirect_with_oauth_error("Google Authentication failed. Please try again.", mode)

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT user_id, username, email, profile_photo, role, status
            FROM users
            WHERE email = %s
            LIMIT 1
            """,
            (email,),
        )
        existing_user = cursor.fetchone()
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        is_new_account = False

        if existing_user:
            user_id = existing_user["user_id"]
            role = (existing_user.get("role") or "user").lower()
            status = (existing_user.get("status") or "active").lower()

            # Banned accounts cannot sign in via Google either.
            if status == "banned":
                _record_login_history(cursor, user_id, "login_fail", now_utc)
                conn.commit()
                cursor.close()
                return _redirect_with_oauth_error(
                    "Sorry, you're banned from our community. For more concern, contact to admin via mail.",
                    "login",
                )

            # Google already proved email ownership, so 'pending' accounts can be activated.
            if status == "pending":
                cursor.execute(
                    """
                    UPDATE users
                    SET status = 'active', failed_login_attempts = 0, locked_until = NULL, updated_at = %s
                    WHERE user_id = %s
                    """,
                    (now_utc, user_id),
                )
                cursor.execute("DELETE FROM otp WHERE user_id = %s AND purpose = 'signup'", (user_id,))
                is_new_account = True
            else:
                cursor.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = 0, locked_until = NULL, updated_at = %s
                    WHERE user_id = %s
                    """,
                    (now_utc, user_id),
                )

            username = existing_user.get("username")
            full_name = existing_user.get("full_name")
            profile_photo = existing_user.get("profile_photo")

        else:
            # Active user with NULL username/photo so they land on profile setup.
            dummy_hash = generate_password_hash(secrets.token_hex(24))
            cursor.execute(
                """
                INSERT INTO users (
                    username, email, password_hash, profile_photo,
                    role, status, failed_login_attempts, locked_until, created_at, updated_at
                )
                VALUES (NULL, %s, %s, NULL, 'user', 'active', 0, NULL, %s, %s)
                """,
                (email, dummy_hash, now_utc, now_utc),
            )
            user_id = cursor.lastrowid
            role = "user"
            username = None
            full_name = None
            profile_photo = None
            is_new_account = True

        _record_login_history(cursor, user_id, "success", now_utc)

        conn.commit()
        cursor.close()

        has_complete_profile = bool(username and profile_photo)

        session.clear()
        session["user_id"] = user_id
        session["user_email"] = email
        session["user_role"] = role
        session["username"] = username
        session["full_name"] = full_name
        session["display_name"] = (
            google_user.get("name")
            or full_name
            or username
            or email.partition("@")[0]
        )
        session["profile_photo"] = profile_photo
        session["profile_completed"] = has_complete_profile

        if role == "admin":
            return redirect("/admin-dashboard")
        if is_new_account or not has_complete_profile:
            return redirect("/profile-setup")
        return redirect("/home")

    except Exception as db_err:
        if conn:
            conn.rollback()
        print(f"[Google OAuth DB Error]: {db_err}")
        return _redirect_with_oauth_error("Database error during Google sign-in.", mode)
    finally:
        if conn and conn.is_connected():
            conn.close()