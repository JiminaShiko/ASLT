
import re
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config.database import get_db_connection
from config.cloudinary import upload_to_cloudinary


admin_profile_bp = Blueprint("admin_profile", __name__)

MAX_PROFILE_IMAGE_SIZE = 5 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_admin():
    return (
        bool(session.get("user_id"))
        and bool(session.get("user_email"))
        and (session.get("user_role") or "").lower() == "admin"
    )


def _auth_error():
    if not session.get("user_id") or not session.get("user_email"):
        return jsonify({
            "success": False,
            "message": "Authentication required."
        }), 401

    if (session.get("user_role") or "").lower() != "admin":
        return jsonify({
            "success": False,
            "message": "Administrator access required."
        }), 403

    return None


def _get_admin(cursor):
    cursor.execute(
        """
        SELECT user_id, username, email, profile_photo, password_hash,
               role, status, created_at, updated_at
        FROM users
        WHERE user_id = %s AND email = %s
        LIMIT 1
        """,
        (session["user_id"], session["user_email"]),
    )

    user = cursor.fetchone()

    if not user:
        return None

    if (user.get("role") or "").lower() != "admin":
        return None

    return user


def _profile_response(user):
    return {
        "user_id": user["user_id"],
        "username": user.get("username"),
        "email": user.get("email"),
        "profile_photo": user.get("profile_photo"),
        "role": user.get("role"),
        "status": user.get("status"),
        "created_at": (
            user["created_at"].isoformat()
            if user.get("created_at") else None
        ),
        "updated_at": (
            user["updated_at"].isoformat()
            if user.get("updated_at") else None
        ),
    }


@admin_profile_bp.get("/api/admin/profile")
def get_admin_profile():
    auth_error = _auth_error()
    if auth_error:
        return auth_error

    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        user = _get_admin(cursor)

        if not user:
            return jsonify({
                "success": False,
                "message": "Administrator profile not found."
            }), 404

        return jsonify({
            "success": True,
            "item": _profile_response(user)
        }), 200

    except Exception:
        current_app.logger.exception("Failed to load admin profile")
        return jsonify({
            "success": False,
            "message": "Could not load administrator profile."
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@admin_profile_bp.put("/api/admin/profile")
def update_admin_profile():
    auth_error = _auth_error()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}

    username = data.get("username")

    if not isinstance(username, str):
        return jsonify({
            "success": False,
            "message": "Name is required."
        }), 400

    username = username.strip()

    # Any name is allowed (spaces, accents, symbols); it only has to be
    # non-empty, at most 100 characters, and not already in use.
    if not username or len(username) > 100:
        return jsonify({
            "success": False,
            "message": "Name must contain 1–100 characters."
        }), 400

    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        user = _get_admin(cursor)

        if not user:
            return jsonify({
                "success": False,
                "message": "Administrator profile not found."
            }), 404

        cursor.execute(
            """
            SELECT user_id FROM users
            WHERE username = %s AND user_id != %s
            LIMIT 1
            """,
            (username, user["user_id"]),
        )

        if cursor.fetchone():
            return jsonify({
                "success": False,
                "message": "That name is already taken. Please choose another."
            }), 409

        cursor.execute(
            """
            UPDATE users
            SET username = %s,
                updated_at = %s
            WHERE user_id = %s
            """,
            (username, _now_utc(), user["user_id"]),
        )

        conn.commit()

        session["username"] = username

        return jsonify({
            "success": True,
            "message": "Profile information updated successfully.",
            "item": {
                "username": username
            }
        }), 200

    except Exception:
        if conn:
            conn.rollback()

        current_app.logger.exception("Failed to update admin profile")

        return jsonify({
            "success": False,
            "message": "Could not update profile information."
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@admin_profile_bp.post("/api/admin/profile/upload-picture")
def upload_admin_profile_picture():
    auth_error = _auth_error()
    if auth_error:
        return auth_error

    image = request.files.get("profile_picture")

    if not image or not image.filename:
        return jsonify({
            "success": False,
            "message": "Please select a profile picture."
        }), 400

    filename = secure_filename(image.filename)
    extension = Path(filename).suffix.lower().lstrip(".")

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({
            "success": False,
            "message": "Only JPG, JPEG, PNG, and WEBP images are allowed."
        }), 400

    image_bytes = image.read(MAX_PROFILE_IMAGE_SIZE + 1)

    if not image_bytes:
        return jsonify({
            "success": False,
            "message": "The selected image is empty."
        }), 400

    if len(image_bytes) > MAX_PROFILE_IMAGE_SIZE:
        return jsonify({
            "success": False,
            "message": "Profile picture must be smaller than 5 MB."
        }), 400

    # Never trust the extension alone.
    valid_signatures = {
        "jpg": image_bytes.startswith(b"\xff\xd8\xff"),
        "jpeg": image_bytes.startswith(b"\xff\xd8\xff"),
        "png": image_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
        "webp": (
            image_bytes.startswith(b"RIFF")
            and image_bytes[8:12] == b"WEBP"
        ),
    }

    if not valid_signatures.get(extension, False):
        return jsonify({
            "success": False,
            "message": "The selected file is not a valid image."
        }), 400

    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        user = _get_admin(cursor)

        if not user:
            return jsonify({
                "success": False,
                "message": "Administrator profile not found."
            }), 404

        image_url = upload_to_cloudinary(
            file_source=image_bytes,
            folder="admin_profiles",
            filename=filename,
            resource_type="image",
        )

        cursor.execute(
            """
            UPDATE users
            SET profile_photo = %s,
                updated_at = %s
            WHERE user_id = %s
            """,
            (image_url, _now_utc(), user["user_id"]),
        )

        conn.commit()

        session["profile_photo"] = image_url

        return jsonify({
            "success": True,
            "message": "Profile picture updated successfully.",
            "profile_photo": image_url
        }), 200

    except Exception:
        if conn:
            conn.rollback()

        current_app.logger.exception("Failed to upload admin profile picture")

        return jsonify({
            "success": False,
            "message": "Could not upload profile picture."
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@admin_profile_bp.post("/api/admin/profile/change-password")
def change_admin_password():
    auth_error = _auth_error()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}

    current_password = data.get("current_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not all(
        isinstance(value, str) and value
        for value in (current_password, new_password, confirm_password)
    ):
        return jsonify({
            "success": False,
            "message": "Please complete all password fields."
        }), 400

    if len(new_password) < 8:
        return jsonify({
            "success": False,
            "message": "New password must contain at least 8 characters."
        }), 400

    if not re.search(r"[A-Z]", new_password):
        return jsonify({
            "success": False,
            "message": "New password must contain at least one uppercase letter."
        }), 400

    if not re.search(r"\d", new_password):
        return jsonify({
            "success": False,
            "message": "New password must contain at least one number."
        }), 400

    if not re.search(r"[^A-Za-z0-9]", new_password):
        return jsonify({
            "success": False,
            "message": "New password must contain at least one special character."
        }), 400

    if new_password != confirm_password:
        return jsonify({
            "success": False,
            "message": "New password and confirmation do not match."
        }), 400

    if current_password == new_password:
        return jsonify({
            "success": False,
            "message": "Your new password must be different from your current password."
        }), 400

    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        user = _get_admin(cursor)

        if not user:
            return jsonify({
                "success": False,
                "message": "Administrator profile not found."
            }), 404

        if not check_password_hash(user["password_hash"], current_password):
            return jsonify({
                "success": False,
                "message": "Current password is incorrect."
            }), 401

        new_password_hash = generate_password_hash(new_password)

        cursor.execute(
            """
            UPDATE users
            SET password_hash = %s,
                updated_at = %s
            WHERE user_id = %s
            """,
            (new_password_hash, _now_utc(), user["user_id"]),
        )

        conn.commit()

        return jsonify({
            "success": True,
            "message": "Password updated successfully."
        }), 200

    except Exception:
        if conn:
            conn.rollback()

        current_app.logger.exception("Failed to change admin password")

        return jsonify({
            "success": False,
            "message": "Could not update password."
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
