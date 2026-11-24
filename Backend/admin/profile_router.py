"""Admin profile settings endpoints; identity comes from g.admin_user, never the browser."""
import os
import re
import uuid
from datetime import date, datetime
from pathlib import Path

from flask import Blueprint, current_app, g, jsonify, request

from .dependencies import get_db, hash_password, require_admin, verify_password

profile_bp = Blueprint("admin_profile", __name__, url_prefix="/api/admin/profile")
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
MAX_PHOTO_BYTES = 5 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@profile_bp.errorhandler(ApiError)
def handle_api_error(error):
    return jsonify({"detail": error.detail}), error.status


def admin_id():
    admin = getattr(g, "admin_user", None)
    try:
        return int(admin.get("user_id") or admin.get("id"))
    except (TypeError, ValueError, AttributeError):
        raise ApiError(401, "Your administrator session is not valid")


def iso(value):
    # MySQL DATETIME is naive; timezone is not applied here.
    return value.isoformat() if isinstance(value, (datetime, date)) else None


def json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError(400, "A JSON object is required")
    return data


def protect_write():
    """Browser-write guard; this is not a replacement for require_admin."""
    if request.headers.get("X-Signrest-Request") != "admin":
        raise ApiError(403, "Invalid admin request")
    if request.mimetype not in ("application/json", "multipart/form-data"):
        raise ApiError(415, "JSON or multipart form data is required")
    origin = request.headers.get("Origin")
    if origin:
        configured = {
            item.strip().rstrip("/")
            for item in os.getenv("SIGNREST_ALLOWED_ORIGINS", "").split(",")
            if item.strip()
        }
        if origin.rstrip("/") not in {request.host_url.rstrip("/"), *configured}:
            raise ApiError(403, "This origin is not allowed")
    # The header is not auth or CSRF protection; require_admin still guards.


def profile_row(db, user_id):
    with db.cursor() as cursor:
        cursor.execute(
            """SELECT user_id, username, email, full_name, profile_photo,
                      role, status, created_at, updated_at
               FROM users WHERE user_id = %s LIMIT 1""",
            (user_id,),
        )
        row = cursor.fetchone()
    if not row:
        raise ApiError(404, "Administrator profile not found")
    return row


def serialized(row):
    # Never expose password_hash, tokens, OTPs, or IPs.
    return {
        "user_id": row[0],
        "username": row[1],
        "email": row[2],
        "full_name": row[3],
        "profile_photo": row[4],
        "role": row[5],
        "status": row[6],
        "created_at": iso(row[7]),
        "updated_at": iso(row[8]),
    }


def duplicate_account(error):
    return bool(getattr(error, "args", ()) and error.args[0] == 1062)


def profile_fields(data):
    full_name = data.get("full_name")
    username = data.get("username")
    email = data.get("email")
    if full_name is not None and not isinstance(full_name, str):
        raise ApiError(422, "Full name must be text")
    if not isinstance(username, str) or not isinstance(email, str):
        raise ApiError(422, "Username and email are required")
    full_name = (full_name or "").strip() or None
    username = username.strip()
    email = email.strip()
    if full_name and len(full_name) > 150:
        raise ApiError(422, "Full name must be 150 characters or fewer")
    if not username or len(username) > 100:
        raise ApiError(422, "Username must contain 1 to 100 characters")
    if len(email) > 255 or not EMAIL_PATTERN.fullmatch(email):
        raise ApiError(422, "Enter a valid email address")
    return full_name, username, email


@profile_bp.get("")
@require_admin
def get_profile():
    db = get_db()
    try:
        return jsonify({"item": serialized(profile_row(db, admin_id()))})
    except ApiError:
        raise
    except Exception:
        current_app.logger.exception("Could not load administrator profile")
        raise ApiError(500, "Could not load administrator profile")


@profile_bp.patch("")
@require_admin
def update_profile():
    db = get_db()
    user_id = admin_id()
    protect_write()
    full_name, username, email = profile_fields(json_object())
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """UPDATE users
                   SET full_name = %s, username = %s, email = %s, updated_at = NOW()
                   WHERE user_id = %s""",
                (full_name, username, email, user_id),
            )
            if cursor.rowcount == 0:
                # MySQL can report zero for unchanged values; confirm existence.
                profile_row(db, user_id)
        row = profile_row(db, user_id)
        db.commit()
        return jsonify({"item": serialized(row)})
    except ApiError:
        db.rollback()
        raise
    except Exception as error:
        db.rollback()
        if duplicate_account(error):
            raise ApiError(409, "Username or email is already in use")
        current_app.logger.exception("Could not save administrator profile")
        raise ApiError(500, "Could not save administrator profile")


def image_extension(data):
    # Sniff magic bytes rather than trusting filename or Content-Type.
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ApiError(422, "Choose a valid PNG, JPEG, or WebP image")


@profile_bp.post("/photo")
@require_admin
def upload_photo():
    db = get_db()
    user_id = admin_id()
    protect_write()
    upload = request.files.get("file")
    if upload is None:
        raise ApiError(422, "Choose an image file")
    data = upload.stream.read(MAX_PHOTO_BYTES + 1)
    upload.close()
    if len(data) > MAX_PHOTO_BYTES:
        raise ApiError(413, "Profile photo must be 5 MB or smaller")
    if not data:
        raise ApiError(422, "Choose an image file")
    extension = image_extension(data)

    upload_dir = Path(os.getenv("SIGNREST_PROFILE_UPLOAD_DIR", "uploads/profile_photos")).resolve()
    public_prefix = "/" + os.getenv("SIGNREST_PROFILE_UPLOAD_URL", "/uploads/profile_photos").strip("/")
    filename = f"{user_id}-{uuid.uuid4().hex}.{extension}"
    destination = upload_dir / filename
    public_path = f"{public_prefix}/{filename}"
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            output.write(data)
        with db.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET profile_photo = %s, updated_at = NOW() WHERE user_id = %s",
                (public_path, user_id),
            )
            if cursor.rowcount == 0:
                raise ApiError(404, "Administrator profile not found")
        row = profile_row(db, user_id)
        db.commit()
        return jsonify({"item": serialized(row)})
    except ApiError:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise
    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        current_app.logger.exception("Could not upload profile photo")
        raise ApiError(500, "Could not upload profile photo")
    # Previous file is kept: another account may share it.


@profile_bp.post("/password")
@require_admin
def update_password():
    db = get_db()
    user_id = admin_id()
    protect_write()
    data = json_object()
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    if not isinstance(current_password, str) or not 1 <= len(current_password) <= 128:
        raise ApiError(422, "Enter your current password")
    if not isinstance(new_password, str) or not 8 <= len(new_password) <= 128:
        raise ApiError(422, "Your new password must contain 8 to 128 characters")
    if new_password == current_password:
        raise ApiError(422, "Choose a password different from your current password")
    try:
        with db.cursor() as cursor:
            cursor.execute("SELECT password_hash FROM users WHERE user_id = %s LIMIT 1", (user_id,))
            row = cursor.fetchone()
            if not row:
                raise ApiError(404, "Administrator profile not found")
            if not verify_password(current_password, row[0]):
                raise ApiError(400, "Current password is incorrect")
            new_hash = hash_password(new_password)
            cursor.execute(
                "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE user_id = %s",
                (new_hash, user_id),
            )
        db.commit()
        return jsonify({"detail": "Password updated"})
    except ApiError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        current_app.logger.exception("Could not update password")
        raise ApiError(500, "Could not update password")