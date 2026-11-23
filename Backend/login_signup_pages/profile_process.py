from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, session

from config.cloudinary import upload_to_cloudinary
from config.database import get_db_connection

profile_bp = Blueprint("profile_bp", __name__)


@profile_bp.route("/api/profile/setup", methods=["POST"])
def save_user_profile():
    """Upload the cropped photo to Cloudinary, save username + photo, redirect home."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "Session expired. Please login again."}), 401

    username = (request.form.get("username") or "").strip()
    photo_file = request.files.get("profile_photo")

    if not username or len(username) < 2 or len(username) > 40:
        return jsonify(
            {"success": False, "message": "Please enter a username (2–40 characters)."}
        ), 400

    if not photo_file or not photo_file.filename:
        return jsonify({"success": False, "message": "Please upload and crop a profile photo."}), 400

    try:
        secure_photo_url = upload_to_cloudinary(
            photo_file.stream,
            folder="profiles",
            filename=f"user_{user_id}_avatar.png",
            resource_type="image",
        )
    except Exception as cloud_err:
        print(f"[Cloudinary Profile Upload Error]: {cloud_err}")
        return jsonify(
            {"success": False, "message": "Photo upload failed. Check Cloudinary configuration."}
        ), 500

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor.execute(
            """
            UPDATE users
            SET username = %s,
                profile_photo = %s,
                updated_at = %s
            WHERE user_id = %s
            """,
            (username, secure_photo_url, now_utc, user_id),
        )
        conn.commit()
        cursor.close()

        session["username"] = username
        session["profile_photo"] = secure_photo_url
        session["profile_completed"] = True

        return jsonify(
            {
                "success": True,
                "message": "Profile saved! Welcome to Signrest.",
                "redirect_url": "/home",
            }
        ), 200

    except Exception as db_err:
        if conn:
            conn.rollback()
        print(f"[Profile DB Update Error]: {db_err}")
        return jsonify({"success": False, "message": "Database error saving profile."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()


@profile_bp.route("/api/profile/skip", methods=["POST"])
def skip_profile_setup():
    """Skip profile setup but keep the session; photo + username still gate completion."""
    if not session.get("user_id"):
        return jsonify({"success": False, "redirect_url": "/login"}), 401

    session["profile_completed"] = bool(
        session.get("username") and session.get("profile_photo")
    )
    return jsonify({"success": True, "redirect_url": "/home"}), 200