import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, send_from_directory, session
from werkzeug.middleware.proxy_fix import ProxyFix

from login_signup_pages.forgot_password_process import forgot_password_bp
from login_signup_pages.google_auth import google_bp
from login_signup_pages.login_process import ensure_login_columns, login_bp
from login_signup_pages.signup_process import ensure_auth_tables, signup_bp
from login_signup_pages.profile_process import profile_bp

BASE_DIR = Path(__file__).resolve().parent
ADMIN_DIR = BASE_DIR / "HTML" / "Admin"
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

# Reads SECRET_KEY strictly from .env (localhost) or Render Environment Variables
app.secret_key = os.getenv("SECRET_KEY", "dev-fallback-secret-key")

# Ensures HTTPS & session cookies work properly behind Render's reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Register signup, login, forgot password, and Google OAuth blueprints
app.register_blueprint(signup_bp)
app.register_blueprint(login_bp)
app.register_blueprint(forgot_password_bp)
app.register_blueprint(google_bp)
app.register_blueprint(profile_bp)

# Ensure `users` columns (`username` nullable, `failed_login_attempts`, `locked_until`) exist
ensure_auth_tables()
ensure_login_columns()


@app.route("/")
def index():
    return redirect("/login")


@app.route("/login")
def login_signup_view():
    return send_from_directory(BASE_DIR / "login_signup_pages", "login_page.html")


@app.route("/forgot-password")
def forgot_password_view():
    return send_from_directory(BASE_DIR / "login_signup_pages", "forgot_password.html")


@app.route("/verify-otp")
def verify_otp_view():
    return send_from_directory(BASE_DIR / "login_signup_pages", "otp_verify.html")


@app.route("/reset-password")
def reset_password_view():
    if not session.get("password_reset_verified_email"):
        return redirect("/forgot-password")
    return send_from_directory(BASE_DIR / "login_signup_pages", "password_reset.html")


@app.route("/api/auth/oauth-flash", methods=["GET"])
def consume_oauth_flash():
    msg = session.pop("oauth_error_message", None)
    mode = session.pop("oauth_error_mode", "login")
    return jsonify({"message": msg, "mode": mode}), 200


@app.route("/profile-setup")
def profile_setup_placeholder():
    if not session.get("user_email"):
        return redirect("/login")
    profile_form = BASE_DIR / "login_signup_pages" / "profile_form.html"
    if profile_form.is_file():
        return send_from_directory(BASE_DIR / "login_signup_pages", "profile_form.html")
    return (
        f"<h2>Profile Setup (profile_form.html coming next)</h2>"
        f"<p>Logged in as {session.get('user_email')}</p>"
        f"<a href='/home'>Skip to Home Page</a>"
    )


@app.route("/home")
def user_home_placeholder():
    if not session.get("user_email"):
        return redirect("/login")
    completed = session.get("profile_completed", False)
    return (
        f"<h2>Welcome to Signrest Home, {session.get('user_email')}!</h2>"
        f"<p>Profile Completed: {completed}</p>"
        f"<a href='/api/logout'>Logout</a>"
    )


@app.route("/admin-dashboard")
def admin_dashboard_placeholder():
    if not session.get("user_email") or session.get("user_role") != "admin":
        return redirect("/login")
    return redirect("/admin-dashboard/")


@app.route("/admin-dashboard/", defaults={"filename": "admin_dashboard.html"})
@app.route("/admin-dashboard/<path:filename>")
def serve_admin_dashboard(filename):
    if not session.get("user_email") or session.get("user_role") != "admin":
        return redirect("/login")
    return send_from_directory(ADMIN_DIR, filename)


@app.route("/api/admin/me")
def admin_profile():
    if not session.get("user_email"):
        return jsonify({"message": "Authentication required."}), 401
    if session.get("user_role") != "admin":
        return jsonify({"message": "Administrator access required."}), 403

    email = session["user_email"]
    display_name = (
        session.get("display_name")
        or session.get("full_name")
        or session.get("username")
        or email.partition("@")[0]
    )
    return jsonify({"name": display_name, "email": email})


# Serves relative files inside login_signup_pages/ (e.g., auth_style.css)
@app.route("/<path:filename>")
def serve_login_page_assets(filename):
    target = BASE_DIR / "login_signup_pages" / filename
    if target.is_file():
        return send_from_directory(BASE_DIR / "login_signup_pages", filename)
    return "Not Found", 404


# Serves ../assets/images/... cleanly for both localhost and Render
@app.route("/assets/<path:filename>")
def serve_global_assets(filename):
    return send_from_directory(BASE_DIR / "assets", filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)