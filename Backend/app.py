import importlib.util
import os
from pathlib import Path
import sys
from flask import Flask, jsonify, redirect, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config.database.db_connection import get_db_connection
from config.environment import load_environment
from admin.overview import admin_overview_bp
from admin.profile import admin_profile_bp
from login_signup_pages.forgot_password_process import forgot_password_bp
from login_signup_pages.google_auth import google_bp
from login_signup_pages.login_process import ensure_login_columns, login_bp
from login_signup_pages.signup_process import ensure_auth_tables, signup_bp
from login_signup_pages.profile_process import profile_bp
from agent_mail import agent_mail_bp
from user.text_to_sign import text_to_sign_bp
from cursor_effects_process import cursor_bp
from user_homepage.learning_process import learning_bp

PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "Front-end"
LOGIN_PAGES_DIR = FRONTEND_DIR / "login_signup_pages"
HTML_DIR = FRONTEND_DIR / "HTML"
ADMIN_DIR = FRONTEND_DIR / "HTML" / "Admin"
load_environment()

app = Flask(__name__)

# Lets the front-end opened from a local static server (VS Code Live Server and
# friends) call the API with the shared session cookie during development.
CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:8000", "http://localhost:8000",
        "http://127.0.0.1:3000", "http://localhost:3000",
    ],
)

# Register the landmark translator routes on the main website app. This keeps
# the model API and the authenticated UI on the same origin and port.
translation_api_path = PROJECT_ROOT / "Model" / "sign language photo model" / "api.py"
translation_api_spec = importlib.util.spec_from_file_location(
    "aslt_translation_api", translation_api_path
)
if translation_api_spec is None or translation_api_spec.loader is None:
    raise RuntimeError(f"Could not load translation API from {translation_api_path}")
translation_api = importlib.util.module_from_spec(translation_api_spec)
sys.modules[translation_api_spec.name] = translation_api
translation_api_spec.loader.exec_module(translation_api)
app.register_blueprint(translation_api.api_bp)

# Text to Sign: word signs from the word model's vocabulary plus alphabet
# fingerspelling, served from the same origin as the page.
app.register_blueprint(text_to_sign_bp)

# AgentMail: the backend agent's own inbox, outbound mail and the signed webhook
# that answers replies.
app.register_blueprint(agent_mail_bp)

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
app.register_blueprint(admin_overview_bp)
app.register_blueprint(admin_profile_bp)
app.register_blueprint(cursor_bp)
app.register_blueprint(learning_bp)


LOOPBACK_ADDRESSES = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


def _translation_auth_bypassed() -> bool:
    """ASLT_TRANSLATION_NO_AUTH=1 lets local requests use the model without
    login, so the page can be tested from Live Server with no database."""
    flag = os.getenv("ASLT_TRANSLATION_NO_AUTH", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return False
    return request.remote_addr in LOOPBACK_ADDRESSES


@app.before_request
def require_login_for_translation_api():
    # Browsers send CORS preflights without cookies, so they must pass here.
    if request.method == "OPTIONS":
        return None
    if (request.blueprint in {"asl_translation_api", "text_to_sign_api"}
            and not session.get("user_email")):
        if _translation_auth_bypassed():
            return None
        return jsonify({"error": "Authentication required."}), 401


# Ensure `users` columns (`username` nullable, `failed_login_attempts`, `locked_until`) exist
ensure_auth_tables()
ensure_login_columns()


@app.route("/")
def index():
    return send_from_directory(HTML_DIR, "Landing_page.html")


@app.route("/login")
def login_signup_view():
    return send_from_directory(LOGIN_PAGES_DIR, "login_page.html")


@app.route("/forgot-password")
def forgot_password_view():
    return send_from_directory(LOGIN_PAGES_DIR, "forgot_password.html")


@app.route("/verify-otp")
def verify_otp_view():
    return send_from_directory(LOGIN_PAGES_DIR, "otp_verify.html")


@app.route("/reset-password")
def reset_password_view():
    if not session.get("password_reset_verified_email"):
        return redirect("/forgot-password")
    return send_from_directory(LOGIN_PAGES_DIR, "password_reset.html")


@app.route("/api/auth/oauth-flash", methods=["GET"])
def consume_oauth_flash():
    msg = session.pop("oauth_error_message", None)
    mode = session.pop("oauth_error_mode", "login")
    return jsonify({"message": msg, "mode": mode}), 200


@app.route("/profile-setup")
def profile_setup_placeholder():
    if not session.get("user_email"):
        return redirect("/login")
    profile_form = LOGIN_PAGES_DIR / "profile_form.html"
    if profile_form.is_file():
        return send_from_directory(LOGIN_PAGES_DIR, "profile_form.html")
    return (
        f"<h2>Profile Setup (profile_form.html coming next)</h2>"
        f"<p>Logged in as {session.get('user_email')}</p>"
        f"<a href='/home'>Skip to Home Page</a>"
    )


@app.route("/home")
def user_home_view():
    if not session.get("user_email"):
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR / "user_homepage", "main_dashboard.html")


@app.route("/user_homepage/<path:filename>")
def serve_user_homepage_files(filename):
    if not session.get("user_email"):
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR / "user_homepage", filename)


@app.route("/text_to_sign_demo/<path:filename>")
def serve_text_to_sign_demo_media(filename):
    if not session.get("user_email"):
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR / "text_to_sign_demo", filename)


@app.route("/api/user/me", methods=["GET"])
def get_current_user():
    user_id = session.get("user_id")
    user_email = session.get("user_email")
    if not user_id and not user_email:
        return jsonify({"authenticated": False}), 401

    username = session.get("username")
    profile_photo = session.get("profile_photo")
    role = session.get("user_role", "user")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if user_id:
            cur.execute(
                "SELECT id, email, username, profile_photo, role FROM users WHERE id = %s",
                (user_id,),
            )
        else:
            cur.execute(
                "SELECT id, email, username, profile_photo, role FROM users WHERE email = %s",
                (user_email,),
            )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            user_id = row[0]
            user_email = row[1]
            username = row[2] or username
            profile_photo = row[3] or profile_photo
            role = row[4] or role
            session["user_id"] = user_id
            session["username"] = username
            session["profile_photo"] = profile_photo
    except Exception:
        pass

    display_name = username or (user_email.partition("@")[0] if user_email else "User")

    return jsonify(
        {
            "authenticated": True,
            "user_id": user_id,
            "email": user_email,
            "username": display_name,
            "profile_photo": profile_photo,
            "role": role,
            "profile_completed": bool(username),
        }
    ), 200


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


# Serves relative files inside Front-end/login_signup_pages/ (e.g., auth_style.css)
@app.route("/<path:filename>")
def serve_login_page_assets(filename):
    target = LOGIN_PAGES_DIR / filename
    if target.is_file():
        return send_from_directory(LOGIN_PAGES_DIR, filename)
    return "Not Found", 404


# Serves Front-end/assets/images/... cleanly for both localhost and Render
@app.route("/assets/<path:filename>")
def serve_global_assets(filename):
    assets_dir = FRONTEND_DIR / "assets"
    target = assets_dir / filename
    if target.is_file():
        return send_from_directory(assets_dir, filename)

    if filename == "images/logo.png":
        return send_from_directory(assets_dir, "images/logo_bgremove.png")

    suffix = target.suffix.lower()
    fallback_name = None
    if suffix == ".jpg":
        fallback_name = filename[:-4] + ".png"
    elif suffix == ".png":
        fallback_name = filename[:-4] + ".jpg"

    if fallback_name:
        fallback = assets_dir / fallback_name
        if fallback.is_file():
            return send_from_directory(assets_dir, fallback_name)

    return "Not Found", 404


@app.route("/css/<path:filename>")
def serve_frontend_css(filename):
    return send_from_directory(FRONTEND_DIR / "css", filename)


@app.route("/js/<path:filename>")
def serve_frontend_js(filename):
    return send_from_directory(FRONTEND_DIR / "js", filename)


@app.route("/HTML/<path:filename>")
def serve_frontend_html(filename):
    if not session.get("user_email"):
        return redirect("/login")
    return send_from_directory(HTML_DIR, filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    if os.getenv("ASLT_TRANSLATION_NO_AUTH", "").strip().lower() in {"1", "true", "yes"}:
        print("[dev] ASLT_TRANSLATION_NO_AUTH is on: loopback requests skip the login check.")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)