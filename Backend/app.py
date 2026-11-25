import os
from pathlib import Path
import sys
from flask import Flask, jsonify, redirect, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config.database.db_connection import get_db_connection  # noqa: E402
from config.environment import load_environment  # noqa: E402
from admin.overview import admin_overview_bp  # noqa: E402
from admin.profile import admin_profile_bp  # noqa: E402
from admin.user_management import (  # noqa: E402
    admin_users_bp,
    ensure_user_management_schema,
    session_guard,
)
from login_signup_pages.forgot_password_process import forgot_password_bp  # noqa: E402
from login_signup_pages.google_auth import google_bp  # noqa: E402
from login_signup_pages.login_process import ensure_login_columns, login_bp  # noqa: E402
from login_signup_pages.signup_process import ensure_auth_tables, signup_bp  # noqa: E402
from login_signup_pages.profile_process import profile_bp  # noqa: E402
from agent_mail import agent_mail_bp  # noqa: E402
from user.api import api_bp as translation_api_bp  # noqa: E402
from user.text_to_sign import text_to_sign_bp  # noqa: E402
from cursor_effects_process import cursor_bp  # noqa: E402
from user.learning import learning_bp  # noqa: E402

PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "Front-end"
LOGIN_PAGES_DIR = FRONTEND_DIR / "login_signup_pages"
HTML_DIR = FRONTEND_DIR / "HTML"
ADMIN_DIR = FRONTEND_DIR / "HTML" / "Admin"
load_environment()

app = Flask(__name__)

# Shared session cookie from a local static server (e.g. Live Server) needs these origins.
CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:8000", "http://localhost:8000",
        "http://127.0.0.1:3000", "http://localhost:3000",
    ],
)

app.register_blueprint(translation_api_bp)
app.register_blueprint(text_to_sign_bp)
app.register_blueprint(agent_mail_bp)

app.secret_key = os.getenv("SECRET_KEY", "dev-fallback-secret-key")

# Required behind Render's reverse proxy so HTTPS and cookies resolve correctly.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

app.register_blueprint(signup_bp)
app.register_blueprint(login_bp)
app.register_blueprint(forgot_password_bp)
app.register_blueprint(google_bp)
app.register_blueprint(profile_bp)
app.register_blueprint(admin_overview_bp)
app.register_blueprint(admin_profile_bp)
app.register_blueprint(admin_users_bp)
app.register_blueprint(cursor_bp)
app.register_blueprint(learning_bp)


LOOPBACK_ADDRESSES = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


def _translation_auth_bypassed() -> bool:
    """Dev escape hatch: ASLT_TRANSLATION_NO_AUTH=1 skips login for loopback only."""
    flag = os.getenv("ASLT_TRANSLATION_NO_AUTH", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return False
    return request.remote_addr in LOOPBACK_ADDRESSES


@app.before_request
def enforce_server_side_sessions():
    return session_guard()


@app.before_request
def require_login_for_translation_api():
    # CORS preflights carry no cookies; they must pass.
    if request.method == "OPTIONS":
        return None
    if (request.blueprint in {"asl_translation_api", "text_to_sign_api"}
            and not session.get("user_email")):
        if _translation_auth_bypassed():
            return None
        return jsonify({"error": "Authentication required."}), 401


# Schema migrations must run before any request touches `users`.
ensure_auth_tables()
ensure_login_columns()
ensure_user_management_schema()


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


ADMIN_SIDEBAR_SLOT = '<div id="adminSidebarHost"></div>'
ADMIN_SHELL_FILE = "admin_shell.html"
# Runs while the page is parsed, so a collapsed sidebar stays collapsed instead
# of flashing open on every admin page change.
ADMIN_SIDEBAR_BOOT = (
    "<script>try{if(localStorage.getItem('signrest-admin-sidebar')==='collapsed')"
    "document.getElementById('app').classList.add('collapsed')}catch(e){}</script>"
)

# The shell mounts one sidebar and loads only the page inside its frame. Embedded
# pages therefore drop the sidebar slot and the sidebar layout offset.
ADMIN_EMBED_STYLE = """<style id="admin-embed-style">
.app > .main,.main{margin-left:0 !important;width:100% !important}
.app{min-height:100vh}
.mobile-overlay{display:none !important}
.mobile-collapse{display:none}
@media (max-width:800px){.mobile-collapse{display:flex}}
</style>"""

ADMIN_EMBED_SCRIPT = """<script id="admin-embed-script">
(function () {
  if (window.parent === window) return;
  var origin = location.origin;
  var files = {overview: 'admin_dashboard', logs: 'accuracy', materials: 'materials', users: 'user_management', profile: 'profile'};
  function post(message) { try { parent.postMessage(message, origin); } catch (error) {} }
  function navigateInside(href) {
    var url;
    try { url = new URL(href, location.href); } catch (error) { return false; }
    if (url.origin !== origin || url.pathname.indexOf('/admin-dashboard/') !== 0) return false;
    post({type: 'signrest-admin-page', path: url.pathname + url.search});
    return true;
  }
  document.addEventListener('click', function (event) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var link = event.target && event.target.closest ? event.target.closest('a[href]') : null;
    if (!link) return;
    if (navigateInside(link.getAttribute('href'))) event.preventDefault();
  }, true);
  function syncShellControls() {
    window.collapseSidebar = function () { post({type: 'signrest-admin-sidebar', action: 'toggle'}); };
    window.closeMobile = function () { post({type: 'signrest-admin-sidebar', action: 'close'}); };
    if (typeof window.navigate === 'function' && !window.navigate.signrestShell) {
      var original = window.navigate;
      var wrapped = function (page, filter) {
        if (document.body.dataset.page === page || !files[page]) return original.apply(this, arguments);
        post({type: 'signrest-admin-page', path: '/admin-dashboard/' + files[page] + '.html' + (filter ? '?filter=' + filter : '')});
      };
      wrapped.signrestShell = true;
      window.navigate = wrapped;
    }
  }
  window.addEventListener('DOMContentLoaded', syncShellControls);
  window.addEventListener('load', syncShellControls);
})();
</script>"""


def _render_admin_embed(html: str) -> str:
    """Render a page for the shell frame: no sidebar, no sidebar layout offset."""
    html = html.replace(ADMIN_SIDEBAR_SLOT, "", 1)
    if "admin-embed-style" not in html:
        html = html.replace("</head>", ADMIN_EMBED_STYLE + "\n</head>", 1)
    if "admin-embed-script" not in html:
        html = html.replace("</body>", ADMIN_EMBED_SCRIPT + "\n</body>", 1)
    return html


def _render_admin_page(page: Path, embed: bool = False) -> str:
    """Inline the shared sidebar so it is part of the first paint.

    Every admin page is wrapped by the shell, so the sidebar is inlined once
    into that shell document instead of into each page.
    """
    html = page.read_text(encoding="utf-8")
    if embed:
        return _render_admin_embed(html)
    if ADMIN_SIDEBAR_SLOT not in html:
        return html
    sidebar_file = ADMIN_DIR / "admin_sidebar.html"
    if not sidebar_file.is_file():
        return html
    sidebar = sidebar_file.read_text(encoding="utf-8")
    html = html.replace(
        ADMIN_SIDEBAR_SLOT,
        '<div id="adminSidebarHost">' + sidebar + "</div>",
        1,
    )
    return html.replace(
        '<div class="app" id="app">',
        '<div class="app" id="app">' + ADMIN_SIDEBAR_BOOT,
        1,
    )


@app.route("/admin-dashboard/")
def serve_admin_root():
    return serve_admin_dashboard("admin_dashboard.html")


@app.route("/admin-dashboard/<path:filename>")
def serve_admin_dashboard(filename):
    if not session.get("user_email") or session.get("user_role") != "admin":
        return redirect("/login")

    try:
        page = (ADMIN_DIR / filename).resolve()
        page.relative_to(ADMIN_DIR.resolve())
    except (OSError, ValueError):
        return "Not Found", 404

    if page.is_file() and page.suffix.lower() == ".html":
        headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-store",
        }
        if request.args.get("embed"):
            return _render_admin_page(page, embed=True), 200, headers
        if filename == "admin_sidebar.html":
            return send_from_directory(ADMIN_DIR, filename)
        # Every admin page is opened inside the one shell, so the sidebar is
        # never re-rendered as part of a page load (including refresh).
        return _render_admin_page(ADMIN_DIR / ADMIN_SHELL_FILE), 200, headers
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


@app.route("/<path:filename>")
def serve_login_page_assets(filename):
    target = LOGIN_PAGES_DIR / filename
    if target.is_file():
        return send_from_directory(LOGIN_PAGES_DIR, filename)
    return "Not Found", 404


# Falls back across .jpg/.png variants and the logo alias because assets differ per deploy.
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