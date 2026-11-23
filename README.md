# ASLT
AI Sign Language Translator
This is a school project.

## Project structure

- `Front-end/` — static website: HTML pages, CSS, JS, images, login/profile pages, the admin dashboard UI, and the logged-in `user_homepage/` (dashboard + translation).
- `Backend/` — Flask application: `app.py`, authentication blueprints (`login_signup_pages/`), the live ASL translation API (`user/api.py`), the Text to Sign planner (`user/text_to_sign.py`), database/mail/Cloudinary config (`config/`), `ca.pem`.
- `Model/` — AI projects: `asl_landmark_app/` (letter recognizer), `videoModel/` (word recognizer), and `EDA/` notebooks/reports.
- `Assignments/`, `Diagrams/`, `Doc/` — course deliverables and design documents.

## Running

The backend also registers the ASL landmark translation API (from
`Backend/user/api.py`) as a blueprint, so a single process
serves the website, the user dashboard and the translation endpoints:

```bash
pip install -r requirements.txt   # root-level consolidated file (Python 3.13+)
cd Backend
python app.py                     # http://127.0.0.1:5000
```

`requirements.txt` at the repository root is the only Python dependency file in
the repo. It merges the dependencies of the backend, the models, the Streamlit
demos and the EDA scripts, so nothing from a subfolder is missed; install it
from the repository root before running anything.

`api.py` does not need to be run separately. It is imported and registered by
`Backend/app.py`; running it on its own (port 5000) is only for isolated API
debugging and cannot run at the same time as the backend.

Front-end pages in `Front-end/HTML/` are plain static files (e.g. VS Code Live Server).
Each folder inside `Model/` has its own README with training instructions.

The recommended way to use the app is through the backend at
http://127.0.0.1:5000/home, which serves every page and the translation API from
one origin. If you preview a page from a static server instead (Live Server on
:5500/:8000/:3000), keep `python app.py` running: the translation page detects
the static origin, calls the backend at http://127.0.0.1:5000, and sends the
login cookie. Log in at http://127.0.0.1:5000 first and use the same host
(`127.0.0.1`, not `localhost`) for both, or the session will not match.

To test the translation page and the model without logging in (no database
needed), start the backend with the local-only dev flag:

```powershell
$env:ASLT_TRANSLATION_NO_AUTH = "1"   # bash: ASLT_TRANSLATION_NO_AUTH=1 python app.py
python app.py
```

Then open the page from Live Server, e.g.
http://127.0.0.1:5500/Front-end/user_homepage/translation.html (the URL depends
on the folder Live Server serves). The flag only skips the login check for
loopback requests to the translation API; the account pages and `/home` still
require a real login.

Login, signup and password reset read their settings from the `.env` file at
the repository root, `D:\Metro\AI Project\ASLT\.env` (`config/environment.py`
also accepts `Backend\.env`, but do not create both). Without `DB_HOST`,
`DB_USER`, `DB_PASS` and `DB_NAME` login cannot reach the database; `MAIL_*` is
needed for the OTP emails, `SECRET_KEY` for sessions, and `GOOGLE_*` /
`CLOUDINARY_*` are optional.

`.env.example` lists the backend agent email settings, empty: `AGENTMAIL_API_KEY`
and `AGENTMAIL_WEBHOOK_SECRET` (`Backend/agent_mail.py`). With a key in `.env`,
`python agent_mail.py` from `Backend/` opens the agent inbox and
`python agent_mail.py you@example.com` sends one test email; point an AgentMail
webhook with the `message.received` event at `POST /api/agent/email/webhook` and
the agent answers each reply with how the app would sign that sentence.
