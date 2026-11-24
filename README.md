# ASLT — AI Sign Language Translator

School project: a Flask web app for ASL fingerspelling recognition (photo/video)
and text-to-sign, backed by a MediaPipe-landmark ML model.

## Structure

- `Front-end/` — static HTML/CSS/JS: landing page, login/signup, user dashboard, admin UI, translation page.
- `Backend/` — Flask app: auth, profiles, admin, AgentMail, translation API (`user/api.py`), text-to-sign planner.
- `Model/asl_landmark_app/` — ASL letter recognizer (MediaPipe landmarks + MLP).
- `Model/videoModel/` — word/sign recognizer.
- `Model/EDA/` — notebooks and dataset reports.
- `Assignments/`, `Diagrams/`, `Doc/` — course deliverables.

## Setup

Requires Python 3.13+.

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in the values
```

Login needs `DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME`; OTP email needs `MAIL_*`;
sessions need `SECRET_KEY`. `ca.pem` is used for the database TLS connection.
See `.env.example` for the AgentMail keys.

## Run

```bash
cd Backend
python app.py            # http://127.0.0.1:5000/home
```

`Backend/app.py` registers the translation API itself, so there is one server and
one origin. Do not also start `user/api.py` (it uses the same port).

Front-end pages can also be opened from a static server (e.g. VS Code Live Server)
while the backend runs on port 5000. To test translation without logging in, start
with `ASLT_TRANSLATION_NO_AUTH=1` (bash) or `$env:ASLT_TRANSLATION_NO_AUTH="1"`
(PowerShell).

Model training and evaluation: see `Model/asl_landmark_app/README.md` and
`Model/videoModel/README.md`.
