# ASLT
AI Sign Language Translator
This is a school project.

## Project structure

- `Front-end/` — static website: HTML pages, CSS, JS, images, login/profile pages, and the admin dashboard UI.
- `Backend/` — Flask application: `app.py`, authentication blueprints (`login_signup_pages/`), database/mail/Cloudinary config (`config/`), `ca.pem`, `requirements.txt`.
- `Model/` — AI projects: `asl_landmark_app/` (letter recognizer), `sign language photo model/` (photo recognizer + live API), `videoModel/` (word recognizer), and `EDA/` notebooks/reports.
- `Assignments/`, `Diagrams/`, `Doc/` — course deliverables and design documents.

## Running

Backend (serves login, profile and admin pages plus `/assets`):

```bash
cd Backend
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
```

Front-end pages in `Front-end/HTML/` are plain static files (e.g. VS Code Live Server).

ASL landmark API used by `translation.html`:

```bash
cd "Model/sign language photo model"
pip install -r requirements.txt
python api.py            # http://127.0.0.1:5000
```

Each folder inside `Model/` has its own README with training instructions.
Environment variables are read from the `.env` file at the repository root.
