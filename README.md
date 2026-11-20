# ASLT
AI Sign Language Translator
This is a school project.

## Project structure

- `Front-end/` — static website: HTML pages, CSS, JS, images, login/profile pages, the admin dashboard UI, and the logged-in `user_homepage/` (dashboard + translation).
- `Backend/` — Flask application: `app.py`, authentication blueprints (`login_signup_pages/`), database/mail/Cloudinary config (`config/`), `ca.pem`, `requirements.txt`.
- `Model/` — AI projects: `asl_landmark_app/` (letter recognizer), `sign language photo model/` (photo recognizer + live API), `videoModel/` (word recognizer), and `EDA/` notebooks/reports.
- `Assignments/`, `Diagrams/`, `Doc/` — course deliverables and design documents.

## Running

The backend also registers the ASL landmark translation API (from
`Model/sign language photo model/api.py`) as a blueprint, so a single process
serves the website, the user dashboard and the translation endpoints:

```bash
cd Backend
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000
```

`api.py` does not need to be run separately. It can still be started on its own
from `Model/sign language photo model/` for the legacy standalone page, but not
at the same time as the backend (both use port 5000).

Front-end pages in `Front-end/HTML/` are plain static files (e.g. VS Code Live Server).
Each folder inside `Model/` has its own README with training instructions.
Environment variables are read from the `.env` file at the repository root.
