# Signrest Admin Dashboard — Foundation

Open `admin_dashboard.html` in a modern browser. No installation or build is needed.
For local serving, run `python3 -m http.server 8000` in this folder and open http://localhost:8000/admin_dashboard.html.

## Structure
- admin_dashboard.html — dashboard shell and Overview entry point
- accuracy.html — AI Accuracy & Logs, read-only model evaluations and predictions
- materials.html — Sign Materials, API-backed library with search, categories, details, add/edit/delete
- css/style.css — shared styles and light/dark theme tokens
- js/app.js — shared page controls, Overview demo, and theme preference
- js/api.js — FastAPI connection adapter used by accuracy.html and materials.html
- js/materials.js — material library and editing controls (load before js/app.js on materials.html)
- assets/images/mist.jpg — light mode cover photo
- assets/images/forest.jpg — dark mode cover photo
- assets/icons/ — 26 standalone SVG files, one per icon used in the interface

## Icons
The pages draw every icon inline from the `paths` list in js/app.js, so the icon
files in assets/icons are not loaded by the HTML. They are included so the same
set can be reused elsewhere (user-side pages, documents, email templates) without
redrawing them. Each file is a 24x24 stroke icon using `currentColor`, matching
the on-screen rendering.

## Scope
This is the foundation setup, not the complete eight-module dashboard.
Overview, AI Accuracy & Logs, and Sign Materials work; the other five navigation entries show a coming-soon message.
Light mode uses sage and parchment; dark mode uses the original deep forest palette.
Theme choice and Overview demo state are saved only in this browser's localStorage.
AI Accuracy & Logs does not use sample records or browser-stored records. It requests FastAPI data and shows loading, empty, or connection-error states. Overview remains a demo and is not converted in this update.
No running FastAPI service or MySQL connection is included. Authentication must be supplied by your existing application; no AI training or prediction review is added.
Sign Materials also contains no demo or cached records. It uses the existing sign_materials fields, including image_path (a public HTTP image path/URL, not a local disk path). No video or status columns are assumed. Changes are confirmed only after successful API responses. Images must be served by your existing app; file uploads are not included.
Fonts use Google Fonts when online, with local serif/sans-serif fallbacks.
Existing Overview charts are vanilla JavaScript; Chart.js can be added during the analytics page step.
Future pages should reuse the shared styles and script, with a separate HTML page per module.

## Adding this to your existing Admin folder
Copy accuracy.html and materials.html alongside admin_dashboard.html, and replace the shared css/style.css and js/app.js with these versions. Also copy js/api.js and js/materials.js. In js/app.js, the overview entry in the files map is already 'admin_dashboard'; change it if your existing Overview file has a different name. Keep assets/images alongside these folders.

## Connect your FastAPI application

See FASTAPI_CONNECTION.md for the exact response contract based on your existing users, translations, and model_accuracy tables. There are no schema changes or new tables required.

Default API base: /api/admin on the same host as the HTML pages. To use a different API host, add this before js/api.js in accuracy.html or materials.html (replace the example with your real URL):

```html
<script>window.SIGNREST_API_CONFIG = { baseUrl: 'https://your-api.example/api/admin' };</script>
```

Do not open accuracy.html or materials.html as file://; serve them using HTTP(S). No credentials belong in HTML or JavaScript. The pages assume an existing secure HttpOnly session cookie, send credentials with requests, and handle 401/403 responses. If your login uses a different method, adapt the request authentication in js/api.js to that existing login flow before connecting.
