# Signrest Admin Dashboard — Foundation

Open `index.html` in a modern browser. No installation or build is needed.
For local serving, run `python3 -m http.server 8000` in this folder and open http://localhost:8000.

## Structure
- index.html — dashboard shell and Overview entry point
- css/style.css — shared styles and light/dark theme tokens
- js/app.js — Overview mock data, chart controls, search, navigation, and theme preference
- assets/images/ — local light and dark cover photos
- assets/icons/ — reserved for future local icons; current icons are inline SVG

## Scope
This is the foundation setup, not the complete eight-module dashboard.
Overview works; the other seven navigation entries show a coming-soon message.
Light mode uses sage and parchment; dark mode uses the original deep forest palette.
Theme choice and demo state are saved only in this browser's localStorage.
No database, FastAPI, authentication, or AI training is connected.
Fonts use Google Fonts when online, with local serif/sans-serif fallbacks.
Existing Overview charts are vanilla JavaScript; Chart.js can be added during the analytics page step.
Future pages should reuse the shared styles and script, with a separate HTML page per module.
