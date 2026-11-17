# AI Accuracy & Logs — FastAPI connection contract

This is a connection-ready frontend, not an installed backend. Implement these read-only endpoints in your existing FastAPI app and connect that app to MySQL. Do not create new audit/review tables for this page.

## Authentication and hosting
- Both endpoints must validate the logged-in session and require the user's actual administrator role before querying or returning data. A visible Admin label is not authorization.
- Return HTTP 401 when signed out and 403 for non-admin accounts. Never make these endpoints public or trust a role sent by the browser.
- The frontend uses your existing secure HttpOnly cookie with credentials: include. It does not implement login or pretend that its Logout control clears a server session; sign out through your existing application.
- Prefer serving HTML and FastAPI from one origin. If separate, configure an explicit allowed frontend origin and allow_credentials=True in FastAPI CORS (not '*'). Use HTTPS; cross-site cookies require SameSite=None; Secure and may still be blocked by browser privacy settings. Same-origin avoids that problem.
- Keep MySQL credentials in backend environment variables. Never include password hashes, tokens, IP addresses, raw media paths, or OTPs in these responses.

## GET /api/admin/model-accuracy
Return application/json with { "items": [...] }. Each item has these fields from model_accuracy:

| Field | JSON type | Existing column |
| --- | --- | --- |
| accuracy_id | integer | accuracy_id |
| model_name | string | model_name |
| model_version | string or null | model_version |
| total_samples | integer or null | total_samples |
| correct_predictions | integer or null | correct_predictions |
| accuracy_percentage | number or null | accuracy_percentage |
| evaluated_at | ISO 8601 string with timezone or null | evaluated_at |

Return the latest evaluation for each model name/version, using evaluated_at descending and accuracy_id descending as a tie-breaker. Return {"items":[]} if no evaluations exist. Send Decimal values as JSON numbers; percentage units are 0–100, not 0–1. Missing accuracy displays as unavailable, not 0%. Do not derive evaluation accuracy from prediction confidence or average model percentages with different sample sizes.

## GET /api/admin/translations
Query parameters:

| Parameter | Values |
| --- | --- |
| page | integer >= 1, default 1 |
| page_size | integer 1–100, frontend sends 5 |
| q | bounded search string, default empty |
| mode | all, Sign to Text, Text to Sign |
| confidence | all, uncertain, high |

Map the two display mode names to your actual stored mode values on the backend; return these display names in each item's mode field. Reject unsupported values. Search translation ID, user name, predicted text, mode, and input type with bound parameters. Apply all filters before counting and pagination.

Return application/json with:

```text
{
  "items": [translation records for the requested page],
  "total": number of records matching all current filters,
  "low_confidence_total": number below 75 matching q and mode (before the confidence filter)
}
```

Each translation record contains:

| Field | JSON type | Existing source |
| --- | --- | --- |
| translation_id | integer | translations.translation_id |
| user_name | string | users.full_name, falling back to users.username |
| mode | string | translations.mode, mapped to the display names above |
| input_type | string or null | translations.input_type |
| predicted_text | string or null | translations.predicted_text |
| confidence_score | number or null | translations.confidence_score |
| created_at | ISO 8601 string with timezone or null | translations.created_at |

LEFT JOIN translations to users by user_id and return only the fields above. Order by translations.created_at DESC, translation_id DESC. Use bound LIMIT and OFFSET (offset = (page - 1) * page_size), never interpolated SQL. Escape LIKE wildcards if search should treat them literally.

- uncertain: confidence_score < 75
- high: confidence_score >= 75
- all: include NULL confidence; NULL belongs to neither confidence band and displays as Not available.
- No records: {"items":[],"total":0,"low_confidence_total":0}.
- Read totals and page rows consistently (one read transaction when needed).
- Add the actual timezone when serializing MySQL DATETIME; do not label local timestamps as UTC without conversion.
- Do not return a successful empty response when the database fails. Return a suitable error status, log details privately, and omit SQL/credential details from the client response.

## Verify after connecting
1. Log in with an actual admin account; open accuracy.html and confirm saved MySQL records display.
2. Check an empty database, unknown confidence, mode/search filters, both sides of the 75% boundary, pagination, and read-only details.
3. Sign out and confirm endpoints return 401; a non-admin account must receive 403.
4. Stop FastAPI and confirm Retry appears without old/demo data. Theme changes should still work.

This update's automated tests exercise the frontend against controlled responses only. A real authenticated FastAPI/MySQL flow remains unverified until your existing backend supplies these endpoints.

## Sign Materials — existing sign_materials table

These are frontend contracts to implement in your external FastAPI application, not installed backend endpoints. Reuse the session/admin checks above for every read and write. No new tables or columns are needed.

### GET /api/admin/materials

Query: page >= 1 (default 1), page_size 1–100 (frontend sends 9), q (bounded search over sign_label, explanation and category), category (exact category or empty for all). Apply filters before counting; order by created_at DESC, material_id DESC. Use bound SQL parameters and a consistent read transaction for rows/count.

Return `{ "items": [...], "total": 0, "categories": [...] }`. Categories are distinct nonempty category strings from the full library, not only this page. Each item contains:

- material_id: integer
- sign_label: string (existing varchar(100))
- category: string (existing varchar(30))
- explanation: string or null
- image_url: public HTTPS image URL or same-origin public path, resolved from image_path by FastAPI; null if absent. Never expose filesystem paths or arbitrary private storage URLs.
- created_at: ISO 8601 with actual timezone, or null

Empty data returns items [], total 0 and categories []. Database failures must return an error, never successful empty data.

### POST /api/admin/materials and PATCH /api/admin/materials/{material_id}

JSON: `{ "sign_label": "...", "category": "...", "image_path": null, "explanation": "..." }`. The form sends all four editable fields. Trim and require sign_label/category; enforce 100/30/500 character column limits, and limit explanation to 10,000 characters. created_at and material_id are generated on the server, never accepted from the browser. Validate image_path as an allowed public media path or URL in your application (no local file paths, script/data URLs, or unrestricted server fetching of remote URLs). Resolve it to image_url for reads; edits must accept the public image URL returned on reads. Uploads are out of scope.

POST returns HTTP 201 and PATCH HTTP 200, both with `{ "item": <material in GET shape> }` only after a committed database write. Bound INSERT/UPDATE queries; allowlist these fields. Return 404 for missing records, 422 for invalid fields and 409 for conflicts.

### DELETE /api/admin/materials/{material_id}

Require an actual admin session, delete using a bound ID, and return 204 only after commit. Reject deletion with 409 if other records depend on the material; do not cascade into unrelated learning/quiz data. Do not delete an image file that could be shared by other records.

### Cookie authentication and write protection

The frontend sends credentials: include for all requests. Writes use JSON and the custom header X-Signrest-Request: admin. FastAPI must validate the Origin against an explicit allowlist, require this custom header and JSON content type for POST/PATCH, and enforce CSRF protection. The header alone is not authorization. Keep credentialed CORS narrow; never allow arbitrary origins or reflect an Origin blindly. If your application uses CSRF tokens, place its public session CSRF token in SIGNREST_API_CONFIG.csrfToken before api.js; the adapter forwards it as X-CSRF-Token and the backend must verify it. No credentials or static secret belongs in this config.

### Verify after implementation

Use a real administrator account to create a material, reload/read it from MySQL, edit it and confirm the changed values, then delete it and confirm absence. Test filters, pagination, empty data, failed writes preserving the form, missing/broken images and refresh failure clearing stale records. Confirm signed-out requests return 401, non-admin requests 403, and cross-origin forged writes are rejected. No external FastAPI/MySQL authenticated flow has been verified here.