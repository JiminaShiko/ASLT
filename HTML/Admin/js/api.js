/* Public settings only; never add database passwords or API secrets.
   Serve over HTTP(S). Set window.SIGNREST_API_CONFIG before this script to
   override baseUrl, e.g. {baseUrl:'https://your-api.example/api/admin'}.
   Authentication uses your FastAPI application's secure HttpOnly cookie. */
window.SignrestAPI = (() => {
  const baseUrl = (window.SIGNREST_API_CONFIG?.baseUrl || '/api/admin').replace(/\/$/, '');
  async function get(path, params, signal) {
    if (!['http:', 'https:'].includes(location.protocol)) throw new Error('Open this page through an HTTP or HTTPS server.');
    const url = new URL(baseUrl + path, location.origin);
    Object.entries(params || {}).forEach(([key, value]) => url.searchParams.set(key, String(value)));
    const response = await fetch(url, { credentials: 'include', headers: { Accept: 'application/json' }, signal, cache: 'no-store' });
    if (response.status === 401) throw new Error('Please sign in through your application, then retry.');
    if (response.status === 403) throw new Error('An administrator account is required to view these records.');
    if (!response.ok) throw new Error('The server could not load these records. Please retry.');
    if (!(response.headers.get('content-type') || '').includes('application/json')) throw new Error('The FastAPI endpoint is not connected yet.');
    return response.json();
  }
  function percentage(value) {
    if (value === null || value === undefined) return null;
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0 || number > 100) throw new Error('The server returned an invalid percentage.');
    return number;
  }
  function normalizeLog(row) {
    if (row.translation_id === undefined || typeof row.mode !== 'string') throw new Error('The server returned an invalid translation record.');
    return { id: String(row.translation_id), user: row.user_name || 'Unknown user', mode: row.mode, source: row.input_type || '—', output: row.predicted_text ?? '—', confidence: percentage(row.confidence_score), createdAt: row.created_at };
  }
  function normalizeMaterial(row) {
    if (!Number.isInteger(row.material_id) || typeof row.sign_label !== 'string' || typeof row.category !== 'string') throw new Error('The server returned an invalid material record.');
    return { id: row.material_id, label: row.sign_label, category: row.category, image: row.image_url || null, explanation: row.explanation || '', createdAt: row.created_at };
  }
  async function mutateMaterial(method, id, body) {
    if (!['http:', 'https:'].includes(location.protocol)) throw new Error('Open this page through an HTTP or HTTPS server.');
    const headers = { Accept: 'application/json', 'Content-Type': 'application/json', 'X-Signrest-Request': 'admin' };
    if (window.SIGNREST_API_CONFIG?.csrfToken) headers['X-CSRF-Token'] = window.SIGNREST_API_CONFIG.csrfToken;
    const response = await fetch(new URL(baseUrl + '/materials' + (id == null ? '' : '/' + encodeURIComponent(id)), location.origin), { method, credentials: 'include', headers, body: body ? JSON.stringify(body) : undefined });
    if (response.status === 401) throw new Error('Please sign in through your application, then retry.');
    if (response.status === 403) throw new Error('Your application did not authorize this change. Please check your administrator session.');
    if (response.status === 409) throw new Error('This material is in use or has changed. Refresh the library and try again.');
    if (!response.ok) throw new Error('The material could not be saved. Check the fields and try again.');
    if (response.status === 204) return;
    if (!(response.headers.get('content-type') || '').includes('application/json')) throw new Error('The FastAPI endpoint is not connected yet.');
    const data = await response.json();
    return normalizeMaterial(data.item);
  }
  function count(value) { if (!Number.isInteger(value) || value < 0) throw new Error('The server returned an invalid overview response.'); return value; }
  return {
    normalizeLog,
    async overview(params, signal) {
      const d = await get('/overview', params, signal);
      if (!d || !d.totals || !Array.isArray(d.activity) || !Array.isArray(d.low_confidence) || !Array.isArray(d.recent_feedback)) throw new Error('The server returned an invalid overview response.');
      const m = d.model;
      return {
        admin_name: typeof d.admin_name === 'string' ? d.admin_name : '',
        totals: { users: count(d.totals.users), translations: count(d.totals.translations), active_sessions: count(d.totals.active_sessions), pending_feedback: count(d.totals.pending_feedback) },
        model: m ? { name: m.model_name || 'Model', version: m.model_version, accuracy: percentage(m.accuracy_percentage), evaluatedAt: m.evaluated_at } : null,
        activity: d.activity.map(a => ({ date: a.date, sign_to_text: count(a.sign_to_text), text_to_sign: count(a.text_to_sign) })),
        low_confidence: d.low_confidence.map(normalizeLog),
        recent_feedback: d.recent_feedback.map(f => ({ id: f.feedback_id, user: f.user_name || 'Unknown user', rating: Number(f.rating_score) || 0, message: f.message, status: f.status || 'pending', createdAt: f.created_at }))
      };
    },
    async materials(params, signal) {
      const data = await get('/materials', params, signal);
      if (!Array.isArray(data.items) || !Number.isInteger(data.total) || data.total < 0 || !Array.isArray(data.categories) || data.categories.some(c => typeof c !== 'string' || !c)) throw new Error('The server returned an invalid materials response.');
      return { items: data.items.map(normalizeMaterial), total: data.total, categories: data.categories };
    },
    createMaterial: body => mutateMaterial('POST', null, body),
    updateMaterial: (id, body) => mutateMaterial('PATCH', id, body),
    deleteMaterial: id => mutateMaterial('DELETE', id),
    async evaluations(signal) {
      const data = await get('/model-accuracy', {}, signal);
      if (!Array.isArray(data.items)) throw new Error('The server returned an invalid evaluation response.');
      return data.items.map(row => ({ name: row.model_name, version: row.model_version, samples: row.total_samples, correct: row.correct_predictions, accuracy: percentage(row.accuracy_percentage), evaluatedAt: row.evaluated_at }));
    },
    async translations(params, signal) {
      const data = await get('/translations', params, signal);
      if (!Array.isArray(data.items) || !Number.isInteger(data.total) || data.total < 0 || !Number.isInteger(data.low_confidence_total) || data.low_confidence_total < 0) throw new Error('The server returned an invalid translation response.');
      return { items: data.items.map(normalizeLog), total: data.total, lowConfidenceTotal: data.low_confidence_total };
    }
  };
})();