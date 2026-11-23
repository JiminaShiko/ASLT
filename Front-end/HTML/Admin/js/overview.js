let overviewState = { status: 'loading', error: '', data: null, days: 7 };
let overviewRequest;
const overviewNumber = value => value === null || value === undefined ? '—' : new Intl.NumberFormat().format(value);
function overviewDate(value, opts) { if (!value) return 'Not available'; const d = new Date(value); return Number.isNaN(d.getTime()) ? 'Not available' : d.toLocaleString(undefined, opts || { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }); }
async function loadOverview() {
  overviewRequest?.abort();
  overviewRequest = new AbortController();
  overviewState = { ...overviewState, status: 'loading', error: '', data: null };
  renderContent(); updateConnection();
  try {
    const data = await SignrestAPI.overview({ days: overviewState.days }, overviewRequest.signal);
    overviewState = { ...overviewState, status: 'ready', data };
    if (data.admin_name) { state.profile.name = data.admin_name; updateProfile(); }
  } catch (error) {
    if (error.name === 'AbortError') return;
    overviewState = { ...overviewState, status: 'error', error: error.message || 'The overview could not be loaded.' };
  }
  renderContent(); updateConnection();
}
function changeOverviewDays(value) { overviewState.days = value === '30' ? 30 : 7; loadOverview(); }
function sectionHeading(text) {
  return '<div class="overview-section-heading"><h2>' + esc(text) + '</h2></div>';
}
function leafCard(label, value, note) {
  return '<article class="leaf-card"><div class="torn"><div class="leaf-label">' + esc(label) + '</div><div class="leaf-value">' + esc(value) + '</div><div class="leaf-note">' + esc(note) + '</div></div></article>';
}
function overviewView() {
  const hello = 'Welcome back, ' + esc((state.profile.name || 'Administrator').split(' ')[0]) + '.';
  const today = new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
  const hero = '<section class="scroll-hero"><span class="roller"></span><div class="scroll-paper"><div class="eyebrow">' + icon('leaf') + 'YOUR SIGNREST WORKSPACE</div><h1>' + hello + '</h1><p>Every gesture brings us closer. Here is how your community is growing.</p><div class="scroll-date">' + icon('calendar') + esc(today) + '</div></div><span class="roller"></span></section>';
  if (overviewState.status === 'loading') return hero + '<div class="paper-state" role="status">Gathering the latest records…</div>';
  if (overviewState.status === 'error') return hero + '<div class="paper-state error" role="alert"><p>' + esc(overviewState.error) + '</p><button class="btn" onclick="loadOverview()">Retry</button></div>';
  const d = overviewState.data, t = d.totals, m = d.model;
  const cards = '<div class="leaf-grid">' +
    leafCard('Total Users', overviewNumber(t.users), 'Registered accounts') +
    leafCard('Total Translations', overviewNumber(t.translations), 'All time') +
    leafCard('Model Accuracy', m && m.accuracy !== null ? m.accuracy.toFixed(1) + '%' : 'Not available', m ? (m.name + (m.version ? ' · ' + m.version : '')) : 'No evaluation yet') +
    leafCard('Pending Feedback', overviewNumber(t.pending_feedback), overviewNumber(t.active_sessions) + ' active sessions now') + '</div>';
  const max = Math.max(1, ...d.activity.map(a => a.sign_to_text + a.text_to_sign));
  const total = d.activity.reduce((s, a) => s + a.sign_to_text + a.text_to_sign, 0);
  const step = d.activity.length > 10 ? Math.ceil(d.activity.length / 8) : 1;
  const chart = '<section class="parchment-panel"><div class="panel-head"><div><h3>Translation activity</h3><p>' + overviewNumber(total) + ' translations in this period</p></div><select class="period" aria-label="Activity period" onchange="changeOverviewDays(this.value)"><option value="7"' + (overviewState.days === 7 ? ' selected' : '') + '>Last 7 days</option><option value="30"' + (overviewState.days === 30 ? ' selected' : '') + '>Last 30 days</option></select></div>' +
    (total ? '<div class="ink-chart">' + d.activity.map((a, i) => '<div class="ink-col" title="' + esc(overviewDate(a.date, { month: 'short', day: 'numeric' })) + ': ' + a.sign_to_text + ' sign to text, ' + a.text_to_sign + ' text to sign"><div class="ink-stack"><span class="ink-bar b" style="height:' + (a.text_to_sign / max * 100) + '%"></span><span class="ink-bar a" style="height:' + (a.sign_to_text / max * 100) + '%"></span></div><small>' + (i % step === 0 ? esc(overviewDate(a.date, { month: 'short', day: 'numeric' })) : '') + '</small></div>').join('') + '</div><div class="ink-legend"><span><i class="a"></i>Sign to Text</span><span><i class="b"></i>Text to Sign</span></div>' : '<p class="paper-empty">No translations in this period yet.</p>') + '</section>';
  const notes = (title, rows, empty, link) => '<section class="parchment-panel note-panel"><div class="panel-head"><h3>' + title + '</h3>' + link + '</div>' + (rows.length ? '<ul class="note-list">' + rows.join('') + '</ul>' : '<p class="paper-empty">' + empty + '</p>') + '</section>';
  const low = d.low_confidence.map(r => '<li><div><strong>' + esc(r.output) + '</strong><small>' + esc(r.user) + ' · ' + esc(r.mode) + ' · ' + esc(overviewDate(r.createdAt)) + '</small></div><span class="seal">' + (r.confidence === null ? '—' : r.confidence.toFixed(1) + '%') + '</span></li>');
  const fb = d.recent_feedback.map(f => '<li><div><strong>' + esc(f.user) + ' <span class="stars" aria-label="' + f.rating + ' of 5">' + '★'.repeat(Math.max(0, Math.min(5, f.rating))) + '</span></strong><small>' + esc(f.message || 'No message') + '</small></div><span class="seal soft">' + esc(f.status) + '</span></li>');
  return hero + sectionHeading('Platform overview') + cards + sectionHeading('Activity & notes') + '<div class="paper-columns">' + chart + notes('Needs a second look', low, 'No low-confidence predictions right now.', '<button class="text-link" onclick="navigate(\'logs\')">Open logs</button>') + '</div>' + '<div class="paper-columns single">' + notes('Recent feedback', fb, 'No feedback yet.', '') + '</div>';
}
