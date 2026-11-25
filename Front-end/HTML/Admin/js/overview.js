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
  const chart = overviewBarChart(d.activity);
  const pie = overviewPieChart(d.activity);
  const notes = (title, rows, empty, link) => '<section class="parchment-panel note-panel"><div class="panel-head"><h3>' + title + '</h3>' + link + '</div>' + (rows.length ? '<ul class="note-list">' + rows.join('') + '</ul>' : '<p class="paper-empty">' + empty + '</p>') + '</section>';
  const low = d.low_confidence.map(r => '<li><div><strong>' + esc(r.output) + '</strong><small>' + esc(r.user) + ' · ' + esc(r.mode) + ' · ' + esc(overviewDate(r.createdAt)) + '</small></div><span class="seal">' + (r.confidence === null ? '—' : r.confidence.toFixed(1) + '%') + '</span></li>');
  const fb = d.recent_feedback.map(f => '<li><div><strong>' + esc(f.user) + ' <span class="stars" aria-label="' + f.rating + ' of 5">' + '★'.repeat(Math.max(0, Math.min(5, f.rating))) + '</span></strong><small>' + esc(f.message || 'No message') + '</small></div><span class="seal soft">' + esc(f.status) + '</span></li>');
  return hero + sectionHeading('Platform overview') + cards + sectionHeading('Translation insights') + '<div class="ov-charts">' + chart + pie + '</div>' + sectionHeading('Activity & notes') + '<div class="paper-columns">' + notes('Needs a second look', low, 'No low-confidence predictions right now.', '<button class="text-link" onclick="navigate(\'logs\')">Open logs</button>') + notes('Recent feedback', fb, 'No feedback yet.', '') + '</div>';
}

/* ---------- Charts (SVG, coloured with the admin theme variables) ---------- */
function overviewPeriodSelect() {
  return '<select class="period" aria-label="Activity period" onchange="changeOverviewDays(this.value)"><option value="7"' + (overviewState.days === 7 ? ' selected' : '') + '>Last 7 days</option><option value="30"' + (overviewState.days === 30 ? ' selected' : '') + '>Last 30 days</option></select>';
}
function overviewBarChart(activity) {
  const total = activity.reduce((s, a) => s + a.sign_to_text + a.text_to_sign, 0);
  const head = '<section class="parchment-panel ov-bar-panel"><div class="panel-head"><div><h3>Translation activity</h3><p>' + overviewNumber(total) + ' translations in this period</p></div>' + overviewPeriodSelect() + '</div>';
  if (!total) return head + '<p class="paper-empty">No translations in this period yet.</p></section>';
  const W = 640, H = 260, L = 38, B = 28, T = 12, plotH = H - B - T, plotW = W - L - 8;
  const max = Math.max(1, ...activity.map(a => Math.max(a.sign_to_text, a.text_to_sign)));
  const nice = Math.ceil(max / 4) * 4 || 4;
  const slot = plotW / activity.length, bw = Math.max(3, Math.min(18, slot * 0.32));
  const step = activity.length > 10 ? Math.ceil(activity.length / 8) : 1;
  let svg = '';
  for (let i = 0; i <= 4; i++) {
    const v = nice * i / 4, y = T + plotH - (v / nice) * plotH;
    svg += '<line class="ov-grid" x1="' + L + '" x2="' + (W - 8) + '" y1="' + y + '" y2="' + y + '"/><text class="ov-axis" x="' + (L - 8) + '" y="' + (y + 4) + '" text-anchor="end">' + overviewNumber(Math.round(v)) + '</text>';
  }
  activity.forEach((a, i) => {
    const cx = L + slot * i + slot / 2, label = overviewDate(a.date, { month: 'short', day: 'numeric' });
    const h1 = a.sign_to_text / nice * plotH, h2 = a.text_to_sign / nice * plotH;
    svg += '<g><title>' + esc(label) + ': ' + a.sign_to_text + ' sign to text, ' + a.text_to_sign + ' text to sign</title>' +
      '<rect class="ov-bar a" rx="3" x="' + (cx - bw - 1) + '" y="' + (T + plotH - h1) + '" width="' + bw + '" height="' + h1 + '"/>' +
      '<rect class="ov-bar b" rx="3" x="' + (cx + 1) + '" y="' + (T + plotH - h2) + '" width="' + bw + '" height="' + h2 + '"/></g>';
    if (i % step === 0) svg += '<text class="ov-axis" x="' + cx + '" y="' + (H - 8) + '" text-anchor="middle">' + esc(label) + '</text>';
  });
  return head + '<svg class="ov-svg" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Bar chart of daily translations">' + svg + '</svg><div class="ink-legend"><span><i class="a"></i>Sign to Text</span><span><i class="b"></i>Text to Sign</span></div></section>';
}
function overviewPieChart(activity) {
  const s = activity.reduce((n, a) => n + a.sign_to_text, 0), t = activity.reduce((n, a) => n + a.text_to_sign, 0), total = s + t;
  const head = '<section class="parchment-panel ov-pie-panel"><div class="panel-head"><div><h3>Mode split</h3><p>Share of translations by direction</p></div></div>';
  if (!total) return head + '<p class="paper-empty">Nothing to compare yet.</p></section>';
  const r = 70, c = 2 * Math.PI * r, sLen = s / total * c;
  const pct = v => (v / total * 100).toFixed(1) + '%';
  const svg = '<svg class="ov-pie" viewBox="0 0 200 200" role="img" aria-label="Pie chart: Sign to Text ' + pct(s) + ', Text to Sign ' + pct(t) + '">' +
    '<circle cx="100" cy="100" r="' + r + '" class="ov-slice b" stroke-width="44" fill="none"/>' +
    (s ? '<circle cx="100" cy="100" r="' + r + '" class="ov-slice a" stroke-width="44" fill="none" stroke-dasharray="' + sLen + ' ' + c + '" transform="rotate(-90 100 100)"/>' : '') +
    '<text x="100" y="96" text-anchor="middle" class="ov-pie-total">' + overviewNumber(total) + '</text><text x="100" y="116" text-anchor="middle" class="ov-axis">total</text></svg>';
  return head + svg + '<ul class="ov-pie-legend"><li><i class="a"></i>Sign to Text<strong>' + overviewNumber(s) + ' · ' + pct(s) + '</strong></li><li><i class="b"></i>Text to Sign<strong>' + overviewNumber(t) + ' · ' + pct(t) + '</strong></li></ul></section>';
}
(function injectOverviewChartStyles() {
  if (document.getElementById('ov-chart-styles')) return;
  const st = document.createElement('style'); st.id = 'ov-chart-styles';
  st.textContent = `
.ov-charts{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:20px;margin-bottom:28px}
.ov-svg{width:100%;height:auto;display:block;margin-top:8px}
.ov-grid{stroke:color-mix(in oklab,var(--text) 14%,transparent);stroke-dasharray:3 4}
.ov-axis{fill:var(--muted);font:11px var(--font-ui)}
.ov-bar.a,.ov-slice.a{fill:var(--sage);stroke:var(--sage)}
.ov-bar.b,.ov-slice.b{fill:var(--tan);stroke:var(--tan)}
.ov-bar{stroke:none!important;transition:opacity .15s}.ov-bar:hover{opacity:.75}
.ov-slice{fill:none!important}
.ov-pie{width:min(210px,100%);height:auto;display:block;margin:10px auto}
.ov-pie-total{fill:var(--text);font:600 24px var(--font-title)}
.ov-pie-legend{list-style:none;padding:0;margin:8px 0 0;display:grid;gap:10px}
.ov-pie-legend li{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:12px}
.ov-pie-legend strong{margin-left:auto;color:var(--text);font-weight:600}
.ov-pie-legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
.ov-pie-legend i.a{background:var(--sage)}.ov-pie-legend i.b{background:var(--tan)}
@media (max-width:900px){.ov-charts{grid-template-columns:1fr}}`;
  document.head.appendChild(st);
})();