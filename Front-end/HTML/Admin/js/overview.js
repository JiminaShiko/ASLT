let overviewState = { status: 'loading', error: '', data: null, days: 7, updatedAt: null };
let overviewRequest;
const overviewFormatter = new Intl.NumberFormat();
const overviewNumber = value => overviewFormatter.format(value);

function overviewDate(value, options = { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) {
  if (!value) return 'Not available';
  // The API's daily buckets are calendar dates, independent of browser time zone.
  const isCalendarDate = /^\d{4}-\d{2}-\d{2}$/.test(value);
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Not available' :
    date.toLocaleString(undefined, isCalendarDate ? { ...options, timeZone: 'UTC' } : options);
}

async function loadOverview(restoreFocus = false) {
  overviewRequest?.abort();
  const request = new AbortController();
  overviewRequest = request;
  overviewState = { ...overviewState, status: 'loading', error: '' };

  const renderOverview = () => {
    const focus = restoreFocus && (
      document.activeElement === document.body || document.activeElement?.id === 'overview-period'
    );
    renderContent();
    updateConnection();
    if (focus) {
      (document.getElementById('overview-period') || document.getElementById('overview-retry'))
        ?.focus({ preventScroll: true });
    }
  };

  renderOverview();
  try {
    const data = await SignrestAPI.overview({ days: overviewState.days }, request.signal);
    if (request.signal.aborted) return;
    overviewState = { ...overviewState, status: 'ready', data, updatedAt: new Date() };
    if (data.admin_name) {
      state.profile.name = data.admin_name;
      updateProfile();
    }
  } catch (error) {
    if (request.signal.aborted || error.name === 'AbortError') return;
    overviewState = { ...overviewState, status: 'error', error: error.message || 'The overview could not be loaded.' };
  }
  renderOverview();
}

function changeOverviewDays(value) {
  overviewState.days = value === '30' ? 30 : 7;
  loadOverview(true);
}

function overviewMetric(label, value, note, symbol, accent = '') {
  return `<article class="card overview-metric">
    <div class="overview-metric-heading">
      <h3>${esc(label)}</h3>
      <span class="overview-icon ${accent}">${icon(symbol)}</span>
    </div>
    <p class="overview-value">${esc(value)}</p>
    <p class="overview-muted">${esc(note)}</p>
  </article>`;
}

function overviewPanelHeading(title, description = '', action = '') {
  return `<div class="overview-panel-heading">
    <div><h3>${title}</h3>${description ? `<p>${description}</p>` : ''}</div>
    ${action}
  </div>`;
}

function overviewView() {
  const name = (state.profile.name || 'Administrator').trim().split(/\s+/)[0] || 'Administrator';
  const today = new Date();
  const hour = today.getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
  const dateLabel = today.toLocaleDateString(undefined, {
    weekday: 'long', month: 'long', day: 'numeric', year: 'numeric'
  });
  const hero = `<section class="overview-hero" aria-labelledby="overview-title">
    <div class="overview-hero-copy">
      <div class="eyebrow">${icon('leaf')}Your Signrest workspace</div>
      <h1 id="overview-title">${greeting}, ${esc(name)}.</h1>
      <p>Every gesture brings us closer. Here’s how your community is growing.</p>
    </div>
    <div class="overview-hero-date">${icon('calendar')}${esc(dateLabel)}</div>
  </section>`;

  const { data, status, error, updatedAt } = overviewState;
  let content;
  if (status === 'error') {
    content = `<div class="card overview-state" role="alert">
      <p>${esc(error)}</p>
      <button class="btn outline" id="overview-retry" type="button" onclick="loadOverview()">Retry</button>
    </div>`;
  } else if (!data) {
    content = '<div class="card overview-state" role="status">Loading overview…</div>';
  } else {
    const { totals, model, activity, low_confidence, recent_feedback } = data;
    const updating = status === 'loading';
    content = `<section class="overview-summary" aria-labelledby="overview-summary-title">
      <div class="overview-section-heading">
        <h2 id="overview-summary-title">Your platform, at a glance</h2>
        <span class="overview-muted">Updated <time datetime="${updatedAt.toISOString()}">${esc(overviewDate(updatedAt, { hour: 'numeric', minute: '2-digit' }))}</time></span>
      </div>
      <div class="overview-metrics">
        ${overviewMetric('Total Users', overviewNumber(totals.users), 'Registered accounts · ' + overviewNumber(totals.active_sessions) + ' active sessions', 'users')}
        ${overviewMetric('Total Translations', overviewNumber(totals.translations), 'All time', 'translate', 'warm')}
        ${overviewMetric('Model Accuracy', model?.accuracy != null ? model.accuracy.toFixed(1) + '%' : '—',
          model?.accuracy != null ? 'Latest evaluation' : 'No accuracy recorded', 'shield')}
        ${overviewMetric('Pending Feedback', overviewNumber(totals.pending_feedback),
          totals.pending_feedback ? 'Awaiting review' : 'No feedback awaiting review', 'message', 'warm')}
      </div>
      <div class="overview-insights">
        ${overviewBarChart(activity, updating)}
        ${overviewPerformance(model)}
      </div>
    </section>
    <section class="overview-activity" aria-labelledby="overview-activity-title">
      <div class="overview-section-heading"><h2 id="overview-activity-title">Activity &amp; notes</h2></div>
      <div class="overview-activity-grid">
        ${overviewModeSplit(activity, updating)}
        ${overviewNotes(low_confidence, recent_feedback)}
      </div>
    </section>`;
  }

  return `<div class="overview-page">${hero}${content}</div>`;
}

function overviewPeriodSelect() {
  return `<select class="period overview-period" id="overview-period" aria-label="Activity period" onchange="changeOverviewDays(this.value)">
    <option value="7"${overviewState.days === 7 ? ' selected' : ''}>Last 7 days</option>
    <option value="30"${overviewState.days === 30 ? ' selected' : ''}>Last 30 days</option>
  </select>`;
}

function overviewBarChart(activity, updating) {
  const total = activity.reduce((sum, day) => sum + day.sign_to_text + day.text_to_sign, 0);
  const heading = overviewPanelHeading('Translation activity', 'Daily translations by direction', overviewPeriodSelect());
  let chart;
  if (updating) {
    chart = '<p class="overview-empty" role="status">Loading translation activity…</p>';
  } else if (!total) {
    chart = '<p class="overview-empty">No translations in this period yet.</p>';
  } else {
    chart = `<div class="overview-legend">
      <span><i></i>Sign to Text</span><span><i class="warm"></i>Text to Sign</span>
    </div>${overviewChartSvg(activity)}`;
  }
  return `<section class="card overview-panel overview-chart">
    ${heading}${chart}
    ${updating ? '' : `<div class="overview-panel-footer">
      <strong>${overviewNumber(total)}</strong> translations in the last ${overviewState.days} days
    </div>`}
  </section>`;
}

function overviewChartSvg(activity) {
  const width = 640, height = 196, left = 56, right = 12, top = 12, base = 169;
  const plotHeight = base - top;
  const max = Math.max(...activity.map(day => Math.max(day.sign_to_text, day.text_to_sign)));
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(1, max / 4)));
  const rawStep = max / 4 / magnitude;
  const tick = magnitude * [1, 2, 2.5, 5, 10].find(step => step >= rawStep);
  const scale = Math.max(1, Math.ceil(tick)) * 4;
  const slot = (width - left - right) / activity.length;
  const barWidth = Math.min(18, (slot - 4) / 2);
  const labelStep = activity.length > 7 ? Math.ceil(activity.length / 7) : 1;
  let svg = '';

  for (let line = 0; line <= 4; line++) {
    const y = base - plotHeight * line / 4;
    svg += `<line class="overview-chart-line" x1="${left}" x2="${width - right}" y1="${y}" y2="${y}"/>
      <text x="${left - 10}" y="${y + 4}" text-anchor="end">${overviewNumber(scale * line / 4)}</text>`;
  }

  const dailyValues = [];
  activity.forEach((day, index) => {
    const x = left + slot * (index + 0.5);
    const label = overviewDate(day.date, activity.length > 7 ? { month: 'short', day: 'numeric' } : { weekday: 'short' });
    const fullDate = overviewDate(day.date, { month: 'long', day: 'numeric', year: 'numeric' });
    const signHeight = day.sign_to_text / scale * plotHeight;
    const textHeight = day.text_to_sign / scale * plotHeight;
    const counts = `${fullDate}: ${day.sign_to_text} Sign to Text, ${day.text_to_sign} Text to Sign`;
    dailyValues.push(counts);
    svg += `<g><title>${esc(counts)}</title>
      <rect class="overview-bar" rx="2" x="${x - barWidth - 2}" y="${base - signHeight}" width="${barWidth}" height="${signHeight}"/>
      <rect class="overview-bar warm" rx="2" x="${x + 2}" y="${base - textHeight}" width="${barWidth}" height="${textHeight}"/>
    </g>`;
    if (index % labelStep === 0) svg += `<text x="${x}" y="${height - 8}" text-anchor="middle">${esc(label)}</text>`;
  });

  return `<div class="overview-chart-plot"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Daily translation counts" aria-describedby="overview-chart-data">${svg}</svg></div>
    <p class="overview-visually-hidden" id="overview-chart-data">${esc(dailyValues.join('. '))}</p>`;
}

function overviewPerformance(model) {
  const accuracy = model?.accuracy;
  const evaluation = model
    ? `<div class="overview-performance-details">
      <div class="overview-performance-label">
        <span>${esc([model.name, model.version].filter(Boolean).join(' · '))}</span>
        <strong>${accuracy == null ? '—' : accuracy.toFixed(1) + '%'}</strong>
      </div>
      ${accuracy == null ? '<p class="overview-muted">No accuracy recorded for this evaluation.</p>' :
        `<meter class="overview-meter" min="0" max="100" value="${accuracy}" aria-label="Latest model accuracy">${accuracy.toFixed(1)}%</meter>`}
      <p class="overview-muted">Evaluated ${esc(overviewDate(model.evaluatedAt, { dateStyle: 'medium' }))}</p>
    </div>`
    : '<p class="overview-empty">No model evaluations yet.</p>';

  return `<section class="card overview-panel">
    ${overviewPanelHeading('Model performance', 'Latest recorded evaluation', icon('chart'))}
    ${evaluation}
    <div class="overview-panel-footer overview-performance-footer">
      <button class="text-button" type="button" onclick="navigate('logs')">View accuracy &amp; logs ${icon('arrow')}</button>
    </div>
  </section>`;
}

function overviewModeSplit(activity, updating) {
  const sign = activity.reduce((sum, day) => sum + day.sign_to_text, 0);
  const text = activity.reduce((sum, day) => sum + day.text_to_sign, 0);
  const total = sign + text;
  const share = value => (value / total * 100).toFixed(1) + '%';
  let split = '<p class="overview-empty">No translations to compare yet.</p>';
  if (updating) {
    split = '<p class="overview-empty" role="status">Loading translation split…</p>';
  } else if (total) {
    split = `<div class="overview-mode">
      <div class="overview-mode-track" aria-hidden="true"><span style="width:${share(sign)}"></span></div>
      <ul class="overview-list overview-mode-list">
        <li><span>Sign to Text</span><strong>${overviewNumber(sign)} · ${share(sign)}</strong></li>
        <li><span>Text to Sign</span><strong>${overviewNumber(text)} · ${share(text)}</strong></li>
      </ul>
    </div>`;
  }
  return `<section class="card overview-panel">${overviewPanelHeading('Mode split', 'Share of translations by direction')}${split}</section>`;
}

function overviewNotes(low, feedback) {
  const notePanel = (title, rows, empty, link = '') => `<section class="card overview-panel">
    ${overviewPanelHeading(title, '', link)}
    ${rows.length ? `<ul class="overview-list overview-notes">${rows.join('')}</ul>` : `<p class="overview-empty">${empty}</p>`}
  </section>`;
  const predictions = low.map(row => `<li>
    <div><strong>${esc(row.output)}</strong><small>${esc(row.user)} · ${esc(row.mode)} · ${esc(overviewDate(row.createdAt))}</small></div>
    <span class="badge coral">${row.confidence == null ? '—' : row.confidence.toFixed(1) + '%'}</span>
  </li>`);
  const feedbackRows = feedback.map(row => `<li>
    <div><strong>${esc(row.user)}${row.rating > 0 ? ` <span class="overview-muted">· ${esc(row.rating)}/5</span>` : ''}</strong><small>${esc(row.message || 'No message')}</small></div>
    <span class="badge neutral">${esc(row.status)}</span>
  </li>`);
  return notePanel('Low-confidence predictions', predictions, 'No low-confidence predictions right now.',
    `<button class="text-button" type="button" onclick="navigate('logs', 'uncertain')">Review ${icon('arrow')}</button>`) +
    notePanel('Recent feedback', feedbackRows, 'No feedback yet.');
}
