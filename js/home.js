document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  document.getElementById('user-name').textContent = A.getProfile().name;

  var historyList = document.getElementById('hlist');
  var dayAgo = Date.now() - 24 * 60 * 60 * 1000;
  var kindIcons = { 'Text to Sign': 'type' };

  function renderHistory() {
    var recent = A.history.list().filter(function (item) {
      return typeof item.t === 'number' && item.t >= dayAgo;
    }).slice(0, 10);
    if (!recent.length) {
      historyList.innerHTML = '<div class="empty">No translations in the last 24 hours.</div>';
      return;
    }
    historyList.innerHTML = recent.map(function (item) {
      return '<div class="hitem' + (item.saved ? ' saved' : '') + '">' +
        '<div class="kind">' + A.icon(kindIcons[item.kind] || 'chat', 22) + '</div>' +
        '<div class="body"><b>' + A.esc(item.text) + '</b><small>' +
        A.esc(item.kind) + ', ' + new Date(item.t).toLocaleString([], {
          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        }) + '</small></div></div>';
    }).join('');
  }

  function renderStats() {
    document.getElementById('st-trans').textContent = A.history.list().length;
    document.getElementById('st-learn').textContent =
      A.learned.list().length + ' / ' + A.learned.total();
    var best = A.quiz.best();
    document.getElementById('st-quiz').textContent = best ? best.score + '/' + best.total : 'None yet';
  }

  document.addEventListener('asl:update', function () {
    renderHistory();
    renderStats();
  });
  renderHistory();
  renderStats();
});
