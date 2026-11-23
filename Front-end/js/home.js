document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var DAY = 24 * 60 * 60 * 1000; // recent history = last 24 hours

  document.getElementById('user-name').textContent = A.getProfile().name;

  var hlist = document.getElementById('hlist');

  function renderHistory() {
    var since = Date.now() - DAY;
    var list = A.history.list().filter(function (h) { return h.t >= since; });
    if (!list.length) {
      hlist.innerHTML = '<div class="empty">No translations in the last 24 hours. Older ones are in History.</div>';
      return;
    }
    hlist.innerHTML = list.map(A.historyItemHtml).join('');
  }

  A.bindHistoryActions(hlist);

  function renderStats() {
    document.getElementById('st-trans').textContent = A.history.list().length;
    document.getElementById('st-learn').textContent = A.learned.list().length + ' / ' + A.learned.total();
    var best = A.quiz.best();
    document.getElementById('st-quiz').textContent = best ? best.score + '/' + best.total : 'None yet';
  }

  document.addEventListener('asl:update', function () { renderHistory(); renderStats(); });
  renderHistory();
  renderStats();
});
