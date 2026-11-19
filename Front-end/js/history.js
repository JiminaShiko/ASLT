document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var hlist = document.getElementById('hlist');
  var count = document.getElementById('h-count');

  function dayLabel(t) {
    var d = new Date(t);
    var today = new Date(); today.setHours(0, 0, 0, 0);
    var that = new Date(d); that.setHours(0, 0, 0, 0);
    var diff = Math.round((today - that) / 86400000);
    if (diff === 0) { return 'Today'; }
    if (diff === 1) { return 'Yesterday'; }
    return d.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
  }

  function renderHistory() {
    var list = A.history.list();
    count.textContent = list.length + (list.length === 1 ? ' record' : ' records');
    if (!list.length) {
      hlist.innerHTML = '<div class="empty">No translations yet. Your results will appear here.</div>';
      return;
    }
    var html = '', last = '';
    list.forEach(function (h) {
      var day = dayLabel(h.t);
      if (day !== last) {
        html += '<h3 class="day-h">' + A.esc(day) + '</h3>';
        last = day;
      }
      html += A.historyItemHtml(h);
    });
    hlist.innerHTML = html;
  }

  A.bindHistoryActions(hlist);

  document.getElementById('clear-history').addEventListener('click', function () {
    var kept = A.history.clearUnsaved();
    A.toast(kept ? 'Cleared. Saved records were kept.' : 'History cleared');
  });

  document.addEventListener('asl:update', renderHistory);
  renderHistory();
});
