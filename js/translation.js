document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;

  var userName = document.getElementById('user-name');
  if (userName) { userName.textContent = A.getProfile().name; }

  /* ---------- Text to Sign ---------- */
  var txt = document.getElementById('txt');
  var signs = document.getElementById('signs');
  var textStatus = document.getElementById('text-status');

  function showSigns() {
    var raw = txt.value.trim();
    var up = raw.toUpperCase();
    signs.innerHTML = '';
    var count = 0, skipped = 0;
    for (var i = 0; i < up.length; i++) {
      var ch = up.charAt(i);
      if (ch >= 'A' && ch <= 'Z') {
        var tile = document.createElement('div');
        tile.className = 'tile';
        tile.appendChild(A.imgBox('signs/' + ch + '.png', ch, 'ASL sign for ' + ch));
        var l = document.createElement('div');
        l.className = 'letter'; l.textContent = ch;
        tile.appendChild(l);
        signs.appendChild(tile);
        count++;
      } else if (ch === ' ') {
        var gap = document.createElement('div');
        gap.className = 'sgap';
        signs.appendChild(gap);
      } else {
        skipped++;
      }
    }
    if (count === 0) {
      signs.innerHTML = '<p class="muted">Type letters A to Z to see the signs.</p>';
      textStatus.textContent = '';
      return;
    }
    textStatus.textContent = skipped
      ? 'Only letters A to Z can be signed. Skipped ' + skipped + ' other character(s).'
      : '';
    A.history.add('Text to Sign', raw);
  }

  document.getElementById('show-sign').addEventListener('click', showSigns);
  txt.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') { showSigns(); }
  });

  var voiceBtn = document.getElementById('voice');
  var voiceLabel = document.getElementById('voice-label');
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceBtn.disabled = true;
    textStatus.textContent = 'Voice input works in Chrome or Edge.';
  } else {
    var recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.onresult = function (event) {
      txt.value = event.results[0][0].transcript;
      showSigns();
    };
    recognition.onend = function () {
      voiceBtn.classList.remove('listening');
      voiceLabel.textContent = 'Speak';
    };
    recognition.onerror = function () {
      textStatus.textContent = 'Could not hear you. Please try again.';
    };
    voiceBtn.addEventListener('click', function () {
      textStatus.textContent = '';
      voiceBtn.classList.add('listening');
      voiceLabel.textContent = 'Listening...';
      try {
        recognition.start();
      } catch (error) {
        textStatus.textContent = 'Voice input is already listening.';
      }
    });
  }

  /* ---------- History ---------- */
  var historyList = document.getElementById('hlist');
  var kindIcons = { 'Text to Sign': 'type' };

  function formatTime(timestamp) {
    return new Date(timestamp).toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function renderHistory() {
    if (!historyList) { return; }
    var items = A.history.list();
    if (!items.length) {
      historyList.innerHTML = '<div class="empty">No translations yet. Your latest results will appear here.</div>';
      return;
    }
    historyList.innerHTML = items.map(function (item) {
      return '<div class="hitem' + (item.saved ? ' saved' : '') + '">' +
        '<div class="kind">' + A.icon(kindIcons[item.kind] || 'chat', 22) + '</div>' +
        '<div class="body"><b>' + A.esc(item.text) + '</b><small>' +
        A.esc(item.kind) + ', ' + formatTime(item.t) + '</small></div>' +
        '<div class="acts">' +
        '<button class="icon-btn" type="button" data-act="save" data-id="' + item.id +
        '" aria-pressed="' + (item.saved ? 'true' : 'false') + '" aria-label="' +
        (item.saved ? 'Unsave' : 'Save') + ' ' + A.esc(item.text) + '">' +
        A.icon('bookmark', 20) + '</button>' +
        '<button class="icon-btn" type="button" data-act="del" data-id="' + item.id +
        '" aria-label="Delete ' + A.esc(item.text) + '">' +
        A.icon('trash', 20) + '</button></div></div>';
    }).join('');
  }

  if (historyList) {
    historyList.addEventListener('click', function (event) {
      var button = event.target.closest('button[data-act]');
      if (!button) { return; }
      if (button.dataset.act === 'save') { A.history.toggleSave(button.dataset.id); }
      if (button.dataset.act === 'del') {
        A.history.remove(button.dataset.id);
        A.toast('Record deleted');
      }
    });

    var clearHistory = document.getElementById('clear-history');
    if (clearHistory) {
      clearHistory.addEventListener('click', function () {
        var kept = A.history.clearUnsaved();
        A.toast(kept ? 'Cleared. Saved records were kept.' : 'History cleared');
      });
    }
  }

  function renderStats() {
    var translations = document.getElementById('st-trans');
    var signsLearned = document.getElementById('st-learn');
    var quizScore = document.getElementById('st-quiz');
    if (!translations || !signsLearned || !quizScore) { return; }
    translations.textContent = A.history.list().length;
    signsLearned.textContent = A.learned.list().length + ' / ' + A.learned.total();
    var best = A.quiz.best();
    quizScore.textContent = best ? best.score + '/' + best.total : 'None yet';
  }

  document.addEventListener('asl:update', function () {
    renderHistory();
    renderStats();
  });
  renderHistory();
  renderStats();
});
