document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var WORDS = A.WORDS;
  var NOTES = A.LETTER_NOTES;
  var TOTAL = A.learned.total();

  function $(id) { return document.getElementById(id); }
  function setTabs(list, active) {
    list.forEach(function (t) { t.setAttribute('aria-selected', t === active ? 'true' : 'false'); });
  }

  /* ---------- Progress ---------- */
  function renderProgress() {
    var n = A.learned.list().length;
    $('prog-text').textContent = n + ' of ' + TOTAL + ' signs learned';
    $('prog-bar').style.width = Math.round(n / TOTAL * 100) + '%';
    $('prog').setAttribute('aria-valuenow', n);
    var best = A.quiz.best();
    $('best-text').textContent = best ? 'Best quiz score: ' + best.score + '/' + best.total : 'No quiz taken yet';
  }

  /* ---------- Top tabs ---------- */
  var tabCharts = $('tab-charts'), tabQuiz = $('tab-quiz');
  var quizStarted = false;
  tabCharts.addEventListener('click', function () {
    setTabs([tabCharts, tabQuiz], tabCharts);
    $('charts').hidden = false; $('quiz').hidden = true;
  });
  tabQuiz.addEventListener('click', function () {
    setTabs([tabCharts, tabQuiz], tabQuiz);
    $('charts').hidden = true; $('quiz').hidden = false;
    if (!quizStarted) { startQuiz(); }
  });
  $('to-charts').addEventListener('click', function () { tabCharts.click(); });

  /* ---------- Chart tabs ---------- */
  var tabAlpha = $('tab-alpha'), tabWord = $('tab-word');
  tabAlpha.addEventListener('click', function () {
    setTabs([tabAlpha, tabWord], tabAlpha);
    $('alpha-view').hidden = false; $('word-view').hidden = true;
  });
  tabWord.addEventListener('click', function () {
    setTabs([tabAlpha, tabWord], tabWord);
    $('alpha-view').hidden = true; $('word-view').hidden = false;
  });

  /* ---------- Alphabet ---------- */
  var grid = $('agrid');
  var alphaDetail = $('alpha-detail');
  var tileByLetter = {};

  function markTile(letter) {
    var tile = tileByLetter[letter];
    if (tile && !tile.querySelector('.done')) {
      var d = document.createElement('span');
      d.className = 'done';
      d.innerHTML = A.icon('check', 14);
      tile.appendChild(d);
    }
  }

  for (var i = 0; i < 26; i++) {
    (function (letter) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'atile';
      b.setAttribute('aria-label', 'Letter ' + letter);
      b.appendChild(A.imgBox('signs/' + letter + '.png', letter, 'ASL sign for ' + letter));
      var t = document.createElement('div');
      t.className = 'letter'; t.textContent = letter;
      b.appendChild(t);
      tileByLetter[letter] = b;
      if (A.learned.has(letter)) { markTile(letter); }
      b.addEventListener('click', function () { showLetter(letter); });
      grid.appendChild(b);
    })(String.fromCharCode(65 + i));
  }

  function showLetter(letter) {
    Object.keys(tileByLetter).forEach(function (k) { tileByLetter[k].classList.toggle('sel', k === letter); });
    alphaDetail.hidden = false;
    alphaDetail.innerHTML = '';
    alphaDetail.appendChild(A.imgBox('signs/' + letter + '.png', letter, 'ASL sign for ' + letter));
    var txt = document.createElement('div');
    txt.className = 'txt';
    txt.innerHTML = '<h3>' + letter + '</h3><p>' + A.esc(NOTES[letter] || '') + '</p>';
    alphaDetail.appendChild(txt);
    A.learned.mark(letter);
    markTile(letter);
    alphaDetail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ---------- Words ---------- */
  var wgrid = $('wgrid');
  var wordDetail = $('word-detail');
  var tileByWord = {};

  function markWordTile(id) {
    var tile = tileByWord[id];
    if (tile && !tile.querySelector('.done')) {
      var d = document.createElement('span');
      d.className = 'done';
      d.innerHTML = A.icon('check', 14);
      tile.appendChild(d);
    }
  }

  WORDS.forEach(function (w) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'atile wtile';
    b.setAttribute('aria-label', 'Word ' + w.label);
    b.appendChild(A.imgBox('signs/words/' + w.id + '.png', w.label, 'ASL sign for ' + w.label));
    var t = document.createElement('div');
    t.className = 'letter word'; t.textContent = w.label;
    b.appendChild(t);
    tileByWord[w.id] = b;
    if (A.learned.has('w:' + w.id)) { markWordTile(w.id); }
    b.addEventListener('click', function () { showWord(w); });
    wgrid.appendChild(b);
  });

  function showWord(w) {
    Object.keys(tileByWord).forEach(function (k) { tileByWord[k].classList.toggle('sel', k === w.id); });
    wordDetail.hidden = false;
    wordDetail.innerHTML = '';
    wordDetail.appendChild(A.imgBox('signs/words/' + w.id + '.png', w.label, 'ASL sign for ' + w.label));
    var txt = document.createElement('div');
    txt.className = 'txt';
    txt.innerHTML = '<h3>' + A.esc(w.label) + '</h3><p>' + A.esc(w.note) + '</p>';
    wordDetail.appendChild(txt);
    A.learned.mark('w:' + w.id);
    markWordTile(w.id);
    wordDetail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* ---------- Quiz ---------- */
  var QUESTIONS = [];
  var letters = Object.keys(NOTES);
  var wordLabels = WORDS.map(function (w) { return w.label; });
  letters.forEach(function (l) {
    QUESTIONS.push({ text: 'Which letter is this sign?', hint: NOTES[l], img: 'signs/' + l + '.png', label: l, answer: l, pool: letters });
  });
  WORDS.forEach(function (w) {
    QUESTIONS.push({ text: 'Which word is this sign?', hint: w.note, img: 'signs/words/' + w.id + '.png', label: w.label, answer: w.label, pool: wordLabels });
  });

  var QUIZ_LEN = 10;
  var order = [], current = 0, score = 0, locked = false;

  function shuffle(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  function startQuiz() {
    quizStarted = true;
    order = shuffle(QUESTIONS).slice(0, QUIZ_LEN);
    current = 0; score = 0;
    $('quiz-play').hidden = false; $('quiz-result').hidden = true;
    showQuestion();
  }

  function showQuestion() {
    var q = order[current];
    locked = false;
    $('q-count').textContent = 'Question ' + (current + 1) + ' of ' + order.length;
    $('q-score').textContent = 'Score: ' + score;
    $('q-bar').style.width = (current / order.length * 100) + '%';
    $('q-text').textContent = q.text;
    $('q-hint').textContent = q.hint;
    var holder = $('q-img');
    holder.innerHTML = '';
    holder.appendChild(A.imgBox(q.img, '?', 'Sign to identify'));
    var fb = $('q-feedback');
    fb.textContent = ''; fb.className = 'fbline';
    $('q-next').hidden = true;

    var others = shuffle(q.pool.filter(function (x) { return x !== q.answer; })).slice(0, 3);
    var opts = shuffle(others.concat([q.answer]));
    var box = $('q-options');
    box.innerHTML = '';
    opts.forEach(function (o) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'opt';
      b.textContent = o;
      b.addEventListener('click', function () { answer(b, o, q); });
      box.appendChild(b);
    });
  }

  function answer(btn, chosen, q) {
    if (locked) { return; }
    locked = true;
    var fb = $('q-feedback');
    document.querySelectorAll('#q-options .opt').forEach(function (b) {
      b.disabled = true;
      if (b.textContent === q.answer) { b.classList.add('right'); }
    });
    if (chosen === q.answer) {
      score++;
      fb.textContent = 'Correct!';
      fb.className = 'fbline right';
    } else {
      btn.classList.add('wrong');
      fb.textContent = 'Not quite. The answer is ' + q.answer + '.';
      fb.className = 'fbline wrong';
    }
    $('q-score').textContent = 'Score: ' + score;
    var next = $('q-next');
    next.textContent = (current === order.length - 1) ? 'See my score' : 'Next question';
    next.hidden = false;
    next.focus();
  }

  $('q-next').addEventListener('click', function () {
    current++;
    if (current < order.length) { showQuestion(); } else { finishQuiz(); }
  });

  function finishQuiz() {
    $('q-bar').style.width = '100%';
    $('quiz-play').hidden = true; $('quiz-result').hidden = false;
    var pct = Math.round(score / order.length * 100);
    $('ring').style.setProperty('--p', pct);
    $('final-score').textContent = score + '/' + order.length;
    var msg = 'Keep practising!';
    if (pct >= 80) { msg = 'Great job!'; } else if (pct >= 50) { msg = 'Good effort!'; }
    $('final-msg').textContent = msg;
    A.quiz.add(score, order.length);
    var best = A.quiz.best();
    $('final-best').textContent = 'Best score: ' + best.score + '/' + best.total;
  }
  $('restart').addEventListener('click', startQuiz);

  document.addEventListener('asl:update', renderProgress);
  renderProgress();
});
