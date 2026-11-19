document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var KEY = 'asl_feedback';
  var FACES = [
    { v: 1, e: '😞', l: 'Very bad' }, { v: 2, e: '🙁', l: 'Bad' }, { v: 3, e: '😐', l: 'Okay' },
    { v: 4, e: '🙂', l: 'Good' }, { v: 5, e: '😍', l: 'Great' }
  ];
  var CATS = ['Translation', 'Learning', 'Design', 'Other'];
  var rating = 0, category = 'Translation';

  var faces = document.getElementById('faces');
  var chips = document.getElementById('chips');
  var err = document.getElementById('err');
  var msg = document.getElementById('msg');

  function radio(container, items, getSel, onPick) {
    var btns = [];
    items.forEach(function (it) {
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', 'false');
      it.build(b);
      b.addEventListener('click', function () { onPick(it); refresh(); });
      container.appendChild(b);
      btns.push({ b: b, it: it });
    });
    function refresh() {
      btns.forEach(function (x) { x.b.setAttribute('aria-checked', getSel(x.it) ? 'true' : 'false'); });
    }
    refresh();
  }

  radio(faces, FACES.map(function (f) {
    return { v: f.v, build: function (b) { b.className = 'face'; b.textContent = f.e; b.setAttribute('aria-label', f.l); } };
  }), function (it) { return it.v === rating; }, function (it) { rating = it.v; });

  radio(chips, CATS.map(function (c) {
    return { v: c, build: function (b) { b.className = 'chip'; b.textContent = c; } };
  }), function (it) { return it.v === category; }, function (it) { category = it.v; });

  function fmt(t) { return new Date(t).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }); }

  function renderList() {
    var list = A.read(KEY, []);
    var box = document.getElementById('flist');
    if (!list.length) {
      box.innerHTML = '<div class="empty">Nothing sent yet. Your feedback will show up here.</div>';
      return;
    }
    box.innerHTML = list.map(function (f, i) {
      var face = FACES[f.rating - 1] ? FACES[f.rating - 1].e + ' ' : '';
      return '<div class="fitem"><div style="display:flex;justify-content:space-between;gap:8px;align-items:center">' +
        '<b>' + face + A.esc(f.category) + '</b>' +
        '<button class="icon-btn" type="button" data-i="' + i + '" aria-label="Delete this feedback" style="width:38px;height:38px">' + A.icon('trash', 18) + '</button></div>' +
        (f.message ? '<span>' + A.esc(f.message) + '</span>' : '') +
        '<small>' + fmt(f.t) + '</small></div>';
    }).join('');
  }

  document.getElementById('flist').addEventListener('click', function (e) {
    var b = e.target.closest('button[data-i]');
    if (!b) { return; }
    var list = A.read(KEY, []);
    list.splice(parseInt(b.dataset.i, 10), 1);
    A.write(KEY, list);
    renderList();
    A.toast('Feedback deleted');
  });

  document.getElementById('fb-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var message = msg.value.trim();
    if (!rating && message.length < 3) {
      err.textContent = 'Pick a face or write a short message first.';
      return;
    }
    err.textContent = '';
    var list = A.read(KEY, []);
    list.unshift({ rating: rating, category: category, message: message, t: Date.now() });
    A.write(KEY, list.slice(0, 30));
    document.getElementById('form-view').hidden = true;
    document.getElementById('done-view').hidden = false;
    renderList();
  });

  document.getElementById('again').addEventListener('click', function () {
    rating = 0; msg.value = '';
    document.querySelectorAll('#faces .face').forEach(function (b) { b.setAttribute('aria-checked', 'false'); });
    document.getElementById('done-view').hidden = true;
    document.getElementById('form-view').hidden = false;
  });

  renderList();
});
