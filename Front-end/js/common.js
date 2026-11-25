/* Shared shell: icons, storage, nav, cursor effects */
(function () {
  'use strict';

  var ICONS = {
    home: '<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/>',
    chat: '<path d="M4 5h16v11H9l-5 4z"/>',
    book: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    star: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    logout: '<path d="M9 4H5v16h4"/><path d="M16 8l4 4-4 4"/><path d="M20 12H9"/>',
    camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    type: '<path d="M5 6V4h14v2"/><path d="M12 4v16"/><path d="M9 20h6"/>',
    upload: '<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 20h16"/>',
    flip: '<path d="M20 11a8 8 0 0 0-14-4L4 9"/><path d="M4 4v5h5"/><path d="M4 13a8 8 0 0 0 14 4l2-2"/><path d="M20 20v-5h-5"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/>',
    hand: '<path d="M8 13V5a1.5 1.5 0 0 1 3 0v6"/><path d="M11 11V3.5a1.5 1.5 0 0 1 3 0V11"/><path d="M14 11V5.5a1.5 1.5 0 0 1 3 0V14"/><path d="M17 9.5a1.5 1.5 0 0 1 3 0V15a7 7 0 0 1-7 7h-1a7 7 0 0 1-5.5-2.7L3 14.5a1.5 1.5 0 0 1 2.3-1.9L8 15"/>',
    trash: '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 13h10l1-13"/>',
    bookmark: '<path d="M6 3h12v18l-6-4-6 4z"/>',
    check: '<path d="M5 12l5 5 9-10"/>',
    x: '<path d="M6 6l12 12M18 6L6 18"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    sparkle: '<path d="M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z"/>',
    arrow: '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 9"/>',
    video: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/>',
    send: '<path d="M21 3L3 11l7 3 3 7z"/><path d="M10 14l11-11"/>',
    alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v5"/><path d="M12 18v.5"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>',
    eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'
  };

  function icon(name, size) {
    size = size || 24;
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + (ICONS[name] || '') + '</svg>';
  }

  function paintIcons(root) {
    (root || document).querySelectorAll('[data-icon]').forEach(function (el) {
      if (el.dataset.painted) { return; }
      el.innerHTML = icon(el.dataset.icon, parseInt(el.dataset.size || '24', 10));
      el.dataset.painted = '1';
    });
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function read(key, fallback) {
    try {
      var v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch (e) { return fallback; }
  }
  function write(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {}
  }
  function emit() { document.dispatchEvent(new CustomEvent('asl:update')); }

  var LETTER_NOTES = {
    A: 'Make a fist and keep your thumb on the side.',
    B: 'Hold your fingers straight up together. Fold your thumb across your palm.',
    C: 'Curve your hand into the shape of the letter C.',
    D: 'Point your index finger up. Touch your thumb to the other fingers to make a circle.',
    E: 'Bend your fingers down and tuck your thumb underneath them.',
    F: 'Touch your thumb and index finger to make a circle. Keep the other three fingers up.',
    G: 'Point your index finger and thumb sideways, side by side.',
    H: 'Point your index and middle fingers sideways together.',
    I: 'Make a fist and raise your little finger.',
    J: 'Raise your little finger, then draw a J in the air.',
    K: 'Raise your index and middle fingers in a V. Place your thumb between them.',
    L: 'Point your index finger up and your thumb out, like the letter L.',
    M: 'Tuck your thumb under your first three fingers.',
    N: 'Tuck your thumb under your first two fingers.',
    O: 'Touch your fingertips to your thumb to make an O.',
    P: 'Make a K shape, then point it downward.',
    Q: 'Make a G shape, then point it downward.',
    R: 'Cross your index and middle fingers.',
    S: 'Make a fist with your thumb across the front of your fingers.',
    T: 'Make a fist and tuck your thumb between your index and middle fingers.',
    U: 'Raise your index and middle fingers together.',
    V: 'Raise your index and middle fingers apart, like a peace sign.',
    W: 'Raise your index, middle and ring fingers and spread them.',
    X: 'Bend your index finger like a hook.',
    Y: 'Stick out your thumb and little finger.',
    Z: 'Point your index finger and draw a Z in the air.'
  };

  var WORDS = [
    { id: 'eat', label: 'Eat', note: 'Bunch your fingertips together with your thumb. Tap them to your mouth.' },
    { id: 'family', label: 'Family', note: 'Make F hands with both hands and touch them together. Circle them outward until your little fingers meet.' },
    { id: 'friend', label: 'Friend', note: 'Hook your index fingers together. Then flip your hands and hook them again the other way.' },
    { id: 'good', label: 'Good', note: 'Touch your flat hand to your chin. Move it down onto your other palm.' },
    { id: 'hello', label: 'Hello', note: 'Hold a flat hand at your forehead like a salute, then move it away.' },
    { id: 'help', label: 'Help', note: 'Make a fist with your thumb up and rest it on your other open palm. Lift both up.' },
    { id: 'name', label: 'Name', note: 'Make H hands with your index and middle fingers together. Tap one across the other twice.' },
    { id: 'sad', label: 'Sad', note: 'Hold both open hands in front of your face. Drop them down slowly with a sad face.' },
    { id: 'school', label: 'School', note: 'Clap your flat hands together twice.' },
    { id: 'drink', label: 'Drink', note: 'Shape your hand like you are holding a cup. Tip it toward your mouth.' },
    { id: 'thanks', label: 'Thanks', note: 'Touch your fingertips to your chin, then move your hand forward and down.' },
    { id: 'love', label: 'Love', note: 'Cross both fists over your chest, like a hug.' },
    { id: 'bye', label: 'Bye', note: 'Hold up an open hand and wave it by bending your fingers down and up.' },
    { id: 'sorry', label: 'Sorry', note: 'Make a fist and rub it in a circle on your chest.' },
    { id: 'yes', label: 'Yes', note: 'Make a fist and nod it up and down, like a head nodding.' },
    { id: 'no', label: 'No', note: 'Tap your index and middle fingers against your thumb, like a beak closing.' },
    { id: 'happy', label: 'Happy', note: 'Brush your flat hand upward on your chest twice, with a smile.' },
    { id: 'please', label: 'Please', note: 'Rub your flat hand in a circle on your chest.' },
    { id: 'bad', label: 'Bad', note: 'Touch your fingertips to your chin. Flip your hand down and away, palm down.' },
    { id: 'home', label: 'Home', note: 'Bunch your fingertips and touch them near your mouth. Then touch them near your cheek.' }
  ];

  var HK = 'asl_history';
  var history = {
    list: function () { return read(HK, []); },
    add: function (kind, text, notes) {
      var l = read(HK, []);
      var rec = { id: Date.now() + '' + Math.floor(Math.random() * 1000), kind: kind, text: text, t: Date.now(), saved: false };
      if (notes && notes.length) { rec.notes = notes; }
      l.unshift(rec);
      var saved = l.filter(function (x) { return x.saved; });
      var rest = l.filter(function (x) { return !x.saved; }).slice(0, 50);
      write(HK, saved.concat(rest).sort(function (a, b) { return b.t - a.t; }));
      emit();
    },
    toggleSave: function (id) {
      var l = read(HK, []);
      l.forEach(function (x) { if (x.id === id) { x.saved = !x.saved; } });
      write(HK, l); emit();
    },
    remove: function (id) {
      write(HK, read(HK, []).filter(function (x) { return x.id !== id; })); emit();
    },
    clearUnsaved: function () {
      var keep = read(HK, []).filter(function (x) { return x.saved; });
      write(HK, keep); emit();
      return keep.length;
    },
    clearAll: function () { write(HK, []); emit(); }
  };

  var learned = {
    list: function () {
      var valid = {};
      WORDS.forEach(function (w) { valid['w:' + w.id] = true; });
      return read('asl_learned', []).filter(function (id) { return /^[A-Z]$/.test(id) || valid[id]; });
    },
    has: function (id) { return read('asl_learned', []).indexOf(id) !== -1; },
    mark: function (id) {
      var l = read('asl_learned', []);
      if (l.indexOf(id) === -1) { l.push(id); write('asl_learned', l); emit(); }
    },
    total: function () { return 26 + WORDS.length; }
  };

  var quiz = {
    results: function () { return read('asl_quiz_results', []); },
    add: function (score, total) {
      var l = read('asl_quiz_results', []);
      l.unshift({ score: score, total: total, t: Date.now() });
      write('asl_quiz_results', l.slice(0, 30)); emit();
    },
    best: function () {
      var l = read('asl_quiz_results', []);
      if (!l.length) { return null; }
      return l.reduce(function (a, b) { return (b.score / b.total) > (a.score / a.total) ? b : a; });
    }
  };

  function getSession() { return read('asl_session', null); }
  function setSession(s) { write('asl_session', s); }
  function logout() {
    try { localStorage.removeItem('asl_session'); } catch (e) {}
    location.href = 'login.html';
  }
  function getProfile() {
    var s = getSession() || {};
    var p = read('asl_profile', {});
    return {
      name: p.name || s.name || 'Guest',
      email: p.email || s.email || '',
      level: p.level || 'Beginner',
      camera: p.camera || 'user'
    };
  }

  function hash(str) {
    if (window.crypto && crypto.subtle && window.TextEncoder) {
      return crypto.subtle.digest('SHA-256', new TextEncoder().encode(str)).then(function (buf) {
        return Array.prototype.map.call(new Uint8Array(buf), function (b) { return ('0' + b.toString(16)).slice(-2); }).join('');
      });
    }
    return Promise.resolve(btoa(unescape(encodeURIComponent(str))));
  }

  function imgBox(src, label, alt) {
    var box = document.createElement('div');
    box.className = 'imgbox';
    var img = document.createElement('img');
    img.alt = alt || label;
    img.loading = 'lazy';
    img.src = src;
    img.onerror = function () {
      box.innerHTML = '<div class="fb' + (String(label).length > 4 ? ' long' : '') + '">' + esc(label) + '<small>add image</small></div>';
    };
    box.appendChild(img);
    return box;
  }

  var toastEl = null, toastTimer = null;
  function toast(msg) {
    if (!toastEl) {
      toastEl = document.createElement('div');
      toastEl.className = 'toast';
      toastEl.setAttribute('role', 'status');
      document.body.appendChild(toastEl);
    }
    toastEl.innerHTML = icon('check', 20) + '<span>' + esc(msg) + '</span>';
    toastEl.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove('show'); }, 2600);
  }


  var KIND_ICON = { 'Sign to Text': 'camera', 'Text to Sign': 'type', 'Upload Media': 'upload' };

  function fmtTime(t) {
    return new Date(t).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function historyItemHtml(h) {
    return '<div class="hitem' + (h.saved ? ' saved' : '') + '">' +
      '<div class="kind">' + icon(KIND_ICON[h.kind] || 'chat', 22) + '</div>' +
      '<div class="body"><b>' + esc(h.text) + '</b><small>' + esc(h.kind) + ', ' + fmtTime(h.t) + '</small></div>' +
      '<div class="acts">' +
      '<button class="icon-btn" type="button" data-act="save" data-id="' + h.id + '" aria-pressed="' + (h.saved ? 'true' : 'false') + '" aria-label="' + (h.saved ? 'Unsave' : 'Save') + ' ' + esc(h.text) + '">' + icon('bookmark', 20) + '</button>' +
      '<button class="icon-btn" type="button" data-act="del" data-id="' + h.id + '" aria-label="Delete ' + esc(h.text) + '">' + icon('trash', 20) + '</button>' +
      '</div></div>';
  }

  function bindHistoryActions(listEl) {
    listEl.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-act]');
      if (!b) { return; }
      if (b.dataset.act === 'save') { history.toggleSave(b.dataset.id); }
      if (b.dataset.act === 'del') { history.remove(b.dataset.id); toast('Record deleted'); }
    });
  }

  function getTheme() {
    var t = read('asl_theme', null);
    if (t === 'light' || t === 'dark') { return t; }
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
  }
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    document.querySelectorAll('.theme-toggle').forEach(function (b) {
      var dark = t === 'dark';
      b.innerHTML = icon(dark ? 'sun' : 'moon', 22);
      b.setAttribute('aria-pressed', dark ? 'true' : 'false');
      b.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
      b.title = dark ? 'Light mode' : 'Dark mode';
    });
  }
  function toggleTheme() {
    var next = getTheme() === 'dark' ? 'light' : 'dark';
    write('asl_theme', next);
    document.documentElement.classList.add('theme-anim');
    applyTheme(next);
    setTimeout(function () { document.documentElement.classList.remove('theme-anim'); }, 400);
  }
  function themeButton(extra) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'icon-btn theme-toggle' + (extra ? ' ' + extra : '');
    b.addEventListener('click', toggleTheme);
    return b;
  }

  var NAV = [
    { key: 'home', href: 'home.html', icon: 'home', label: 'Home' },
    { key: 'translate', href: 'translation.html', icon: 'chat', label: 'Translation' },
    { key: 'learning', href: 'learning.html', icon: 'book', label: 'Learning' },
    { key: 'history', href: 'history.html', icon: 'clock', label: 'History' },
    { key: 'feedback', href: 'feedback.html', icon: 'star', label: 'Feedback' },
    { key: 'profile', href: 'profile.html', icon: 'user', label: 'Profile' }
  ];

  function buildShell(page) {
    var top = document.getElementById('topbar');
    if (!top) { return; }
    var p = getProfile();
    var initial = (p.name || 'G').trim().charAt(0).toUpperCase();

    var links = NAV.map(function (n) {
      return '<a class="nav-item" href="' + n.href + '"' + (n.key === page ? ' aria-current="page"' : '') + '>' +
        icon(n.icon, 20) + '<span>' + n.label + '</span></a>';
    }).join('');

    top.innerHTML =
      '<div class="topbar-in">' +
      '<a class="brand" href="home.html"><span class="brand-mark">' + icon('hand', 24) + '</span>' +
      '<span class="brand-name">ASL Translator</span></a>' +
      '<nav class="topnav" id="topnav" aria-label="Main menu">' + links + '</nav>' +
      '<div class="top-actions">' +
      '<button class="icon-btn theme-toggle" id="theme-toggle" type="button"></button>' +
      '<button class="icon-btn" id="fx-toggle" type="button" aria-pressed="true" aria-label="Cursor effects" title="Cursor effects">' + icon('sparkle', 22) + '</button>' +
      '<a class="avatar" href="profile.html" id="top-avatar" aria-label="Profile: ' + esc(p.name) + '" title="' + esc(p.name) + '">' + esc(initial) + '</a>' +
      '<button class="icon-btn" id="logout-btn" type="button" aria-label="Log out" title="Log out">' + icon('logout', 22) + '</button>' +
      '<button class="icon-btn menu-btn" id="menu-btn" type="button" aria-label="Open menu" aria-expanded="false">' + icon('menu', 22) + '</button>' +
      '</div></div>';

    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    var nav = document.getElementById('topnav');
    var menuBtn = document.getElementById('menu-btn');
    function setMenu(open) {
      nav.classList.toggle('open', open);
      menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    menuBtn.addEventListener('click', function () { setMenu(!nav.classList.contains('open')); });
    nav.addEventListener('click', function (e) { if (e.target.closest('a.nav-item')) { setMenu(false); } });
    document.addEventListener('click', function (e) { if (!top.contains(e.target)) { setMenu(false); } });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { setMenu(false); } });
    document.getElementById('logout-btn').addEventListener('click', logout);

    window.addEventListener('scroll', function () {
      top.classList.toggle('scrolled', window.scrollY > 8);
    }, { passive: true });
  }

  function initFx() {
    var btn = document.getElementById('fx-toggle');
    var fine = window.matchMedia('(pointer: fine)').matches;
    var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!fine || reduce) {
      if (btn) { btn.hidden = true; }
      document.body.classList.add('fx-off');
      return;
    }

    var enabled = read('asl_fx', true) !== false;
    var ring = document.createElement('div'); ring.className = 'fx-ring';
    var dot = document.createElement('div'); dot.className = 'fx-dot';
    document.body.appendChild(ring); document.body.appendChild(dot);

    var x = -100, y = -100, rx = -100, ry = -100, lastTrail = 0, raf = null;
    var HOVER = 'a,button,input,textarea,select,label,[role="tab"],[role="radio"],[role="switch"],.atile,.drop';

    function loop() {
      rx += (x - rx) * 0.18;
      ry += (y - ry) * 0.18;
      ring.style.transform = 'translate(' + rx + 'px,' + ry + 'px)';
      dot.style.transform = 'translate(' + x + 'px,' + y + 'px)';
      raf = requestAnimationFrame(loop);
    }

    function apply() {
      document.body.classList.toggle('fx-off', !enabled);
      if (btn) { btn.setAttribute('aria-pressed', enabled ? 'true' : 'false'); }
      if (enabled && !raf) { loop(); }
      if (!enabled && raf) { cancelAnimationFrame(raf); raf = null; }
    }

    document.addEventListener('mousemove', function (e) {
      if (!enabled) { return; }
      x = e.clientX; y = e.clientY;
      ring.classList.add('on'); dot.classList.add('on');

      var now = performance.now();
      if (now - lastTrail > 45) {
        lastTrail = now;
        var t = document.createElement('span');
        t.className = 'fx-trail';
        t.style.left = x + 'px'; t.style.top = y + 'px';
        t.addEventListener('animationend', function () { t.remove(); });
        document.body.appendChild(t);
      }

      var card = e.target.closest ? e.target.closest('.card') : null;
      if (card) {
        var r = card.getBoundingClientRect();
        card.style.setProperty('--mx', (e.clientX - r.left) + 'px');
        card.style.setProperty('--my', (e.clientY - r.top) + 'px');
      }
    });

    document.addEventListener('mouseover', function (e) {
      ring.classList.toggle('hover', !!(e.target.closest && e.target.closest(HOVER)));
    });
    document.addEventListener('mousedown', function () { ring.classList.add('down'); });
    document.addEventListener('mouseup', function () { ring.classList.remove('down'); });
    document.addEventListener('mouseleave', function () { ring.classList.remove('on'); dot.classList.remove('on'); });
    document.addEventListener('click', function (e) {
      if (!enabled) { return; }
      var r = document.createElement('span');
      r.className = 'fx-ripple';
      r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
      r.addEventListener('animationend', function () { r.remove(); });
      document.body.appendChild(r);
    });

    if (btn) {
      btn.addEventListener('click', function () {
        enabled = !enabled;
        write('asl_fx', enabled);
        apply();
        toast(enabled ? 'Cursor effects on' : 'Cursor effects off');
      });
    }
    apply();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var page = document.body.dataset.page || '';
    if (page !== 'login' && !getSession()) {
      location.replace('login.html');
      return;
    }
    if (page !== 'login') {
      var skip = document.createElement('a');
      skip.className = 'skip'; skip.href = '#main'; skip.textContent = 'Skip to content';
      document.body.insertBefore(skip, document.body.firstChild);
      buildShell(page);

      var foot = document.createElement('footer');
      foot.className = 'footer';
      foot.innerHTML = '<span><b>ASL Translator</b> &middot; Sign language, made visual</span>' +
        '<span>Accessible by design. Your data stays in this browser.</span>';
      var wrap = document.querySelector('.content');
      if (wrap) { wrap.appendChild(foot); }
    } else {
      document.body.appendChild(themeButton('theme-fab'));
    }
    applyTheme(getTheme());
    paintIcons();
    initFx();
    document.dispatchEvent(new CustomEvent('asl:ready'));
  });

  window.ASL = {
    icon: icon, paintIcons: paintIcons, esc: esc, read: read, write: write,
    history: history, learned: learned, quiz: quiz,
    LETTER_NOTES: LETTER_NOTES, WORDS: WORDS,
    getSession: getSession, setSession: setSession, getProfile: getProfile, logout: logout, hash: hash,
    imgBox: imgBox, toast: toast, getTheme: getTheme, toggleTheme: toggleTheme,
    historyItemHtml: historyItemHtml, bindHistoryActions: bindHistoryActions
  };
})();
