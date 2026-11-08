document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var nameEl = document.getElementById('name');
  var emailEl = document.getElementById('email');
  var levelEl = document.getElementById('level');
  var cameraEl = document.getElementById('camera');
  var err = document.getElementById('err');

  function fill() {
    var p = A.getProfile();
    nameEl.value = p.name;
    emailEl.value = p.email;
    levelEl.value = p.level;
    cameraEl.value = p.camera;
    document.getElementById('avatar-lg').textContent = (p.name || 'G').trim().charAt(0).toUpperCase();
    var s = A.getSession() || {};
    document.getElementById('pf-sub').textContent = s.guest ? 'You are using a guest account.' : (p.email || '');
  }
  fill();

  document.getElementById('pf-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var name = nameEl.value.trim();
    var email = emailEl.value.trim();
    if (!name) { err.textContent = 'Enter your name.'; return; }
    if (email && (email.indexOf('@') < 1 || email.indexOf('.') < 0)) { err.textContent = 'Enter a valid email address.'; return; }
    err.textContent = '';

    A.write('asl_profile', { name: name, email: email, level: levelEl.value, camera: cameraEl.value });
    var s = A.getSession() || {};
    s.name = name; s.email = email;
    A.setSession(s);

    // Update the top bar avatar without reloading
    var av = document.getElementById('top-avatar');
    if (av) { av.textContent = name.charAt(0).toUpperCase(); av.title = name; }
    fill();
    A.toast('Changes saved');
  });

  /* Cursor effects switch (uses the top bar toggle when it exists) */
  var sw = document.getElementById('fx-switch');
  function paintSwitch() { sw.setAttribute('aria-checked', A.read('asl_fx', true) !== false ? 'true' : 'false'); }
  paintSwitch();
  sw.addEventListener('click', function () {
    var topBtn = document.getElementById('fx-toggle');
    if (topBtn && !topBtn.hidden) {
      topBtn.click();
    } else {
      A.write('asl_fx', A.read('asl_fx', true) === false);
      A.toast('Saved');
    }
    paintSwitch();
  });

  document.getElementById('clear-all').addEventListener('click', function () {
    if (!window.confirm('Delete your translation history and learning progress?')) { return; }
    ['asl_history', 'asl_learned', 'asl_quiz_results', 'asl_feedback'].forEach(function (k) {
      try { localStorage.removeItem(k); } catch (e) {}
    });
    A.toast('History and progress deleted');
  });

  document.getElementById('logout').addEventListener('click', A.logout);
});
