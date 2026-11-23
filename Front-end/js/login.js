document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var mode = 'login';

  if (A.getSession() && location.search.indexOf('switch') === -1) {
    location.replace('home.html');
    return;
  }

  var tabLogin = document.getElementById('tab-login');
  var tabSignup = document.getElementById('tab-signup');
  var nameField = document.getElementById('name-field');
  var title = document.getElementById('form-title');
  var submit = document.getElementById('submit');
  var err = document.getElementById('err');
  var pass = document.getElementById('password');

  function setMode(m) {
    mode = m;
    var signup = (m === 'signup');
    tabLogin.setAttribute('aria-selected', signup ? 'false' : 'true');
    tabSignup.setAttribute('aria-selected', signup ? 'true' : 'false');
    nameField.hidden = !signup;
    title.textContent = signup ? 'Create your account' : 'Welcome back';
    submit.textContent = signup ? 'Create account' : 'Log in';
    pass.autocomplete = signup ? 'new-password' : 'current-password';
    err.textContent = '';
  }
  tabLogin.addEventListener('click', function () { setMode('login'); });
  tabSignup.addEventListener('click', function () { setMode('signup'); });

  document.getElementById('guest').addEventListener('click', function () {
    A.setSession({ name: 'Guest', email: '', guest: true });
    location.href = 'home.html';
  });

  document.getElementById('auth-form').addEventListener('submit', function (e) {
    e.preventDefault();
    err.textContent = '';
    var email = document.getElementById('email').value.trim().toLowerCase();
    var password = pass.value;
    var name = document.getElementById('name').value.trim();

    if (email.indexOf('@') < 1 || email.indexOf('.') < 0) { err.textContent = 'Enter a valid email address.'; return; }
    if (password.length < 6) { err.textContent = 'Password needs at least 6 characters.'; return; }
    if (mode === 'signup' && !name) { err.textContent = 'Enter your name.'; return; }

    var users = A.read('asl_users', {});
    A.hash(password).then(function (h) {
      if (mode === 'signup') {
        if (users[email]) { err.textContent = 'This email already has an account. Log in instead.'; return; }
        users[email] = { name: name, hash: h };
        A.write('asl_users', users);
        A.setSession({ name: name, email: email });
        location.href = 'home.html';
      } else {
        if (!users[email] || users[email].hash !== h) { err.textContent = 'Email or password is not correct.'; return; }
        A.setSession({ name: users[email].name, email: email });
        location.href = 'home.html';
      }
    });
  });
});
