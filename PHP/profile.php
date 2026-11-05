<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Profile Settings - ASL Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/style.css">
</head>
<body data-page="profile" data-title="Profile Settings">
<div class="app">
  <aside class="sidebar" id="sidebar"></aside>
  <div class="scrim" id="scrim"></div>
  <div class="content">
    <header class="topbar" id="topbar"></header>

    <main class="page" id="main">
      <div class="twocol">

        <section class="card" aria-labelledby="pf-h">
          <div class="profile-top">
            <div class="avatar-lg" id="avatar-lg" aria-hidden="true">G</div>
            <div>
              <h2 id="pf-h">Your profile</h2>
              <p class="muted" id="pf-sub"></p>
            </div>
          </div>

          <form id="pf-form" novalidate style="display:flex;flex-direction:column;gap:18px;margin-top:24px">
            <div class="formgrid">
              <div class="field">
                <label for="name">Full name</label>
                <input class="input" id="name" type="text" autocomplete="name">
              </div>
              <div class="field">
                <label for="email">Email</label>
                <input class="input" id="email" type="email" autocomplete="email">
              </div>
              <div class="field">
                <label for="level">Learning level</label>
                <select class="select" id="level">
                  <option>Beginner</option><option>Intermediate</option><option>Advanced</option>
                </select>
              </div>
              <div class="field">
                <label for="camera">Default camera</label>
                <select class="select" id="camera">
                  <option value="user">Front camera</option>
                  <option value="environment">Back camera</option>
                </select>
              </div>
            </div>
            <div class="err" id="err" role="alert"></div>
            <div><button class="btn" type="submit"><span data-icon="check"></span>Save changes</button></div>
          </form>
        </section>

        <div style="display:flex;flex-direction:column;gap:24px">
          <section class="card" aria-labelledby="pr-h">
            <h2 id="pr-h">Preferences</h2>
            <div class="toggle" style="margin-top:16px">
              <div><b>Cursor effects</b><div class="muted" style="font-size:16px">Glow, trail and click ripple</div></div>
              <button class="switch" id="fx-switch" type="button" role="switch" aria-checked="true" aria-label="Cursor effects"></button>
            </div>
          </section>

          <section class="card danger" aria-labelledby="dz-h">
            <h2 id="dz-h">Your data</h2>
            <p class="muted" style="margin-top:6px">Everything is saved in this browser only.</p>
            <div style="display:flex;flex-direction:column;gap:12px;margin-top:16px">
              <button class="btn btn-danger" type="button" id="clear-all"><span data-icon="trash"></span>Delete history and progress</button>
              <button class="btn btn-ghost" type="button" id="logout"><span data-icon="logout"></span>Log out</button>
            </div>
          </section>
        </div>

      </div>
    </main>
  </div>
</div>
<script src="js/common.js"></script>
<script src="js/profile.js"></script>
</body>
</html>
