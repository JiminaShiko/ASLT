<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Home - ASL Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../css/style.css">
</head>
<body data-page="home" data-title="Home">
<div class="app">
  <aside class="sidebar" id="sidebar"></aside>
  <div class="scrim" id="scrim"></div>
  <div class="content">
    <header class="topbar" id="topbar"></header>

    <main class="page" id="main">
      <section class="card hero" aria-label="Welcome">
        <div>
          <h1>Hello, <span id="user-name">there</span>!</h1>
          <p>Translate a sign, turn words into signs, or practise a few letters today.</p>
          <div class="hero-actions">
            <a class="btn" href="translation.php"><span data-icon="chat"></span>Start translating</a>
            <a class="btn btn-ghost" href="learning.php"><span data-icon="book"></span>Open Learning</a>
          </div>
        </div>
        <div class="signing" aria-hidden="true">
          <div class="t" style="--i:0"><b>H</b></div>
          <div class="t" style="--i:1"><b>E</b></div>
          <div class="t" style="--i:2"><b>L</b></div>
          <div class="t" style="--i:3"><b>L</b></div>
          <div class="t" style="--i:4"><b>O</b></div>
        </div>
      </section>

      <section class="card" id="history" aria-labelledby="h-h">
        <div class="card-head">
          <div>
            <h2 id="h-h">Recent history</h2>
            <p class="muted" style="margin-top:4px">Your translations from the last 24 hours.</p>
          </div>
          <a class="btn btn-ghost btn-sm" href="history.php"><span data-icon="clock" data-size="20"></span>View all history</a>
        </div>
        <div class="hlist" id="hlist" style="margin-top:16px"></div>
      </section>

      <section class="card strip" aria-label="Your progress">
        <div><b id="st-trans">0</b><span>Translations</span></div>
        <div><b id="st-learn">0 / 31</b><span>Signs learned</span></div>
        <div><b id="st-quiz">None yet</b><span>Best quiz score</span></div>
      </section>
    </main>
  </div>
</div>
<script src="../js/common.js"></script>
<script src="../js/home.js"></script>
</body>
</html>
