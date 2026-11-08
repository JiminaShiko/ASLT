<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Translation - ASL Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../css/style.css">
</head>
<body data-page="translate" data-title="Translation">
<div class="app">
  <aside class="sidebar" id="sidebar"></aside>
  <div class="scrim" id="scrim"></div>
  <div class="content">
    <header class="topbar" id="topbar"></header>

    <main class="page" id="main">

    <section class="card" id="translate" aria-labelledby="t-h">
      <h2 id="t-h">Text to Sign</h2>
      <div class="panel" id="panel-text">
        <div class="input-row">
          <div class="field">
            <label for="txt">Type your text</label>
            <input class="input" id="txt" type="text" placeholder="For example: hello" autocomplete="off">
          </div>
          <button class="btn btn-ghost mic" type="button" id="voice"><span data-icon="mic"></span><span id="voice-label">Speak</span></button>
        </div>
        <div class="status" id="text-status" role="status" aria-live="polite"></div>
        <button class="btn" type="button" id="show-sign"><span data-icon="hand"></span>Show signs</button>
        <div class="result" style="flex-direction:column;gap:12px">
          <div class="label">Sign images</div>
          <div class="tiles" id="signs"><p class="muted">Type a word, then tap "Show signs".</p></div>
        </div>
      </div>

    </section>

    </main>
  </div>
</div>
<script src="../js/common.js"></script>
<script src="../js/translation.js"></script>
</body>
</html>
