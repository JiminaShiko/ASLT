<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>History - ASL Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../css/style.css">
</head>
<body data-page="history" data-title="History">
<div class="app">
  <aside class="sidebar" id="sidebar"></aside>
  <div class="scrim" id="scrim"></div>
  <div class="content">
    <header class="topbar" id="topbar"></header>

    <main class="page" id="main">

      <section class="card" aria-labelledby="h-h">
        <div class="card-head">
          <div>
            <h2 id="h-h">All history</h2>
            <p class="muted" id="h-count" style="margin-top:4px"></p>
          </div>
          <button class="btn btn-ghost btn-sm" type="button" id="clear-history"><span data-icon="trash" data-size="20"></span>Clear unsaved</button>
        </div>
        <div class="hlist hlist-all" id="hlist" style="margin-top:16px"></div>
      </section>

    </main>
  </div>
</div>
<script src="../js/common.js"></script>
<script src="../js/history.js"></script>
</body>
</html>
