<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feedback - ASL Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/style.css">
</head>
<body data-page="feedback" data-title="Feedback">
<div class="app">
  <aside class="sidebar" id="sidebar"></aside>
  <div class="scrim" id="scrim"></div>
  <div class="content">
    <header class="topbar" id="topbar"></header>

    <main class="page" id="main">
      <div class="twocol">

        <section class="card" aria-labelledby="f-h">
          <div id="form-view">
            <h2 id="f-h">Tell us what you think</h2>
            <p class="muted" style="margin-top:6px">Your feedback helps us improve translations and lessons.</p>

            <form id="fb-form" novalidate style="display:flex;flex-direction:column;gap:22px;margin-top:22px">
              <div>
                <div class="muted" id="rate-l" style="font-weight:700;font-size:16px;margin-bottom:8px">How was your experience?</div>
                <div class="faces" role="radiogroup" aria-labelledby="rate-l" id="faces"></div>
              </div>
              <div>
                <div class="muted" id="cat-l" style="font-weight:700;font-size:16px;margin-bottom:8px">What is it about?</div>
                <div class="chips" role="radiogroup" aria-labelledby="cat-l" id="chips"></div>
              </div>
              <div class="field">
                <label for="msg">Your message</label>
                <textarea class="textarea" id="msg" placeholder="Write what worked well and what could be better."></textarea>
              </div>
              <div class="err" id="err" role="alert"></div>
              <div><button class="btn" type="submit"><span data-icon="send"></span>Send feedback</button></div>
            </form>
          </div>

          <div class="success" id="done-view" hidden>
            <div class="tick"><span data-icon="check" data-size="44"></span></div>
            <h2>Feedback sent</h2>
            <p class="muted">Thank you. We read every message.</p>
            <button class="btn btn-ghost" type="button" id="again">Write more feedback</button>
          </div>
        </section>

        <aside class="card" aria-labelledby="p-h">
          <h2 id="p-h">Your previous feedback</h2>
          <div id="flist" style="display:flex;flex-direction:column;gap:10px;margin-top:16px"></div>
        </aside>

      </div>
    </main>
  </div>
</div>
<script src="js/common.js"></script>
<script src="js/feedback.js"></script>
</body>
</html>
