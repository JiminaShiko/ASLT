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
      <h2 id="t-h">Translate</h2>
      <div class="seg" role="tablist" aria-label="Translation mode" style="margin-top:16px">
        <button type="button" role="tab" id="tab-sign" aria-selected="true" aria-controls="panel-sign"><span data-icon="camera"></span>Sign to Text</button>
        <button type="button" role="tab" id="tab-text" aria-selected="false" aria-controls="panel-text"><span data-icon="type"></span>Text to Sign</button>
        <button type="button" role="tab" id="tab-upload" aria-selected="false" aria-controls="panel-upload"><span data-icon="upload"></span>Upload Media</button>
      </div>

      <!-- Sign to Text -->
      <div class="panel" id="panel-sign" role="tabpanel" aria-labelledby="tab-sign">
        <div class="cam" id="cam-box">
          <video id="cam" autoplay playsinline muted hidden></video>
          <img class="snap" id="snap" alt="Captured sign" hidden>
          <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
          <div class="ph" id="cam-ph">
            <span data-icon="hand" data-size="64"></span>
            <b>Camera is off</b>
            <span>Tap "Start camera", then show your sign inside the frame.</span>
          </div>
          <button class="icon-btn flip" id="flip" type="button" aria-label="Flip camera" disabled><span data-icon="flip"></span></button>
        </div>
        <div class="row">
          <button class="btn btn-ghost" type="button" id="cam-toggle"><span data-icon="camera"></span><span id="cam-toggle-label">Start camera</span></button>
          <button class="btn" type="button" id="capture" disabled><span data-icon="check"></span>Capture sign</button>
        </div>
        <div class="status" id="cam-status" role="status" aria-live="polite"></div>
        <div class="result" id="sign-result" aria-live="polite"></div>
      </div>

      <!-- Text to Sign -->
      <div class="panel" id="panel-text" role="tabpanel" aria-labelledby="tab-text" hidden>
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

      <!-- Upload Media -->
      <div class="panel" id="panel-upload" role="tabpanel" aria-labelledby="tab-upload" hidden>
        <div class="drop" id="drop">
          <div class="icon-lg"><span data-icon="upload" data-size="34"></span></div>
          <div class="big">Upload a photo or video of a sign</div>
          <span class="muted">Drag a file here, or choose one below.</span>
          <div class="row" style="justify-content:center">
            <button class="btn" type="button" id="pick-photo" style="flex:0 1 auto"><span data-icon="image"></span>Photo</button>
            <button class="btn btn-ghost" type="button" id="pick-video" style="flex:0 1 auto"><span data-icon="video"></span>Video</button>
          </div>
          <input type="file" id="file-photo" accept="image/*" hidden>
          <input type="file" id="file-video" accept="video/*" hidden>
        </div>
        <div class="preview" id="preview" hidden></div>
        <div class="status" id="upload-status" role="status" aria-live="polite"></div>
        <button class="btn" type="button" id="translate-upload" disabled><span data-icon="sparkle"></span>Translate</button>
        <div class="result" id="upload-result" aria-live="polite"></div>
      </div>
    </section>

    </main>
  </div>
</div>
<script src="../js/common.js"></script>
<script src="../js/translation.js"></script>
</body>
</html>
