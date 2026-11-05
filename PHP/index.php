<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>User Home Page - ASL Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{--ink:#0B2E4A;--muted:#3F5F7A;--primary:#0B72C7;--sky:#E6F4FE;--sky-soft:#F0F8FF;--border:#B9DFFB;--border-soft:#D3EBFC}
*{box-sizing:border-box}
body{margin:0;font-family:'Atkinson Hyperlegible',system-ui,sans-serif;color:var(--ink);background:#fff}
a{text-decoration:none;color:var(--primary)}
button,input{font-family:inherit}
.layout{display:flex;flex-wrap:wrap;min-height:100vh}
.sidebar{flex:1 0 260px;max-width:100%;padding:28px 20px;background:var(--sky-soft);border-right:2px solid var(--border-soft);display:flex;flex-direction:column;gap:8px}
.brand{display:flex;align-items:center;gap:12px;padding:0 8px 24px}
.brand-logo{width:48px;height:48px;border-radius:14px;background:var(--primary);display:flex;align-items:center;justify-content:center}
.brand-name{font-size:22px;font-weight:700;line-height:1.1}
.nav a{display:flex;align-items:center;gap:14px;min-height:52px;padding:0 16px;border-radius:14px;color:var(--ink);font-size:18px}
.nav a.active{background:var(--primary);color:#fff;font-weight:700}
.nav a:hover:not(.active){background:var(--sky)}
.spacer{flex-grow:1;min-height:24px}
.logout{border:2px solid var(--border);background:#fff}
.main{flex:999 1 640px;min-width:0;padding:32px 40px 48px;display:flex;flex-direction:column;gap:32px}
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.top h1{margin:0;font-size:36px}
.top p{margin:6px 0 0;font-size:20px;color:var(--muted)}
.avatar{width:56px;height:56px;border-radius:50%;background:var(--sky);border:2px solid var(--border);display:flex;align-items:center;justify-content:center}
.row{display:flex;flex-wrap:wrap;gap:28px;align-items:stretch}
.card{border:2px solid var(--border);border-radius:24px;padding:24px;background:#fff;display:flex;flex-direction:column;gap:16px}
.translate{flex:999 1 520px;min-width:0;gap:20px}
.history{flex:1 1 280px;min-width:0}
.tabs{display:flex;flex-wrap:wrap;gap:12px}
.tab{flex:1 1 150px;min-height:56px;border-radius:16px;border:2px solid var(--primary);font-size:18px;font-weight:700;display:flex;align-items:center;justify-content:center;gap:10px;background:#fff;color:var(--primary);cursor:pointer}
.tab[aria-selected="true"]{background:var(--primary);color:#fff}
.panel[hidden]{display:none}
.panel{display:flex;flex-direction:column;gap:16px}
.camera{position:relative;width:100%;aspect-ratio:16/9;max-height:380px;border-radius:20px;border:2px solid var(--border);background:var(--sky);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;font-size:20px;color:var(--muted)}
.flip{position:absolute;top:14px;right:14px;width:52px;height:52px;border-radius:50%;border:2px solid var(--border);background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center}
.btn{min-height:60px;border-radius:16px;border:none;background:var(--primary);color:#fff;font-size:20px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0 28px}
.btn-outline{background:#fff;color:var(--primary);border:2px solid var(--primary)}
.result{border-radius:16px;background:var(--sky-soft);border:2px solid var(--border-soft);padding:18px 20px;display:flex;flex-direction:column;gap:12px}
.label{font-size:16px;font-weight:700;color:var(--muted)}
.result-text{font-size:26px;font-weight:700;min-height:34px}
.field{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end}
.field > div{flex:1 1 260px;display:flex;flex-direction:column;gap:6px}
.input{min-height:60px;width:100%;border-radius:16px;border:2px solid var(--border);padding:0 18px;font-size:20px;color:var(--ink)}
.voice{min-height:60px;border-radius:16px;border:2px solid var(--primary);background:#fff;color:var(--primary);font-size:18px;font-weight:700;padding:0 18px;display:flex;align-items:center;gap:8px;cursor:pointer}
.signs{display:flex;flex-wrap:wrap;gap:12px}
.sign{width:84px;display:flex;flex-direction:column;align-items:center;gap:6px;font-size:20px;font-weight:700}
.sign-img{width:84px;height:84px;border-radius:14px;border:2px dashed #7CC4F8;background:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:400;color:var(--muted)}
.drop{border-radius:20px;border:2px dashed #7CC4F8;background:var(--sky);padding:40px 20px;display:flex;flex-direction:column;align-items:center;gap:16px;text-align:center;font-size:22px;font-weight:700}
.drop .btns{display:flex;flex-wrap:wrap;gap:12px;justify-content:center}
.history-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.history-head h2{margin:0;font-size:24px}
.small-btn{min-height:44px;padding:0 16px;border-radius:12px;border:2px solid var(--border);background:#fff;color:var(--ink);font-size:16px;cursor:pointer}
.history-item{border-radius:14px;background:var(--sky-soft);border:2px solid var(--border-soft);padding:12px 16px}
.history-item small{display:block;font-size:14px;color:var(--muted)}
.history-item strong{font-size:20px}
.history-list{display:flex;flex-direction:column;gap:10px}
.learning h2{margin:0 0 16px;font-size:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
.card h3{margin:0;font-size:22px}
.tiles{display:flex;flex-wrap:wrap;gap:10px}
.tile{min-width:64px;height:76px;padding:0 16px;border-radius:14px;background:var(--sky);border:2px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:700}
.link-btn{min-height:52px;border-radius:14px;background:var(--primary);color:#fff;font-size:18px;font-weight:700;display:flex;align-items:center;justify-content:center;margin-top:auto}
.score{display:flex;align-items:center;gap:16px;font-size:18px;color:var(--muted)}
.score-ring{width:84px;height:84px;border-radius:50%;border:6px solid var(--primary);background:var(--sky);display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:700;color:var(--ink)}
.feedback{border-radius:24px;background:var(--sky);border:2px solid var(--border);padding:24px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:16px}
.feedback h2{margin:0;font-size:24px}
.feedback p{margin:6px 0 0;font-size:18px;color:var(--muted)}
.feedback .btn{background:#fff;color:var(--primary);border:2px solid var(--primary);font-size:18px;min-height:56px}

.camera{overflow:hidden}
.camera video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.camera .flip{z-index:2}
.row-btns{display:flex;flex-wrap:wrap;gap:12px}
.row-btns .btn{flex:1 1 200px}
.btn[disabled]{opacity:.45;cursor:not-allowed}
.status{font-size:18px;color:var(--muted);min-height:26px}
.status.error{color:#9A2A00;font-weight:700}
.sign-img img{width:100%;height:100%;object-fit:cover;border-radius:12px}
.sign-gap{width:40px}
.preview{border-radius:20px;border:2px solid var(--border);background:var(--sky);overflow:hidden;display:flex;align-items:center;justify-content:center;max-height:380px}
.preview img,.preview video{max-width:100%;max-height:380px;display:block}
.voice.listening{background:var(--primary);color:#fff}
.history-empty{font-size:18px;color:var(--muted)}
a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid #0B2E4A;outline-offset:2px}
</style>
</head>
<body>
<div class="layout">

  <nav class="sidebar" aria-label="Main menu">
    <div class="brand">
      <div class="brand-logo">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 13V5a1.5 1.5 0 0 1 3 0v6"/><path d="M11 11V3.5a1.5 1.5 0 0 1 3 0V11"/><path d="M14 11V5.5a1.5 1.5 0 0 1 3 0V14"/><path d="M17 9.5a1.5 1.5 0 0 1 3 0V15a7 7 0 0 1-7 7h-1a7 7 0 0 1-5.5-2.7L3 14.5a1.5 1.5 0 0 1 2.3-1.9L8 15"/></svg>
      </div>
      <div class="brand-name">ASL<br>Translator</div>
    </div>
    <div class="nav">
      <a class="active" href="home.php">Home</a>
      <a href="#translate">Translation</a>
      <a href="#learning">Learning</a>
      <a href="#history">History</a>
      <a href="#feedback">Feedback</a>
      <a href="profile.php">Profile Settings</a>
    </div>
    <div class="spacer"></div>
    <div class="nav"><a class="logout" href="login.php">Logout</a></div>
  </nav>

  <main class="main">
    <header class="top">
      <div>
        <h1>Hello!</h1>
        <p>What would you like to do today?</p>
      </div>
      <a class="avatar" href="profile.php" aria-label="Profile">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0B2E4A" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>
      </a>
    </header>

    <div class="row" id="translate">
      <section class="card translate" aria-label="Translation">
        <div class="tabs" role="tablist">
          <button class="tab" role="tab" id="tab-sign" aria-selected="true" aria-controls="panel-sign">Sign to Text</button>
          <button class="tab" role="tab" id="tab-text" aria-selected="false" aria-controls="panel-text">Text to Sign</button>
          <button class="tab" role="tab" id="tab-upload" aria-selected="false" aria-controls="panel-upload">Upload Media</button>
        </div>

        <!-- Sign to Text -->
        <div class="panel" id="panel-sign" role="tabpanel" aria-labelledby="tab-sign">
          <div class="camera" id="camera">
            <video id="cam" autoplay playsinline muted hidden></video>
            <div id="cam-placeholder" style="display:flex;flex-direction:column;align-items:center;gap:12px">
              <svg width="88" height="88" viewBox="0 0 24 24" fill="none" stroke="#0B72C7" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 13V5a1.5 1.5 0 0 1 3 0v6"/><path d="M11 11V3.5a1.5 1.5 0 0 1 3 0V11"/><path d="M14 11V5.5a1.5 1.5 0 0 1 3 0V14"/><path d="M17 9.5a1.5 1.5 0 0 1 3 0V15a7 7 0 0 1-7 7h-1a7 7 0 0 1-5.5-2.7L3 14.5a1.5 1.5 0 0 1 2.3-1.9L8 15"/></svg>
              <div>Tap "Start Camera" to begin</div>
            </div>
            <button class="flip" id="flip" type="button" aria-label="Flip camera" disabled>
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#0B2E4A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14-4L4 9"/><path d="M4 4v5h5"/><path d="M4 13a8 8 0 0 0 14 4l2-2"/><path d="M20 20v-5h-5"/></svg>
            </button>
          </div>
          <div class="row-btns">
            <button class="btn btn-outline" type="button" id="cam-toggle">Start Camera</button>
            <button class="btn" type="button" id="capture" disabled>Start Live Translation</button>
          </div>
          <div class="field">
            <div>
              <label class="label" for="camera-model-choice">Model</label>
              <select class="input" id="camera-model-choice">
                <option value="cnn3d">3D CNN</option>
                <option value="cnn_gru">CNN-GRU</option>
              </select>
            </div>
          </div>
          <div class="status" id="cam-status" role="status" aria-live="polite"></div>
          <div class="result">
            <div class="label">Result</div>
            <div class="result-text" id="sign-result">Your text will appear here</div>
          </div>
        </div>

        <!-- Text to Sign -->
        <div class="panel" id="panel-text" role="tabpanel" aria-labelledby="tab-text" hidden>
          <div class="field">
            <div>
              <label class="label" for="txt">Type your text</label>
              <input class="input" id="txt" type="text" placeholder="Type a word or sentence" autocomplete="off">
            </div>
            <button class="voice" type="button" id="voice" aria-label="Speak with voice">Voice</button>
          </div>
          <div class="status" id="text-status" role="status" aria-live="polite"></div>
          <button class="btn" type="button" id="show-sign">Show Sign</button>
          <div class="result">
            <div class="label">Sign images</div>
            <div class="signs" id="signs">
              <div class="history-empty">Type something to see the signs</div>
            </div>
          </div>
        </div>

        <!-- Upload Media -->
        <div class="panel" id="panel-upload" role="tabpanel" aria-labelledby="tab-upload" hidden>
          <div class="drop" id="drop">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#0B72C7" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 20h16"/></svg>
            <div>Upload a photo or video of a sign</div>
            <div class="btns">
              <button class="btn" type="button" id="pick-photo" style="min-height:56px;font-size:18px">Photo</button>
              <button class="btn btn-outline" type="button" id="pick-video" style="min-height:56px;font-size:18px">Video</button>
            </div>
            <input type="file" id="file-photo" accept="image/*" hidden>
            <input type="file" id="file-video" accept="video/*" hidden>
          </div>
          <div class="preview" id="preview" hidden></div>
          <div class="field">
            <div>
              <label class="label" for="model-choice">Model</label>
              <select class="input" id="model-choice">
                <option value="cnn3d">3D CNN</option>
                <option value="cnn_gru">CNN-GRU</option>
              </select>
            </div>
          </div>
          <div class="status" id="upload-status" role="status" aria-live="polite"></div>
          <button class="btn" type="button" id="translate-upload" disabled>Translate</button>
          <div class="result">
            <div class="label">Result</div>
            <div class="result-text" id="upload-result">Your text will appear here</div>
          </div>
        </div>
      </section>

      <aside class="card history" id="history" aria-label="Translation history">
        <div class="history-head">
          <h2>History</h2>
          <button class="small-btn" type="button" id="clear-history">Clear</button>
        </div>
        <div class="history-list" id="history-list"></div>
      </aside>
    </div>

    <section class="learning" id="learning" aria-label="Learning">
      <h2>Learning</h2>
      <div class="grid">
        <div class="card">
          <h3>Alphabet Chart</h3>
          <div class="tiles">
            <div class="tile">A</div><div class="tile">B</div><div class="tile">C</div>
            <div class="tile">D</div><div class="tile">E</div><div class="tile">F</div>
          </div>
          <a class="link-btn" href="alphabet.php">Open Alphabet</a>
        </div>
        <div class="card">
          <h3>Word Chart</h3>
          <div class="tiles">
            <div class="tile" style="font-size:22px">Mom</div>
            <div class="tile" style="font-size:22px">Dad</div>
            <div class="tile" style="font-size:22px">Thank you</div>
          </div>
          <a class="link-btn" href="words.php">Open Words</a>
        </div>
        <div class="card">
          <h3>Quiz</h3>
          <div class="score"><div class="score-ring">9/10</div>Last score</div>
          <a class="link-btn" href="quiz.php">Start Quiz</a>
        </div>
      </div>
    </section>

    <section class="feedback" id="feedback" aria-label="Feedback">
      <div>
        <h2>Tell us what you think</h2>
        <p>Your feedback helps us make the app better.</p>
      </div>
      <a class="btn" href="feedback.php">Give Feedback</a>
    </section>
  </main>
</div>

<script>
/* Video predictions are sent to predict.php. */
async function runPrediction(videoFile, model) {
  if (!(videoFile instanceof Blob)) {
    return { error: 'Choose a video clip to translate.' };
  }
  try {
    var body = new FormData();
    body.append('video', videoFile, videoFile.name || 'camera.webm');
    body.append('model', model);
    var response = await fetch('predict.php', { method: 'POST', body: body });
    if (!response.ok) {
      var safeErrors = {
        400: 'Please choose a supported video file and try again.',
        413: 'This video is too large. Choose a video under 200 MB.'
      };
      return {
        error: safeErrors[response.status] ||
          'We could not translate this video. Please try again later.'
      };
    }
    var result = await response.json();
    if (!result.predictions || !result.predictions.length) {
      return { error: 'We could not recognize a sign in this video. Please try again.' };
    }
    return {
      text: result.predictions[0].label,
      score: result.predictions[0].score,
      model: result.model
    };
  } catch (err) {
    console.error('Video prediction request failed:', err);
    return { error: 'We could not connect to the translation service. Please try again later.' };
  }
}

function showResult(el, r) {
  if (r.error) { el.textContent = r.error; return false; }
  var pct = (typeof r.score === 'number') ? ' (' + Math.round(r.score * 100) + '% model score)' : '';
  el.textContent = r.text + pct;
  return true;
}

/* ---------------- Tabs ---------------- */
var tabs = document.querySelectorAll('.tab');
var panels = { 'tab-sign': 'panel-sign', 'tab-text': 'panel-text', 'tab-upload': 'panel-upload' };
tabs.forEach(function (tab) {
  tab.addEventListener('click', function () {
    tabs.forEach(function (t) {
      var active = t === tab;
      t.setAttribute('aria-selected', active ? 'true' : 'false');
      document.getElementById(panels[t.id]).hidden = !active;
    });
    if (tab.id !== 'tab-sign') { stopCamera(); }
  });
});

/* ---------------- History (saved in this browser) ---------------- */
var HKEY = 'asl_history';
var historyList = document.getElementById('history-list');

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch (e) { return []; }
}
function saveHistory(list) {
  try { localStorage.setItem(HKEY, JSON.stringify(list)); } catch (e) {}
}
function renderHistory() {
  var list = loadHistory();
  historyList.innerHTML = '';
  if (list.length === 0) {
    historyList.innerHTML = '<div class="history-empty">No translations yet</div>';
    return;
  }
  list.forEach(function (h) {
    var d = document.createElement('div');
    d.className = 'history-item';
    var sm = document.createElement('small'); sm.textContent = h.kind;
    var st = document.createElement('strong'); st.textContent = h.text;
    d.appendChild(sm); d.appendChild(st);
    historyList.appendChild(d);
  });
}
function addHistory(kind, text) {
  var list = loadHistory();
  list.unshift({ kind: kind, text: text });
  saveHistory(list.slice(0, 20));
  renderHistory();
}
document.getElementById('clear-history').addEventListener('click', function () {
  saveHistory([]);
  renderHistory();
});
renderHistory();

/* ---------------- Sign to Text (camera) ---------------- */
var cam = document.getElementById('cam');
var camPlaceholder = document.getElementById('cam-placeholder');
var camToggle = document.getElementById('cam-toggle');
var captureBtn = document.getElementById('capture');
var flipBtn = document.getElementById('flip');
var camStatus = document.getElementById('cam-status');
var signResult = document.getElementById('sign-result');
var stream = null;
var facing = 'user';
var liveTranslation = false;
var activeRecorder = null;
var CAMERA_CLIP_DURATION_MS = 2250;
var RETRY_DELAY_MS = 1000;

function setCamStatus(msg, isError) {
  camStatus.textContent = msg;
  camStatus.className = isError ? 'status error' : 'status';
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCamStatus('Camera is not available. Open this page with https or localhost.', true);
    return;
  }
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facing,
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 }
      },
      audio: false
    });
    cam.srcObject = stream;
    cam.hidden = false;
    camPlaceholder.hidden = true;
    captureBtn.disabled = false;
    flipBtn.disabled = false;
    camToggle.textContent = 'Stop Camera';
    setCamStatus('Camera is on. Start live translation when you are ready.', false);
  } catch (err) {
    setCamStatus('Could not open the camera. Please allow camera permission.', true);
  }
}

function stopCamera() {
  liveTranslation = false;
  if (activeRecorder && activeRecorder.state === 'recording') {
    activeRecorder.stop();
  }
  if (stream) {
    stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
  }
  cam.srcObject = null;
  cam.hidden = true;
  captureBtn.disabled = true;
  captureBtn.textContent = 'Start Live Translation';
  flipBtn.disabled = true;
  camToggle.textContent = 'Start Camera';
  camPlaceholder.hidden = false;
}

camToggle.addEventListener('click', function () {
  if (stream) { stopCamera(); setCamStatus('', false); } else { startCamera(); }
});

flipBtn.addEventListener('click', function () {
  facing = (facing === 'user') ? 'environment' : 'user';
  startCamera();
});

function recordCameraClip() {
  return new Promise(function (resolve, reject) {
    var supportedTypes = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    var mimeType = supportedTypes.find(function (type) {
      return MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type);
    });
    var recorder;
    try {
      recorder = mimeType ? new MediaRecorder(stream, { mimeType: mimeType }) : new MediaRecorder(stream);
    } catch (err) {
      reject(new Error('This browser cannot record camera video for translation.'));
      return;
    }

    activeRecorder = recorder;
    var chunks = [];
    recorder.ondataavailable = function (event) {
      if (event.data.size > 0) { chunks.push(event.data); }
    };
    recorder.onerror = function () {
      if (activeRecorder === recorder) { activeRecorder = null; }
      reject(new Error('Could not record a camera clip. Please try again.'));
    };
    recorder.onstop = function () {
      if (activeRecorder === recorder) { activeRecorder = null; }
      resolve(new Blob(chunks, { type: recorder.mimeType || 'video/webm' }));
    };
    recorder.start();
    window.setTimeout(function () {
      if (recorder.state === 'recording') { recorder.stop(); }
    }, CAMERA_CLIP_DURATION_MS);
  });
}

async function runLiveTranslation() {
  var lastPrediction = '';
  var translatedWords = [];
  while (stream && liveTranslation) {
    try {
      setCamStatus('Recording a sign...', false);
      var clip = await recordCameraClip();
      if (!liveTranslation) { break; }
      if (!clip.size) { throw new Error('No camera video was recorded. Please try again.'); }

      setCamStatus('Analyzing this sign; this can take several seconds...', false);
      var model = document.getElementById('camera-model-choice').value;
      var result = await runPrediction(clip, model);
      if (!liveTranslation) { break; }
      if (result.error) {
        lastPrediction = '';
        setCamStatus('Could not read that clip. Keep signing; trying another clip...', true);
        await new Promise(function (resolve) { window.setTimeout(resolve, RETRY_DELAY_MS); });
        continue;
      }

      if (result.text !== lastPrediction) {
        translatedWords.push(result.text);
        addHistory('Sign to Text', result.text);
        lastPrediction = result.text;
      }
      signResult.textContent = translatedWords.join(' ');
      setCamStatus('Live translation is on. Show one sign at a time.', false);
    } catch (error) {
      if (!liveTranslation) { break; }
      console.error('Camera clip translation failed:', error);
      lastPrediction = '';
      setCamStatus('Could not record that clip. Keep signing; trying again...', true);
      await new Promise(function (resolve) { window.setTimeout(resolve, RETRY_DELAY_MS); });
    }
  }
}

captureBtn.addEventListener('click', function () {
  if (liveTranslation) {
    liveTranslation = false;
    if (activeRecorder && activeRecorder.state === 'recording') {
      activeRecorder.stop();
    }
    captureBtn.textContent = 'Start Live Translation';
    setCamStatus('Live translation paused.', false);
    return;
  }
  if (!stream) { return; }
  if (!window.MediaRecorder) {
    setCamStatus('Live video recording is not supported by this browser.', true);
    return;
  }
  liveTranslation = true;
  captureBtn.textContent = 'Stop Live Translation';
  signResult.textContent = 'Your text will appear here';
  runLiveTranslation();
});

/* ---------------- Text to Sign ---------------- */
var txt = document.getElementById('txt');
var signsBox = document.getElementById('signs');
var textStatus = document.getElementById('text-status');
var SIGN_IMAGE_PATH = 'signs/';   // put A.png ... Z.png in this folder

function makeSignTile(ch) {
  var item = document.createElement('div');
  item.className = 'sign';
  var box = document.createElement('div');
  box.className = 'sign-img';
  var img = document.createElement('img');
  img.alt = 'ASL sign for ' + ch;
  img.src = SIGN_IMAGE_PATH + ch + '.png';
  img.onerror = function () { box.removeChild(img); box.textContent = 'ASL image'; };
  box.appendChild(img);
  item.appendChild(box);
  item.appendChild(document.createTextNode(ch));
  return item;
}

function showSigns() {
  var value = txt.value.trim();
  signsBox.innerHTML = '';
  var count = 0;
  var upper = value.toUpperCase();
  for (var i = 0; i < upper.length; i++) {
    var ch = upper.charAt(i);
    if (ch >= 'A' && ch <= 'Z') {
      signsBox.appendChild(makeSignTile(ch));
      count++;
    } else if (ch === ' ') {
      var gap = document.createElement('div');
      gap.className = 'sign-gap';
      signsBox.appendChild(gap);
    }
  }
  if (count === 0) {
    signsBox.innerHTML = '<div class="history-empty">Type letters A-Z to see the signs</div>';
    return;
  }
  addHistory('Text to Sign', value);
}

document.getElementById('show-sign').addEventListener('click', showSigns);
txt.addEventListener('keydown', function (e) { if (e.key === 'Enter') { showSigns(); } });

// Voice input (Web Speech API)
var voiceBtn = document.getElementById('voice');
var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
var recognizer = null;
if (!SR) {
  voiceBtn.disabled = true;
  voiceBtn.style.opacity = '.45';
  textStatus.textContent = 'Voice input works in Chrome or Edge.';
} else {
  recognizer = new SR();
  recognizer.lang = 'en-US';
  recognizer.interimResults = false;
  recognizer.onresult = function (e) {
    txt.value = e.results[0][0].transcript;
    showSigns();
  };
  recognizer.onend = function () {
    voiceBtn.classList.remove('listening');
    voiceBtn.textContent = 'Voice';
  };
  recognizer.onerror = function () {
    textStatus.textContent = 'Could not hear you. Please try again.';
  };
  voiceBtn.addEventListener('click', function () {
    textStatus.textContent = '';
    voiceBtn.classList.add('listening');
    voiceBtn.textContent = 'Listening...';
    try { recognizer.start(); } catch (err) {}
  });
}

/* ---------------- Upload Media ---------------- */
var preview = document.getElementById('preview');
var uploadStatus = document.getElementById('upload-status');
var uploadResult = document.getElementById('upload-result');
var translateBtn = document.getElementById('translate-upload');
var filePhoto = document.getElementById('file-photo');
var fileVideo = document.getElementById('file-video');
var currentFile = null;
var currentUrl = null;

document.getElementById('pick-photo').addEventListener('click', function () { filePhoto.click(); });
document.getElementById('pick-video').addEventListener('click', function () { fileVideo.click(); });

function loadFile(file, kind) {
  if (!file) { return; }
  currentFile = kind === 'video' ? file : null;
  if (currentUrl) { URL.revokeObjectURL(currentUrl); }
  currentUrl = URL.createObjectURL(file);
  preview.innerHTML = '';
  var el;
  if (kind === 'photo') {
    el = document.createElement('img');
    el.alt = 'Uploaded sign photo';
  } else {
    el = document.createElement('video');
    el.controls = true;
    el.playsInline = true;
  }
  el.src = currentUrl;
  preview.appendChild(el);
  preview.hidden = false;
  translateBtn.disabled = kind !== 'video';
  uploadResult.textContent = 'Your text will appear here';
  uploadStatus.textContent = kind === 'video' ? file.name : 'The video model requires a video clip.';
  uploadStatus.className = 'status';
}
filePhoto.addEventListener('change', function () { loadFile(filePhoto.files[0], 'photo'); filePhoto.value = ''; });
fileVideo.addEventListener('change', function () { loadFile(fileVideo.files[0], 'video'); fileVideo.value = ''; });

translateBtn.addEventListener('click', async function () {
  if (!currentFile) { return; }
  uploadStatus.textContent = 'Reading your sign...';
  var r = await runPrediction(currentFile, document.getElementById('model-choice').value);
  var ok = showResult(uploadResult, r);
  if (ok) { addHistory('Upload Media', r.text); uploadStatus.textContent = 'Done.'; }
  else { uploadStatus.textContent = 'File loaded.'; }
});
</script>
</body>
</html>
