document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var CONF_MIN = 0.6; // below this, show the "uncertain" result

  /* =========================================================
     AI model hook - connect your model here.
     Define window.predictSign(source) anywhere on the page.
       source = <canvas>, <img> or <video>
       return (or resolve to): { text: "Hello", confidence: 0.92 }
     ========================================================= */
  async function runPrediction(source) {
    if (typeof window.predictSign !== 'function') {
      return { error: 'The AI model is not connected yet.' };
    }
    try {
      var r = await window.predictSign(source);
      if (!r || !r.text) { return { error: 'No sign found. Please try again.' }; }
      return r;
    } catch (err) {
      return { error: 'Something went wrong. Please try again.' };
    }
  }

  /* ---------- Result box ---------- */
  function renderResult(el, state, text, note, conf) {
    var ico = { idle: 'eye', ok: 'check', warn: 'alert' }[state] || 'eye';
    el.className = 'result' + (state === 'ok' || state === 'warn' ? ' ' + state : '');
    var meter = (typeof conf === 'number')
      ? '<div class="meter" role="img" aria-label="Confidence ' + Math.round(conf * 100) + ' percent"><i style="width:' + Math.round(conf * 100) + '%"></i></div>'
      : '';
    el.innerHTML =
      '<div class="ico">' + A.icon(ico, 26) + '</div>' +
      '<div><div class="label">Result</div><div class="text">' + A.esc(text) + '</div>' +
      (note ? '<div class="note">' + A.esc(note) + '</div>' : '') + meter + '</div>';
  }

  // Returns true when a confident result was shown and saved
  function handlePrediction(r, kind, el) {
    if (r.error) {
      renderResult(el, 'idle', 'Your text will appear here', r.error);
      return false;
    }
    if (typeof r.confidence === 'number' && r.confidence < CONF_MIN) {
      renderResult(el, 'warn', 'Not sure: maybe "' + r.text + '"',
        'Try again with better light and a plain background.', r.confidence);
      return false;
    }
    renderResult(el, 'ok', r.text, '', r.confidence);
    A.history.add(kind, r.text);
    return true;
  }

  var signResult = document.getElementById('sign-result');
  var uploadResult = document.getElementById('upload-result');
  renderResult(signResult, 'idle', 'Your text will appear here');
  renderResult(uploadResult, 'idle', 'Your text will appear here');

  /* ---------- Tabs ---------- */
  var tabs = [
    { tab: document.getElementById('tab-sign'), panel: document.getElementById('panel-sign') },
    { tab: document.getElementById('tab-text'), panel: document.getElementById('panel-text') },
    { tab: document.getElementById('tab-upload'), panel: document.getElementById('panel-upload') }
  ];
  tabs.forEach(function (item) {
    item.tab.addEventListener('click', function () {
      tabs.forEach(function (t) {
        var on = (t === item);
        t.tab.setAttribute('aria-selected', on ? 'true' : 'false');
        t.panel.hidden = !on;
      });
      if (item.tab.id !== 'tab-sign') { stopCamera(); }
    });
  });

  /* ---------- Sign to Text (camera) ---------- */
  var camBox = document.getElementById('cam-box');
  var cam = document.getElementById('cam');
  var snap = document.getElementById('snap');
  var camPh = document.getElementById('cam-ph');
  var camToggle = document.getElementById('cam-toggle');
  var camLabel = document.getElementById('cam-toggle-label');
  var captureBtn = document.getElementById('capture');
  var flipBtn = document.getElementById('flip');
  var camStatus = document.getElementById('cam-status');
  var stream = null;
  var facing = A.getProfile().camera === 'environment' ? 'environment' : 'user';

  function setCamStatus(msg, isError) {
    camStatus.textContent = msg;
    camStatus.className = isError ? 'status error' : 'status';
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCamStatus('Camera is not available. Open this page with https or localhost.', true);
      return;
    }
    stopCamera(true);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing }, audio: false });
      cam.srcObject = stream;
      cam.hidden = false; snap.hidden = true; camPh.hidden = true;
      camBox.classList.add('live');
      captureBtn.disabled = false; flipBtn.disabled = false;
      camLabel.textContent = 'Stop camera';
      setCamStatus('Camera is on. Show your sign, then tap "Capture sign".', false);
    } catch (err) {
      setCamStatus('Could not open the camera. Please allow camera permission.', true);
    }
  }

  function stopCamera(keepStatus) {
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    cam.srcObject = null; cam.hidden = true;
    camBox.classList.remove('live');
    captureBtn.disabled = true; flipBtn.disabled = true;
    camLabel.textContent = 'Start camera';
    if (snap.hidden) { camPh.hidden = false; }
    if (!keepStatus) { setCamStatus('', false); }
  }

  camToggle.addEventListener('click', function () { if (stream) { stopCamera(); } else { startCamera(); } });
  flipBtn.addEventListener('click', function () {
    facing = (facing === 'user') ? 'environment' : 'user';
    startCamera();
  });

  captureBtn.addEventListener('click', async function () {
    if (!stream || !cam.videoWidth) { return; }
    var canvas = document.createElement('canvas');
    canvas.width = cam.videoWidth; canvas.height = cam.videoHeight;
    canvas.getContext('2d').drawImage(cam, 0, 0);
    snap.src = canvas.toDataURL('image/jpeg', 0.9);
    snap.hidden = false; cam.hidden = true; camPh.hidden = true;
    setCamStatus('Reading your sign...', false);
    var r = await runPrediction(canvas);
    var ok = handlePrediction(r, 'Sign to Text', signResult);
    setCamStatus(ok ? 'Done.' : 'Photo captured. Capture again to retry.', false);
    setTimeout(function () { if (stream) { cam.hidden = false; snap.hidden = true; } }, 1800);
  });

  /* ---------- Text to Sign ---------- */
  var txt = document.getElementById('txt');
  var signs = document.getElementById('signs');
  var textStatus = document.getElementById('text-status');

  function showSigns() {
    var raw = txt.value.trim();
    var up = raw.toUpperCase();
    signs.innerHTML = '';
    var count = 0, skipped = 0;
    for (var i = 0; i < up.length; i++) {
      var ch = up.charAt(i);
      if (ch >= 'A' && ch <= 'Z') {
        var tile = document.createElement('div');
        tile.className = 'tile';
        tile.appendChild(A.imgBox('signs/' + ch + '.png', ch, 'ASL sign for ' + ch));
        var l = document.createElement('div');
        l.className = 'letter'; l.textContent = ch;
        tile.appendChild(l);
        signs.appendChild(tile);
        count++;
      } else if (ch === ' ') {
        var gap = document.createElement('div');
        gap.className = 'sgap';
        signs.appendChild(gap);
      } else {
        skipped++;
      }
    }
    if (count === 0) {
      signs.innerHTML = '<p class="muted">Type letters A to Z to see the signs.</p>';
      textStatus.textContent = '';
      return;
    }
    textStatus.textContent = skipped ? 'Only letters A to Z can be signed. Skipped ' + skipped + ' other character(s).' : '';
    A.history.add('Text to Sign', raw);
  }
  document.getElementById('show-sign').addEventListener('click', showSigns);
  txt.addEventListener('keydown', function (e) { if (e.key === 'Enter') { showSigns(); } });

  // Voice input
  var voiceBtn = document.getElementById('voice');
  var voiceLabel = document.getElementById('voice-label');
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    voiceBtn.disabled = true;
    textStatus.textContent = 'Voice input works in Chrome or Edge.';
  } else {
    var rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.onresult = function (e) { txt.value = e.results[0][0].transcript; showSigns(); };
    rec.onend = function () { voiceBtn.classList.remove('listening'); voiceLabel.textContent = 'Speak'; };
    rec.onerror = function () { textStatus.textContent = 'Could not hear you. Please try again.'; };
    voiceBtn.addEventListener('click', function () {
      textStatus.textContent = '';
      voiceBtn.classList.add('listening');
      voiceLabel.textContent = 'Listening...';
      try { rec.start(); } catch (err) {}
    });
  }

  /* ---------- Upload Media ---------- */
  var preview = document.getElementById('preview');
  var uploadStatus = document.getElementById('upload-status');
  var translateBtn = document.getElementById('translate-upload');
  var filePhoto = document.getElementById('file-photo');
  var fileVideo = document.getElementById('file-video');
  var drop = document.getElementById('drop');
  var currentMedia = null, currentUrl = null;

  document.getElementById('pick-photo').addEventListener('click', function () { filePhoto.click(); });
  document.getElementById('pick-video').addEventListener('click', function () { fileVideo.click(); });

  function loadFile(file) {
    if (!file) { return; }
    var isImage = file.type.indexOf('image/') === 0;
    var isVideo = file.type.indexOf('video/') === 0;
    if (!isImage && !isVideo) {
      uploadStatus.textContent = 'Please choose a photo or a video.';
      uploadStatus.className = 'status error';
      return;
    }
    if (currentUrl) { URL.revokeObjectURL(currentUrl); }
    currentUrl = URL.createObjectURL(file);
    preview.innerHTML = '';
    var el = document.createElement(isImage ? 'img' : 'video');
    if (isImage) { el.alt = 'Uploaded sign photo'; } else { el.controls = true; el.playsInline = true; }
    el.src = currentUrl;
    preview.appendChild(el);
    preview.hidden = false;
    currentMedia = el;
    translateBtn.disabled = false;
    renderResult(uploadResult, 'idle', 'Your text will appear here');
    uploadStatus.textContent = file.name;
    uploadStatus.className = 'status';
  }
  filePhoto.addEventListener('change', function () { loadFile(filePhoto.files[0]); filePhoto.value = ''; });
  fileVideo.addEventListener('change', function () { loadFile(fileVideo.files[0]); fileVideo.value = ''; });

  ['dragenter', 'dragover'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); });
  });
  drop.addEventListener('drop', function (e) { loadFile(e.dataTransfer.files[0]); });

  translateBtn.addEventListener('click', async function () {
    if (!currentMedia) { return; }
    uploadStatus.textContent = 'Reading your sign...';
    var r = await runPrediction(currentMedia);
    var ok = handlePrediction(r, 'Upload Media', uploadResult);
    uploadStatus.textContent = ok ? 'Done.' : 'File loaded.';
  });
});
