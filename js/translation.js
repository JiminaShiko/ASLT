document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var CONF_MIN = 0.6;
  var API_BASE = 'http://127.0.0.1:5000';
  var LIVE_INTERVAL = 300;
  var LIVE_MAX_SIDE = 640;

  /* ---------- Shared helpers ---------- */
  async function modelRequest(path, payload) {
    var response;
    try {
      response = await fetch(API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
    } catch (err) {
      throw new Error('Could not reach the photo model. Start it with "python api.py" in the photo model folder.');
    }
    var result;
    try {
      result = await response.json();
    } catch (err) {
      throw new Error('The photo model returned an invalid response.');
    }
    if (!response.ok) {
      throw new Error(result.error || 'The photo model could not make a prediction.');
    }
    return result;
  }

  function sourceToDataURL(source, maxSide) {
    var w = source.videoWidth || source.naturalWidth || source.width;
    var h = source.videoHeight || source.naturalHeight || source.height;
    if (!w || !h) { throw new Error('The image is not ready yet. Please try again.'); }
    var scale = 1;
    if (maxSide) { scale = Math.min(1, maxSide / Math.max(w, h)); }
    var canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(w * scale));
    canvas.height = Math.max(1, Math.round(h * scale));
    var ctx = canvas.getContext('2d');
    if (!ctx) { throw new Error('Could not prepare the image.'); }
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.88);
  }

  /* Single-shot prediction (used by Upload Media) */
  window.predictSign = async function (source) {
    var dataUrl = sourceToDataURL(source, 1280);
    var result = await modelRequest('/api/predict-photo', { image: dataUrl });
    if (result.hand_found === false || result.text === 'nothing') {
      throw new Error('No hand sign was found. Show the whole hand clearly and try again.');
    }
    return { text: result.text, confidence: result.confidence };
  };

  async function runPrediction(source) {
    if (typeof window.predictSign !== 'function') {
      return { error: 'The AI model is not connected yet.' };
    }
    try {
      var r = await window.predictSign(source);
      if (!r || !r.text) { return { error: 'No sign found. Please try again.' }; }
      return r;
    } catch (err) {
      return { error: err.message || 'Something went wrong. Please try again.' };
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

  /* ---------- Sign to Text (camera + live) ---------- */
  var camBox = document.getElementById('cam-box');
  var cam = document.getElementById('cam');
  var camPh = document.getElementById('cam-ph');
  var camToggle = document.getElementById('cam-toggle');
  var camLabel = document.getElementById('cam-toggle-label');
  var captureBtn = document.getElementById('capture');
  var captureLabel = document.getElementById('capture-label');
  var flipBtn = document.getElementById('flip');
  var camStatus = document.getElementById('cam-status');
  var mirrorLive = document.getElementById('mirror-live');
  var leftHandLive = document.getElementById('left-hand-live');
  var liveText = document.getElementById('live-text');
  var liveTop = document.getElementById('live-top');
  var clearLiveBtn = document.getElementById('clear-live');
  var liveDebugBox = document.getElementById('live-debug-box');
  var liveDebug = document.getElementById('live-debug');
  var stream = null;
  var liveSession = null;
  var liveActive = false;
  var liveRunId = 0;
  var nextLiveFrameAt = 0;
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
      cam.hidden = false;
      camPh.hidden = true;
      camBox.classList.add('live');
      captureBtn.disabled = false;
      flipBtn.disabled = false;
      camLabel.textContent = 'Stop camera';
      setCamStatus('Camera is on. Start live translation when you are ready.', false);
    } catch (err) {
      setCamStatus('Could not open the camera. Please allow camera permission.', true);
    }
  }

  function renderLiveFrame(frame) {
    if (frame.letter === '-') {
      renderResult(signResult, 'idle', 'Waiting for a sign', 'Show your hand inside the green box.');
    } else {
      var isNothing = frame.letter === 'nothing';
      var note = !frame.hand_found
        ? 'The camera does not see a hand. Improve the light and keep your whole hand visible.'
        : (frame.confidence < CONF_MIN
          ? 'The model is not sure. Show the whole hand clearly and closer to the camera.'
          : 'Confidence: ' + (frame.confidence * 100).toFixed(2) + '%');
      renderResult(signResult,
        frame.hand_found && frame.confidence >= CONF_MIN ? 'ok' : 'warn',
        isNothing ? 'NO SIGN' : frame.letter.toUpperCase(),
        note,
        frame.confidence);
    }

    liveText.value = frame.text || '';
    liveTop.innerHTML = '';
    (frame.top_predictions || []).forEach(function (p) {
      var item = document.createElement('li');
      item.textContent = p.text.toUpperCase() + ': ' + (p.confidence * 100).toFixed(2) + '%';
      liveTop.appendChild(item);
    });
    if (!frame.top_predictions || !frame.top_predictions.length) {
      var empty = document.createElement('li');
      empty.className = 'muted';
      empty.textContent = 'Waiting for a sign.';
      liveTop.appendChild(empty);
    }
    if (frame.processed_image) {
      liveDebug.src = frame.processed_image;
      liveDebugBox.hidden = false;
    }
  }

  function drawLiveFrame() {
    var width = cam.videoWidth;
    var height = cam.videoHeight;
    if (!width || !height) { return null; }
    var scale = Math.min(1, LIVE_MAX_SIDE / Math.max(width, height));
    var canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    var ctx = canvas.getContext('2d');
    if (!ctx) { throw new Error('Could not prepare the camera frame.'); }
    ctx.drawImage(cam, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  }

  async function captureLiveFrame(runId) {
    if (!liveActive || runId !== liveRunId || !stream) { return; }
    var image;
    try {
      image = drawLiveFrame();
    } catch (err) {
      setCamStatus(err.message, true);
      await stopLiveTranslation(false);
      return;
    }
    if (!image) {
      setTimeout(function () { captureLiveFrame(runId); }, 100);
      return;
    }
    try {
      var frame = await modelRequest('/api/live/frame', {
        session_id: liveSession,
        image: image,
        left_hand: leftHandLive.checked
      });
      if (!liveActive || runId !== liveRunId) { return; }
      renderLiveFrame(frame);
    } catch (err) {
      if (runId === liveRunId) {
        await stopLiveTranslation(false);
        setCamStatus(err.message, true);
      }
      return;
    }
    nextLiveFrameAt += LIVE_INTERVAL;
    setTimeout(function () {
      captureLiveFrame(runId);
    }, Math.max(0, nextLiveFrameAt - Date.now()));
  }

  async function startLiveTranslation() {
    if (!stream || liveActive) { return; }
    captureBtn.disabled = true;
    setCamStatus('Starting live translation...', false);
    var runId = ++liveRunId;
    try {
      var result = await modelRequest('/api/live/start', {});
      if (runId !== liveRunId || !stream) {
        modelRequest('/api/live/stop', { session_id: result.session_id }).catch(function () {});
        return;
      }
      liveSession = result.session_id;
      liveActive = true;
      captureBtn.disabled = false;
      captureLabel.textContent = 'Stop live translation';
      clearLiveBtn.disabled = false;
      liveText.value = '';
      liveDebugBox.hidden = true;
      liveTop.innerHTML = '<li class="muted">Waiting for a sign.</li>';
      renderResult(signResult, 'idle', 'Waiting for a sign', 'Show your hand inside the green box.');
      setCamStatus('Live translation is on. Hold each sign steady inside the green box.', false);
      nextLiveFrameAt = Date.now();
      captureLiveFrame(runId);
    } catch (err) {
      captureBtn.disabled = !stream;
      setCamStatus(err.message, true);
    }
  }

  async function stopLiveTranslation(saveHistory) {
    var sessionId = liveSession;
    var previousText = liveText ? liveText.value : '';
    liveActive = false;
    liveSession = null;
    liveRunId++;
    if (captureLabel) { captureLabel.textContent = 'Start live translation'; }
    if (captureBtn) { captureBtn.disabled = !stream; }
    if (clearLiveBtn) { clearLiveBtn.disabled = true; }
    if (!sessionId) { return; }

    var finalText = previousText;
    try {
      var result = await modelRequest('/api/live/stop', { session_id: sessionId });
      finalText = result.text || '';
      if (liveText) { liveText.value = finalText; }
    } catch (err) {
      if (saveHistory) { setCamStatus(err.message, true); }
    }
    if (saveHistory && finalText && finalText.trim()) {
      A.history.add('Sign to Text', finalText);
    }
  }

  function stopCamera(keepStatus) {
    stopLiveTranslation(true);
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    cam.srcObject = null;
    cam.hidden = true;
    camBox.classList.remove('live');
    if (captureBtn) { captureBtn.disabled = true; }
    if (flipBtn) { flipBtn.disabled = true; }
    if (clearLiveBtn) { clearLiveBtn.disabled = true; }
    if (camLabel) { camLabel.textContent = 'Start camera'; }
    if (camPh) { camPh.hidden = false; }
    if (!keepStatus) { setCamStatus('', false); }
  }

  camToggle.addEventListener('click', function () {
    if (stream) { stopCamera(); } else { startCamera(); }
  });
  captureBtn.addEventListener('click', function () {
    if (liveActive) {
      stopLiveTranslation(true);
      setCamStatus('Live translation stopped.', false);
    } else {
      startLiveTranslation();
    }
  });
  clearLiveBtn.addEventListener('click', async function () {
    if (!liveSession) {
      liveText.value = '';
      return;
    }
    clearLiveBtn.disabled = true;
    try {
      var result = await modelRequest('/api/live/clear', { session_id: liveSession });
      liveText.value = result.text || '';
    } catch (err) {
      setCamStatus(err.message, true);
    } finally {
      clearLiveBtn.disabled = !liveActive;
    }
  });
  mirrorLive.addEventListener('change', function () {
    cam.style.transform = mirrorLive.checked ? 'scaleX(-1)' : '';
  });
  flipBtn.addEventListener('click', function () {
    facing = (facing === 'user') ? 'environment' : 'user';
    startCamera();
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
        l.className = 'letter';
        l.textContent = ch;
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
    textStatus.textContent = skipped
      ? 'Only letters A to Z can be signed. Skipped ' + skipped + ' other character(s).'
      : '';
    A.history.add('Text to Sign', raw);
  }
  document.getElementById('show-sign').addEventListener('click', showSigns);
  txt.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { showSigns(); }
  });

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
    rec.onresult = function (e) {
      txt.value = e.results[0][0].transcript;
      showSigns();
    };
    rec.onend = function () {
      voiceBtn.classList.remove('listening');
      voiceLabel.textContent = 'Speak';
    };
    rec.onerror = function () {
      textStatus.textContent = 'Could not hear you. Please try again.';
    };
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
    if (isImage) {
      el.alt = 'Uploaded sign photo';
    } else {
      el.controls = true;
      el.playsInline = true;
    }
    el.src = currentUrl;
    preview.appendChild(el);
    preview.hidden = false;
    currentMedia = el;
    translateBtn.disabled = false;
    renderResult(uploadResult, 'idle', 'Your text will appear here');
    uploadStatus.textContent = file.name;
    uploadStatus.className = 'status';
  }
  filePhoto.addEventListener('change', function () {
    loadFile(filePhoto.files[0]);
    filePhoto.value = '';
  });
  fileVideo.addEventListener('change', function () {
    loadFile(fileVideo.files[0]);
    fileVideo.value = '';
  });

  ['dragenter', 'dragover'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      drop.classList.add('over');
    });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      drop.classList.remove('over');
    });
  });
  drop.addEventListener('drop', function (e) {
    loadFile(e.dataTransfer.files[0]);
  });

  translateBtn.addEventListener('click', async function () {
    if (!currentMedia) { return; }
    uploadStatus.textContent = 'Reading your sign...';
    var r = await runPrediction(currentMedia);
    var ok = handlePrediction(r, 'Upload Media', uploadResult);
    uploadStatus.textContent = ok ? 'Done.' : 'File loaded.';
  });
});
