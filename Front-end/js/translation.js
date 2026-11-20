document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;
  var STATIC_PREVIEW_PORTS = { '5500': true, '8000': true, '3000': true };
  var API_BASE = (location.protocol === 'file:' || STATIC_PREVIEW_PORTS[location.port])
    ? 'http://127.0.0.1:5000'
    : '';
  var API_CREDENTIALS = API_BASE ? 'include' : 'same-origin';
  var LIVE_INTERVAL = 55;
  var LIVE_MAX_SIDE = 640;

  /* ---------- Shared helpers ---------- */
  async function modelRequest(path, payload) {
    var response;
    try {
      response = await fetch(API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: API_CREDENTIALS,
        body: JSON.stringify(payload || {})
      });
    } catch (err) {
      throw new Error('Could not reach the ASL model API. Start the backend with "python app.py" in the Backend folder.');
    }
    var result;
    var raw = await response.text();
    try {
      result = JSON.parse(raw);
    } catch (err) {
      throw new Error(
        'The ASL model API returned an invalid response (HTTP ' + response.status +
        ' from ' + (response.url || path) + '). Open the site through the Flask backend: ' +
        'http://127.0.0.1:5000/home.'
      );
    }
    if (!response.ok) {
      throw new Error(result.error || 'The ASL model could not make a prediction.');
    }
    return result;
  }

  async function loadModelInfo() {
    try {
      var response = await fetch(API_BASE + '/api/health', { credentials: API_CREDENTIALS });
      if (!response.ok) { throw new Error('HTTP ' + response.status); }
      var info = await response.json();
      var profile = document.getElementById('profile-live');
      profile.innerHTML = '';
      (info.profiles || []).forEach(function (name) {
        var option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        profile.appendChild(option);
      });
      profile.value = info.default_profile || 'all';
      var settingControls = {
        conf: 'conf-live',
        margin: 'margin-live',
        vote: 'vote-live',
        hold: 'hold-live',
        motion: 'motion-live',
        landmarks: 'landmarks-live',
        motion_debug: 'motion-debug-live',
        mirror: 'mirror-live',
        recover: 'recover-live',
        recover_every: 'recover-every-live',
        recover_conf: 'recover-conf-live',
        recover_flip: 'recover-flip-live',
        jz: 'jz-live',
        calibrate_reps: 'calibrate-reps-live'
      };
      Object.keys(settingControls).forEach(function (key) {
        var value = info.default_settings[key];
        var control = document.getElementById(settingControls[key]);
        if (control && value !== null && typeof value !== 'object') {
          if (control.type === 'checkbox') { control.checked = Boolean(value); }
          else { control.value = value; }
        }
      });
      applyMirrorPreview();
      await refreshCameraOptions();
      return info;
    } catch (err) {
      setCamStatus('Could not load model settings. Start the ASL model API with "python api.py".', true);
      return null;
    }
  }

  function liveSettings() {
    var customMargins = document.getElementById('pair-custom-live').value.trim();
    var pairMargins = customMargins || (
      document.getElementById('pairs-live').checked ? null : 'none'
    );
    return {
      profile: document.getElementById('profile-live').value,
      backend: document.getElementById('backend-live').value,
      conf: Number(document.getElementById('conf-live').value),
      margin: Number(document.getElementById('margin-live').value),
      pair_margins: pairMargins,
      vote: Number(document.getElementById('vote-live').value),
      hold: Number(document.getElementById('hold-live').value),
      motion: document.getElementById('motion-live').checked,
      landmarks: document.getElementById('landmarks-live').checked,
      motion_debug: document.getElementById('motion-debug-live').checked,
      mirror: document.getElementById('mirror-live').checked,
      recover: Number(document.getElementById('recover-live').value),
      recover_every: Number(document.getElementById('recover-every-live').value),
      recover_conf: Number(document.getElementById('recover-conf-live').value),
      recover_flip: document.getElementById('recover-flip-live').checked,
      jz: document.getElementById('jz-live').value.trim(),
      calibrate_reps: Number(document.getElementById('calibrate-reps-live').value)
    };
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
    var result = await modelRequest('/api/predict-photo', {
      image: dataUrl,
      settings: liveSettings()
    });
    if (result.hand_found === false || result.text === 'nothing') {
      throw new Error('No hand sign was found. Show the whole hand clearly and try again.');
    }
    return {
      text: result.text,
      confidence: result.confidence,
      uncertain: result.uncertain,
      low_conf: result.low_conf,
      top_predictions: result.top_predictions
    };
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
    if (r.uncertain || r.low_conf) {
      renderResult(el, 'warn', 'Not sure: maybe "' + r.text + '"',
        r.uncertain
          ? 'The frame did not clear the console typer’s top-two margin, so it would cast no vote.'
          : 'The frame is below the console typer’s confidence threshold, so it would cast no vote.',
        r.confidence);
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
  var camRender = document.getElementById('cam-render');
  var mirrorToggle = document.getElementById('mirror-live');
  var camPh = document.getElementById('cam-ph');
  var camToggle = document.getElementById('cam-toggle');
  var camLabel = document.getElementById('cam-toggle-label');
  var captureBtn = document.getElementById('capture');
  var captureLabel = document.getElementById('capture-label');
  var commitBtn = document.getElementById('commit-live');
  var backspaceBtn = document.getElementById('backspace-live');
  var calibrateBtn = document.getElementById('calibrate-live');
  var cancelCalibrationBtn = document.getElementById('cancel-calibration-live');
  var camStatus = document.getElementById('cam-status');
  var liveText = document.getElementById('live-text');
  var liveTop = document.getElementById('live-top');
  var clearLiveBtn = document.getElementById('clear-live');
  var readBtn = document.getElementById('read-live');
  var stream = null;
  var liveSession = null;
  var liveActive = false;
  var liveRunId = 0;
  var nextLiveFrameAt = 0;
  var facing = A.getProfile().camera === 'environment' ? 'environment' : 'user';
  var calibrationStatus = document.getElementById('calibration-status');
  var calibrationProgress = document.getElementById('calibration-progress');
  var calibrationReport = document.getElementById('calibration-report');

  function setCamStatus(msg, isError) {
    camStatus.textContent = msg;
    camStatus.className = isError ? 'status error' : 'status';
  }

  function applyMirrorPreview() {
    camBox.classList.toggle('mirrored', mirrorToggle.checked);
  }

  mirrorToggle.addEventListener('change', applyMirrorPreview);
  applyMirrorPreview();

  async function refreshCameraOptions() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) { return; }
    try {
      var selected = document.getElementById('camera-live').value;
      var devices = await navigator.mediaDevices.enumerateDevices();
      var selector = document.getElementById('camera-live');
      selector.innerHTML = '<option value="">Default camera</option>';
      devices.forEach(function (device, index) {
        if (device.kind !== 'videoinput') { return; }
        var option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || 'Camera ' + (index + 1);
        selector.appendChild(option);
      });
      if (selected) { selector.value = selected; }
    } catch (err) {
      setCamStatus('Could not list camera devices: ' + err.message, true);
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCamStatus('Camera is not available. Open this page with https or localhost.', true);
      return;
    }
    stopCamera(true);
    try {
      var requestedSize = document.getElementById('camera-size-live').value.split('x');
      var videoConstraints = {
        width: { ideal: Number(requestedSize[0]) },
        height: { ideal: Number(requestedSize[1]) },
        frameRate: { ideal: 24 }
      };
      var selectedCamera = document.getElementById('camera-live').value;
      if (selectedCamera) {
        videoConstraints.deviceId = { exact: selectedCamera };
      } else {
        videoConstraints.facingMode = { ideal: facing };
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false
      });
      await refreshCameraOptions();
      cam.srcObject = stream;
      cam.hidden = false;
      camRender.hidden = true;
      camPh.hidden = true;
      camBox.classList.add('live');
      captureBtn.disabled = false;
      camLabel.textContent = 'Stop camera';
      setCamStatus('Camera is on. Start the typer when you are ready.', false);
    } catch (err) {
      setCamStatus('Could not open the camera. Please allow camera permission.', true);
    }
  }

  function renderLiveFrame(frame) {
    if (!frame.hand_found) {
      renderResult(signResult, 'idle', 'NO HAND', 'Show your whole hand clearly in the camera view.');
    } else if (frame.held) {
      renderResult(signResult, 'ok', frame.held.toUpperCase(),
        'Hold the sign until the progress bar completes.', frame.held_confidence);
    } else if (frame.uncertain) {
      var runnerUp = frame.top_predictions && frame.top_predictions[1]
        ? frame.top_predictions[1].text : '-';
      var pairNote = frame.pair
        ? ' - ' + frame.pair[0] + '/' + frame.pair[1] + ' raised pair margin'
        : '';
      renderResult(signResult, 'warn', 'UNCERTAIN: ' + (frame.letter || '-'),
        'Only ' + (frame.margin * 100).toFixed(0) + ' points ahead of ' +
        runnerUp + ' (needs ' + (frame.required_margin * 100).toFixed(0) +
        ')' + pairNote + '; this frame casts no vote.', frame.confidence);
    } else if (frame.low_conf) {
      renderResult(signResult, 'warn', 'LOW CONFIDENCE: ' + (frame.letter || '-'),
        'Below the ' + Number(document.getElementById('conf-live').value).toFixed(2) +
        ' confidence gate; this frame casts no vote.', frame.confidence);
    } else {
      renderResult(signResult, 'idle', 'WAITING FOR A SIGN',
        frame.letter ? 'Raw best: ' + frame.letter + ' (' +
          (frame.confidence * 100).toFixed(0) + '%).' : '');
    }

    liveText.textContent = frame.text || '';
    refreshReadButton();
    liveTop.innerHTML = '';
    (frame.top_predictions || []).forEach(function (p) {
      var item = document.createElement('li');
      item.textContent = p.text.toUpperCase() + ': ' + (p.confidence * 100).toFixed(0) + '%';
      liveTop.appendChild(item);
    });
    if (!frame.top_predictions || !frame.top_predictions.length) {
      var empty = document.createElement('li');
      empty.className = 'muted';
      empty.textContent = frame.hand_found ? 'No model prediction.' : 'No hand detected.';
      liveTop.appendChild(empty);
    }
    if (frame.frame_image) {
      camRender.src = frame.frame_image;
      camRender.hidden = false;
    }
    document.getElementById('live-fps').textContent =
      Number(frame.fps || 0).toFixed(1) + ' fps';
    document.getElementById('live-hand-rate').textContent =
      Math.round(100 * (frame.hands || 0) / Math.max(frame.frames || 0, 1)) + '%';
    document.getElementById('live-hand-count').textContent =
      (frame.hands || 0) + ' of ' + (frame.frames || 0) + ' frames';
    document.getElementById('live-frame-ms').textContent =
      Number(frame.frame_ms || 0).toFixed(0) + ' ms';
    document.getElementById('live-backend').textContent = frame.backend || 'torch';
    document.getElementById('live-profile').textContent =
      (frame.model_profile || 'all') + ' profile';
    if (frame.calibration) {
      calibrationStatus.textContent = frame.calibration.prompt;
      calibrationProgress.value = frame.calibration.progress || 0;
      calibrationProgress.hidden = false;
      calibrationReport.hidden = true;
      cancelCalibrationBtn.disabled = false;
    } else if (frame.calibration_result) {
      var consistency = frame.calibration_result.consistency;
      var baseline = frame.calibration_result.baseline_consistency;
      calibrationStatus.textContent = 'Calibration complete: ' +
        frame.calibration_result.n_strokes + ' strokes; ' +
        consistency.fired + '/' + consistency.n +
        ' now fire their own letter. Thresholds apply only until the typer stops.';
      calibrationProgress.value = 1;
      calibrationProgress.hidden = true;
      calibrationReport.hidden = false;
      document.getElementById('calibration-summary').textContent =
        'Personal thresholds: ' + consistency.fired + '/' + consistency.n +
        ' strokes fire correctly (shipped thresholds: ' +
        baseline.fired + '/' + baseline.n + '). No recording or settings are saved.';
      var thresholds = document.getElementById('calibration-thresholds');
      thresholds.innerHTML = '';
      [
        ['Threshold', 'Shipped', 'This session', 'How it was derived']
      ].concat(Object.keys(frame.calibration_result.defaults || {}).sort().map(function (key) {
        return [
          key,
          Number(frame.calibration_result.defaults[key]).toFixed(2),
          Number(frame.calibration_result.thresholds[key]).toFixed(2),
          frame.calibration_result.notes[key] || ''
        ];
      })).forEach(function (row) {
        row.forEach(function (value) {
          var cell = document.createElement('div');
          cell.textContent = value;
          thresholds.appendChild(cell);
        });
      });
      var command = document.getElementById('calibration-command');
      command.textContent = frame.calibration_result.spec
        ? 'To reuse these thresholds in the console: python src/live.py --jz ' +
          frame.calibration_result.spec
        : 'Your strokes already fit the shipped thresholds; no console override is needed.';
      cancelCalibrationBtn.disabled = true;
    } else {
      calibrationStatus.textContent = frame.recovery || '';
      calibrationProgress.hidden = true;
      calibrationReport.hidden = true;
      cancelCalibrationBtn.disabled = !liveActive;
    }
    calibrateBtn.disabled = !liveActive ||
      !document.getElementById('motion-live').checked;
    clearLiveBtn.disabled = !liveActive;
    commitBtn.disabled = !liveActive;
    backspaceBtn.disabled = !liveActive;
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
        image: image
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
    var settings = liveSettings();
    mirrorToggle.disabled = true;
    try {
      var result = await modelRequest('/api/live/start', { settings: settings });
      if (runId !== liveRunId || !stream) {
        modelRequest('/api/live/stop', { session_id: result.session_id }).catch(function () {});
        return;
      }
      liveSession = result.session_id;
      liveActive = true;
      captureBtn.disabled = false;
      captureLabel.textContent = 'Stop live typer';
      clearLiveBtn.disabled = false;
      liveText.textContent = '';
      refreshReadButton();
      liveTop.innerHTML = '<li class="muted">Waiting for a sign.</li>';
      renderResult(signResult, 'idle', 'Waiting for a sign', 'Hold each letter steady until it is committed.');
      setCamStatus('Live typer is on. Space commits; Backspace deletes; C clears; Q or Esc stops.', false);
      nextLiveFrameAt = Date.now();
      captureLiveFrame(runId);
    } catch (err) {
      captureBtn.disabled = !stream;
      mirrorToggle.disabled = false;
      setCamStatus(err.message, true);
    }
  }

  async function stopLiveTranslation(saveHistory) {
    var sessionId = liveSession;
    var previousText = liveText ? liveText.textContent : '';
    liveActive = false;
    liveSession = null;
    liveRunId++;
    if (mirrorToggle) { mirrorToggle.disabled = false; }
    if (captureLabel) { captureLabel.textContent = 'Start live typer'; }
    if (captureBtn) { captureBtn.disabled = !stream; }
    if (clearLiveBtn) { clearLiveBtn.disabled = true; }
    if (commitBtn) { commitBtn.disabled = true; }
    if (backspaceBtn) { backspaceBtn.disabled = true; }
    if (calibrateBtn) { calibrateBtn.disabled = true; }
    if (cancelCalibrationBtn) { cancelCalibrationBtn.disabled = true; }
    refreshReadButton();
    if (!sessionId) { return; }

    var finalText = previousText;
    try {
      var result = await modelRequest('/api/live/stop', { session_id: sessionId });
      finalText = result.text || '';
      if (liveText) { liveText.textContent = finalText; }
      refreshReadButton();
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
    camRender.hidden = true;
    camRender.removeAttribute('src');
    camBox.classList.remove('live');
    if (captureBtn) { captureBtn.disabled = true; }
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
      setCamStatus('Live typer stopped.', false);
    } else {
      startLiveTranslation();
    }
  });
  async function sendLiveAction(action) {
    if (!liveSession) { return; }
    try {
      var result = await modelRequest('/api/live/action', {
        session_id: liveSession,
        action: action
      });
      renderLiveFrame(result);
    } catch (err) {
      setCamStatus(err.message, true);
    }
  }
  commitBtn.addEventListener('click', function () { sendLiveAction('commit'); });
  backspaceBtn.addEventListener('click', function () { sendLiveAction('backspace'); });
  clearLiveBtn.addEventListener('click', function () { sendLiveAction('clear'); });

  var speechAudio = null;
  var speechUrl = null;
  var speechLoading = false;

  function refreshReadButton() {
    readBtn.disabled = speechLoading || !(liveText.textContent || '').trim();
  }

  async function readTypedText() {
    var text = (liveText.textContent || '').trim();
    if (!text) {
      setCamStatus('There is no typed text to read yet.', true);
      return;
    }
    speechLoading = true;
    refreshReadButton();
    setCamStatus('Loading the Google voice...', false);
    try {
      var response;
      try {
        response = await fetch(API_BASE + '/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: API_CREDENTIALS,
          body: JSON.stringify({ text: text, lang: 'en' })
        });
      } catch (err) {
        throw new Error('Could not reach the speech API. Start the backend with "python app.py" in the Backend folder.');
      }
      if (!response.ok) {
        var message = 'Google TTS could not read the text.';
        try {
          var result = await response.json();
          if (result && result.error) { message = result.error; }
        } catch (err) {}
        throw new Error(message);
      }
      var blob = await response.blob();
      if (speechAudio) { speechAudio.pause(); }
      if (speechUrl) { URL.revokeObjectURL(speechUrl); }
      speechUrl = URL.createObjectURL(blob);
      var audio = new Audio(speechUrl);
      speechAudio = audio;
      audio.addEventListener('ended', function () {
        if (speechAudio === audio && speechUrl) {
          URL.revokeObjectURL(speechUrl);
          speechUrl = null;
        }
      });
      await audio.play();
      setCamStatus('Reading the typed text aloud.', false);
    } catch (err) {
      setCamStatus(err.message || 'Google TTS could not read the text.', true);
    } finally {
      speechLoading = false;
      refreshReadButton();
    }
  }
  readBtn.addEventListener('click', readTypedText);

  calibrateBtn.addEventListener('click', function () { sendLiveAction('calibrate'); });
  cancelCalibrationBtn.addEventListener('click', function () {
    sendLiveAction('cancel_calibration');
  });
  document.addEventListener('keydown', function (event) {
    if (!liveActive || event.altKey || event.ctrlKey || event.metaKey ||
        /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { return; }
    if (event.code === 'Space') {
      event.preventDefault();
      sendLiveAction('commit');
    } else if (event.key === 'Backspace') {
      event.preventDefault();
      sendLiveAction('backspace');
    } else if (event.key.toLowerCase() === 'c') {
      sendLiveAction('clear');
    } else if (event.key.toLowerCase() === 'q' || event.key === 'Escape') {
      stopLiveTranslation(true);
      setCamStatus('Live typer stopped.', false);
    }
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

  loadModelInfo();
});
