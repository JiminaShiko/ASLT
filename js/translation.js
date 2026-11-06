document.addEventListener('DOMContentLoaded', function () {
  var A = window.ASL;

  var userName = document.getElementById('user-name');
  if (userName) { userName.textContent = A.getProfile().name; }

  async function runPrediction(videoFile, model, filename) {
    if (!(videoFile instanceof Blob)) {
      return { error: 'Choose a video clip to translate.' };
    }
    try {
      var body = new FormData();
      body.append('video', videoFile, filename || videoFile.name || 'camera.webm');
      body.append('model', model);
      var response = await fetch('predict.php', { method: 'POST', body: body });
      var result = await response.json();
      if (!response.ok) {
        return { error: result.error || 'We could not translate this video. Please try again.' };
      }
      if (!result.predictions || !result.predictions.length) {
        return { error: 'The model returned no predictions. Please try another clip.' };
      }
      return {
        text: result.predictions[0].label,
        score: result.predictions[0].score
      };
    } catch (err) {
      console.error('Video prediction request failed:', err);
      return { error: 'Could not connect to the video model. Check the server setup and try again.' };
    }
  }

  /* ---------- Result box ---------- */
  function renderResult(el, state, text, note, score) {
    var ico = { idle: 'eye', ok: 'check', warn: 'alert' }[state] || 'eye';
    el.className = 'result' + (state === 'ok' || state === 'warn' ? ' ' + state : '');
    var boundedScore = typeof score === 'number' ? Math.max(0, Math.min(1, score)) : null;
    var meter = boundedScore !== null
      ? '<div class="note">Model score: ' + Math.round(boundedScore * 100) + '% (not calibrated confidence)</div>' +
        '<div class="meter" role="img" aria-label="Model score ' + Math.round(boundedScore * 100) + ' percent"><i style="width:' + Math.round(boundedScore * 100) + '%"></i></div>'
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
    renderResult(el, 'ok', r.text, '', r.score);
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
  var camPh = document.getElementById('cam-ph');
  var camToggle = document.getElementById('cam-toggle');
  var camLabel = document.getElementById('cam-toggle-label');
  var captureBtn = document.getElementById('capture');
  var flipBtn = document.getElementById('flip');
  var camStatus = document.getElementById('cam-status');
  var signResult = document.getElementById('sign-result');
  var stream = null;
  var activeRecorder = null;
  var facing = A.getProfile().camera === 'environment' ? 'environment' : 'user';
  var CAMERA_CLIP_DURATION_MS = 2250;

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
      cam.hidden = false; camPh.hidden = true;
      camBox.classList.add('live');
      captureBtn.disabled = false; flipBtn.disabled = false;
      camLabel.textContent = 'Stop camera';
      setCamStatus('Camera is on. Show one sign, then capture a short clip.', false);
    } catch (err) {
      setCamStatus('Could not open the camera. Please allow camera permission.', true);
    }
  }

  function stopCamera(keepStatus) {
    if (activeRecorder && activeRecorder.state === 'recording') { activeRecorder.stop(); }
    activeRecorder = null;
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    cam.srcObject = null; cam.hidden = true;
    camBox.classList.remove('live');
    captureBtn.disabled = true; flipBtn.disabled = true;
    camLabel.textContent = 'Start camera';
    camPh.hidden = false;
    if (!keepStatus) { setCamStatus('', false); }
  }

  camToggle.addEventListener('click', function () { if (stream) { stopCamera(); } else { startCamera(); } });
  flipBtn.addEventListener('click', function () {
    facing = (facing === 'user') ? 'environment' : 'user';
    startCamera();
  });

  function recordCameraClip() {
    return new Promise(function (resolve, reject) {
      if (!window.MediaRecorder) {
        reject(new Error('Video recording is not supported by this browser.'));
        return;
      }
      var formats = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
      var mimeType = formats.find(function (type) {
        return MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type);
      });
      var recorder;
      try {
        recorder = mimeType ? new MediaRecorder(stream, { mimeType: mimeType }) : new MediaRecorder(stream);
      } catch (error) {
        reject(new Error('This browser cannot record a camera clip.'));
        return;
      }
      activeRecorder = recorder;
      var chunks = [];
      recorder.ondataavailable = function (event) {
        if (event.data.size) { chunks.push(event.data); }
      };
      recorder.onerror = function () {
        if (activeRecorder === recorder) { activeRecorder = null; }
        reject(new Error('Could not record the camera clip. Please try again.'));
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

  captureBtn.addEventListener('click', async function () {
    if (!stream || !cam.videoWidth) { return; }
    captureBtn.disabled = true;
    setCamStatus('Recording a 2.25-second clip...', false);
    try {
      var clip = await recordCameraClip();
      if (!stream) { return; }
      if (!clip.size) { throw new Error('No camera video was recorded. Please try again.'); }
      setCamStatus('Analyzing your sign. This may take several seconds...', false);
      var extension = clip.type.indexOf('mp4') >= 0 ? 'mp4' : 'webm';
      var result = await runPrediction(
        clip,
        document.getElementById('camera-model-choice').value,
        'camera.' + extension
      );
      var ok = handlePrediction(result, 'Sign to Text', signResult);
      setCamStatus(ok ? 'Done.' : 'Try again with one sign centered in the frame.', !ok);
    } catch (error) {
      console.error('Camera video translation failed:', error);
      renderResult(signResult, 'idle', 'Your text will appear here', error.message);
      setCamStatus(error.message, true);
    } finally {
      if (stream) { captureBtn.disabled = false; }
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
  var fileVideo = document.getElementById('file-video');
  var drop = document.getElementById('drop');
  var currentFile = null, currentUrl = null;

  document.getElementById('pick-video').addEventListener('click', function () { fileVideo.click(); });

  function loadFile(file) {
    if (!file) { return; }
    var isVideo = file.type.indexOf('video/') === 0 ||
      /\.(avi|m4v|mkv|mov|mp4|webm)$/i.test(file.name);
    if (!isVideo) {
      currentFile = null;
      translateBtn.disabled = true;
      uploadStatus.textContent = 'The video model can only translate video clips. Choose a video to continue.';
      uploadStatus.className = 'status error';
      return;
    }
    currentFile = file;
    if (currentUrl) { URL.revokeObjectURL(currentUrl); }
    currentUrl = URL.createObjectURL(file);
    preview.innerHTML = '';
    var el = document.createElement('video');
    el.controls = true;
    el.playsInline = true;
    el.src = currentUrl;
    preview.appendChild(el);
    preview.hidden = false;
    translateBtn.disabled = file.size > 200 * 1024 * 1024;
    renderResult(uploadResult, 'idle', 'Your text will appear here');
    uploadStatus.textContent = file.size > 200 * 1024 * 1024
      ? 'This video exceeds the 200 MB upload limit.'
      : file.name;
    uploadStatus.className = translateBtn.disabled ? 'status error' : 'status';
  }
  fileVideo.addEventListener('change', function () { loadFile(fileVideo.files[0]); fileVideo.value = ''; });

  ['dragenter', 'dragover'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); });
  });
  drop.addEventListener('drop', function (e) { loadFile(e.dataTransfer.files[0]); });

  translateBtn.addEventListener('click', async function () {
    if (!currentFile) { return; }
    translateBtn.disabled = true;
    uploadStatus.textContent = 'Reading your sign...';
    uploadStatus.className = 'status';
    try {
      var r = await runPrediction(currentFile, document.getElementById('model-choice').value);
      var ok = handlePrediction(r, 'Upload Media', uploadResult);
      uploadStatus.textContent = ok ? 'Done.' : r.error;
      uploadStatus.className = ok ? 'status' : 'status error';
    } finally {
      translateBtn.disabled = !currentFile || currentFile.size > 200 * 1024 * 1024;
    }
  });

  /* ---------- History ---------- */
  var hlist = document.getElementById('hlist');
  var KIND_ICON = { 'Sign to Text': 'camera', 'Text to Sign': 'type', 'Upload Media': 'upload' };

  function fmt(t) {
    return new Date(t).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function renderHistory() {
    if (!hlist) { return; }
    var list = A.history.list();
    if (!list.length) {
      hlist.innerHTML = '<div class="empty">No translations yet. Your latest results will appear here.</div>';
      return;
    }
    hlist.innerHTML = list.map(function (h) {
      return '<div class="hitem' + (h.saved ? ' saved' : '') + '">' +
        '<div class="kind">' + A.icon(KIND_ICON[h.kind] || 'chat', 22) + '</div>' +
        '<div class="body"><b>' + A.esc(h.text) + '</b><small>' + A.esc(h.kind) + ', ' + fmt(h.t) + '</small></div>' +
        '<div class="acts">' +
        '<button class="icon-btn" type="button" data-act="save" data-id="' + h.id + '" aria-pressed="' + (h.saved ? 'true' : 'false') + '" aria-label="' + (h.saved ? 'Unsave' : 'Save') + ' ' + A.esc(h.text) + '">' + A.icon('bookmark', 20) + '</button>' +
        '<button class="icon-btn" type="button" data-act="del" data-id="' + h.id + '" aria-label="Delete ' + A.esc(h.text) + '">' + A.icon('trash', 20) + '</button>' +
        '</div></div>';
    }).join('');
  }

  if (hlist) {
    hlist.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-act]');
      if (!b) { return; }
      if (b.dataset.act === 'save') { A.history.toggleSave(b.dataset.id); }
      if (b.dataset.act === 'del') { A.history.remove(b.dataset.id); A.toast('Record deleted'); }
    });

    var clearHistory = document.getElementById('clear-history');
    if (clearHistory) {
      clearHistory.addEventListener('click', function () {
        var kept = A.history.clearUnsaved();
        A.toast(kept ? 'Cleared. Saved records were kept.' : 'History cleared');
      });
    }
  }

  function renderStats() {
    var translations = document.getElementById('st-trans');
    var signsLearned = document.getElementById('st-learn');
    var quizScore = document.getElementById('st-quiz');
    if (!translations || !signsLearned || !quizScore) { return; }
    translations.textContent = A.history.list().length;
    signsLearned.textContent = A.learned.list().length + ' / ' + A.learned.total();
    var best = A.quiz.best();
    quizScore.textContent = best ? best.score + '/' + best.total : 'None yet';
  }

  document.addEventListener('asl:update', function () { renderHistory(); renderStats(); });
  renderHistory();
  renderStats();
});
