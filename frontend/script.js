/**
 * Chipset Defect Vision — Frontend Logic
 * ────────────────────────────────────────
 * Vanilla JS, async/await, zero dependencies.
 */

'use strict';

// ── Config ─────────────────────────────────────────────────────────────────
const API_BASE = '';          // same origin; change to http://localhost:8080 for dev
const HEALTH_INTERVAL = 30_000;   // ms between health-checks

// ── Label display mapping ───────────────────────────────────────────────────
// Maps raw model class names to user-facing display names.
// Does NOT affect API calls, model weights, or training data.
const labelMap = {
  Solder_defect: 'Solder Defect',
};

// ── DOM Refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const loader          = $('loader');
const loaderBar       = $('loaderBar');
const loaderStatus    = $('loaderStatus');
const app             = $('app');

const statusDot       = $('statusDot');
const statusText      = $('statusText');

const btnUploadMode   = $('btnUploadMode');
const btnCameraMode   = $('btnCameraMode');
const uploadPanel     = $('uploadPanel');
const cameraPanel     = $('cameraPanel');

const dropZone        = $('dropZone');
const fileInput       = $('fileInput');

const cameraFeed      = $('cameraFeed');
const cameraViewport  = $('cameraViewport');
const cameraPlaceholder=$('cameraPlaceholder');
const captureCanvas   = $('captureCanvas');
const btnStartCamera  = $('btnStartCamera');
const btnCapture      = $('btnCapture');
const btnStopCamera   = $('btnStopCamera');

const inputCard       = $('inputCard');
const previewCard     = $('previewCard');
const previewImg      = $('previewImg');
const previewBadge    = $('previewBadge');
const btnScan         = $('btnScan');
const btnClearPreview = $('btnClearPreview');

const processingOverlay = $('processingOverlay');
const stepPre           = $('stepPre');
const stepInfer         = $('stepInfer');
const stepPost          = $('stepPost');

const resultsCard     = $('resultsCard');
const statTotal       = $('statTotal');
const statGood        = $('statGood');
const statDefect      = $('statDefect');
const statQuality     = $('statQuality');
const resultImg       = $('resultImg');
const detectionsList  = $('detectionsList');
const btnNewScan      = $('btnNewScan');
const footerModel     = $('footerModel');

const toastContainer  = $('toastContainer');

// ── State ──────────────────────────────────────────────────────────────────
let currentFile       = null;   // File object or null
let currentDataUrl    = null;   // base64 data URL
let cameraStream      = null;   // MediaStream or null
let currentMode       = 'upload';

// ══════════════════════════════════════════════════════════════ LOADER ══════
const LOADER_STEPS = [
  { pct: 10, msg: 'Loading model weights...' },
  { pct: 35, msg: 'Initializing inference engine...' },
  { pct: 60, msg: 'Warming up YOLOv8 pipeline...' },
  { pct: 85, msg: 'Checking server health...' },
  { pct: 100, msg: 'Ready.' },
];

async function runLoader() {
  for (let i = 0; i < LOADER_STEPS.length; i++) {
    const s = LOADER_STEPS[i];
    loaderBar.style.width   = s.pct + '%';
    loaderStatus.textContent = s.msg;
    await sleep(i === LOADER_STEPS.length - 1 ? 400 : 480);
  }
  await sleep(300);
  loader.classList.add('fade-out');
  await sleep(600);
  loader.classList.add('hidden');
  app.classList.remove('hidden');
  requestAnimationFrame(() => app.classList.add('visible'));
}

// ══════════════════════════════════════════════════════════════ HEALTH ═══════
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(10_000) });
    const data = await res.json();
    if (data.status === 'ok') {
      setStatus('online', 'System Online');
      if (data.model_loaded) footerModel.textContent = `API ${data.version} | Model: ${data.model}`;
    } else {
      setStatus('offline', 'Degraded');
    }
  } catch (e) {
    setStatus('offline', `Offline — ${e.name}`);
  }
}

function setStatus(cls, text) {
  statusDot.className  = 'status-dot ' + cls;
  statusText.textContent = text;
}

// ══════════════════════════════════════════════════════════════ MODE TOGGLE ═
function switchMode(mode) {
  currentMode = mode;
  btnUploadMode.classList.toggle('active', mode === 'upload');
  btnCameraMode.classList.toggle('active', mode === 'camera');
  uploadPanel.classList.toggle('hidden', mode !== 'upload');
  cameraPanel.classList.toggle('hidden', mode !== 'camera');
  if (mode !== 'camera' && cameraStream) stopCamera();
}

btnUploadMode.addEventListener('click', () => switchMode('upload'));
btnCameraMode.addEventListener('click', () => switchMode('camera'));

// ══════════════════════════════════════════════════════════════ UPLOAD ═══════
dropZone.addEventListener('click', e => {
  if (e.target !== fileInput) fileInput.click();
});

fileInput.addEventListener('change', e => {
  const file = e.target.files?.[0];
  if (file) handleFileSelected(file);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    handleFileSelected(file);
  } else {
    toast('Please drop a valid image file.', 'error');
  }
});

function handleFileSelected(file) {
  if (!file.type.startsWith('image/')) {
    toast('Invalid file type. Please select an image.', 'error');
    return;
  }
  currentFile    = file;
  currentDataUrl = null;
  const reader   = new FileReader();
  reader.onload  = e => showPreview(e.target.result);
  reader.readAsDataURL(file);
}

// ══════════════════════════════════════════════════════════════ CAMERA ════════
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Camera API not supported in this browser.', 'error');
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
    });
    cameraFeed.srcObject = cameraStream;
    await cameraFeed.play();

    cameraPlaceholder.classList.add('hidden');
    cameraViewport.classList.add('active');

    btnStartCamera.classList.add('hidden');
    btnCapture.classList.remove('hidden');
    btnStopCamera.classList.remove('hidden');
    toast('Camera started', 'success');
  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Camera permission denied. Please allow access.'
      : `Camera error: ${err.message}`;
    toast(msg, 'error');
  }
}

function captureFrame() {
  if (!cameraStream) return;
  const video  = cameraFeed;
  const canvas = captureCanvas;
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
  currentFile    = null;
  currentDataUrl = dataUrl;
  showPreview(dataUrl);
  toast('Frame captured', 'success');
}

function stopCamera() {
  cameraStream?.getTracks().forEach(t => t.stop());
  cameraStream = null;
  cameraFeed.srcObject = null;
  cameraViewport.classList.remove('active');
  cameraPlaceholder.classList.remove('hidden');
  btnStartCamera.classList.remove('hidden');
  btnCapture.classList.add('hidden');
  btnStopCamera.classList.add('hidden');
}

btnStartCamera.addEventListener('click', startCamera);
btnCapture.addEventListener('click',     captureFrame);
btnStopCamera.addEventListener('click',  stopCamera);

// ══════════════════════════════════════════════════════════════ PREVIEW ════════
function showPreview(dataUrl) {
  previewImg.src        = dataUrl;
  previewBadge.textContent = 'Ready to scan';
  previewBadge.style.color = '';
  previewCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
  previewCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

btnClearPreview.addEventListener('click', () => {
  currentFile    = null;
  currentDataUrl = null;
  fileInput.value = '';
  previewCard.classList.add('hidden');
  resultsCard.classList.add('hidden');
});

// ══════════════════════════════════════════════════════════════ SCAN ══════════
btnScan.addEventListener('click', async () => {
  if (!currentFile && !currentDataUrl) {
    toast('No image selected.', 'error');
    return;
  }
  await runInference();
});

async function runInference() {
  showProcessing(true);
  animateStep('pre');

  try {
    let body;
    if (currentFile) {
      body = new FormData();
      body.append('file', currentFile);
    } else {
      body = new FormData();
      // Strip data-URL prefix → raw base64
      const b64 = currentDataUrl.includes(',')
        ? currentDataUrl.split(',')[1]
        : currentDataUrl;
      body.append('image_base64', b64);
    }

    await sleep(600);
    animateStep('infer');

    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body,
      signal: AbortSignal.timeout(60_000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Server error');
    }

    const data = await res.json();
    await sleep(400);
    animateStep('post');
    await sleep(500);

    showProcessing(false);
    displayResults(data);

  } catch (err) {
    showProcessing(false);
    const msg = err.name === 'TimeoutError'
      ? 'Request timed out. Is the server running?'
      : err.name === 'TypeError'
        ? `Network error — is the server running? (${err.message})`
        : err.message;
    toast(msg, 'error');
    previewBadge.textContent = 'Scan failed';
    previewBadge.style.color = 'var(--red)';
  }
}

function animateStep(step) {
  [stepPre, stepInfer, stepPost].forEach(el => el.classList.remove('active', 'done'));
  const map = { pre: [stepPre], infer: [stepPre, stepInfer], post: [stepPre, stepInfer, stepPost] };
  const all = [stepPre, stepInfer, stepPost];
  const active = { pre: stepPre, infer: stepInfer, post: stepPost }[step];
  all.forEach((el, i) => {
    if (map[step].includes(el)) {
      el.classList.add(el === active ? 'active' : 'done');
    }
  });
}

function showProcessing(show) {
  processingOverlay.classList.toggle('hidden', !show);
  if (show) {
    [stepPre, stepInfer, stepPost].forEach(el => el.classList.remove('active','done'));
  }
}

// ══════════════════════════════════════════════════════════════ RESULTS ══════
function displayResults(data) {
  const detections = data.detections ?? [];
  const summary = data.summary ?? {};
  const total = summary.total ?? data.total ?? detections.length;
  const defect_count = summary.defect_count ?? data.defect_count ?? 0;
  const good_count = summary.good_count ?? data.good_count ?? 0;
  const image = data.annotated_image_base64 ?? data.image;
  const model = data.model;

  // Stats
  statTotal.textContent  = total;
  statGood.textContent   = good_count;
  statDefect.textContent = defect_count;

  const quality = computeQuality(good_count, defect_count, total);
  statQuality.textContent        = quality.label;
  statQuality.style.color        = quality.color;

  // Annotated image
  resultImg.src = `data:image/jpeg;base64,${image}`;

  // Footer
  if (model) footerModel.textContent = `Model: ${model}`;

  // Detection list
  detectionsList.innerHTML = '';
  if (detections.length === 0) {
    detectionsList.innerHTML = '<p class="no-detections">No solder regions detected in this image.</p>';
  } else {
    detections.forEach((det, i) => {
      const item = document.createElement('div');
      const cssClass = det.label === 'Good' ? 'good' : 'defect'; // red for all non-Good labels
      const displayLabel = labelMap[det.label] || det.label;      // apply display mapping
      item.className = `detection-item ${cssClass}`;
      const [x1,y1,x2,y2] = det.bbox;
      item.innerHTML = `
        <span class="di-num">#${String(i+1).padStart(2,'0')}</span>
        <span class="di-badge ${cssClass}">${displayLabel.toUpperCase()}</span>
        <span class="di-conf">${(det.confidence * 100).toFixed(1)}%</span>
        <span class="di-bbox">[${x1},${y1} → ${x2},${y2}]</span>
      `;
      item.style.animationDelay = `${i * 60}ms`;
      detectionsList.appendChild(item);
    });
  }

  previewBadge.textContent = 'Scanned';
  previewBadge.style.color = 'var(--green)';

  // Show result in right panel — preview stays visible in left panel
  resultsCard.classList.remove('hidden');
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const msg = defect_count > 0
    ? `⚠ ${defect_count} defect${defect_count > 1 ? 's' : ''} detected`
    : '✓ No defects found';
  toast(msg, defect_count > 0 ? 'error' : 'success');
}

function computeQuality(good, defect, total) {
  if (total === 0) return { label: 'N/A', color: 'var(--text-muted)' };
  const ratio = defect / total;
  if (ratio === 0)      return { label: 'PASS', color: 'var(--green)' };
  if (ratio <= 0.2)     return { label: 'WARN', color: 'var(--yellow)' };
  return                       { label: 'FAIL', color: 'var(--red)' };
}

// ══════════════════════════════════════════════════════════════ NEW SCAN ═════
btnNewScan.addEventListener('click', () => {
  resultsCard.classList.add('hidden');
  previewCard.classList.add('hidden');
  currentFile    = null;
  currentDataUrl = null;
  fileInput.value = '';
  previewBadge.textContent = 'Ready to scan';
  previewBadge.style.color = '';
  inputCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

// ══════════════════════════════════════════════════════════════ TOAST ════════
function toast(msg, type = 'info') {
  const icons = { info: 'ℹ', success: '✓', error: '✕' };
  const el    = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] ?? 'ℹ'}</span><span class="toast-msg">${msg}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add('hiding');
    setTimeout(() => el.remove(), 320);
  }, 3500);
}

// ══════════════════════════════════════════════════════════════ UTILS ═════════
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ══════════════════════════════════════════════════════════════ INIT ══════════
(async function init() {
  await runLoader();
  await checkHealth();
  setInterval(checkHealth, HEALTH_INTERVAL);
})();
