/**
 * Chipset Defect Vision — Frontend Logic
 * ────────────────────────────────────────
 * Vanilla JS, async/await, zero dependencies.
 *
 * Architecture overview:
 *   • All DOM element references are grabbed once at load time (DOM Refs section).
 *   • State is tracked in four module-level variables (State section).
 *   • The INIT block at the bottom runs the startup sequence: loader → health poll.
 *
 * Data flow for a scan:
 *   User selects image (upload or camera) → handleFileSelected / captureFrame
 *     → showPreview()          (show image in left panel)
 *     → btnScan click          (user confirms)
 *     → runInference()         (builds FormData, POST /predict)
 *     → displayResults()       (render detections in right panel)
 *
 * Backend connection:
 *   All fetch() calls go to API_BASE + route (e.g. /health, /predict).
 *   API_BASE is empty string by default → same-origin requests served by FastAPI.
 *   Set API_BASE = 'http://localhost:8000' to point at a separately-running server.
 */

'use strict';

// ── Config ─────────────────────────────────────────────────────────────────
// API_BASE: prefix prepended to every fetch URL.  Empty = same origin as the
// frontend page (FastAPI serves both static files and API from the same port).
const API_BASE = '';
// How often the frontend pings GET /health to update the status indicator dot.
const HEALTH_INTERVAL = 30_000;   // ms between health-checks

// ── Label display mapping ───────────────────────────────────────────────────
// Maps raw model class names to user-facing display names.
// Does NOT affect API calls, model weights, or training data.
const labelMap = {
  missing_hole:        'Missing Hole',
  mouse_bite:          'Mouse Bite',
  open_circuit:        'Open Circuit',
  short:               'Solder Short',
  spur:                'Spur',
  spurious_copper:     'Spurious Copper',
  solder_bridge:       'Solder Bridge',
  excess_solder:       'Excess Solder',
  insufficient_solder: 'Insufficient Solder',
  good:                'Good',
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
// These four variables represent the complete UI state.  All event handlers
// read/write them to coordinate between the upload, camera, preview, and
// results sections.
let currentFile       = null;      // File object from file picker / drag-drop, or null
let currentDataUrl    = null;      // base64 data-URL string from camera capture, or null
let cameraStream      = null;      // Live MediaStream while camera is active, or null
let currentMode       = 'upload';  // 'upload' | 'camera' — which input panel is visible

// ══════════════════════════════════════════════════════════════ LOADER ══════
// The loader screen is shown on first page load while the browser fetches
// assets and the server warms up YOLO.  It animates a progress bar through
// LOADER_STEPS, then fades out and reveals the main app shell.
// This is purely cosmetic — actual server readiness is checked by checkHealth().
const LOADER_STEPS = [
  { pct: 10,  msg: 'Loading model weights...' },
  { pct: 35,  msg: 'Initializing inference engine...' },
  { pct: 60,  msg: 'Warming up YOLOv8 pipeline...' },
  { pct: 85,  msg: 'Checking server health...' },
  { pct: 100, msg: 'Ready.' },
];

async function runLoader() {
  // Advance progress bar through each step with a short pause
  for (let i = 0; i < LOADER_STEPS.length; i++) {
    const s = LOADER_STEPS[i];
    loaderBar.style.width    = s.pct + '%';
    loaderStatus.textContent = s.msg;
    // Last step gets a shorter pause before the fade-out begins
    await sleep(i === LOADER_STEPS.length - 1 ? 400 : 480);
  }
  await sleep(300);
  loader.classList.add('fade-out');  // CSS opacity transition (0.6s)
  await sleep(600);                  // wait for fade-out to complete
  loader.classList.add('hidden');    // remove from layout
  app.classList.remove('hidden');    // reveal the main app shell
  // Defer .visible to next paint frame so the CSS transition triggers
  requestAnimationFrame(() => app.classList.add('visible'));
}

// ══════════════════════════════════════════════════════════════ HEALTH ═══════
// checkHealth() polls GET /health (see HealthResponse schema in app/schemas.py).
// Called once after the loader completes, then every HEALTH_INTERVAL ms.
// Updates the status indicator dot in the header and the footer model label.
async function checkHealth() {
  try {
    // 10 s timeout — if the server is slow to start, we show "Offline" briefly
    const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(10_000) });
    const data = await res.json();
    if (data.status === 'ok') {
      setStatus('online', 'System Online');
      // Show version + model name in footer once model is confirmed loaded
      if (data.model_loaded) footerModel.textContent = `API ${data.version} | Model: ${data.model}`;
    } else {
      setStatus('offline', 'Degraded');   // server up but model not ready
    }
  } catch (e) {
    // Network error (server not running) or timeout
    setStatus('offline', `Offline — ${e.name}`);
  }
}

/**
 * Update the header status indicator dot and text.
 * @param {string} cls  - CSS class: 'online' | 'offline'
 * @param {string} text - Display text next to the dot
 */
function setStatus(cls, text) {
  statusDot.className    = 'status-dot ' + cls;
  statusText.textContent = text;
}

// ══════════════════════════════════════════════════════════════ MODE TOGGLE ═
// The Input Card has two panels: Upload and Camera.  switchMode() swaps
// between them and stops the camera stream if the user leaves Camera mode,
// preventing the browser from keeping the camera active in the background.
function switchMode(mode) {
  currentMode = mode;
  // Active button styling
  btnUploadMode.classList.toggle('active', mode === 'upload');
  btnCameraMode.classList.toggle('active', mode === 'camera');
  // Show/hide the corresponding panel
  uploadPanel.classList.toggle('hidden', mode !== 'upload');
  cameraPanel.classList.toggle('hidden', mode !== 'camera');
  // Release camera hardware when leaving camera mode
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
/**
 * Render a completed PredictionResponse (from POST /predict) into the Results Card.
 *
 * Layout updated:
 *   Left panel  — preview badge updated to "Scanned" (image stays visible)
 *   Right panel — results card shown with stats, annotated image, detection list
 *
 * @param {Object} data - PredictionResponse JSON parsed from the /predict response
 */
function displayResults(data) {
  // ── Extract fields with fallbacks for schema evolution ────────────────────
  // summary.* fields are preferred; top-level duplicates serve as fallbacks
  // for older API versions that may not include a summary block.
  const detections   = data.detections ?? [];
  const summary      = data.summary ?? {};
  const total        = summary.total        ?? data.total        ?? detections.length;
  const defect_count = summary.defect_count ?? data.defect_count ?? 0;
  const good_count   = summary.good_count   ?? data.good_count   ?? 0;
  // annotated_image_base64 is the primary key; 'image' is the legacy fallback
  const image = data.annotated_image_base64 ?? data.image;
  const model = data.model;

  // ── Stats row ─────────────────────────────────────────────────────────────
  statTotal.textContent  = total;
  statGood.textContent   = good_count;
  statDefect.textContent = defect_count;

  const quality = computeQuality(good_count, defect_count, total);
  statQuality.textContent = quality.label;
  statQuality.style.color = quality.color;

  // ── Annotated result image ────────────────────────────────────────────────
  // image is a plain base64 string — prefix with the data-URL scheme for img.src
  resultImg.src = `data:image/jpeg;base64,${image}`;

  // ── Footer model label ────────────────────────────────────────────────────
  if (model) footerModel.textContent = `Model: ${model}`;

  // ── Detection list ────────────────────────────────────────────────────────
  detectionsList.innerHTML = '';
  if (detections.length === 0) {
    // No boxes means the model found no regions above the confidence threshold
    detectionsList.innerHTML = '<p class="no-detections">No solder regions detected in this image.</p>';
  } else {
    detections.forEach((det, i) => {
      const item = document.createElement('div');
      // Colour coding: 'good' class gets green styling; all other labels are defects
      const cssClass    = det.label === 'Good' ? 'good' : 'defect';
      // labelMap translates raw model names (snake_case) to human-readable labels
      const displayLabel = labelMap[det.label] || det.label;
      item.className = `detection-item ${cssClass}`;
      const [x1,y1,x2,y2] = det.bbox;
      item.innerHTML = `
        <span class="di-num">#${String(i+1).padStart(2,'0')}</span>
        <span class="di-badge ${cssClass}">${displayLabel.toUpperCase()}</span>
        <span class="di-conf">${(det.confidence * 100).toFixed(1)}%</span>
        <span class="di-bbox">[${x1},${y1} → ${x2},${y2}]</span>
      `;
      // Stagger entry animation for a cascade reveal effect
      item.style.animationDelay = `${i * 60}ms`;
      detectionsList.appendChild(item);
    });
  }

  // ── Preview badge update ──────────────────────────────────────────────────
  previewBadge.textContent = 'Scanned';
  previewBadge.style.color = 'var(--green)';

  // Show results card in right panel; left panel preview stays visible
  resultsCard.classList.remove('hidden');
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Toast summary notification
  const msg = defect_count > 0
    ? `⚠ ${defect_count} defect${defect_count > 1 ? 's' : ''} detected`
    : '✓ No defects found';
  toast(msg, defect_count > 0 ? 'error' : 'success');
}

/**
 * Compute board quality rating from detection counts.
 *
 * Thresholds:
 *   0 % defects  → PASS (green)
 *   1–20 % defects → WARN (yellow) — borderline board, needs review
 *   > 20 % defects → FAIL (red)    — significant defect density
 *
 * @returns {{ label: string, color: string }}
 */
function computeQuality(good, defect, total) {
  if (total === 0)    return { label: 'N/A',  color: 'var(--text-muted)' };
  const ratio = defect / total;
  if (ratio === 0)    return { label: 'PASS', color: 'var(--green)' };
  if (ratio <= 0.2)   return { label: 'WARN', color: 'var(--yellow)' };
  return                     { label: 'FAIL', color: 'var(--red)' };
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
/**
 * Display a transient notification in the bottom-right corner.
 *
 * Toasts are appended to #toastContainer, auto-hidden after 3.5 s, then
 * removed from the DOM after the CSS fade-out transition completes.
 *
 * @param {string} msg  - Text to display
 * @param {string} type - 'info' | 'success' | 'error'
 */
function toast(msg, type = 'info') {
  const icons = { info: 'ℹ', success: '✓', error: '✕' };
  const el    = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] ?? 'ℹ'}</span><span class="toast-msg">${msg}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add('hiding');         // triggers CSS opacity/transform transition
    setTimeout(() => el.remove(), 320); // remove after transition completes
  }, 3500);                             // visible duration in ms
}

// ══════════════════════════════════════════════════════════════ UTILS ═════════
/** Promisified setTimeout — used to sequence async UI animations. */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ══════════════════════════════════════════════════════════════ INIT ══════════
// IIFE immediately invoked when the script loads (after DOM is ready via defer/end-of-body).
// Sequence:
//   1. runLoader() — animate the splash screen, then reveal the app
//   2. checkHealth() — first health poll; sets the status dot
//   3. setInterval(checkHealth, …) — continuous background health polling
(async function init() {
  await runLoader();
  await checkHealth();
  setInterval(checkHealth, HEALTH_INTERVAL);
})();
