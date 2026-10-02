const $ = (id) => document.getElementById(id);
const state = {
  audioWs: null,
  devices: null,
  audioCtx: null,
  micWorkletNode: null,
  micPlayer: null,
  micScriptPlayer: null,
  micWorkletReady: null,
  micWorkletFailed: false,
  micWorkletFormatKey: "",
  toneSending: false,
  filePlaying: false,
  fileStop: false,
  fileTx: null,
  spkResetting: false,
  spkRefillBudget: 0,
  spkBudgetWaitSince: 0,
  spkBufferWaitSince: 0,
  spkLastBudgetWaitLog: 0,
  spkLastBufferWaitLog: 0,
  micFrames: 0,
  micLevel: 0,
  cameraWs: null,
  cameraObjectUrl: null,
  cameraStreamActive: false,
  cameraStopRequested: false,
  cameraFirstFrameLogged: false,
  cameraLastConnected: false,
  cameraLastEnabled: false,
  cropEnabled: false,
  cameraCropBusy: false,
  crop: { left: 0, top: 0, width: 100, height: 100 },
  // Mirrors the constants in main/app_ai_patch.h. The red AI crop is fixed
  // to a 40-pixel grid and is recalculated from the actual source resolution.
  // It cannot be dragged or resized.
  aiPatchSize: 40,
  // Placeholder defaults only, overwritten as soon as the device's /ai/patch
  // status JSON is fetched (see max_cols/max_rows below) - kept in sync with
  // APP_AI_PATCH_MAX_COLS/ROWS in main/app_ai_patch.h just so the very first
  // render (before that fetch resolves) isn't obviously wrong.
  aiPatchMaxCols: 20,
  aiPatchMaxRows: 15,
  // blob: URLs created for the currently-displayed patch grid, so the next
  // render can revoke them (avoids leaking memory every time P4 is pressed).
  aiPatchObjectUrls: [],
  aiInferObjectUrl: null,
  aiSessionCount: 0,
  aiSessionState: "idle",
  aiCoresetObjectUrl: null,
};
const SPK_MAX_PAYLOAD = 4092;
const SPK_WS_BUFFER_LIMIT = 65536;
const CAMERA_HEADER_SIZE = 16;
const CAMERA_FORMAT_MJPEG = 1;

function log(message) {
  const el = $("log");
  const time = new Date().toLocaleTimeString();
  el.textContent = `[${time}] ${message}\n` + el.textContent;
}

window.addEventListener("error", (event) => {
  log(`JS error: ${event.message}`);
});

window.addEventListener("unhandledrejection", (event) => {
  log(`Promise error: ${event.reason && event.reason.message ? event.reason.message : event.reason}`);
});

function sendAudioJson(obj) {
  if (state.audioWs && state.audioWs.readyState === WebSocket.OPEN) {
    state.audioWs.send(JSON.stringify(obj));
  }
}

function setPill(el, text, cls) {
  el.className = `pill ${cls || ""}`.trim();
  el.textContent = text;
}

function formatLabel(alt) {
  const freqs = alt.sample_freq_type === 0 ? `${alt.freqs[0]}-${alt.freqs[1]} Hz` : `${alt.freqs.join(",")} Hz`;
  return `Alt ${alt.alt}: ${alt.channels}ch ${alt.bit_resolution}bit ${freqs}`;
}

function fillFormats(select, stream) {
  select.innerHTML = "";
  if (!stream || !stream.connected) return;
  for (const alt of stream.alts) {
    const freqs = alt.sample_freq_type === 0 ? [stream.sample_freq || alt.freqs[0]] : alt.freqs;
    for (const freq of freqs) {
      const option = document.createElement("option");
      option.value = `${alt.alt}:${freq}`;
      option.textContent = formatLabel({ ...alt, freqs: [freq] });
      option.selected = alt.alt === stream.selected_alt && freq === stream.sample_freq;
      select.appendChild(option);
    }
  }
}

function updateControls(streamName, stream) {
  const enabled = $(`${streamName}Enabled`);
  const mute = $(`${streamName}Mute`);
  const volume = $(`${streamName}Volume`);
  const format = $(`${streamName}Format`);
  const connected = stream && stream.connected;
  enabled.disabled = !connected;
  mute.disabled = !connected || !stream.mute_supported;
  volume.disabled = !connected || !stream.volume_supported;
  format.disabled = !connected || stream.enabled;
  enabled.textContent = stream.enabled ? "Close" : "Open";
  enabled.dataset.enabled = stream.enabled ? "1" : "0";
  enabled.classList.toggle("is-open-action", connected && !stream.enabled);
  enabled.classList.toggle("is-close-action", connected && stream.enabled);
  mute.checked = !!stream.mute;
  volume.value = stream.volume || 80;
  fillFormats(format, stream);
}

function cameraFormatByName(camera, name) {
  if (!camera || !Array.isArray(camera.formats)) return null;
  return camera.formats.find((item) => item.format === name) || null;
}

function currentCameraFormat(camera) {
  return cameraFormatByName(camera, camera && camera.selected_format) || cameraFormatByName(camera, "mjpeg");
}

function fillCameraFormats(camera) {
  const select = $("cameraFormat");
  select.innerHTML = "";
  const formats = camera && Array.isArray(camera.formats) ? camera.formats : [];
  for (const item of formats) {
    const option = document.createElement("option");
    option.value = item.format;
    option.textContent = item.format.toUpperCase();
    option.selected = item.format === camera.selected_format;
    select.appendChild(option);
  }
  const h264 = document.createElement("option");
  h264.value = "h264";
  h264.textContent = "H.264 (unsupported)";
  h264.disabled = true;
  if (!formats.some((item) => item.format === "h264")) select.appendChild(h264);
}

function fillCameraResolutions(camera) {
  const select = $("cameraResolution");
  select.innerHTML = "";
  const format = currentCameraFormat(camera);
  const resolutions = format && Array.isArray(format.resolutions) ? format.resolutions : [];
  for (const item of resolutions) {
    const option = document.createElement("option");
    option.value = String(item.index);
    option.textContent = `${item.width}x${item.height}@${Math.round(item.fps || 0)}fps`;
    option.selected = item.index === camera.selected_resolution;
    select.appendChild(option);
  }
}

function stopCameraStream(text) {
  state.cameraStopRequested = true;
  if (state.cameraWs) {
    state.cameraWs.close();
    state.cameraWs = null;
  }
  if (state.cameraObjectUrl) {
    URL.revokeObjectURL(state.cameraObjectUrl);
    state.cameraObjectUrl = null;
  }
  const img = $("cameraStream");
  img.removeAttribute("src");
  img.classList.remove("active");
  const canvas = $("cameraCanvas");
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("cameraPlaceholder").textContent = text || "Camera disabled";
  $("cameraPlaceholder").classList.remove("hidden");
  state.cameraStreamActive = false;
  state.cameraFirstFrameLogged = false;
  updateCropOverlay();
}

// Derive the fixed AI crop from the real camera source resolution. The crop
// dimensions are always multiples of 40, so the patch count scales with
// resolution (640x480 -> 192 patches, 800x600 -> 300 patches, ...). When a
// mode is not divisible by 40, keep the largest centered rectangle that is
// divisible by 40. The AI pool is only capped at the device-reported
// max_cols x max_rows safety ceiling (see APP_AI_PATCH_MAX_COLS/ROWS).
function computeFixedCropPx(sourceW, sourceH) {
  const size = state.aiPatchSize || 40;
  const maxW = (state.aiPatchMaxCols || 20) * size;
  const maxH = (state.aiPatchMaxRows || 15) * size;
  let w = Math.floor(sourceW / size) * size;
  let h = Math.floor(sourceH / size) * size;
  w = Math.min(w, maxW);
  h = Math.min(h, maxH);
  if (w < size || h < size) return null;
  const x = Math.floor((sourceW - w) / 2);
  const y = Math.floor((sourceH - h) / 2);
  return { x, y, width: w, height: h, cols: w / size, rows: h / size, count: (w / size) * (h / size) };
}

function applyFixedCropFromImage() {
  const img = $("cameraStream");
  if (!img || !img.naturalWidth || !img.naturalHeight) return null;
  const fixed = computeFixedCropPx(img.naturalWidth, img.naturalHeight);
  if (!fixed) return null;
  state.crop = {
    left: (fixed.x / img.naturalWidth) * 100,
    top: (fixed.y / img.naturalHeight) * 100,
    width: (fixed.width / img.naturalWidth) * 100,
    height: (fixed.height / img.naturalHeight) * 100,
  };
  return fixed;
}

function updateCropReadout() {
  const el = $("cropReadout");
  if (!el) return;
  const img = $("cameraStream");
  const fixed = img && img.naturalWidth && img.naturalHeight ? computeFixedCropPx(img.naturalWidth, img.naturalHeight) : null;
  if (fixed) {
    el.textContent = `Crop CO DINH: ${fixed.width}x${fixed.height}px • ${fixed.cols}x${fixed.rows} = ${fixed.count} patch 40x40 • source ${img.naturalWidth}x${img.naturalHeight}`;
  } else {
    el.textContent = `Crop CO DINH: chờ nhận được độ phân giải camera`;
  }
}

// Computes the actual on-screen rectangle of the <img>, since it is displayed with
// object-fit: contain and may be letterboxed inside the (fixed aspect-ratio) container.
function computeContainRect(containerW, containerH, naturalW, naturalH) {
  if (!containerW || !containerH || !naturalW || !naturalH) {
    return { x: 0, y: 0, w: containerW || 0, h: containerH || 0 };
  }
  const containerRatio = containerW / containerH;
  const naturalRatio = naturalW / naturalH;
  let w, h;
  if (naturalRatio > containerRatio) {
    w = containerW;
    h = containerW / naturalRatio;
  } else {
    h = containerH;
    w = containerH * naturalRatio;
  }
  return { x: (containerW - w) / 2, y: (containerH - h) / 2, w, h };
}

// Reproduces app_ai_patch.c's grid math exactly (1 cell per ~aiPatchSize
// source pixels, clamped to at least 1 and at most aiPatchMax{Cols,Rows}),
// given the crop rectangle's size in *source* pixels. Keeping this in one
// place means the live preview grid and the actual on-device cut can never
// drift apart.
function computeAiPatchGrid(cropWpx, cropHpx) {
  const size = state.aiPatchSize || 40;
  const maxCols = state.aiPatchMaxCols || 20;
  const maxRows = state.aiPatchMaxRows || 15;
  const cols = Math.min(maxCols, Math.floor(cropWpx / size));
  const rows = Math.min(maxRows, Math.floor(cropHpx / size));
  return { cols: Math.max(0, cols), rows: Math.max(0, rows) };
}
// Draws the yellow cols x rows grid lines inside the fixed red crop box so you can
// see, live, exactly how the source frame will be diced into 40x40 patches.
// The geometry is recalculated only from the camera source resolution.
function updateCropGridLines() {
  const lines = $("cropGridLines");
  if (!lines) return;
  const img = $("cameraStream");
  if (!state.cropEnabled || !img.classList.contains("active") || !img.naturalWidth || !img.naturalHeight) {
    lines.style.backgroundImage = "none";
    return;
  }
  applyFixedCropFromImage();
  updateCropReadout();
  const fixed = computeFixedCropPx(img.naturalWidth, img.naturalHeight);
  if (!fixed) { lines.style.backgroundImage = "none"; return; }
  const { cols, rows } = computeAiPatchGrid(fixed.width, fixed.height);
  lines.style.backgroundImage =
    "linear-gradient(to right, rgba(255,210,0,.85) 1px, transparent 1px)," +
    "linear-gradient(to bottom, rgba(255,210,0,.85) 1px, transparent 1px)";
  lines.style.backgroundSize = `${100 / cols}% 100%, 100% ${100 / rows}%`;
}

// Positions the fixed red AI crop box over the currently displayed image.
// The box geometry is derived from the camera source resolution and never edited
// by pointer interaction.
function updateCropOverlay() {
  const overlay = $("cropOverlay");
  const view = $("cameraView");
  const img = $("cameraStream");
  const box = $("cropBox");
  if (!state.cropEnabled || !img.classList.contains("active")) {
    overlay.classList.add("hidden");
    updateCropGridLines();
    return;
  }
  applyFixedCropFromImage();
  overlay.classList.remove("hidden");
  const rect = computeContainRect(view.clientWidth, view.clientHeight, img.naturalWidth, img.naturalHeight);
  state.cropImgRect = rect;
  box.style.left = `${rect.x + (state.crop.left / 100) * rect.w}px`;
  box.style.top = `${rect.y + (state.crop.top / 100) * rect.h}px`;
  box.style.width = `${(state.crop.width / 100) * rect.w}px`;
  box.style.height = `${(state.crop.height / 100) * rect.h}px`;
  updateCropGridLines();
}

// Draws only the selected sub-rectangle of the decoded frame into the preview canvas
// at native pixel resolution (no pre-scale blur), then CSS scales the smaller canvas
// up to fill the preview box — this is what makes the cropped area look sharper than
// the full, stretched frame.
function drawCroppedFrame(bitmap) {
  const canvas = $("cameraCanvas");
  const cw = bitmap.width;
  const ch = bitmap.height;
  const fixed = computeFixedCropPx(cw, ch);
  if (!fixed) return;
  const sx = fixed.x;
  const sy = fixed.y;
  const sw = fixed.width;
  const sh = fixed.height;
  if (canvas.width !== sw || canvas.height !== sh) {
    canvas.width = sw;
    canvas.height = sh;
  }
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
}

// Shows the results of a finished ESP32-side capture (see
// app_ai_patch_capture() in app_ai_patch.c): the exact crop-rectangle photo
// it was cut from (served fresh from /api/ai_patch/photo) and the 40x40
// patch grid (/api/ai_patch/patch?index=N) - both guaranteed to come from
// that SAME on-device frame, so the "Anh da chup" panel and the patches
// below it can never show two different moments in time. `data` is the
// parsed JSON from either /api/ai_patch/capture or /api/ai_patch/status.
function showAiCaptureResult(data, fetchInfer = true) {
  const stamp = Date.now();
  const img = $("cropCapturedImg");
  img.src = `/api/ai_patch/photo?t=${stamp}`;
  $("cropCaptureWrap").classList.add("active");

  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const filename = `crop_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.bmp`;
  const link = $("cropCapturedDownload");
  link.href = `/api/ai_patch/photo?t=${stamp}`;
  link.download = filename;

  const readout = $("aiPatchReadout");
  if (readout) {
    readout.textContent = `OK: luoi ${data.cols}x${data.rows} = ${data.count} patch, moi patch ${data.patch_size}x${data.patch_size}px (nguon ${data.source_width}x${data.source_height})`;
  }
  renderAiPatchGrid(data.cols, data.rows, data.count);
  if (fetchInfer) fetchAndRenderAiInfer();
}

// Reads back whatever /api/ai_infer/features currently holds - the 32-d
// feature vector app_ai_infer.c computed for each patch, one model run per
// patch, right after the last successful capture (see main.c /
// ai_patch_capture_handler() in app_web.c). Like showLastAiCapture(), this
// only *displays* the result; it never triggers the model itself.
async function fetchAndRenderAiInfer() {
  const readout = $("aiInferReadout");
  try {
    const res = await fetch("/api/ai_infer/features");
    const data = await res.json();
    if (!data || !data.ready) {
      if (readout) readout.textContent = "Model dac trung chua san sang tren ESP32 (kiem tra log khoi dong).";
      renderAiInferTable(null);
      return;
    }
    if (!data.ok || !data.vectors || !data.vectors.length) {
      const reason = (data && data.error) || "chua co vecto nao";
      if (readout) readout.textContent = `Chua co vecto dac trung (${reason}).`;
      renderAiInferTable(null);
      return;
    }
    if (readout) {
      const perPatchMs = data.count ? (data.last_run_ms / data.count).toFixed(2) : "0";
      readout.textContent = `OK: ${data.count} vecto ${data.feature_dim}-chieu, tong ${data.last_run_ms.toFixed(1)}ms (~${perPatchMs}ms/patch)`;
    }
    renderAiInferTable(data.vectors);
    log(`Trich xuat dac trung OK: ${data.count} vecto ${data.feature_dim}-chieu`);
  } catch (err) {
    if (readout) readout.textContent = `Tai vecto dac trung that bai: ${err.message}`;
    renderAiInferTable(null);
    log(`Tai vecto dac trung that bai: ${err.message}`);
  }
}

// Renders one row per patch, one column per embedding dimension. `vectors`
// is the raw array-of-float-arrays from /api/ai_infer/features (or null to
// clear the table). Also refreshes the "Tai vecto dac trung (JSON)"
// download link with a fresh Blob of exactly what's shown.
function renderAiInferTable(vectors) {
  const head = $("aiInferTableHead");
  const body = $("aiInferTableBody");
  const downloadLink = $("aiInferDownload");
  if (!head || !body) return;

  if (state.aiInferObjectUrl) {
    URL.revokeObjectURL(state.aiInferObjectUrl);
    state.aiInferObjectUrl = null;
  }
  if (downloadLink) downloadLink.classList.add("hidden");

  head.innerHTML = "";
  body.innerHTML = "";
  if (!vectors || !vectors.length) {
    return;
  }

  const dim = vectors[0].length;
  const headRow = document.createElement("tr");
  const cornerTh = document.createElement("th");
  cornerTh.textContent = "Patch";
  headRow.appendChild(cornerTh);
  for (let d = 0; d < dim; d++) {
    const th = document.createElement("th");
    th.textContent = `d${d}`;
    headRow.appendChild(th);
  }
  head.appendChild(headRow);

  vectors.forEach((vec, i) => {
    const row = document.createElement("tr");
    const idxTd = document.createElement("td");
    idxTd.textContent = `#${i + 1}`;
    row.appendChild(idxTd);
    for (let d = 0; d < dim; d++) {
      const td = document.createElement("td");
      td.textContent = vec[d].toFixed(3);
      row.appendChild(td);
    }
    body.appendChild(row);
  });

  if (downloadLink) {
    const blob = new Blob([JSON.stringify(vectors, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    state.aiInferObjectUrl = url;
    downloadLink.href = url;
    downloadLink.classList.remove("hidden");
  }
}

// NOTE: multi-photo capture is intentionally NOT triggerable from the web page.
// P4 (GPIO4) controls capture and P3 (GPIO3) starts processing. That removes the
// confusing "Chup anh (P4)" web button that actually fired its own HTTP
// capture independent of the physical button, which is what made "capture
// works from the web but not from the physical button" look like a bug -
// they were really two different trigger paths. The web page now only ever
// *displays* whatever the ESP32 last captured (showLastAiCapture() below),
// either because the physical button's WebSocket notification arrived, or
// because the user pressed "Xem lai anh & patch da chup (P4)".

// Loads and displays whatever the ESP32's LAST successful capture already
// produced (photo + patches) WITHOUT triggering a new one. This is the
// function both:
//  - the physical P4 button notification calls (the device has already
//    finished capturing by the time that notification arrives - see
//    main.c), and
//  - the "Cat patch 40x40 (ESP32)" button calls, so it cuts patches out of
//    the photo that was actually captured (by "Chup anh (P4)" or the
//    physical button) instead of quietly grabbing a brand new live frame
//    of its own. That mismatch - patches coming from a fresh, independent
//    capture instead of the photo just shown - was exactly the bug: it
//    made the crop look like it kept following the live camera instead of
//    staying locked to the captured photo.
async function showLastAiCapture(logPrefix, fetchInfer = true) {
  const readout = $("aiPatchReadout");
  const btn = $("aiPatchCapture");
  if (btn) btn.disabled = true;
  if (readout) readout.textContent = "Dang tai ket qua tu ESP32...";
  try {
    const res = await fetch("/api/ai_patch/status");
    const data = await res.json();
    if (!data || !data.ok) {
      const reason = (data && data.error) || "chua co anh nao duoc chup";
      if (readout) {
        readout.textContent = `Chua co patch (${reason}). Hay bam nut vat ly P4 tren ESP32 (noi GPIO4 xuong GND) de chup truoc.`;
      }
      log(`${logPrefix || "Cat patch"} that bai: ${reason}`);
      return;
    }
    showAiCaptureResult(data, fetchInfer);
    if (!fetchInfer) {
      const inferReadout = $("aiInferReadout");
      if (inferReadout) inferReadout.textContent = "Đã chụp ảnh mới; chưa chạy model. Bấm P3 để trích xuất vector cho toàn bộ phiên.";
      renderAiInferTable(null);
      const sessionDownload = $("aiSessionDownload");
      if (sessionDownload) sessionDownload.classList.add("hidden");
    }
    log(`${logPrefix || "Cat patch"} OK: ${data.count} patch (${data.cols}x${data.rows}) tu anh da chup, khong chup lai frame moi`);
  } catch (err) {
    if (readout) readout.textContent = `Tai ket qua that bai: ${err.message}`;
    log(`Tai ket qua patch that bai: ${err.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Tells the ESP32 to grab its own frame straight from the USB camera,
// decode the JPEG on-device, cut the current crop rectangle into a grid of
// tiles and resize each one to exactly 40x40 - ready to feed into the AI
// model later. No model runs yet; this only produces/stores the patches on
// the device. Sends the up-to-date crop box first so results match what's
// shown on screen even if a drag hasn't finished syncing yet.
// Renders one <img> per captured patch, each pointing at the ESP32's
// /api/ai_patch/patch?index=N endpoint (see app_web.c). A cache-busting
// timestamp is added so a second capture doesn't just re-show the browser's
// cached copy of the same index from the previous capture.
//
// Patches come back from the device in row-major order over a `cols` x
// `rows` grid (index = row * cols + col - see app_ai_patch.c). This sets
// the CSS grid to that SAME column count (instead of letting it auto-wrap
// to whatever fits the panel width) so patch N+1 always renders directly
// to the right of patch N when they're really neighbours in the source
// image, and wraps to a new row exactly when the device did. Getting this
// wrong is what made an already-correct crop look scrambled before.
// How many /api/ai_patch/patch requests are allowed in flight at once.
// The ESP32's httpd only keeps a handful of TCP sockets open at a time
// (ESP-IDF's default max_open_sockets), and it is also busy serving the
// live MJPEG WebSocket at the same time. Setting every <img>'s src at once
// (up to 100 for a 10x10 grid) fired that many simultaneous connections -
// the device could only actually service a few of them, so whichever
// requests didn't get a free socket in time came back as browser
// connection errors, which is exactly the "cho mat cho khong" (some
// patches show, some don't) symptom. Loading a handful at a time (with a
// couple of retries for the rare timeout) fixes that without touching the
// device's crop/grid math, which was already correct.
const AI_PATCH_LOAD_CONCURRENCY = 2;
const AI_PATCH_LOAD_RETRIES = 2;

function revokeAiPatchObjectUrls() {
  for (const url of state.aiPatchObjectUrls) {
    URL.revokeObjectURL(url);
  }
  state.aiPatchObjectUrls = [];
}

async function fetchAiPatchImageUrl(index, stamp) {
  let lastErr;
  for (let attempt = 0; attempt <= AI_PATCH_LOAD_RETRIES; attempt++) {
    try {
      const res = await fetch(`/api/ai_patch/patch?index=${index}&t=${stamp}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const blob = await res.blob();
      return URL.createObjectURL(blob);
    } catch (err) {
      lastErr = err;
      if (attempt < AI_PATCH_LOAD_RETRIES) {
        await new Promise((resolve) => setTimeout(resolve, 150));
      }
    }
  }
  throw lastErr;
}

async function renderAiPatchGrid(cols, rows, count) {
  const grid = $("aiPatchGrid");
  if (!grid) {
    return;
  }
  revokeAiPatchObjectUrls();
  grid.innerHTML = "";
  const safeCols = Math.max(1, cols || 1);
  grid.style.gridTemplateColumns = `repeat(${safeCols}, 40px)`;
  const stamp = Date.now();
  const images = [];
  for (let i = 0; i < count; i++) {
    const cell = document.createElement("div");
    cell.className = "ai-patch-cell";
    const img = document.createElement("img");
    const r = Math.floor(i / safeCols);
    const c = i % safeCols;
    img.alt = `patch ${i}`;
    img.title = `patch #${i + 1} - hang ${r}, cot ${c} (grid ${cols}x${rows})`;
    const badge = document.createElement("span");
    badge.className = "ai-patch-badge";
    badge.textContent = String(i + 1);
    cell.appendChild(img);
    cell.appendChild(badge);
    grid.appendChild(cell);
    images.push(img);
  }

  // A small pool of "workers" pulls the next un-fetched index instead of
  // firing every request up front, so at most AI_PATCH_LOAD_CONCURRENCY
  // patches are ever in flight to the device at the same time.
  let nextIndex = 0;
  let failCount = 0;
  async function worker() {
    while (nextIndex < count) {
      const i = nextIndex++;
      try {
        const url = await fetchAiPatchImageUrl(i, stamp);
        state.aiPatchObjectUrls.push(url);
        images[i].src = url;
      } catch (err) {
        failCount++;
        images[i].alt = `patch ${i} loi`;
        images[i].title = `patch #${i + 1}: tai that bai (${err.message})`;
      }
    }
  }
  const workers = [];
  for (let w = 0; w < Math.min(AI_PATCH_LOAD_CONCURRENCY, Math.max(1, count)); w++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  if (failCount > 0) {
    log(`Cat patch: ${failCount}/${count} patch tai loi (se thu lai o lan chup sau)`);
  }
}

// Loads the device's actual patch-grid constants (patch size + max cols/rows)
// once at startup, so the live grid-line preview (updateCropGridLines) uses
// real firmware values instead of the JS-side defaults. Safe to skip on
// failure - the defaults in `state` match the current firmware anyway.
async function fetchAiPatchInfo() {
  try {
    const res = await fetch("/api/ai_patch/status");
    const data = await res.json();
    if (data && data.patch_size) state.aiPatchSize = data.patch_size;
    if (data && data.max_cols) state.aiPatchMaxCols = data.max_cols;
    if (data && data.max_rows) state.aiPatchMaxRows = data.max_rows;
    updateCropGridLines();
  } catch (err) {
    /* keep JS-side defaults */
  }
}

async function fetchAiCoresetStatus() {
  try {
    const res = await fetch(`/api/ai_coreset/status?t=${Date.now()}`);
    const data = await res.json();
    const summary = $("aiCoresetSummary");
    const body = $("aiCoresetTableBody");
    if (!summary || !body) return data;
    body.innerHTML = "";
    const ready = !!data.ready;
    const bank = Number(data.bank_count || 0);
    const total = Number(data.total_vectors || 0);
    if (!ready) {
      summary.textContent = "Chưa xây Memory Bank (sẽ tự chạy sau khi P3 xử lý xong toàn bộ ảnh).";
      const dl = $("aiCoresetDownload"); if (dl) dl.classList.add("hidden");
      return data;
    }
    summary.textContent = `Memory Bank: ${bank}/${total} vector • tỉ lệ coreset=${Number((data.coreset_ratio || 0) * 100).toFixed(0)}% • K tối thiểu=${Number(data.min_bank_count || 0)} • K mục tiêu (theo tỉ lệ, tối đa=${Number(data.max_bank_count || 0)})=${Number(data.target_bank_count || 0)} • giữ ${Number(data.kept_percent || 0).toFixed(1)}% • giảm ${Number(data.reduction_percent || 0).toFixed(1)}% • ε=${Number(data.epsilon || 0).toFixed(3)} (α=${Number(data.epsilon_alpha || 0).toFixed(2)}, tự dừng khi dist≤ε sau khi đủ K tối thiểu) • train max=${Number(data.train_max_score || 0).toFixed(4)} • Threshold=${Number(data.threshold || 0).toFixed(4)} • build=${Number(data.build_ms || 0).toFixed(0)} ms`;
    for (const e of (data.entries || [])) {
      const tr = document.createElement("tr");
      const vals = Array.isArray(e.preview_vector) ? e.preview_vector : [];
      const cells = [Number(e.index || 0) + 1, Number(e.source_image || 0), Number(e.source_patch || 0), Number(e.selection_distance || 0).toFixed(4), ...vals.map(v => Number(v).toFixed(4))];
      while (cells.length < 12) cells.push("-");
      for (const value of cells) { const td = document.createElement("td"); td.textContent = String(value); tr.appendChild(td); }
      body.appendChild(tr);
    }
    const dl = $("aiCoresetDownload");
    if (dl) {
      dl.classList.remove("hidden");
      dl.textContent = `Tải Memory Bank (${bank} vector)`;
      dl.onclick = async (ev) => {
        ev.preventDefault();
        try {
          const r = await fetch(`/api/ai_coreset/bank?t=${Date.now()}`);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const json = await r.json();
          const blob = new Blob([JSON.stringify(json, null, 2)], {type:"application/json"});
          if (state.aiCoresetObjectUrl) URL.revokeObjectURL(state.aiCoresetObjectUrl);
          state.aiCoresetObjectUrl = URL.createObjectURL(blob);
          const a = document.createElement("a"); a.href = state.aiCoresetObjectUrl; a.download = `patchcore_memory_bank_${Date.now()}.json`;
          document.body.appendChild(a); a.click(); a.remove();
        } catch (err) { log(`Tai Memory Bank that bai: ${err.message}`); }
      };
    }
    return data;
  } catch (err) {
    const summary = $("aiCoresetSummary");
    if (summary) summary.textContent = `Không đọc được Memory Bank: ${err.message}`;
    return null;
  }
}

async function fetchAiSessionStatus() {
  try {
    const res = await fetch(`/api/ai_session/status?t=${Date.now()}`);
    const data = await res.json();
    state.aiSessionCount = Number(data.image_count || 0);
    state.aiSessionState = data.state || "idle";
    const el = $("aiSessionReadout");
    if (el) {
      const processing = Number(data.processed_count || 0);
      const total = Number(data.image_count || 0);
      const patches = Number(data.total_patches || 0);
      const sessionDownload = $("aiSessionDownload");
      if (state.aiSessionState === "collecting" || state.aiSessionState === "idle") {
        if (sessionDownload) sessionDownload.classList.add("hidden");
        el.textContent = `Đang chụp: ${total} ảnh. Tiếp tục bấm P4; bấm P3 để bắt đầu trích xuất. ${patches ? `Đã lưu ${Math.round((Number(data.stored_rgb_bytes || 0))/1024)} KB.` : ""}`;
      } else if (state.aiSessionState === "processing") {
        if (sessionDownload) sessionDownload.classList.add("hidden");
        const current = Number(data.processing_image || 0);
        el.textContent = `Đang trích xuất: ${processing}/${total} ảnh${current ? ` (ảnh #${current})` : ""}, đã có ${patches} patch.`;
      } else if (state.aiSessionState === "building_coreset") {
        if (sessionDownload) sessionDownload.classList.add("hidden");
        el.textContent = `Đã trích xuất ${processing}/${total} ảnh, ${patches} patch. Đang chọn vector đại diện để xây Memory Bank...`;
      } else if (state.aiSessionState === "finished") {
        el.textContent = `Hoàn tất: ${processing}/${total} ảnh, tổng ${patches} patch → mỗi patch ${32}-D. Có thể bấm P4 để tạo phiên mới.`;
      } else {
        el.textContent = "Chưa có phiên chụp. Bấm P4 để chụp ảnh đầu tiên.";
      }
      if (data.error || data.last_error) {
        el.textContent += ` Lỗi: ${data.error || data.last_error}.`;
      }
    }
    fetchAiCoresetStatus();
    return data;
  } catch (err) {
    const el = $("aiSessionReadout");
    if (el) el.textContent = `Không đọc được trạng thái phiên: ${err.message}`;
    return null;
  }
}

async function fetchSessionImageFeatures(imageIndex) {
  const res = await fetch(`/api/ai_session/features?image=${imageIndex}&t=${Date.now()}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function downloadAllSessionFeatures() {
  const link = $("aiSessionDownload");
  if (!link) return;
  try {
    const status = await fetchAiSessionStatus();
    const count = Number(status && status.image_count || 0);
    const processed = Number(status && status.processed_count || 0);
    if (!count || processed !== count) {
      log("Chua co du vecto cua tat ca anh de tai");
      return;
    }
    link.textContent = "Đang gom vector...";
    link.classList.add("disabled");
    const images = [];
    for (let i = 0; i < count; i++) {
      images.push(await fetchSessionImageFeatures(i));
    }
    const blob = new Blob([JSON.stringify({feature_dim: 32, image_count: count, images}, null, 2)], {type:"application/json"});
    if (state.aiInferObjectUrl) URL.revokeObjectURL(state.aiInferObjectUrl);
    state.aiInferObjectUrl = URL.createObjectURL(blob);
    link.href = state.aiInferObjectUrl;
    link.download = `multi_image_feature_vectors_${Date.now()}.json`;
    link.textContent = `Tải toàn bộ ${count} ảnh + vector`;
    link.classList.remove("disabled", "hidden");
  } catch (err) {
    link.classList.remove("disabled");
    link.textContent = "Tải toàn bộ vector thất bại";
    log(`Tai vector session that bai: ${err.message}`);
  }
}

async function captureAiPatches() {
  await showLastAiCapture("Xem ket qua gan nhat");
}

async function showCameraJpeg(payload) {
  const oldUrl = state.cameraObjectUrl;
  const blob = new Blob([payload], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  state.cameraObjectUrl = url;
  const img = $("cameraStream");
  img.src = url;
  img.classList.add("active");
  $("cameraPlaceholder").classList.add("hidden");
  if (oldUrl) URL.revokeObjectURL(oldUrl);

  if (state.cropEnabled && typeof createImageBitmap === "function" && !state.cameraCropBusy) {
    state.cameraCropBusy = true;
    try {
      const bitmap = await createImageBitmap(blob);
      drawCroppedFrame(bitmap);
      bitmap.close();
    } catch (err) {
      log(`Camera crop preview failed: ${err.message}`);
    } finally {
      state.cameraCropBusy = false;
    }
  }
}

function handleCameraBinary(buf) {
  if (buf.byteLength < CAMERA_HEADER_SIZE) {
    log(`Camera WS short packet: len=${buf.byteLength}`);
    return;
  }
  const dv = new DataView(buf);
  if (dv.getUint8(0) !== 0x43 || dv.getUint8(1) !== 1) {
    log(`Camera WS bad header: magic=${dv.getUint8(0)} version=${dv.getUint8(1)} len=${buf.byteLength}`);
    return;
  }
  const format = dv.getUint8(2);
  const payloadLen = dv.getUint32(4, true);
  if (payloadLen + CAMERA_HEADER_SIZE !== buf.byteLength) {
    log(`Camera WS bad length: payload=${payloadLen} packet=${buf.byteLength}`);
    return;
  }
  if (format === CAMERA_FORMAT_MJPEG) {
    if (!state.cameraFirstFrameLogged) {
      log(`Camera first MJPEG frame: ${dv.getUint16(12, true)}x${dv.getUint16(14, true)} len=${payloadLen}`);
      state.cameraFirstFrameLogged = true;
    }
    showCameraJpeg(new Uint8Array(buf, CAMERA_HEADER_SIZE, payloadLen));
  } else {
    log(`Camera WS unsupported frame format: ${format}`);
  }
}

function startCameraStream() {
  if (state.cameraStreamActive) return;
  state.cameraStopRequested = false;
  state.cameraStreamActive = true;
  const url = `ws://${location.host}/camera_ws`;
  log(`Camera WebSocket opening: ${url}`);
  try {
    state.cameraWs = new WebSocket(url);
  } catch (err) {
    state.cameraStreamActive = false;
    log(`Camera WebSocket constructor failed: ${err.message}`);
    return;
  }
  state.cameraWs.binaryType = "arraybuffer";
  state.cameraWs.onopen = () => {
    log("Camera WebSocket connected");
    const camera = state.devices && state.devices.camera;
    if (camera && camera.enabled && camera.selected_format === "mjpeg") {
      state.cameraWs.send(JSON.stringify({ type: "set_camera_enabled", enabled: true }));
    }
    // Let the ESP32 know the current fixed crop geometry when source dimensions
    // are already known. The firmware independently recomputes it on capture.
    state.cameraWs.send(JSON.stringify({
      type: "set_crop",
      left: state.crop.left,
      top: state.crop.top,
      width: state.crop.width,
      height: state.crop.height,
    }));
  };
  state.cameraWs.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
      handleCameraBinary(event.data);
      return;
    }
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      log(`Camera WS non-JSON text message: ${event.data}`);
      return;
    }
    if (msg && msg.type === "capture_photo") {
      log("P4: anh moi da duoc luu vao phien chup");
      await showLastAiCapture("Nut P4", false);
      await fetchAiSessionStatus();
    } else if (msg && msg.type === "ai_session") {
      await fetchAiSessionStatus();
      if (msg.event === "capture") {
        await showLastAiCapture("Nut P4", false);
      } else if (msg.event === "coreset_start") {
        await fetchAiCoresetStatus();
      } else if (msg.event === "finish" || msg.event === "finish_error") {
        await showLastAiCapture("Ket qua sau P3", true);
        const link = $("aiSessionDownload");
        if (link) link.classList.remove("hidden");
      }
    } else if (msg && msg.type === "error") {
      log(`Camera WS error: ${msg.code || "unknown"} ${msg.message || ""}`);
    } else {
      log(`Camera WS unhandled message: ${event.data}`);
    }
  };
  state.cameraWs.onerror = () => log(`Camera WebSocket error: ${url} state=${state.cameraWs ? state.cameraWs.readyState : "-"}`);
  state.cameraWs.onclose = (event) => {
    log(`Camera WebSocket disconnected: code=${event.code} reason=${event.reason || "-"} clean=${event.wasClean}`);
    state.cameraWs = null;
    state.cameraStreamActive = false;
    const camera = state.devices && state.devices.camera;
    if (!state.cameraStopRequested && camera && camera.connected && camera.enabled && camera.streaming) {
      setTimeout(startCameraStream, 500);
    }
  };
}

function sendCameraJson(obj) {
  startCameraStream();
  const payload = JSON.stringify(obj);
  const send = (retry = 0) => {
    if (state.cameraWs && state.cameraWs.readyState === WebSocket.OPEN) {
      state.cameraWs.send(payload);
    } else if (retry < 10) {
      setTimeout(() => send(retry + 1), 100);
    } else {
      log(`Camera WS command dropped: ${obj.type}`);
    }
  };
  send();
}

function sendCameraFormat() {
  sendCameraJson({
    type: "set_camera_format",
    format: $("cameraFormat").value,
    resolution: Number($("cameraResolution").value),
  });
}

// Keeps the ESP32-side crop -> 40x40 patch cut (app_ai_patch.c) aligned with
// the red crop box the user sees on this page. Safe/cheap to call often -
// the firmware just clamps and stores the four numbers.
function sendCropToDevice() {
  applyFixedCropFromImage();
  sendCameraJson({
    type: "set_crop",
    left: state.crop.left,
    top: state.crop.top,
    width: state.crop.width,
    height: state.crop.height,
  });
}

function updateCamera(camera) {
  const connected = camera && camera.connected;
  const format = currentCameraFormat(camera);
  const resolutions = format && Array.isArray(format.resolutions) ? format.resolutions : [];
  const enabled = connected && camera.enabled;
  const streaming = enabled && camera.streaming && camera.selected_format === "mjpeg";
  const shouldOpenCameraWs = enabled && camera.selected_format === "mjpeg";
  setPill($("cameraState"), connected ? (enabled ? "Enabled" : "Ready") : "Disconnected", connected ? "ok" : "");
  $("cameraEnabled").disabled = !connected || !cameraFormatByName(camera, "mjpeg") || !resolutions.length;
  $("cameraEnabled").textContent = enabled ? "Close" : "Open";
  $("cameraEnabled").dataset.enabled = enabled ? "1" : "0";
  $("cameraEnabled").classList.toggle("is-open-action", connected && !enabled && !!cameraFormatByName(camera, "mjpeg") && !!resolutions.length);
  $("cameraEnabled").classList.toggle("is-close-action", connected && enabled);
  fillCameraFormats(camera);
  fillCameraResolutions(camera);
  $("cameraFormat").disabled = !connected || enabled;
  $("cameraResolution").disabled = !connected || enabled || !resolutions.length;
  if (!connected) {
    stopCameraStream("Camera disconnected");
  } else if (!enabled) {
    stopCameraStream("Camera disabled");
  } else if (!shouldOpenCameraWs) {
    stopCameraStream("Camera stream unavailable");
  } else {
    if (!streaming) $("cameraPlaceholder").textContent = "Camera starting";
    startCameraStream();
  }
  if (connected !== state.cameraLastConnected) {
    log(connected ? "Camera connected" : "Camera disconnected");
    state.cameraLastConnected = connected;
  }
  if (enabled !== state.cameraLastEnabled) {
    log(enabled ? "Camera enabled" : "Camera disabled");
    state.cameraLastEnabled = enabled;
  }
}

function setToolGroupDisabled(groupId, disabled) {
  const group = $(groupId);
  group.classList.toggle("disabled", disabled);
  group.setAttribute("aria-disabled", disabled ? "true" : "false");
  group.querySelectorAll("input, select, button").forEach((control) => {
    control.disabled = disabled;
  });
}

function resetFilePlaybackState(reason) {
  if (!state.filePlaying && !state.fileTx) return;
  state.fileStop = true;
  state.spkRefillBudget = 0;
  if (state.fileTx && !state.fileTx.done) {
    const reject = state.fileTx.reject;
    state.fileTx.done = true;
    state.fileTx = null;
    reject(new Error(reason || "device_disconnected"));
  } else {
    state.fileTx = null;
  }
  state.filePlaying = false;
  $("fileProgress").value = 0;
  $("fileInfo").textContent = "Playback reset";
  log("File playback reset");
}

function render(data) {
  state.devices = data;
  const mic = data.mic;
  const spk = data.spk;
  const camera = data.camera || { connected: false, enabled: false, formats: [] };
  const micState = mic.connected ? (mic.enabled ? "enabled" : "ready") : "off";
  const spkState = spk.connected ? (spk.enabled ? "enabled" : "ready") : "off";
  const cameraState = camera.connected ? (camera.enabled ? "enabled" : "ready") : "off";
  log(`State updated: mic=${micState} spk=${spkState} camera=${cameraState}`);
  if (!spk.connected) resetFilePlaybackState("device_disconnected");
  const any = mic.connected || spk.connected || camera.connected;
  setPill($("usbStatus"), any ? "USB connected" : "USB idle", any ? "ok" : "");
  setPill($("micState"), mic.connected ? (mic.enabled ? "Enabled" : "Ready") : "Disconnected", mic.connected ? "ok" : "");
  setPill($("spkState"), spk.connected ? (spk.enabled ? "Enabled" : "Ready") : "Disconnected", spk.connected ? "ok" : "");
  $("deviceSummary").textContent = any ? "USB device ready" : "Waiting for USB device";
  const audioVid = mic.vid || spk.vid;
  const audioPid = mic.pid || spk.pid;
  $("vidPid").textContent = audioVid && audioPid ? `${audioVid.toString(16)}:${audioPid.toString(16)}` : "--";
  $("product").textContent = mic.product || spk.product || "--";
  $("interfaces").textContent = `MIC ${mic.iface_num || "--"} / SPK ${spk.iface_num || "--"} / CAM ${camera.connected ? "yes" : "--"}`;
  updateControls("mic", mic);
  updateControls("spk", spk);
  updateCamera(camera);
  const spkReady = spk.connected && spk.enabled;
  setToolGroupDisabled("toneGroup", !spkReady || state.filePlaying || state.toneSending);
  setToolGroupDisabled("fileGroup", !spkReady);
  if (spkReady) {
    $("audioFile").disabled = state.filePlaying;
    $("playFile").disabled = state.filePlaying || state.spkResetting || !$("audioFile").files.length;
    $("stopFile").disabled = !state.filePlaying;
  }
  if (!mic.connected || !mic.enabled) updateMicMeter(0);
}

async function refreshDevices() {
  try {
    const res = await fetch("/api/devices");
    render(await res.json());
  } catch (err) {
    log(`Device refresh failed: ${err.message}`);
  }
}

function ensureAudio() {
  if (!state.audioCtx) {
    state.audioCtx = new AudioContext();
  }
  if (state.audioCtx.state === "suspended") state.audioCtx.resume();
  return state.audioCtx;
}

async function ensureMicPlayer(format) {
  const ctx = ensureAudio();
  if (!ctx.audioWorklet || state.micWorkletFailed) return ensureScriptMicPlayer(ctx, format);
  try {
    if (!state.micWorkletReady) state.micWorkletReady = ctx.audioWorklet.addModule("/assets/mic-player-worklet.js");
    await state.micWorkletReady;
  } catch (err) {
    state.micWorkletReady = null;
    state.micWorkletFailed = true;
    log(`MIC worklet unavailable, fallback to script player: ${err.message}`);
    return ensureScriptMicPlayer(ctx, format);
  }

  const channels = Math.max(1, Number(format.channels) || 1);
  const formatKey = `${format.sampleRate}:${channels}:${format.bitDepth}:${format.subframeSize}`;
  if (state.micPlayer && state.micWorkletNode && state.micWorkletFormatKey === formatKey) return state.micPlayer;
  if (state.micWorkletNode) state.micWorkletNode.disconnect();
  if (state.micScriptPlayer) {
    state.micScriptPlayer.node.disconnect();
    state.micScriptPlayer = null;
  }

  let node = null;
  try {
    node = new AudioWorkletNode(ctx, "mic-player", { outputChannelCount: [channels] });
  } catch (err) {
    state.micWorkletFailed = true;
    log(`MIC worklet node failed, fallback to script player: ${err.message}`);
    return ensureScriptMicPlayer(ctx, format);
  }
  node.port.onmessage = (event) => {
    if (event.data && event.data.type === "level") updateMicMeter(Math.max(event.data.level || 0, state.micLevel * 0.72));
  };
  node.connect(ctx.destination);
  node.port.postMessage({ type: "format", sampleRate: format.sampleRate, channels, bitDepth: format.bitDepth, subframeSize: format.subframeSize });
  state.micWorkletNode = node;
  state.micWorkletFormatKey = formatKey;
  state.micPlayer = { send: (pcmBuffer) => node.port.postMessage({ type: "pcm", pcm: pcmBuffer }, [pcmBuffer]) };
  log(`MIC worklet ready: ${format.sampleRate}Hz ${channels}ch ${format.bitDepth}bit`);
  return state.micPlayer;
}

function currentMicFormat() {
  const mic = state.devices && state.devices.mic;
  if (!mic || !mic.sample_freq) return null;
  const alt = mic.alts.find((item) => item.alt === mic.selected_alt) || mic.alts[0];
  if (!alt) return null;
  return {
    mic,
    alt,
    sampleRate: mic.sample_freq,
    channels: alt.channels,
    bitDepth: alt.bit_resolution,
    subframeSize: alt.subframe_size || Math.ceil(alt.bit_resolution / 8),
  };
}

function readPcmSample(dv, bytes, offset, bitDepth, subframeSize) {
  if (bitDepth === 8 && subframeSize === 1) return (bytes[offset] - 128) / 128;
  if (bitDepth <= 16 && subframeSize >= 2) return dv.getInt16(offset, true) / 32768;
  if (bitDepth <= 24 && subframeSize >= 3) {
    let value = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
    if (value & 0x800000) value |= 0xff000000;
    return value / 8388608;
  }
  if (bitDepth <= 32 && subframeSize >= 4) return dv.getInt32(offset, true) / 2147483648;
  return 0;
}

function ensureScriptMicPlayer(ctx, format) {
  const channels = Math.max(1, Number(format.channels) || 1);
  const formatKey = `script:${format.sampleRate}:${channels}:${format.bitDepth}:${format.subframeSize}`;
  if (state.micPlayer && state.micScriptPlayer && state.micWorkletFormatKey === formatKey) return state.micPlayer;
  if (state.micWorkletNode) {
    state.micWorkletNode.disconnect();
    state.micWorkletNode = null;
  }
  if (state.micScriptPlayer) state.micScriptPlayer.node.disconnect();

  const capacityFrames = Math.max(1024, Math.ceil(format.sampleRate * 0.6));
  const player = {
    node: ctx.createScriptProcessor(1024, 0, channels),
    channels,
    capacityFrames,
    buffer: new Float32Array(capacityFrames * channels),
    readFrame: 0,
    writeFrame: 0,
    availableFrames: 0,
    readFrac: 0,
  };
  player.node.onaudioprocess = (event) => processScriptMicPlayer(event, player, format);
  player.node.connect(ctx.destination);
  state.micScriptPlayer = player;
  state.micWorkletFormatKey = formatKey;
  state.micPlayer = { send: (pcmBuffer) => pushScriptMicPcm(player, format, new Uint8Array(pcmBuffer)) };
  log(`MIC script ring player ready: ${format.sampleRate}Hz ${channels}ch ${format.bitDepth}bit`);
  return state.micPlayer;
}

function pushScriptMicPcm(player, format, bytes) {
  const frameBytes = player.channels * format.subframeSize;
  const frames = Math.floor(bytes.byteLength / frameBytes);
  if (frames <= 0) return;
  if (frames >= player.capacityFrames) bytes = bytes.subarray((frames - player.capacityFrames + 1) * frameBytes);
  const usableFrames = Math.floor(bytes.byteLength / frameBytes);
  const overflow = Math.max(0, player.availableFrames + usableFrames - player.capacityFrames);
  if (overflow > 0) {
    player.readFrame = (player.readFrame + overflow) % player.capacityFrames;
    player.availableFrames -= overflow;
    player.readFrac = 0;
  }

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let sumSquares = 0;
  let peak = 0;
  for (let i = 0; i < usableFrames; i++) {
    for (let ch = 0; ch < player.channels; ch++) {
      const sample = readPcmSample(dv, bytes, i * frameBytes + ch * format.subframeSize, format.bitDepth, format.subframeSize);
      player.buffer[(player.writeFrame * player.channels) + ch] = sample;
      sumSquares += sample * sample;
      peak = Math.max(peak, Math.abs(sample));
    }
    player.writeFrame = (player.writeFrame + 1) % player.capacityFrames;
  }
  player.availableFrames += usableFrames;
  const rms = Math.sqrt(sumSquares / Math.max(1, usableFrames * player.channels));
  updateMicMeter(Math.max(Math.min(1, Math.max(rms * 4, peak * 0.8)), state.micLevel * 0.72));
}

function processScriptMicPlayer(event, player, format) {
  const output = event.outputBuffer;
  const frames = output.length;
  const step = format.sampleRate / output.sampleRate;
  for (let i = 0; i < frames; i++) {
    if (player.availableFrames <= Math.ceil(player.readFrac)) {
      for (let ch = 0; ch < output.numberOfChannels; ch++) output.getChannelData(ch)[i] = 0;
      continue;
    }
    const sourceOffset = Math.floor(player.readFrac);
    const sourceFrame = (player.readFrame + sourceOffset) % player.capacityFrames;
    for (let ch = 0; ch < output.numberOfChannels; ch++) {
      const sourceCh = Math.min(ch, player.channels - 1);
      output.getChannelData(ch)[i] = player.buffer[(sourceFrame * player.channels) + sourceCh];
    }
    player.readFrac += step;
    const consumeFrames = Math.floor(player.readFrac);
    if (consumeFrames > 0) {
      player.readFrame = (player.readFrame + consumeFrames) % player.capacityFrames;
      player.availableFrames = Math.max(0, player.availableFrames - consumeFrames);
      player.readFrac -= consumeFrames;
    }
  }
}

function updateMicMeter(level) {
  state.micLevel = Math.max(0, Math.min(1, level));
  const percent = Math.round(state.micLevel * 100);
  const bar = $("micMeter").firstElementChild;
  bar.style.width = `${percent}%`;
  bar.classList.toggle("hot", percent >= 90);
  $("micLevelText").textContent = `${percent}%`;
}

async function playMicPcm(pcmBuffer) {
  const format = currentMicFormat();
  if (!format) return;
  const { sampleRate, channels, bitDepth, subframeSize } = format;
  const frameBytes = channels * subframeSize;
  const frames = Math.floor(pcmBuffer.byteLength / frameBytes);
  if (frames <= 0) return;
  try {
    const player = await ensureMicPlayer(format);
    player.send(pcmBuffer);
    state.micFrames++;
    if (state.micFrames === 1) log(`MIC playback started: ${sampleRate}Hz ${channels}ch ${bitDepth}bit`);
  } catch (err) {
    log(`MIC playback failed: ${err.message}`);
  }
}

function handleBinary(buf) {
  const view = new Uint8Array(buf);
  if (view.length <= 4 || view[0] !== 0x01) return;
  const len = view[2] | (view[3] << 8);
  if (len !== view.length - 4) return;
  playMicPcm(buf.slice(4));
}

function connectAudioWs() {
  const url = `ws://${location.host}/ws`;
  let audioWs = null;
  try {
    audioWs = new WebSocket(url);
  } catch (err) {
    log(`Audio WebSocket constructor failed: ${err.message}`);
    return;
  }
  audioWs.binaryType = "arraybuffer";
  state.audioWs = audioWs;
  audioWs.onopen = () => {
    setPill($("audioWsStatus"), "Audio WS online", "ok");
    log("Audio WebSocket connected");
    sendAudioJson({ type: "get_state" });
    refreshDevices();
  };
  audioWs.onclose = (event) => {
    setPill($("audioWsStatus"), "Audio WS offline", "warn");
    log(`Audio WebSocket disconnected: code=${event.code} reason=${event.reason || "-"} clean=${event.wasClean}`);
    setTimeout(connectAudioWs, 1000);
  };
  audioWs.onerror = () => log(`Audio WebSocket error: ${url} state=${audioWs.readyState}`);
  audioWs.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "state") {
        render(msg);
      }
      if (msg.type === "error") log(`${msg.code}: ${msg.message}`);
      if (msg.type === "spk_reset_done") handleSpkResetDone();
      if (msg.type === "spk_refill_budget") handleSpkRefillBudget(msg.bytes || 0);
    } else {
      handleBinary(event.data);
    }
  };
}

function selectedFormat(prefix) {
  const [alt, freq] = $(`${prefix}Format`).value.split(":").map((v) => Number(v));
  return { alt, sample_freq: freq };
}

function writePcmSample(dv, bytes, offset, sample, bitDepth, subframeSize) {
  const max = Math.pow(2, bitDepth - 1) - 1;
  const min = -Math.pow(2, bitDepth - 1);
  const value = Math.max(min, Math.min(max, Math.round(sample * max)));
  if (bitDepth === 8 && subframeSize === 1) {
    bytes[offset] = Math.max(0, Math.min(255, Math.round((sample + 1) * 127.5)));
  } else if (bitDepth <= 16 && subframeSize >= 2) {
    dv.setInt16(offset, value, true);
  } else if (bitDepth <= 24 && subframeSize >= 3) {
    bytes[offset] = value & 0xff;
    bytes[offset + 1] = (value >> 8) & 0xff;
    bytes[offset + 2] = (value >> 16) & 0xff;
    if (subframeSize >= 4) bytes[offset + 3] = value < 0 ? 0xff : 0x00;
  } else if (bitDepth <= 32 && subframeSize >= 4) {
    dv.setInt32(offset, value, true);
  }
}

function currentSpkFormat() {
  const spk = state.devices.spk;
  const alt = spk.alts.find((item) => item.alt === spk.selected_alt) || spk.alts[0];
  return {
    spk,
    alt,
    sampleRate: spk.sample_freq,
    channels: alt.channels,
    bitDepth: alt.bit_resolution,
    subframeSize: alt.subframe_size || Math.ceil(alt.bit_resolution / 8),
  };
}

function sendSpkPacket(part) {
  const packet = new Uint8Array(part.length + 4);
  packet[0] = 0x02;
  packet[1] = 0;
  packet[2] = part.length & 0xff;
  packet[3] = (part.length >> 8) & 0xff;
  packet.set(part, 4);
  state.audioWs.send(packet);
}

function refillBudgetPayloadBytes(frameBytes, remainingBytes) {
  const budget = Math.min(state.spkRefillBudget, SPK_MAX_PAYLOAD, remainingBytes);
  return Math.floor(budget / frameBytes) * frameBytes;
}

function makeTonePcm() {
  const { sampleRate, channels, bitDepth, subframeSize } = currentSpkFormat();
  const freq = Number($("toneFreq").value || 1000);
  const ms = Number($("toneMs").value || 1000);
  const frames = Math.floor(sampleRate * ms / 1000);
  const pcm = new Uint8Array(frames * channels * subframeSize);
  const dv = new DataView(pcm.buffer);
  for (let i = 0; i < frames; i++) {
    const sample = Math.sin(2 * Math.PI * freq * i / sampleRate) * 0.35;
    for (let ch = 0; ch < channels; ch++) {
      writePcmSample(dv, pcm, (i * channels + ch) * subframeSize, sample, bitDepth, subframeSize);
    }
  }
  return pcm;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function sendTone() {
  if (state.toneSending || state.filePlaying || !state.audioWs || state.audioWs.readyState !== WebSocket.OPEN) return;
  state.toneSending = true;
  $("playTone").disabled = true;
  try {
    const pcm = makeTonePcm();
    const format = currentSpkFormat();
    await sendPcmByRefillBudget(pcm, format, () => {});
    log("Test tone sent");
  } catch (err) {
    log(`Test tone failed: ${err.message}`);
  } finally {
    state.toneSending = false;
    render(state.devices);
  }
}

async function decodeAudioFile(file) {
  const ctx = ensureAudio();
  const data = await file.arrayBuffer();
  return await ctx.decodeAudioData(data);
}

async function resampleAudioBuffer(buffer, sampleRate) {
  if (buffer.sampleRate === sampleRate) return buffer;
  const length = Math.max(1, Math.ceil(buffer.duration * sampleRate));
  const offline = new OfflineAudioContext(buffer.numberOfChannels, length, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);
  return await offline.startRendering();
}

function audioBufferToSpkPcm(buffer, format) {
  return audioBufferChunkToSpkPcm(buffer, format, 0, buffer.length);
}

function audioBufferChunkToSpkPcm(buffer, format, frameOff, frames) {
  const { channels, bitDepth, subframeSize } = format;
  const pcm = new Uint8Array(frames * channels * subframeSize);
  const dv = new DataView(pcm.buffer);
  const inputs = [];
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) inputs.push(buffer.getChannelData(ch));
  for (let i = 0; i < frames; i++) {
    const srcIndex = frameOff + i;
    for (let ch = 0; ch < channels; ch++) {
      let sample = 0;
      if (buffer.numberOfChannels === 1) {
        sample = inputs[0][srcIndex];
      } else if (channels === 1) {
        for (let src = 0; src < buffer.numberOfChannels; src++) sample += inputs[src][srcIndex];
        sample /= buffer.numberOfChannels;
      } else {
        sample = inputs[Math.min(ch, buffer.numberOfChannels - 1)][srcIndex];
      }
      writePcmSample(dv, pcm, (i * channels + ch) * subframeSize, sample, bitDepth, subframeSize);
    }
  }
  return pcm;
}

function pumpFileTx() {
  const tx = state.fileTx;
  if (!tx || tx.done) return;
  try {
    while (tx.off < tx.pcm.length) {
      if (state.fileStop) throw new Error("stopped");
      if (!state.audioWs || state.audioWs.readyState !== WebSocket.OPEN) throw new Error("Audio WebSocket disconnected");
      const partLen = refillBudgetPayloadBytes(tx.frameBytes, tx.pcm.length - tx.off);
      if (partLen <= 0) {
        logSpkBudgetWait("pcm", tx.pcm.length - tx.off);
        return;
      }
      clearSpkBudgetWait();
      if (state.audioWs.bufferedAmount > SPK_WS_BUFFER_LIMIT) {
        logSpkBufferWait("pcm");
        setTimeout(pumpFileTx, 5);
        return;
      }
      clearSpkBufferWait();
      const part = tx.pcm.subarray(tx.off, tx.off + partLen);
      sendSpkPacket(part);
      tx.off += part.length;
      state.spkRefillBudget -= part.length;
      tx.onProgress(Math.min(100, Math.round(tx.off * 100 / tx.pcm.length)));
    }
    if (tx.off >= tx.pcm.length) {
      tx.done = true;
      const resolve = tx.resolve;
      state.fileTx = null;
      resolve();
    }
  } catch (err) {
    tx.done = true;
    const reject = tx.reject;
    state.fileTx = null;
    reject(err);
  }
}

function pumpBufferTx() {
  const tx = state.fileTx;
  if (!tx || tx.done) return;
  try {
    while (tx.frameOff < tx.buffer.length) {
      if (state.fileStop) throw new Error("stopped");
      if (!state.audioWs || state.audioWs.readyState !== WebSocket.OPEN) throw new Error("Audio WebSocket disconnected");
      const partLen = refillBudgetPayloadBytes(tx.frameBytes, (tx.buffer.length - tx.frameOff) * tx.frameBytes);
      if (partLen <= 0) {
        logSpkBudgetWait("buffer", (tx.buffer.length - tx.frameOff) * tx.frameBytes);
        return;
      }
      clearSpkBudgetWait();
      const frames = Math.min(Math.floor(partLen / tx.frameBytes), tx.buffer.length - tx.frameOff);
      if (state.audioWs.bufferedAmount > SPK_WS_BUFFER_LIMIT) {
        logSpkBufferWait("buffer");
        setTimeout(pumpBufferTx, 5);
        return;
      }
      clearSpkBufferWait();
      const part = audioBufferChunkToSpkPcm(tx.buffer, tx.format, tx.frameOff, frames);
      sendSpkPacket(part);
      tx.frameOff += frames;
      state.spkRefillBudget -= part.length;
      tx.onProgress(Math.min(100, Math.round(tx.frameOff * 100 / tx.buffer.length)));
    }
    if (tx.frameOff >= tx.buffer.length) {
      tx.done = true;
      const resolve = tx.resolve;
      state.fileTx = null;
      resolve();
    }
  } catch (err) {
    tx.done = true;
    const reject = tx.reject;
    state.fileTx = null;
    reject(err);
  }
}

function handleSpkRefillBudget(bytes) {
  if (state.spkResetting || !state.fileTx) {
    log(`Drop stale SPK refill budget: bytes=${bytes} resetting=${state.spkResetting}`);
    return;
  }
  state.spkRefillBudget = Math.max(state.spkRefillBudget, Math.max(0, Number(bytes) || 0));
  clearSpkBudgetWait();
  log(`SPK refill budget received: bytes=${bytes} budget=${state.spkRefillBudget}`);
  if (state.fileTx && state.fileTx.buffer) {
    pumpBufferTx();
  } else {
    pumpFileTx();
  }
}

function handleSpkResetDone() {
  state.spkRefillBudget = 0;
  state.spkResetting = false;
  clearSpkBudgetWait();
  clearSpkBufferWait();
  log("SPK reset done");
  if (state.devices) render(state.devices);
}

function logSpkBudgetWait(kind, remainingBytes) {
  const now = performance.now();
  if (!state.spkBudgetWaitSince) state.spkBudgetWaitSince = now;
  if (now - state.spkBudgetWaitSince >= 100 && now - state.spkLastBudgetWaitLog >= 500) {
    state.spkLastBudgetWaitLog = now;
    log(`SPK waiting refill budget: mode=${kind} wait=${Math.round(now - state.spkBudgetWaitSince)}ms remaining=${remainingBytes} budget=${state.spkRefillBudget}`);
  }
}

function clearSpkBudgetWait() {
  state.spkBudgetWaitSince = 0;
}

function logSpkBufferWait(kind) {
  const now = performance.now();
  if (!state.spkBufferWaitSince) state.spkBufferWaitSince = now;
  if (now - state.spkBufferWaitSince >= 100 && now - state.spkLastBufferWaitLog >= 500) {
    state.spkLastBufferWaitLog = now;
    log(`SPK WebSocket buffered wait: mode=${kind} wait=${Math.round(now - state.spkBufferWaitSince)}ms buffered=${state.audioWs ? state.audioWs.bufferedAmount : 0}`);
  }
}

function clearSpkBufferWait() {
  state.spkBufferWaitSince = 0;
}

function sendPcmByRefillBudget(pcm, format, onProgress) {
  const { channels, subframeSize } = format;
  const frameBytes = channels * subframeSize;
  return new Promise((resolve, reject) => {
    state.spkRefillBudget = 0;
    clearSpkBudgetWait();
    clearSpkBufferWait();
    state.fileTx = { pcm, frameBytes, off: 0, onProgress, resolve, reject, done: false };
    sendAudioJson({ type: "request_spk_refill_budget" });
  });
}

function sendAudioBufferByRefillBudget(buffer, format, onProgress) {
  const { channels, subframeSize } = format;
  const frameBytes = channels * subframeSize;
  return new Promise((resolve, reject) => {
    state.spkRefillBudget = 0;
    clearSpkBudgetWait();
    clearSpkBufferWait();
    state.fileTx = { buffer, format, frameBytes, frameOff: 0, onProgress, resolve, reject, done: false };
    sendAudioJson({ type: "request_spk_refill_budget" });
  });
}

async function playSelectedFile() {
  const file = $("audioFile").files[0];
  if (!file || state.filePlaying || state.spkResetting || !state.audioWs || state.audioWs.readyState !== WebSocket.OPEN) return;
  state.filePlaying = true;
  state.fileStop = false;
  render(state.devices);
  try {
    const format = currentSpkFormat();
    $("fileInfo").textContent = `Decoding ${file.name}`;
    $("fileProgress").value = 0;
    const decoded = await decodeAudioFile(file);
    if (state.fileStop) throw new Error("device_disconnected");
    const rendered = await resampleAudioBuffer(decoded, format.sampleRate);
    if (state.fileStop) throw new Error("device_disconnected");
    $("fileInfo").textContent = `${file.name} -> ${format.sampleRate}Hz ${format.channels}ch ${format.bitDepth}bit`;
    log(`File playback started: ${file.name}`);
    await sendAudioBufferByRefillBudget(rendered, format, (value) => ($("fileProgress").value = value));
    $("fileInfo").textContent = `Done: ${file.name}`;
    log("File playback done");
  } catch (err) {
    if (err.message === "stopped") {
      $("fileInfo").textContent = "Stopped";
      log("File playback stopped");
    } else if (err.message === "device_disconnected") {
      $("fileProgress").value = 0;
      $("fileInfo").textContent = "Playback reset";
    } else {
      $("fileInfo").textContent = "File playback failed";
      log(`File playback failed: ${err.message}`);
    }
  } finally {
    state.fileTx = null;
    state.filePlaying = false;
    state.fileStop = false;
    render(state.devices);
  }
}

function stopFilePlayback() {
  if (!state.filePlaying) return;
  state.fileStop = true;
  state.spkResetting = true;
  state.spkRefillBudget = 0;
  clearSpkBudgetWait();
  clearSpkBufferWait();
  if (state.fileTx && !state.fileTx.done) {
    const reject = state.fileTx.reject;
    state.fileTx.done = true;
    state.fileTx = null;
    reject(new Error("stopped"));
  }
  sendAudioJson({ type: "stop_spk_playback" });
  log("SPK stop command sent");
  if (state.devices) render(state.devices);
}

function bindControls() {
  $("micEnabled").onclick = (e) => {
    const enabled = e.currentTarget.dataset.enabled !== "1";
    if (enabled) ensureAudio();
    sendAudioJson({ type: "set_mic_enabled", enabled });
  };
  $("spkEnabled").onclick = (e) => {
    const enabled = e.currentTarget.dataset.enabled !== "1";
    sendAudioJson({ type: "set_spk_enabled", enabled });
  };
  $("micMute").onchange = (e) => sendAudioJson({ type: "set_mic_mute", mute: e.target.checked });
  $("spkMute").onchange = (e) => sendAudioJson({ type: "set_spk_mute", mute: e.target.checked });
  $("micVolume").onchange = (e) => sendAudioJson({ type: "set_mic_volume", volume: Number(e.target.value) });
  $("spkVolume").onchange = (e) => sendAudioJson({ type: "set_spk_volume", volume: Number(e.target.value) });
  $("micFormat").onchange = () => sendAudioJson({ type: "set_mic_format", ...selectedFormat("mic") });
  $("spkFormat").onchange = () => sendAudioJson({ type: "set_spk_format", ...selectedFormat("spk") });
  $("cameraEnabled").onclick = (e) => {
    const enabled = e.currentTarget.dataset.enabled !== "1";
    if (enabled) {
      $("cameraPlaceholder").textContent = "Camera starting";
    }
    sendCameraJson({ type: "set_camera_enabled", enabled });
    if (!enabled) {
      setTimeout(() => stopCameraStream("Camera disabled"), 50);
    }
  };
  $("cameraFormat").onchange = (e) => {
    log(`Camera format selected: ${e.target.value}`);
    sendCameraFormat();
  };
  $("cameraResolution").onchange = (e) => {
    log(`Camera resolution selected: ${e.target.options[e.target.selectedIndex].textContent}`);
    sendCameraFormat();
  };
  $("cameraStream").onerror = () => {
    log("Camera stream load error");
    stopCameraStream("Camera stream error");
  };
  $("cameraStream").onload = () => {
    applyFixedCropFromImage();
    updateCropOverlay();
    updateCropReadout();
    sendCropToDevice();
  };
  $("cropEnabled").onchange = (e) => {
    state.cropEnabled = e.target.checked;
    log(`Camera crop ${state.cropEnabled ? "enabled" : "disabled"}`);
    $("cropPreviewWrap").classList.toggle("active", state.cropEnabled);
    updateCropOverlay();
    if (!state.cropEnabled) {
      const canvas = $("cameraCanvas");
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };
  $("cropReset").onclick = () => {
    applyFixedCropFromImage();
    updateCropOverlay();
    updateCropReadout();
    sendCropToDevice();
  };
  $("aiPatchCapture").onclick = captureAiPatches;
  const sessionDownload = $("aiSessionDownload");
  if (sessionDownload) sessionDownload.onclick = (event) => { event.preventDefault(); downloadAllSessionFeatures(); };
  fetchAiSessionStatus();
  // AI crop is intentionally fixed; no drag/resize handlers are bound.

  window.addEventListener("resize", () => {
    if (state.cropResizeQueued) return;
    state.cropResizeQueued = true;
    requestAnimationFrame(() => {
      state.cropResizeQueued = false;
      updateCropOverlay();
    });
  });
  $("playTone").onclick = sendTone;
  $("audioFile").onchange = () => {
    const file = $("audioFile").files[0];
    $("fileInfo").textContent = file ? `${file.name} (${Math.round(file.size / 1024)} KB)` : "No file selected";
    $("fileProgress").value = 0;
    render(state.devices);
  };
  $("playFile").onclick = playSelectedFile;
  $("stopFile").onclick = stopFilePlayback;
}

// AI crop is fixed to the camera resolution / 40-pixel grid. This function is
// retained as a compatibility stub so other code does not need to change.
function bindCropDrag() {
  return;
}

try {
  bindControls();
  updateCropReadout();
  refreshDevices();
  connectAudioWs();
  fetchAiPatchInfo();
} catch (err) {
  log(`Startup failed: ${err.message}`);
}
