const $ = (id) => document.getElementById(id);

let streamImage = $('stream');
const inspectionImage = $('inspection-image');
const overlay = $('overlay');
const overlayCtx = overlay.getContext('2d');
const inspectionStage = document.querySelector('.inspection-stage');

/* Chưa có ảnh -> ẩn <img> để không lộ icon ảnh vỡ + chữ alt như trong screenshot. */
streamImage.style.visibility = 'hidden';
inspectionImage.style.visibility = 'hidden';

let latestStatus = null;
let streamRetryTimer = null;
let lastStatusSeq = -1;

/* Ảnh inspection: chỉ coi là "đã có ảnh" khi ảnh THỰC SỰ đã tải xong và hiển thị.
 * (Bản cũ dựa vào check.capture_id — luôn = 0 trong LEARN — nên lớp "Chưa có ảnh
 * inspection" bị bật lại sau mỗi lần poll dù ảnh đã nằm sẵn bên dưới.) */
let inspectionShownId = 0;      /* capture id của ảnh đang hiển thị */
let inspectionLoadingId = 0;    /* capture id đang tải */
let inspectionFailAt = 0;
let inspectionBlobUrl = null;

const RESULT_PASS = 0;
const RESULT_FAIL = 1;

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function formatFloat(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
}

function formatMs(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))} ms` : '—';
}

function stageFamily(stage) {
  if (!stage) return 'STANDBY';
  if (stage.startsWith('LEARN')) return 'LEARN';
  if (stage.startsWith('TRAIN')) return 'TRAIN';
  if (stage.startsWith('CHECK')) return 'CHECK';
  return stage === 'ERROR' ? 'ERROR' : 'STANDBY';
}

function isCheckStage(stage) {
  return stage && stage.startsWith('CHECK_');
}

function updatePipeline(stage) {
  const family = stageFamily(stage);
  document.querySelectorAll('.pipeline-step').forEach((el) => {
    const step = el.dataset.step;
    el.classList.toggle('active', family === step);
    el.classList.toggle('done',
      (step === 'LEARN' && (stage === 'TRAINING' || stage === 'TRAINED' || family === 'CHECK')) ||
      (step === 'TRAIN' && (stage === 'TRAINED' || family === 'CHECK'))
    );
  });
  setText('learn-state', family === 'LEARN' ? 'ACTIVE' : (family === 'TRAIN' || family === 'CHECK' ? 'DONE' : 'WAIT'));
  setText('train-state', family === 'TRAIN' ? 'ACTIVE' : (stage === 'TRAINED' || family === 'CHECK' ? 'DONE' : 'WAIT'));
  setText('check-state', family === 'CHECK' ? 'ACTIVE' : 'WAIT');
}

function resultMeta(status) {
  const stage = status.stage || 'STANDBY';
  const check = status.check || {};
  const finalResult = Number(check.final_result);
  const visualResult = Number(check.visual_result);
  const powerResult = Number(check.power_result);
  const fc = Number(check.patch_fail_count) || 0;
  const tc = Number(check.patch_total) || 0;
  const prev = status.summary || {};

  if (stage === 'CHECK_VISUAL') return ['AI RUNNING', 'Đang chụp ảnh và chấm từng patch 40×40.', 'working'];
  if (stage === 'CHECK_VISUAL_FAIL') return ['VISUAL FAIL', `${fc} / ${tc} patch vượt ngưỡng — PCB FAIL, bỏ qua đo dòng.`, 'fail'];
  if (stage === 'CHECK_VISUAL_PASS') return ['VISUAL PASS', `Ngoại quan đạt (${fc} / ${tc} patch lỗi) — STM32 sẽ chuyển sang đo dòng.`, 'pass'];
  if (stage === 'CHECK_POWER') return ['POWER RUNNING', 'Đang kiểm tra Imax / Iavg do STM32 gửi sang.', 'working'];
  if (stage === 'CHECK_POWER_PASS') return ['FINAL PASS', `Ngoại quan PASS (${fc} / ${tc} patch lỗi) + dòng điện PASS.`, 'pass'];
  if (stage === 'CHECK_POWER_FAIL') return ['FINAL FAIL', `Ngoại quan PASS (${fc} / ${tc} patch lỗi) nhưng dòng điện FAIL.`, 'fail'];
  if (stage === 'CHECK_WAIT_PCB') {
    const last = prev.valid ? ` PCB #${prev.board} vừa xong: ${resultWord(prev.final)} (xem Tổng kết bên dưới).` : '';
    return ['WAITING PCB', `Chờ PCB #${status.board || '—'} từ cảm biến IR.${last}`, 'neutral'];
  }
  if (stage === 'TRAINED') return ['TRAINED', 'Memory Bank đã sẵn sàng cho CHECK.', 'pass'];
  if (stage === 'TRAINING') return ['TRAINING', 'ESP32-S3 đang xây Memory Bank.', 'working'];
  if (stage === 'LEARN_READY_TRAIN') return ['READY TO TRAIN', 'Đủ 5 ảnh + 5 mẫu dòng. STM32 sẽ gửi TRAIN_START.', 'pass'];
  if (stage === 'ERROR') return ['ERROR', `ESP32 báo lỗi code ${status.error?.code ?? '—'}.`, 'fail'];
  if (familyStage(stage) === 'LEARN') return ['LEARNING', status.last_event || 'Đang thu mẫu.', 'working'];
  return ['WAITING', status.last_event || 'Chưa có kết quả inspection.', 'neutral'];
}

function familyStage(stage) {
  return stageFamily(stage);
}

function updateResultPanel(status) {
  const check = status.check || {};
  const [title, subtitle, kind] = resultMeta(status);
  const banner = $('result-banner');
  banner.className = `result-banner ${kind}`;
  setText('result-text', title);
  setText('result-subtext', subtitle);
  setText('board-large', status.board ? `#${status.board}` : '#—');

  const fail = Number(check.patch_fail_count);
  const total = Number(check.patch_total);
  setText('metric-patches', Number.isFinite(fail) && Number.isFinite(total) && total > 0 ? `${fail} / ${total}` : '—');
  setText('metric-dmin', formatFloat(check.max_dmin, 3));
  /* Chỉ hiện số đo dòng khi PCB hiện tại đã có kết quả dòng (tránh dính số của PCB trước).
   * Trong ngoặc: độ lệch cho phép quanh giá trị chuẩn (mean ± dev) học được ở LEARN. */
  const pr = Number(check.power_result);
  const havePower = pr === RESULT_PASS || pr === RESULT_FAIL;
  const imaxDev = check.imax_dev ?? check.imax_threshold;
  const iavgDev = check.iavg_dev ?? check.iavg_threshold;
  setText('metric-imax', havePower ? `${formatFloat(check.imax_measured, 1)} (±${formatFloat(imaxDev, 1)})` : '—');
  setText('metric-iavg', havePower ? `${formatFloat(check.iavg_measured, 1)} (±${formatFloat(iavgDev, 1)})` : '—');
  setText('metric-threshold', formatFloat(check.threshold, 3));

  const powerText = powerResultText(Number(check.power_result));
  setText('metric-power', powerText);
}

function powerResultText(result) {
  if (result === RESULT_PASS) return 'PASS';
  if (result === RESULT_FAIL) return 'FAIL';
  return '—';
}

function resultWord(r) {
  r = Number(r);
  return r === RESULT_PASS ? 'PASS' : (r === RESULT_FAIL ? 'FAIL' : '—');
}

function currentText(measured, mean, dev) {
  return `${formatFloat(measured, 1)} (chuẩn ${formatFloat(mean, 1)} ± ${formatFloat(dev, 1)})`;
}

let lastSummaryKey = '';

function updateSummary(status) {
  const s = status.summary || {};
  const hist = Array.isArray(status.history) ? status.history : [];
  const key = JSON.stringify([s, hist]);
  if (key === lastSummaryKey) return;          /* không vẽ lại khi dữ liệu không đổi */
  lastSummaryKey = key;

  const banner = $('summary-banner');
  const valid = !!s.valid;
  const total = Number(s.total) || 0;
  setText('summary-totals', `Đã kiểm ${total} PCB · PASS ${Number(s.pass) || 0} · FAIL ${Number(s.fail) || 0}`);

  if (!valid) {
    banner.className = 'result-banner neutral';
    setText('summary-board', '#—');
    setText('summary-text', 'CHƯA CÓ');
    setText('summary-sub', 'Chưa hoàn tất PCB nào trong phiên CHECK này.');
    ['sum-visual', 'sum-patches', 'sum-dmin', 'sum-vtime', 'sum-power', 'sum-imax', 'sum-iavg', 'sum-ptime']
      .forEach((id) => setText(id, '—'));
  } else {
    const pass = Number(s.final) === RESULT_PASS;
    const powerMeasured = Number(s.power) === RESULT_PASS || Number(s.power) === RESULT_FAIL;
    const patches = `${Number(s.patch_fail) || 0} / ${Number(s.patch_total) || 0}`;
    banner.className = `result-banner ${pass ? 'pass' : 'fail'}`;
    setText('summary-board', `#${s.board}`);
    setText('summary-text', `PCB #${s.board} · ${pass ? 'PASS' : 'FAIL'}`);
    setText('summary-sub', powerMeasured
      ? `Ngoại quan ${resultWord(s.visual)} (${patches} patch lỗi) · Dòng điện ${resultWord(s.power)}.`
      : `Ngoại quan FAIL: ${patches} patch lỗi · không đo dòng điện.`);
    setText('sum-visual', resultWord(s.visual));
    setText('sum-patches', patches);
    setText('sum-dmin', `${formatFloat(s.max_dmin, 3)} / ${formatFloat(s.v_threshold, 3)}`);
    setText('sum-vtime', formatMs(s.visual_ms));
    setText('sum-power', powerMeasured ? resultWord(s.power) : 'Không đo');
    setText('sum-imax', powerMeasured ? currentText(s.imax, s.imax_mean, s.imax_dev) : '—');
    setText('sum-iavg', powerMeasured ? currentText(s.iavg, s.iavg_mean, s.iavg_dev) : '—');
    setText('sum-ptime', powerMeasured ? formatMs(s.power_ms) : '—');
  }

  const body = $('history-body');
  if (!body) return;
  body.textContent = '';
  const rows = hist.slice().reverse();          /* mới nhất ở trên cùng */
  if (!rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 6; td.className = 'history-empty'; td.textContent = 'Chưa có PCB nào hoàn tất.';
    tr.appendChild(td); body.appendChild(tr);
    return;
  }
  rows.forEach((h) => {
    const measured = Number(h.power) === RESULT_PASS || Number(h.power) === RESULT_FAIL;
    const cells = [
      [`#${h.board}`, ''],
      [resultWord(h.visual), Number(h.visual) === RESULT_PASS ? 'pass' : 'fail'],
      [`${Number(h.patch_fail) || 0} / ${Number(h.patch_total) || 0}`, ''],
      [measured ? formatFloat(h.imax, 1) : '—', measured ? '' : 'skip'],
      [measured ? formatFloat(h.iavg, 1) : '—', measured ? '' : 'skip'],
      [resultWord(h.final), Number(h.final) === RESULT_PASS ? 'pass' : 'fail'],
    ];
    const tr = document.createElement('tr');
    cells.forEach(([text, cls]) => {
      const td = document.createElement('td');
      td.textContent = text;
      if (cls) td.className = cls;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function updateProcess(status) {
  const stage = status.stage || 'STANDBY';
  setText('stage-title', stage.replaceAll('_', ' '));
  setText('stage-code', stage);
  setText('process-message', status.last_event || 'Chờ lệnh từ STM32.');

  const learn = status.learn || {};
  const total = Number(learn.total) || 5;
  const visualDone = clamp(Number(learn.visual_done) || 0, 0, total);
  const powerDone = clamp(Number(learn.power_done) || 0, 0, total);
  setText('learn-visual-progress', `${visualDone} / ${total}`);
  setText('learn-power-progress', `${powerDone} / ${total}`);
  $('learn-visual-bar').style.width = `${(visualDone / total) * 100}%`;
  $('learn-power-bar').style.width = `${(powerDone / total) * 100}%`;

  setText('camera-status', status.camera ? 'ONLINE' : 'WAIT');
  setText('ai-status', status.ai_ready ? 'READY' : 'WAIT');
  setText('bank-status', status.bank_ready ? `${status.train?.bank_count || 0} vectors` : 'NOT TRAINED');
  setText('capture-id', status.last_capture_id || '—');
}

function updateTraining(status) {
  const train = status.train || {};
  const trained = !!status.model_trained;
  const pill = $('model-pill');
  pill.textContent = trained ? 'TRAINED' : 'NOT TRAINED';
  pill.className = `status-pill ${trained ? '' : 'neutral'}`;
  setText('train-bank', trained ? (train.bank_count ?? '—') : '—');
  setText('train-threshold', trained ? formatFloat(train.image_threshold, 3) : '—');
  setText('train-imax-dev', trained ? formatFloat(train.imax_max_dev, 2) : '—');
  setText('train-iavg-dev', trained ? formatFloat(train.iavg_max_dev, 2) : '—');
  setText('train-time', trained ? formatMs(train.elapsed_ms) : '—');
}

function updateSystem(status) {
  const cameraPill = $('camera-pill');
  cameraPill.textContent = status.camera ? 'ONLINE' : 'WAITING';
  cameraPill.className = `status-pill ${status.camera ? '' : 'neutral'}`;
  setText('camera-detail', status.camera ? 'UVC · MJPEG · live' : 'UVC · waiting');
  setText('live-message', status.camera ? 'Live preview' : 'Chờ camera UVC');
  setText('event-text', status.last_event || 'Đang chờ STM32...');
  setText('board-chip', `PCB ${status.board ? `#${status.board}` : '—'}`);
  setText('seq-chip', `SEQ ${status.event_seq ?? '—'}`);
  document.body.classList.toggle('has-error', status.stage === 'ERROR');
}

function updateInspectionPill(status) {
  const check = status.check || {};
  const stage = status.stage || '';
  const pill = $('visual-pill');
  let text = 'NO RESULT';
  let cls = 'status-pill neutral';
  if (stage === 'CHECK_VISUAL') { text = 'AI RUNNING'; cls = 'status-pill working'; }
  else if (check.visual_result === RESULT_FAIL || stage === 'CHECK_VISUAL_FAIL') { text = 'VISUAL FAIL'; cls = 'status-pill fail'; }
  else if (check.visual_result === RESULT_PASS) { text = 'VISUAL PASS'; cls = 'status-pill pass'; }
  pill.textContent = text;
  pill.className = cls;
}

function setInspectionMode(status) {
  const check = status.check || {};
  const stage = status.stage || 'STANDBY';
  const hasCurrent = Number(check.capture_id) > 0;
  const hasImage = inspectionShownId > 0;
  const inspectionWorking = $('inspection-working');
  const placeholder = $('inspection-placeholder');

  const isCheckVisualWorking = stage === 'CHECK_VISUAL' && !hasCurrent;
  const isLearnCapturing = stage === 'LEARN_CAPTURING';
  const isWorking = (isCheckVisualWorking || isLearnCapturing) && isStatusFresh();
  inspectionWorking.classList.toggle('hidden', !isWorking);
  /* Placeholder CHỈ hiện khi chưa từng có ảnh nào được hiển thị. */
  placeholder.classList.toggle('hidden', isWorking || hasImage);

  if (isCheckVisualWorking) {
    setText('working-title', `PCB #${status.board || '—'} · ESP32-S3 đang chụp + AI`);
    setText('working-subtitle', 'STM32 đang chờ VISUAL_RESULT');
  } else if (isLearnCapturing) {
    setText('working-title', 'LEARN · ESP32-S3 đang chụp mẫu');
    setText('working-subtitle', `Mẫu ngoại quan ${status.learn?.visual_working || '—'} / ${status.learn?.total || 5}`);
  }

  if (!hasImage && stage === 'CHECK_WAIT_PCB') {
    placeholder.querySelector('strong').textContent = `Chờ PCB #${status.board || '—'}`;
    placeholder.querySelector('span').textContent = 'Ảnh sẽ tự xuất hiện khi STM32 yêu cầu kiểm tra ngoại quan.';
  }
}

function wantedCaptureId(status) {
  const check = status.check || {};
  const stage = status.stage || '';
  if (Number(check.capture_id) > 0) return Number(check.capture_id);
  if (stageFamily(stage) === 'LEARN' && Number(status.last_capture_id) > 0) return Number(status.last_capture_id);
  if (stage === 'TRAINED' && Number(status.last_capture_id) > 0) return Number(status.last_capture_id);
  return 0;
}

function refreshCaptureImage(status) {
  const id = wantedCaptureId(status);
  if (!id || id === inspectionShownId || id === inspectionLoadingId) return;
  /* Tải lỗi (ESP đang bận AI) -> chờ 1 s rồi tự thử lại ở lần poll kế tiếp. */
  if (Date.now() - inspectionFailAt < 1000) return;
  loadInspectionImage(id);
}

/* Tải ảnh bằng fetch -> blob: ảnh cũ vẫn hiển thị cho tới khi ảnh mới sẵn sàng
 * (không chớp trắng/đen), và có timeout + retry thay vì bỏ cuộc im lặng. */
async function loadInspectionImage(id) {
  inspectionLoadingId = id;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  try {
    const response = await fetch(`/api/last-image?id=${id}&t=${Date.now()}`, { cache: 'no-store', signal: ctrl.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    if (blob.size < 100) throw new Error('empty image');
    const url = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      inspectionImage.onload = resolve;
      inspectionImage.onerror = () => reject(new Error('decode failed'));
      inspectionImage.src = url;
    });
    if (inspectionBlobUrl) URL.revokeObjectURL(inspectionBlobUrl);
    inspectionBlobUrl = url;
    inspectionImage.style.visibility = 'visible';
    inspectionShownId = id;
    inspectionFailAt = 0;
    $('inspection-placeholder').classList.add('hidden');
    if (latestStatus) {
      setInspectionMode(latestStatus);
      drawOverlay();
      renderFailedPatches(latestStatus);
    }
    setText('capture-detail', `Capture #${id}`);
  } catch (error) {
    inspectionFailAt = Date.now();
    console.warn('Tải ảnh inspection lỗi:', error);
    if (inspectionShownId === 0) setText('capture-detail', `Đang thử lại ảnh capture #${id}...`);
  } finally {
    clearTimeout(timer);
    if (inspectionLoadingId === id) inspectionLoadingId = 0;
  }
}

function drawOverlay() {
  const status = latestStatus;
  if (!status) return;
  const check = status.check || {};
  if (!inspectionImage.naturalWidth || !inspectionImage.naturalHeight || !inspectionStage.clientWidth || !inspectionStage.clientHeight) return;

  const cols = Number(check.cols) || 0;
  const rows = Number(check.rows) || 0;
  const cropWidth = Number(check.crop_width) || 0;
  const cropHeight = Number(check.crop_height) || 0;
  const threshold = Number(check.threshold);
  const scores = Array.isArray(check.scores) ? check.scores : [];
  if (!cols || !rows || !cropWidth || !cropHeight || !Number.isFinite(threshold) || !scores.length || Number(check.capture_id) <= 0 || Number(check.capture_id) !== inspectionShownId) {
    overlay.width = 1;
    overlay.height = 1;
    overlayCtx.clearRect(0, 0, 1, 1);
    return;
  }

  const rect = inspectionStage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(rect.width * dpr));
  overlay.height = Math.max(1, Math.round(rect.height * dpr));
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.clearRect(0, 0, rect.width, rect.height);

  const imageRatio = inspectionImage.naturalWidth / inspectionImage.naturalHeight;
  const boxRatio = rect.width / rect.height;
  let drawWidth, drawHeight, imageLeft, imageTop;
  if (imageRatio > boxRatio) {
    drawWidth = rect.width;
    drawHeight = drawWidth / imageRatio;
    imageLeft = 0;
    imageTop = (rect.height - drawHeight) / 2;
  } else {
    drawHeight = rect.height;
    drawWidth = drawHeight * imageRatio;
    imageTop = 0;
    imageLeft = (rect.width - drawWidth) / 2;
  }

  const cropX = Math.max(0, (inspectionImage.naturalWidth - cropWidth) / 2);
  const cropY = Math.max(0, (inspectionImage.naturalHeight - cropHeight) / 2);
  const scaleX = drawWidth / inspectionImage.naturalWidth;
  const scaleY = drawHeight / inspectionImage.naturalHeight;
  const cellW = (cropWidth / cols) * scaleX;
  const cellH = (cropHeight / rows) * scaleY;
  const baseX = imageLeft + cropX * scaleX;
  const baseY = imageTop + cropY * scaleY;

  // Draw a very light grid, then strongly highlight only failing patches.
  overlayCtx.lineWidth = 1;
  overlayCtx.strokeStyle = 'rgba(255,255,255,0.16)';
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i = r * cols + c;
      const x = baseX + c * cellW;
      const y = baseY + r * cellH;
      overlayCtx.strokeRect(x, y, cellW, cellH);
      const score = Number(scores[i]);
      if (Number.isFinite(score) && score > threshold) {
        overlayCtx.fillStyle = 'rgba(235, 38, 52, 0.32)';
        overlayCtx.fillRect(x, y, cellW, cellH);
        overlayCtx.strokeStyle = '#ff283c';
        overlayCtx.lineWidth = Math.max(2, dpr);
        overlayCtx.strokeRect(x + 1, y + 1, Math.max(0, cellW - 2), Math.max(0, cellH - 2));
        overlayCtx.strokeStyle = 'rgba(255,255,255,0.16)';
        overlayCtx.lineWidth = 1;
      }
    }
  }
}

function renderFailedPatches(status) {
  const list = $('patch-list');
  const empty = $('patch-empty');
  list.innerHTML = '';
  const check = status.check || {};
  const cols = Number(check.cols) || 0;
  const rows = Number(check.rows) || 0;
  const cropWidth = Number(check.crop_width) || 0;
  const cropHeight = Number(check.crop_height) || 0;
  const threshold = Number(check.threshold);
  const scores = Array.isArray(check.scores) ? check.scores : [];
  const failCount = Number(check.patch_fail_count) || 0;
  setText('danger-count', String(failCount));

  if (!inspectionImage.naturalWidth || !cols || !rows || !cropWidth || !cropHeight || !Number.isFinite(threshold) || failCount === 0 || Number(check.capture_id) !== inspectionShownId) {
    const node = document.createElement('div');
    node.className = 'patch-empty';
    node.textContent = 'Chưa có patch lỗi vượt ngưỡng.';
    list.appendChild(node);
    return;
  }

  const cropX = Math.max(0, (inspectionImage.naturalWidth - cropWidth) / 2);
  const cropY = Math.max(0, (inspectionImage.naturalHeight - cropHeight) / 2);
  const srcCellW = cropWidth / cols;
  const srcCellH = cropHeight / rows;
  const failed = scores
    .map((score, index) => ({ score: Number(score), index }))
    .filter((item) => Number.isFinite(item.score) && item.score > threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);

  failed.forEach(({ score, index }) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    const wrap = document.createElement('div');
    wrap.className = 'patch-card';

    const canvas = document.createElement('canvas');
    canvas.width = 120;
    canvas.height = 120;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(
      inspectionImage,
      cropX + col * srcCellW,
      cropY + row * srcCellH,
      srcCellW,
      srcCellH,
      0, 0, canvas.width, canvas.height
    );
    ctx.strokeStyle = '#ff283c';
    ctx.lineWidth = 5;
    ctx.strokeRect(2.5, 2.5, canvas.width - 5, canvas.height - 5);
    wrap.appendChild(canvas);

    const meta = document.createElement('div');
    meta.className = 'patch-meta';
    meta.innerHTML = `<strong>Patch ${index + 1}</strong><span>${score.toFixed(3)} &gt; ${threshold.toFixed(3)}</span>`;
    wrap.appendChild(meta);
    list.appendChild(wrap);
  });
}

function applyStatus(status) {
  latestStatus = status;
  updateSystem(status);
  updatePipeline(status.stage);
  updateProcess(status);
  updateResultPanel(status);
  updateSummary(status);
  updateTraining(status);
  updateInspectionPill(status);
  setInspectionMode(status);
  refreshCaptureImage(status);

  const check = status.check || {};
  if ((status.stage === 'CHECK_WAIT_PCB' || status.stage === 'CHECK_VISUAL') && Number(check.capture_id) <= 0) {
    $('patch-list').innerHTML = '<div class="patch-empty">Đang chờ ảnh inspection của PCB hiện tại.</div>';
    setText('danger-count', '0');
  }
  const failCount = Number(check.patch_fail_count) || 0;
  if (check.capture_id > 0) {
    setText('overlay-detail', failCount > 0 ? `${failCount} patch đỏ · vượt ngưỡng` : 'Visual PASS · không có patch đỏ');
    setText('capture-detail', `Capture #${check.capture_id} · ${check.cols || 0}×${check.rows || 0} · ${formatMs(check.elapsed_ms)}`);
  }
  if (lastStatusSeq !== Number(status.event_seq)) {
    lastStatusSeq = Number(status.event_seq);
    if (check.capture_id > 0 && inspectionShownId > 0) { drawOverlay(); renderFailedPatches(status); }
  }
}

let statusFailCount = 0;
let lastUptime = -1;
let lastStatusOkAt = 0;
const STATUS_TIMEOUT_MS = 4000;
const STATUS_FAIL_OFFLINE = 5;      /* chỉ báo OFFLINE sau 5 lần lỗi liên tiếp */

function isStatusFresh() {
  return Date.now() - lastStatusOkAt < 3000;
}

async function refreshStatus() {
  /* Timeout + chạy tuần tự (không chồng request): trước đây setInterval 500 ms
   * không có timeout nên khi ESP32 bận (chụp/AI) các request chồng lên nhau,
   * httpd nghẽn và trạng thái web bị đứng ở "đang chụp". */
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), STATUS_TIMEOUT_MS);
  try {
    const response = await fetch('/api/status', { cache: 'no-store', signal: ctrl.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    statusFailCount = 0;
    lastStatusOkAt = Date.now();
    if (Number.isFinite(Number(status.uptime_s))) {
      if (lastUptime >= 0 && Number(status.uptime_s) + 2 < lastUptime) {
        /* uptime giảm => ESP32 vừa reset (brown-out / crash / WDT) */
        console.warn('ESP32 vừa khởi động lại, reset_reason =', status.reset_reason);
        setText('event-text', `ESP32 vừa khởi động lại (lý do: ${status.reset_reason || '?'})`);
        startStream(true);
      }
      lastUptime = Number(status.uptime_s);
    }
    applyStatus(status);
  } catch (error) {
    statusFailCount++;
    /* Nếu video vẫn đang về thì ESP32 còn sống, chỉ là /api/status chậm. */
    const streamAlive = Date.now() - lastFrameAt < 3000;
    if (statusFailCount >= STATUS_FAIL_OFFLINE && !streamAlive) {
      setText('event-text', 'Mất kết nối ESP32-S3 Web Console - đang thử lại...');
      $('camera-pill').textContent = 'OFFLINE';
      $('camera-pill').className = 'status-pill fail';
      /* Mất kết nối thì không để spinner "đang chụp" quay mãi */
      $('inspection-working').classList.add('hidden');
    }
  } finally {
    clearTimeout(timer);
    setTimeout(refreshStatus, statusFailCount ? 800 : 400);
  }
}

/* ---------------- LIVE STREAM (MJPEG qua fetch) ----------------
 * Bản cũ gán thẳng <img src=".../stream">. Với multipart/x-mixed-replace trình
 * duyệt KHÔNG báo lỗi khi luồng treo/đứt nên khung cứ đen hoặc đứng hình mà JS
 * không biết để nối lại. Bản này tự đọc luồng, tách từng JPEG, và:
 *  - hiển thị frame mới bằng blob (ảnh cũ giữ nguyên tới khi frame mới giải mã xong
 *    => không chớp đen),
 *  - watchdog: >4 s không có frame => huỷ kết nối và nối lại (backoff),
 *  - chỉ có đúng MỘT kết nối stream tại mọi thời điểm,
 *  - tab bị ẩn thì ngắt stream để nhường băng thông Wi-Fi cho ESP32. */
const STREAM_URL = `http://${location.hostname}:81/stream`;
const STREAM_STALL_MS = 4000;
const STREAM_MAX_FRAME = 2 * 1024 * 1024;
let streamAbort = null;
let streamRunning = false;
let streamWanted = true;
let streamBackoffMs = 500;
let streamBlobUrl = null;
let lastFrameAt = 0;
let fetchFailStreak = 0;
let imgFallback = false;
const latin1 = new TextDecoder('latin1');
const HEADER_RE = /--vedcframe\r\nContent-Type: image\/jpeg\r\nContent-Length: (\d+)\r\n\r\n/;

function showStreamFrame(bytes) {
  const url = URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
  const prev = streamBlobUrl;
  streamBlobUrl = url;
  streamImage.src = url;
  streamImage.style.visibility = 'visible';
  if (prev) URL.revokeObjectURL(prev);
}

async function runStream() {
  if (streamRunning) return;
  streamRunning = true;
  const ctrl = new AbortController();
  streamAbort = ctrl;
  lastFrameAt = Date.now();
  const watchdog = setInterval(() => {
    if (Date.now() - lastFrameAt > STREAM_STALL_MS) ctrl.abort();
  }, 1000);

  let buf = new Uint8Array(256 * 1024);
  let len = 0;
  let gotFrame = false;
  try {
    const response = await fetch(`${STREAM_URL}?t=${Date.now()}`, { mode: 'cors', signal: ctrl.signal });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (len + value.length > buf.length) {
        const bigger = new Uint8Array(Math.max(buf.length * 2, len + value.length));
        bigger.set(buf.subarray(0, len));
        buf = bigger;
      }
      buf.set(value, len);
      len += value.length;

      /* tách tất cả frame hoàn chỉnh đang có trong buffer; chỉ hiển thị frame cuối */
      let latest = null;
      while (true) {
        const m = HEADER_RE.exec(latin1.decode(buf.subarray(0, Math.min(len, 512))));
        if (!m) {
          if (len > 2048) { buf.copyWithin(0, len - 256, len); len = 256; }   /* mất đồng bộ -> bỏ rác */
          break;
        }
        const size = Number(m[1]);
        const start = m.index + m[0].length;
        if (!size || size > STREAM_MAX_FRAME) { buf.copyWithin(0, start, len); len -= start; continue; }
        if (len < start + size) break;                                          /* chưa đủ dữ liệu */
        latest = buf.slice(start, start + size);
        buf.copyWithin(0, start + size, len);
        len -= start + size;
      }
      if (latest) {
        showStreamFrame(latest);
        lastFrameAt = Date.now();
        if (!gotFrame) { gotFrame = true; streamBackoffMs = 500; setText('live-message', 'Live preview'); }
      }
    }
  } catch (error) {
    /* AbortError do watchdog/ẩn tab là bình thường */
  } finally {
    clearInterval(watchdog);
    streamRunning = false;
    streamAbort = null;
    fetchFailStreak = gotFrame ? 0 : fetchFailStreak + 1;
    if (streamWanted && fetchFailStreak >= 3 && !imgFallback) {
      startImgFallback();               /* fetch bị chặn/không hỗ trợ -> dùng <img> MJPEG kiểu cũ */
    } else if (streamWanted && !imgFallback) {
      setText('live-message', 'Đang kết nối lại camera...');   /* frame cuối vẫn hiển thị */
      clearTimeout(streamRetryTimer);
      streamRetryTimer = setTimeout(runStream, streamBackoffMs);
      streamBackoffMs = Math.min(streamBackoffMs * 2, 4000);
    }
  }
}

/* Đường lùi: <img src=/stream> thuần. Trình duyệt tự giải multipart; nếu lỗi thì
 * gán lại src sau 1.5 s (không đụng tới DOM nên frame cuối vẫn hiển thị). */
function startImgFallback() {
  imgFallback = true;
  console.warn('Stream fetch thất bại nhiều lần, chuyển sang <img> MJPEG');
  const connect = () => {
    if (!streamWanted) return;
    streamImage.src = `${STREAM_URL}?t=${Date.now()}`;
  };
  streamImage.onload = () => {
    streamImage.style.visibility = 'visible';
    lastFrameAt = Date.now();
    setText('live-message', 'Live preview');
  };
  streamImage.onerror = () => {
    setText('live-message', 'Đang kết nối lại camera...');
    clearTimeout(streamRetryTimer);
    streamRetryTimer = setTimeout(connect, 1500);
  };
  connect();
}

function startStream(force = false) {
  streamWanted = true;
  if (imgFallback) { streamImage.src = `${STREAM_URL}?t=${Date.now()}`; return; }
  if (streamRunning) {
    if (force && streamAbort) streamAbort.abort();   /* finally{} sẽ nối lại */
    return;
  }
  clearTimeout(streamRetryTimer);
  runStream();
}

function stopStream() {
  streamWanted = false;
  clearTimeout(streamRetryTimer);
  if (streamAbort) streamAbort.abort();
  if (imgFallback) streamImage.removeAttribute('src');   /* ngắt kết nối khi ẩn tab */
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopStream();
  else { streamBackoffMs = 500; startStream(); }
});
window.addEventListener('pagehide', stopStream);
window.addEventListener('resize', () => {
  drawOverlay();
  if (latestStatus) renderFailedPatches(latestStatus);
});

startStream();
refreshStatus();
