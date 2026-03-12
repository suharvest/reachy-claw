// ── Emotion Mirror Dashboard — Fullscreen HUD ──────────────────────

// ── Config ──────────────────────────────────────────────────────────
const VISION_HOST = location.hostname + ':8630';
const VISION_API = `http://${VISION_HOST}`;
const DASHBOARD_WS = `ws://${location.host}/ws`;

const EMOTION_COLORS = {
    angry: '#e94560', happy: '#2ecc71', neutral: '#3498db',
    sad: '#2c3e50', surprised: '#f39c12', fear: '#9b59b6',
    thinking: '#f39c12', confused: '#a0a0c0', curious: '#00d2ff',
    excited: '#2ecc71', laugh: '#2ecc71', listening: '#3498db',
    Anger: '#e94560', Happiness: '#2ecc71', Neutral: '#3498db',
    Sadness: '#2c3e50', Surprise: '#f39c12', Fear: '#9b59b6',
    Contempt: '#a0a0c0', Disgust: '#8b5e3c',
};

// ── State ───────────────────────────────────────────────────────────
let visionWs = null;
let dashboardWs = null;
let latestFaces = null;
let currentLlmText = '';
let lastLlmText = '';
let currentRunId = null;
let isStreaming = false;
let currentMode = 'conversation';
let uploadFiles = [];
let asrIdleTimer = null;

// ── DOM refs ────────────────────────────────────────────────────────
const videoEl = document.getElementById('video-stream');
const canvasEl = document.getElementById('overlay-canvas');
const ctx = canvasEl.getContext('2d');
const noVideoEl = document.getElementById('no-video');
const asrTextEl = document.getElementById('asr-text');
const monologueTextEl = document.getElementById('monologue-text');
const monologueBubble = document.getElementById('monologue-bubble');

// ── Toast ───────────────────────────────────────────────────────────
function showToast(msg, isError = false) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = isError ? 'toast show error' : 'toast show';
    setTimeout(() => { el.className = 'toast'; }, 2500);
}

// ── Vision WebSocket (face detection) ───────────────────────────────
let visionRetry = 1000;

function connectVision() {
    const url = `ws://${VISION_HOST}/ws`;
    visionWs = new WebSocket(url);

    visionWs.onopen = () => {
        console.log('Vision WS connected');
        visionRetry = 1000;
        document.getElementById('dot-vision').className = 'dot live';
    };

    visionWs.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            latestFaces = data.faces || [];
        } catch(err) {
            console.error('Vision WS parse error:', err);
        }
    };

    visionWs.onclose = () => {
        document.getElementById('dot-vision').className = 'dot off';
        latestFaces = null;
        visionRetry = Math.min(visionRetry * 1.5, 10000);
        setTimeout(connectVision, visionRetry);
    };

    visionWs.onerror = () => visionWs.close();
}

// ── Dashboard WebSocket (ASR, LLM, state) ───────────────────────────
let dashRetry = 1000;

function connectDashboard() {
    dashboardWs = new WebSocket(DASHBOARD_WS);

    dashboardWs.onopen = () => {
        console.log('Dashboard WS connected');
        dashRetry = 1000;
        document.getElementById('dot-jetson').className = 'dot live';
    };

    dashboardWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            handleDashboardMsg(msg);
        } catch(err) {
            console.error('Dashboard WS parse error:', err);
        }
    };

    dashboardWs.onclose = () => {
        document.getElementById('dot-jetson').className = 'dot off';
        dashRetry = Math.min(dashRetry * 1.5, 10000);
        setTimeout(connectDashboard, dashRetry);
    };

    dashboardWs.onerror = () => dashboardWs.close();
}

function handleDashboardMsg(msg) {
    switch(msg.type) {
        case 'asr_partial':
            asrTextEl.innerHTML = msg.text;
            asrTextEl.className = 'asr-text partial';
            resetAsrIdleTimer();
            break;

        case 'asr_final':
            asrTextEl.innerHTML = msg.text;
            asrTextEl.className = 'asr-text';
            resetAsrIdleTimer();
            break;

        case 'observation':
            updateObservation(msg.text);
            break;

        case 'llm_delta':
            if (msg.run_id !== currentRunId) {
                currentRunId = msg.run_id;
                currentLlmText = '';
                isStreaming = true;
            }
            currentLlmText += msg.text;
            updateMonologue();
            break;

        case 'llm_end':
            if (msg.run_id === currentRunId) {
                currentLlmText = msg.full_text;
                lastLlmText = currentLlmText;
                isStreaming = false;
                currentRunId = null;
                updateMonologue();
            }
            break;

        case 'state':
            updateState(msg.state);
            break;

        case 'emotion':
            break;

        case 'robot_state':
            updateRobotState(msg);
            break;

        case 'mode_changed':
            currentMode = msg.mode;
            syncModeUI();
            showToast('Mode: ' + msg.mode);
            break;

        case 'prompts':
            document.getElementById('prompt-conversation').value = msg.conversation || '';
            document.getElementById('prompt-monologue').value = msg.monologue || '';
            break;

        case 'prompt_saved':
            showToast('Prompt saved: ' + msg.mode);
            break;
    }
}

// ── ASR idle timer ──────────────────────────────────────────────────
function resetAsrIdleTimer() {
    if (asrIdleTimer) clearTimeout(asrIdleTimer);
    asrIdleTimer = setTimeout(() => {
        asrTextEl.innerHTML = '<i>\u503E\u542C\u4E2D...</i>';
        asrTextEl.className = 'asr-text idle';
    }, 5000);
}

// ── Observation (monologue mode input) ──────────────────────────────
let lastObsText = '';
function updateObservation(text) {
    lastObsText = text;
    // Show observation as a small line above the monologue bubble
    const obsEl = document.getElementById('observation-text');
    if (obsEl) {
        obsEl.textContent = text;
        obsEl.style.display = 'block';
    }
}

// ── Monologue bubble ────────────────────────────────────────────────
function updateMonologue() {
    if (isStreaming) {
        monologueTextEl.innerHTML = currentLlmText + '<span class="typing-cursor"></span>';
        monologueTextEl.className = 'monologue-text';
    } else {
        const text = lastLlmText || '...';
        monologueTextEl.textContent = text;
        monologueTextEl.className = 'monologue-text idle';
    }
}

// ── State & Robot State ─────────────────────────────────────────────
function updateState(state) {
    const el = document.getElementById('robot-state');
    if (el) {
        el.textContent = state;
        el.dataset.state = state;
    }
}

function updateRobotState(msg) {
    // Mode
    if (msg.mode) {
        currentMode = msg.mode;
        syncModeUI();
    }

    // Robot connected indicator
    document.getElementById('dot-robot').className = 'dot live';
}

// ── Canvas overlay (face detection) ─────────────────────────────────
function getImageRect() {
    // Calculate actual rendered image area within object-fit: contain element
    const rect = videoEl.getBoundingClientRect();
    const natW = videoEl.naturalWidth || 640;
    const natH = videoEl.naturalHeight || 360;
    const scale = Math.min(rect.width / natW, rect.height / natH);
    const imgW = natW * scale;
    const imgH = natH * scale;
    const offsetX = (rect.width - imgW) / 2;
    const offsetY = (rect.height - imgH) / 2;
    return { offsetX, offsetY, imgW, imgH, fullW: rect.width, fullH: rect.height };
}

function drawOverlay() {
    const rect = videoEl.getBoundingClientRect();
    canvasEl.width = rect.width;
    canvasEl.height = rect.height;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!latestFaces || latestFaces.length === 0) {
        requestAnimationFrame(drawOverlay);
        return;
    }

    // Map normalized coords to actual image area (not full canvas)
    const ir = getImageRect();
    const cw = ir.imgW;
    const ch = ir.imgH;
    const ox = ir.offsetX;
    const oy = ir.offsetY;

    for (const face of latestFaces) {
        const [x1, y1, x2, y2] = face.bbox;
        const color = EMOTION_COLORS[face.emotion] || '#3498db';

        const bx = ox + x1 * cw;
        const by = oy + y1 * ch;
        const bw = (x2 - x1) * cw;
        const bh = (y2 - y1) * ch;

        const cornerLen = Math.min(bw, bh) * 0.2;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;

        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(bx, by + cornerLen);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + cornerLen, by);
        ctx.stroke();

        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + cornerLen);
        ctx.stroke();

        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - cornerLen);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + cornerLen, by + bh);
        ctx.stroke();

        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - cornerLen);
        ctx.stroke();

        ctx.globalAlpha = 1.0;

        // ── Identity label (above bbox) ──
        const identity = face.identity;
        if (identity && identity !== '?') {
            ctx.font = 'bold 12px monospace';
            const idMetrics = ctx.measureText(identity);
            const idW = idMetrics.width + 12;
            const idH = 18;
            let idY = by - idH - 4;
            if (idY < 2) idY = by + 4;
            const idX = bx + (bw - idW) / 2;

            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            roundRect(ctx, idX, idY, idW, idH, 4);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.fillText(identity, idX + 6, idY + 13);
        }

        // ── Emotion pill (below bbox) ──
        const emotion = face.emotion || 'neutral';
        const conf = ((face.emotion_confidence || 0) * 100).toFixed(0);
        const pillText = conf > 0 ? `${emotion} ${conf}%` : emotion;
        ctx.font = '11px monospace';
        const pillMetrics = ctx.measureText(pillText);
        const pillW = pillMetrics.width + 16;
        const pillH = 20;
        const pillX = bx + (bw - pillW) / 2;
        let pillY = by + bh + 6;
        if (pillY + pillH > oy + ch - 4) pillY = by + bh - pillH - 4;

        // Pill background
        const pillAlpha = 0.7;
        ctx.globalAlpha = pillAlpha;
        ctx.fillStyle = hexToRgba(color, 0.25);
        roundRect(ctx, pillX, pillY, pillW, pillH, 10);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(color, 0.5);
        ctx.lineWidth = 1;
        roundRect(ctx, pillX, pillY, pillW, pillH, 10);
        ctx.stroke();

        ctx.globalAlpha = 1.0;
        ctx.fillStyle = color;
        ctx.fillText(pillText, pillX + 8, pillY + 14);

        // ── 5-point landmarks ──
        if (face.landmarks) {
            ctx.fillStyle = '#00d2ff';
            for (const [lx, ly] of face.landmarks) {
                ctx.beginPath();
                ctx.arc(ox + lx * cw, oy + ly * ch, 2.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }

    requestAnimationFrame(drawOverlay);
}

// ── Canvas helpers ──────────────────────────────────────────────────
function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// ── MJPEG video (fetch + ReadableStream, vision-trt has CORS enabled) ─
function setupVideo() {
    const streamUrl = `http://${VISION_HOST}/stream`;

    async function readStream() {
        try {
            const res = await fetch(streamUrl);
            const reader = res.body.getReader();
            let buf = new Uint8Array(0);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Append chunk
                const tmp = new Uint8Array(buf.length + value.length);
                tmp.set(buf); tmp.set(value, buf.length);
                buf = tmp;

                // Extract complete JPEG frames (FFD8..FFD9)
                let start = -1;
                for (let i = 0; i < buf.length - 1; i++) {
                    if (buf[i] === 0xFF && buf[i+1] === 0xD8) start = i;
                    if (buf[i] === 0xFF && buf[i+1] === 0xD9 && start >= 0) {
                        const jpeg = buf.slice(start, i + 2);
                        const blob = new Blob([jpeg], { type: 'image/jpeg' });
                        const url = URL.createObjectURL(blob);
                        const prev = videoEl.src;
                        videoEl.src = url;
                        videoEl.style.display = 'block';
                        noVideoEl.style.display = 'none';
                        if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
                        buf = buf.slice(i + 2);
                        break;  // process one frame per read cycle
                    }
                }
            }
        } catch (e) {
            console.warn('MJPEG stream error:', e);
            videoEl.style.display = 'none';
            noVideoEl.style.display = 'flex';
            setTimeout(readStream, 3000);
        }
    }

    readStream();
}

// ── Settings Panel ──────────────────────────────────────────────────
function initSettings() {
    const overlay = document.getElementById('settings-overlay');
    document.getElementById('settings-open').onclick = () => {
        overlay.classList.add('open');
        loadFaces();
    };
    document.getElementById('settings-close').onclick = () => {
        overlay.classList.remove('open');
    };
    overlay.onclick = (e) => {
        if (e.target === overlay) overlay.classList.remove('open');
    };

    // Tabs
    document.querySelectorAll('.settings-tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            if (tab.dataset.tab === 'prompt') loadPrompts();
        };
    });

    // Mode selection
    document.querySelectorAll('.mode-option').forEach(opt => {
        opt.onclick = () => {
            const mode = opt.dataset.mode;
            if (mode === currentMode) return;
            setMode(mode);
        };
    });

    // Live enroll
    document.getElementById('enroll-live-btn').onclick = enrollLive;

    // File upload
    const uploadArea = document.getElementById('upload-area');
    const uploadInput = document.getElementById('upload-input');

    uploadArea.onclick = () => uploadInput.click();
    uploadInput.onchange = () => handleUploadFiles(uploadInput.files);

    uploadArea.ondragover = (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); };
    uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
    uploadArea.ondrop = (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleUploadFiles(e.dataTransfer.files);
    };

    document.getElementById('upload-btn').onclick = uploadAndEnroll;

    // Export / Import
    document.getElementById('export-btn').onclick = exportFaces;
    document.getElementById('import-btn').onclick = () => document.getElementById('import-input').click();
    document.getElementById('import-input').onchange = importFaces;

    // Prompt tab
    document.getElementById('prompt-conv-save').onclick = () => savePrompt('conversation');
    document.getElementById('prompt-mono-save').onclick = () => savePrompt('monologue');
    document.getElementById('prompt-conv-reset').onclick = () => {
        document.getElementById('prompt-conversation').value = '';
        savePrompt('conversation', '');
    };
    document.getElementById('prompt-mono-reset').onclick = () => {
        document.getElementById('prompt-monologue').value = '';
        savePrompt('monologue', '');
    };
}

function syncModeUI() {
    document.querySelectorAll('.mode-option').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.mode === currentMode);
        opt.querySelector('input').checked = opt.dataset.mode === currentMode;
    });
    document.getElementById('mode-status').textContent = 'Current: ' + currentMode;
}

function setMode(mode) {
    if (!dashboardWs || dashboardWs.readyState !== 1) {
        showToast('Not connected', true);
        return;
    }
    dashboardWs.send(JSON.stringify({ type: 'set_mode', mode }));
}

// ── Face Management API ─────────────────────────────────────────────
async function loadFaces() {
    try {
        const res = await fetch(`${VISION_API}/api/faces`);
        const data = await res.json();
        renderFaceList(data.faces || []);
    } catch (e) {
        renderFaceList([]);
    }
}

function renderFaceList(faces) {
    const el = document.getElementById('face-list');
    if (!faces.length) {
        el.innerHTML = '<div class="face-empty">No faces registered</div>';
        return;
    }
    el.innerHTML = faces.map(name =>
        `<div class="face-item">
            <span class="face-name">${name}</span>
            <button class="face-item-btn" onclick="deleteFace('${name}')">Delete</button>
        </div>`
    ).join('');
}

async function deleteFace(name) {
    try {
        await fetch(`${VISION_API}/api/faces/${encodeURIComponent(name)}`, { method: 'DELETE' });
        showToast(`Deleted: ${name}`);
        loadFaces();
    } catch (e) {
        showToast('Delete failed', true);
    }
}

async function enrollLive() {
    const name = document.getElementById('enroll-name').value.trim();
    if (!name) { showToast('Enter a name', true); return; }

    const btn = document.getElementById('enroll-live-btn');
    btn.disabled = true;
    try {
        const res = await fetch(`${VISION_API}/api/faces/enroll?name=${encodeURIComponent(name)}`, { method: 'POST' });
        const data = await res.json();
        if (data.error) { showToast(data.error, true); return; }
        showToast(`Enrolled: ${name}`);
        document.getElementById('enroll-name').value = '';
        loadFaces();
    } catch (e) {
        showToast('Enroll failed', true);
    } finally {
        btn.disabled = false;
    }
}

function handleUploadFiles(files) {
    uploadFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    const preview = document.getElementById('upload-preview');
    preview.innerHTML = '';
    uploadFiles.forEach(f => {
        const img = document.createElement('img');
        img.className = 'upload-thumb';
        img.src = URL.createObjectURL(f);
        preview.appendChild(img);
    });
    document.getElementById('upload-btn').disabled = uploadFiles.length === 0;
}

async function uploadAndEnroll() {
    const name = document.getElementById('upload-name').value.trim();
    if (!name) { showToast('Enter a name', true); return; }
    if (!uploadFiles.length) return;

    const btn = document.getElementById('upload-btn');
    btn.disabled = true;
    let ok = 0, fail = 0;

    for (const file of uploadFiles) {
        const fd = new FormData();
        fd.append('name', name);
        fd.append('image', file);
        try {
            const res = await fetch(`${VISION_API}/api/faces/enroll-image`, { method: 'POST', body: fd });
            const data = await res.json();
            if (data.error) { fail++; } else { ok++; }
        } catch (e) {
            fail++;
        }
    }

    showToast(`Enrolled ${ok}/${uploadFiles.length} images` + (fail ? ` (${fail} failed)` : ''));
    uploadFiles = [];
    document.getElementById('upload-preview').innerHTML = '';
    document.getElementById('upload-input').value = '';
    btn.disabled = true;
    loadFaces();
}

async function exportFaces() {
    try {
        const res = await fetch(`${VISION_API}/api/faces/export`);
        if (!res.ok) { showToast('Export failed', true); return; }
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'faces.zip';
        a.click();
        URL.revokeObjectURL(a.href);
        showToast('Exported faces.zip');
    } catch (e) {
        showToast('Export failed', true);
    }
}

async function importFaces() {
    const input = document.getElementById('import-input');
    if (!input.files.length) return;

    const fd = new FormData();
    fd.append('file', input.files[0]);

    try {
        const res = await fetch(`${VISION_API}/api/faces/import`, { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) { showToast(data.error, true); return; }
        showToast(`Imported ${(data.faces || []).length} faces`);
        loadFaces();
    } catch (e) {
        showToast('Import failed', true);
    }
    input.value = '';
}

// ── Prompt Management ────────────────────────────────────────────────
function loadPrompts() {
    if (!dashboardWs || dashboardWs.readyState !== 1) return;
    dashboardWs.send(JSON.stringify({ type: 'get_prompts' }));
}

function savePrompt(mode, text) {
    if (!dashboardWs || dashboardWs.readyState !== 1) {
        showToast('Not connected', true);
        return;
    }
    if (text === undefined) {
        const id = mode === 'conversation' ? 'prompt-conversation' : 'prompt-monologue';
        text = document.getElementById(id).value;
    }
    dashboardWs.send(JSON.stringify({ type: 'set_prompt', mode, prompt: text }));
}

// ── Init ────────────────────────────────────────────────────────────
function init() {
    setupVideo();
    connectVision();
    connectDashboard();
    initSettings();
    requestAnimationFrame(drawOverlay);
}

init();
