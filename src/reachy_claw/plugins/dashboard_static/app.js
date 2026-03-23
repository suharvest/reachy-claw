// ── Emotion Mirror Dashboard — 3-Column Layout ─────────────────────

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

const EMOTION_EMOJI = {
    happy: '\u{1F60A}', sad: '\u{1F622}', thinking: '\u{1F914}',
    surprised: '\u{1F631}', curious: '\u{1F9D0}', excited: '\u{1F929}',
    neutral: '\u{1F610}', confused: '\u{1F615}', angry: '\u{1F620}',
    laugh: '\u{1F602}', fear: '\u{1F628}', listening: '\u{1F3A7}',
};

// ── State ───────────────────────────────────────────────────────────
let visionWs = null;
let dashboardWs = null;
let latestFaces = null;      // from vision WS
let latestFacesTs = 0;       // Date.now() of last vision WS update
let latestVisionFaces = [];  // from dashboard WS (vision_faces event)
let currentLlmText = '';
let currentRunId = null;
let isStreaming = false;
let currentMode = 'conversation';
let uploadFiles = [];
let asrIdleTimer = null;
let captureCount = 0;
let asrActive = false;
let asrActiveTimer = null;

// Thought bubble history (max 3)
const MAX_THOUGHTS = 5;

// ── DOM refs ────────────────────────────────────────────────────────
const videoEl = document.getElementById('video-stream');
const canvasEl = document.getElementById('overlay-canvas');
const ctx = canvasEl.getContext('2d');
const noVideoEl = document.getElementById('no-video');
const asrTextEl = document.getElementById('asr-text');
const faceCropCanvas = document.getElementById('face-crop-canvas');
const faceCropCtx = faceCropCanvas.getContext('2d');
const emotionPill = document.getElementById('emotion-pill');
const emotionLabel = document.getElementById('emotion-label');
const captureCountEl = document.getElementById('capture-count');
const thoughtList = document.getElementById('thought-list');

// ── Toast ───────────────────────────────────────────────────────────
function showToast(msg, isError = false) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = isError ? 'toast show error' : 'toast show';
    setTimeout(() => { el.className = 'toast'; }, 2500);
}

// ── Waveform bars init ──────────────────────────────────────────────
const waveformImg = document.getElementById('waveform-img');

function animateWaveform(active) {
    if (waveformImg) {
        if (active) {
            waveformImg.style.opacity = '1';
            waveformImg.style.filter = 'drop-shadow(0 0 8px rgba(163, 230, 53, 0.6))';
        } else {
            waveformImg.style.opacity = '0.4';
            waveformImg.style.filter = 'drop-shadow(0 0 2px rgba(163, 230, 53, 0.2))';
        }
    }
}

// Run waveform animation loop (now just purely state based from ASR)
setInterval(() => animateWaveform(asrActive), 200);

// ── Vision WebSocket (face detection data) ──────────────────────────
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
            latestFacesTs = Date.now();
            updateEmotionDisplay();
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
        // Clear restart status on reconnect (reachy-claw restarted itself)
        const restartStatus = document.getElementById('restart-status');
        const restartBtn = document.getElementById('restart-btn');
        if (restartStatus && restartStatus.textContent) {
            restartStatus.textContent = 'All services restarted.';
            setTimeout(() => { restartStatus.textContent = ''; }, 3000);
        }
        if (restartBtn) restartBtn.disabled = false;
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
        // Reset streaming state so a mid-stream disconnect doesn't leave
        // a permanent "typing" card with blinking cursor.
        if (isStreaming) {
            isStreaming = false;
            currentRunId = null;
            finalizeThoughtCard(null);
        }
        dashRetry = Math.min(dashRetry * 1.5, 10000);
        setTimeout(connectDashboard, dashRetry);
    };

    dashboardWs.onerror = () => dashboardWs.close();
}

function handleDashboardMsg(msg) {
    switch(msg.type) {
        case 'asr_partial':
            asrTextEl.textContent = msg.text;
            asrTextEl.className = 'asr-text partial';
            triggerAsrActive();
            resetAsrIdleTimer();
            break;

        case 'asr_final':
            if (msg.text) {
                asrTextEl.textContent = msg.text;
                asrTextEl.className = 'asr-text';
                triggerAsrActive();
                resetAsrIdleTimer();
            } else {
                // Empty transcript — reset to listening immediately
                if (asrIdleTimer) clearTimeout(asrIdleTimer);
                asrTextEl.innerHTML = '<i>Listening...</i>';
                asrTextEl.className = 'asr-text idle';
                asrActive = false;
            }
            break;

        case 'observation':
            // Observation context in monologue mode (vision description)
            break;

        case 'llm_delta':
            if (msg.run_id !== currentRunId) {
                currentRunId = msg.run_id;
                currentLlmText = '';
                isStreaming = true;
                addStreamingCard();
            }
            currentLlmText += msg.text;
            updateStreamingCard();
            break;

        case 'llm_end':
            if (msg.run_id === currentRunId) {
                currentLlmText = msg.full_text;
                isStreaming = false;
                currentRunId = null;
                finalizeThoughtCard(msg.emotion);
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

        case 'vision_faces':
            latestVisionFaces = msg.faces || [];
            updateEmotionDisplay();
            break;

        case 'smile_capture':
            captureCount = msg.count || 0;
            captureCountEl.textContent = captureCount;
            spawnFloatPlus();
            break;

        case 'capture_reset':
            captureCount = 0;
            captureCountEl.textContent = '0';
            break;

        case 'mode_changed':
            currentMode = msg.mode;
            syncModeUI();
            showToast('Mode: ' + msg.mode);
            break;

        case 'interpreter_langs_changed':
            document.getElementById('interpreter-source').value = msg.source;
            document.getElementById('interpreter-target').value = msg.target;
            break;

        case 'prompts':
            document.getElementById('prompt-conversation').value = msg.conversation || '';
            document.getElementById('prompt-monologue').value = msg.monologue || '';
            document.getElementById('prompt-interpreter').value = msg.interpreter || '';
            break;

        case 'prompt_saved':
            showToast('Prompt saved: ' + msg.mode);
            break;

        case 'llm_settings':
            syncLlmUI(msg.backend, msg.model);
            break;

        case 'volume':
            setVolumeUI(msg.volume);
            isMuted = msg.volume === 0;
            if (!isMuted) volumeBeforeMute = msg.volume;
            updateMuteUI();
            break;

        case 'history':
            setHistoryUI(msg.turns || 0);
            break;

        case 'restart_status':
            handleRestartStatus(msg);
            break;

        case 'capture_info':
            document.getElementById('capture-data-count').textContent = (msg.count || 0) + ' photos';
            document.getElementById('capture-storage-path').textContent = msg.path || '--';
            break;

        case 'voice_settings':
            setVoiceUI(msg.speaker_id, msg.pitch_shift, msg.speed);
            break;

        case 'motor_state':
            motorEnabled = msg.enabled !== false;
            motorPreset = msg.preset || 'moderate';
            document.getElementById('motor-toggle').classList.toggle('active', motorEnabled);
            document.getElementById('motor-presets').classList.toggle('disabled', !motorEnabled);
            syncMotorPresetUI();
            updateMotorStatus();
            break;

        case 'vlm_state': {
            const vlmToggle = document.getElementById('vlm-toggle');
            if (vlmToggle) vlmToggle.classList.toggle('active', !!msg.enabled);
            break;
        }

        case 'bargein_state': {
            const bargeinToggle = document.getElementById('bargein-toggle');
            if (bargeinToggle) bargeinToggle.classList.toggle('active', !!msg.enabled);
            break;
        }
    }
}

// ── ASR active / idle ───────────────────────────────────────────────
function triggerAsrActive() {
    asrActive = true;
    if (asrActiveTimer) clearTimeout(asrActiveTimer);
    asrActiveTimer = setTimeout(() => { asrActive = false; }, 500);
}

function resetAsrIdleTimer() {
    if (asrIdleTimer) clearTimeout(asrIdleTimer);
    asrIdleTimer = setTimeout(() => {
        asrTextEl.innerHTML = '<i>Listening...</i>';
        asrTextEl.className = 'asr-text idle';
        asrActive = false;
    }, 5000);
}

// ── Emotion display (center column) ─────────────────────────────────
function updateEmotionDisplay() {
    // Use vision_faces from dashboard WS, or fall back to direct vision WS data
    const faces = (latestVisionFaces && latestVisionFaces.length > 0)
        ? latestVisionFaces
        : latestFaces;

    if (!faces || faces.length === 0) {
        emotionPill.textContent = 'Neutral';
        emotionLabel.textContent = 'Smiling face';
        return;
    }

    // Largest face (first in list from dashboard, or pick largest from vision WS)
    const face = faces[0];
    const rawEmotion = face.emotion || 'neutral';
    // Normalize: capitalize first letter for display
    const emotion = rawEmotion.charAt(0).toUpperCase() + rawEmotion.slice(1);
    const conf = Math.round((face.emotion_confidence || 0) * 100);
    const color = EMOTION_COLORS[rawEmotion] || EMOTION_COLORS.neutral;

    emotionPill.textContent = conf > 0 ? `${emotion} ${conf}%` : emotion;
    emotionPill.style.borderColor = color;
    emotionPill.style.color = color;

    // Label is fixed — it's the title for the smile capture counter
    // (emotionLabel always shows "Smiling face")
}

// ── Thought bubbles (right column) ──────────────────────────────────
function showThinkingCard() {
    // Don't add if already streaming or thinking
    if (thoughtList.querySelector('.thought-card.streaming')) return;
    if (thoughtList.querySelector('.thought-card.thinking')) return;
    const card = document.createElement('div');
    card.className = 'thought-card thinking';
    card.innerHTML = '<div class="thinking-dots"><span>.</span><span>.</span><span>.</span></div>';
    thoughtList.prepend(card);
}

function removeThinkingCard() {
    const card = thoughtList.querySelector('.thought-card.thinking');
    if (card) card.remove();
}

function addStreamingCard() {
    removeThinkingCard();
    // Remove existing streaming card if any
    const existing = thoughtList.querySelector('.thought-card.streaming');
    if (existing) existing.remove();

    const card = document.createElement('div');
    card.className = 'thought-card streaming';
    card.innerHTML = '<div class="thought-text"></div>';
    thoughtList.prepend(card);

    trimThoughts();
}

function updateStreamingCard() {
    const card = thoughtList.querySelector('.thought-card.streaming');
    if (!card) return;
    const textEl = card.querySelector('.thought-text');
    textEl.textContent = currentLlmText;
    // Append typing cursor element (safe — no user content in innerHTML)
    let cursor = textEl.querySelector('.typing-cursor');
    if (!cursor) {
        cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        textEl.appendChild(cursor);
    }
}

function finalizeThoughtCard(emotion) {
    const card = thoughtList.querySelector('.thought-card.streaming');
    const text = currentLlmText.trim();

    // Empty response (e.g. only emotion tags) — remove card
    if (!text) {
        if (card) card.remove();
        return;
    }

    if (!card) {
        // No streaming card, create completed one
        const newCard = document.createElement('div');
        newCard.className = 'thought-card';
        const emoji = emotion ? (EMOTION_EMOJI[emotion] || '') : '';
        newCard.innerHTML = `<div class="thought-text">${escapeHtml(text)}</div>` +
            (emoji ? `<span class="thought-emoji">${emoji}</span>` : '');
        thoughtList.prepend(newCard);
    } else {
        card.classList.remove('streaming');
        const textEl = card.querySelector('.thought-text');
        textEl.textContent = text;
        const emoji = emotion ? (EMOTION_EMOJI[emotion] || '') : '';
        if (emoji) {
            const emojiEl = document.createElement('span');
            emojiEl.className = 'thought-emoji';
            emojiEl.textContent = emoji;
            card.appendChild(emojiEl);
        }
    }

    trimThoughts();
}

function trimThoughts() {
    const cards = thoughtList.querySelectorAll('.thought-card');
    if (cards.length > MAX_THOUGHTS) {
        for (let i = MAX_THOUGHTS; i < cards.length; i++) {
            cards[i].classList.add('fading');
            setTimeout(() => cards[i].remove(), 300);
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Capture counter ─────────────────────────────────────────────────
function spawnFloatPlus() {
    const area = document.getElementById('capture-float-area');
    const el = document.createElement('div');
    el.className = 'capture-float';
    el.textContent = '+1';
    area.appendChild(el);
    setTimeout(() => el.remove(), 1300);
}

const captureResetBtn = document.getElementById('capture-reset');
if (captureResetBtn) {
    captureResetBtn.onclick = () => {
        if (dashboardWs && dashboardWs.readyState === 1) {
            dashboardWs.send(JSON.stringify({ type: 'clear_captures' }));
        }
    };
}

// ── State ───────────────────────────────────────────────────────────
function updateState(state) {
    const el = document.getElementById('robot-state');
    if (el) {
        el.textContent = state;
        el.dataset.state = state;
    }
    if (state === 'thinking') showThinkingCard();
    // When backend returns to idle/listening, reset ASR display immediately
    if (state === 'idle' || state === 'listening') {
        removeThinkingCard();
        if (asrIdleTimer) clearTimeout(asrIdleTimer);
        asrTextEl.innerHTML = '<i>Listening...</i>';
        asrTextEl.className = 'asr-text idle';
        asrActive = false;
    }
}

function updateRobotState(msg) {
    if (msg.mode) {
        currentMode = msg.mode;
        syncModeUI();
    }
    if (msg.capture_count !== undefined) {
        captureCount = msg.capture_count;
        captureCountEl.textContent = captureCount;
    }
    // Sync VLM and barge-in toggles
    const vlmToggle = document.getElementById('vlm-toggle');
    if (vlmToggle && msg.vlm_enabled !== undefined) {
        vlmToggle.classList.toggle('active', !!msg.vlm_enabled);
    }
    const bargeinToggle = document.getElementById('bargein-toggle');
    if (bargeinToggle && msg.barge_in_enabled !== undefined) {
        bargeinToggle.classList.toggle('active', !!msg.barge_in_enabled);
    }
    // Sync audio detection sliders
    if (msg.silero_threshold !== undefined) {
        const el = document.getElementById('vad-threshold');
        if (el) { el.value = msg.silero_threshold; }
        const lbl = document.getElementById('vad-threshold-value');
        if (lbl) { lbl.textContent = msg.silero_threshold.toFixed(2); }
    }
    if (msg.barge_in_energy_threshold !== undefined) {
        const el = document.getElementById('energy-threshold');
        if (el) { el.value = msg.barge_in_energy_threshold; }
        const lbl = document.getElementById('energy-threshold-value');
        if (lbl) { lbl.textContent = msg.barge_in_energy_threshold.toFixed(3); }
    }
    if (msg.llm_backend !== undefined) {
        syncLlmUI(msg.llm_backend, msg.ollama_model);
    }
    document.getElementById('dot-robot').className = 'dot live';
}

// ── Face crop pipeline ──────────────────────────────────────────────
function drawFaceCrop() {
    const vp = document.querySelector('.face-viewport');

    // Expire stale face data after 2s without updates
    if (latestFaces && latestFacesTs && Date.now() - latestFacesTs > 2000) {
        latestFaces = null;
    }

    if (!latestFaces || latestFaces.length === 0 || videoEl.style.display === 'none') {
        faceCropCtx.clearRect(0, 0, faceCropCanvas.width, faceCropCanvas.height);
        vp.classList.remove('has-face');
        requestAnimationFrame(drawFaceCrop);
        return;
    }

    // Find largest face
    let largest = latestFaces[0];
    let maxArea = 0;
    for (const f of latestFaces) {
        const area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
        if (area > maxArea) { maxArea = area; largest = f; }
    }

    vp.classList.add('has-face');

    // Map normalized bbox to video natural size
    const natW = videoEl.naturalWidth || 640;
    const natH = videoEl.naturalHeight || 360;
    const [x1, y1, x2, y2] = largest.bbox;
    let sx = x1 * natW;
    let sy = y1 * natH;
    let sw = (x2 - x1) * natW;
    let sh = (y2 - y1) * natH;

    // Add padding (30%)
    const padW = sw * 0.3;
    const padH = sh * 0.3;
    sx = Math.max(0, sx - padW);
    sy = Math.max(0, sy - padH);
    sw = Math.min(natW - sx, sw + padW * 2);
    sh = Math.min(natH - sy, sh + padH * 2);

    // Draw cropped face — clear first so failures never show stale content
    faceCropCtx.clearRect(0, 0, faceCropCanvas.width, faceCropCanvas.height);
    try {
        if (videoEl.complete && videoEl.naturalWidth > 0) {
            faceCropCtx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, faceCropCanvas.width, faceCropCanvas.height);
        }
    } catch(e) {
        // drawImage failed — canvas already cleared
    }

    requestAnimationFrame(drawFaceCrop);
}

// ── Canvas overlay (face detection boxes on video) ──────────────────
function getImageRect() {
    const container = document.querySelector('.video-container');
    const rect = container.getBoundingClientRect();
    const natW = videoEl.naturalWidth || 640;
    const natH = videoEl.naturalHeight || 360;
    // CSS uses object-fit: cover — match with Math.max (not contain/min)
    const scale = Math.max(rect.width / natW, rect.height / natH);
    const imgW = natW * scale;
    const imgH = natH * scale;
    const offsetX = (rect.width - imgW) / 2;
    const offsetY = (rect.height - imgH) / 2;
    return { offsetX, offsetY, imgW, imgH, fullW: rect.width, fullH: rect.height };
}

function drawOverlay() {
    const container = document.querySelector('.video-container');
    const rect = container.getBoundingClientRect();
    canvasEl.width = rect.width;
    canvasEl.height = rect.height;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!latestFaces || latestFaces.length === 0) {
        requestAnimationFrame(drawOverlay);
        return;
    }

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

        const cornerLen = Math.min(bw, bh) * 0.25;
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.globalAlpha = 0.9;

        // Corner brackets
        ctx.beginPath();
        ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
        ctx.stroke();

        ctx.globalAlpha = 1.0;

        // Identity label
        const identity = face.identity;
        if (identity && identity !== '?') {
            ctx.font = 'bold 14px sans-serif';
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

        // Emotion pill below bbox
        const emotion = face.emotion || 'neutral';
        const conf = ((face.emotion_confidence || 0) * 100).toFixed(0);
        const pillText = conf > 0 ? `${emotion} ${conf}%` : emotion;
        ctx.font = 'bold 14px sans-serif';
        const pillMetrics = ctx.measureText(pillText);
        const pillW = pillMetrics.width + 20;
        const pillH = 24;
        const pillX = bx + (bw - pillW) / 2;
        let pillY = by + bh + 6;
        if (pillY + pillH > oy + ch - 4) pillY = by + bh - pillH - 4;

        ctx.globalAlpha = 0.7;
        ctx.fillStyle = hexToRgba(color, 0.25);
        roundRect(ctx, pillX, pillY, pillW, pillH, 10);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(color, 0.5);
        ctx.lineWidth = 1;
        roundRect(ctx, pillX, pillY, pillW, pillH, 10);
        ctx.stroke();

        ctx.globalAlpha = 1.0;
        ctx.fillStyle = color;
        ctx.fillText(pillText, pillX + 10, pillY + 17);

        // Landmarks
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

// ── MJPEG video stream ──────────────────────────────────────────────
function setupVideo() {
    // Use same-origin proxy to avoid CORS/cross-port issues
    const streamUrl = `/stream`;
    const STALL_TIMEOUT = 5000; // ms — restart if no data for this long

    let abortCtrl = null;

    async function readStream() {
        // Abort any previous stream
        if (abortCtrl) { try { abortCtrl.abort(); } catch(_) {} }
        abortCtrl = new AbortController();

        try {
            const res = await fetch(streamUrl, { signal: abortCtrl.signal });
            const reader = res.body.getReader();
            let buf = new Uint8Array(0);
            let stallTimer = setTimeout(onStall, STALL_TIMEOUT);

            function onStall() {
                console.warn('MJPEG stream stalled, reconnecting...');
                reader.cancel();
            }

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Reset stall timer on every chunk
                clearTimeout(stallTimer);
                stallTimer = setTimeout(onStall, STALL_TIMEOUT);

                const tmp = new Uint8Array(buf.length + value.length);
                tmp.set(buf); tmp.set(value, buf.length);
                buf = tmp;

                // Extract ALL complete JPEG frames in buffer, keep only the last one
                let lastJpeg = null;
                let lastEnd = -1;
                let start = -1;
                for (let i = 0; i < buf.length - 1; i++) {
                    if (buf[i] === 0xFF && buf[i+1] === 0xD8) start = i;
                    if (buf[i] === 0xFF && buf[i+1] === 0xD9 && start >= 0) {
                        lastJpeg = buf.slice(start, i + 2);
                        lastEnd = i + 2;
                        start = -1;
                    }
                }

                if (lastJpeg) {
                    // Display only the most recent frame (skip stale ones)
                    const blob = new Blob([lastJpeg], { type: 'image/jpeg' });
                    const url = URL.createObjectURL(blob);
                    const prev = videoEl.src;
                    videoEl.src = url;
                    videoEl.style.display = 'block';
                    noVideoEl.style.display = 'none';
                    if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
                    // Keep only unprocessed bytes after last complete frame
                    buf = buf.slice(lastEnd);
                }

                // Safety: cap buffer at 512KB to prevent memory leak
                if (buf.length > 512 * 1024) {
                    buf = new Uint8Array(0);
                }
            }

            clearTimeout(stallTimer);
        } catch (e) {
            if (e.name !== 'AbortError') {
                console.warn('MJPEG stream error:', e);
            }
        }

        videoEl.style.display = 'none';
        noVideoEl.style.display = 'flex';
        setTimeout(readStream, 2000);
    }

    readStream();
}

// ── Settings Panel ──────────────────────────────────────────────────
function initSettings() {
    const overlay = document.getElementById('settings-overlay');
    document.getElementById('settings-open').onclick = () => {
        overlay.classList.add('open');
        loadFaces();
        if (dashboardWs && dashboardWs.readyState === 1) {
            dashboardWs.send(JSON.stringify({ type: 'get_volume' }));
            dashboardWs.send(JSON.stringify({ type: 'get_motor' }));
            dashboardWs.send(JSON.stringify({ type: 'get_voice' }));
        }
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
            if (tab.dataset.tab === 'faces') loadCaptureInfo();
            if (tab.dataset.tab === 'detail') {
                if (dashboardWs && dashboardWs.readyState === 1) {
                    dashboardWs.send(JSON.stringify({ type: 'get_voice' }));
                    dashboardWs.send(JSON.stringify({ type: 'get_motor' }));
                }
            }
            if (tab.dataset.tab === 'prompt') loadPrompts();
        };
    });

    // Mode selection (only options with data-mode, not motor presets)
    document.querySelectorAll('.mode-option[data-mode]').forEach(opt => {
        opt.onclick = () => {
            const mode = opt.dataset.mode;
            if (mode === currentMode) return;
            setMode(mode);
        };
    });

    // Interpreter language selectors
    const interpSrc = document.getElementById('interpreter-source');
    const interpTgt = document.getElementById('interpreter-target');
    if (interpSrc && interpTgt) {
        const sendLangs = () => {
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({
                    type: 'set_interpreter_langs',
                    source: interpSrc.value,
                    target: interpTgt.value,
                }));
            }
        };
        interpSrc.onchange = sendLangs;
        interpTgt.onchange = sendLangs;
    }

    // LLM backend/model selection
    const llmBackend = document.getElementById('llm-backend');
    const ollamaModel = document.getElementById('ollama-model');
    if (llmBackend) {
        llmBackend.onchange = () => {
            const backend = llmBackend.value;
            const model = backend === 'ollama' ? ollamaModel.value : undefined;
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({ type: 'set_llm', backend, model }));
            }
            syncLlmUI(backend, model);
        };
    }
    if (ollamaModel) {
        ollamaModel.onchange = () => {
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({ type: 'set_llm', backend: 'ollama', model: ollamaModel.value }));
            }
        };
    }

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
    document.getElementById('prompt-interp-save').onclick = () => savePrompt('interpreter');
    document.getElementById('prompt-interp-reset').onclick = () => {
        document.getElementById('prompt-interpreter').value = '';
        savePrompt('interpreter', '');
    };
    document.getElementById('prompt-mono-reset').onclick = () => {
        document.getElementById('prompt-monologue').value = '';
        savePrompt('monologue', '');
    };

    // Voice settings (speaker, pitch, speed)
    initVoice();
    initAudioDetection();

    // Volume control
    initVolume();

    // Memory (history turns)
    initHistory();

    // Motor control
    initMotor();

    // VLM toggle
    const vlmToggle = document.getElementById('vlm-toggle');
    if (vlmToggle) {
        vlmToggle.onclick = () => {
            const enabled = !vlmToggle.classList.contains('active');
            vlmToggle.classList.toggle('active', enabled);
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({ type: 'set_vlm', enabled }));
            }
        };
    }

    // Barge-in toggle
    const bargeinToggle = document.getElementById('bargein-toggle');
    if (bargeinToggle) {
        bargeinToggle.onclick = () => {
            const enabled = !bargeinToggle.classList.contains('active');
            bargeinToggle.classList.toggle('active', enabled);
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({ type: 'set_bargein', enabled }));
            }
        };
    }

    // Restart services
    initRestart();
}

let currentVolume = 80;
let isMuted = false;
let volumeBeforeMute = 80;

function initVolume() {
    const slider = document.getElementById('volume-slider');
    const valueEl = document.getElementById('volume-value');
    const muteBtn = document.getElementById('volume-mute');

    slider.oninput = () => {
        const vol = parseInt(slider.value);
        valueEl.textContent = vol + '%';
        currentVolume = vol;
        if (vol > 0) { isMuted = false; volumeBeforeMute = vol; }
        else { isMuted = true; }
        updateMuteUI();
    };
    slider.onchange = () => {
        sendVolume(parseInt(slider.value));
    };
    muteBtn.onclick = () => {
        isMuted = !isMuted;
        if (isMuted) {
            volumeBeforeMute = currentVolume || 80;
            setVolumeUI(0);
            sendVolume(0);
        } else {
            setVolumeUI(volumeBeforeMute);
            sendVolume(volumeBeforeMute);
        }
        updateMuteUI();
    };

    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({ type: 'get_volume' }));
    }
}

function setVolumeUI(vol) {
    currentVolume = vol;
    document.getElementById('volume-slider').value = vol;
    document.getElementById('volume-value').textContent = vol + '%';
}

function updateMuteUI() {
    const btn = document.getElementById('volume-mute');
    btn.classList.toggle('muted', isMuted);
    document.getElementById('volume-icon-on').style.display = isMuted ? 'none' : '';
    document.getElementById('volume-icon-off').style.display = isMuted ? '' : 'none';
}

function sendVolume(vol) {
    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({ type: 'set_volume', volume: vol }));
    }
}

function initHistory() {
    const slider = document.getElementById('history-slider');
    const valueEl = document.getElementById('history-value');

    slider.oninput = () => {
        valueEl.textContent = slider.value + ' turns';
    };
    slider.onchange = () => {
        const turns = parseInt(slider.value);
        if (dashboardWs && dashboardWs.readyState === 1) {
            dashboardWs.send(JSON.stringify({ type: 'set_history', turns }));
        }
    };

    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({ type: 'get_history' }));
    }
}

function setHistoryUI(turns) {
    document.getElementById('history-slider').value = turns;
    document.getElementById('history-value').textContent = turns + ' turns';
}

function syncModeUI() {
    document.querySelectorAll('.mode-option[data-mode]').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.mode === currentMode);
        opt.querySelector('input').checked = opt.dataset.mode === currentMode;
    });
    document.getElementById('mode-status').textContent = 'Current: ' + currentMode;
    const toggles = document.getElementById('mode-toggles');
    if (toggles) toggles.style.display = currentMode === 'conversation' ? '' : 'none';
    const interpSettings = document.getElementById('interpreter-settings');
    if (interpSettings) interpSettings.style.display = currentMode === 'interpreter' ? '' : 'none';
    const mindLabel = document.getElementById('mind-label');
    if (mindLabel) {
        mindLabel.textContent = currentMode === 'conversation' ? 'Says'
            : currentMode === 'interpreter' ? 'Translation' : 'Mind';
    }
}

function syncLlmUI(backend, model) {
    const backendEl = document.getElementById('llm-backend');
    const modelEl = document.getElementById('ollama-model');
    const modelRow = document.getElementById('ollama-model-row');
    if (backendEl) backendEl.value = backend;
    if (modelRow) modelRow.style.display = backend === 'ollama' ? '' : 'none';
    if (modelEl && model) modelEl.value = model;
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
        const res = await fetch(`${VISION_API}/api/faces/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) { showToast('Delete failed', true); return; }
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
        if (!res.ok || data.error) { showToast(data.error || 'Enroll failed', true); return; }
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
            if (!res.ok || data.error) { fail++; } else { ok++; }
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

// ── Capture Data Management ──────────────────────────────────────────
function loadCaptureInfo() {
    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({ type: 'get_capture_info' }));
    }
}

document.getElementById('capture-clear-btn').onclick = async () => {
    if (!confirm('Clear all smile capture photos? This cannot be undone.')) return;
    try {
        const res = await fetch(`${VISION_API}/api/captures`, { method: 'DELETE' });
        if (res.ok) {
            showToast('Captures cleared');
            captureCount = 0;
            captureCountEl.textContent = '0';
            loadCaptureInfo();
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({ type: 'clear_captures' }));
            }
        } else {
            showToast('Clear failed', true);
        }
    } catch (e) {
        showToast('Clear failed', true);
    }
};

document.getElementById('capture-export-btn').onclick = async () => {
    try {
        const res = await fetch(`${VISION_API}/api/captures/export`);
        if (!res.ok) { showToast('Export failed', true); return; }
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'smile-captures.zip';
        a.click();
        URL.revokeObjectURL(a.href);
        showToast('Exported captures');
    } catch (e) {
        showToast('Export failed', true);
    }
};

// ── Voice Settings ──────────────────────────────────────────────────
let voiceSid = 3;
let voicePitch = 1.5;
let voiceSpeed = 0.8;

function initVoice() {
    const sidInput = document.getElementById('voice-sid');
    const pitchSlider = document.getElementById('voice-pitch');
    const pitchValue = document.getElementById('voice-pitch-value');
    const speedSlider = document.getElementById('voice-speed');
    const speedValue = document.getElementById('voice-speed-value');

    sidInput.onchange = () => {
        voiceSid = Math.max(0, parseInt(sidInput.value) || 0);
        sidInput.value = voiceSid;
        sendVoice();
    };

    pitchSlider.oninput = () => {
        voicePitch = parseFloat(pitchSlider.value);
        pitchValue.textContent = (voicePitch >= 0 ? '+' : '') + voicePitch.toFixed(1);
    };
    pitchSlider.onchange = () => sendVoice();

    speedSlider.oninput = () => {
        voiceSpeed = parseFloat(speedSlider.value);
        speedValue.textContent = voiceSpeed.toFixed(1) + 'x';
    };
    speedSlider.onchange = () => sendVoice();
}

function setVoiceUI(sid, pitch, speed) {
    voiceSid = sid;
    voicePitch = pitch;
    voiceSpeed = speed;
    document.getElementById('voice-sid').value = sid;
    document.getElementById('voice-pitch').value = pitch;
    document.getElementById('voice-pitch-value').textContent = (pitch >= 0 ? '+' : '') + pitch.toFixed(1);
    document.getElementById('voice-speed').value = speed;
    document.getElementById('voice-speed-value').textContent = speed.toFixed(1) + 'x';
}

function sendVoice() {
    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({
            type: 'set_voice',
            speaker_id: voiceSid,
            pitch_shift: voicePitch,
            speed: voiceSpeed,
        }));
    }
}

// ── Audio detection sliders ──────────────────────────────────────────
function initAudioDetection() {
    const vadSlider = document.getElementById('vad-threshold');
    const vadValue = document.getElementById('vad-threshold-value');
    const energySlider = document.getElementById('energy-threshold');
    const energyValue = document.getElementById('energy-threshold-value');

    if (vadSlider) {
        vadSlider.oninput = () => {
            vadValue.textContent = parseFloat(vadSlider.value).toFixed(2);
        };
        vadSlider.onchange = () => {
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({
                    type: 'set_vad_threshold',
                    value: parseFloat(vadSlider.value),
                }));
            }
        };
    }
    if (energySlider) {
        energySlider.oninput = () => {
            energyValue.textContent = parseFloat(energySlider.value).toFixed(3);
        };
        energySlider.onchange = () => {
            if (dashboardWs && dashboardWs.readyState === 1) {
                dashboardWs.send(JSON.stringify({
                    type: 'set_energy_threshold',
                    value: parseFloat(energySlider.value),
                }));
            }
        };
    }
}

// ── Motor Control ───────────────────────────────────────────────────
let motorEnabled = true;
let motorPreset = 'moderate';

function initMotor() {
    const toggle = document.getElementById('motor-toggle');
    const presets = document.getElementById('motor-presets');

    toggle.onclick = () => {
        motorEnabled = !motorEnabled;
        toggle.classList.toggle('active', motorEnabled);
        presets.classList.toggle('disabled', !motorEnabled);
        updateMotorStatus();
        sendMotorState();
    };

    document.querySelectorAll('#motor-presets .mode-option').forEach(opt => {
        opt.onclick = (e) => {
            e.preventDefault(); // prevent label→radio double-fire
            const preset = opt.dataset.motor;
            if (preset === motorPreset) return;
            motorPreset = preset;
            syncMotorPresetUI();
            updateMotorStatus();
            sendMotorState();
        };
    });

    // Apply initial state
    syncMotorPresetUI();
    updateMotorStatus();
}

function syncMotorPresetUI() {
    document.querySelectorAll('#motor-presets .mode-option').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.motor === motorPreset);
        opt.querySelector('input').checked = opt.dataset.motor === motorPreset;
    });
}

function updateMotorStatus() {
    const el = document.getElementById('motor-status');
    if (el) {
        el.textContent = motorEnabled ? 'Motor: ' + motorPreset : 'Motor: sleep (disabled)';
    }
}

function sendMotorState() {
    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({
            type: 'set_motor',
            enabled: motorEnabled,
            preset: motorPreset,
        }));
    }
}

// ── Prompt Management ───────────────────────────────────────────────
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
        const idMap = { conversation: 'prompt-conversation', monologue: 'prompt-monologue', interpreter: 'prompt-interpreter' };
        text = document.getElementById(idMap[mode]).value;
    }
    dashboardWs.send(JSON.stringify({ type: 'set_prompt', mode, prompt: text }));
}

// ── Restart Services ─────────────────────────────────────────────
function initRestart() {
    const btn = document.getElementById('restart-btn');
    if (!btn) return;
    btn.onclick = () => {
        if (!dashboardWs || dashboardWs.readyState !== 1) {
            showToast('Not connected', true);
            return;
        }
        if (!confirm('Restart all services? The dashboard will briefly disconnect.')) return;
        btn.disabled = true;
        document.getElementById('restart-status').textContent = 'Sending restart command...';
        dashboardWs.send(JSON.stringify({ type: 'restart_services' }));
    };
}

function handleRestartStatus(msg) {
    const statusEl = document.getElementById('restart-status');
    const btn = document.getElementById('restart-btn');
    if (!statusEl) return;

    switch (msg.status) {
        case 'starting':
            statusEl.textContent = 'Restarting services...';
            break;
        case 'restarting':
            statusEl.textContent = `Restarting ${msg.container}...`;
            break;
        case 'done':
            statusEl.textContent = 'All services restarted. Reconnecting...';
            if (btn) btn.disabled = false;
            break;
        case 'error':
            statusEl.textContent = 'Error: ' + (msg.error || 'unknown');
            if (btn) btn.disabled = false;
            showToast('Restart failed', true);
            break;
    }
}

// ── Init ────────────────────────────────────────────────────────────
function init() {
    setupVideo();
    connectVision();
    connectDashboard();
    initSettings();
    requestAnimationFrame(drawOverlay);
    requestAnimationFrame(drawFaceCrop);
}

init();
