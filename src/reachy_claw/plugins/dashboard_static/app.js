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
const MAX_THOUGHTS = 3;

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
            triggerAsrActive();
            resetAsrIdleTimer();
            break;

        case 'asr_final':
            asrTextEl.innerHTML = msg.text;
            asrTextEl.className = 'asr-text';
            triggerAsrActive();
            resetAsrIdleTimer();
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

        case 'prompts':
            document.getElementById('prompt-conversation').value = msg.conversation || '';
            document.getElementById('prompt-monologue').value = msg.monologue || '';
            break;

        case 'prompt_saved':
            showToast('Prompt saved: ' + msg.mode);
            break;

        case 'volume':
            setVolumeUI(msg.volume);
            isMuted = msg.volume === 0;
            if (!isMuted) volumeBeforeMute = msg.volume;
            updateMuteUI();
            break;

        case 'restart_status':
            handleRestartStatus(msg);
            break;

        case 'motor_state':
            motorEnabled = msg.enabled !== false;
            motorPreset = msg.preset || 'moderate';
            document.getElementById('motor-toggle').classList.toggle('active', motorEnabled);
            document.getElementById('motor-presets').classList.toggle('disabled', !motorEnabled);
            document.querySelector('.motor-sleep-label').textContent =
                motorEnabled ? 'Motor Enabled' : 'Motor Disabled (Sleep)';
            syncMotorPresetUI();
            updateMotorStatus();
            break;
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

    // Face description label
    const FACE_LABELS = {
        happy: 'Smiling face', happiness: 'Smiling face',
        sad: 'Sad face', sadness: 'Sad face',
        angry: 'Angry face', anger: 'Angry face',
        surprised: 'Surprised face', surprise: 'Surprised face',
        fear: 'Fearful face',
        neutral: 'Neutral face',
        contempt: 'Contemptuous face',
        disgust: 'Disgusted face',
    };
    const identity = face.identity;
    if (identity && identity !== '?') {
        emotionLabel.textContent = identity;
    } else {
        emotionLabel.textContent = FACE_LABELS[rawEmotion.toLowerCase()] || (emotion + ' face');
    }
}

// ── Thought bubbles (right column) ──────────────────────────────────
function addStreamingCard() {
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
    textEl.innerHTML = currentLlmText + '<span class="typing-cursor"></span>';
}

function finalizeThoughtCard(emotion) {
    const card = thoughtList.querySelector('.thought-card.streaming');
    if (!card) {
        // No streaming card, create completed one
        const newCard = document.createElement('div');
        newCard.className = 'thought-card';
        const emoji = emotion ? (EMOTION_EMOJI[emotion] || '') : '';
        newCard.innerHTML = `<div class="thought-text">${escapeHtml(currentLlmText)}</div>` +
            (emoji ? `<span class="thought-emoji">${emoji}</span>` : '');
        thoughtList.prepend(newCard);
    } else {
        card.classList.remove('streaming');
        const textEl = card.querySelector('.thought-text');
        textEl.textContent = currentLlmText;
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
    document.getElementById('dot-robot').className = 'dot live';
}

// ── Face crop pipeline ──────────────────────────────────────────────
function drawFaceCrop() {
    const vp = document.querySelector('.face-viewport');

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

    // Draw cropped face
    faceCropCtx.clearRect(0, 0, faceCropCanvas.width, faceCropCanvas.height);
    try {
        faceCropCtx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, faceCropCanvas.width, faceCropCanvas.height);
    } catch(e) {
        // Image not loaded yet
    }

    requestAnimationFrame(drawFaceCrop);
}

// ── Canvas overlay (face detection boxes on video) ──────────────────
function getImageRect() {
    const container = document.querySelector('.video-container');
    const rect = container.getBoundingClientRect();
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

    async function readStream() {
        try {
            const res = await fetch(streamUrl);
            const reader = res.body.getReader();
            let buf = new Uint8Array(0);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const tmp = new Uint8Array(buf.length + value.length);
                tmp.set(buf); tmp.set(value, buf.length);
                buf = tmp;

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
                        break;
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
        if (dashboardWs && dashboardWs.readyState === 1) {
            dashboardWs.send(JSON.stringify({ type: 'get_volume' }));
            dashboardWs.send(JSON.stringify({ type: 'get_motor' }));
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

    // Volume control
    initVolume();

    // Motor control
    initMotor();

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
        document.querySelector('.motor-sleep-label').textContent =
            motorEnabled ? 'Motor Enabled' : 'Motor Disabled (Sleep)';
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
    if (!motorEnabled) {
        el.textContent = 'Motor: sleep (disabled)';
    } else {
        el.textContent = 'Motor: ' + motorPreset;
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
        const id = mode === 'conversation' ? 'prompt-conversation' : 'prompt-monologue';
        text = document.getElementById(id).value;
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
