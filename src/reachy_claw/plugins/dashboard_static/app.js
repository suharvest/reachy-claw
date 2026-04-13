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
        // Query TTS capabilities and cloned voices
        dashboardWs.send(JSON.stringify({ type: 'get_tts_capabilities' }));
        dashboardWs.send(JSON.stringify({ type: 'get_cloned_voices' }));
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
            syncLlmUI(msg.backend, msg.model, msg.ollama_url, msg.gateway_host, msg.gateway_port);
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
            voiceSid = msg.speaker_id;
            voicePitch = msg.pitch_shift;
            voiceSpeed = msg.speed;
            voiceCloneSupported = msg.voice_clone_supported || false;
            selectedVoiceName = msg.cloned_voice_name || null;
            updateVoiceUI();
            setVoiceInputs();
            break;

        case 'tts_capabilities':
            voiceCloneSupported = msg.voice_clone || false;
            updateVoiceUI();
            break;

        case 'cloned_voices':
            clonedVoices = msg.voices || [];
            updateVoiceSelect();
            break;

        case 'clone_voice_result':
            if (msg.success) {
                clonedVoices.push(msg.voice);
                updateVoiceSelect();
                document.getElementById('voice-select').value = msg.voice.name;
                selectedVoiceName = msg.voice.name;
                showToast('Voice cloned: ' + msg.voice.name);
            } else {
                showToast('Clone failed: ' + msg.error, true);
            }
            closeCloneModal();
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

        // Diary narration messages
        case 'diary_narrate_focus':
        case 'diary_narrate_navigate':
        case 'diary_narrate_end':
            if (typeof handleNarrationMessage === 'function') {
                handleNarrationMessage(msg);
            }
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
    const ollamaUrl = document.getElementById('ollama-url');
    const gatewayHost = document.getElementById('gateway-host');
    const gatewayPort = document.getElementById('gateway-port');
    const applyLlmBtn = document.getElementById('apply-llm-btn');

    // Show/hide fields when backend changes
    function updateLlmFieldsVisibility() {
        const backend = llmBackend ? llmBackend.value : 'ollama';
        const isOllama = backend === 'ollama';
        const modelRow = document.getElementById('ollama-model-row');
        const ollamaUrlRow = document.getElementById('ollama-url-row');
        const gatewayHostRow = document.getElementById('gateway-host-row');
        const gatewayPortRow = document.getElementById('gateway-port-row');
        if (modelRow) modelRow.style.display = isOllama ? '' : 'none';
        if (ollamaUrlRow) ollamaUrlRow.style.display = isOllama ? '' : 'none';
        if (gatewayHostRow) gatewayHostRow.style.display = isOllama ? 'none' : '';
        if (gatewayPortRow) gatewayPortRow.style.display = isOllama ? 'none' : '';
    }

    if (llmBackend) {
        llmBackend.onchange = updateLlmFieldsVisibility;
    }

    // Apply button: send all LLM settings
    if (applyLlmBtn) {
        applyLlmBtn.onclick = () => {
            if (!dashboardWs || dashboardWs.readyState !== 1) {
                showToast('Not connected', true);
                return;
            }
            const backend = llmBackend ? llmBackend.value : 'ollama';
            const model = backend === 'ollama' && ollamaModel ? ollamaModel.value : undefined;
            const ollama_url = backend === 'ollama' && ollamaUrl ? ollamaUrl.value : undefined;
            const gateway_host = backend === 'gateway' && gatewayHost ? gatewayHost.value : undefined;
            const gateway_port = backend === 'gateway' && gatewayPort ? parseInt(gatewayPort.value) : undefined;

            dashboardWs.send(JSON.stringify({
                type: 'set_llm',
                backend,
                model,
                ollama_url,
                gateway_host,
                gateway_port
            }));
            showToast('Applying LLM settings...');
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

function syncLlmUI(backend, model, ollamaUrl, gatewayHost, gatewayPort) {
    const backendEl = document.getElementById('llm-backend');
    const modelEl = document.getElementById('ollama-model');
    const modelRow = document.getElementById('ollama-model-row');
    const ollamaUrlRow = document.getElementById('ollama-url-row');
    const ollamaUrlEl = document.getElementById('ollama-url');
    const gatewayHostRow = document.getElementById('gateway-host-row');
    const gatewayHostEl = document.getElementById('gateway-host');
    const gatewayPortRow = document.getElementById('gateway-port-row');
    const gatewayPortEl = document.getElementById('gateway-port');

    if (backendEl) backendEl.value = backend;

    // Show/hide fields based on backend
    const isOllama = backend === 'ollama';
    if (modelRow) modelRow.style.display = isOllama ? '' : 'none';
    if (ollamaUrlRow) ollamaUrlRow.style.display = isOllama ? '' : 'none';
    if (gatewayHostRow) gatewayHostRow.style.display = isOllama ? 'none' : '';
    if (gatewayPortRow) gatewayPortRow.style.display = isOllama ? 'none' : '';

    if (modelEl && model) modelEl.value = model;
    if (ollamaUrlEl && ollamaUrl) ollamaUrlEl.value = ollamaUrl;
    if (gatewayHostEl && gatewayHost) gatewayHostEl.value = gatewayHost;
    if (gatewayPortEl && gatewayPort) gatewayPortEl.value = gatewayPort;
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
let voiceCloneSupported = false;
let clonedVoices = [];
let selectedVoiceName = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = null;
let cloneAudioBlob = null;

function initVoice() {
    const sidInput = document.getElementById('voice-sid');
    const voiceSelect = document.getElementById('voice-select');
    const cloneBtn = document.getElementById('clone-voice-btn');
    const pitchSlider = document.getElementById('voice-pitch');
    const pitchValue = document.getElementById('voice-pitch-value');
    const speedSlider = document.getElementById('voice-speed');
    const speedValue = document.getElementById('voice-speed-value');

    // Speaker ID input (fallback mode)
    if (sidInput) {
        sidInput.onchange = () => {
            voiceSid = Math.max(0, parseInt(sidInput.value) || 0);
            sidInput.value = voiceSid;
            sendVoice();
        };
    }

    // Voice select (clone mode)
    if (voiceSelect) {
        voiceSelect.onchange = () => {
            selectedVoiceName = voiceSelect.value || null;
            sendVoice();
        };
    }

    // Clone button opens modal
    if (cloneBtn) {
        cloneBtn.onclick = () => openCloneModal();
    }

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

    // Query TTS capabilities on init
    if (dashboardWs && dashboardWs.readyState === 1) {
        dashboardWs.send(JSON.stringify({ type: 'get_tts_capabilities' }));
        dashboardWs.send(JSON.stringify({ type: 'get_cloned_voices' }));
    }
}

function updateVoiceUI() {
    const cloneRow = document.getElementById('voice-clone-row');
    const speakerRow = document.getElementById('speaker-id-row');
    if (voiceCloneSupported) {
        if (cloneRow) cloneRow.style.display = 'flex';
        if (speakerRow) speakerRow.style.display = 'none';
    } else {
        if (cloneRow) cloneRow.style.display = 'none';
        if (speakerRow) speakerRow.style.display = 'flex';
    }
}

function updateVoiceSelect() {
    const select = document.getElementById('voice-select');
    if (!select) return;
    select.innerHTML = '<option value="">-- Select Voice --</option>';
    for (const v of clonedVoices) {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = v.name;
        select.appendChild(opt);
    }
    if (selectedVoiceName) {
        select.value = selectedVoiceName;
    }
}

function setVoiceInputs() {
    document.getElementById('voice-sid').value = voiceSid;
    document.getElementById('voice-pitch').value = voicePitch;
    document.getElementById('voice-pitch-value').textContent = (voicePitch >= 0 ? '+' : '') + voicePitch.toFixed(1);
    document.getElementById('voice-speed').value = voiceSpeed;
    document.getElementById('voice-speed-value').textContent = voiceSpeed.toFixed(1) + 'x';
    if (selectedVoiceName) {
        const select = document.getElementById('voice-select');
        if (select) select.value = selectedVoiceName;
    }
}

function sendVoice() {
    if (!dashboardWs || dashboardWs.readyState !== 1) return;
    const msg = {
        type: 'set_voice',
        pitch_shift: voicePitch,
        speed: voiceSpeed,
    };
    if (voiceCloneSupported && selectedVoiceName) {
        msg.voice_name = selectedVoiceName;
    } else {
        msg.speaker_id = voiceSid;
    }
    dashboardWs.send(JSON.stringify(msg));
}

// ── Voice Clone Modal ────────────────────────────────────────────────
function openCloneModal() {
    document.getElementById('clone-overlay').style.display = 'flex';
    document.getElementById('clone-name').value = '';
    document.getElementById('clone-submit-btn').disabled = true;
    document.getElementById('record-status').textContent = '';
    document.getElementById('record-timer').textContent = '0:00';
    resetRecordState();
}

function closeCloneModal() {
    document.getElementById('clone-overlay').style.display = 'none';
    resetRecordState();
}

function resetRecordState() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    mediaRecorder = null;
    audioChunks = [];
    recordingStartTime = null;
    cloneAudioBlob = null;
    const btn = document.getElementById('record-start-btn');
    if (btn) {
        btn.classList.remove('recording');
        btn.querySelector('span').textContent = 'Start Recording';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const closeBtn = document.getElementById('clone-close');
    if (closeBtn) closeBtn.onclick = closeCloneModal;

    // Method toggle
    const recordBtn = document.getElementById('clone-record-btn');
    const uploadBtn = document.getElementById('clone-upload-btn');
    const recordArea = document.getElementById('clone-record-area');
    const uploadArea = document.getElementById('clone-upload-area');

    if (recordBtn) {
        recordBtn.onclick = () => {
            recordBtn.classList.add('active');
            uploadBtn.classList.remove('active');
            recordArea.style.display = 'flex';
            uploadArea.style.display = 'none';
        };
    }
    if (uploadBtn) {
        uploadBtn.onclick = () => {
            uploadBtn.classList.add('active');
            recordBtn.classList.remove('active');
            recordArea.style.display = 'none';
            uploadArea.style.display = 'flex';
        };
    }

    // Recording logic
    const startBtn = document.getElementById('record-start-btn');
    if (startBtn) {
        startBtn.onclick = async () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
            } else {
                await startRecording();
            }
        };
    }

    // Upload logic
    const selectBtn = document.getElementById('clone-select-btn');
    const audioInput = document.getElementById('clone-audio-input');
    if (selectBtn && audioInput) {
        selectBtn.onclick = () => audioInput.click();
        audioInput.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            document.getElementById('upload-filename').textContent = file.name;
            cloneAudioBlob = file;
            document.getElementById('clone-submit-btn').disabled = false;
        };
    }

    // Submit clone
    const submitBtn = document.getElementById('clone-submit-btn');
    if (submitBtn) {
        submitBtn.onclick = async () => {
            const name = document.getElementById('clone-name').value.trim();
            if (!name) {
                showToast('Enter a voice name', true);
                return;
            }
            if (!cloneAudioBlob) {
                showToast('Record or upload audio first', true);
                return;
            }
            submitBtn.disabled = true;
            submitBtn.textContent = 'Cloning...';

            const reader = new FileReader();
            reader.onload = async () => {
                const b64 = reader.result.split(',')[1];
                dashboardWs.send(JSON.stringify({
                    type: 'clone_voice',
                    name: name,
                    audio_b64: b64,
                }));
            };
            reader.onerror = () => {
                showToast('Failed to read audio', true);
                submitBtn.disabled = false;
                submitBtn.textContent = 'Clone Voice';
            };
            reader.readAsDataURL(cloneAudioBlob);
        };
    }
});

async function startRecording() {
    // Check if HTTPS or localhost (required for microphone access)
    if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        showToast('Microphone requires HTTPS. Use https:// or localhost.', true);
        return;
    }
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            cloneAudioBlob = blob;
            document.getElementById('clone-submit-btn').disabled = false;
            document.getElementById('record-status').textContent = 'Recording saved (' + Math.round(blob.size / 1024) + ' KB)';
            stream.getTracks().forEach(t => t.stop());
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();
        const btn = document.getElementById('record-start-btn');
        btn.classList.add('recording');
        btn.querySelector('span').textContent = 'Stop';
        updateRecordTimer();
    } catch (e) {
        console.error('Microphone error:', e);
        if (e.name === 'NotAllowedError') {
            showToast('Microphone blocked. Click the lock icon in address bar to allow.', true);
        } else if (e.name === 'NotFoundError') {
            showToast('No microphone found on this device.', true);
        } else {
            showToast('Microphone error: ' + e.message, true);
        }
    }
}

function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
    }
    const btn = document.getElementById('record-start-btn');
    btn.classList.remove('recording');
    btn.querySelector('span').textContent = 'Start Recording';
}

function updateRecordTimer() {
    if (!recordingStartTime || !mediaRecorder || mediaRecorder.state !== 'recording') return;
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    document.getElementById('record-timer').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    requestAnimationFrame(updateRecordTimer);
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

// ── Page Tab Switching ──────────────────────────────────────────────

let currentPage = 'live';
let diaryInitialized = false;

function switchPage(page) {
    if (page === currentPage) return;
    currentPage = page;

    document.getElementById('page-live').style.display = page === 'live' ? '' : 'none';
    document.getElementById('page-diary').style.display = page === 'diary' ? '' : 'none';

    document.querySelectorAll('.page-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.page === page);
    });

    // Lazy-init diary on first visit
    if (page === 'diary' && !diaryInitialized && typeof initDiary === 'function') {
        diaryInitialized = true;
        initDiary();
    }

    // No need to pause/resume video — fetch-based MJPEG handles reconnection
}

// Expose for diary.js narration navigation
window.switchPage = switchPage;

function initPageTabs() {
    document.querySelectorAll('.page-tab').forEach(tab => {
        tab.addEventListener('click', () => switchPage(tab.dataset.page));
    });
}

// ── Send WS message helper ─────────────────────────────────────────

function sendDashboardWs(data) {
    if (dashboardWs && dashboardWs.readyState === WebSocket.OPEN) {
        dashboardWs.send(JSON.stringify(data));
    }
}

window.sendDashboardWs = sendDashboardWs;

// ── Smile Gallery Overlay ───────────────────────────────────────────

const GALLERY_COLORS = [
    '#f0883e', '#58a6ff', '#00d68f', '#bc8cff',
    '#f85149', '#f7dc6f', '#a8e6cf', '#ff8a80',
];

let smileGalleryEl = null;
let smileLightboxEl = null;

function createSmileGallery() {
    if (smileGalleryEl) return;

    // Overlay
    const el = document.createElement('div');
    el.className = 'smile-gallery-overlay';
    el.id = 'smile-gallery';
    el.innerHTML = `
        <div class="smile-gallery-inner">
            <div class="smile-gallery-header">
                <div class="smile-gallery-title">
                    <span class="smile-gallery-count" id="gallery-count">0</span>
                    <span class="smile-gallery-label">smiles collected</span>
                </div>
                <button class="smile-gallery-close" id="gallery-close">\u2715 Close</button>
            </div>
            <div class="smile-gallery-scatter" id="gallery-grid"></div>
            <div class="smile-gallery-timeline" id="gallery-timeline" style="display:none">
                <span class="smile-timeline-label" id="timeline-start"></span>
                <input type="range" class="smile-timeline-slider" id="timeline-slider" min="0" max="100" value="100">
                <span class="smile-timeline-label right" id="timeline-end"></span>
                <span class="smile-timeline-count" id="timeline-count"></span>
            </div>
        </div>
    `;
    document.body.appendChild(el);
    smileGalleryEl = el;

    document.getElementById('gallery-close').addEventListener('click', closeSmileGallery);
    el.addEventListener('click', (e) => {
        if (e.target === el) closeSmileGallery();
    });

    // Lightbox
    const lb = document.createElement('div');
    lb.className = 'smile-lightbox';
    lb.id = 'smile-lightbox';
    lb.innerHTML = `
        <img src="" alt="smile">
        <div class="smile-lightbox-info">
            <div class="smile-lightbox-filename" id="lightbox-filename"></div>
        </div>
    `;
    lb.addEventListener('click', () => {
        lb.classList.remove('active');
    });
    document.body.appendChild(lb);
    smileLightboxEl = lb;
}

// All loaded capture items with metadata
let galleryItems = []; // { file, ts, el }

async function openSmileGallery() {
    createSmileGallery();
    smileGalleryEl.classList.add('active');

    const grid = document.getElementById('gallery-grid');
    const countEl = document.getElementById('gallery-count');
    const timeline = document.getElementById('gallery-timeline');
    grid.innerHTML = '<div class="smile-gallery-loading" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">Loading smiles...</div>';
    timeline.style.display = 'none';

    try {
        const res = await fetch('/api/captures/list?limit=500');
        const data = await res.json();
        const files = (data.files || []).filter(f => f.endsWith('.jpg'));

        countEl.textContent = data.total || files.length;

        if (files.length === 0) {
            grid.innerHTML = '<div class="smile-gallery-empty" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">No smiles captured yet</div>';
            return;
        }

        grid.innerHTML = '';
        galleryItems = [];

        // Use full viewport for scatter
        const areaW = window.innerWidth;
        const areaH = window.innerHeight;
        const padTop = 80;    // header
        const padBot = 80;    // timeline
        const padSide = 24;

        files.forEach((file, i) => {
            const color = GALLERY_COLORS[i % GALLERY_COLORS.length];
            const item = document.createElement('div');
            item.className = 'smile-gallery-item';

            // Random size (70–180px, varied)
            const size = 70 + Math.random() * 110;
            item.style.width = size + 'px';
            item.style.height = size + 'px';

            // Random position across full screen
            const x = padSide + Math.random() * (areaW - size - padSide * 2);
            const y = padTop + Math.random() * (areaH - size - padTop - padBot);
            const rot = (Math.random() - 0.5) * 30;
            const baseTransform = `translate(${x}px, ${y}px)`;

            item.style.transform = `${baseTransform} rotate(${rot}deg)`;
            item.style.setProperty('--base-transform', baseTransform);
            item.style.setProperty('--item-color-solid', color);
            item.style.zIndex = i;

            // Parse timestamp from filename
            let ts = 0;
            const match = file.match(/smile_(\d+)_/);
            if (match) ts = parseInt(match[1]);
            const timeStr = ts ? new Date(ts * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }) : '';

            item.innerHTML = `
                <img src="/api/captures/image/${file}" alt="smile" loading="lazy">
                ${timeStr ? `<div class="smile-gallery-time">${timeStr}</div>` : ''}
            `;

            item.addEventListener('click', (e) => {
                e.stopPropagation();
                openLightbox(file);
            });

            grid.appendChild(item);
            galleryItems.push({ file, ts, el: item });

            // Staggered drop-in
            setTimeout(() => item.classList.add('revealed'), i * 15 + 80);
        });

        // Setup timeline slider
        setupTimelineSlider(galleryItems);

    } catch (e) {
        console.warn('Failed to load captures:', e);
        grid.innerHTML = '<div class="smile-gallery-empty" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">Could not load smiles</div>';
    }
}

function setupTimelineSlider(items) {
    const timeline = document.getElementById('gallery-timeline');
    const slider = document.getElementById('timeline-slider');
    const startLabel = document.getElementById('timeline-start');
    const endLabel = document.getElementById('timeline-end');
    const countLabel = document.getElementById('timeline-count');

    // Get time range
    const timestamps = items.filter(it => it.ts > 0).map(it => it.ts);
    if (timestamps.length < 2) {
        timeline.style.display = 'none';
        return;
    }

    const minTs = Math.min(...timestamps);
    const maxTs = Math.max(...timestamps);

    function formatDate(ts) {
        const d = new Date(ts * 1000);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
               d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    }

    startLabel.textContent = formatDate(minTs);
    endLabel.textContent = formatDate(maxTs);
    slider.value = 100;
    countLabel.textContent = items.length;
    timeline.style.display = 'flex';

    slider.oninput = () => {
        const pct = parseInt(slider.value);
        const cutoffTs = minTs + (maxTs - minTs) * (pct / 100);
        let visible = 0;

        for (const it of items) {
            const show = it.ts === 0 || it.ts <= cutoffTs;
            it.el.style.opacity = show ? '' : '0';
            it.el.style.pointerEvents = show ? '' : 'none';
            if (show) visible++;
        }

        countLabel.textContent = visible;
        endLabel.textContent = formatDate(Math.round(cutoffTs));
    };
}

function closeSmileGallery() {
    if (smileGalleryEl) smileGalleryEl.classList.remove('active');
    if (smileLightboxEl) smileLightboxEl.classList.remove('active');
}

function openLightbox(filename) {
    if (!smileLightboxEl) return;
    const img = smileLightboxEl.querySelector('img');
    const info = document.getElementById('lightbox-filename');
    img.src = `/api/captures/image/${filename}`;
    if (info) info.textContent = filename;
    smileLightboxEl.classList.add('active');
}

function initSmileGallery() {
    // Click on capture counter → open gallery
    const counter = document.getElementById('capture-counter');
    if (counter) {
        counter.style.cursor = 'pointer';
        counter.addEventListener('click', openSmileGallery);
    }

    // ESC to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (smileLightboxEl?.classList.contains('active')) {
                smileLightboxEl.classList.remove('active');
            } else if (smileGalleryEl?.classList.contains('active')) {
                closeSmileGallery();
            }
        }
    });
}

// ── Init ────────────────────────────────────────────────────────────
function init() {
    initPageTabs();
    setupVideo();
    connectVision();
    connectDashboard();
    initSettings();
    initSmileGallery();
    requestAnimationFrame(drawOverlay);
    requestAnimationFrame(drawFaceCrop);
}

init();
