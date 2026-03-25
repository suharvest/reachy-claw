// ── Daily Diary Page — Storyboard-inspired ──────────────────────────
// Design ref: Reachymini-Storyboard (ziyu-wei)

// ── State ───────────────────────────────────────────────────────────
let diaryDates = [];
let diaryCurrentIdx = -1;
let diaryData = null;
let diaryNarrating = false;

const DIARY_EMOJI = {
    happy: '\u{1F60A}', sad: '\u{1F622}', thinking: '\u{1F914}',
    surprised: '\u{1F631}', curious: '\u{1F9D0}', excited: '\u{1F929}',
    neutral: '\u{1F610}', confused: '\u{1F615}', angry: '\u{1F620}',
    laugh: '\u{1F602}', fear: '\u{1F628}', listening: '\u{1F3A7}',
    contemplative: '\u{1F914}',
};

const STAT_COLORS = ['green', 'orange', 'blue', 'purple', 'green', 'orange'];

// ── DOM refs ────────────────────────────────────────────────────────
const diaryContent = document.getElementById('diary-content');
const diaryDateLabel = document.getElementById('diary-date-label');
const diaryPrevBtn = document.getElementById('diary-prev');
const diaryNextBtn = document.getElementById('diary-next');
const narrateBtn = document.getElementById('narrate-btn');

// ── Init ────────────────────────────────────────────────────────────

async function initDiary() {
    await loadDiaryList();
    diaryPrevBtn?.addEventListener('click', () => navigateDiary(1));   // older
    diaryNextBtn?.addEventListener('click', () => navigateDiary(-1));  // newer
    narrateBtn?.addEventListener('click', toggleNarration);

    document.addEventListener('keydown', (e) => {
        if (document.getElementById('page-diary')?.style.display === 'none') return;
        if (e.key === 'ArrowLeft') navigateDiary(1);
        if (e.key === 'ArrowRight') navigateDiary(-1);
    });
}

async function loadDiaryList() {
    try {
        const res = await fetch('/api/diaries');
        const data = await res.json();
        diaryDates = data.dates || [];
        if (diaryDates.length > 0) {
            diaryCurrentIdx = 0;
            await loadDiary(diaryDates[0]);
        } else {
            showDiaryEmpty();
        }
    } catch (e) {
        console.warn('Failed to load diary list:', e);
        showDiaryEmpty();
    }
    updateDateNav();
}

async function loadDiary(date) {
    if (!diaryContent) return;
    diaryContent.innerHTML = '<div class="diary-loading">Loading</div>';

    try {
        const res = await fetch(`/api/diary/${date}`);
        if (!res.ok) { showDiaryEmpty(date); return; }
        diaryData = await res.json();
        renderDiary(diaryData);
    } catch (e) {
        console.warn('Failed to load diary:', e);
        showDiaryEmpty(date);
    }
}

function navigateDiary(delta) {
    const newIdx = diaryCurrentIdx + delta;
    if (newIdx < 0 || newIdx >= diaryDates.length) return;
    diaryCurrentIdx = newIdx;
    loadDiary(diaryDates[diaryCurrentIdx]);
    updateDateNav();
}

function updateDateNav() {
    if (!diaryDateLabel) return;
    if (diaryDates.length === 0) {
        diaryDateLabel.textContent = 'No diaries';
        if (diaryPrevBtn) diaryPrevBtn.disabled = true;
        if (diaryNextBtn) diaryNextBtn.disabled = true;
        return;
    }
    const date = diaryDates[diaryCurrentIdx];
    const isToday = date === new Date().toISOString().slice(0, 10);
    diaryDateLabel.textContent = isToday ? `${date} · Today` : date;
    if (diaryPrevBtn) diaryPrevBtn.disabled = diaryCurrentIdx >= diaryDates.length - 1;
    if (diaryNextBtn) diaryNextBtn.disabled = diaryCurrentIdx <= 0;
}

// ── Render ──────────────────────────────────────────────────────────

function renderDiary(diary) {
    if (!diaryContent) return;

    let html = '<div class="diary-inner">';

    // Header (Storyboard style)
    html += `
        <div class="diary-header">
            <div class="diary-subtitle">Day ${diary.date || ''}</div>
            <div class="diary-title">${esc(diary.title || 'Daily Diary')}</div>
        </div>
    `;

    // Timeline sections
    html += '<div class="diary-sections">';
    for (const section of (diary.sections || [])) {
        html += renderSection(section);
    }
    html += '</div>';

    // Stats footer
    const faceSection = (diary.sections || []).find(s => s.id === 'faces');
    if (faceSection?.data) {
        html += renderFooterStats(faceSection.data);
    }

    html += '</div>'; // /diary-inner

    diaryContent.innerHTML = html;

    // Scroll-triggered section reveal
    requestAnimationFrame(() => {
        observeSections();
        // Draw charts
        for (const section of (diary.sections || [])) {
            if (section.type === 'chart' && section.data) {
                drawMoodChart(section.id, section.data);
            }
        }
    });
}

function renderSection(section) {
    const labelMap = {
        summary: 'Summary', mood_curve: 'Emotional Journey',
        conversations: 'Conversations', faces: 'People & Smiles',
        thoughts: 'Reflections', environment: 'Environment',
        smile_wall: 'Smile Wall',
    };
    const label = labelMap[section.id] || section.id;

    let inner = '';

    switch (section.type) {
        case 'narrative':
            inner = `<div class="diary-section-text">${esc(section.content)}</div>`;
            break;

        case 'chart':
            inner = `
                <div class="diary-section-text">${esc(section.content)}</div>
                <div class="mood-chart-container">
                    <canvas class="mood-chart" id="chart-${section.id}"></canvas>
                </div>
            `;
            break;

        case 'highlights':
            inner = `<div class="diary-section-text">${esc(section.content)}</div>`;
            if (section.id === 'conversations' && section.items) {
                inner += renderConversations(section.items);
            } else if (section.items) {
                inner += renderThoughts(section.items);
            }
            break;

        case 'stats':
            inner = `<div class="diary-section-text">${esc(section.content)}</div>`;
            if (section.data) {
                inner += renderStats(section.data);
            }
            break;

        case 'sensors':
            inner = `<div class="diary-section-text">${esc(section.content)}</div>`;
            if (section.data) {
                inner += renderSensors(section.data);
            }
            break;

        default:
            inner = `<div class="diary-section-text">${esc(section.content || '')}</div>`;
    }

    return `
        <div class="diary-section" id="diary-section-${section.id}" data-section-id="${section.id}">
            <div class="diary-section-label">${esc(label)}</div>
            ${inner}
        </div>
    `;
}

function renderConversations(items) {
    let html = '<div class="diary-conv-list">';
    for (const item of items) {
        const emoji = DIARY_EMOJI[item.emotion] || '';
        html += `<div class="diary-conv-item">`;
        html += `<div class="diary-conv-time">${esc(item.time)}</div>`;

        // User bubble (right aligned)
        if (item.user) {
            html += `
                <div class="diary-conv-bubble user">
                    <div class="conv-sender">Visitor</div>
                    ${esc(item.user)}
                </div>
            `;
        }

        // Bot reply (left aligned)
        if (item.reply) {
            html += `
                <div class="diary-conv-bubble reply">
                    <div class="conv-sender">
                        <span>\u{1F916}</span>
                        <span>Reachy Mini</span>
                    </div>
                    ${esc(item.reply)} ${emoji}
                </div>
            `;
        }

        html += `</div>`;
    }
    html += '</div>';
    return html;
}

function renderThoughts(items) {
    let html = '<div class="diary-thought-list">';
    for (const item of items) {
        const emoji = DIARY_EMOJI[item.emotion] || '';
        html += `
            <div class="diary-thought-card">
                <div class="diary-thought-time">${esc(item.time)} <span class="diary-thought-emoji">${emoji}</span></div>
                <div class="diary-thought-text">${esc(item.text)}</div>
            </div>
        `;
    }
    html += '</div>';
    return html;
}

function renderStats(data) {
    const items = [
        { value: data.faces_seen ?? '-', label: 'People', color: 'orange' },
        { value: data.smiles_captured ?? '-', label: 'Smiles', color: 'green' },
        { value: data.peak_hour ?? '-', label: 'Peak', color: 'blue' },
    ];

    const known = data.known_people || {};
    const knownCount = Object.keys(known).length;
    if (knownCount > 0) {
        items.push({ value: knownCount, label: 'Recognized', color: 'purple' });
    }

    let html = '<div class="diary-stats-row">';
    for (const item of items) {
        html += `
            <div class="diary-stat-card">
                <div class="diary-stat-value" data-color="${item.color}" data-target="${item.value}">${item.value}</div>
                <div class="diary-stat-label">${esc(item.label)}</div>
            </div>
        `;
    }
    html += '</div>';
    return html;
}

function renderSensors(data) {
    const sensors = [];
    if (data.temperature != null) sensors.push({ icon: '\u{1F321}', value: `${data.temperature}\u00B0C`, label: 'Temperature' });
    if (data.humidity != null) sensors.push({ icon: '\u{1F4A7}', value: `${data.humidity}%`, label: 'Humidity' });
    if (data.weather) sensors.push({ icon: weatherIcon(data.weather), value: data.weather, label: 'Weather' });
    if (data.location) sensors.push({ icon: '\u{1F4CD}', value: data.location, label: 'Location' });

    let html = '<div class="diary-sensor-row">';
    for (const s of sensors) {
        html += `
            <div class="diary-sensor-card">
                <span class="diary-sensor-icon">${s.icon}</span>
                <div>
                    <div class="diary-sensor-value">${esc(s.value)}</div>
                    <div class="diary-sensor-label">${esc(s.label)}</div>
                </div>
            </div>
        `;
    }
    html += '</div>';
    return html;
}

function renderFooterStats(data) {
    const items = [
        { value: data.faces_seen ?? 0, label: 'Total Faces', color: '#f0883e' },
        { value: data.smiles_captured ?? 0, label: 'Smiles', color: '#00d68f' },
        { value: data.peak_hour || '--', label: 'Peak Hour', color: '#58a6ff' },
    ];

    let html = `
        <div class="diary-footer">
            <div class="diary-footer-label">Data Summary</div>
            <div class="diary-stats-row">
    `;
    for (const item of items) {
        html += `
            <div class="diary-stat-card">
                <div class="diary-stat-value" style="color: ${item.color}; text-shadow: 0 0 30px ${item.color}33;"
                     data-target="${item.value}">${item.value}</div>
                <div class="diary-stat-label">${esc(item.label)}</div>
            </div>
        `;
    }
    html += '</div></div>';
    return html;
}

function weatherIcon(weather) {
    const w = (weather || '').toLowerCase();
    if (w.includes('rain')) return '\u{1F327}';
    if (w.includes('cloud')) return '\u{2601}';
    if (w.includes('snow')) return '\u{2744}';
    if (w.includes('sun') || w.includes('clear')) return '\u{2600}';
    return '\u{1F324}';
}

// ── Scroll-triggered section animation ──────────────────────────────

function observeSections() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Animate number counters inside this section
                entry.target.querySelectorAll('.diary-stat-value[data-target]').forEach(el => {
                    animateNumber(el);
                });
            }
        });
    }, { threshold: 0.15, root: document.querySelector('.diary-page') });

    document.querySelectorAll('.diary-section, .diary-footer').forEach(el => {
        observer.observe(el);
    });
}

// ── Animated number counter ─────────────────────────────────────────

function animateNumber(el) {
    if (el.dataset.animated) return;
    el.dataset.animated = '1';

    const target = el.dataset.target;
    if (!target) return;

    // Skip non-numeric values (time strings, dashes)
    const num = parseInt(target);
    if (isNaN(num)) return;

    const duration = 1200;
    const start = performance.now();

    function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Cubic ease-out
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = String(Math.round(num * eased));
        if (progress < 1) requestAnimationFrame(tick);
    }
    el.textContent = '0';
    requestAnimationFrame(tick);
}

// ── Mood Chart (Canvas 2D — Storyboard style) ───────────────────────

function drawMoodChart(sectionId, data, canvasId) {
    const canvas = document.getElementById(canvasId || `chart-${sectionId}`);
    if (!canvas || !data.length) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 24, right: 16, bottom: 28, left: 36 };
    const chartW = W - pad.left - pad.right;
    const chartH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    // Y-axis labels (subtle)
    ctx.fillStyle = '#333';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (chartH / 4) * i;
        const val = 100 - (i * 25);
        ctx.fillText(val, pad.left - 8, y + 3);
        // Grid line
        ctx.strokeStyle = '#1a1a1f';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
    }

    // X-axis labels
    ctx.textAlign = 'center';
    ctx.fillStyle = '#333';
    for (let i = 0; i < data.length; i++) {
        const x = pad.left + (chartW / Math.max(data.length - 1, 1)) * i;
        ctx.fillText(data[i].t, x, H - 6);
    }

    // Points
    const points = data.map((d, i) => ({
        x: pad.left + (chartW / Math.max(data.length - 1, 1)) * i,
        y: pad.top + chartH - ((d.v - 20) / 80) * chartH, // Scale 20-100 range
    }));

    // Smooth curve helper (Catmull-Rom → Bezier)
    function drawSmoothLine(pts) {
        if (pts.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[Math.max(i - 1, 0)];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            const p3 = pts[Math.min(i + 2, pts.length - 1)];
            const cp1x = p1.x + (p2.x - p0.x) / 6;
            const cp1y = p1.y + (p2.y - p0.y) / 6;
            const cp2x = p2.x - (p3.x - p1.x) / 6;
            const cp2y = p2.y - (p3.y - p1.y) / 6;
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
        }
    }

    // Gradient fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    grad.addColorStop(0, 'rgba(0, 214, 143, 0.25)');
    grad.addColorStop(1, 'rgba(0, 214, 143, 0)');

    drawSmoothLine(points);
    // Close for fill
    ctx.lineTo(points[points.length - 1].x, H - pad.bottom);
    ctx.lineTo(points[0].x, H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line stroke
    drawSmoothLine(points);
    ctx.strokeStyle = '#00d68f';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Peak marker
    const peakIdx = points.reduce((mi, p, i, arr) => p.y < arr[mi].y ? i : mi, 0);
    const peak = points[peakIdx];
    const peakVal = data[peakIdx].v;

    // Glow ring
    ctx.beginPath();
    ctx.arc(peak.x, peak.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 214, 143, 0.2)';
    ctx.fill();

    // Solid dot
    ctx.beginPath();
    ctx.arc(peak.x, peak.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#00d68f';
    ctx.fill();
    ctx.strokeStyle = '#0a0a0c';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Peak label
    ctx.fillStyle = '#00d68f';
    ctx.font = 'bold 11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`Peak: ${peakVal}`, peak.x, peak.y - 16);
}

// ── Fullscreen Narration Overlay ─────────────────────────────────────

let narrationOverlay = null;
let narrationSections = [];
let narrationCurrentIdx = -1;

const SECTION_LABEL_MAP = {
    summary: 'Summary', mood_curve: 'Emotional Journey',
    conversations: 'Conversations', faces: 'People & Smiles',
    thoughts: 'Reflections', environment: 'Environment',
};

const SECTION_ACCENT = {
    summary: 'green', mood_curve: 'green', conversations: 'blue',
    faces: 'orange', thoughts: 'purple', environment: 'blue',
};

function createNarrationOverlay() {
    if (narrationOverlay) return;

    const el = document.createElement('div');
    el.className = 'narration-overlay';
    el.id = 'narration-overlay';
    el.innerHTML = `
        <button class="narration-stop" id="narration-stop-btn">\u2715 Stop</button>
        <div class="narration-label" id="narration-label"></div>
        <div class="narration-content" id="narration-content-area"></div>
        <div class="narration-progress" id="narration-progress"></div>
    `;

    // Floating particles
    for (let i = 0; i < 10; i++) {
        const p = document.createElement('div');
        p.className = 'narration-particle';
        const size = Math.random() * 2.5 + 1;
        p.style.cssText = `
            left: ${10 + Math.random() * 80}%;
            top: ${5 + Math.random() * 90}%;
            width: ${size}px; height: ${size}px;
            --dur: ${2 + Math.random() * 3}s;
            --delay: ${Math.random() * 2}s;
        `;
        el.appendChild(p);
    }

    document.body.appendChild(el);
    narrationOverlay = el;

    document.getElementById('narration-stop-btn').addEventListener('click', () => {
        stopNarrationUI();
        if (typeof sendDashboardWs === 'function') {
            sendDashboardWs({ type: 'diary_narrate_stop' });
        }
    });
}

function toggleNarration() {
    if (!diaryData) return;
    diaryNarrating = !diaryNarrating;

    if (diaryNarrating) {
        narrateBtn?.classList.add('active');
        narrationSections = (diaryData.sections || []).filter(s => s.content);
        narrationCurrentIdx = -1;
        openNarrationOverlay();
        if (typeof sendDashboardWs === 'function') {
            sendDashboardWs({ type: 'diary_narrate_start', date: diaryData.date });
        }
    } else {
        stopNarrationUI();
        if (typeof sendDashboardWs === 'function') {
            sendDashboardWs({ type: 'diary_narrate_stop' });
        }
    }
}

function openNarrationOverlay() {
    createNarrationOverlay();

    // Build progress dots
    const progressEl = document.getElementById('narration-progress');
    progressEl.innerHTML = narrationSections.map((s, i) =>
        `<div class="narration-dot" data-idx="${i}"></div>`
    ).join('');

    // Show with title card first
    const contentArea = document.getElementById('narration-content-area');
    const labelEl = document.getElementById('narration-label');
    labelEl.textContent = diaryData.date || '';
    contentArea.innerHTML = `<div class="narration-text">${esc(diaryData.title || 'Daily Diary')}</div>`;

    narrationOverlay.classList.add('active');
}

function showNarrationSection(sectionId) {
    if (!narrationOverlay || !diaryData) return;

    const idx = narrationSections.findIndex(s => s.id === sectionId);
    if (idx < 0) return;

    const section = narrationSections[idx];
    const prevIdx = narrationCurrentIdx;
    narrationCurrentIdx = idx;

    const contentArea = document.getElementById('narration-content-area');
    const labelEl = document.getElementById('narration-label');
    const accent = SECTION_ACCENT[section.id] || 'green';
    narrationOverlay.setAttribute('data-accent', accent);

    // Fade out current
    contentArea.classList.add('fade-out');

    setTimeout(() => {
        // Update label
        labelEl.textContent = SECTION_LABEL_MAP[section.id] || section.id;

        // Build section-specific content
        let html = `<div class="narration-text">${esc(section.content)}</div>`;

        // Add rich content below text
        if (section.type === 'chart' && section.data) {
            html += `<canvas class="narration-chart" id="narration-chart"></canvas>`;
        }

        if (section.type === 'stats' && section.data) {
            const d = section.data;
            html += `<div class="narration-stats">`;
            if (d.faces_seen != null) html += `
                <div class="narration-stat">
                    <div class="narration-stat-value" style="color: #f0883e; text-shadow: 0 0 40px rgba(240,136,62,0.3);"
                         data-target="${d.faces_seen}">${d.faces_seen}</div>
                    <div class="narration-stat-label">People</div>
                </div>`;
            if (d.smiles_captured != null) html += `
                <div class="narration-stat">
                    <div class="narration-stat-value" style="color: #00d68f; text-shadow: 0 0 40px rgba(0,214,143,0.3);"
                         data-target="${d.smiles_captured}">${d.smiles_captured}</div>
                    <div class="narration-stat-label">Smiles</div>
                </div>`;
            if (d.peak_hour) html += `
                <div class="narration-stat">
                    <div class="narration-stat-value" style="color: #58a6ff; text-shadow: 0 0 40px rgba(88,166,255,0.3);">${d.peak_hour}</div>
                    <div class="narration-stat-label">Peak</div>
                </div>`;
            html += `</div>`;
        }

        if (section.type === 'highlights' && section.items) {
            if (section.id === 'conversations') {
                html += '<div class="narration-conv">';
                // Show top 2 conversations in narration
                for (const item of section.items.slice(0, 2)) {
                    const emoji = DIARY_EMOJI[item.emotion] || '';
                    if (item.user) html += `<div class="diary-conv-bubble user" style="max-width:90%"><div class="conv-sender">Visitor</div>${esc(item.user)}</div>`;
                    if (item.reply) html += `<div class="diary-conv-bubble reply" style="max-width:90%"><div class="conv-sender"><span>\u{1F916}</span><span>Reachy Mini</span></div>${esc(item.reply)} ${emoji}</div>`;
                }
                html += '</div>';
            } else {
                // Thoughts — show first one large
                const first = section.items[0];
                if (first) {
                    const emoji = DIARY_EMOJI[first.emotion] || '';
                    html += `<div style="margin-top: 20px; font-family: 'Noto Serif SC', serif; font-size: 18px; color: #8b949e; line-height: 1.7; font-style: italic;">${emoji} ${esc(first.text)}</div>`;
                }
            }
        }

        if (section.type === 'sensors' && section.data) {
            const d = section.data;
            html += `<div class="narration-stats" style="margin-top: 24px;">`;
            if (d.temperature != null) html += `<div class="narration-stat"><div class="narration-stat-value" style="color: #f0883e; font-size: 36px;">${d.temperature}\u00B0</div><div class="narration-stat-label">Temperature</div></div>`;
            if (d.humidity != null) html += `<div class="narration-stat"><div class="narration-stat-value" style="color: #58a6ff; font-size: 36px;">${d.humidity}%</div><div class="narration-stat-label">Humidity</div></div>`;
            if (d.weather) html += `<div class="narration-stat"><div class="narration-stat-value" style="color: #bc8cff; font-size: 36px;">${weatherIcon(d.weather)}</div><div class="narration-stat-label">${esc(d.weather)}</div></div>`;
            html += `</div>`;
        }

        contentArea.innerHTML = html;
        contentArea.classList.remove('fade-out');

        // Update progress dots
        document.querySelectorAll('.narration-dot').forEach((dot, i) => {
            dot.classList.toggle('active', i === idx);
            dot.classList.toggle('done', i < idx);
        });

        // Draw chart if needed
        if (section.type === 'chart' && section.data) {
            requestAnimationFrame(() => drawMoodChart('narration-chart', section.data, 'narration-chart'));
        }

        // Animate numbers
        contentArea.querySelectorAll('.narration-stat-value[data-target]').forEach(el => {
            el.dataset.animated = '';
            animateNumber(el);
        });

    }, 500); // Wait for fade-out
}

function stopNarrationUI() {
    diaryNarrating = false;
    narrationCurrentIdx = -1;

    narrateBtn?.classList.remove('active');
    if (narrateBtn) narrateBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg> Narrate';

    if (narrationOverlay) {
        narrationOverlay.classList.remove('active');
    }
}

function handleNarrationMessage(data) {
    switch (data.type) {
        case 'diary_narrate_focus': {
            if (!narrationOverlay || !narrationOverlay.classList.contains('active')) {
                // If overlay not open yet, open it
                if (diaryData) {
                    narrationSections = (diaryData.sections || []).filter(s => s.content);
                    openNarrationOverlay();
                }
            }
            if (data.state === 'speaking') {
                showNarrationSection(data.section_id);
            }
            // Also highlight in background diary page
            document.querySelectorAll('.diary-section.narration-spotlight').forEach(el => {
                el.classList.remove('narration-spotlight');
            });
            const bgEl = document.getElementById(`diary-section-${data.section_id}`);
            if (bgEl) bgEl.classList.add('narration-spotlight');
            break;
        }
        case 'diary_narrate_navigate': {
            if (data.action === 'switch_date' && data.date) {
                const idx = diaryDates.indexOf(data.date);
                if (idx >= 0) {
                    diaryCurrentIdx = idx;
                    loadDiary(data.date);
                    updateDateNav();
                }
            }
            if (data.action === 'switch_tab' && data.tab) {
                stopNarrationUI();
                if (typeof switchPage === 'function') switchPage(data.tab);
            }
            break;
        }
        case 'diary_narrate_end':
            // Final fade before closing
            if (narrationOverlay) {
                const contentArea = document.getElementById('narration-content-area');
                const labelEl = document.getElementById('narration-label');
                contentArea.classList.add('fade-out');
                setTimeout(() => {
                    labelEl.textContent = '';
                    contentArea.innerHTML = '<div class="narration-text" style="color: #555;">End of diary</div>';
                    contentArea.classList.remove('fade-out');
                    // Close after a moment
                    setTimeout(() => stopNarrationUI(), 2000);
                }, 500);
            } else {
                stopNarrationUI();
            }
            break;
    }
}

// ── Empty state ─────────────────────────────────────────────────────

function showDiaryEmpty(date) {
    if (!diaryContent) return;
    diaryContent.innerHTML = `
        <div class="diary-inner">
            <div class="diary-empty">
                <div class="diary-empty-icon">\u{1F4D6}</div>
                <div class="diary-empty-text">${date ? `No diary for ${esc(date)}` : 'No diaries yet'}</div>
                <div class="diary-empty-hint">Diaries are generated daily from interaction data</div>
            </div>
        </div>
    `;
}

// ── Helpers ─────────────────────────────────────────────────────────

function esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Expose for WS handler and page switching
window.initDiary = initDiary;
window.loadDiary = loadDiary;
window.handleNarrationMessage = handleNarrationMessage;
window.stopNarrationUI = stopNarrationUI;
