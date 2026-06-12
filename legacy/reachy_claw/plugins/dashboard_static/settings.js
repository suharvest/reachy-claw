// Rest + Diary settings — wires existing DOM to /api/settings/{rest,diary}
// and /api/diary/{status,generate,publish}. Uses styles from style.css
// (toggle-row / motor-sleep-toggle / url-input / restart-btn / face-list).
"use strict";

async function fetchSettings(ns) {
  const r = await fetch(`/api/settings/${ns}`);
  if (!r.ok) throw new Error(`GET /api/settings/${ns} -> ${r.status}`);
  return r.json();
}

async function putSettings(ns, body) {
  const r = await fetch(`/api/settings/${ns}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `PUT /api/settings/${ns} -> ${r.status}`);
  }
  return r.json();
}

function setToggle(btn, on) {
  if (!btn) return;
  btn.classList.toggle("active", !!on);
}

function flashStatus(el, msg, ok) {
  if (!el) return;
  el.textContent = msg;
  el.style.color = ok ? "var(--green)" : "var(--red)";
  setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 3000);
}

// ── Rest Window (lives inside the General tab) ──────────────────────
let _restPoller = null;

async function fetchRestStatus() {
  const r = await fetch("/api/rest/status");
  if (!r.ok) throw new Error(`GET /api/rest/status -> ${r.status}`);
  return r.json();
}

async function postRestForce(action) {
  const r = await fetch("/api/rest/force", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `POST /api/rest/force -> ${r.status}`);
  }
  return r.json();
}

function updateRestBadge(state) {
  const badge = document.getElementById("rest-state-badge");
  if (!badge) return;
  badge.classList.remove("active", "resting", "forced");
  if (state.resting) {
    badge.classList.add("resting");
    badge.textContent = state.force_state === true ? "Resting (forced)" : "Resting";
  } else {
    badge.classList.add("active");
    badge.textContent = state.force_state === false ? "Awake (forced)" : "Awake";
  }
  if (state.force_state !== null && state.force_state !== undefined) {
    badge.classList.add("forced");
  }
}

async function refreshRestState() {
  try {
    const s = await fetchRestStatus();
    updateRestBadge(s);
  } catch (e) { /* silent */ }
}

async function bindRestSettings() {
  const enabledBtn = document.getElementById("rest-enabled-toggle");
  const startInput = document.getElementById("rest-window-start");
  const endInput = document.getElementById("rest-window-end");
  const tzInput = document.getElementById("rest-timezone");
  const saveBtn = document.getElementById("rest-save");
  const statusEl = document.getElementById("rest-status");
  const enterBtn = document.getElementById("rest-force-enter");
  const exitBtn = document.getElementById("rest-force-exit");
  const clearBtn = document.getElementById("rest-force-clear");
  if (!enabledBtn || !saveBtn) return;

  // Always re-load current state on bind (modal reopened).
  try {
    const cur = await fetchSettings("rest");
    setToggle(enabledBtn, cur.enabled);
    startInput.value = cur.window_start;
    endInput.value = cur.window_end;
    tzInput.value = cur.timezone;
  } catch (e) {
    flashStatus(statusEl, "Load failed: " + e.message, false);
  }
  await refreshRestState();

  if (saveBtn.dataset.bound) {
    // Already wired — only state and inputs needed refresh.
    return;
  }
  saveBtn.dataset.bound = "1";

  enabledBtn.addEventListener("click", () => enabledBtn.classList.toggle("active"));

  saveBtn.addEventListener("click", async () => {
    saveBtn.disabled = true;
    const body = {
      enabled: enabledBtn.classList.contains("active"),
      window_start: startInput.value,
      window_end: endInput.value,
      timezone: tzInput.value,
    };
    try {
      await putSettings("rest", body);
      flashStatus(statusEl, "Saved", true);
      await refreshRestState();
    } catch (e) {
      flashStatus(statusEl, e.message, false);
    } finally {
      saveBtn.disabled = false;
    }
  });

  enterBtn.addEventListener("click", async () => {
    try {
      await postRestForce("enter");
      flashStatus(statusEl, "Forced into rest", true);
      await refreshRestState();
    } catch (e) { flashStatus(statusEl, e.message, false); }
  });
  exitBtn.addEventListener("click", async () => {
    try {
      await postRestForce("exit");
      flashStatus(statusEl, "Forced awake", true);
      await refreshRestState();
    } catch (e) { flashStatus(statusEl, e.message, false); }
  });
  clearBtn.addEventListener("click", async () => {
    try {
      await postRestForce("clear");
      flashStatus(statusEl, "Following schedule", true);
      await refreshRestState();
    } catch (e) { flashStatus(statusEl, e.message, false); }
  });

  // Poll status every 5s while modal is open (cheap; one row in JSON).
  if (_restPoller) clearInterval(_restPoller);
  _restPoller = setInterval(refreshRestState, 5000);
}

// ── Diary Publishing (its own tab) ──────────────────────────────────
async function bindDiarySettings() {
  const autoPubBtn = document.getElementById("diary-auto-publish-toggle");
  const lintBtn = document.getElementById("diary-privacy-linter-toggle");
  const repoInput = document.getElementById("diary-site-repo-url");
  const pathInput = document.getElementById("diary-site-diary-path");
  const branchInput = document.getElementById("diary-site-branch");
  const saveBtn = document.getElementById("diary-save");
  const statusEl = document.getElementById("diary-status");
  if (!saveBtn) return;

  // Always reload settings + history on tab activation so it stays fresh.
  try {
    const cur = await fetchSettings("diary");
    setToggle(autoPubBtn, cur.auto_publish);
    setToggle(lintBtn, cur.privacy_linter);
    repoInput.value = cur.site_repo_url || "";
    pathInput.value = cur.site_diary_path || "src/content/docs";
    branchInput.value = cur.site_branch || "main";
  } catch (e) {
    flashStatus(statusEl, "Load failed: " + e.message, false);
  }

  if (!saveBtn.dataset.bound) {
    saveBtn.dataset.bound = "1";

    autoPubBtn.addEventListener("click", () => autoPubBtn.classList.toggle("active"));
    lintBtn.addEventListener("click", () => lintBtn.classList.toggle("active"));

    saveBtn.addEventListener("click", async () => {
      saveBtn.disabled = true;
      const body = {
        auto_publish: autoPubBtn.classList.contains("active"),
        privacy_linter: lintBtn.classList.contains("active"),
        site_repo_url: repoInput.value.trim(),
        site_diary_path: pathInput.value.trim(),
        site_branch: branchInput.value.trim(),
      };
      try {
        await putSettings("diary", body);
        flashStatus(statusEl, "Saved", true);
      } catch (e) {
        flashStatus(statusEl, e.message, false);
      } finally {
        saveBtn.disabled = false;
      }
    });
  }

  await renderDiaryHistory();
}

async function renderDiaryHistory() {
  const root = document.getElementById("diary-history");
  if (!root) return;
  try {
    const r = await fetch("/api/diary/status");
    const { dates } = await r.json();
    if (!dates || dates.length === 0) {
      root.innerHTML = `<div class="face-empty">No diary records yet.</div>`;
      return;
    }
    root.innerHTML = "";
    for (const d of dates) {
      const row = document.createElement("div");
      row.className = "toggle-row";
      row.style.marginBottom = "6px";

      const status = d.published
        ? `<span style="color: var(--green); font-weight: 600;">✓ Published</span>`
        : (d.generated
            ? `<span style="color: #f59e0b; font-weight: 600;">⚠ Unpublished</span>`
            : `<span style="color: var(--text-dim);">— Missing</span>`);

      const action = d.published
        ? { label: "Regenerate", op: () => trigger("generate", d.date, true) }
        : (d.generated
            ? { label: "Publish", op: () => trigger("publish", d.date, false) }
            : { label: "Generate + Publish", op: async () => {
                await trigger("generate", d.date, false);
                // small wait so the row reflects "generated"
                await new Promise(res => setTimeout(res, 600));
                await trigger("publish", d.date, false);
              } });

      row.innerHTML = `
        <span class="toggle-label" style="font-family: monospace;">${d.date}</span>
        <span style="flex: 1; text-align: center;">${status}</span>
        <button class="enroll-btn">${action.label}</button>
      `;
      const btn = row.querySelector("button");
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.textContent = "Working...";
        try { await action.op(); } catch (e) { alert(e.message); }
        await new Promise(res => setTimeout(res, 800));
        await renderDiaryHistory();
      });
      root.appendChild(row);
    }
  } catch (e) {
    root.innerHTML = `<div class="face-empty">Failed: ${e.message}</div>`;
  }
}

async function trigger(kind, date, force) {
  const r = await fetch(`/api/diary/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, force }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `POST /api/diary/${kind} -> ${r.status}`);
  }
  return r.json();
}

// Public hooks called from app.js when the modal opens / Diary tab is clicked.
window.bindRestSettings = bindRestSettings;
window.bindDiarySettings = bindDiarySettings;
