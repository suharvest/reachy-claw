// HA Sensors settings binding. Loaded by index.html as a classic script
// (NOT type="module") to match the rest of the dashboard. Exposes
// window.bindHASettings / window.refreshHAEntities for app.js.
(function () {
"use strict";

let _haEntitiesCache = null;       // last fetched groups payload
let _haSelectedSet = new Set();    // entity_ids currently checked

function _byId(id) { return document.getElementById(id); }

function _setStatus(elId, msg, kind) {
  // kind: "ok" | "err" | "info" | null
  const el = _byId(elId);
  if (!el) return;
  el.textContent = msg || "";
  el.style.color = kind === "ok" ? "var(--green)"
                 : kind === "err" ? "var(--red)"
                 : "var(--text-dim)";
}

function _renderTestResult(ok, msg) {
  if (ok === null) {
    _setStatus("ha-test-result", msg, "info");
  } else {
    _setStatus("ha-test-result", msg || (ok ? "Connected" : "Error"), ok ? "ok" : "err");
  }
}

function _updateSelectionCount() {
  const el = _byId("ha-selection-count");
  if (!el) return;
  const total = _haEntitiesCache
    ? _haEntitiesCache.groups.reduce((n, g) => n + g.count, 0)
    : 0;
  el.textContent = `${_haSelectedSet.size} selected of ${total}`;
}

function _renderEntities(groups) {
  const root = _byId("ha-entities-tree");
  if (!root) return;
  root.innerHTML = "";
  if (!groups || groups.length === 0) {
    root.innerHTML = '<div class="face-empty">No entities returned by HA.</div>';
    return;
  }
  for (const g of groups) {
    const details = document.createElement("details");
    // Auto-open groups that contain a checked entity.
    if (g.entities.some((e) => _haSelectedSet.has(e.entity_id))) {
      details.open = true;
    }
    const summary = document.createElement("summary");
    summary.textContent = `${g.domain} (${g.count})`;
    details.appendChild(summary);
    const list = document.createElement("div");
    list.className = "ha-entity-list";
    for (const e of g.entities) {
      const label = document.createElement("label");
      label.className = "ha-entity-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = e.entity_id;
      cb.checked = _haSelectedSet.has(e.entity_id);
      cb.addEventListener("change", () => {
        if (cb.checked) _haSelectedSet.add(e.entity_id);
        else _haSelectedSet.delete(e.entity_id);
        _updateSelectionCount();
      });
      label.appendChild(cb);
      const txt = document.createElement("span");
      const fn = e.friendly_name ? `  · ${e.friendly_name}` : "";
      txt.textContent = ` ${e.entity_id} = ${e.state}${fn}`;
      label.appendChild(txt);
      list.appendChild(label);
    }
    details.appendChild(list);
    root.appendChild(details);
  }
  _updateSelectionCount();
}

async function _loadCurrentSettings() {
  const r = await fetch("/api/settings/ha");
  if (!r.ok) return null;
  return await r.json();
}

async function _saveCurrentSettings(payload) {
  const r = await fetch("/api/settings/ha", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${r.status}`);
  }
}

function _toast(msg, ok = true) {
  // Use existing toast if available, otherwise alert.
  if (typeof window.showToast === "function") {
    window.showToast(msg, ok ? "success" : "error");
  } else {
    console[ok ? "log" : "error"](msg);
  }
}

async function bindHASettings() {
  const cur = await _loadCurrentSettings();
  if (cur) {
    _byId("ha-url").value = cur.url || "";
    _byId("ha-token").value = cur.token || "";
    _haSelectedSet = new Set(cur.entities || []);
  }

  _byId("ha-token-toggle").addEventListener("click", () => {
    const f = _byId("ha-token");
    f.type = f.type === "password" ? "text" : "password";
  });

  let saveTimer = null;
  function debouncedSave(field, value) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try {
        await _saveCurrentSettings({ [field]: value });
        _toast(`Saved ${field}`);
      } catch (e) {
        _toast(`Save failed: ${e.message}`, false);
      }
    }, 500);
  }
  _byId("ha-url").addEventListener("input", (e) => debouncedSave("url", e.target.value.trim()));
  _byId("ha-token").addEventListener("input", (e) => debouncedSave("token", e.target.value));

  _byId("ha-test-btn").addEventListener("click", async () => {
    _renderTestResult(null, "Testing…");
    const body = {
      url: _byId("ha-url").value.trim(),
      token: _byId("ha-token").value,
    };
    const r = await fetch("/api/ha/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    _renderTestResult(j.ok, j.message || (j.ok ? "Connected" : `HTTP ${j.status}`));
  });

  _byId("ha-refresh-btn").addEventListener("click", refreshHAEntities);
  _byId("ha-save-btn").addEventListener("click", async () => {
    _setStatus("ha-save-status", "Saving…", "info");
    try {
      await _saveCurrentSettings({ entities: Array.from(_haSelectedSet).sort() });
      _setStatus("ha-save-status", `Saved ${_haSelectedSet.size} entities.`, "ok");
      _toast(`Saved ${_haSelectedSet.size} entities`);
    } catch (e) {
      _setStatus("ha-save-status", `Save failed: ${e.message}`, "err");
      _toast(`Save failed: ${e.message}`, false);
    }
  });
}

async function refreshHAEntities() {
  const tree = _byId("ha-entities-tree");
  if (!tree) return;
  tree.innerHTML = '<div class="face-empty">Loading…</div>';
  let r;
  try {
    r = await fetch("/api/ha/entities");
  } catch (e) {
    tree.innerHTML = `<div class="face-empty" style="color:var(--red)">Network error: ${e.message}</div>`;
    return;
  }
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    tree.innerHTML = `<div class="face-empty" style="color:var(--red)">${body.error || `HTTP ${r.status}`}</div>`;
    return;
  }
  _haEntitiesCache = await r.json();
  _renderEntities(_haEntitiesCache.groups);
}

// Expose to other classic scripts (app.js).
window.bindHASettings = bindHASettings;
window.refreshHAEntities = refreshHAEntities;
})();