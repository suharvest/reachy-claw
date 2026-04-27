// HA Sensors settings binding. Loaded by index.html as a classic script
// (NOT type="module") to match the rest of the dashboard. Exposes
// window.bindHASettings / window.refreshHAEntities for app.js.
(function () {
"use strict";

let _haEntitiesCache = null;       // last fetched groups payload
let _haSelectedSet = new Set();    // entity_ids currently checked

function _byId(id) { return document.getElementById(id); }

function _renderTestResult(ok, msg) {
  const el = _byId("ha-test-result");
  if (!el) return;
  el.textContent = msg || (ok ? "Connected" : "Error");
  el.style.color = ok ? "#4caf50" : "#e57373";
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
  for (const g of groups) {
    const details = document.createElement("details");
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
      txt.textContent = ` ${e.entity_id} — ${e.state}` +
                        (e.friendly_name ? `  (${e.friendly_name})` : "");
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
    try {
      await _saveCurrentSettings({ entities: Array.from(_haSelectedSet).sort() });
      _toast(`Saved ${_haSelectedSet.size} entities`);
    } catch (e) {
      _toast(`Save failed: ${e.message}`, false);
    }
  });
}

async function refreshHAEntities() {
  const tree = _byId("ha-entities-tree");
  tree.innerHTML = "<em>Loading…</em>";
  const r = await fetch("/api/ha/entities");
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    tree.innerHTML = `<span style="color:#e57373">${body.error || `HTTP ${r.status}`}</span>`;
    return;
  }
  _haEntitiesCache = await r.json();
  _renderEntities(_haEntitiesCache.groups);
}

// Expose to other classic scripts (app.js).
window.bindHASettings = bindHASettings;
window.refreshHAEntities = refreshHAEntities;
})();