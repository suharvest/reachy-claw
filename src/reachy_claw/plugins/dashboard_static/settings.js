// src/reachy_claw/plugins/dashboard_static/settings.js
"use strict";

const SECTIONS = [];

function registerSection(section) { SECTIONS.push(section); }

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

// --- Section: Rest Window ---
registerSection({
  id: "rest-window",
  title: "休整时段 / Rest Window",
  async render(div) {
    const cur = await fetchSettings("rest");
    div.innerHTML = `
      <label>开始 <input type="time" data-k="window_start" value="${cur.window_start}"></label>
      <label>结束 <input type="time" data-k="window_end" value="${cur.window_end}"></label>
      <label>时区 <input type="text" data-k="timezone" value="${cur.timezone}"></label>
      <label><input type="checkbox" data-k="enabled" ${cur.enabled ? "checked" : ""}> 启用</label>
      <button class="btn-save">保存</button>
      <span class="msg"></span>
    `;
    div.querySelector(".btn-save").addEventListener("click", async () => {
      const body = {};
      div.querySelectorAll("[data-k]").forEach(el => {
        const k = el.dataset.k;
        body[k] = el.type === "checkbox" ? el.checked : el.value;
      });
      const msg = div.querySelector(".msg");
      try {
        await putSettings("rest", body);
        msg.textContent = "✓ 已保存";
        msg.className = "msg ok";
      } catch (e) {
        msg.textContent = "✗ " + e.message;
        msg.className = "msg err";
      }
    });
  },
});

// --- Section: Diary Publishing ---
registerSection({
  id: "diary-publishing",
  title: "日记发布 / Diary Publishing",
  async render(div) {
    const cur = await fetchSettings("diary");
    div.innerHTML = `
      <label><input type="checkbox" data-k="auto_publish" ${cur.auto_publish ? "checked" : ""}> 自动每日发布</label>
      <label><input type="checkbox" data-k="privacy_linter" ${cur.privacy_linter ? "checked" : ""}> 隐私 linter</label>
      <label>站点 repo <input type="text" data-k="site_repo_url" value="${cur.site_repo_url}" placeholder="git@github.com:org/site.git"></label>
      <label>路径 <input type="text" data-k="site_diary_path" value="${cur.site_diary_path}"></label>
      <label>分支 <input type="text" data-k="site_branch" value="${cur.site_branch}"></label>
      <button class="btn-save">保存</button>
      <span class="msg"></span>
      <h4>历史</h4>
      <table class="diary-history"><tbody></tbody></table>
    `;
    div.querySelector(".btn-save").addEventListener("click", async () => {
      const body = {};
      div.querySelectorAll("[data-k]").forEach(el => {
        const k = el.dataset.k;
        body[k] = el.type === "checkbox" ? el.checked : el.value;
      });
      const msg = div.querySelector(".msg");
      try {
        await putSettings("diary", body);
        msg.textContent = "✓ 已保存";
        msg.className = "msg ok";
      } catch (e) {
        msg.textContent = "✗ " + e.message;
        msg.className = "msg err";
      }
    });
    await renderDiaryHistory(div.querySelector(".diary-history tbody"));
  },
});

async function renderDiaryHistory(tbody) {
  const r = await fetch("/api/diary/status");
  const { dates } = await r.json();
  tbody.innerHTML = "";
  for (const d of dates) {
    const tr = document.createElement("tr");
    const status = d.published ? "✓" : (d.generated ? "⚠" : "✗");
    const action = d.published
      ? { label: "重新生成", op: () => trigger("generate", d.date, true) }
      : (d.generated
          ? { label: "发布", op: () => trigger("publish", d.date, false) }
          : { label: "生成+发布", op: async () => { await trigger("generate", d.date, false); await trigger("publish", d.date, false); } });
    tr.innerHTML = `<td>${d.date}</td><td>${status}</td><td><button>${action.label}</button></td>`;
    tr.querySelector("button").addEventListener("click", async () => {
      tr.querySelector("button").disabled = true;
      try { await action.op(); } catch (e) { alert(e.message); }
      tr.querySelector("button").disabled = false;
      // Refresh row.
      await new Promise(r => setTimeout(r, 800));
      await renderDiaryHistory(tbody);
    });
    tbody.appendChild(tr);
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

window.renderSettings = function(root) {
  root.innerHTML = "<h2>SETTINGS</h2>";
  for (const s of SECTIONS) {
    const wrap = document.createElement("section");
    wrap.className = "settings-section";
    wrap.innerHTML = `<h3>${s.title}</h3><div class="content"></div>`;
    root.appendChild(wrap);
    s.render(wrap.querySelector(".content")).catch(e => {
      wrap.querySelector(".content").innerHTML = `<span class="msg err">加载失败：${e.message}</span>`;
    });
  }
};
