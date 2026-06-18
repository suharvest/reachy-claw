/* deferred.js — dashboard controls whose backend isn't built yet.
 *
 * The reachy-voice app (ovs_agent + SLV core) wires a focused set of live
 * controls (language, barge-in, VAD, memory, volume, restart, faces). The rest
 * of the inherited UI — conversation modes, VLM, rest window, LLM switching,
 * voice/clone, motor presets, prompts, diary, Home Assistant — belongs to
 * features not yet ported (the "Line 3" roadmap). Rather than let those controls
 * silently no-op (send a WS message nobody handles) or 404, render them visibly
 * DISABLED: greyed, non-clickable, with a "coming soon" badge/tooltip.
 *
 * When a control's backend lands, delete its entry from the lists below.
 * Spec: docs/superpowers/specs/2026-06-14-dashboard-runtime-overrides-design.md
 */
(function () {
  "use strict";

  var TIP = "暂未支持 / Coming soon"; // 暂未支持 / Coming soon

  // Whole sections — disabled + a "coming soon" badge.
  var CARDS = [
    ".mode-group",   // conversation / monologue / interpreter mode picker
    "#mode-status",
    "#interpreter-settings",
    "#tab-prompt",   // Prompt tab (system prompts — tied to modes)
    "#tab-diary",    // Diary settings tab
    "#tab-ha",       // Home Assistant Sensors tab
    "#page-diary",   // Diary page
  ];

  // Cards reached via an always-present anchor inside them (disable the whole
  // enclosing card, not just the anchor).
  var ANCHOR_CARDS = [
    ["#rest-enabled-toggle", ".restart-section"], // Rest Window
    ["#apply-llm-btn", ".detail-section"],        // LLM backend/model switch
    ["#voice-pitch", ".detail-section"],          // Voice / clone
    ["#motor-toggle", ".detail-section"],         // Motor enable + presets
  ];

  // Small inline controls — disabled + greyed, no badge (no room).
  var CONTROLS = [
    "#vlm-toggle",        // Camera Vision (VLM) toggle
    "#energy-threshold",  // Energy threshold slider (VAD threshold IS wired)
  ];

  // Buttons that navigate to a deferred view — disable so the click is inert.
  var NAV = [
    '.settings-tab[data-tab="prompt"]',
    '.settings-tab[data-tab="diary"]',
    '.settings-tab[data-tab="ha"]',
    '.page-tab[data-page="diary"]',
  ];

  function disable(el, badge) {
    if (!el || el.dataset.deferred) return;
    el.dataset.deferred = "1";
    el.classList.add("deferred-control");
    if (badge) el.classList.add("deferred-card");
    el.setAttribute("title", TIP);
    var fields = el.matches("input,select,button,textarea") ? [el] : [];
    el.querySelectorAll("input,select,button,textarea").forEach(function (f) {
      fields.push(f);
    });
    fields.forEach(function (f) { f.disabled = true; });
  }

  function apply() {
    CARDS.forEach(function (s) {
      document.querySelectorAll(s).forEach(function (e) { disable(e, true); });
    });
    ANCHOR_CARDS.forEach(function (pair) {
      var a = document.querySelector(pair[0]);
      if (a) disable(a.closest(pair[1]), true);
    });
    CONTROLS.forEach(function (s) {
      document.querySelectorAll(s).forEach(function (e) {
        disable(e.closest(".volume-row, .toggle-row") || e, false);
      });
    });
    NAV.forEach(function (s) {
      document.querySelectorAll(s).forEach(function (e) { disable(e, false); });
    });
  }

  function injectStyle() {
    var css =
      ".deferred-control{opacity:.45!important;cursor:not-allowed!important;" +
      "pointer-events:none!important;}" +
      ".deferred-control *{pointer-events:none!important;}" +
      ".deferred-card{position:relative;}" +
      ".deferred-card::after{content:\"\\6682\\672A\\652F\\6301\";" + // 暂未支持
      "position:absolute;top:6px;right:8px;z-index:2;font-size:9px;" +
      "letter-spacing:.05em;font-weight:600;opacity:.6;padding:1px 6px;" +
      "border:1px solid currentColor;border-radius:7px;white-space:nowrap;}";
    var st = document.createElement("style");
    st.textContent = css;
    document.head.appendChild(st);
  }

  function init() { injectStyle(); apply(); }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
