# reachy_voice Client-Loop Migration — Phase 1 Implementation Spec
_2026-06-15 · branch `feat/reachy-voice-clientloop` (off master) · test target: orin-nx 100.82.225.102_

Goal: bring master's Wave-1 `src/reachy_voice/` (pure conversation, `tools_enabled=False`)
up to full CLIENT-LOOP tool-calling, mirroring the proven legacy impl in
`legacy/reachy_claw/clientloop/`. ovs library (`~/project/seeed-local-voice/agent`) is
feature-complete; no lib changes needed for Phase 1.

## 0. Reuse principle (MUST FOLLOW) — thin shell on a complete ovs agent
ovs_agent IS the voice agent. reachy_voice must NOT reimplement anything ovs provides.
Verified ownership:
- **ovs (reuse as-is)**: conversation loop, ASR/TTS/VAD, `tool_registry` + `dispatch` +
  client-loop `runner.stream_with_tools`, and `CompanionRobotApp` scaffolding which already exposes
  `app.reachy`, `app.head_target_bus`, `app.current_emotion`, `app.motor_enabled`
  (`companion_robot/app.py:51-61`). Mirror the canonical companion tool shape in
  `ovs_agent/apps/companion_robot/demo_tools.py` (mock `move_head`/`play_emotion`) — replace the mock
  bodies with real SDK→compositor calls.
- **reachy_voice (thin, app-specific ONLY)**: the 4 real motion tool BODIES (SDK→compositor glue) and
  the bracket/`[Faces:]` tag stripping — confirmed ovs has NO tag-stripping (grep empty), so this is
  not duplication. Use `app.current_emotion` / `app.head_target_bus` slots rather than inventing parallel state.

## 1. Tool inventory (4 tools, ported from legacy/reachy_claw/clientloop/tools_plugin.py)
- `move_head(yaw, pitch, roll=0.0)` — legacy `tools_plugin.py:95-101` → SDK `goto_target(head=...)`.
  NEW: route through compositor `MotionController.command_head(...)` (motion.py:447-453), NOT direct SDK.
- `move_antennas(left, right)` — legacy `tools_plugin.py:103-117` → `goto_target(antennas=[right,left])`.
  NEW: `MotionController.command_antennas(...)` (motion.py:454-456). Mind SDK order `[right,left]`.
- `play_emotion(emotion)` — already exists: `MotionController.play_emotion` (motion.py:216-228).
- `dance(dance_name)` — NEW `MotionController.play_dance(name)` queued through compositor like
  `_play_presence()` (motion.py:373-393); dance names in `DANCES_DATASET` (motion.py:42-43).
- NO HA / vision / skill tools in Phase 1 (matches legacy tools_plugin.py:148-150).

## 2. Plugin design
- New `src/reachy_voice/plugins/motion_tools.py` → `ReachyMotionToolsPlugin(ovs_agent.plugin.Plugin)`,
  pattern = `ovs_agent/plugins/actuator_actions.py` ArmPlugin.
- `setup()` captures `motion: MotionController` (+ optional dashboard hub), registers the 4 tools via
  `self.app.tool_registry.tool(...)` (registry.py:154-201), with `preamble_text` per legacy.
- Wire in `ConversationEngine.start()` after motion is created (conversation.py:224-232), register the
  plugin BEFORE `_app.run()` (conversation.py:271), near the dashboard plugin reg (conversation.py:264-266).
  `main.py` does NOT instantiate plugins directly.

## 3. Config / wiring
- `conversation.py build_ovs_config()`: flip `tools_enabled=True`; add `tools_default_allowlist=[]`,
  `tools_max_iterations=3` (legacy config.yaml:36-43). Keep `server_loop` false → client-loop auto-selected
  (app_mode.py:95-118, stream_with_tools at app_mode.py:337-357). ovs Config already has these fields
  (config.py:293-336). Only touch `reachy_voice/config.py` if YAML/env control of tool settings is wanted
  (load_config drops unknown keys, config.py:121-127).

## 4. Vision-tag stripper port
- Source: `legacy/reachy_claw/llm.py:34-92` (`_RESPONSE_STRIP_RE`, `_StreamingBracketStripper`).
- LAYER into existing `_TtsTagFilter` (conversation.py:36, wraps slv.send_text at :233-237): keep emotion
  tags firing `_on_emotion` (:63-67), additionally DROP `[Faces: ...]` silently.
- Also fix `DashboardPlugin._strip` (dashboard.py:28, :133-142) to drop `[Faces: ...]`.

## 5. Risks / gotchas
- **SDK ownership**: reachy handle already passed in (main.py:124-125 → conversation.py:208-210). Do NOT
  double-connect (legacy created its own at motion_plugin.py:83-147).
- **Compositor contention (KEY)**: MotionController is single-writer (motion.py:145-148). Tool bodies must
  go through compositor commands, NOT blocking SDK `goto_target` off-loop, or they fight presence/gaze/emotion.
- **Tool timeout**: long dances can exceed default 10s dispatch timeout (registry.py:294-307) → set
  `timeout_s` in `registry.tool(...)`. Preamble TTS runs during dispatch (app_mode.py:257-278) — don't block loop.
- **Audio**: app forces `no_media` (main.py:96-99) + DuplexAudioIO (conversation.py:211-219). Don't open USB audio elsewhere.
- **Vision**: only prompt-context `[Faces:]` today (conversation.py:111-118); no capture_image — VLM not 1:1.

## 6. Test plan (orin-nx 100.82.225.102)
- Unit (tests/voice/): registry contains 4 names (cf proof_clientloop.py:63-66); dispatch with fake
  MotionController (proof_clientloop.py:128-181); tag-filter for split `[Faces: Alice]` / `[happy]`.
- On-device smoke: deploy, restart ONLY reachy-voice. Confirm `tools_enabled=True` in logs. Speak
  "look left" / "raise your antennas" / "be happy" / "do a dance". Grep `tool_call_started`/`tool_call_completed`
  (app_mode.py:245-256,298-327). Verify `/debug/motion` (main.py:243-257) + robot moves + speech continues
  (dashboard llm_delta/llm_end). `[Faces:]`/emotion tags never spoken.

## Ordered checklist
1. `src/reachy_voice/plugins/__init__.py`
2. `src/reachy_voice/plugins/motion_tools.py` (ReachyMotionToolsPlugin)
3. motion.py: add `command_head` (compositor-safe)
4. motion.py: add `command_antennas`
5. motion.py: add `play_dance`
6. conversation.py: `tools_enabled=True` + allowlist + max_iterations
7. conversation.py: register ReachyMotionToolsPlugin before `_app.run()`
8. conversation.py: extend `_TtsTagFilter` to drop `[Faces: ...]`
9. dashboard.py: strip `[Faces: ...]` in `_strip`
10. config.py: only if YAML/env tool control needed
11-13. tests/voice/: test_motion_tools_plugin, test_response_tag_stripper, test_clientloop_tool_config
14. `uv run pytest tests/voice -v`
15. `uv run ruff check .`
16-20. deploy orin-nx → restart reachy-voice only → voice smoke → grep tool_call events → verify motion+speech
