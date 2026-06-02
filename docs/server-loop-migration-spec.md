# reachy-claw → CompanionRobotApp + Server-Loop 迁移设计 spec

> ⚠️ **方向已修正(2026-06-02 FINAL):改走 route A —— CLIENT-LOOP,不是 server-loop。**
> 迁移 `ReachyClawApp` → 继承 `CompanionRobotApp`,但**照 `apps/voice_arm` 范式**:
> `llm_backend: edge_llm` + `tools_enabled: true` + 一个插件 `setup()` 把工具注册进
> `app.tool_registry`(照 `ArmPlugin` / `ovs_agent.tools.action_tools.register_arm_tools`);
> ovs_agent `tools/runner.py` 在 **agent 侧**跑 LLM↔工具循环;**引擎保持 pass-through**。
> **不切 server-loop 镜像、不开 flag、引擎不动。** 原因:ovs_agent 自带 client-loop 工具能力
> (`runner.py`),reachy 自己也有(`llm.py:_tool_call_loop`);server-loop 会强加"引擎 edge-llm 必须
> 支持 tools"的依赖,而验证(2026-06-02)发现部署的 Qwen3-4B-AWQ 无视 tools。client-loop 绕开该坑、
> 保留 gateway/token 事件。工具 LLM 仍需 agent 侧支持,但 voice_arm 证明 `edge_llm` 经 overlay
> patch 0006 可 tool-call → reachy 的 edge-llm 端点需带该 overlay(部署核查项,非阻塞)。
>
> 下文以 server-loop 视角写的部分(§2 tool advertise、§7 Phase 0 server-loop 原语、deploy 的
> OVS_V2V_SERVER_LOOP)**作废**;但去重处置表(§1 DELETE 项)、dashboard add_tab、ElevenLabs、
> 分层原则**仍适用**(与 loop 模式无关)。工具机制以 voice_arm client-loop 范式为准。
> server-loop 留作未来优化(引擎 edge-llm tool-calling 在设备上核实后再说;voxedge 闭环本身已证实真实)。
>
> ---
> *(以下为原 route B 设计,部分作废,见上方修正)* 路线 B:迁移 `ReachyClawApp` → 继承 `CompanionRobotApp`(ovs_agent),
> 切 server-loop —— V2V 引擎独占 ASR→LLM(+tools)→TTS,reachy 只 advertise 工具 + 本地执行
> `SERVER_TOOL_CALL`。核心约束:**被 server-loop 砍掉的能力必须设计如何补回来,并明确分层
> (BASE = ovs_agent/引擎,robot-agnostic;APP = clawd-reachy-mini,robot-specific)。**
>
> 跨两仓库:BASE = `/Users/harvest/project/seeed-local-voice`(`agent/ovs_agent` + `server/`);
> APP = `/Users/harvest/project/clawd-reachy-mini`(`src/reachy_claw`)。
> 关联背景见 memory `project-v2v-server-loop-vs-reachy`。

---

### 1. 逐模块处置表

| File | Disposition | Layer | Anchors / Notes |
|---|---|---|---|
| `__init__.py` | ADAPT | APP | 保留最小导出 |
| `app.py` | ADAPT | APP | 基类换 `CompanionRobotApp`;保留 Reachy daemon/db 接线(`app.py:21-41`)、插件注册(`351-363`) |
| `audio.py` | DELETE | BASE 重复 | 本地麦克风管线归 server-loop |
| `backend_registry.py` | DELETE | BASE 重复 | STT/TTS/VAD 注册(`config.py:39-70`)不再 app 拥有 |
| `config.py` | ADAPT | APP | 留 Reachy/motion/vision/diary/HA/dashboard;加 `server_loop: true`;兼容期后移除 app 侧 LLM/STT/TTS/VAD 执行项(`110-149`) |
| `edge_llm.py` | DELETE / BASE 补 | BASE | app LLM 流式+emotion(`337-387`)由 BASE `EdgeLLMBackend` + 事件 hook 取代 |
| `elevenlabs.py` | PROMOTE-TO-BASE | BASE | 通用 TTS provider → 实现为 server/SLV TTS 后端;app 只选 voice/provider |
| `event_bus.py` | DELETE | BASE 重复 | 用 ovs_agent 事件总线 |
| `gateway.py` | DROP / BASE 后端口 | BASE | OpenClaw 流(`37-51`,`320-400`);若仍需则实现为引擎通用 LLM 后端 |
| `ha_client.py` | KEEP-AS-TOOL | APP | 无状态 HA 客户端(`79-190`)→ advertise 工具 + diary 输入 |
| `healthcheck.py` | ADAPT | APP | 容器健康检查;否则用 base app health |
| `llm.py` | DELETE | BASE 重复 | Ollama/client LLM → 引擎后端 |
| `main.py` | ADAPT | APP | 注册(`284-335`)→ `ReachyClawApp(CompanionRobotApp)` |
| `mode.py` / `modes/*` | DELETE / BASE map | BASE+APP | mode 机制 → BASE;Reachy 人设文本留 APP,经 advertise 下发 |
| `motion/dances.py` | KEEP-AS-TOOL | APP | `dance(name)` 工具(`conversation_plugin.py:1628-1650`) |
| `motion/emotion_mapper.py` | ADAPT | APP | emotion→头/天线映射(`69-215`)留本地 |
| `motion/head_target.py` | ADAPT | APP | Reachy 专属(`1-24`) |
| `motion/head_wobbler.py` | ADAPT(需 BASE 事件) | APP | 映射留本地,由 BASE 句/TTS 事件触发,不再靠本地 token 循环 |
| `plugin.py` | DELETE | BASE 重复 | 自定义协议(`17-57`)→ `ovs_agent.plugin.Plugin` |
| `reachy_app.py` | ADAPT | APP | Reachy daemon 包装(`23-45`) |
| `settings_schema.py` | ADAPT | APP | dashboard 可调项留 APP |
| `skill_loader.py` | KEEP-AS-TOOL / ADAPT | APP | `SKILL.md`(`83-120`)、meta-tool(`282-301`)→ `ToolRegistry` 注册 |
| `storage/*` | ADAPT | APP | diary 持久化(`storage/db.py:58-176`) |
| `stt.py` / `tts.py` / `vad.py` | DELETE(provider 提升) | BASE | 服务端拥有 ASR/TTS/VAD |
| `v2v_client.py` | DELETE | BASE 重复 | → `ovs_agent.slv_client.SLVClient`(`508-571`) |
| `vision/*` | ADAPT | APP | 摄像头/头追踪(`mediapipe_tracker.py:19-94` 等) |
| `plugins/conversation_plugin.py` | SPLIT | APP+BASE delete | 删本地 ASR/LLM/TTS 循环(`637-1220`);留工具 handler(`1556-1871`);叙述 helper 由 BASE 事件驱动 |
| `plugins/motion_plugin.py` | ADAPT | APP | 转 Plugin;留预设(`40-59`)、循环(`135-142`)、头追踪(`193-267`)、天线动画(`269-311`) |
| `plugins/face_tracker_plugin.py` | ADAPT | APP | 转协议;留摄像头检查(`61-109`) |
| `plugins/vision_client_plugin.py` | ADAPT / KEEP-AS-TOOL | APP | ZMQ 视觉(`126-187`);暴露 `capture_image`/`vision_status`/`describe_scene` |
| `plugins/daily_log_plugin.py` | ADAPT | APP | 订阅 BASE assistant/user 事件(原 `41-48`) |
| `plugins/dashboard_plugin.py` | SPLIT | BASE+APP | BASE tab API + APP tabs:diary(`397-415`)/prompt save(`482-554`)/motor(`590-616`)/HA(`1754-1810`)/restart+captures(`1149-1569`) |
| `plugins/rest_plugin.py` | ADAPT | APP | rest 调度 + ZMQ 视觉暂停(`54-153`) |
| `plugins/housekeeping_tasks.py` | ADAPT | APP | diary 生成/发布(`30-68`) |
| `plugins/dashboard_static/*` | ADAPT | APP tabs | BASE tab API 后重打包为贡献式 tab |
| `assets/silero_vad.onnx` | DELETE | — | 本地 VAD 资产不再需要 |

### 2. 能力补回来 & 分层矩阵(本 spec 核心)

| 风险能力 | naive server-loop 下丢什么 | 补法 | 层 | 需新 BASE 原语? |
|---|---|---|---|---|
| 自有 LLM / gateway 后端 | `gateway.py` OpenClaw 流 + 工具派发消失 | **已定:语音链路不用 gateway。** 引擎 edge-llm 当对话大脑 + advertise 工具;`gateway.py` 从语音路径 DROP(flag 门控)。**砍掉 BASE LLM 后端选择器。** | APP 工具 | 否 |
| ~~token 级动作驱动~~ **(codex 误判,已纠正)** | 无 —— `_on_stream_delta`(`conversation_plugin.py:1175-1198`)只转发 token 给引擎 TTS,**不驱动动作** | 无需补 | — | **否**(原"token 事件流"原语已砍) |
| 头部跟随 | 无 —— 视觉驱动(`face_tracker_plugin.py:262` → `app.head_targets` bus),与对话/LLM 无耦合 | 不动,整条留 APP 本地视觉环路 | APP | 否 |
| emotion 叠加 | client 侧 emotion 抽取(`edge_llm.py:256-387`)消失 | reachy 已有服务端 emotion 通道 `_on_emotion`(`conversation_plugin.py:1305-1310`)+ `play_emotion` 工具(`1621-1626`)。server-loop 下:LLM 调 `play_emotion` 工具(零新增)**或**引擎广播它已 strip 的 emotion tag(a3b6d71)→ 复用 `_on_emotion` | APP 消费侧已存在 / BASE 仅需 emotion 事件投递 | 可选(优先用 `play_emotion` 工具,免新原语) |
| HA 控制/传感 | LLM 不再调本地 HA | advertise 工具 `ha_*`(`ha_client.py:79-190`) | APP | 否 |
| Skills | Ollama skill loader 绑 app LLM | `SKILL.md`→`ToolRegistry`;`skill_load` 动态注册+重 advertise | APP(registry 已 BASE) | 也许:显式重 advertise API |
| ElevenLabs 语音 | app 侧 `elevenlabs.py`/`tts.py` 不跑 | **已定:本期要 → 先实现为引擎/SLV TTS provider(云 TTS 接入),再删 app 侧。** Reachy 只选 provider/voice。**Phase 1 前新增一项 BASE 任务。** | BASE provider / APP 选择 | **是**:SLV TTS provider 选择器 |
| Diary | 旧 daily log 监听本地 `asr_final`/`llm_end`/`emotion`/`vision_faces`(`41-48`) | BASE 转发 user/assistant final+工具+emotion+TTS;APP 持久化 | BASE 事件转发 / APP diary | **是**:user/assistant final 事件对齐 |
| Dashboard 面板 | 1812 行 Reachy dashboard 不适配;base 无 tab API | BASE `add_tab(...)`;Reachy 贡献 Motor/Vision/Diary/HA tab | BASE 扩展 API / APP tabs | **是**:`debug_dashboard.py` 无 add_tab |

### 3. 工具 advertise 设计

链路:Reachy 注册工具进 `self.tool_registry`(类型提示生成 OpenAI schema,`tools/registry.py:154-201`,advertise schema 带 preamble/completion/response_mode,`222-257`)→ `BaseApp._advertise_tools_if_server_loop()` 在 `server_loop` 时发 `list_advertise_tools()`+人设+LLM 参数(`app_base.py:1331-1383`)→ 引擎 wire 进 EdgeLLMBackend,匹配发 `SERVER_TOOL_CALL` → `SLVClient` 解析(`slv_client.py:714-725`)→ `BaseApp._handle_server_tool_call()` 经 `registry.dispatch()` 执行并回 `CLIENT_TOOL_RESULT`(`app_base.py:1470-1515`)。

Reachy advertise 工具(全 APP):`move_head`(`conversation_plugin.py:1590-1603`)、`move_antennas`(`1605-1619`)、`play_emotion`(`1621-1626`)、`dance`(`1628-1650`)、`capture_image`(`1652-1677`)、`set_volume`(`1679-1760`)、`robot_status`(`1762-1779`)、`stop/resume_conversation`(`1781-1796`)、`sensecraft_*`(`1800-1871`)、`ha_*`(`ha_client.py:79-190`)、`skill_load`+动态技能工具。

### 4. 人设 / system prompt 下发
主路径:`tool_advertise.system_prompt`(`slv_client.py:525-532`);`BaseApp._resolve_chat_system_prompt()`(`app_base.py:1320-1329`)纳入 advertise(`1367-1383`)。兜底:`OVS_V2V_SYSTEM_PROMPT`(`server/main.py:2807-2822`,部署级默认)。**dashboard 运行时改 prompt(保 b576300 修复):** 把 `dashboard_plugin.py:518-546` 的热应用分支改为"更新 config.system_prompt + 调重 advertise",下一轮生效无需重启。BASE 拥有传输+重 advertise;APP 拥有人设文本+编辑 UI。

### 5. Dashboard 原语(Phase 0 BASE)
`agent/ovs_agent/plugins/debug_dashboard.py` 经 rg 确认**无** `add_tab`/`contribute_tab`,固定路由+静态资产(`97-155`)、事件中继(`1450-1585`)。需新增 `add_tab(name,label,html_url,static_dir,ws_topics,routes)`;APP tab 挂 `/plugin-tabs/<name>/`,API 挂 `/api/<name>/`。README 已提议(`apps/companion_robot/README.md:83-99`)。

### 6. 部署/配置 delta
引擎 compose `deploy/jetson/voice/docker-compose.yml`:镜像(line 13)→ `server-loop-prod-v6-asrlock`;加 env `OVS_V2V_SERVER_LOOP=1`、`EDGE_LLM_BASE_URL=http://127.0.0.1:11435/v1`、`EDGE_LLM_MODEL`、`OVS_V2V_SYSTEM_PROMPT`(仅兜底)、可选 temperature/max_tokens。
reachy compose `deploy/jetson/reachy/docker-compose.yml`(`45-63`):加 `OVS_AGENT_SERVER_LOOP=1`。
app 配置 `deploy/jetson/reachy-claw.jetson.yaml`:`llm.backend: edge_llm_v2v` 留兼容注释;加 `server_loop: true`;`v2v.url`(`13-19`)指向 `ws://localhost:8621/v2v/stream` 不变。

### 7. 分阶段
- **Phase 0 — BASE 原语(seeed-local-voice)** *(2026-06-02 决策后精简版)*:
  ① dashboard `add_tab` API;
  ② emotion 事件投递(引擎广播已 strip 的 emotion tag → 客户端 `_on_emotion`)—— **可选**,若优先用 `play_emotion` 工具则不阻塞 Phase 1;
  ③ diary 用的 user/assistant final 事件对齐(`tts_started` 句级 + asr_final 已有,补 assistant final);
  ④ **ElevenLabs 引擎/SLV TTS provider**(本期要,新增);
  ⑤ 显式重 advertise API(skill_load 后)。
  ~~引擎 LLM 后端选择器~~(已砍 —— 语音链路不用 gateway);~~token 级事件流~~(已砍 —— 动作非 token 驱动)。
  验收:server-loop ON 无工具也能答;dashboard 显示一个示例贡献 tab;EL provider 出声;日志见 advertise + 重 advertise。
- **Phase 1 — 最小可用 reachy server-loop(APP):** `ReachyClawApp(CompanionRobotApp)`;删客户端循环代码(`v2v_client/conversation_plugin:637-1220/audio/stt/tts/vad/backend_registry/event_bus/mode`);注册 `MotionPlugin` + `play_emotion`/`move_head`;人设经 advertise;头部摆动/emotion 由 BASE 句/TTS 事件驱动。验收:说话→引擎 LLM 答+引擎 TTS;LLM 调 `play_emotion`/`move_head`;dashboard 改 prompt 下轮生效无需重启。
- **Phase 2 — 产品能力工具化(APP):** 移植 vision_client + face_tracker(ZMQ+head bus);HA 工具;skill_loader→registry+重 advertise;补 `dance/capture_image/sensecraft_*/set_volume/robot_status`;rest 暂停。验收:LLM 调 HA/vision/status;视觉驱动 head bus;rest 暂停/恢复干净。
- **Phase 3 — Dashboard tabs + Diary(APP):** 拆 dashboard_plugin 为 Motor/Vision/HA/Diary 贡献 tab;DailyLog 接 BASE 事件名;diary 生成发布留 housekeeping。验收:diary 含 user/assistant 文本+emotion+vision+HA;tab 工作无需 fork base dashboard。

### 8. 风险 / 待人决策
1. ~~**OpenClaw gateway 范围**~~ **【已定 2026-06-02】语音链路不用 gateway** → feature flag 后删 app gateway 循环,不做 BASE 后端选择器。
2. ~~**token vs 句级动作粒度**~~ **【已定/纠正】伪命题** —— 动作非 token 驱动:头部跟随是视觉环路(不碰对话);emotion 用 `play_emotion` 工具 + 引擎 emotion 事件。无需 token 事件流。
3. ~~**ElevenLabs SKU**~~ **【已定 2026-06-02】本期要** → Phase 1 前先把 EL 做成引擎/SLV TTS provider,再删 app `tts.py`/`elevenlabs.py`。
4. **Dashboard 鉴权**:base dashboard 无鉴权仅 loopback(`debug_dashboard.py:157-164`);Reachy 产品要鉴权/外网吗?影响扩展 base vs 独立 app server。
5. **动态 skill 重 advertise 契约**:`_readvertise_*` 已有重连重广播,但 `skill_load` 后"工具集变更"需稳定 public API(`trigger_readvertise()`?)。
6. **`ovs_agent.Plugin` 生命周期兼容**:Reachy `Plugin.setup()` 返回 bool,ovs_agent 签名可能不同(`apps/companion_robot/README.md:70-72`),执行体移植前确认。

---

## 9. Phase 0 架构勘查结论(2026-06-02,已核验)

**⚠️ BASE 层其实是 3 个仓库**,不止 ovs_agent + server/:
- **`voxedge`**(server-loop 引擎本体)源码在 `/Users/harvest/project/voxedge/voxedge`(editable 安装,`server/main.py:2671` `from voxedge.engine.conversation import ConversationEngine`)。

**✅ server-loop 工具闭环已在 voxedge 实现(迁移核心成立):**
- `voxedge/engine/conversation.py:474-475` 收 `CLIENT_TOOL_ADVERTISE` → `_handle_tool_advertise`(`480-519`)注册客户端工具
- `voxedge/engine/tool_registry.py:339-354` LLM 选中 → 发 `{"type":"tool_call",call_id,name,arguments}`,await 对应 `tool_result`
- `conversation.py:455` 收 `CLIENT_TOOL_RESULT` → 回填 future;`496-501` server-loop off 时 no-op(向后兼容)

**① TTS / ElevenLabs(seeed-local-voice):**
- backend ABC `server/core/tts_backend.py:31`;能力枚举 `:21`;注册表 `_TTS_REGISTRY` `:145`(加 `"cloud.elevenlabs"` 于 `:153`);工厂 `create_tts_backend()` `:195`
- 选择是 **profile 驱动**(`current_profile()["tts_backend"]` `:168`),非 `TTS_BACKEND=` env;config frame **无** per-connection backend 字段 → 需新建 profile JSON `"tts_backend":"cloud.elevenlabs"`
- 必实现方法:`name`/`capabilities`/`sample_rate`/`is_ready`/`preload`/`synthesize`/`generate_streaming`(`tts_backend.py:43-118`)
- **阻抗不匹配(必须做适配):** 引擎流式契约 = 首帧 4 字节 SR(uint32 LE)+ 裸 int16 PCM 分块(`server/main.py:3971`/`4025`,agent 解析 `slv_client.py:663`);现 `reachy_claw/elevenlabs.py` 输出 mp3_44100(`:23`)且返回整包(`:97`)→ 要改用 EL 流式端点 + MP3/WAV→mono int16 PCM 边解边吐

**② Dashboard `add_tab()`(ovs_agent,aiohttp):**
- `debug_dashboard.py`:`web.Application()` `:97`,静态 `/static/` `:152`,浏览器 WS `/ws` `:100`,广播 `:1245`;tab strip 是固定 inline HTML(`static/dashboard.html:54`)
- 注入点:状态 `__init__:39`;`add_tab()` 方法 `:73`;贡献静态挂 `/plugin-tabs/<name>/`、API 挂 `/api/<name>/`(均在 `:156` 前);snapshot 加 `tabs[]` `:301`;动态 topic relay `:192`
- **⚠️ aiohttp 路由在 `AppRunner.setup()`(`:156`)冻结** → 贡献 tab 必须在各插件 `setup()` 阶段收集,dashboard 在 `start()` 的 `:156` 之前 mount(插件生命周期 `__init__→setup→start` 保证可行)
- 前端要改:`dashboard.html:54` tab strip 改为按 snapshot `tabs[]` 动态生成;`dashboard.js:1025` snapshot handler 创建 tab 按钮 + panel

**③ 事件清单 → Phase 0 几乎不用新增事件:**
- 已有:`asr_partial/asr_final/tts_started(含句文本)/tts_sentence_done/tts_done/vad_event/error/tool_call`
- **`emotion` 事件不存在** → 用 `play_emotion` **工具**(已确认 tool_call 闭环可用)→ **零新增事件**
- `assistant_final` 不存在 → diary 可从 `tts_started` 句文本重建,或 Phase 3 再加(非 Phase 0 阻塞)

**→ Phase 0 收敛为 2 个真任务:① ElevenLabs 引擎 TTS backend(+MP3→PCM 适配);② dashboard `add_tab()` API(+前端动态 tab)。** 二者独立,可并行。emotion 走工具、assistant_final 延后,均不阻塞。
