# Edge LLM + V2V Integration Spec

## 1. File-by-file change list

### `/Users/harvest/project/clawd-reachy-mini/src/reachy_claw/edge_llm.py` (new)

```python
@dataclass
class EdgeLLMConfig:
    """Configuration for the OpenAI-compatible TensorRT-Edge-LLM chat service."""
    base_url: str = "http://localhost:8080"
    model: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_history: int = 3
    max_tokens: int = 80
    prefix_cache: bool = True

class EdgeLLMClient:
    """Streaming LLM client with the same callback contract as DesktopRobotClient."""

    def __init__(self, config: EdgeLLMConfig): ...
    @property
    def is_connected(self) -> bool: ...
    async def connect(self) -> None: ...
    async def warmup_session(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def send_message_streaming(self, text: str) -> None: ...
    async def send_interrupt(self) -> None: ...
    async def send_state_change(self, state: str) -> None: ...
    async def send_robot_result(self, command_id: str, result: dict) -> None: ...
    async def _stream_chat(self, user_text: str) -> None: ...
    def _build_messages(self, text: str, image_b64: str | None = None) -> list[dict]: ...
```

### `/Users/harvest/project/clawd-reachy-mini/src/reachy_claw/v2v_client.py` (new)

```python
@dataclass
class V2VConfig:
    """Configuration for the unified ASR/TTS/VAD WebSocket service."""
    url: str = "ws://localhost:8621/v2v/stream"
    sample_rate: int = 16000
    asr_language: str = "auto"
    tts_language: str = "auto"
    vad: str = "silero"
    vad_silence_ms: int = 700
    tts_voice: str | None = None
    tts_speed: float | None = None
    multi_utterance: bool = True

class V2VClient:
    """Bidirectional V2V WebSocket client for mic PCM in and ASR/TTS events out."""

    def __init__(self, config: V2VConfig): ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...
    async def send_text_delta(self, text: str) -> None: ...
    async def flush_tts(self) -> None: ...
    async def abort(self) -> None: ...
```

### `/Users/harvest/project/clawd-reachy-mini/src/reachy_claw/plugins/conversation_plugin.py` (modify)

- `conversation_plugin.py:24-33` — imports only local STT/TTS/VAD and gateway/Ollama clients -> also import `EdgeLLMClient`, `EdgeLLMConfig`, `V2VClient`, and `V2VConfig`.
- `conversation_plugin.py:70` — `_client` accepts `DesktopRobotClient | OllamaClient | None` -> include `EdgeLLMClient`.
- `conversation_plugin.py:71-74` — local `_stt`, `_tts`, `_vad`, `_audio` are always built -> when `llm_backend == "edge_llm_v2v"`, keep `AudioCapture` for mic chunks but do not create local STT, TTS, or VAD backends.
- `conversation_plugin.py:123-132` — Phase 1 always creates STT/TTS/VAD/AudioCapture -> branch so V2V creates only audio capture plus a V2V client; ASR, VAD, and TTS are delegated to the V2V service.
- `conversation_plugin.py:154` — `_init_gateway()` currently branches only `standalone_mode`, `ollama`, and gateway. Add a first-class `config.llm_backend == "edge_llm_v2v"` branch that constructs `EdgeLLMClient`, connects it, constructs/connects `V2VClient`, wires both callback sets, and raises on V2V connect failure.
- `conversation_plugin.py:160-185` — Ollama branch remains unchanged.
- `conversation_plugin.py:186-191` — gateway fallback remains unchanged and must not mask `edge_llm_v2v` connection errors.
- `conversation_plugin.py:217-219` — `_init_tts()` always warms local TTS -> skip this task for `edge_llm_v2v`.
- `conversation_plugin.py:237-243` — startup `gather()` always includes `_init_stt()`, `_init_vad()`, `_init_gateway()`, `_init_tts()`, `_init_robot()` -> omit `_init_stt`, `_init_vad`, and `_init_tts` for `edge_llm_v2v`.
- `conversation_plugin.py:268-273` — always starts `_audio_loop`, `_sentence_accumulator`, `_tts_worker`, `_output_pipeline` -> for `edge_llm_v2v`, start new `_v2v_audio_loop`, `_v2v_event_loop`, and `_edge_llm_stream_bridge` only. The old `_audio_loop`, `_sentence_accumulator`, and `_tts_worker` must not start. `_output_pipeline` also should not start unless binary V2V audio is adapted into `_audio_queue`; the preferred design plays V2V audio directly.
- `conversation_plugin.py:291-313` — `stop()` disconnects audio/client/TTS -> also disconnect V2V after stopping audio and before client disconnect.
- `conversation_plugin.py:430-442` — `_setup_callbacks()` wires LLM stream callbacks -> keep for `EdgeLLMClient`; add `_setup_v2v_callbacks()` for ASR/TTS/VAD/binary audio.
- `conversation_plugin.py:452` — exact stream delta callback is `async def _on_stream_delta(self, text: str, run_id: str) -> None`; Edge LLM SSE deltas must call this shape.
- `conversation_plugin.py:470` — `_on_stream_end(full_text, run_id)` currently queues `None` for local sentence splitting -> under V2V, send `tts_flush` instead of feeding `_sentence_accumulator`.
- `conversation_plugin.py:1133` — `_audio_loop()` owns local VAD/STT/barge-in -> do not run for V2V; replace with `_v2v_audio_loop()` that forwards PCM chunks to V2V.
- `conversation_plugin.py:1488-1585` — `_process_and_send()` path remains the LLM entry point; V2V ASR final handlers should call `_process_and_send(text)`.
- `conversation_plugin.py:1625` — `_sentence_accumulator()` splits LLM text locally -> do not run; V2V `SentenceBuffer` at `app/core/v2v.py:92-210` owns sentence buffering.
- `conversation_plugin.py:1701` — `_tts_worker()` synthesizes local TTS -> do not run.
- `conversation_plugin.py:1763` — `_output_pipeline()` plays local TTS queue -> replace or bypass with V2V binary audio playback.
- `conversation_plugin.py:1938-1966` — `_fire_interrupt()` drains queues and calls `send_interrupt()` -> also deduplicate and send V2V `{"type":"abort"}` for active V2V TTS.

### `/Users/harvest/project/clawd-reachy-mini/src/reachy_claw/config.py` (modify)

- `config.py:110-119` — LLM fields currently support `"gateway"` and `"ollama"` -> add `"edge_llm_v2v"` plus Edge LLM/V2V fields in Section 6.
- `config.py:222-310` — YAML mapping lacks Edge LLM/V2V keys -> add `llm.edge_llm_url`, `llm.edge_llm_prefix_cache`, and `v2v.*` mappings.
- `config.py:312-325` — env mapping lacks new variables -> add `CLAWD_EDGE_LLM_URL`, `CLAWD_V2V_URL`, and related fields.

### `/Users/harvest/project/clawd-reachy-mini/src/reachy_claw/llm.py` (modify or leave as reference)

Prefer a new `edge_llm.py`; if colocated, insert after `OllamaClient`. Reuse `StreamCallbacks` from `gateway.py:37-52`, and mirror the history window at `llm.py:249-251` and `llm.py:317-322`.

### `/Users/harvest/project/clawd-reachy-mini/src/reachy_claw/gateway.py` (no behavior change)

Use `StreamCallbacks` as the shared callback contract. `gateway.py:199-207` shows nonblocking `send_message_streaming`; `gateway.py:209-214` shows interrupt semantics; `gateway.py:313-415` is the event-to-callback pattern Edge/V2V should emulate.

## 2. Message/protocol mapping tables

### V2V incoming server frames to ConversationPlugin

| V2V frame | Source lines | clawd handler |
| --- | --- | --- |
| `asr_partial` | `v2v.py:41`, `main.py:1077-1078` | `_handle_v2v_asr_partial(text, is_stable)` emits `asr_partial`; mirrors `conversation_plugin.py:1364`. |
| `asr_endpoint` | `v2v.py:42`, `main.py:1085`, `main.py:1107-1108` | `_handle_v2v_asr_endpoint()` sets `TRANSCRIBING` if useful. |
| `asr_final` | `v2v.py:43`, `main.py:1087-1089`, `main.py:1118-1124` | `_handle_v2v_asr_final(text, session_complete)` calls `_process_and_send(text)`. |
| `tts_started` | `v2v.py:44`, `main.py:1173-1175` | `_handle_v2v_tts_started(sentence)` sets `app.is_speaking=True`, `_set_state(SPEAKING)`. |
| `tts_sentence_done` | `v2v.py:45`, `main.py:1184` | `_handle_v2v_tts_sentence_done(sentence)` records observability only. |
| `tts_done` | `v2v.py:46`, `main.py:1204-1205` | `_handle_v2v_tts_done()` calls `_finish_speaking()` equivalent without local TTS drain. |
| `error` | `v2v.py:47`, `main.py:970-971` | `_handle_v2v_error(error)` logs and resets to IDLE if needed. |
| binary audio | `main.py:1169-1184` | `_handle_v2v_audio(data)` parses first 4-byte sample-rate header, then streams PCM to Reachy/local audio output. |
| `vad_event` `speech_start`/`speech_end` | not currently emitted; VAD is internal at `main.py:995-1014` | proposed `_handle_vad_event(event)`. |

### clawd outgoing V2V frames

| Frame | Purpose |
| --- | --- |
| `{"type":"config", "asr_language":"auto", "tts_language":"auto", "sample_rate":16000, "vad":"silero", "vad_silence_ms":700, "multi_utterance":true}` | Required first frame; see `main.py:843-858` and config parsing at `main.py:860-884`. |
| binary PCM16 chunks | Mic input to V2V ASR/VAD; consumed at `main.py:983-1018`. |
| `{"type":"text","text": delta}` | LLM token/delta input for V2V TTS; consumed at `main.py:1028-1030`. |
| `{"type":"tts_flush"}` | Flush remaining sentence buffer after LLM stream end; consumed at `main.py:1031-1035`. |
| `{"type":"asr_eos"}` | Manual ASR finalization during shutdown or single-shot tests; consumed at `main.py:1036-1037`. |
| `{"type":"abort"}` | Barge-in cancel for active TTS; defined at `v2v.py:36`, consumed at `main.py:1038-1048`. |

### Edge LLM SSE delta to stream callback

`server.py:91-135` mounts the upstream OpenAI-compatible FastAPI app and protects `/v1/chat/completions`; `guard.py:420-433` confirms the guarded path. The request body is OpenAI-style `messages` (`README.md:29-31`) with cache flags (`README.md:38-60`). For streaming, parse SSE lines shaped like:

| SSE field | EdgeLLMClient action |
| --- | --- |
| `choices[0].delta.content` | append to `full_text`; call `callbacks.on_stream_delta(delta_text, run_id)` matching `conversation_plugin.py:452`. |
| `choices[0].finish_reason` or `data: [DONE]` | call `callbacks.on_stream_end(clean_full_text, run_id)` matching `conversation_plugin.py:470`. |
| HTTP/client exception or cancellation | call `callbacks.on_stream_abort(reason, run_id)` matching `conversation_plugin.py:483`. |

Mapping summary: `asr_partial` -> `_handle_v2v_asr_partial`; `asr_final` -> `_handle_v2v_asr_final`; `tts_started` -> `_handle_v2v_tts_started`; `tts_sentence_done` -> `_handle_v2v_tts_sentence_done`; binary audio -> `_handle_v2v_audio`; VAD `speech_start`/`speech_end` -> `_handle_vad_event`.

## 3. Message history management

`EdgeLLMClient.history` should be `list[dict[str, str]]` in OpenAI message format, e.g. `{"role":"user","content":text}` and `{"role":"assistant","content":reply}`. The system prompt is not stored in history; `_build_messages()` creates `[{"role":"system","content":system}, *history_window, {"role":"user","content":text}]` each request.

Mirror Ollama’s sliding window exactly: `llm.py:249-251` extends messages with `self._history[-(self._config.max_history * 2):]`, and `llm.py:317-322` appends user/assistant pairs then truncates to `max_history * 2`. Append the user message and assistant message only after a successful stream end, so aborted turns do not poison context.

For cache behavior, first request sends `save_system_prompt_kv_cache=True` and may include `return_cache_metrics=True` for diagnostics; `README.md:38-45` documents this flag. Subsequent turns send `prefix_cache=True` when `edge_llm_prefix_cache` is enabled; `README.md:51-60` documents the multi-turn cache shape. Do not send both cache modes blindly on every turn. Track `_system_prompt_cache_saved: bool`; first successful request flips it true.

## 4. Concurrent task lifecycle

```
+-----------------------+-------------+-----------------------------------------+
| Task                  | Start order | Responsibility                          |
+-----------------------+-------------+-----------------------------------------+
| v2v_event_loop        | 1           | Read JSON/binary V2V frames and dispatch |
| v2v_audio_loop        | 2           | Read mic chunks and send PCM16 to V2V    |
| edge_llm_stream_chat  | per ASR final | POST SSE chat request to Edge LLM       |
| edge_llm_to_v2v_tts   | per delta   | Forward stream deltas as V2V text frames |
+-----------------------+-------------+-----------------------------------------+
```

Shutdown order: set `_running=False`; stop audio capture; send `asr_eos` if V2V is connected; send `abort` if speaking; drain local queues for compatibility; cancel per-turn Edge LLM task; close V2V WebSocket; close Edge LLM HTTP client. If V2V drops mid-conversation, fire synthetic abort semantics, reset `app.is_speaking`, set IDLE unless already LISTENING, reconnect with exponential backoff for idle drops, and raise/log for startup drops so the orchestrator can select gateway.

## 5. Barge-in flow

1. V2V sends `{"type":"vad_event","event":"speech_start"}`. This is proposed; current V2V only handles speech start internally at `main.py:995-1005`.
2. clawd `_handle_vad_event("speech_start")` drains pending V2V audio/local compatibility queues, sets `_interrupt_event`, clears `_current_run_id`, and stops local playback.
3. clawd sends `{"type":"abort"}` to V2V. V2V already consumes this at `main.py:1038-1048`.
4. Guard with `_v2v_abort_sent_for_run_id` or `_v2v_abort_in_flight`; skip duplicate aborts until the next `tts_started` or LLM run.
5. Emotion/state transition: local state should move `SPEAKING -> LISTENING` immediately on speech_start, because user experience depends on local playback stopping now. Server-side TTS cancellation can complete later and should not move state back to IDLE if VAD/ASR is already listening.

## 6. Configuration additions

Add these fields near `config.py:110-119`:

```python
edge_llm_url: str = "http://localhost:8080"
edge_llm_model: str = ""
edge_llm_prefix_cache: bool = True
edge_llm_max_tokens: int = 80
v2v_url: str = "ws://localhost:8621/v2v/stream"
v2v_asr_language: str = "auto"
v2v_tts_language: str = "auto"
v2v_vad: str = "silero"
v2v_vad_silence_ms: int = 700
v2v_multi_utterance: bool = True
```

YAML keys: `llm.backend=edge_llm_v2v`, `llm.edge_llm_url`, `llm.edge_llm_model`, `llm.edge_llm_prefix_cache`, `llm.edge_llm_max_tokens`, `v2v.url`, `v2v.asr_language`, `v2v.tts_language`, `v2v.vad`, `v2v.vad_silence_ms`, `v2v.multi_utterance`.

Environment variables: `CLAWD_EDGE_LLM_URL`, `CLAWD_EDGE_LLM_MODEL`, `CLAWD_EDGE_LLM_PREFIX_CACHE`, `CLAWD_EDGE_LLM_MAX_TOKENS`, `CLAWD_V2V_URL`, `CLAWD_V2V_ASR_LANGUAGE`, `CLAWD_V2V_TTS_LANGUAGE`, `CLAWD_V2V_VAD`, `CLAWD_V2V_VAD_SILENCE_MS`, `CLAWD_V2V_MULTI_UTTERANCE`.

## 7. Future multimodal extension hook

`EdgeLLMClient.send_message_streaming()` should create a task just like `OllamaClient.send_message_streaming()` at `llm.py:204-209`; inside `_stream_chat()`, call `_build_messages(text)` before posting. Keep all message assembly in:

```python
def _build_messages(self, text: str, image_b64: str | None = None) -> list[dict]: ...
```

When Edge LLM adds VLM support, only `_build_messages()` changes to emit OpenAI multimodal content parts or image fields. The streaming parser, history policy, cache flags, and V2V bridge stay unchanged.

## 8. Test plan

Unit seams: mock `V2VClient` with async methods recording `send_audio`, `send_text_delta`, `flush_tts`, and `abort`; inject JSON/binary events into the plugin handler without opening a WebSocket. Mock `EdgeLLMClient` by manually invoking `on_stream_start`, `on_stream_delta`, and `on_stream_end` and verifying V2V receives `text` then `tts_flush`. Separately test `EdgeLLMClient` with an `httpx.MockTransport` SSE stream and assert history truncation matches `llm.py:317-322`.

Integration smoke: configure `llm_backend="edge_llm_v2v"`, mock V2V to emit `asr_final` for “say hello”, mock Edge LLM to stream “Hello there.”, then mock V2V to emit `tts_started`, binary audio, and `tts_done`. Assert clawd emits `asr_final`, enters THINKING, enters SPEAKING, and returns to IDLE/listening.

## 9. Migration and rollback

Migration is a config switch: `llm_backend = "edge_llm_v2v"` uses Edge LLM plus V2V; `llm_backend = "gateway"` keeps `DesktopRobotClient`; `llm_backend = "ollama"` keeps `OllamaClient`. Rollback is changing the field back and restarting.

Add logs for: selected backend, Edge LLM URL/model, V2V URL/config, first cache mode used, TTFT, first V2V audio byte latency, abort sent/deduped, V2V reconnect start/success/failure, and startup V2V connect failure. Startup fallback policy: if the V2V WebSocket connect fails in `_init_gateway()`, log an error and raise. Do not silently fall back inside the plugin; let the orchestrator choose gateway.

## Open Questions

- **Q1 (RESOLVED)**: V2V `vad_event` frame WILL exist (`{"type":"vad_event","event":"speech_start"|"speech_end"}`) — parallel task adds this in seeed-local-voice. clawd designs barge-in around it directly; no local VAD fallback needed once integrated.
- **Q2 (PINNED: reuse `_output_pipeline`)**: V2V binary audio is parsed by `V2VClient` (strip 4-byte LE uint32 SR header), then dispatched via the existing `_audio_queue` so `_output_pipeline()` plays it. If header SR differs from device output SR, log a warning. Resampling can be added later.
- **Q3 (PINNED: client-generated uuid)**: `EdgeLLMClient` generates `run_id = uuid.uuid4().hex` per `send_message_streaming()` call (mirrors `DesktopRobotClient` at `gateway.py:199-207`). No external run_id source.
- **Q4 (RESOLVED)**: Edge LLM service exposes `GET /v1/models` returning `{"data":[{"id":"...","object":"model"}]}`. When `config.edge_llm_model == ""`, `EdgeLLMClient.connect()` queries this endpoint and uses the first id. Result is cached on the client.
- **Q5 (RESOLVED)**: Only two cache flags are accepted: `save_system_prompt_kv_cache=True` (first request only — track `_system_prompt_cache_saved: bool`); `prefix_cache=True` from turn 2 onward when `edge_llm_prefix_cache` is true. `save_prefix_cache` is NOT sent.
- **Q6 (deferred)**: V2V audio sample-rate handling beyond warning log (e.g. resampling). Out of scope for Wave 1.
- **Q7 (deferred)**: Monologue/interpreter modes for `edge_llm_v2v`. Wave 1+2 cover conversation mode only.

## Error & Observability Notes

- Edge LLM returns structured errors: `{"error":{"code":"...","message":"...","context":{"request_id":"..."}}}` for failures.
- Every Edge LLM HTTP response carries an `X-Request-Id` header. `EdgeLLMClient` logs this header on every request (success or failure) for traceability.
- `EdgeLLMClient` exceptions surface `error.code` and `error.message` from the structured error body.
- V2V `asr_final` carries `session_complete: bool` and `duplicate_of_streamed: bool`. Always connect with `multi_utterance=true` so a single WS handles all turns; only send `{"type":"asr_eos"}` at shutdown.
- `EdgeLLMClient.send_state_change()` and `send_robot_result()` are interface-compatibility stubs (debug log, return). They are not meaningful for Edge LLM but keep `StreamCallbacks` callers polymorphic.

## 8.5 Implementation Waves

The full design lands in three waves to keep blast radius small:

### Wave 1 — Foundational clients + tests (this commit set)

- New `src/reachy_claw/edge_llm.py` (`EdgeLLMConfig`, `EdgeLLMClient`).
- New `src/reachy_claw/v2v_client.py` (`V2VConfig`, `V2VClient`).
- `src/reachy_claw/config.py`: additive fields for `edge_llm_*` and `v2v_*`; `"edge_llm_v2v"` accepted as `llm_backend`; YAML + env mappings.
- `tests/test_edge_llm_client.py`, `tests/test_v2v_client.py`.
- No changes to `conversation_plugin.py`, `gateway.py`, or `llm.py`.
- All existing tests must remain green.

### Wave 2 — ConversationPlugin integration

- Wire `EdgeLLMClient` + `V2VClient` into `conversation_plugin.py` per §1 and §4 above.
- Add `_handle_v2v_*` event handlers and `_v2v_audio_loop`.
- Barge-in via `_handle_vad_event("speech_start")` per §5.
- Skip local STT/TTS/VAD init in V2V branch.
- Plugin-level unit + integration tests.

### Wave 3 — End-to-end smoke on Jetson

- Deploy edge-llm-chat-service + seeed-local-voice + clawd-reachy-mini together.
- Verify TTFT, first-audio-byte latency, barge-in, reconnect.
- Capture cache-hit metrics with `return_cache_metrics=True`.
