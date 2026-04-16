# Mode State Machine Design

## Problem

`ConversationPlugin.switch_mode()` is a 90-line if/elif/else method that manages:
- Boolean flags (`_monologue_mode`, `_interpreter_mode`)
- LLM config mutation (7+ fields on `_client._config`)
- Sub-task lifecycle (monologue timer, interpreter sequencer)
- Global config side effects (barge-in, conversation_mode)

Adding a new mode means adding another elif branch plus scattering mode checks across `_audio_loop`, `_process_and_send`, `switch_backend`, and emotion handling. The flags are mutually exclusive but maintained manually.

## Solution

Extract each mode into a **Mode strategy object** managed by a **ModeManager**. Modes declare their config overrides and implement enter/exit lifecycle hooks. The ConversationPlugin delegates to the current Mode instead of branching on boolean flags.

## Existing Modes

| Mode | Trigger | Core Behavior | Special Resources |
|------|---------|---------------|-------------------|
| **conversation** | User voice → VAD | STT → LLM → TTS, barge-in enabled | OllamaClient with history |
| **monologue** | Auto-timer (configurable interval) | Vision + background speech → LLM monologue → TTS | `_monologue_timer` task, background listening |
| **interpreter** | User voice → VAD | STT → concurrent translation → ordered TTS | `_InterpreterSequencer` (httpx, bypasses OllamaClient) |

## Architecture

### Mode Base Class (`src/reachy_claw/mode.py`)

```python
class Mode(ABC):
    name: str

    # Declarative config overrides (None = use app.config default)
    barge_in: bool | None = None
    temperature: float | None = None
    max_history: int | None = None
    skip_emotion_extraction: bool = False
    enable_vlm: bool | None = None
    system_prompt: str | None = None
    play_emotions: bool = True

    def get_ollama_config(self, app_config) -> dict:
        """Return dict of OllamaConfig field overrides for this mode."""

    async def enter(self, ctx: ModeContext) -> None:
        """Called when mode becomes active. Start sub-tasks here."""

    async def exit(self, ctx: ModeContext) -> None:
        """Called when mode is deactivated. Cleanup sub-tasks here."""

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        """Mode-specific preprocessing of user input before sending to LLM.
        Return None to indicate the mode handles sending itself (e.g., interpreter)."""
        return text

    def on_speaking_audio(self, chunk, ctx: ModeContext) -> str:
        """Behavior when audio arrives during SPEAKING state.
        Returns: "barge_in" | "bg_listen" | "ignore"."""
        return "barge_in" if self.barge_in else "ignore"
```

### ModeContext (`src/reachy_claw/mode.py`)

Facade over ConversationPlugin internals, limiting what Mode objects can access:

```python
@dataclass
class ModeContext:
    app: ReachyClawApp
    sentence_queue: asyncio.Queue
    audio_queue: asyncio.Queue
    events: EventBus
    spawn_task: Callable      # plugin._spawn_task
    get_vision_context: Callable
    capture_frame: Callable
```

### ModeManager (`src/reachy_claw/mode.py`)

```python
class ModeManager:
    def __init__(self, ctx: ModeContext): ...

    def register(self, mode: Mode) -> None: ...

    async def switch(self, name: str) -> None:
        """Exit current mode, restore config, apply new mode config, enter new mode."""
        # 1. current.exit(ctx)
        # 2. Restore config snapshot
        # 3. Snapshot current config
        # 4. new_mode.apply_config(ctx)
        # 5. new_mode.enter(ctx)
        # 6. events.emit("mode_change", {"mode": name, "prev": prev_name})

    @property
    def current(self) -> Mode: ...
```

Config snapshot/restore ensures each mode starts from a clean baseline, preventing config leaks between modes.

### Mode Implementations (`src/reachy_claw/modes/`)

**ConversationMode** (`modes/conversation.py`):
- `barge_in = True`
- All other config overrides are None (use defaults)
- `preprocess_utterance`: injects face recognition context (`[Faces: ...]`)
- `on_speaking_audio`: returns `"barge_in"`
- `enter`/`exit`: no-op

**MonologueMode** (`modes/monologue.py`):
- `barge_in = False`, `temperature = 0.9` (minimum), `play_emotions = False`
- `enter`: spawns `_timer_loop` task, initializes `_last_speech_time`, `_pending_speech`, `_bg_speech_frames`
- `exit`: cancels timer task, clears background listening state
- `preprocess_utterance`: calls `_compose_monologue_prompt(text, ctx)`
- `on_speaking_audio`: returns `"bg_listen"`
- Contains `_timer_loop`, `_compose_monologue_prompt`, `_bg_listen` (moved from conversation_plugin.py)

**InterpreterMode** (`modes/interpreter.py`):
- `barge_in = False`, `temperature = 0.3`, `max_history = 0`, `skip_emotion_extraction = True`, `enable_vlm = False`
- `enter`: creates and starts `_InterpreterSequencer`
- `exit`: stops sequencer, drains sentence_queue and audio_queue
- `preprocess_utterance`: submits text to sequencer, returns `None` (mode handles sending itself — sequencer calls Ollama `/api/chat` directly with concurrent requests and ordered output)
- Contains `_InterpreterSequencer` class (moved from conversation_plugin.py, unchanged)

### ConversationPlugin Changes

**Initialization:**
```python
# Remove: self._monologue_mode, self._interpreter_mode, self._interp_sequencer
# Add: self._mode_manager = ModeManager(ctx) with 3 registered modes
```

**`_audio_loop` SPEAKING state** (replaces 3-branch if/elif):
```python
action = self._mode_manager.current.on_speaking_audio(chunk, ctx)
if action == "bg_listen":
    await self._mode_manager.current.bg_listen(chunk, streaming_stt)
elif action == "ignore":
    continue
# action == "barge_in" → existing barge-in detection logic (unchanged)
```

**LLM send** (replaces interpreter/monologue/conversation branching):
```python
result = self._mode_manager.current.preprocess_utterance(text, ctx)
if result is not None:
    await self._process_and_send(result)
```

**`switch_mode()`** (replaces 90-line method):
```python
def switch_mode(self, mode: str) -> None:
    asyncio.ensure_future(self._mode_manager.switch(mode))
```

**`switch_backend()`** (replaces if/elif config block):
```python
overrides = self._mode_manager.current.get_ollama_config(config)
ollama_cfg = OllamaConfig(**{**defaults, **overrides})
```

**Emotion/thinking guards** (replaces `not self._monologue_mode` checks):
```python
if self.app.config.play_emotions and self._mode_manager.current.play_emotions:
    self.app.emotions.queue_emotion("thinking")
```

### Mode Transition Rules

- All modes can switch to any other mode freely
- Hard constraint: `exit()` must complete before `enter()` of the new mode (sequential, not concurrent)
- Switching to the current mode is a no-op
- On switch: LLM message history is cleared (existing behavior preserved)
- EventBus broadcasts `mode_change` event for other plugins to observe

### What Does NOT Change

- `ConvState` enum and its state machine inside `_audio_loop` — orthogonal to modes
- `_sentence_accumulator`, `_tts_worker`, `_output_pipeline` — shared pipeline, all modes use it
- VAD/STT logic in `_audio_loop` — all modes need audio capture
- EventBus, HeadTargetBus, Plugin base class, app.py
- config.py, llm.py, dashboard_plugin.py

## File Changes

| Action | File | Description |
|--------|------|-------------|
| New | `src/reachy_claw/mode.py` | Mode base class, ModeContext, ModeManager |
| New | `src/reachy_claw/modes/__init__.py` | Re-export ConversationMode, MonologueMode, InterpreterMode |
| New | `src/reachy_claw/modes/conversation.py` | ConversationMode |
| New | `src/reachy_claw/modes/monologue.py` | MonologueMode + timer + bg_listen logic |
| New | `src/reachy_claw/modes/interpreter.py` | InterpreterMode + _InterpreterSequencer |
| Modify | `src/reachy_claw/plugins/conversation_plugin.py` | Remove switch_mode body, boolean flags; delegate to ModeManager |
| New | `tests/test_modes.py` | Mode + ModeManager unit tests |
| Modify | `tests/test_conversation_plugin.py` | Adapt existing tests to new interface |

## Testing Strategy

**Mode unit tests** (`tests/test_modes.py`):
- Each Mode's `get_ollama_config()` returns correct overrides
- `preprocess_utterance` behavior per mode
- `on_speaking_audio` returns correct action string
- MonologueMode enter/exit starts/stops timer
- InterpreterMode enter/exit starts/stops sequencer

**ModeManager unit tests** (same file):
- `switch()` calls exit → enter in correct order
- Config snapshot/restore works correctly
- Duplicate switch is no-op
- `mode_change` event emitted with correct payload

**Integration tests** (modify existing `tests/test_conversation_plugin.py`):
- Existing monologue/interpreter tests pass with Mode objects underneath
- `switch_mode("interpreter")` → sequencer started
- `switch_mode("conversation")` → sequencer stopped, queues drained
