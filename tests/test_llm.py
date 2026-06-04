"""Tests for the lightweight Ollama LLM client."""

import json

import pytest

from reachy_claw.llm import (
    DEFAULT_SYSTEM_PROMPT,
    OllamaClient,
    OllamaConfig,
    _extract_emotion,
    _StreamingBracketStripper,
)


# ── Emotion extraction ──────────────────────────────────────────────


class TestExtractEmotion:
    def test_happy_prefix(self):
        text, emotion = _extract_emotion("[happy] Hello!")
        assert emotion == "happy"
        assert "Hello!" in text
        assert "[happy]" not in text

    def test_sad_prefix(self):
        text, emotion = _extract_emotion("[sad] I'm sorry")
        assert emotion == "sad"

    def test_no_emotion(self):
        text, emotion = _extract_emotion("Just a normal sentence")
        assert emotion is None
        assert text == "Just a normal sentence"

    def test_unknown_emotion_ignored(self):
        text, emotion = _extract_emotion("[blorp] weird")
        assert emotion is None

    def test_inline_emotion(self):
        text, emotion = _extract_emotion("Hey [excited] wow!")
        assert emotion == "excited"
        assert "[excited]" not in text

    def test_thinking(self):
        text, emotion = _extract_emotion("[thinking] Let me consider...")
        assert emotion == "thinking"

    def test_chinese_text(self):
        text, emotion = _extract_emotion("[curious] \u4f60\u597d\uff01\u4f60\u662f\u8c01\uff1f")
        assert emotion == "curious"
        assert "\u4f60\u597d" in text


# ── OllamaConfig ────────────────────────────────────────────────────


class TestOllamaConfig:
    def test_defaults(self):
        cfg = OllamaConfig()
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.model == "qwen3.5:0.8b"
        assert cfg.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert cfg.max_history == 0

    def test_custom(self):
        cfg = OllamaConfig(model="qwen3.5:2b", max_history=5)
        assert cfg.model == "qwen3.5:2b"
        assert cfg.max_history == 5


# ── OllamaClient callbacks ──────────────────────────────────────────


class TestOllamaClientCallbacks:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        cfg = OllamaConfig()
        client = OllamaClient(cfg)
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_send_state_change_noop(self):
        """send_state_change should be a no-op (no server)."""
        cfg = OllamaConfig()
        client = OllamaClient(cfg)
        await client.connect()
        await client.send_state_change("listening")  # should not raise
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_send_robot_result_noop(self):
        """send_robot_result should be a no-op (no tool calling)."""
        cfg = OllamaConfig()
        client = OllamaClient(cfg)
        await client.connect()
        await client.send_robot_result("cmd-1", {"status": "ok"})
        await client.disconnect()


# ── Streaming with mock server ──────────────────────────────────────


class TestOllamaStreaming:
    @pytest.mark.asyncio
    async def test_stream_with_emotion(self, monkeypatch):
        """Simulate Ollama streaming response with emotion tag."""
        events = []

        # Mock httpx streaming
        chunks = [
            {"message": {"content": "[happy]"}, "done": False},
            {"message": {"content": " Hello"}, "done": False},
            {"message": {"content": " there!"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]

        class FakeResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for c in chunks:
                    yield json.dumps(c)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeClient:
            def stream(self, method, url, json=None):
                return FakeResponse()

        cfg = OllamaConfig()
        client = OllamaClient(cfg)
        client._http = FakeClient()
        client._connected = True

        cb = client.callbacks
        cb.on_stream_start = lambda rid: events.append(("start", rid))
        cb.on_stream_delta = lambda text, rid: events.append(("delta", text))
        cb.on_stream_end = lambda text, rid: events.append(("end", text))
        cb.on_emotion = lambda e: events.append(("emotion", e))

        await client._stream_chat("hi")

        # Should have: start, emotion, delta(s), end
        types = [e[0] for e in events]
        assert "start" in types
        assert "emotion" in types
        assert "end" in types

        # Emotion should be "happy"
        emotion_events = [e for e in events if e[0] == "emotion"]
        assert emotion_events[0][1] == "happy"

        # End text should not contain emotion tag
        end_events = [e for e in events if e[0] == "end"]
        assert "[happy]" not in end_events[0][1]

    @pytest.mark.asyncio
    async def test_stream_without_emotion(self, monkeypatch):
        """Response without emotion tag should still work."""
        events = []

        chunks = [
            {"message": {"content": "Sure, I can help "}, "done": False},
            {"message": {"content": "with that."}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]

        class FakeResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for c in chunks:
                    yield json.dumps(c)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeClient:
            def stream(self, method, url, json=None):
                return FakeResponse()

        cfg = OllamaConfig()
        client = OllamaClient(cfg)
        client._http = FakeClient()
        client._connected = True

        cb = client.callbacks
        cb.on_stream_start = lambda rid: events.append(("start", rid))
        cb.on_stream_delta = lambda text, rid: events.append(("delta", text))
        cb.on_stream_end = lambda text, rid: events.append(("end", text))
        cb.on_emotion = lambda e: events.append(("emotion", e))

        await client._stream_chat("help me")

        types = [e[0] for e in events]
        assert "emotion" not in types
        assert "end" in types

    @pytest.mark.asyncio
    async def test_history_management(self):
        """With max_history > 0, history should accumulate."""
        chunks = [
            {"message": {"content": "[neutral] OK"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]

        class FakeResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for c in chunks:
                    yield json.dumps(c)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeClient:
            def stream(self, method, url, json=None):
                return FakeResponse()

        cfg = OllamaConfig(max_history=2)
        client = OllamaClient(cfg)
        client._http = FakeClient()
        client._connected = True
        client.callbacks.on_stream_start = lambda rid: None
        client.callbacks.on_stream_end = lambda text, rid: None

        await client._stream_chat("hello")
        assert len(client._history) == 2  # user + assistant

        await client._stream_chat("again")
        assert len(client._history) == 4  # 2 turns

        await client._stream_chat("third")
        assert len(client._history) == 4  # trimmed to max_history=2 turns


# ── Streaming bracket-tag stripper ──────────────────────────────────


class TestStreamingBracketStripper:
    """Strip [emotion] and [Faces: ...] tags from a streamed token feed.

    The LLM often splits a tag across token deltas (e.g. "[", "happy", "]"),
    so a per-token regex misses it and the raw tag leaks into TTS. The
    stripper buffers across feeds until a tag closes.
    """

    def _drain(self, stripper, tokens):
        """Feed all tokens then flush; return the concatenated output."""
        out = "".join(stripper.feed(t) for t in tokens)
        return out + stripper.flush()

    def test_emotion_tag_single_feed(self):
        s = _StreamingBracketStripper()
        assert self._drain(s, ["[happy] Hello"]) == "Hello"

    def test_emotion_tag_split_across_feeds(self):
        s = _StreamingBracketStripper()
        assert self._drain(s, ["[", "happy", "] hi"]) == "hi"

    def test_faces_tag_split_across_feeds(self):
        s = _StreamingBracketStripper()
        assert self._drain(s, ["[Fac", "es: Alice", "] hello"]) == "hello"

    def test_trailing_space_eaten_across_feed_boundary(self):
        # Tag closes exactly at end of a feed; the leading space of the next
        # feed must be eaten so "[Faces: X]" + " hello" → "hello".
        s = _StreamingBracketStripper()
        assert self._drain(s, ["[Faces: X]", " hello"]) == "hello"

    def test_unclosed_bracket_is_preserved_via_flush(self):
        # A bare "[" with no closing "]" is held, not dropped; flush() emits it.
        s = _StreamingBracketStripper()
        assert self._drain(s, ["hi [incomplete"]) == "hi [incomplete"

    def test_non_tag_bracket_is_kept(self):
        # Bracketed text with spaces isn't a tag — leave it untouched.
        s = _StreamingBracketStripper()
        assert self._drain(s, ["[hello world]"]) == "[hello world]"

    def test_plain_text_passthrough(self):
        s = _StreamingBracketStripper()
        assert self._drain(s, ["just ", "plain ", "text"]) == "just plain text"

    def test_tag_then_text_then_tag(self):
        s = _StreamingBracketStripper()
        assert self._drain(s, ["[happy] hi ", "[Faces: Bob] there"]) == "hi there"
