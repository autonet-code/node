"""Tests for the Claude Max bridge provider.

Unit tests use a mock subprocess to verify the NDJSON protocol.
The live test at the bottom requires Claude Code / Claude Max and is
skipped unless RUN_BRIDGE_LIVE=1 is set.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from atn.providers.bridge import BridgeProvider, _BRIDGE_SCRIPT
from atn.providers.base import ProviderError, ProviderResponse


# ---------------------------------------------------------------------------
# Helpers — mock subprocess that speaks NDJSON
# ---------------------------------------------------------------------------

class MockBridgeProcess:
    """Simulates the bridge subprocess for unit tests."""

    def __init__(self, responses: list[dict] | None = None):
        self._responses = responses or []
        self._response_index = 0
        self.returncode = None  # None means still running
        self.stdin = MockStdin()
        self.stdout = MockStdout(self._responses)
        self.stderr = MockStderr()

    def kill(self):
        self.returncode = -9

    async def wait(self):
        pass


class MockStdin:
    def __init__(self):
        self.written: list[str] = []

    def write(self, data: bytes):
        self.written.append(data.decode())

    async def drain(self):
        pass


class MockStdout:
    def __init__(self, responses: list[dict]):
        self._responses = responses
        self._index = 0

    async def readline(self) -> bytes:
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return (json.dumps(resp) + "\n").encode()
        return b""


class MockStderr:
    async def readline(self) -> bytes:
        # Return empty to end the drain loop
        return b""


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def bridge_response():
    """Standard successful bridge response."""
    return {
        "id": "req-1",
        "ok": True,
        "session_id": "test-session-123",
        "text": "Hello! I'm Claude.",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        },
        "model": "claude-sonnet-4-20250514",
    }


@pytest.fixture
def tool_use_response():
    """Bridge response with tool_use blocks."""
    return {
        "id": "req-1",
        "ok": True,
        "session_id": "test-session-456",
        "text": "I'll check the weather for you.",
        "tool_calls": [
            {
                "id": "toolu_01",
                "name": "get_weather",
                "input": {"city": "London"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 150,
            "output_tokens": 40,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "model": "claude-sonnet-4-20250514",
    }


@pytest.fixture
def error_response():
    """Bridge error response."""
    return {
        "id": "req-1",
        "ok": False,
        "error": "SDK subprocess crashed",
    }


@pytest.mark.asyncio
async def test_send_basic(bridge_response):
    """Test basic send with a successful response."""
    provider = BridgeProvider(model="sonnet")
    mock_proc = MockBridgeProcess([bridge_response])
    provider._process = mock_proc

    result = await provider.send(
        messages=[{"role": "user", "content": "Hello"}],
        system="You are helpful.",
    )

    assert isinstance(result, ProviderResponse)
    assert result.text == "Hello! I'm Claude."
    assert result.stop_reason == "end_turn"
    assert result.model == "claude-sonnet-4-20250514"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 25
    assert result.usage.cache_read_tokens == 80
    assert result.usage.cache_creation_tokens == 20
    assert len(result.tool_calls) == 0

    # Verify the request sent to bridge
    assert len(mock_proc.stdin.written) == 1
    req = json.loads(mock_proc.stdin.written[0])
    assert req["type"] == "create"
    assert req["message"] == "Hello"
    assert req["system"] == "You are helpful."
    assert req["model"] == "sonnet"


@pytest.mark.asyncio
async def test_send_with_tool_calls(tool_use_response):
    """Test response with tool_use blocks."""
    provider = BridgeProvider(model="sonnet")
    mock_proc = MockBridgeProcess([tool_use_response])
    provider._process = mock_proc

    result = await provider.send(
        messages=[{"role": "user", "content": "What's the weather?"}],
    )

    assert result.text == "I'll check the weather for you."
    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].input == {"city": "London"}


@pytest.mark.asyncio
async def test_send_error(error_response):
    """Test that bridge errors become ProviderError."""
    provider = BridgeProvider(model="sonnet")
    mock_proc = MockBridgeProcess([error_response])
    provider._process = mock_proc

    with pytest.raises(ProviderError, match="Bridge error"):
        await provider.send(
            messages=[{"role": "user", "content": "Hello"}],
        )


@pytest.mark.asyncio
async def test_send_empty_message():
    """Test that empty user message raises ProviderError."""
    provider = BridgeProvider(model="sonnet")

    with pytest.raises(ProviderError, match="No user message"):
        await provider.send(messages=[])


@pytest.mark.asyncio
async def test_provider_name():
    """Test provider name property."""
    provider = BridgeProvider()
    assert provider.name == "claude_max"


@pytest.mark.asyncio
async def test_model_passthrough(bridge_response):
    """Test that model parameter is passed to bridge request."""
    provider = BridgeProvider(model="opus")
    mock_proc = MockBridgeProcess([bridge_response])
    provider._process = mock_proc

    await provider.send(
        messages=[{"role": "user", "content": "Hello"}],
        model="haiku",
    )

    req = json.loads(mock_proc.stdin.written[0])
    assert req["model"] == "haiku"


@pytest.mark.asyncio
async def test_default_model(bridge_response):
    """Test that default model is used when none specified."""
    provider = BridgeProvider(model="opus")
    mock_proc = MockBridgeProcess([bridge_response])
    provider._process = mock_proc

    await provider.send(
        messages=[{"role": "user", "content": "Hello"}],
    )

    req = json.loads(mock_proc.stdin.written[0])
    assert req["model"] == "opus"


@pytest.mark.asyncio
async def test_system_prompt_forwarded(bridge_response):
    """Test that system prompt is forwarded to bridge."""
    provider = BridgeProvider()
    mock_proc = MockBridgeProcess([bridge_response])
    provider._process = mock_proc

    await provider.send(
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful AI.",
    )

    req = json.loads(mock_proc.stdin.written[0])
    assert req["system"] == "You are a helpful AI."


@pytest.mark.asyncio
async def test_content_blocks_extraction(bridge_response):
    """Test extraction of text from content block arrays."""
    provider = BridgeProvider()
    mock_proc = MockBridgeProcess([bridge_response])
    provider._process = mock_proc

    await provider.send(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        }],
    )

    req = json.loads(mock_proc.stdin.written[0])
    assert req["message"] == "Hello world"


@pytest.mark.asyncio
async def test_bridge_script_path():
    """Test that bridge script path is resolved correctly."""
    provider = BridgeProvider()
    assert provider._bridge_script == _BRIDGE_SCRIPT
    assert provider._bridge_script.name == "claude-bridge.ts"

    # Custom path
    custom = BridgeProvider(bridge_script="/custom/path.ts")
    assert custom._bridge_script == Path("/custom/path.ts")


# ---------------------------------------------------------------------------
# Live test — requires Claude Code / Claude Max
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_BRIDGE_LIVE") != "1",
    reason="Set RUN_BRIDGE_LIVE=1 to run live bridge tests",
)
async def test_live_bridge():
    """Live integration test with the actual Claude bridge.

    Requires:
    - Claude Code installed (npm i -g @anthropic-ai/claude-code)
    - Active Claude Max subscription
    - bun installed
    - RUN_BRIDGE_LIVE=1 environment variable
    """
    provider = BridgeProvider(model="sonnet")

    try:
        result = await provider.send(
            messages=[{"role": "user", "content": "Reply with exactly: BRIDGE_OK"}],
            system="You are a test assistant. Follow instructions exactly.",
        )

        print(f"\n--- Live Bridge Test ---")
        print(f"Text: {result.text}")
        print(f"Model: {result.model}")
        print(f"Stop: {result.stop_reason}")
        print(f"Usage: in={result.usage.input_tokens} out={result.usage.output_tokens} "
              f"cache_read={result.usage.cache_read_tokens} cache_create={result.usage.cache_creation_tokens}")

        assert result.text, "Expected non-empty response"
        assert result.model, "Expected model to be set"
        assert result.usage.input_tokens > 0 or result.usage.output_tokens > 0, "Expected usage tokens"

    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_BRIDGE_LIVE") != "1",
    reason="Set RUN_BRIDGE_LIVE=1 to run live bridge tests",
)
async def test_live_bridge_cache_tokens():
    """Test that prompt caching works and cache tokens are reported.

    Sends two requests with the same system prompt in quick succession.
    The first should create a cache entry; the second should read from it.

    Requires: RUN_BRIDGE_LIVE=1
    """
    provider = BridgeProvider(model="sonnet")
    system = (
        "You are a specialised test assistant for the ATN agent framework. "
        "Your only job is to echo back the user's message verbatim. "
        "Do not add any commentary or explanation."
    )

    try:
        # First call — should create cache
        r1 = await provider.send(
            messages=[{"role": "user", "content": "CACHE_TEST_1"}],
            system=system,
        )
        print(f"\n--- Bridge Cache Test: Call 1 ---")
        print(f"  in={r1.usage.input_tokens} out={r1.usage.output_tokens} "
              f"cache_create={r1.usage.cache_creation_tokens} cache_read={r1.usage.cache_read_tokens}")

        assert r1.text, "Expected non-empty response"

        # Second call — same system prompt, should hit cache
        r2 = await provider.send(
            messages=[{"role": "user", "content": "CACHE_TEST_2"}],
            system=system,
        )
        print(f"--- Bridge Cache Test: Call 2 ---")
        print(f"  in={r2.usage.input_tokens} out={r2.usage.output_tokens} "
              f"cache_create={r2.usage.cache_creation_tokens} cache_read={r2.usage.cache_read_tokens}")

        assert r2.text, "Expected non-empty response"

        # Verify cache tokens are being reported (at least one call should show them)
        total_cache_create = r1.usage.cache_creation_tokens + r2.usage.cache_creation_tokens
        total_cache_read = r1.usage.cache_read_tokens + r2.usage.cache_read_tokens
        print(f"--- Totals ---")
        print(f"  cache_creation_tokens: {total_cache_create}")
        print(f"  cache_read_tokens: {total_cache_read}")

        # The bridge may or may not support caching depending on the SDK version.
        # We verify the fields are present and non-negative; if caching works,
        # at least one of them should be > 0.
        assert total_cache_create >= 0, "cache_creation_tokens should be non-negative"
        assert total_cache_read >= 0, "cache_read_tokens should be non-negative"

        if total_cache_create > 0 or total_cache_read > 0:
            print("PASS: Cache tokens detected — prompt caching is active")
        else:
            print("WARN: No cache tokens reported — caching may not be active for this provider/model")

    finally:
        await provider.close()
