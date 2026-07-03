"""§9 Ollama adapter: structured tool history + num_ctx.

Covers _translate_history (canonical Anthropic-shape history → ollama native
tool_calls / role:"tool" linkage) and the presence of options.num_ctx sized
from model_specs.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atn.providers.ollama import (
    OllamaProvider,
    _num_ctx_for,
    _translate_history,
)


# ---------------------------------------------------------------------------
# _translate_history — structured tool round-trip
# ---------------------------------------------------------------------------

class TestTranslateHistory:
    def test_plain_text_user_message(self):
        out = _translate_history([{"role": "user", "content": "hello"}])
        assert out == [{"role": "user", "content": "hello"}]

    def test_plain_text_block_list(self):
        out = _translate_history([
            {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
        ])
        assert out == [{"role": "user", "content": "hi there"}]

    def test_assistant_tool_use_becomes_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "t1", "name": "bash",
                 "input": {"command": "ls"}},
            ],
        }
        out = _translate_history([msg])
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "let me check"
        assert out[0]["tool_calls"] == [
            {"function": {"name": "bash", "arguments": {"command": "ls"}}},
        ]

    def test_tool_result_becomes_tool_message_with_name(self):
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash",
                     "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": "file1\nfile2"},
                ],
            },
        ]
        out = _translate_history(history)
        # assistant tool_call, then a role:"tool" message
        assert out[0]["role"] == "assistant"
        assert out[1] == {
            "role": "tool",
            "content": "file1\nfile2",
            "tool_name": "bash",
        }

    def test_multiple_tool_results_become_separate_tool_messages(self):
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "a", "name": "read_file",
                     "input": {"path": "/x"}},
                    {"type": "tool_use", "id": "b", "name": "bash",
                     "input": {"command": "pwd"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "X"},
                    {"type": "tool_result", "tool_use_id": "b", "content": "/home"},
                ],
            },
        ]
        out = _translate_history(history)
        assert out[0]["tool_calls"] == [
            {"function": {"name": "read_file", "arguments": {"path": "/x"}}},
            {"function": {"name": "bash", "arguments": {"command": "pwd"}}},
        ]
        assert out[1] == {"role": "tool", "content": "X", "tool_name": "read_file"}
        assert out[2] == {"role": "tool", "content": "/home", "tool_name": "bash"}

    def test_two_turn_conversation_preserves_linkage(self):
        """Round-trip: a 2-turn tool conversation keeps the tool_use →
        tool_result linkage all the way through translation."""
        history = [
            {"role": "user", "content": "list files then read one"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "listing"},
                    {"type": "tool_use", "id": "c1", "name": "bash",
                     "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1",
                     "content": "a.txt"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c2", "name": "read_file",
                     "input": {"path": "a.txt"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c2",
                     "content": "contents"},
                ],
            },
        ]
        out = _translate_history(history)
        roles = [m["role"] for m in out]
        assert roles == ["user", "assistant", "tool", "assistant", "tool"]
        # First turn linkage
        assert out[1]["tool_calls"][0]["function"]["name"] == "bash"
        assert out[2]["tool_name"] == "bash"
        assert out[2]["content"] == "a.txt"
        # Second turn linkage (past turn 1 — the regression this fixes)
        assert out[3]["tool_calls"][0]["function"]["name"] == "read_file"
        assert out[4]["tool_name"] == "read_file"
        assert out[4]["content"] == "contents"


# ---------------------------------------------------------------------------
# num_ctx sizing
# ---------------------------------------------------------------------------

class TestNumCtx:
    def test_known_local_model(self):
        assert _num_ctx_for("qwen3.5:4b") == 16384

    def test_unknown_local_model_defaults_16384(self):
        assert _num_ctx_for("llama3.3:70b") == 16384

    def test_never_returns_ollama_default_4096(self):
        assert _num_ctx_for("anything") >= 16384


# ---------------------------------------------------------------------------
# Request body — num_ctx present, structured history used
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRequestBody:
    async def test_send_sets_num_ctx_and_translates(self, monkeypatch):
        captured: dict = {}

        async def fake_send(self, body):
            captured["body"] = body
            return {"message": {"content": "ok"}, "done_reason": "stop"}

        monkeypatch.setattr(OllamaProvider, "_send_with_retry", fake_send)

        prov = OllamaProvider(default_model="qwen3.5:4b")
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash",
                     "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "out"},
                ],
            },
        ]
        await prov.send(messages=history, model="qwen3.5:4b", max_tokens=256)

        body = captured["body"]
        assert body["options"]["num_ctx"] == 16384
        # The tool linkage survived into the request body.
        msgs = body["messages"]
        assert any(m.get("tool_calls") for m in msgs)
        assert any(m["role"] == "tool" and m.get("tool_name") == "bash" for m in msgs)
