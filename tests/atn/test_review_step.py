"""v3 review step in the agentic loop (docs/tool_substrate.md, 2026-07-08).

When a cognitive run used registered tools (use_tool / register_tool)
and never called attest_tools, the base orchestrate loop injects ONE
closing review turn before finalizing — the harness-enforced beat that
produces the per-axis review signal routing tool discovery. Covers:
injection, one-shot behavior, compliance path, self-gating, opt-out.
"""

from __future__ import annotations

from typing import Any

import pytest

from atn.providers.base import Provider, ProviderResponse, ToolCall, Usage


class ScriptedProvider(Provider):
    """send_stream() plays back a scripted list (same pattern as
    tests/atn/test_loop_hardening.py)."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.stream_calls = 0
        self.seen_messages: list[list[dict]] = []
        self._active_model = "test-model"

    @property
    def name(self) -> str:
        return "scripted"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        return ProviderResponse(text="SUMMARY", stop_reason="end_turn")

    async def send_stream(self, *, messages, system="", model="",
                          max_tokens=1024, tools=None, temperature=0.0,
                          on_chunk=None, on_thinking=None) -> ProviderResponse:
        self.stream_calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(**kw) -> ProviderResponse:
    kw.setdefault("usage", Usage(input_tokens=5, output_tokens=2))
    return ProviderResponse(**kw)


def _tool_turn(tool: str, args: dict | None = None) -> ProviderResponse:
    return _resp(
        text="",
        stop_reason="tool_use",
        tool_calls=[ToolCall(id=f"id-{tool}", name=tool,
                             input=args or {"x": 1})],
    )


async def _noop_executor(name, inp):
    return {"ok": True}


def _injected_review(messages_seen: list[list[dict]]) -> bool:
    return any(
        isinstance(m.get("content"), str) and "attest_tools" in m["content"]
        and "Work item closing" in m["content"]
        for msgs in messages_seen for m in msgs
    )


class TestReviewInjection:
    @pytest.mark.asyncio
    async def test_injects_once_when_registered_tools_unreviewed(self):
        p = ScriptedProvider([
            _tool_turn("use_tool", {"name": "summarize_csv"}),
            _resp(text="done", stop_reason="end_turn"),
            # Agent ignores the review prompt → one-shot, finalize anyway.
            _resp(text="still done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor)
        assert p.stream_calls == 3
        assert resp.stop_reason == "end_turn"
        assert resp.text == "still done"
        assert _injected_review(p.seen_messages)
        # The agent's pre-review answer stayed in history.
        assert any(m.get("content") == "done"
                   for m in p.seen_messages[2] if m["role"] == "assistant")

    @pytest.mark.asyncio
    async def test_agent_attests_on_review_turn(self):
        p = ScriptedProvider([
            _tool_turn("use_tool", {"name": "summarize_csv"}),
            _resp(text="done", stop_reason="end_turn"),
            _tool_turn("attest_tools", {
                "judgments": [{"tool": "summarize_csv", "ok": True,
                               "axes": {"correctness": 0.8}}],
                "context": "csv work"}),
            _resp(text="reviewed and done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor)
        assert p.stream_calls == 4
        assert resp.text == "reviewed and done"

    @pytest.mark.asyncio
    async def test_register_tool_does_not_trigger(self):
        """Authoring a tool must NOT prompt a review of it.

        Registration never invokes the tool, so any review it produced
        would be ungrounded — and the close discards it regardless,
        because the author's own household is excluded from position
        drift. The trigger was spending model tokens on rows consensus
        throws away. (This test previously asserted the opposite.)
        """
        p = ScriptedProvider([
            _tool_turn("register_tool", {"name": "new_tool", "code": "..."}),
            _resp(text="registered", stop_reason="end_turn"),
        ])
        await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor)
        assert p.stream_calls == 2
        assert not _injected_review(p.seen_messages)

    @pytest.mark.asyncio
    async def test_register_then_use_still_triggers(self):
        """Registering AND using in one run still earns a review — the
        trigger is the use, not the registration."""
        p = ScriptedProvider([
            _tool_turn("register_tool", {"name": "new_tool", "code": "..."}),
            _tool_turn("use_tool", {"tool": "new_tool"}),
            _resp(text="used it", stop_reason="end_turn"),
            _resp(text="ok", stop_reason="end_turn"),
        ])
        await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor)
        assert _injected_review(p.seen_messages)


class TestReviewSelfGating:
    @pytest.mark.asyncio
    async def test_no_injection_without_registered_tool_usage(self):
        p = ScriptedProvider([
            _tool_turn("get_snapshot"),
            _resp(text="done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor)
        assert p.stream_calls == 2
        assert resp.text == "done"
        assert not _injected_review(p.seen_messages)

    @pytest.mark.asyncio
    async def test_no_injection_when_already_attested(self):
        p = ScriptedProvider([
            _tool_turn("use_tool", {"name": "summarize_csv"}),
            _tool_turn("attest_tools", {
                "judgments": [{"tool": "summarize_csv", "ok": True}],
                "context": "work"}),
            _resp(text="done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor)
        assert p.stream_calls == 3
        assert resp.text == "done"
        assert not _injected_review(p.seen_messages)

    @pytest.mark.asyncio
    async def test_opt_out_flag(self):
        p = ScriptedProvider([
            _tool_turn("use_tool", {"name": "summarize_csv"}),
            _resp(text="done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor,
            review_tools=False)
        assert p.stream_calls == 2
        assert resp.text == "done"
        assert not _injected_review(p.seen_messages)


class TestReinvokePredicate:
    """The caller-side path for providers whose loop can't inject (the
    SDK bridge): delegate_prompts.needs_review_reinvoke."""

    class _BridgeLike:
        handles_review_step = False

    class _GenericLike:
        handles_review_step = True

    def _calls(self, *names):
        return [{"tool": n, "args": {}, "result": {}, "success": True}
                for n in names]

    def test_bridge_needs_reinvoke(self):
        from atn.delegate_prompts import needs_review_reinvoke
        assert needs_review_reinvoke(
            self._BridgeLike(), self._calls("use_tool"), "end_turn")

    def test_generic_provider_handles_it_itself(self):
        from atn.delegate_prompts import needs_review_reinvoke
        assert not needs_review_reinvoke(
            self._GenericLike(), self._calls("use_tool"), "end_turn")

    def test_attested_run_skips(self):
        from atn.delegate_prompts import needs_review_reinvoke
        assert not needs_review_reinvoke(
            self._BridgeLike(),
            self._calls("use_tool", "attest_tools"), "end_turn")

    def test_no_registered_usage_skips(self):
        from atn.delegate_prompts import needs_review_reinvoke
        assert not needs_review_reinvoke(
            self._BridgeLike(), self._calls("get_snapshot"), "end_turn")

    def test_abnormal_stop_reasons_skip(self):
        from atn.delegate_prompts import needs_review_reinvoke
        for reason in ("interrupted", "budget_exceeded", "context_overflow",
                       "provider_error", "loop_detected"):
            assert not needs_review_reinvoke(
                self._BridgeLike(), self._calls("use_tool"), reason), reason

    def test_register_tool_does_not_trigger_and_missing_attr_defaults_safe(self):
        from atn.delegate_prompts import needs_review_reinvoke
        # Authoring a tool is not using it: registration never invokes the
        # tool, and the close discards an author's review of their own work
        # anyway (author household excluded from drift). Previously this
        # asserted the opposite.
        assert not needs_review_reinvoke(
            self._BridgeLike(), self._calls("register_tool"), "end_turn")
        # Using one still does.
        assert needs_review_reinvoke(
            self._BridgeLike(), self._calls("use_tool"), "end_turn")
        # An object without the attribute is treated as self-handling
        # (never double-prompt by default).
        assert not needs_review_reinvoke(
            object(), self._calls("use_tool"), "end_turn")
