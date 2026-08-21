"""Tests for the generic-loop hardening spec (docs/agentic_loop.md §1–§6, §8).

Covers the send_orchestrate upgrades on atn/providers/base.py:
  §1  loop-level retry gate (transient / overflow / fatal routing)
  §2  two-tier context reduction (prune, then compact) + spiral guard + fallback
  §3  orphan repair on every exit path
  §4  loop detection (exact-repeat threshold 3, oscillation, one warning turn)
  §5  mid-turn steering queue (send_user_message + undelivered surfacing)
  §6  tool-result head+tail truncation (see test_provider_base.py too)
  §8  malformed tool-argument guard ({"raw": ...} -> synthetic error result)

All tests drive Provider.send_orchestrate directly against a mock provider
whose send_stream() is scripted per turn (a response OR an exception to raise).
"""
from __future__ import annotations

from typing import Any

import pytest

from atn.providers.base import (
    ContextOverflowError,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    Usage,
    classify_provider_error,
    _is_malformed_args,
    _is_oscillating,
)


# ---------------------------------------------------------------------------
# Mock provider: each scripted step is a ProviderResponse OR an Exception.
# ---------------------------------------------------------------------------

class ScriptedProvider(Provider):
    """Provider whose send_stream() plays back a scripted list.

    Each element is either a ``ProviderResponse`` (returned) or an
    ``Exception`` instance (raised). ``send()`` (used by the compaction
    summarizer) is scripted separately via ``summary_responses``.
    """

    def __init__(
        self,
        script: list[Any],
        summary_responses: list[Any] | None = None,
    ) -> None:
        self._script = list(script)
        self._summary = list(summary_responses or [])
        self.stream_calls = 0
        self.summary_calls = 0
        self.seen_messages: list[list[dict]] = []
        self._active_model = "test-model"

    @property
    def name(self) -> str:
        return "scripted"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        # Used by the compaction summarizer.
        self.summary_calls += 1
        if self._summary:
            item = self._summary.pop(0)
        else:
            item = ProviderResponse(text="SUMMARY", stop_reason="end_turn")
        if isinstance(item, Exception):
            raise item
        return item

    async def send_stream(self, *, messages, system="", model="", max_tokens=1024,
                          tools=None, temperature=0.0, on_chunk=None,
                          on_thinking=None) -> ProviderResponse:
        self.stream_calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _resp(**kw) -> ProviderResponse:
    kw.setdefault("usage", Usage(input_tokens=5, output_tokens=2))
    return ProviderResponse(**kw)


def _tc(name="t", **inp) -> ToolCall:
    return ToolCall(id=f"id-{name}", name=name, input=inp or {"x": 1})


async def _noop_executor(name, inp):
    return {"ok": True}


# ---------------------------------------------------------------------------
# §1 Error classification
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_overflow_class(self):
        assert classify_provider_error(ContextOverflowError("too big")) == "overflow"

    def test_overflow_message_on_400(self):
        e = ProviderError("prompt is too long: 300000 tokens", status_code=400)
        assert classify_provider_error(e) == "overflow"

    def test_transient_status(self):
        for code in (429, 500, 502, 503, 529):
            assert classify_provider_error(ProviderError("x", status_code=code)) == "transient"

    def test_fatal_400_non_overflow(self):
        assert classify_provider_error(ProviderError("bad request", status_code=400)) == "fatal"

    def test_fatal_auth(self):
        assert classify_provider_error(ProviderError("unauthorized", status_code=401)) == "fatal"

    def test_transient_by_exception_name(self):
        class TimeoutException(Exception):
            pass
        assert classify_provider_error(TimeoutException()) == "transient"


# ---------------------------------------------------------------------------
# §1 Loop-level retry gate
# ---------------------------------------------------------------------------

class TestRetryGate:
    @pytest.mark.asyncio
    async def test_transient_retried_then_succeeds(self, monkeypatch):
        # Avoid real backoff sleeps.
        import atn.providers.base as base
        async def _fast_sleep(_):
            return None
        monkeypatch.setattr(base.asyncio, "sleep", _fast_sleep)

        p = ScriptedProvider([
            ProviderError("overloaded", status_code=503),   # transient, retry
            _resp(text="recovered", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(message="go", tools=[], tool_executor=_noop_executor)
        assert resp.text == "recovered"
        assert p.stream_calls == 2

    @pytest.mark.asyncio
    async def test_transient_exhausts_then_provider_error(self, monkeypatch):
        import atn.providers.base as base
        async def _fast_sleep(_):
            return None
        monkeypatch.setattr(base.asyncio, "sleep", _fast_sleep)

        # 4 transient failures: 1 initial + 3 retries all fail -> abort.
        p = ScriptedProvider([ProviderError("overloaded", status_code=503)] * 4)
        resp = await p.send_orchestrate(message="go", tools=[], tool_executor=_noop_executor)
        assert resp.stop_reason == "provider_error"
        assert p.stream_calls == 4

    @pytest.mark.asyncio
    async def test_fatal_aborts_immediately(self):
        p = ScriptedProvider([ProviderError("invalid api key", status_code=401)])
        resp = await p.send_orchestrate(message="go", tools=[], tool_executor=_noop_executor)
        assert resp.stop_reason == "provider_error"
        assert p.stream_calls == 1
        assert "provider error" in resp.text.lower()


# ---------------------------------------------------------------------------
# §1 / §2 Overflow -> reduction -> re-send
# ---------------------------------------------------------------------------

class TestOverflowReduction:
    @pytest.mark.asyncio
    async def test_overflow_triggers_reduction_then_resend(self):
        # First send overflows; reduction runs (summarizer returns a short
        # summary), re-send succeeds.
        p = ScriptedProvider(
            script=[
                ContextOverflowError("context length exceeded", status_code=400),
                _resp(text="after reduction", stop_reason="end_turn"),
            ],
            summary_responses=[_resp(text="tiny summary", stop_reason="end_turn")],
        )
        # Seed a big history so reduction has something to fold and the
        # post-reduction estimate actually shrinks.
        history = [
            {"role": "user", "content": "old ask " + "a" * 5000},
            {"role": "assistant", "content": "old answer " + "b" * 5000},
        ]
        resp = await p.send_orchestrate(
            message="now do X", tools=[], tool_executor=_noop_executor,
            history=history,
        )
        assert resp.text == "after reduction"
        assert p.stream_calls == 2
        assert p.summary_calls == 1  # compaction summarizer was called

    @pytest.mark.asyncio
    async def test_overflow_after_reduction_aborts(self):
        # Both sends overflow -> abort context_overflow (never a third send).
        p = ScriptedProvider(
            script=[
                ContextOverflowError("context length exceeded", status_code=400),
                ContextOverflowError("context length exceeded", status_code=400),
            ],
            summary_responses=[_resp(text="tiny", stop_reason="end_turn")],
        )
        history = [
            {"role": "user", "content": "x" * 8000},
            {"role": "assistant", "content": "y" * 8000},
        ]
        resp = await p.send_orchestrate(
            message="go", tools=[], tool_executor=_noop_executor, history=history,
        )
        assert resp.stop_reason == "context_overflow"
        assert p.stream_calls == 2


# ---------------------------------------------------------------------------
# §16 Verify step: closing verification turn after code edits
# ---------------------------------------------------------------------------

class TestVerifyStep:
    @pytest.mark.asyncio
    async def test_code_edit_injects_one_verify_turn(self):
        p = ScriptedProvider([
            _resp(text="editing", stop_reason="tool_use",
                  tool_calls=[_tc("edit_file", path="/repo/mod.py",
                                  old_string="a", new_string="b")]),
            _resp(text="done", stop_reason="end_turn"),      # natural end → verify turn
            _resp(text="verified, done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(message="fix", tools=[],
                                        tool_executor=_noop_executor)
        assert resp.text == "verified, done"
        assert p.stream_calls == 3
        last_user = [m for m in p.seen_messages[-1] if m["role"] == "user"][-1]
        assert "mod.py" in str(last_user["content"])

    @pytest.mark.asyncio
    async def test_no_code_edit_no_verify_turn(self):
        p = ScriptedProvider([
            _resp(text="looking", stop_reason="tool_use",
                  tool_calls=[_tc("read_file", path="/repo/mod.py")]),
            _resp(text="done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(message="look", tools=[],
                                        tool_executor=_noop_executor)
        assert resp.text == "done"
        assert p.stream_calls == 2


# ---------------------------------------------------------------------------
# §2 Compaction: retention, fallback, spiral guard
# ---------------------------------------------------------------------------

class TestCompaction:
    @pytest.mark.asyncio
    async def test_summarizer_failure_falls_back_to_hard_truncation(self):
        from atn.providers.base import _hard_truncate
        # A summarizer that raises must NOT no-op: _compact_messages returns
        # a hard-truncated list.
        p = ScriptedProvider(script=[], summary_responses=[RuntimeError("boom")])
        # User turns large enough that older ones fall outside the ~20k user-char
        # retention window, so hard truncation genuinely drops turns.
        messages = [
            {"role": "user", "content": "task " + "a" * 25_000},
            {"role": "assistant", "content": "step " + "b" * 25_000},
            {"role": "user", "content": "more " + "c" * 25_000},
            {"role": "assistant", "content": "done " + "d" * 25_000},
            {"role": "user", "content": "again " + "e" * 25_000},
            {"role": "assistant", "content": "yes " + "f" * 25_000},
        ]
        out = await p._compact_messages(messages, system="", model="m",
                                        original_request="task")
        # Hard truncation: strictly shorter, first message is the truncation
        # header, original request preserved in it.
        assert len(out) < len(messages)
        assert "truncated" in out[0]["content"].lower()
        assert "task" in out[0]["content"]

    @pytest.mark.asyncio
    async def test_spiral_triggers_hard_reset_then_aborts_at_cap(self):
        # Summarizer returns a summary as long as the input -> <20% reduction.
        # Tier 3: the first failures rebuild the stack (hard reset) instead of
        # aborting; once the per-run reset cap is spent, abort as before.
        def build_messages():
            huge = "z" * 60_000
            return [
                {"role": "user", "content": huge},
                {"role": "assistant", "content": huge},
                {"role": "user", "content": huge},
                {"role": "assistant", "content": huge},
            ]
        p = ScriptedProvider(
            script=[],
            summary_responses=[_resp(text="z" * 60_000, stop_reason="end_turn")] * 5,
        )
        messages = build_messages()
        await p._reduce_context(
            messages, system="", model="claude-haiku-4-5",
            original_request="the original task", max_tokens=8192, force=True,
        )
        # Reset happened: single small user message carrying the task.
        assert p._hard_resets_this_run == 1
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "the original task" in messages[0]["content"]
        assert "[CONTEXT RESET]" in messages[0]["content"]
        assert p._compactions_this_run == 0  # fresh budget after reset

        p._hard_resets_this_run = 2  # cap spent
        messages = build_messages()
        with pytest.raises(ContextOverflowError):
            await p._reduce_context(
                messages, system="", model="claude-haiku-4-5",
                original_request="task", max_tokens=8192, force=True,
            )

    @pytest.mark.asyncio
    async def test_max_two_compactions_per_run(self):
        p = ScriptedProvider(
            script=[],
            summary_responses=[_resp(text="s", stop_reason="end_turn")] * 5,
        )
        p._compactions_this_run = 2  # already at the cap
        p._hard_resets_this_run = 2  # tier 3 also spent — abort is the floor
        messages = [
            {"role": "user", "content": "x" * 60_000},
            {"role": "assistant", "content": "y" * 60_000},
        ]
        with pytest.raises(ContextOverflowError):
            await p._reduce_context(
                messages, system="", model="claude-haiku-4-5",
                original_request="task", max_tokens=8192, force=True,
            )

    def test_prune_saves_and_stubs_old_tool_results(self):
        from atn.providers.base import _prune_tool_results
        # Build a history with several old tool_result bodies plus recent turns.
        big = "R" * 30_000
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "t", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": big}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "b", "name": "t", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "b", "content": big}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c", "name": "t", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c", "content": big}]},
            {"role": "assistant", "content": "final"},
        ]
        saved = _prune_tool_results(messages)
        assert saved > 0
        # The oldest result body should be stubbed; the most recent protected.
        oldest = messages[2]["content"][0]["content"]
        assert oldest.startswith("[pruned:")


# ---------------------------------------------------------------------------
# §3 Orphan repair
# ---------------------------------------------------------------------------

class TestOrphanRepair:
    def test_repair_appends_cancelled_results(self):
        from atn.providers.base import _repair_orphan_tool_uses
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "x", "name": "t", "input": {}},
                {"type": "tool_use", "id": "y", "name": "t", "input": {}},
            ]},
        ]
        _repair_orphan_tool_uses(messages, "interrupted")
        assert messages[-1]["role"] == "user"
        blocks = messages[-1]["content"]
        ids = {b["tool_use_id"] for b in blocks}
        assert ids == {"x", "y"}
        assert all(b["is_error"] for b in blocks)
        assert all('"cancelled": true' in b["content"] for b in blocks)

    def test_no_repair_when_results_present(self):
        from atn.providers.base import _repair_orphan_tool_uses
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "x", "name": "t", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
        ]
        before = len(messages)
        _repair_orphan_tool_uses(messages, "abort")
        assert len(messages) == before

    @pytest.mark.asyncio
    async def test_provider_error_abort_repairs_orphans(self):
        # Turn 1 emits a tool call; before results are appended the loop only
        # appends the assistant turn on the NEXT iteration path. Here we abort
        # via a fatal error on the very first send while there is a trailing
        # assistant tool_use in the message stack — simulate by scripting a
        # tool_use response then a fatal error on the follow-up send.
        p = ScriptedProvider([
            _resp(tool_calls=[_tc("t")], stop_reason="tool_use"),
            ProviderError("boom", status_code=400),  # fatal on 2nd send
        ])
        resp = await p.send_orchestrate(
            message="go",
            tools=[{"name": "t", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=_noop_executor,
        )
        assert resp.stop_reason == "provider_error"


# ---------------------------------------------------------------------------
# §4 Loop detection
# ---------------------------------------------------------------------------

class TestLoopDetection:
    def test_is_oscillating_ab_pattern(self):
        fps = ["A", "B", "A", "B", "A", "B"]
        assert _is_oscillating(fps) is True

    def test_is_oscillating_needs_full_window(self):
        assert _is_oscillating(["A", "B", "A", "B"]) is False

    def test_not_oscillating_when_diverse(self):
        assert _is_oscillating(["A", "B", "C", "D", "E", "F"]) is False

    @pytest.mark.asyncio
    async def test_exact_repeat_warns_then_aborts(self):
        # Same call every turn. Default threshold 3 -> after 3 identical the
        # loop issues ONE warning turn, then aborts if it repeats again.
        script = [_resp(tool_calls=[_tc("loop")], stop_reason="tool_use") for _ in range(10)]
        p = ScriptedProvider(script)
        resp = await p.send_orchestrate(
            message="go",
            tools=[{"name": "loop", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=_noop_executor,
            max_turns=20,
        )
        assert resp.stop_reason in ("repeat_call_limit", "loop_detected")
        # A warning turn was injected before the abort — the model got a chance.
        warned = any(
            isinstance(m.get("content"), str) and "looping" in m["content"].lower()
            for msgs in p.seen_messages for m in msgs
        )
        assert warned

    @pytest.mark.asyncio
    async def test_warning_lets_model_self_correct(self):
        # 3 identical -> warning; then the model changes approach and concludes.
        script = [
            _resp(tool_calls=[_tc("loop")], stop_reason="tool_use"),
            _resp(tool_calls=[_tc("loop")], stop_reason="tool_use"),
            _resp(tool_calls=[_tc("loop")], stop_reason="tool_use"),
            _resp(text="ok, concluding", stop_reason="end_turn"),
        ]
        p = ScriptedProvider(script)
        resp = await p.send_orchestrate(
            message="go",
            tools=[{"name": "loop", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=_noop_executor,
            max_turns=20,
        )
        assert resp.stop_reason == "end_turn"
        assert resp.text == "ok, concluding"


# ---------------------------------------------------------------------------
# §5 Mid-turn steering
# ---------------------------------------------------------------------------

class TestSteering:
    @pytest.mark.asyncio
    async def test_send_user_message_false_when_idle(self):
        p = ScriptedProvider([])
        # No orchestration running -> queue is None -> returns False.
        assert await p.send_user_message("hi") is False

    @pytest.mark.asyncio
    async def test_steering_injected_at_iteration_boundary(self):
        # Turn 1 returns a tool call; we enqueue a steering message during the
        # executor, and it should appear as a user turn before turn 2's send.
        p = ScriptedProvider([
            _resp(tool_calls=[_tc("t")], stop_reason="tool_use"),
            _resp(text="done", stop_reason="end_turn"),
        ])

        async def executor(name, inp):
            # Runs while orchestration is active -> queue exists -> True.
            ok = await p.send_user_message("STEER: also check Y")
            assert ok is True
            return {"ok": True}

        await p.send_orchestrate(
            message="go",
            tools=[{"name": "t", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=executor,
            max_turns=5,
        )
        # The second send's messages must contain the injected steering turn.
        injected = any(
            isinstance(m.get("content"), str) and "STEER" in m["content"]
            for m in p.seen_messages[1]
        )
        assert injected

    @pytest.mark.asyncio
    async def test_undrained_steering_surfaced_in_result(self):
        # A steering message enqueued but never drained (run ends on the same
        # turn) must be surfaced as undelivered_steering, not lost.
        p = ScriptedProvider([_resp(text="done", stop_reason="end_turn")])

        # Kick off an orchestration but pre-seed the queue via a wrapper: we
        # enqueue after the queue is created by patching send to enqueue.
        orig_send_stream = p.send_stream

        async def wrapped(**kw):
            await p.send_user_message("late steer")  # enqueued, never drained
            return await orig_send_stream(**kw)

        p.send_stream = wrapped  # type: ignore[assignment]
        resp = await p.send_orchestrate(message="go", tools=[], tool_executor=_noop_executor)
        assert resp.undelivered_steering == ["late steer"]


# ---------------------------------------------------------------------------
# §8 Malformed tool arguments
# ---------------------------------------------------------------------------

class TestMalformedArgs:
    def test_is_malformed_args_detects_raw_shape(self):
        assert _is_malformed_args({"raw": "not json"}) is True

    def test_is_malformed_args_rejects_normal(self):
        assert _is_malformed_args({"path": "/tmp"}) is False
        assert _is_malformed_args({"raw": 123}) is False   # raw must be a str
        assert _is_malformed_args({"raw": "x", "y": 1}) is False

    @pytest.mark.asyncio
    async def test_malformed_args_not_executed(self):
        executed = []

        async def executor(name, inp):
            executed.append((name, inp))
            return {"ok": True}

        # Turn 1: a malformed tool call; Turn 2: conclude.
        p = ScriptedProvider([
            _resp(tool_calls=[ToolCall(id="tc1", name="t", input={"raw": "{bad json"})],
                  stop_reason="tool_use"),
            _resp(text="done", stop_reason="end_turn"),
        ])
        resp = await p.send_orchestrate(
            message="go",
            tools=[{"name": "t", "description": "", "input_schema": {"type": "object"}}],
            tool_executor=executor,
            max_turns=5,
        )
        assert resp.text == "done"
        # The executor was never called for the malformed call.
        assert executed == []
        # A synthetic error tool_result was fed back.
        second = p.seen_messages[1]
        tr = second[-1]["content"][0]
        assert tr["type"] == "tool_result"
        assert tr.get("is_error") is True
        assert "not valid json" in tr["content"].lower()
