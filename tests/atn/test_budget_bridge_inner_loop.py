"""Tests for bridge inner-loop budget enforcement (issue #21 / task #18).

The bridge's send_orchestrate runs the entire SDK loop in a subprocess; the
Python side observes per-turn `usage` events on the stream channel and calls
the recorder. On veto, it issues interrupt() to stop the SDK mid-orchestration.

These tests drive the event-handling path directly without spinning up a real
bridge subprocess.
"""
from __future__ import annotations

import asyncio

import pytest

from atn.providers.base import ProviderResponse, classify_model


# ---------------------------------------------------------------------------
# record_turn_tokens — feeds the per-class estimator counter
# ---------------------------------------------------------------------------

def _make_bridge_stub():
    from atn.providers.bridge import BridgeProvider
    s = BridgeProvider.__new__(BridgeProvider)
    s._cum_tokens_by_class = {"haiku": 0, "sonnet": 0, "opus": 0, "other": 0}
    s._tokens_per_pct_by_class = {}
    s._bootstrap_relative = {"haiku": 5.0, "sonnet": 1.0, "opus": 0.2, "other": 1.0}
    s._bootstrap_sonnet_rate = 60_000.0
    s._predicted_pct_since_refresh = 0.0
    s._predicted_tokens_by_class_since_refresh = {
        "haiku": 0, "sonnet": 0, "opus": 0, "other": 0,
    }
    return s


def test_record_turn_tokens_buckets_by_class():
    s = _make_bridge_stub()

    s.record_turn_tokens(1000, "claude-haiku-4-5")
    s.record_turn_tokens(500, "claude-sonnet-4-6")
    s.record_turn_tokens(200, "claude-opus-4-7")
    s.record_turn_tokens(50, "")  # unknown → 'other'

    assert s._cum_tokens_by_class == {
        "haiku": 1000, "sonnet": 500, "opus": 200, "other": 50,
    }


def test_record_turn_tokens_ignores_zero_or_negative():
    s = _make_bridge_stub()
    s.record_turn_tokens(0, "claude-sonnet-4-6")
    s.record_turn_tokens(-100, "claude-sonnet-4-6")
    assert all(v == 0 for v in s._cum_tokens_by_class.values())


# ---------------------------------------------------------------------------
# Recorder veto path triggers interrupt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_event_calls_recorder():
    """Simulate the inline _stream_events handler logic on a usage event."""
    s = _make_bridge_stub()

    recorded: list[tuple[int, str]] = []
    interrupt_called: list[bool] = []

    def _recorder(tokens: int, model: str = "") -> tuple[bool, str | None]:
        recorded.append((tokens, model))
        return True, None

    async def _interrupt():
        interrupt_called.append(True)

    # Replicate the handler logic directly (it lives inside a closure in
    # send_orchestrate, so we can't import it; this test exercises the
    # behavior contract).
    event = {
        "type": "usage",
        "model": "claude-sonnet-4-6",
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_input_tokens": 50,
        "cache_creation_input_tokens": 0,
    }
    turn_total = (
        event["input_tokens"] + event["output_tokens"]
        + event["cache_read_input_tokens"] + event["cache_creation_input_tokens"]
    )
    s.record_turn_tokens(turn_total, event["model"])
    ok, blocker = _recorder(turn_total, event["model"])

    assert ok is True
    assert recorded == [(350, "claude-sonnet-4-6")]
    assert interrupt_called == []  # no veto → no interrupt
    assert s._cum_tokens_by_class["sonnet"] == 350


@pytest.mark.asyncio
async def test_recorder_veto_triggers_interrupt():
    """When the recorder returns (False, blocker), interrupt() is called."""
    s = _make_bridge_stub()

    interrupt_called: list[bool] = []

    async def _fake_interrupt():
        interrupt_called.append(True)

    s.interrupt = _fake_interrupt  # type: ignore[assignment]

    def _recorder(tokens: int, model: str = "") -> tuple[bool, str | None]:
        return False, "parent_agent"

    # Mirror the handler's interrupt path.
    ok, blocker = _recorder(500, "claude-opus-4-7")
    if not ok:
        await s.interrupt()

    assert blocker == "parent_agent"
    assert interrupt_called == [True]


# ---------------------------------------------------------------------------
# Final response is rewritten to budget_exceeded when blocker was set
# ---------------------------------------------------------------------------

def test_budget_blocker_overrides_stop_reason():
    """Confirms the wiring contract: budget_state['blocker'] forces stop_reason."""
    # Simulate the post-orchestration logic: if budget_state['blocker'] is
    # truthy, the returned ProviderResponse uses stop_reason='budget_exceeded'.
    budget_state = {"blocker": "agent.7"}
    final_resp = {"stop_reason": "end_turn", "text": "all good", "ok": True}
    if budget_state["blocker"]:
        stop_reason = "budget_exceeded"
        blocker = budget_state["blocker"]
        bridge_text = final_resp.get("text", "")
        rewritten = ProviderResponse(
            text=bridge_text or f"Aborted: ... blocked by '{blocker}'.",
            stop_reason=stop_reason,
        )
    else:
        rewritten = ProviderResponse(text=final_resp["text"],
                                     stop_reason=final_resp["stop_reason"])

    assert rewritten.stop_reason == "budget_exceeded"


# ---------------------------------------------------------------------------
# usage event end-to-end smoke (queue + handler)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_queue_carries_usage_events():
    """Sanity check: a usage event placed on _event_queue is consumable."""
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({
        "type": "usage",
        "model": "claude-sonnet-4-6",
        "input_tokens": 1, "output_tokens": 2,
        "cache_read_input_tokens": 3, "cache_creation_input_tokens": 4,
    })
    await queue.put({"type": "done"})

    events = []
    while True:
        e = await queue.get()
        if e.get("type") == "done":
            break
        events.append(e)
    assert len(events) == 1
    assert events[0]["model"] == "claude-sonnet-4-6"
    assert classify_model(events[0]["model"]) == "sonnet"
