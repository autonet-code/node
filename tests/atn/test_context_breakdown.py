"""Context-breakdown surface (context_inspect + provider live snapshot)."""
from __future__ import annotations

import pytest

from atn.context_inspect import (
    breakdown_from_parts,
    breakdown_from_provider,
    _MAX_MESSAGE_ITEMS,
)


def _stats(**over):
    base = {
        "active_model": "claude-sonnet-5",
        "context_window": 200_000,
        "last_input_tokens": 0,
        "num_turns": 2,
        "total_cost_usd": 0.0,
        "compaction_count": 0,
        "cumulative_cache_read": 0,
        "cumulative_cache_creation": 0,
        "cumulative_input_tokens": 0,
        "cumulative_output_tokens": 0,
        "session_id": "",
        "context_used_pct": None,
    }
    base.update(over)
    return base


def test_parts_categories_and_free_space():
    system = "x" * 4000  # 1000 tokens
    tools = [{"name": "t1", "description": "d", "input_schema": {}}]
    messages = [
        {"role": "user", "content": "y" * 400},          # 100 tokens
        {"role": "assistant", "content": [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "name": "t1", "input": {"a": 1}},
        ]},
    ]
    bd = breakdown_from_parts(
        system=system, tools=tools, messages=messages,
        stats=_stats(), source="live_inprocess",
        output_reserve_tokens=16_384,
    )
    assert bd["system_prompt"]["est_tokens"] == 1000
    assert bd["tools"]["count"] == 1
    assert bd["messages"]["count"] == 2
    est = bd["est_used_tokens"]
    assert est == (bd["system_prompt"]["est_tokens"]
                   + bd["tools"]["est_tokens"]
                   + bd["messages"]["est_tokens"])
    # window - used - reserve - buffer
    assert bd["free_tokens"] == 200_000 - est - 16_384 - 8_000
    # message itemization carries role/kinds/tool names
    items = bd["messages"]["items"]
    assert items[0]["role"] == "user" and items[0]["kinds"] == ["text"]
    assert items[1]["tool_names"] == ["t1"]
    assert "tool_use" in items[1]["kinds"]


def test_measured_beats_estimate_and_unaccounted():
    bd = breakdown_from_parts(
        system="s" * 400, tools=[], messages=[],
        stats=_stats(last_input_tokens=50_000),
        source="reconstructed",
    )
    assert bd["used_tokens"] == 50_000          # measured wins when larger
    assert bd["unaccounted_tokens"] == 50_000 - bd["est_used_tokens"]


def test_message_itemization_cap_aggregates_oldest():
    messages = [{"role": "user", "content": "m" * 40}
                for _ in range(_MAX_MESSAGE_ITEMS + 50)]
    bd = breakdown_from_parts(
        system="", tools=[], messages=messages,
        stats=_stats(), source="reconstructed",
    )
    items = bd["messages"]["items"]
    assert len(items) == _MAX_MESSAGE_ITEMS + 1
    assert items[0]["role"] == "aggregate"
    assert items[0]["count"] == 50
    # totals include the aggregated mass
    assert bd["messages"]["est_tokens"] == sum(
        len("m" * 40) for _ in messages) // 4


def test_breakdown_from_provider_requires_live_snapshot():
    class Bare:
        pass
    assert breakdown_from_provider(Bare(), "live_worker") is None

    class Live:
        _live_system = "sys " * 100
        _live_tools = [{"name": "t", "description": "", "input_schema": {}}]
        _live_messages = [{"role": "user", "content": "hello"}]
        _live_max_tokens = 4_096
        session_stats = _stats(last_input_tokens=1234)

    bd = breakdown_from_provider(Live(), "live_worker")
    assert bd is not None
    assert bd["source"] == "live_worker"
    assert bd["output_reserve_tokens"] == 4_096
    assert bd["measured_last_input_tokens"] == 1234


def test_send_orchestrate_publishes_live_snapshot():
    """The base loop must expose the exact list object it mutates."""
    from atn.providers.base import Provider

    class Stub(Provider):
        @property
        def name(self):
            return "stub"

        async def send(self, **kwargs):  # pragma: no cover
            raise NotImplementedError

        async def send_stream(self, *, messages, system="", model="",
                              max_tokens=1024, tools=None, temperature=0.0,
                              on_chunk=None):
            from atn.providers.base import ProviderResponse, Usage
            return ProviderResponse(
                text="done", stop_reason="end_turn", model="m",
                usage=Usage(input_tokens=10, output_tokens=2),
            )

    prov = Stub()
    prov._active_model = "claude-sonnet-5"
    import asyncio
    resp = asyncio.run(prov.send_orchestrate(
        message="hi", system="SYS", model="claude-sonnet-5",
        tools=[{"name": "t", "description": "", "input_schema": {}}],
    ))
    assert resp.stop_reason == "end_turn"
    assert prov._live_system == "SYS"
    assert isinstance(prov._live_messages, list)
    roles = [m["role"] for m in prov._live_messages]
    assert roles[0] == "user"
    assert prov._live_max_tokens > 0
    bd = breakdown_from_provider(prov, "live_inprocess")
    assert bd is not None and bd["messages"]["count"] >= 1
