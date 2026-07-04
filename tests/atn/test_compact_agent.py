"""Tests for manual compaction — docs/agentic_loop.md §15.

Covers the compact_agent tool + the provider-level compact-requested flag:
  - permission matrix (owner -> any, parent -> direct child ok;
    sibling / self / grandchild rejected, no side effects)
  - idle path: persisted store summarized in place via the agent's own
    provider (mock summarizer), original history archived, provider evicted
  - running generic-loop path: request_compaction() honored at the iteration
    boundary via _reduce_context(force=True)
  - CONTEXT_COMPACTION event emission with manual: true + requested_by

Idle/permission tests use a real Runtime with mocked providers; the
running-loop test drives Provider.send_orchestrate directly against a
scripted mock provider (matching tests/atn/test_loop_hardening.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from atn.events import EventBus, EventType
from atn.models import AgentDefinition, AgentMode
from atn.orchestrator import ORCHESTRATOR_ID
from atn.orchestrator.tools import _compact_agent
from atn.providers.base import Provider, ProviderResponse, ToolCall, Usage


# ---------------------------------------------------------------------------
# Real-runtime helpers (permission matrix + idle path)
# ---------------------------------------------------------------------------

def _make_runtime(bus: EventBus, tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(bus, data_dir=data_dir, config=config)


async def _register(rt, agent_id: str, parent_id: str | None = None):
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE, parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


def _capture(events: list) -> Any:
    async def _cb(e):
        events.append(e)
    return _cb


def _seed_conversation(rt, agent_id: str, n_turns: int) -> None:
    """Give an idle agent a persisted conversation of n user/assistant turns."""
    store = rt.get_agent_conversation_store(agent_id)
    for i in range(n_turns):
        if i % 2 == 0:
            store.add_user_turn(f"user turn {i}")
        else:
            store.add_assistant_turn(f"assistant turn {i}")


# ---------------------------------------------------------------------------
# Permission matrix (§15)
# ---------------------------------------------------------------------------

class TestPermissions:
    @pytest.mark.asyncio
    async def test_owner_can_compact_any_agent(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "parent-a")
        await _register(rt, "child-a", parent_id="parent-a")
        _seed_conversation(rt, "child-a", 6)

        # Owner caller (orchestrator). Idle agent -> compacted.
        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="SUMMARY"))
        mock.close = AsyncMock()
        with patch.object(rt.providers, "resolve_provider_with_fallback", return_value=mock):
            res = await _compact_agent(rt, {"agent_id": "child-a",
                                            "_caller_id": ORCHESTRATOR_ID})
        assert res["status"] == "compacted"

    @pytest.mark.asyncio
    async def test_parent_can_compact_direct_child(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "parent-b")
        await _register(rt, "child-b", parent_id="parent-b")
        _seed_conversation(rt, "child-b", 6)

        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="SUMMARY"))
        mock.close = AsyncMock()
        with patch.object(rt.providers, "resolve_provider_with_fallback", return_value=mock):
            res = await _compact_agent(rt, {"agent_id": "child-b",
                                            "_caller_id": "parent-b"})
        assert res["status"] == "compacted"

    @pytest.mark.asyncio
    async def test_self_compaction_rejected(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "agent-self")
        _seed_conversation(rt, "agent-self", 6)

        res = await _compact_agent(rt, {"agent_id": "agent-self",
                                        "_caller_id": "agent-self"})
        assert "error" in res
        assert "itself" in res["error"]
        # No side effect: history intact.
        assert rt.get_agent_conversation_store("agent-self").turn_count() == 6

    @pytest.mark.asyncio
    async def test_sibling_compaction_rejected(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "parent-c")
        await _register(rt, "child-c1", parent_id="parent-c")
        await _register(rt, "child-c2", parent_id="parent-c")
        _seed_conversation(rt, "child-c2", 6)

        res = await _compact_agent(rt, {"agent_id": "child-c2",
                                        "_caller_id": "child-c1"})
        assert "error" in res
        assert "direct children" in res["error"]
        assert rt.get_agent_conversation_store("child-c2").turn_count() == 6

    @pytest.mark.asyncio
    async def test_grandchild_compaction_rejected(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "gp")
        await _register(rt, "parent-d", parent_id="gp")
        await _register(rt, "grandchild-d", parent_id="parent-d")
        _seed_conversation(rt, "grandchild-d", 6)

        # gp is the grandparent, not the direct parent -> rejected.
        res = await _compact_agent(rt, {"agent_id": "grandchild-d",
                                        "_caller_id": "gp"})
        assert "error" in res
        assert "direct children" in res["error"]
        assert rt.get_agent_conversation_store("grandchild-d").turn_count() == 6

    @pytest.mark.asyncio
    async def test_unknown_target_errors(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        res = await _compact_agent(rt, {"agent_id": "nope",
                                        "_caller_id": ORCHESTRATOR_ID})
        assert "error" in res
        assert "not found" in res["error"]


# ---------------------------------------------------------------------------
# Idle path — store compaction (§15)
# ---------------------------------------------------------------------------

class TestIdlePath:
    @pytest.mark.asyncio
    async def test_idle_store_compacted_and_archived(self, tmp_path):
        bus = EventBus()
        events: list[Any] = []
        bus.subscribe(None, _capture(events))
        rt = _make_runtime(bus, tmp_path)
        await _register(rt, "idle-1")
        _seed_conversation(rt, "idle-1", 6)

        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="A DENSE SUMMARY"))
        mock.close = AsyncMock()
        with patch.object(rt.providers, "resolve_provider_with_fallback", return_value=mock):
            res = await _compact_agent(rt, {"agent_id": "idle-1",
                                            "_caller_id": ORCHESTRATOR_ID})

        assert res["status"] == "compacted"
        assert res["turns_before"] == 6
        # 1 summary turn + last 2 retained.
        assert res["turns_after"] == 3
        assert res.get("archived_session_id")

        # The summarizer was actually called.
        mock.send.assert_awaited()

        store = rt.get_agent_conversation_store("idle-1")
        turns = store.get_turns()
        assert len(turns) == 3
        assert "A DENSE SUMMARY" in turns[0].content
        # Last two turns preserved verbatim.
        assert turns[-1].content == "assistant turn 5"
        assert turns[-2].content == "user turn 4"

        # Original history archived, not destroyed.
        sessions = store.list_sessions()
        assert any(s["turn_count"] == 6 for s in sessions)

    @pytest.mark.asyncio
    async def test_idle_emits_manual_event(self, tmp_path):
        bus = EventBus()
        events: list[Any] = []
        bus.subscribe(None, _capture(events))
        rt = _make_runtime(bus, tmp_path)
        await _register(rt, "idle-2")
        _seed_conversation(rt, "idle-2", 6)

        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="SUMMARY"))
        mock.close = AsyncMock()
        with patch.object(rt.providers, "resolve_provider_with_fallback", return_value=mock):
            await _compact_agent(rt, {"agent_id": "idle-2",
                                      "_caller_id": ORCHESTRATOR_ID})

        comp = [e for e in events if e.type == EventType.CONTEXT_COMPACTION]
        assert comp, "expected CONTEXT_COMPACTION events"
        assert all(e.data.get("manual") is True for e in comp)
        assert all(e.data.get("requested_by") == ORCHESTRATOR_ID for e in comp)
        statuses = {e.data.get("status") for e in comp}
        assert "in_progress" in statuses and "completed" in statuses

    @pytest.mark.asyncio
    async def test_idle_evicts_cached_provider(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "idle-3")
        _seed_conversation(rt, "idle-3", 6)

        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="SUMMARY"))
        mock.close = AsyncMock()
        # Pre-seed a cached provider so the eviction path runs.
        rt.providers._active_providers["idle-3"] = mock

        await _compact_agent(rt, {"agent_id": "idle-3",
                                  "_caller_id": ORCHESTRATOR_ID})

        assert "idle-3" not in rt.providers._active_providers
        mock.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_idle_summarizer_failure_leaves_store_intact(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "idle-4")
        _seed_conversation(rt, "idle-4", 6)

        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="   "))  # empty
        mock.close = AsyncMock()
        with patch.object(rt.providers, "resolve_provider_with_fallback", return_value=mock):
            res = await _compact_agent(rt, {"agent_id": "idle-4",
                                            "_caller_id": ORCHESTRATOR_ID})

        assert "error" in res
        # Never destroy history on summarizer failure.
        assert rt.get_agent_conversation_store("idle-4").turn_count() == 6

    @pytest.mark.asyncio
    async def test_idle_short_history_no_op(self, tmp_path):
        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "idle-5")
        _seed_conversation(rt, "idle-5", 2)  # == retained tail

        mock = AsyncMock()
        mock.send = AsyncMock(return_value=ProviderResponse(text="SUMMARY"))
        with patch.object(rt.providers, "resolve_provider_with_fallback", return_value=mock):
            res = await _compact_agent(rt, {"agent_id": "idle-5",
                                            "_caller_id": ORCHESTRATOR_ID})

        assert res["status"] == "compacted"
        assert res["turns_before"] == res["turns_after"] == 2
        # Nothing to summarize -> summarizer not called.
        mock.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Running bridge path (§15) — unsupported_while_running
# ---------------------------------------------------------------------------

class TestRunningBridge:
    @pytest.mark.asyncio
    async def test_running_bridge_unsupported(self, tmp_path):
        from atn.providers.bridge import BridgeProvider

        rt = _make_runtime(EventBus(), tmp_path)
        await _register(rt, "bridge-1")

        # A running bridge provider: no SDK input compact control -> unsupported.
        fake_bridge = BridgeProvider.__new__(BridgeProvider)
        rt.providers._active_providers["bridge-1"] = fake_bridge
        rt.registry._running_count["bridge-1"] = 1

        res = await _compact_agent(rt, {"agent_id": "bridge-1",
                                        "_caller_id": ORCHESTRATOR_ID})
        assert res["status"] == "unsupported_while_running"


# ---------------------------------------------------------------------------
# Running generic-loop path (§15) — flag honored at iteration boundary
# ---------------------------------------------------------------------------

class ScriptedProvider(Provider):
    """Provider whose send_stream() plays back a scripted list (per test_loop_hardening)."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.stream_calls = 0
        self.summary_calls = 0
        self.seen_messages: list[list[dict]] = []
        self._active_model = "test-model"

    @property
    def name(self) -> str:
        return "scripted"

    async def send(self, *, messages, system="", model="", max_tokens=1024,
                   tools=None, temperature=0.0) -> ProviderResponse:
        self.summary_calls += 1
        return ProviderResponse(text="SUMMARY", stop_reason="end_turn")

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


class TestRunningLoopHonorsFlag:
    @pytest.mark.asyncio
    async def test_request_compaction_false_when_idle(self):
        p = ScriptedProvider([])
        # No orchestration active -> steering queue is None -> False.
        assert p.request_compaction(requested_by="owner") is False

    @pytest.mark.asyncio
    async def test_flag_honored_at_iteration_boundary(self):
        # Turn 1 returns a tool call; during the executor we request compaction.
        # The loop must run a forced _reduce_context at the next boundary and
        # emit a CONTEXT_COMPACTION event with manual: true.
        p = ScriptedProvider([
            _resp(tool_calls=[_tc("t")], stop_reason="tool_use"),
            _resp(text="done", stop_reason="end_turn"),
        ])

        events: list[Any] = []

        class _Bus:
            async def emit(self, e):
                events.append(e)

        p.event_bus = _Bus()
        p.source_agent_id = "agent-x"

        reduce_calls: list[dict] = []
        orig_reduce = Provider._reduce_context

        async def _spy_reduce(self, messages, system, model, original_request,
                              max_tokens, force=False, manual=False, requested_by=""):
            reduce_calls.append({"force": force, "manual": manual,
                                 "requested_by": requested_by})
            # Emit the completed event the real compactor would, so the test can
            # assert on the manual fields without a real summarizer round-trip.
            if manual:
                from atn.events import Event, EventType
                await self.event_bus.emit(Event(
                    type=EventType.CONTEXT_COMPACTION,
                    source=self.source_agent_id,
                    data={"agent_id": self.source_agent_id, "status": "completed",
                          "manual": manual, "requested_by": requested_by},
                ))

        async def executor(name, inp):
            # Orchestration is active here -> flag accepted.
            assert p.request_compaction(requested_by="owner") is True
            return {"ok": True}

        with patch.object(Provider, "_reduce_context", _spy_reduce):
            await p.send_orchestrate(
                message="go",
                tools=[{"name": "t", "description": "",
                        "input_schema": {"type": "object"}}],
                tool_executor=executor,
                max_turns=5,
            )

        # A forced, manual reduction ran exactly once, tagged with the requester.
        manual_calls = [c for c in reduce_calls if c["manual"]]
        assert len(manual_calls) == 1
        assert manual_calls[0]["force"] is True
        assert manual_calls[0]["requested_by"] == "owner"

        # Flag consumed (does not re-fire).
        assert p._compact_requested is False

        comp = [e for e in events if e.type == EventType.CONTEXT_COMPACTION]
        assert comp and comp[0].data.get("manual") is True
        assert comp[0].data.get("requested_by") == "owner"
