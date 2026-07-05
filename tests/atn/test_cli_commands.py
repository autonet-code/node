"""Headless-console command surface — atn/cli.py `_handle_command`.

The console is the fallback control surface when no frontend is connected.
These tests drive `_handle_command` directly against a real Runtime (built
like tests/atn/test_tool_store.py's `_make_runtime`) and capture the module
`console` output to assert on what the operator sees.

Covers the UX-pass deliverables:
  - `msg <id> <text>` to an idle (active) agent: queued + run triggered,
    reusing the same send_agent_message rail as the WS chat surface.
  - `msg` to an unknown agent: a clean error, no crash.
  - `agents` listing: renders per-agent operational detail incl. a pid column.
  - tool-adoption approve/reject flow over the same ToolStore methods the
    WS handlers call.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import atn.cli as cli
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


class _CaptureConsole:
    """Stand-in for the rich Console: records every printed renderable as its
    plain-text form so tests can substring-match. Handles both str markup and
    rich Table renderables (Table.title carries the identifying text)."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args, **kwargs) -> None:
        for a in args:
            if isinstance(a, str):
                self.lines.append(a)
            else:
                # Table (or any renderable): capture its title + any column
                # header text we can reach so assertions can key off them.
                title = getattr(a, "title", None)
                if title:
                    self.lines.append(str(title))
                cols = getattr(a, "columns", None)
                if cols:
                    for c in cols:
                        hdr = getattr(c, "header", None)
                        if hdr:
                            self.lines.append(str(hdr))
                        cells = getattr(c, "_cells", None)
                        if cells:
                            self.lines.extend(str(x) for x in cells)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def cap(monkeypatch):
    c = _CaptureConsole()
    monkeypatch.setattr(cli, "console", c)
    return c


async def _register_active_agent(rt, agent_id, parent_id=None, model="claude-sonnet-5"):
    defn = AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model=model,
        provider=model,
        parent_id=parent_id,
    )
    await rt.register_agent(defn)
    await rt.activate_agent(agent_id)
    return defn


# ---------------------------------------------------------------------------
# msg
# ---------------------------------------------------------------------------


class TestMsg:
    @pytest.mark.asyncio
    async def test_msg_idle_agent_queues_and_triggers(self, tmp_path, cap, monkeypatch):
        rt = _make_runtime(tmp_path)
        await _register_active_agent(rt, "worker")

        # Isolate the console/rail wiring from real LLM execution: stub the
        # engine's trigger_run so send_agent_message's idle-nudge is observable
        # without spawning a turn. This is the exact call send_agent_message
        # makes for an ACTIVE agent.
        triggered: list[str] = []

        async def fake_trigger_run(agent_id, source=None, **kw):
            triggered.append(agent_id)
            return "exec1234deadbeef"

        monkeypatch.setattr(rt.engine, "trigger_run", fake_trigger_run)

        keep = await cli._handle_command("msg worker please do the thing", rt)
        assert keep is True

        # The message landed in the agent's inbox (queued) ...
        assert rt.inbox.count("worker") >= 1
        # ... and an idle agent was nudged into a run.
        assert triggered == ["worker"]
        assert "triggered a run" in cap.text
        assert "exec1234" in cap.text

    @pytest.mark.asyncio
    async def test_msg_unknown_agent_clean_error(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        keep = await cli._handle_command("msg nope hello there", rt)
        assert keep is True
        assert "not found" in cap.text.lower()

    @pytest.mark.asyncio
    async def test_msg_missing_text_shows_usage(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        await _register_active_agent(rt, "worker")
        keep = await cli._handle_command("msg worker", rt)
        assert keep is True
        assert "Usage: msg" in cap.text

    @pytest.mark.asyncio
    async def test_msg_preserves_multiword_text(self, tmp_path, cap, monkeypatch):
        rt = _make_runtime(tmp_path)
        await _register_active_agent(rt, "worker")

        async def fake_trigger_run(agent_id, source=None, **kw):
            return "e" * 16

        monkeypatch.setattr(rt.engine, "trigger_run", fake_trigger_run)
        await cli._handle_command("msg worker one two   three", rt)

        # The exact free text (collapsed by the inbox path) must reach the inbox.
        drained = rt.inbox.drain("worker")
        assert drained, "expected a queued message"
        instr = drained[0].data.get("instruction", "")
        assert "one two   three" == instr


# ---------------------------------------------------------------------------
# agents listing
# ---------------------------------------------------------------------------


class TestAgentsListing:
    @pytest.mark.asyncio
    async def test_agents_renders_with_pid_column(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        await _register_active_agent(rt, "parent-agent")
        await _register_active_agent(rt, "child-agent", parent_id="parent-agent")

        await cli._handle_command("agents", rt)
        out = cap.text
        assert "parent-agent" in out
        assert "child-agent" in out
        # Human-readable operational detail: model, parent, and a pid column.
        assert "model=" in out
        assert "parent=" in out
        # Isolation off => no supervised workers => pid renders as "-".
        assert "pid=-" in out

    @pytest.mark.asyncio
    async def test_agents_shows_worker_pid_when_supervised(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        await _register_active_agent(rt, "iso-agent")

        # Simulate an active isolated worker by inserting a stub into the
        # supervisor's worker map (the same structure the real spawn path
        # populates). The listing must surface its OS PID.
        class _StubWorker:
            pid = 54321

        rt.supervisor._workers["iso-agent"] = _StubWorker()

        await cli._handle_command("agents", rt)
        assert "pid=54321" in cap.text

    @pytest.mark.asyncio
    async def test_agents_empty(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        # Only whatever the runtime auto-registers (none by default here).
        await cli._handle_command("agents", rt)
        # No crash; either lists nothing or the auto-registered set. The
        # command must at minimum not error.
        assert cap.text is not None


# ---------------------------------------------------------------------------
# tool-adoption approve / reject
# ---------------------------------------------------------------------------


def _seed_pending_proposal(rt, digest: str, *, name="risky_tool", by="child"):
    """Insert a pending adoption proposal directly into the ToolStore, matching
    the shape propose_adoption writes. reject_adoption only needs a pending
    row; this keeps the test off the network fetch path."""
    rt.tool_store._proposals[digest] = {
        "digest": digest,
        "name": name,
        "author": "0xauthor",
        "proposed_by": by,
        "reason": "need it for the work",
        "ts": 1,
        "status": "pending",
        "capabilities": {},
        "provenance": {},
    }


class TestAdoptionFlow:
    _DIGEST = "a" * 64

    @pytest.mark.asyncio
    async def test_tools_lists_pending(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        _seed_pending_proposal(rt, self._DIGEST)
        await cli._handle_command("tools", rt)
        out = cap.text
        assert "Pending Tool-Adoption Proposals" in out
        assert self._DIGEST[:16] in out
        assert "risky_tool" in out

    @pytest.mark.asyncio
    async def test_tools_empty(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        await cli._handle_command("tools", rt)
        assert "No pending tool-adoption proposals" in cap.text

    @pytest.mark.asyncio
    async def test_reject_resolves_proposal(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        _seed_pending_proposal(rt, self._DIGEST)
        keep = await cli._handle_command(f"reject {self._DIGEST}", rt)
        assert keep is True
        assert "Rejected" in cap.text
        # State moved off pending — same method the WS handler uses.
        assert rt.tool_store._proposals[self._DIGEST]["status"] == "rejected"
        # And it no longer shows in the pending listing.
        assert rt.tool_store.list_adoption_proposals("pending") == []

    @pytest.mark.asyncio
    async def test_reject_unknown_digest_clean_error(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        await cli._handle_command("reject " + ("b" * 64), rt)
        assert "no pending proposal" in cap.text.lower()

    @pytest.mark.asyncio
    async def test_approve_no_pending_clean_error(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        # No proposal seeded → approve must fail loud, not crash.
        await cli._handle_command("approve " + ("c" * 64), rt)
        assert "no pending proposal" in cap.text.lower()

    @pytest.mark.asyncio
    async def test_reject_missing_digest_shows_usage(self, tmp_path, cap):
        rt = _make_runtime(tmp_path)
        await cli._handle_command("reject", rt)
        assert "Usage: reject" in cap.text
