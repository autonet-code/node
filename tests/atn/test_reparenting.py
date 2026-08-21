"""Reparenting the agent tree + pushing child completions to parents.

Two tracks, both on the parent/child rails:

A. REPARENTING ("regional manager"): a formerly top-level agent gets a parent
   placed over it (or a subtree is re-homed). The full parent rail must
   re-activate for the new relationship — child->parent notification, the
   parent's delegate probes, the budget cascade over the new subtree, and
   author-lineage tool scoping. Cycles and limit/budget violations are
   rejected and roll back.

B. PUSH CHILD COMPLETIONS: a child reaching a terminal state pushes a compact
   notification to its parent — live-injected when the parent is running,
   otherwise batched into the parent's next natural run (NORMAL priority, no
   wake) unless the parent opts into wake_parent_on_child. Failures always
   wake (HIGH). No infinite loops when parent and child would notify each
   other.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from atn.events import EventBus
from atn.models import (
    AgentDefinition,
    AgentMode,
    ExecutionRecord,
    ExecutionStatus,
    MessagePriority,
    MessageType,
)
from atn.agent_tools import execute_tool


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

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


def _budget(limit: int) -> dict:
    return {"claude_max": {"limit": limit, "unit": "tokens", "period": "none"}}


async def _register(rt, agent_id, parent_id=None, limit=None, **kw):
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5",
        parent_id=parent_id,
        budgets=_budget(limit) if limit else {},
        **kw,
    )
    await rt.register_agent(defn)
    return defn


def _record(status=ExecutionStatus.COMPLETED, output=None, error=None,
            execution_id="exec-1") -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=execution_id,
        agent_id="child",
        status=status,
        trigger_source="test",
        output=output,
        error=error,
    )


# ===========================================================================
# Track A — reparenting
# ===========================================================================

@pytest.mark.asyncio
async def test_reparent_places_manager_over_toplevel(tmp_path):
    """The core 'regional manager' move: a top-level agent gets a parent."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker")  # top-level, no parent

    assert rt.get_agent("worker").parent_id is None
    assert rt.registry.get_children("manager") == []

    res = await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "manager"},
        rt, caller_id="")  # owner
    assert "parent_id" in res.get("changed", []), res

    # parent_id is live everywhere the rail reads it.
    assert rt.get_agent("worker").parent_id == "manager"
    child_ids = [c.id for c in rt.registry.get_children("manager")]
    assert "worker" in child_ids


@pytest.mark.asyncio
async def test_reparent_delegate_probe_sees_new_child(tmp_path):
    """get_children_status (a delegate probe) picks up the new child."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker")

    before = await execute_tool(
        "get_children_status", {}, rt, caller_id="manager")
    assert before["count"] == 0

    await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "manager"},
        rt, caller_id="")

    after = await execute_tool(
        "get_children_status", {}, rt, caller_id="manager")
    assert after["count"] == 1
    assert after["children"][0]["id"] == "worker"


@pytest.mark.asyncio
async def test_reparent_notifies_new_parent_on_completion(tmp_path):
    """Child->parent completion messaging follows the NEW relationship."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker")

    # Before reparenting: a completion notifies nobody (worker is top-level).
    await rt.registry.on_agent_completed(
        "worker", _record(), rt.providers._active_providers)
    assert rt.inbox.count("manager") == 0

    await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "manager"},
        rt, caller_id="")

    # After: the completion lands in the new parent's inbox.
    await rt.registry.on_agent_completed(
        "worker", _record(), rt.providers._active_providers)
    msgs = rt.inbox.peek("manager")
    child_msgs = [m for m in msgs if m.data.get("type") == "child_completed"]
    assert len(child_msgs) == 1
    assert child_msgs[0].data["child_id"] == "worker"


@pytest.mark.asyncio
async def test_reparent_budget_cascade_covers_new_subtree(tmp_path):
    """The moved subtree's caps are re-validated against the new ancestry.

    A child whose cap fit under its OLD (roomy) parent but exceeds the NEW
    (tight) parent's headroom is rejected — the move rolls back."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "roomy", limit=100_000)
    await _register(rt, "worker", parent_id="roomy", limit=80_000)
    await _register(rt, "tight", limit=50_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "tight"},
        rt, caller_id="")
    assert "error" in res, res
    # Rolled back — still under roomy.
    assert rt.get_agent("worker").parent_id == "roomy"


@pytest.mark.asyncio
async def test_reparent_budget_cascade_allows_within_headroom(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "roomy", limit=100_000)
    await _register(rt, "worker", parent_id="roomy", limit=20_000)
    await _register(rt, "other", limit=60_000)

    res = await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "other"},
        rt, caller_id="")
    assert "parent_id" in res.get("changed", []), res
    assert rt.get_agent("worker").parent_id == "other"


@pytest.mark.asyncio
async def test_reparent_tool_lineage_scoping_picks_up_new_ancestry(tmp_path):
    """A tool authored by the child becomes visible to the NEW parent
    (author-lineage scoping walks parent_id, which reparenting updates)."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker")

    store = rt.tool_store
    reg = store.register(
        name="worker_tool", description="does a thing",
        input_schema={"type": "object", "properties": {}},
        author="worker",
        code="import sys,json; json.load(sys.stdin); print('{}')",
    )
    record = store.get(reg["digest"])

    # Before: manager is not in worker's lineage → not allowed.
    assert not store.allowed("manager", record)

    await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "manager"},
        rt, caller_id="")

    # After: manager is now worker's parent → an ancestor → allowed.
    assert store.allowed("manager", record)


@pytest.mark.asyncio
async def test_reparent_cycle_rejected(tmp_path):
    """Pointing a parent under its own descendant is a cycle → rejected."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "a")
    await _register(rt, "b", parent_id="a")
    await _register(rt, "c", parent_id="b")

    # Try to move 'a' under 'c' (a is c's ancestor) → cycle.
    err = rt.registry.reparent_agent("a", "c")
    assert err is not None and "cycle" in err.lower()
    assert rt.get_agent("a").parent_id is None


@pytest.mark.asyncio
async def test_reparent_self_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "a")
    err = rt.registry.reparent_agent("a", "a")
    assert err is not None
    assert rt.get_agent("a").parent_id is None


@pytest.mark.asyncio
async def test_reparent_promote_to_root(tmp_path):
    """parent_id=null promotes a child back to top-level."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker", parent_id="manager")

    res = await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": None},
        rt, caller_id="")
    assert "parent_id" in res.get("changed", []), res
    assert rt.get_agent("worker").parent_id is None
    assert rt.registry.get_children("manager") == []


@pytest.mark.asyncio
async def test_reparent_access_control(tmp_path):
    """A stranger agent cannot reparent an unrelated agent."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker")
    await _register(rt, "stranger")

    res = await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "manager"},
        rt, caller_id="stranger")
    assert "error" in res
    assert rt.get_agent("worker").parent_id is None


@pytest.mark.asyncio
async def test_reparent_new_parent_may_adopt(tmp_path):
    """The prospective new parent may adopt an agent under itself."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "manager")
    await _register(rt, "worker")

    res = await execute_tool(
        "update_agent", {"agent_id": "worker", "parent_id": "manager"},
        rt, caller_id="manager")  # the new parent adopts
    assert "parent_id" in res.get("changed", []), res
    assert rt.get_agent("worker").parent_id == "manager"


@pytest.mark.asyncio
async def test_reparent_unknown_new_parent_rejected(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "worker")
    err = rt.registry.reparent_agent("worker", "nope")
    assert err is not None and "unknown new parent" in err.lower()


# ===========================================================================
# Track B — push child completions to parents
# ===========================================================================

@pytest.mark.asyncio
async def test_completion_batches_by_default(tmp_path):
    """Default (wake_parent_on_child=False): completion queues at NORMAL,
    which does NOT wake an idle parent (has_wake_priority is False)."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent")
    await _register(rt, "child", parent_id="parent")

    await rt.registry.on_agent_completed(
        "child", _record(execution_id="e1"), rt.providers._active_providers)

    msgs = rt.inbox.peek("parent")
    child_msgs = [m for m in msgs if m.data.get("type") == "child_completed"]
    assert len(child_msgs) == 1
    assert child_msgs[0].priority == MessagePriority.NORMAL
    # NORMAL is below the wake threshold → an idle parent is not woken.
    assert not rt.inbox.has_wake_priority("parent")


@pytest.mark.asyncio
async def test_completion_wakes_when_opted_in(tmp_path):
    """wake_parent_on_child=True → HIGH priority, which wakes an idle parent."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent", wake_parent_on_child=True)
    await _register(rt, "child", parent_id="parent")

    await rt.registry.on_agent_completed(
        "child", _record(), rt.providers._active_providers)

    msgs = rt.inbox.peek("parent")
    child_msgs = [m for m in msgs if m.data.get("type") == "child_completed"]
    assert len(child_msgs) == 1
    assert child_msgs[0].priority == MessagePriority.HIGH
    assert rt.inbox.has_wake_priority("parent")


@pytest.mark.asyncio
async def test_completion_includes_result_summary(tmp_path):
    """The notification carries a cheap one-line result summary."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent")
    await _register(rt, "child", parent_id="parent")

    rec = _record(output={"result": "Found 3 issues in the config."})
    await rt.registry.on_agent_completed(
        "child", rec, rt.providers._active_providers)

    msg = rt.inbox.peek("parent")[0]
    assert msg.data["summary"] == "Found 3 issues in the config."
    assert "Found 3 issues" in msg.data["instruction"]


@pytest.mark.asyncio
async def test_completion_live_injects_into_running_parent(tmp_path):
    """Parent running → the completion is live-injected into its provider
    loop (not the inbox)."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent")
    await _register(rt, "child", parent_id="parent")

    # Simulate a running parent with a live provider.
    provider = AsyncMock()
    provider.send_user_message = AsyncMock()
    rt.providers._active_providers["parent"] = provider
    rt.registry._running_count["parent"] = 1

    await rt.registry.on_agent_completed(
        "child", _record(output={"result": "done"}),
        rt.providers._active_providers)

    provider.send_user_message.assert_awaited_once()
    injected = provider.send_user_message.await_args.args[0]
    assert "child" in injected
    # Live injection means NOTHING queued in the inbox.
    assert rt.inbox.count("parent") == 0


@pytest.mark.asyncio
async def test_killed_child_notifies_parent(tmp_path):
    """A KILLED child is terminal and notifies (child_completed, status killed)."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent")
    await _register(rt, "child", parent_id="parent")

    await rt.registry.on_agent_completed(
        "child", _record(status=ExecutionStatus.KILLED),
        rt.providers._active_providers)

    msgs = rt.inbox.peek("parent")
    child_msgs = [m for m in msgs if m.data.get("type") == "child_completed"]
    assert len(child_msgs) == 1
    assert child_msgs[0].data["status"] == "killed"


@pytest.mark.asyncio
async def test_notify_parent_false_suppresses_push(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent")
    await _register(rt, "child", parent_id="parent", notify_parent=False)

    await rt.registry.on_agent_completed(
        "child", _record(), rt.providers._active_providers)
    assert rt.inbox.count("parent") == 0


@pytest.mark.asyncio
async def test_failure_always_wakes_parent(tmp_path):
    """A failure notifies via child_error at HIGH even with wake off."""
    rt = _make_runtime(tmp_path)
    await _register(rt, "parent")  # wake_parent_on_child default False
    await _register(rt, "child", parent_id="parent", notify_parent=False)

    rec = _record(status=ExecutionStatus.FAILED, error="boom")
    await rt.registry.notify_parent_of_failure(
        "child", rec, rt.providers._active_providers)

    msgs = rt.inbox.peek("parent")
    err_msgs = [m for m in msgs if m.data.get("type") == "child_error"]
    assert len(err_msgs) == 1
    assert err_msgs[0].priority == MessagePriority.HIGH
    assert err_msgs[0].type == MessageType.ALERT
    assert "boom" in err_msgs[0].data["instruction"]


@pytest.mark.asyncio
async def test_mutual_notify_no_infinite_loop(tmp_path):
    """Parent and child pointing at each other (a would-be notify cycle):
    the batching default means neither wakes the other, and the self-parent
    guard prevents a node from notifying itself. No unbounded fan-out.

    We can't build a real parent_id cycle (the reparent guard forbids it), so
    this exercises the closest legal shape: two agents that are each other's
    declared 'parent' via direct registration, and confirms each completion
    produces at most ONE message and never re-triggers a cascade."""
    rt = _make_runtime(tmp_path)
    # a is b's parent; b is a's parent — a 2-cycle built by direct
    # registration (bypasses the reparent guard on purpose for the test).
    await _register(rt, "a")
    await _register(rt, "b", parent_id="a")
    rt.get_agent("a").parent_id = "b"  # force the cycle

    # a completes → notifies b (one message, NORMAL, no wake).
    await rt.registry.on_agent_completed(
        "a", _record(execution_id="ea"), rt.providers._active_providers)
    # b completes → notifies a (one message, NORMAL, no wake).
    await rt.registry.on_agent_completed(
        "b", _record(execution_id="eb"), rt.providers._active_providers)

    a_msgs = [m for m in rt.inbox.peek("a") if m.data.get("type") == "child_completed"]
    b_msgs = [m for m in rt.inbox.peek("b") if m.data.get("type") == "child_completed"]
    # Exactly one each — no cascade, no re-wake (NORMAL priority batches).
    assert len(a_msgs) == 1
    assert len(b_msgs) == 1
    assert not rt.inbox.has_wake_priority("a")
    assert not rt.inbox.has_wake_priority("b")
