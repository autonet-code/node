"""Tool substrate ATN surface — register_tool + author-lineage scoping.

Covers manifest registration (pinned/attested, persistence, reload),
author derivation from caller, name collision guards, the two-point
scoping contract (listing visibility AND call-time enforcement), owner
grants/revokes, and pinned-code subprocess execution end-to-end.
Design: docs/tool_substrate.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.orchestrator.tools import execute_tool
from atn.tool_registry import ToolCategory
from atn.tool_store import ToolStore


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


async def _register_agent(rt, agent_id, parent_id=None):
    defn = AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5",
        parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


async def _family(rt):
    """grandparent -> parent -> child, plus an unrelated sibling of parent."""
    await _register_agent(rt, "grandparent")
    await _register_agent(rt, "parent", parent_id="grandparent")
    await _register_agent(rt, "child", parent_id="parent")
    await _register_agent(rt, "sibling", parent_id="grandparent")


ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


async def _register_echo(rt, caller_id="child", name="echo_tool"):
    return await execute_tool(
        "register_tool",
        {"name": name, "description": "Echo x back.",
         "input_schema": SCHEMA, "code": ECHO_CODE},
        rt, caller_id=caller_id,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.asyncio
    async def test_register_pinned_from_agent(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        assert "error" not in res
        assert res["trust_class"] == "pinned"
        assert res["author"] == "child"          # derived, never accepted
        assert len(res["digest"]) == 64
        record = rt.tool_store.get(res["digest"])
        assert record.manifest["code_digest"]

    @pytest.mark.asyncio
    async def test_owner_author_is_user(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = await execute_tool(
            "register_tool",
            {"name": "owner_tool", "description": "d", "input_schema": SCHEMA,
             "endpoint": "https://example.test/api"},
            rt, caller_id="",  # owner caller
        )
        assert res["author"] == "user"
        assert res["trust_class"] == "attested"

    @pytest.mark.asyncio
    async def test_core_name_and_reserved_prefix_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await execute_tool(
            "register_tool",
            {"name": "get_snapshot", "description": "d", "input_schema": SCHEMA,
             "code": ECHO_CODE}, rt, caller_id="child")
        assert "core ATN tool" in res["error"]
        res = await execute_tool(
            "register_tool",
            {"name": "reg_sneaky", "description": "d", "input_schema": SCHEMA,
             "code": ECHO_CODE}, rt, caller_id="child")
        assert "may not start" in res["error"]

    @pytest.mark.asyncio
    async def test_no_backing_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await execute_tool(
            "register_tool",
            {"name": "ghost", "description": "d", "input_schema": SCHEMA},
            rt, caller_id="child")
        assert "error" in res  # neither code nor endpoint/connector

    @pytest.mark.asyncio
    async def test_persistence_reload(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        rt.tool_store.grant(res["digest"], "sibling")

        reloaded = ToolStore(rt, rt._config.data_dir / "tools")
        record = reloaded.get(res["digest"])
        assert record is not None
        assert record.name == "echo_tool"
        assert record.grants == {"sibling"}


# ---------------------------------------------------------------------------
# Scoping — visibility and call-time enforcement
# ---------------------------------------------------------------------------


class TestScoping:
    @pytest.mark.asyncio
    async def test_author_lineage_visibility(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)  # author = child
        store = rt.tool_store
        record = store.get(res["digest"])

        assert store.allowed("child", record)        # author
        assert store.allowed("parent", record)       # direct superior
        assert store.allowed("grandparent", record)  # ancestor chain
        assert store.allowed(None, record)           # owner
        assert store.allowed("", record)             # owner
        assert not store.allowed("sibling", record)  # out of lineage

    @pytest.mark.asyncio
    async def test_owner_grant_and_revoke(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        store = rt.tool_store
        record = store.get(res["digest"])

        assert store.grant(res["digest"], "sibling")
        assert store.allowed("sibling", record)
        assert store.revoke(res["digest"], "sibling")
        assert not store.allowed("sibling", record)

    @pytest.mark.asyncio
    async def test_listing_filtered_by_caller(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)

        def names(caller):
            entries = rt.tool_registry.list_all(
                category=ToolCategory.REGISTERED, caller_id=caller)
            return [e["tool_name"] for e in entries]

        assert names("parent") == ["echo_tool"]
        assert names("sibling") == []
        assert names(None) == ["echo_tool"]  # owner

    @pytest.mark.asyncio
    async def test_call_time_enforcement_even_with_digest(self, tmp_path):
        """Knowing the digest out of band must not bypass the lineage gate."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        unified = f"reg_{res['digest'][:12]}"

        blocked = await rt.tool_registry.call_tool(
            unified, {"x": "hi"}, caller_id="sibling")
        assert "author-lineage" in blocked["error"]

    @pytest.mark.asyncio
    async def test_disabled_tool_blocked_for_agents(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        rt.tool_store.set_enabled(res["digest"], False)
        record = rt.tool_store.get(res["digest"])
        assert not rt.tool_store.allowed("child", record)


# ---------------------------------------------------------------------------
# Resolution + execution
# ---------------------------------------------------------------------------


class TestResolutionAndExecution:
    @pytest.mark.asyncio
    async def test_resolve_by_name_prefix_and_ambiguity(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        store = rt.tool_store

        assert store.resolve("echo_tool").digest == res["digest"]
        assert store.resolve(f"reg_{res['digest'][:12]}").digest == res["digest"]
        assert store.resolve(res["digest"]).digest == res["digest"]

        # Same name registered again (different code) -> plain name ambiguous
        await execute_tool(
            "register_tool",
            {"name": "echo_tool", "description": "v2",
             "input_schema": SCHEMA, "code": ECHO_CODE + "# v2\n"},
            rt, caller_id="child")
        assert store.resolve("echo_tool") is None

    @pytest.mark.asyncio
    async def test_pinned_execution_end_to_end(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)

        out = await rt.tool_registry.call_tool(
            "echo_tool", {"x": "round-trip"}, caller_id="child")
        assert out == {"result": {"echo": "round-trip"}}

    @pytest.mark.asyncio
    async def test_receipts_and_fee_ledger(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await execute_tool(
            "register_tool",
            {"name": "paid_echo", "description": "Echo, for a fee.",
             "input_schema": SCHEMA, "code": ECHO_CODE, "fee_atn": 0.5},
            rt, caller_id="child")

        sunk: list[dict] = []
        rt.tool_store.event_sink = sunk.append

        await rt.tool_registry.call_tool("paid_echo", {"x": "1"}, caller_id="parent")
        await rt.tool_registry.call_tool("paid_echo", {"x": "2"}, caller_id="child")

        # Consensus events emitted, caller attested, fee carried.
        assert len(sunk) == 2
        assert sunk[0]["kind"] == "tool_used"
        assert sunk[0]["author_agent"] == "parent"
        assert sunk[0]["tool_author"] == "child"
        assert sunk[0]["manifest_digest"] == res["digest"]
        assert sunk[0]["ok"] is True
        assert sunk[0]["fee_atn"] == 0.5

        # Off-chain ledger: author earned, callers spent.
        balances = rt.tool_store.balances()
        assert balances["earned"] == {"child": 1.0}
        assert balances["spent"] == {"parent": 0.5, "child": 0.5}
        assert balances["usage"][res["digest"]]["ok_count"] == 2

        # Receipt seq survives a reload (no duplicate seq after restart).
        reloaded = ToolStore(rt, rt._config.data_dir / "tools")
        assert reloaded._receipt_seq == 2

    @pytest.mark.asyncio
    async def test_failed_call_recorded_without_fee(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        bad_code = "import sys\nsys.exit(3)\n"
        await execute_tool(
            "register_tool",
            {"name": "broken", "description": "Always fails.",
             "input_schema": SCHEMA, "code": bad_code, "fee_atn": 0.5},
            rt, caller_id="child")

        out = await rt.tool_registry.call_tool("broken", {"x": "1"}, caller_id="child")
        assert "error" in out

        balances = rt.tool_store.balances()
        assert balances["earned"] == {}          # not served = not paid
        entry = next(iter(balances["usage"].values()))
        assert entry["count"] == 1 and entry["ok_count"] == 0

    @pytest.mark.asyncio
    async def test_use_tool_path_carries_caller(self, tmp_path):
        """The agent-facing use_tool executor must thread caller_id into
        the scoping check."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)

        blocked = await execute_tool(
            "use_tool", {"name": "echo_tool", "arguments": {"x": "hi"}},
            rt, caller_id="sibling")
        assert "error" in blocked

        ok = await execute_tool(
            "use_tool", {"name": "echo_tool", "arguments": {"x": "hi"}},
            rt, caller_id="parent")
        assert ok == {"result": {"echo": "hi"}}
