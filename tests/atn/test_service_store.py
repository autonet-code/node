"""ServiceStore + services-market WS rail — provider-side coverage.

Design: docs/services_market.md. Covers spec build/validation, register +
reload persistence, version_of lineage on update_ask, retire, the
provider-side service_request dispatch end-to-end through a registered
echo tool, and the request log + summary the reviews later ride on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.orchestrator.tools import execute_tool
from atn.service_spec import (
    build_service_spec,
    canonical_service_bytes,
    validate_service_spec,
)
from atn.service_store import ServiceStore


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


ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}

ASK = {"token": "0xToKeN", "amount": "1000000", "unit": "per_item"}


async def _register_echo_tool(rt, caller_id="child", name="echo_tool"):
    return await execute_tool(
        "register_tool",
        {"name": name, "description": "Echo x back.",
         "input_schema": SCHEMA, "code": ECHO_CODE},
        rt, caller_id=caller_id,
    )


# ---------------------------------------------------------------------------
# Spec build + validation
# ---------------------------------------------------------------------------


class TestServiceSpec:
    def test_build_valid(self):
        spec = build_service_spec(
            name="transcribe", description="Speech to text.",
            input_schema=SCHEMA, author="agent-1", ask=ASK, created_ts=7)
        assert spec["kind"] == "service_spec"
        assert spec["name"] == "transcribe"
        assert spec["ask"] == ASK
        assert spec["version_of"] is None
        assert spec["created_ts"] == 7
        assert validate_service_spec(spec) == []

    def test_missing_fields_rejected(self):
        errors = validate_service_spec({"kind": "service_spec"})
        assert any("name" in e for e in errors)
        assert any("description" in e for e in errors)
        assert any("input_schema" in e for e in errors)
        assert any("ask" in e for e in errors)

    def test_bad_ask_rejected(self):
        with pytest.raises(ValueError):
            build_service_spec(
                name="s", description="d", input_schema=SCHEMA,
                author="a", ask={"amount": "x"})  # no token, non-int amount
        errors = validate_service_spec({
            "kind": "service_spec", "name": "s", "description": "d",
            "input_schema": SCHEMA, "author": "a",
            "ask": {"token": "0x1", "amount": "-5", "unit": "weird"}})
        assert any("non-negative" in e for e in errors)
        assert any("unit" in e for e in errors)

    def test_canonical_bytes_excludes_sig(self):
        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", ask=ASK, created_ts=1)
        base = canonical_service_bytes(spec)
        spec["author_sig"] = "0xdeadbeef"
        assert canonical_service_bytes(spec) == base  # sig excluded

    def test_canonical_bytes_deterministic_and_sorted(self):
        a = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", ask=ASK, created_ts=1)
        b = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", ask=ASK, created_ts=1)
        assert canonical_service_bytes(a) == canonical_service_bytes(b)


# ---------------------------------------------------------------------------
# Store: register, persist, lineage, retire
# ---------------------------------------------------------------------------


class TestServiceStore:
    @pytest.mark.asyncio
    async def test_register_and_reload(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        res = rt.service_store.register(
            name="svc", description="A service.", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool="deadbeef")
        assert len(res["digest"]) == 64
        assert res["spec"]["author"] == "child"

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        record = reloaded.get(res["digest"])
        assert record is not None
        assert record.name == "svc"
        assert record.backing_tool == "deadbeef"
        assert not record.retired

    @pytest.mark.asyncio
    async def test_resolve_by_name_prefix_digest(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="uniq", description="d", input_schema=SCHEMA,
            author="user", ask=ASK)
        store = rt.service_store
        assert store.resolve("uniq").digest == res["digest"]
        assert store.resolve(f"svc_{res['digest'][:12]}").digest == res["digest"]
        assert store.resolve(res["digest"]).digest == res["digest"]

    @pytest.mark.asyncio
    async def test_update_ask_lineage(self, tmp_path):
        rt = _make_runtime(tmp_path)
        v1 = rt.service_store.register(
            name="svc", description="d", input_schema=SCHEMA,
            author="user", ask=ASK, backing_tool="tool-a")
        new_ask = {"token": "0xToKeN", "amount": "2000000", "unit": "per_item"}
        v2 = rt.service_store.update_ask(v1["digest"], new_ask)

        assert v2["digest"] != v1["digest"]              # changed ask = new digest
        assert v2["spec"]["version_of"] == v1["digest"]  # lineage link
        assert v2["spec"]["ask"] == new_ask
        # backing_tool carried forward
        assert rt.service_store.get(v2["digest"]).backing_tool == "tool-a"
        # predecessor soft-retired; live list resolves to current
        assert rt.service_store.get(v1["digest"]).retired
        assert rt.service_store.resolve("svc").digest == v2["digest"]

    @pytest.mark.asyncio
    async def test_retire(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="svc", description="d", input_schema=SCHEMA,
            author="user", ask=ASK)
        assert rt.service_store.retire(res["digest"])
        assert rt.service_store.get(res["digest"]).retired
        assert rt.service_store.list() == []
        assert len(rt.service_store.list(include_retired=True)) == 1
        assert not rt.service_store.retire("nope")


# ---------------------------------------------------------------------------
# Provider-side service_request dispatch (WS rail) + request log
# ---------------------------------------------------------------------------


def _server(rt):
    from atn.ws_server import WebSocketBridge
    return WebSocketBridge(rt)


async def _svc_request(rt, spec_digest, request_id, args, client="0xClient"):
    server = _server(rt)
    return await server._handle_service_request(
        {"spec_digest": spec_digest, "request_id": request_id,
         "args": args, "client": client},
        msg_id="m1",
    )


class TestServiceRequestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_through_backing_tool(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="Echo service.", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool=tool["digest"])

        out = await _svc_request(rt, svc["digest"], "req-1", {"x": "hi"})
        assert out["ok"] is True
        assert out["result"]["request_id"] == "req-1"
        assert out["result"]["result"] == {"echo": "hi"}

        # Provider request log recorded the served item.
        summary = rt.service_store.summary()
        entry = summary[svc["digest"]]
        assert entry["count"] == 1
        assert entry["ok_count"] == 1
        assert entry["success_rate"] == 1.0
        assert entry["unique_clients"] == 1

    @pytest.mark.asyncio
    async def test_unknown_and_retired_and_no_backing(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")

        miss = await _svc_request(rt, "f" * 64, "r", {"x": "1"})
        assert miss["ok"] is False and "Unknown" in miss["error"]

        # No backing tool -> failure recorded.
        svc = rt.service_store.register(
            name="ghost", description="d", input_schema=SCHEMA,
            author="child", ask=ASK)
        out = await _svc_request(rt, svc["digest"], "r", {"x": "1"})
        assert out["ok"] is False and "backing" in out["error"]

        # Retired -> refused.
        rt.service_store.retire(svc["digest"])
        out = await _svc_request(rt, svc["digest"], "r2", {"x": "1"})
        assert out["ok"] is False and "retired" in out["error"].lower()

        # The no-backing attempt is on the provider log as a failure.
        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 1 and entry["ok_count"] == 0

    @pytest.mark.asyncio
    async def test_request_log_survives_reload(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool=tool["digest"])
        await _svc_request(rt, svc["digest"], "req-1", {"x": "a"})
        await _svc_request(rt, svc["digest"], "req-2", {"x": "b"}, client="0xOther")

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded._request_seq == 2
        entry = reloaded.summary()[svc["digest"]]
        assert entry["count"] == 2
        assert entry["unique_clients"] == 2
