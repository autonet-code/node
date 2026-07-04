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
             "connector_id": "google_calendar"},
            rt, caller_id="",  # owner caller
        )
        assert res["author"] == "user"
        assert res["trust_class"] == "attested"

    @pytest.mark.asyncio
    async def test_endpoint_rejected_as_service(self, tmp_path):
        """Remote paid APIs are Services, not tools (spec v2)."""
        rt = _make_runtime(tmp_path)
        res = await execute_tool(
            "register_tool",
            {"name": "remote_api", "description": "d", "input_schema": SCHEMA,
             "endpoint": "https://example.test/api"},
            rt, caller_id="")
        assert "Services" in res["error"]

    @pytest.mark.asyncio
    async def test_private_by_default_publish_deliberate(self, tmp_path):
        """Three tiers (spec v2): private is the default — no substrate
        push; publish=true (or set_published) fires the manifest sink."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        pushed: list[tuple] = []
        rt.tool_store.manifest_sink = lambda m, a: pushed.append((m["name"], a))

        res = await _register_echo(rt)                     # default: private
        assert res["published"] is False
        assert pushed == []
        assert rt.tool_store.push_all_manifests() == 0     # backfill skips private

        # The sink receives the CONSENSUS author (the agent's 0x
        # address), not the local id — chain-claimable attribution.
        child_addr = rt.get_agent("child").identity.address
        res2 = await execute_tool(
            "register_tool",
            {"name": "public_echo", "description": "Echo.", "input_schema": SCHEMA,
             "code": ECHO_CODE, "publish": True},
            rt, caller_id="child")
        assert res2["published"] is True
        assert pushed == [("public_echo", child_addr)]

        # Owner flips the private one to published later.
        assert rt.tool_store.set_published(res["digest"], True)
        assert pushed[-1] == ("echo_tool", child_addr)
        # Publish state survives reload.
        reloaded = ToolStore(rt, rt._config.data_dir / "tools")
        assert reloaded.get(res["digest"]).published is True

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
        # Owner sees echo_tool PLUS the platform-authored reference harness
        # distro manifests (author "user"), which bootstrap at runtime init.
        owner_names = names(None)
        assert "echo_tool" in owner_names
        assert all(n == "echo_tool" or n.startswith("atn_")
                   for n in owner_names)

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
    async def test_receipts_and_usage_ledger(self, tmp_path):
        """Fees left tools with spec v2 (they belong to Services); the
        receipt rail remains: local ledger + consensus events."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)

        sunk: list[dict] = []
        rt.tool_store.event_sink = sunk.append

        await rt.tool_registry.call_tool("echo_tool", {"x": "1"}, caller_id="parent")
        await rt.tool_registry.call_tool("echo_tool", {"x": "2"}, caller_id="child")

        # Consensus events emitted, caller attested — the rail carries
        # 0x identities (chain-claimable); local ids stay in the jsonl.
        assert len(sunk) == 2
        assert sunk[0]["kind"] == "tool_used"
        assert sunk[0]["author_agent"] == rt.get_agent("parent").identity.address
        assert sunk[0]["tool_author"] == rt.get_agent("child").identity.address
        assert sunk[0]["manifest_digest"] == res["digest"]
        assert sunk[0]["ok"] is True

        balances = rt.tool_store.balances()
        assert balances["usage"][res["digest"]]["ok_count"] == 2
        assert balances["earned"] == {}   # no fees on tools in v2

        # Receipt seq survives a reload (no duplicate seq after restart).
        reloaded = ToolStore(rt, rt._config.data_dir / "tools")
        assert reloaded._receipt_seq == 2

    @pytest.mark.asyncio
    async def test_failed_call_recorded(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        bad_code = "import sys\nsys.exit(3)\n"
        await execute_tool(
            "register_tool",
            {"name": "broken", "description": "Always fails.",
             "input_schema": SCHEMA, "code": bad_code},
            rt, caller_id="child")

        out = await rt.tool_registry.call_tool("broken", {"x": "1"}, caller_id="child")
        assert "error" in out

        balances = rt.tool_store.balances()
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


# ---------------------------------------------------------------------------
# Cognitive attestation (docs/tool_substrate.md — two receipt tiers)
# ---------------------------------------------------------------------------


def _fake_embedder(text):
    """Deterministic, dependency-free stand-in for the usefulness embedder
    so tests never spawn the torch subprocess worker."""
    return (0.1, 0.2, 0.3) if text else ()


class TestAttestation:
    @pytest.mark.asyncio
    async def test_attest_end_to_end_emits_attested_event(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)  # author = child
        store = rt.tool_store
        store._embedder = _fake_embedder  # avoid heavy embedder

        sunk: list[dict] = []
        store.event_sink = sunk.append

        out = await execute_tool(
            "attest_tools",
            {"judgments": [{"tool": "echo_tool", "ok": True, "score": 0.9,
                            "note": "did the job"}],
             "context": "wiring up an echo round-trip"},
            rt, caller_id="child",
        )
        assert out == {"attested": 1, "skipped": []}
        assert len(sunk) == 1
        ev = sunk[0]
        child_addr = rt.get_agent("child").identity.address
        assert ev["kind"] == "tool_used"
        assert ev["attested"] is True
        assert ev["author_agent"] == child_addr
        assert ev["tool_author"] == child_addr
        assert ev["manifest_digest"] == res["digest"]
        assert ev["score"] == 0.9
        assert ev["fee_atn"] == 0.0
        assert ev["problem_coords"] == [0.1, 0.2, 0.3]
        assert ev["review_digest"]  # note was blob-stored

    @pytest.mark.asyncio
    async def test_out_of_lineage_judgment_skipped(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)  # author = child
        store = rt.tool_store
        store._embedder = _fake_embedder

        sunk: list[dict] = []
        store.event_sink = sunk.append

        out = await execute_tool(
            "attest_tools",
            {"judgments": [{"tool": "echo_tool", "ok": True},
                           {"tool": "nonexistent", "ok": True}],
             "context": "sibling tries to attest"},
            rt, caller_id="sibling",  # out of child's lineage
        )
        assert out["attested"] == 0
        errs = {s["tool"]: s["error"] for s in out["skipped"]}
        assert "lineage" in errs["echo_tool"]
        assert "not found" in errs["nonexistent"]
        assert sunk == []  # nothing emitted for a wholly-skipped batch

    @pytest.mark.asyncio
    async def test_note_text_lands_in_blob_store(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)
        store = rt.tool_store
        store._embedder = _fake_embedder

        sunk: list[dict] = []
        store.event_sink = sunk.append

        await execute_tool(
            "attest_tools",
            {"judgments": [{"tool": "echo_tool", "ok": False, "score": 0.1,
                            "note": "flaky on empty input"}],
             "context": "stress-testing echo"},
            rt, caller_id="parent",
        )
        review_digest = sunk[0]["review_digest"]
        blob = store._blob_store().get_json(review_digest)
        assert blob["kind"] == "tool_review"
        assert blob["note"] == "flaky on empty input"
        assert blob["caller"] == "parent"
        assert blob["context"] == "stress-testing echo"

    @pytest.mark.asyncio
    async def test_attestation_summary_aggregates(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        store = rt.tool_store
        store._embedder = _fake_embedder

        store.attest_usage(
            "child", [{"tool": "echo_tool", "ok": True, "score": 0.8}],
            "work one")
        store.attest_usage(
            "child", [{"tool": "echo_tool", "ok": False, "score": 0.4}],
            "work two")

        summary = store.attestation_summary()
        entry = summary[res["digest"]]
        assert entry["attested_count"] == 2
        assert entry["ok_count"] == 1
        assert entry["avg_score"] == pytest.approx(0.6)
        assert entry["last_ts"] > 0

    @pytest.mark.asyncio
    async def test_mechanical_receipts_unaffected(self, tmp_path):
        """Mechanical (per-call) receipts carry NO attested field — only
        the cognitive path sets it."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)
        store = rt.tool_store
        store._embedder = _fake_embedder

        sunk: list[dict] = []
        store.event_sink = sunk.append

        await rt.tool_registry.call_tool("echo_tool", {"x": "1"}, caller_id="child")
        assert len(sunk) == 1
        assert "attested" not in sunk[0]

    @pytest.mark.asyncio
    async def test_seq_continuity_across_receipt_and_attestation(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)
        store = rt.tool_store
        store._embedder = _fake_embedder

        sunk: list[dict] = []
        store.event_sink = sunk.append

        await rt.tool_registry.call_tool("echo_tool", {"x": "1"}, caller_id="child")
        store.attest_usage(
            "child", [{"tool": "echo_tool", "ok": True}], "some work")
        await rt.tool_registry.call_tool("echo_tool", {"x": "2"}, caller_id="child")

        seqs = [ev["seq"] for ev in sunk]
        assert seqs == [1, 2, 3]  # unique + monotonic across both tiers

        # Seq counter spans both tiers on reload — no collision after restart.
        reloaded = ToolStore(rt, rt._config.data_dir / "tools")
        assert reloaded._receipt_seq == 3

    @pytest.mark.asyncio
    async def test_degrades_without_embedder(self, tmp_path):
        """If the embedder is unavailable, problem_coords is [] and the
        attestation still records — never crashes the agent."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)
        store = rt.tool_store
        store._embedder = lambda text: (_ for _ in ()).throw(RuntimeError("no torch"))

        sunk: list[dict] = []
        store.event_sink = sunk.append

        out = store.attest_usage(
            "child", [{"tool": "echo_tool", "ok": True}], "work item")
        assert out["attested"] == 1
        assert sunk[0]["problem_coords"] == []


# ---------------------------------------------------------------------------
# Composition — tools calling tools (docs/tool_substrate.md — Composition)
# ---------------------------------------------------------------------------


# A composite that reads its args from the FIRST stdin line, calls a
# declared dependency by name via a {"call": ...} frame, reads the result
# line back, and emits {"return": ...} combining both. Keeps stdin open by
# using readline() (the sandbox contract) rather than json.load(sys.stdin).
COMPOSITE_CODE = (
    "import sys, json\n"
    "args = json.loads(sys.stdin.readline())\n"
    "sys.stdout.write(json.dumps("
    "{'call': args['dep'], 'args': {'x': args['x']}}) + '\\n')\n"
    "sys.stdout.flush()\n"
    "reply = json.loads(sys.stdin.readline())\n"
    "sys.stdout.write(json.dumps("
    "{'return': {'wrapped': reply, 'seen': args['x']}}) + '\\n')\n"
    "sys.stdout.flush()\n"
)

# A composite that tries to call a tool it did NOT declare, gets an error
# frame back, then returns anyway (proving reject-and-continue).
UNDECLARED_COMPOSITE_CODE = (
    "import sys, json\n"
    "args = json.loads(sys.stdin.readline())\n"
    "sys.stdout.write(json.dumps("
    "{'call': args['dep'], 'args': {'x': 'nope'}}) + '\\n')\n"
    "sys.stdout.flush()\n"
    "reply = json.loads(sys.stdin.readline())\n"
    "sys.stdout.write(json.dumps({'return': {'dep_reply': reply}}) + '\\n')\n"
    "sys.stdout.flush()\n"
)


async def _register_composite(rt, dep_digest, caller_id="child",
                              name="composite_tool", code=COMPOSITE_CODE,
                              declare=True):
    return await execute_tool(
        "register_tool",
        {"name": name, "description": "Calls a dep then combines.",
         "input_schema": {
             "type": "object",
             "properties": {"x": {"type": "string"},
                            "dep": {"type": "string"}}},
         "code": code,
         "dependencies": [dep_digest] if declare else []},
        rt, caller_id=caller_id,
    )


class TestComposition:
    @pytest.mark.asyncio
    async def test_composite_calls_dep_and_returns(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        dep = await _register_echo(rt)  # author = child
        comp = await _register_composite(rt, dep["digest"])
        assert "error" not in comp

        sunk: list[dict] = []
        rt.tool_store.event_sink = sunk.append

        out = await rt.tool_registry.call_tool(
            "composite_tool", {"x": "hi", "dep": "echo_tool"},
            caller_id="child")

        # End-to-end: the dep echoed, the composite combined.
        assert out == {"result": {"wrapped": {"result": {"echo": "hi"}},
                                  "seen": "hi"}}

        # TWO mechanical receipts: the nested dep call tagged via the
        # composite digest, the top-level composite call not tagged.
        assert len(sunk) == 2
        by_digest = {ev["manifest_digest"]: ev for ev in sunk}
        dep_ev = by_digest[dep["digest"]]
        comp_ev = by_digest[comp["digest"]]
        assert dep_ev["via"] == comp["digest"]
        assert "via" not in comp_ev

        # Same in the persisted ledger.
        rows = list(rt.tool_store._iter_receipts())
        via_rows = [r for r in rows if r.get("via")]
        assert len(via_rows) == 1
        assert via_rows[0]["manifest_digest"] == dep["digest"]
        assert via_rows[0]["via"] == comp["digest"]

    @pytest.mark.asyncio
    async def test_undeclared_call_gets_error_frame_but_composite_returns(
            self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        dep = await _register_echo(rt)  # author = child
        # Register a SECOND (undeclared) tool the composite will try to hit.
        other = await _register_echo(rt, name="other_tool")
        # Composite declares only `dep`, but its code calls `other_tool`.
        comp = await _register_composite(
            rt, dep["digest"], code=UNDECLARED_COMPOSITE_CODE)
        assert "error" not in comp

        out = await rt.tool_registry.call_tool(
            "composite_tool", {"x": "hi", "dep": "other_tool"},
            caller_id="child")
        # The dep call was rejected with an error frame; the composite
        # received it and still returned.
        assert out["result"]["dep_reply"] == {"error": "undeclared dependency"}

    @pytest.mark.asyncio
    async def test_register_rejects_nonexistent_dependency(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        fake_digest = "a" * 64
        comp = await _register_composite(rt, fake_digest)
        assert "error" in comp
        assert "not registered" in comp["error"]

    @pytest.mark.asyncio
    async def test_nested_call_scoped_under_original_caller(self, tmp_path):
        """The dep runs under the ORIGINAL caller's authority. A sibling
        cannot reach child's dep even through a composite it may call."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        dep = await _register_echo(rt)  # author = child, sibling out of lineage
        comp = await _register_composite(rt, dep["digest"])
        # Owner grants the composite (only) to the sibling.
        rt.tool_store.grant(comp["digest"], "sibling")

        out = await rt.tool_registry.call_tool(
            "composite_tool", {"x": "hi", "dep": "echo_tool"},
            caller_id="sibling")
        # Sibling may run the composite, but the nested dep call is scoped
        # to the sibling (original caller) — who lacks access to the dep.
        assert out["result"]["wrapped"] == {
            "error": "caller not authorized for dependency"}

    @pytest.mark.asyncio
    async def test_legacy_no_deps_tool_still_sealed(self, tmp_path):
        """A tool WITHOUT dependencies keeps the sealed json.load(stdin)
        contract — must still round-trip."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        await _register_echo(rt)
        out = await rt.tool_registry.call_tool(
            "echo_tool", {"x": "sealed"}, caller_id="child")
        assert out == {"result": {"echo": "sealed"}}
