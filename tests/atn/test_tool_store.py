"""Tool substrate ATN surface — register_tool + author-lineage scoping.

Covers manifest registration (pinned/attested, persistence, reload),
author derivation from caller, name collision guards, the two-point
scoping contract (listing visibility AND call-time enforcement), owner
grants/revokes, and pinned-code subprocess execution end-to-end.
Design: docs/tool_substrate.md.
"""
from __future__ import annotations

import json
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

        # register_tool NEVER publishes for agents (user, 2026-07-05):
        # publishing is the separate publish_tool capability.
        res_rej = await execute_tool(
            "register_tool",
            {"name": "public_echo", "description": "Echo.", "input_schema": SCHEMA,
             "code": ECHO_CODE, "publish": True},
            rt, caller_id="child")
        assert "publish_tool" in res_rej["error"]

        # The author publishes its own tool via publish_tool; the sink
        # receives the CONSENSUS author (0x address), not the local id.
        child_addr = rt.get_agent("child").identity.address
        pub = await execute_tool(
            "publish_tool", {"digest": res["digest"]}, rt, caller_id="child")
        assert pub["published"] is True
        assert pushed == [("echo_tool", child_addr)]

        # A non-author agent cannot publish someone else's tool.
        res2 = await execute_tool(
            "register_tool",
            {"name": "sibling_tool", "description": "d", "input_schema": SCHEMA,
             "code": ECHO_CODE + "# sib\n"},
            rt, caller_id="sibling")
        denied = await execute_tool(
            "publish_tool", {"digest": res2["digest"]}, rt, caller_id="child")
        assert "you authored" in denied["error"]

        # Owner can flip publish state directly (WS surface).
        assert rt.tool_store.set_published(res2["digest"], True)
        assert pushed[-1][0] == "sibling_tool"
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


class TestAdoption:
    """Adoption rail (docs/tool_substrate.md — Adoption): agent proposes,
    owner approves per tool, foreign code runs contained."""

    async def _publish_on_author_daemon(self, tmp_path, code=ECHO_CODE,
                                        capabilities=None):
        """A separate 'network' daemon that authored + published a tool;
        returns (runtime, digest, blob_store) for the adopter to fetch
        from."""
        rt = _make_runtime(tmp_path / "author_daemon")
        await _register_agent(rt, "foreign-author")
        args = {"name": "net_tool", "description": "A network-published tool.",
                "input_schema": SCHEMA, "code": code}
        if capabilities:
            args["capabilities"] = capabilities
        res = await execute_tool("register_tool", args, rt,
                                 caller_id="foreign-author")
        assert "error" not in res, res
        return rt, res["digest"], rt.tool_store._blob_store()

    async def _adopter(self, tmp_path, source_blobs):
        rt = _make_runtime(tmp_path / "adopter_daemon")
        await _register_agent(rt, "worker")

        async def _fetch(digest: str) -> bytes | None:
            return source_blobs.get_bytes(digest)
        rt.tool_store.blob_fetcher = _fetch
        return rt

    @pytest.mark.asyncio
    async def test_propose_approve_call_roundtrip(self, tmp_path):
        _, digest, blobs = await self._publish_on_author_daemon(tmp_path)
        rt = await self._adopter(tmp_path, blobs)
        store = rt.tool_store

        out = await execute_tool(
            "adopt_tool", {"digest": digest, "reason": "need echo for X"},
            rt, caller_id="worker")
        assert out["status"] == "pending"
        assert store.get(digest) is None            # NOT installed yet

        pending = store.list_adoption_proposals("pending")
        assert len(pending) == 1
        assert pending[0]["proposed_by"] == "worker"
        assert pending[0]["provenance"]["signed"] is True

        approved = await store.approve_adoption(digest)
        assert approved["origin"] == "adopted"
        record = store.get(digest)
        assert record is not None
        assert record.origin == "adopted"
        # Original author preserved (their mint keeps flowing);
        # scoping keys on the adopter.
        assert record.author != "worker"
        assert record.local_author == "worker"
        assert store.allowed("worker", record)

        result = await store.call(record, {"x": "hi"}, caller_id="worker")
        assert result == {"result": {"echo": "hi"}}

    @pytest.mark.asyncio
    async def test_reason_required_and_reject_flow(self, tmp_path):
        _, digest, blobs = await self._publish_on_author_daemon(tmp_path)
        rt = await self._adopter(tmp_path, blobs)
        out = await execute_tool(
            "adopt_tool", {"digest": digest}, rt, caller_id="worker")
        assert "reason" in out["error"]

        await execute_tool(
            "adopt_tool", {"digest": digest, "reason": "want it"},
            rt, caller_id="worker")
        rej = rt.tool_store.reject_adoption(digest, reason="not needed")
        assert rej["status"] == "rejected"
        assert rt.tool_store.get(digest) is None
        # A rejected proposal doesn't block a later re-proposal cycle
        # via approve (status pending required).
        res = await rt.tool_store.approve_adoption(digest)
        assert "error" in res

    @pytest.mark.asyncio
    async def test_adopted_cannot_be_republished(self, tmp_path):
        _, digest, blobs = await self._publish_on_author_daemon(tmp_path)
        rt = await self._adopter(tmp_path, blobs)
        store = rt.tool_store
        await store.propose_adoption("worker", digest, reason="r")
        await store.approve_adoption(digest)
        out = await execute_tool(
            "publish_tool", {"digest": digest}, rt, caller_id="worker")
        assert "adopted" in out["error"]
        assert store.set_published(digest, True) is False

    @pytest.mark.asyncio
    async def test_undeclared_network_hard_fails(self, tmp_path):
        evil = (
            "import sys, json, socket\n"
            "json.load(sys.stdin)\n"
            "s = socket.socket()\n"
            "print(json.dumps({'ok': True}))\n"
        )
        _, digest, blobs = await self._publish_on_author_daemon(
            tmp_path, code=evil)
        rt = await self._adopter(tmp_path, blobs)
        store = rt.tool_store
        await store.propose_adoption("worker", digest, reason="r")
        await store.approve_adoption(digest)
        record = store.get(digest)
        result = await store.call(record, {"x": "1"}, caller_id="worker")
        assert "error" in result
        assert "undeclared capability: net" in result["error"]

    @pytest.mark.asyncio
    async def test_declared_network_is_allowed(self, tmp_path):
        code = (
            "import sys, json, socket\n"
            "json.load(sys.stdin)\n"
            "s = socket.socket(); s.close()\n"
            "print(json.dumps({'made_socket': True}))\n"
        )
        _, digest, blobs = await self._publish_on_author_daemon(
            tmp_path, code=code, capabilities={"net": True})
        rt = await self._adopter(tmp_path, blobs)
        store = rt.tool_store
        await store.propose_adoption("worker", digest, reason="r")
        await store.approve_adoption(digest)
        record = store.get(digest)
        result = await store.call(record, {"x": "1"}, caller_id="worker")
        assert result == {"result": {"made_socket": True}}

    @pytest.mark.asyncio
    async def test_env_scrubbed_for_adopted_code(self, tmp_path,
                                                 monkeypatch):
        monkeypatch.setenv("ATN_TEST_SECRET", "hunter2")
        code = (
            "import sys, json, os\n"
            "json.load(sys.stdin)\n"
            "print(json.dumps({'secret': os.environ.get('ATN_TEST_SECRET'),"
            " 'policy': os.environ.get('ATN_TOOL_POLICY')}))\n"
        )
        _, digest, blobs = await self._publish_on_author_daemon(
            tmp_path, code=code)
        rt = await self._adopter(tmp_path, blobs)
        store = rt.tool_store
        await store.propose_adoption("worker", digest, reason="r")
        await store.approve_adoption(digest)
        record = store.get(digest)
        result = await store.call(record, {"x": "1"}, caller_id="worker")
        # Secrets never reach foreign code; the policy var is popped
        # before the tool runs.
        assert result == {"result": {"secret": None, "policy": None}}

    @pytest.mark.asyncio
    async def test_declared_env_passes_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATN_TEST_TOKEN", "tok123")
        code = (
            "import sys, json, os\n"
            "json.load(sys.stdin)\n"
            "print(json.dumps({'tok': os.environ.get('ATN_TEST_TOKEN')}))\n"
        )
        _, digest, blobs = await self._publish_on_author_daemon(
            tmp_path, code=code, capabilities={"env": ["ATN_TEST_TOKEN"]})
        rt = await self._adopter(tmp_path, blobs)
        store = rt.tool_store
        await store.propose_adoption("worker", digest, reason="r")
        await store.approve_adoption(digest)
        record = store.get(digest)
        result = await store.call(record, {"x": "1"}, caller_id="worker")
        assert result == {"result": {"tok": "tok123"}}

    @pytest.mark.asyncio
    async def test_authored_tools_run_unguarded(self, tmp_path,
                                                monkeypatch):
        """The guard is adoption-specific: an agent's OWN tool still
        sees the parent environment (author judged their own code)."""
        monkeypatch.setenv("ATN_TEST_SECRET", "mine")
        rt = _make_runtime(tmp_path)
        await _family(rt)
        code = (
            "import sys, json, os\n"
            "json.load(sys.stdin)\n"
            "print(json.dumps({'secret': os.environ.get('ATN_TEST_SECRET')}))\n"
        )
        res = await execute_tool(
            "register_tool",
            {"name": "own_tool", "description": "Own tool.",
             "input_schema": SCHEMA, "code": code},
            rt, caller_id="child")
        record = rt.tool_store.get(res["digest"])
        result = await rt.tool_store.call(record, {}, caller_id="child")
        assert result == {"result": {"secret": "mine"}}


class TestVetting:
    """Validator rail (docs/tool_substrate.md — Vetting section):
    inspect → attest, self-vet rejection, report requirement, and the
    network-fetch path for foreign manifests."""

    @pytest.mark.asyncio
    async def test_inspect_returns_manifest_and_code(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)  # author = child
        out = await execute_tool(
            "vet_tool", {"digest": res["digest"]}, rt, caller_id="sibling",
        )
        assert out["manifest"]["name"] == "echo_tool"
        assert out["code"] == ECHO_CODE
        assert "verdict" not in out

    @pytest.mark.asyncio
    async def test_vet_pass_emits_vet_event(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        store = rt.tool_store
        sunk: list[dict] = []
        store.event_sink = sunk.append

        out = await execute_tool(
            "vet_tool",
            {"digest": res["digest"], "verdict": "pass",
             "report": "read the code; echoes x, no fs/net access"},
            rt, caller_id="sibling",
        )
        assert out["verdict"] == "pass"
        assert len(sunk) == 1
        ev = sunk[0]
        sibling_addr = rt.get_agent("sibling").identity.address
        assert ev["kind"] == "tool_used"
        assert ev["vet"] is True
        assert ev["ok"] is True
        assert ev["author_agent"] == sibling_addr
        assert ev.get("attested") is None       # a vet is NOT usage
        report = store._blob_store().get_json(ev["review_digest"])
        assert report["kind"] == "tool_vet_report"
        assert report["validator"] == sibling_addr
        # Local ledgers: vet_summary sees it, attestation_summary must not.
        assert store.vet_summary()[res["digest"]]["pass_count"] == 1
        assert res["digest"] not in store.attestation_summary()

    @pytest.mark.asyncio
    async def test_self_vet_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)  # author = child
        out = await execute_tool(
            "vet_tool",
            {"digest": res["digest"], "verdict": "pass", "report": "lgtm"},
            rt, caller_id="child",
        )
        assert "your own tool" in out["error"]

    @pytest.mark.asyncio
    async def test_verdict_requires_report(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        out = await execute_tool(
            "vet_tool", {"digest": res["digest"], "verdict": "pass"},
            rt, caller_id="sibling",
        )
        assert "report" in out["error"]

    @pytest.mark.asyncio
    async def test_attested_class_not_vettable(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await execute_tool(
            "register_tool",
            {"name": "cal_lookup", "description": "Calendar lookup.",
             "input_schema": SCHEMA, "connector_id": "gcal"},
            rt, caller_id="child",
        )
        out = await execute_tool(
            "vet_tool", {"digest": res["digest"]}, rt, caller_id="sibling",
        )
        assert "pinned" in out["error"]

    @pytest.mark.asyncio
    async def test_foreign_manifest_fetched_and_vetted(self, tmp_path):
        """A validator daemon that never registered the tool pulls the
        manifest + code by digest over the blob fetcher (digest-verified,
        cached) and vets it."""
        rt_author = _make_runtime(tmp_path / "a")
        await _family(rt_author)
        res = await _register_echo(rt_author)
        author_blobs = rt_author.tool_store._blob_store()
        code_digest = res["manifest"] if isinstance(res.get("manifest"), str) \
            else rt_author.tool_store.get(res["digest"]).manifest["code_digest"]

        rt_val = _make_runtime(tmp_path / "b")
        await _register_agent(rt_val, "validator")
        store_val = rt_val.tool_store

        async def _fetch(digest: str) -> bytes | None:
            return author_blobs.get_bytes(digest)
        store_val.blob_fetcher = _fetch
        sunk: list[dict] = []
        store_val.event_sink = sunk.append

        inspect = await store_val.vet_tool("validator", res["digest"])
        assert inspect["code"] == ECHO_CODE

        out = await store_val.vet_tool(
            "validator", res["digest"], verdict="pass",
            report="fetched by digest; code matches manifest",
        )
        assert out["verdict"] == "pass"
        assert sunk[0]["vet"] is True
        assert sunk[0]["manifest_digest"] == res["digest"]
        # The fetched blobs are now cached locally (content-addressed).
        assert store_val._blob_store().get_bytes(code_digest) is not None

    @pytest.mark.asyncio
    async def test_corrupt_fetch_rejected(self, tmp_path):
        """A peer returning bytes that don't hash to the requested
        digest is ignored — content addressing IS the auth."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "validator")
        store = rt.tool_store

        async def _evil_fetch(digest: str) -> bytes:
            return b'{"kind": "tool_manifest", "name": "evil"}'
        store.blob_fetcher = _evil_fetch

        out = await store.vet_tool("validator", "cd" * 32)
        assert "cannot resolve" in out["error"]


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


# ---------------------------------------------------------------------------
# Evidence-replay CON (docs/tool_substrate.md — Evidence section)
# ---------------------------------------------------------------------------


# A tool that errors when x is missing/empty, else echoes it. Gives us a
# reproducible failing invocation (empty x) AND a wrong-answer path.
BUGGY_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "x = args.get('x')\n"
    "if not x:\n"
    "    raise ValueError('x is required')\n"
    "print(json.dumps({'echo': x}))\n"
)


def _result_digest(payload):
    import hashlib
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class TestEvidenceReplay:
    async def _register_buggy(self, rt, caller_id="child"):
        return await execute_tool(
            "register_tool",
            {"name": "buggy_tool", "description": "Errors on empty x.",
             "input_schema": SCHEMA, "code": BUGGY_CODE},
            rt, caller_id=caller_id)

    @pytest.mark.asyncio
    async def test_replay_confirms_error_evidence(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await self._register_buggy(rt)
        evidence = {"args_json": {"x": ""}, "expected_error": "x is required"}
        out = await rt.tool_store.replay_evidence(evidence, res["digest"])
        assert out["confirmed"] is True
        assert out["kind"] == "error"
        assert "x is required" in out["observed_error"]

    @pytest.mark.asyncio
    async def test_replay_refutes_error_that_does_not_reproduce(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await self._register_buggy(rt)
        # CON claims x='ok' errors — but it doesn't.
        evidence = {"args_json": {"x": "ok"}, "expected_error": "boom"}
        out = await rt.tool_store.replay_evidence(evidence, res["digest"])
        assert out["confirmed"] is False
        assert out["kind"] == "result"

    @pytest.mark.asyncio
    async def test_replay_confirms_wrong_answer_evidence(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)   # echoes x
        # CON claims echo('hi') should produce {"echo": "HELLO"} (the
        # "correct" contract) but the tool returns {"echo": "hi"}.
        correct = _result_digest({"echo": "HELLO"})
        actual = _result_digest({"echo": "hi"})
        evidence = {"args_json": {"x": "hi"}, "expected_digest": correct,
                    "actual_digest": actual}
        out = await rt.tool_store.replay_evidence(evidence, res["digest"])
        assert out["confirmed"] is True          # observed != correct
        assert out["reproduced_actual"] is True  # matches the recorded actual
        assert out["observed_digest"] == actual

    @pytest.mark.asyncio
    async def test_replay_refutes_when_output_matches_expected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        correct = _result_digest({"echo": "hi"})   # the tool IS correct
        evidence = {"args_json": {"x": "hi"}, "expected_digest": correct}
        out = await rt.tool_store.replay_evidence(evidence, res["digest"])
        assert out["confirmed"] is False

    @pytest.mark.asyncio
    async def test_args_json_accepts_string(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await self._register_buggy(rt)
        evidence = {"args_json": json.dumps({"x": ""}),
                    "expected_error": "required"}
        out = await rt.tool_store.replay_evidence(evidence, res["digest"])
        assert out["confirmed"] is True

    @pytest.mark.asyncio
    async def test_evidence_requires_a_criterion(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await _register_echo(rt)
        out = await rt.tool_store.replay_evidence(
            {"args_json": {"x": "hi"}}, res["digest"])
        assert out["confirmed"] is False
        assert "expected_error or expected_digest" in out["error"]

    @pytest.mark.asyncio
    async def test_unresolvable_tool(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "worker")
        out = await rt.tool_store.replay_evidence(
            {"args_json": {}, "expected_error": "x"}, "ab" * 32)
        assert out["confirmed"] is False
        assert "resolvable" in out["error"]

    @pytest.mark.asyncio
    async def test_check_evidence_verify_then_support(self, tmp_path):
        """Confirmed replay posts a PRO support sprout under the CON via
        the wired support_sink; a refuted one posts nothing."""
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await self._register_buggy(rt)

        posted: list = []

        def fake_support_sink(con_node_id, claim):
            posted.append((con_node_id, claim))
            return {"support_node_id": "pro_x", "target_node_id": con_node_id}

        rt.tool_store.support_sink = fake_support_sink

        # Confirmed evidence → support posted.
        confirmed = await execute_tool(
            "check_evidence",
            {"manifest_digest": res["digest"],
             "evidence": {"args_json": {"x": ""}, "expected_error": "req"},
             "con_node_id": "con_abc"},
            rt, caller_id="sibling")
        assert confirmed["confirmed"] is True
        assert confirmed["supported"]["support_node_id"] == "pro_x"
        assert posted == [("con_abc", posted[0][1])]

        # Refuted evidence → nothing posted.
        posted.clear()
        refuted = await execute_tool(
            "check_evidence",
            {"manifest_digest": res["digest"],
             "evidence": {"args_json": {"x": "ok"}, "expected_error": "req"},
             "con_node_id": "con_def"},
            rt, caller_id="sibling")
        assert refuted["confirmed"] is False
        assert refuted["supported"] is None
        assert posted == []

    @pytest.mark.asyncio
    async def test_check_evidence_confirmed_without_sink_posts_nothing(
            self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _family(rt)
        res = await self._register_buggy(rt)
        # No support_sink wired, no con_node_id: replay only.
        out = await rt.tool_store.check_evidence(
            "sibling", res["digest"],
            {"args_json": {"x": ""}, "expected_error": "req"})
        assert out["confirmed"] is True
        assert out["supported"] is None
