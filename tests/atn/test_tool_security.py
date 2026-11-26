"""Tool rail (docs/tool_substrate.md, Decisions 2026-10-13 and 2026-11-09):
novelty claims, same-author successors, stake on publish, no run gate (an
adopted tool runs bonded or not), and search's bond + rating context.

No chain: ToolBond is a fake client / checker-backed gate throughout.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from atn.agent_tools import execute_tool
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime.tool_bond import DEV_OVERRIDE_ENV, ToolBondGate
from atn.tool_security import (
    RunOptIns, annotate_probe_rows, publish_with_stake, rating_for,
)
from nodes.common.library.tool_manifest import (
    novelty_errors, publish_novelty_errors, states_nothing_comparable,
    validate_manifest,
)



SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}
ECHO = "import json,sys\nprint(json.dumps({'echo': json.load(sys.stdin).get('x')}))\n"
NOVEL = {"statement": "No comparable tool exists: echoes x.", "prior_art": []}
D1, D2, D3 = "a1" * 32, "b2" * 32, "c3" * 32
TOKEN, TOKEN18 = "0x" + "c6" * 20, "0x" + "d8" * 20   # index tokens, 6 and 18 decimals


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("ATN_TOOL_BOND_ADDRESS", raising=False)
    # The registry seed names a live RepToken; never read it from tests.
    monkeypatch.setattr("atn.tool_security._rep_reader", lambda r: None)


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


async def _agent(rt, agent_id, *, on_chain=True):
    defn = AgentDefinition(id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
                           system_prompt="x", cognitive_model="claude-sonnet-5")
    await rt.register_agent(defn)
    defn = rt.get_agent(agent_id)
    if on_chain:
        defn.identity.registered_on_chain = True
    return defn


async def _register(rt, caller, name="echo_tool", **extra):
    args = {"name": name, "description": "Echo x.", "input_schema": SCHEMA,
            "code": ECHO, **extra}
    res = await execute_tool("register_tool", args, rt, caller_id=caller)
    assert "error" not in res, res
    return res


class FakeClient:
    """ToolBondClient stand-in: records stakes, serves canned status."""

    def __init__(self, statuses=None, *, available=True, ok=True):
        self.statuses = statuses or {}
        self._available = available
        self.ok = ok
        self.bonds: list[tuple[str, str]] = []
        self.bond_args: list[dict] = []
        self.status_calls: list[str] = []

    def available(self):
        return self._available

    async def register_and_bond(self, key, digest, amount=None, token=None):
        self.bonds.append((key, digest))
        self.bond_args.append({"amount": amount, "token": token})
        if not self.ok:
            return {"success": False, "error": "would revert: NotAuthor",
                    "step": "bond", "txs": ["0xreg"]}
        return {"success": True, "tx_hash": "0xbond",
                "txs": ["0xreg", "0xapprove", "0xbond"], "amount": amount or 100,
                "token": token or TOKEN, "value": 100}

    async def status(self, digest):
        self.status_calls.append(digest)
        st = self.statuses.get(digest)
        if isinstance(st, Exception):
            raise st
        return st


def _st(bonded=True):
    return {"bonded": bonded, "stake": 100, "author": AUTH,
            "unbond_requested_at": 0, "unbond_epoch": 0, "pending_rounds": 0}


def _rating(malware=None, correctness=None, mass=0.3):
    """A rating as the close carries it (federated_reconcile.tool_rating)."""
    return {"correctness": correctness, "simplicity": None, "novelty": None,
            "malware": malware, "axes": {}, "value": None, "mass": mass}


def _ratings(rt, table):
    rt.tool_store.rating_provider = table.get


# ---------------------------------------------------------------------------
# 1. Novelty claim (manifest structure)
# ---------------------------------------------------------------------------

class TestNoveltyStructure:
    def test_wellformed(self):
        assert novelty_errors({"statement": "2x faster than X", "prior_art": [D1]}) == []
        assert novelty_errors(NOVEL) == []

    def test_empty_prior_art_needs_explicit_statement(self):
        errs = novelty_errors({"statement": "It echoes.", "prior_art": []})
        assert errs and "nothing comparable" in errs[0]
        for s in ("Nothing like this exists yet.", "Comparable tools: none.",
                  "Similar tools do not exist."):
            assert states_nothing_comparable(s), s

    @pytest.mark.parametrize("bad", [
        "x", {"prior_art": []}, {"statement": "  ", "prior_art": []},
        {"statement": "s", "prior_art": ["nothex"]},
        {"statement": "s", "prior_art": [D1, D1]},
        {"statement": "s", "prior_art": [D1], "extra": 1},
    ])
    def test_malformed(self, bad):
        assert novelty_errors(bad)

    def test_optional_in_manifest_required_to_publish(self):
        base = {"kind": "tool_manifest", "name": "t", "description": "d",
                "input_schema": SCHEMA, "author": "0x" + "1" * 40,
                "trust_class": "pinned", "code_digest": D1}
        assert validate_manifest(base) == []
        assert publish_novelty_errors(base)
        assert validate_manifest({**base, "novelty": {"statement": ""}})
        assert publish_novelty_errors({**base, "novelty": NOVEL}) == []


# ---------------------------------------------------------------------------
# 2. Registration: novelty in the digest, successor rule
# ---------------------------------------------------------------------------

class TestRegistration:
    async def test_novelty_is_part_of_identity(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        plain = await _register(rt, "a")
        claimed = await _register(rt, "a", novelty={**NOVEL, "prior_art": [D1.upper()]})
        assert claimed["digest"] != plain["digest"]
        m = rt.tool_store.get(claimed["digest"]).manifest
        assert m["novelty"]["prior_art"] == [D1]          # normalized
        again = await _register(rt, "a", novelty={**NOVEL, "prior_art": [D1]})
        assert again["digest"] == claimed["digest"]        # idempotent

    async def test_bad_novelty_rejected(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        res = await execute_tool("register_tool", {
            "name": "t1", "description": "d", "input_schema": SCHEMA, "code": ECHO,
            "novelty": {"statement": "echo", "prior_art": []}}, rt, caller_id="a")
        assert "nothing comparable" in res["error"]

    async def test_successor_same_author_only(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        await _agent(rt, "b")
        v1 = await _register(rt, "a")
        v2 = await _register(rt, "a", code=ECHO + "# v2\n", version_of=v1["digest"])
        assert rt.tool_store.get(v2["digest"]).manifest["version_of"] == v1["digest"]
        res = await execute_tool("register_tool", {
            "name": "fork", "description": "d", "input_schema": SCHEMA,
            "code": ECHO + "# fork\n", "version_of": v1["digest"]}, rt, caller_id="b")
        assert "own tool" in res["error"]
        res = await execute_tool("register_tool", {
            "name": "orphan", "description": "d", "input_schema": SCHEMA,
            "code": ECHO, "version_of": D3}, rt, caller_id="a")
        assert "not a tool manifest known" in res["error"]


# ---------------------------------------------------------------------------
# 3. Publish = novelty + stake
# ---------------------------------------------------------------------------

class TestPublish:
    async def test_stakes_with_author_agent_key(self, tmp_path):
        rt = _make_runtime(tmp_path)
        defn = await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        pushed = []
        rt.tool_store.manifest_sink = lambda m, a: pushed.append(a)
        res = await _register(rt, "a", novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="a")
        assert out["published"] is True, out
        assert out["stake"]["txs"] == ["0xreg", "0xapprove", "0xbond"]
        assert client.bonds == [(rt.registry.get_agent_key("a"), res["digest"])]
        assert pushed == [defn.identity.address]

    async def test_missing_novelty_refused_before_chain(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        res = await _register(rt, "a")
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="a")
        assert "novelty claim" in out["error"]
        assert client.bonds == [] and not rt.tool_store.get(res["digest"]).published

    async def test_novelty_at_publish_restamps(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        res = await _register(rt, "a")
        rt.tool_store.grant(res["digest"], "someone")
        out = await execute_tool("publish_tool", {"digest": res["digest"],
                                                  "novelty": NOVEL}, rt, caller_id="a")
        assert out["published"] is True and out["digest"] != res["digest"]
        assert out["replaces_private"] == res["digest"]
        assert rt.tool_store.get(res["digest"]) is None
        new = rt.tool_store.get(out["digest"])
        assert new.manifest["novelty"] == NOVEL and "someone" in new.grants
        assert client.bonds[0][1] == out["digest"]

    async def test_restamp_keeps_a_referenced_original(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        res = await _register(rt, "a")
        await _register(rt, "a", name="composite", code=ECHO + "# c\n",
                        dependencies=[res["digest"]])
        stamped = rt.tool_store.restamp_novelty(res["digest"], NOVEL)
        assert stamped["replaced"] == ""
        assert rt.tool_store.get(res["digest"]) is not None

    async def test_composite_needs_published_dependencies(self, tmp_path):
        """#38: a composite whose own dependency is private is refused,
        naming it; publishing the dependency first unblocks it."""
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        dep = await _register(rt, "a", name="dep_tool", novelty=NOVEL)
        comp = await _register(rt, "a", name="composite", code=ECHO + "# c\n",
                               dependencies=[dep["digest"]], novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": comp["digest"]}, rt,
                                 caller_id="a")
        assert "not published" in out["error"] and "dep_tool" in out["error"]
        assert not rt.tool_store.get(comp["digest"]).published and client.bonds == []

        out = await execute_tool("publish_tool", {"digest": dep["digest"]}, rt,
                                 caller_id="a")
        assert out["published"] is True and out["digest"] == dep["digest"]
        out = await execute_tool("publish_tool", {"digest": comp["digest"]}, rt,
                                 caller_id="a")
        assert out["published"] is True

    async def test_composite_on_adopted_dependency_publishes(self, tmp_path):
        rt, net = await _adopted(tmp_path)
        rt.tool_bond_client = FakeClient()
        comp = await _register(rt, "worker", name="composite", code=ECHO + "# c\n",
                               dependencies=[net.digest], novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": comp["digest"]}, rt,
                                 caller_id="worker")
        assert out["published"] is True, out

    async def test_fails_closed_without_tool_bond(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = FakeClient(available=False)
        res = await _register(rt, "a", novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="a")
        assert "ToolBond" in out["error"]
        assert not rt.tool_store.get(res["digest"]).published

    async def test_failed_stake_leaves_tool_private(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = FakeClient(ok=False)
        res = await _register(rt, "a", novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="a")
        assert "at bond" in out["error"] and out["txs"] == ["0xreg"]
        assert not rt.tool_store.get(res["digest"]).published

    async def test_the_bond_is_in_the_named_index_token(self, tmp_path):
        """publish_tool passes the token and amount to the bond, and the
        stake reports the token and the value posted."""
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        res = await _register(rt, "a", novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": res["digest"], "token": TOKEN,
                                                  "amount": "250000000"}, rt, caller_id="a")
        assert out["published"] is True, out
        assert client.bond_args == [{"amount": 250_000_000, "token": TOKEN}]
        assert out["stake"]["token"] == TOKEN and out["stake"]["value"] == 100

    async def test_bad_amount_refused_before_chain(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        res = await _register(rt, "a", novelty=NOVEL)
        for bad in ("-5", "0", "ten"):
            out = await execute_tool("publish_tool", {"digest": res["digest"],
                                                      "amount": bad}, rt, caller_id="a")
            assert "amount" in out["error"], out
        assert client.bonds == [] and not rt.tool_store.get(res["digest"]).published

    async def test_no_token_to_choose_names_the_index(self, tmp_path):
        """Several (or no) index tokens held and none named: the bond is
        refused with the index listed, and the tool stays private."""
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        client = FakeClient()

        async def choose(key, digest, amount=None, token=None):
            return {"success": False, "step": "bond", "txs": ["0xreg"],
                    "error": "name the index token to bond in (`token`): you hold several",
                    "index_tokens": [TOKEN, TOKEN18]}
        client.register_and_bond = choose
        rt.tool_bond_client = client
        res = await _register(rt, "a", novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="a")
        assert "name the index token" in out["error"]
        assert out["index_tokens"] == [TOKEN, TOKEN18]
        assert not rt.tool_store.get(res["digest"]).published

    async def test_owner_authored_refused(self, tmp_path):
        rt = _make_runtime(tmp_path)
        rt._config.autonet.owner_wallet = "0x" + "44" * 20
        rt.tool_bond_client = client = FakeClient()
        res = await execute_tool("register_tool", {
            "name": "own", "description": "d", "input_schema": SCHEMA, "code": ECHO,
            "novelty": NOVEL, "publish": True}, rt, caller_id=None)
        assert "owner-authored" in res["error"] and "agent" in res["error"]
        assert res["published"] is False and client.bonds == []

    async def test_unregistered_agent_refused(self, tmp_path):
        rt = _make_runtime(tmp_path)
        rt._config.autonet.owner_wallet = "0x" + "44" * 20
        await _agent(rt, "a", on_chain=False)
        rt.tool_bond_client = client = FakeClient()
        res = await _register(rt, "a", novelty=NOVEL)
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="a")
        assert "not registered on chain" in out["error"] and client.bonds == []
        assert "Autonet app" in out["error"]

    async def test_dev_override_skips_chain(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
        rt = _make_runtime(tmp_path)
        rt._config.autonet.owner_wallet = "0x" + "44" * 20
        rt.tool_bond_client = client = FakeClient(available=False)
        res = await execute_tool("register_tool", {
            "name": "own", "description": "d", "input_schema": SCHEMA, "code": ECHO,
            "novelty": NOVEL, "publish": True}, rt, caller_id=None)
        assert res["published"] is True and res["stake"]["skipped"] == "dev override"
        assert client.bonds == []

    async def test_successor_rechecked_at_publish(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = FakeClient()
        v1 = await _register(rt, "a")
        v2 = await _register(rt, "a", code=ECHO + "#2\n", version_of=v1["digest"],
                             novelty=NOVEL)
        rec = rt.tool_store.get(v2["digest"])
        rec.manifest["author"] = "0x" + "99" * 20   # simulate a foreign-author row
        out = await publish_with_stake(rt, rec)
        assert "own tool" in out["error"]


# ---------------------------------------------------------------------------
# 4. Run gate for network (adopted) tools
# ---------------------------------------------------------------------------

async def _adopted(tmp_path):
    """An adopter daemon holding a tool authored on another daemon."""
    author_rt = _make_runtime(tmp_path / "author")
    await _agent(author_rt, "foreign")
    res = await _register(author_rt, "foreign", name="net_tool")
    blobs = author_rt.tool_store._blob_store()
    rt = _make_runtime(tmp_path / "adopter")
    await _agent(rt, "worker")

    async def _fetch(digest):
        return blobs.get_bytes(digest)
    rt.tool_store.blob_fetcher = _fetch
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: True)
    await rt.tool_store.propose_adoption("worker", res["digest"], reason="need it")
    approved = await rt.tool_store.approve_adoption(res["digest"])
    assert "error" not in approved, approved
    return rt, rt.tool_store.get(res["digest"])


class TestNoRunGate:
    async def test_unbonded_adopted_tool_runs(self, tmp_path):
        """No run gate (Decision 2026-11-09): what runs is the household's
        call; search shows the bond."""
        rt, rec = await _adopted(tmp_path)
        rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: False)
        out = await rt.tool_store.call(rec, {"x": "hi"}, caller_id="worker")
        assert out == {"result": {"echo": "hi"}}

    async def test_optin_is_a_note(self, tmp_path):
        rt, rec = await _adopted(tmp_path)
        from atn.tool_security import run_optins
        run_optins(rt.tool_store).set(rec.digest, True, name=rec.name)
        reloaded = RunOptIns(Path(rt.tool_store._dir) / "run_optins.json")
        assert reloaded.allows(rec.digest)
        run_optins(rt.tool_store).set(rec.digest, False)
        assert not run_optins(rt.tool_store).allows(rec.digest)

    async def test_corrupt_optin_file_reads_empty(self, tmp_path):
        p = tmp_path / "optins.json"
        p.write_text("{not json", encoding="utf-8")
        assert RunOptIns(p).list() == []
        with pytest.raises(ValueError):
            RunOptIns(p).set("short", True)

    async def test_gate_cached_and_dev_override_and_no_vetted_flag(self, monkeypatch):
        calls = []
        g = ToolBondGate(None, checker=lambda d: calls.append(d) or True)
        assert (await g.is_bonded(D1))[0] is True
        assert (await g.is_bonded(D1))[1].endswith("(cached)")
        assert calls == [D1]
        assert not hasattr(g, "is_vetted")
        monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
        assert await ToolBondGate(None).is_bonded(D2) == (True, "dev override")


# ---------------------------------------------------------------------------
# 5. probe_tools: drop unbonded, add bond + author context
# ---------------------------------------------------------------------------

AUTH = "0x" + "ab" * 20


def _rows():
    return [{"digest": d, "name": n, "author": AUTH}
            for d, n in ((D1, "one"), (D2, "two"), (D3, "three"))]


class TestProbeContext:
    async def test_keeps_every_row_and_annotates(self, tmp_path):
        rt = _make_runtime(tmp_path)
        rt.tool_bond_client = FakeClient({
            D1: _st(),
            D2: _st(bonded=False),
            D3: RuntimeError("rpc down"),
        })
        _ratings(rt, {D1: _rating(malware=0.1, correctness=0.8),
                      D2: _rating(malware=0.9)})
        rows = await annotate_probe_rows(rt, _rows())
        assert [r["digest"] for r in rows] == [D1, D2, D3]
        one, two, three = rows
        assert one["bonded"] is True and "vetted" not in one
        assert one["rating"]["malware"] == 0.1 and one["rating"]["correctness"] == 0.8
        assert two["bonded"] is False and two["rating"]["malware"] == 0.9
        assert three["bonded"] is None and three["rating"] is None  # fault: unknown
        assert not any(k in one for k in ("harm", "trust", "containment"))
        ctx = one["author_context"]
        assert {o["digest"] for o in ctx["other_tools"]} == {D2, D3}
        assert ctx["max_other_malware"] == 0.9
        assert ctx["rep"] is None          # no RepToken configured here

    async def test_status_cached(self, tmp_path):
        rt = _make_runtime(tmp_path)
        rt.tool_bond_client = client = FakeClient({D1: _st(), D2: _st(), D3: _st()})
        await annotate_probe_rows(rt, _rows())
        n = len(client.status_calls)
        await annotate_probe_rows(rt, _rows())
        assert len(client.status_calls) == n == 3

    async def test_no_tool_bond_keeps_rows_unknown(self, tmp_path):
        rt = _make_runtime(tmp_path)
        rt.tool_bond_client = FakeClient(available=False)
        rows = await annotate_probe_rows(rt, _rows())
        assert len(rows) == 3 and all(r["bonded"] is None for r in rows)
        assert rows[0]["author_context"]["max_other_malware"] is None

    async def test_author_rep_read(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path)
        rt.tool_bond_client = FakeClient({D1: _st()})

        async def _rep(author):
            return 5 * 10**18
        monkeypatch.setattr("atn.tool_security._rep_reader", lambda r: _rep)
        rows = await annotate_probe_rows(rt, _rows()[:1])
        assert rows[0]["author_context"]["rep"] == str(5 * 10**18)
        assert rows[0]["author_context"]["max_other_malware"] is None

    async def test_probe_tools_lists_unbonded_tools_with_their_bond(self, tmp_path):
        rt = _make_runtime(tmp_path)
        rt.tool_bond_client = FakeClient({D1: _st(), D2: _st(bonded=False)})
        manifests = {d: {"kind": "tool_manifest", "name": n, "description": "echo",
                         "author": AUTH, "trust_class": "pinned"}
                     for d, n in ((D1, "one"), (D2, "two"))}
        lib = SimpleNamespace(artifact_digests=lambda: list(manifests),
                              blob_json=manifests.get)
        rt.autonet = SimpleNamespace(_service=SimpleNamespace(_library=lib))
        out = await execute_tool("probe_tools", {"query": "echo", "k": 5}, rt,
                                 caller_id=None)
        bonded = {m["digest"]: m["bonded"] for m in out["matches"]}
        assert bonded == {D1: True, D2: False}

    def test_rating_for_survives_a_failing_provider(self, tmp_path):
        rt = _make_runtime(tmp_path)

        def boom(digest):
            raise RuntimeError("no close yet")
        rt.tool_store.rating_provider = boom
        assert rating_for(rt, D1) is None


# ---------------------------------------------------------------------------
# 6. WS owner surface: run opt-ins, bond status, staked set_tool_published
# ---------------------------------------------------------------------------

def _bridge(rt):
    from atn.ws_server import WebSocketBridge
    srv = WebSocketBridge.__new__(WebSocketBridge)
    srv.runtime = rt
    return srv


def _sess(scoped=False):
    return SimpleNamespace(scope_ids={"agent-1"} if scoped else None,
                           local=not scoped, owner="0xowner", authed=True)


class TestWsSurface:
    async def test_optin_roundtrip(self, tmp_path):
        rt, rec = await _adopted(tmp_path)
        rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: True)
        srv = _bridge(rt)
        r = await srv._handle_message({"type": "set_tool_run_optin", "msg_id": "1",
                                       "digest": rec.digest, "allowed": True}, _sess())
        assert r["ok"] and r["result"] == {"digest": rec.digest, "allowed": True}
        r = await srv._handle_message({"type": "list_tool_run_optins", "msg_id": "2"},
                                      _sess())
        assert [o["digest"] for o in r["result"]["optins"]] == [rec.digest]
        assert r["result"]["optins"][0]["name"] == "net_tool"
        rt.tool_bond_client = FakeClient({rec.digest: _st()})
        _ratings(rt, {rec.digest: _rating(malware=0.25)})
        r = await srv._handle_message({"type": "tool_bond_status", "msg_id": "3",
                                       "digest": rec.digest}, _sess())
        assert r["result"]["status"]["bonded"] is True
        assert r["result"]["rating"]["malware"] == 0.25
        assert r["result"]["optin"] is True and r["result"]["network_tool"] is True
        assert "runnable" not in r["result"]
        r = await srv._handle_message({"type": "tool_bond_status", "msg_id": "3b",
                                       "digests": [rec.digest, "ab" * 32]}, _sess())
        views = r["result"]["statuses"]
        assert views[rec.digest]["status"]["bonded"] is True
        assert views["ab" * 32]["network_tool"] is False
        r = await srv._handle_message({"type": "set_tool_run_optin", "msg_id": "4",
                                       "digest": rec.digest, "allowed": "yes"}, _sess())
        assert r["ok"] is False

    async def test_scoped_session_refused(self, tmp_path):
        rt = _make_runtime(tmp_path)
        srv = _bridge(rt)
        for t in ("set_tool_run_optin", "list_tool_run_optins", "tool_bond_status"):
            r = await srv._handle_message({"type": t, "msg_id": "1", "digest": D1,
                                           "allowed": True}, _sess(scoped=True))
            assert r["ok"] is False and "owner-only" in r["error"], t

    async def test_set_tool_published_stakes(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _agent(rt, "a")
        rt.tool_bond_client = client = FakeClient()
        res = await _register(rt, "a")
        srv = _bridge(rt)
        r = await srv._handle_message({"type": "set_tool_published", "msg_id": "1",
                                       "digest": res["digest"], "published": True},
                                      _sess())
        assert r["ok"] is False and "novelty" in r["error"]
        r = await srv._handle_message({"type": "set_tool_published", "msg_id": "2",
                                       "digest": res["digest"], "published": True,
                                       "novelty": NOVEL}, _sess())
        assert r["ok"] is True, r
        new = r["result"]["digest"]
        assert rt.tool_store.get(new).published and client.bonds[0][1] == new
        r = await srv._handle_message({"type": "list_registered_tools", "msg_id": "3"},
                                      _sess())
        row = next(t for t in r["result"]["tools"] if t["digest"] == new)
        assert row["novelty"] == NOVEL


async def test_publish_after_registration_reauthors_without_a_redo(tmp_path):
    rt = _make_runtime(tmp_path)
    rt._config.autonet.owner_wallet = "0x" + "44" * 20
    defn = await _agent(rt, "a", on_chain=False)
    rt.tool_bond_client = client = FakeClient()
    res = await _register(rt, "a", novelty=NOVEL)
    before = rt.tool_store.get(res["digest"]).manifest["author"]
    defn.identity.registered_on_chain = True     # the owner approved it
    out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                             caller_id="a")
    assert out.get("published") is True, out
    new = rt.tool_store.get(out["digest"])
    assert new.manifest["author"].lower() == defn.identity.address.lower()
    if before.lower() != defn.identity.address.lower():
        assert out["digest"] != res["digest"]
    assert client.bonds and client.bonds[-1][1] == out["digest"]


def _real_client(monkeypatch, plan, *, author="0x" + "00" * 20):
    """The real client with the chain stubbed: ``plan`` is what _plan_bond
    answers; every write is recorded by its step."""
    from atn.runtime.tool_bond import ToolBondClient
    client = ToolBondClient(None)
    steps = []

    async def tool_author(digest):
        return author

    async def write(key, build, gas_floor=300_000):
        steps.append("write")
        return {"success": True, "tx_hash": f"0x{len(steps)}"}

    async def approve(key, token, amount):
        steps.append(("approve", token, amount))
        return {"success": True, "tx_hash": "0xapprove"}
    monkeypatch.setattr(client, "tool_author", tool_author)
    monkeypatch.setattr(client, "_write", write)
    monkeypatch.setattr(client, "_approve", approve)
    monkeypatch.setattr(client, "_plan_bond", lambda me, d, t, a: dict(plan))
    return client, steps


async def test_register_and_bond_registers_approves_then_bonds(monkeypatch):
    """The real client: registerTool when the digest is unclaimed, the
    token's approve when the allowance falls short, then the bond."""
    client, steps = _real_client(monkeypatch, {
        "token": TOKEN, "amount": 100_000_000, "value": 100_000_000, "approve": True})
    out = await client.register_and_bond("0x" + "11" * 32, "ab" * 32)
    assert out["success"] and out["token"] == TOKEN and out["amount"] == 100_000_000
    assert steps == ["write", ("approve", TOKEN, 100_000_000), "write"]
    assert out["txs"] == ["0x1", "0xapprove", out["tx_hash"]]


async def test_register_and_bond_skips_what_is_done(monkeypatch):
    from eth_account import Account
    key = "0x" + "11" * 32
    me = Account.from_key(key).address
    # Already registered to the signer, allowance enough: the bond only.
    client, steps = _real_client(monkeypatch, {
        "token": TOKEN18, "amount": 5 * 10 ** 17, "value": 100, "approve": False},
        author=me)
    out = await client.register_and_bond(key, "ab" * 32, token=TOKEN18)
    assert out["success"] and steps == ["write"]
    # Already bonded: nothing at all.
    client, steps = _real_client(monkeypatch, {
        "already_bonded": True, "token": TOKEN, "amount": 0, "value": 100}, author=me)
    out = await client.register_and_bond(key, "ab" * 32)
    assert out["success"] and out["note"] == "already bonded" and steps == []


async def test_register_and_bond_refuses_a_plan_error(monkeypatch):
    client, steps = _real_client(monkeypatch, {
        "error": "name the index token to bond in (`token`): you hold several",
        "index_tokens": [TOKEN, TOKEN18]})
    out = await client.register_and_bond("0x" + "11" * 32, "ab" * 32)
    assert not out["success"] and out["step"] == "bond"
    assert out["index_tokens"] == [TOKEN, TOKEN18] and steps == ["write"]


def test_bond_reverts_read_as_reasons():
    """ToolBond f297209: DustAmount for a deposit worth under one value base
    unit, NotInIndex only for a token outside the index; by name or by the
    custom error's selector, as web3 reports it."""
    from eth_utils import keccak

    from atn.runtime.tool_bond import bond_error
    assert "DustAmount" in bond_error("DustAmount") and "bond more" in bond_error("DustAmount")
    sel = "0x" + keccak(text="DustAmount()")[:4].hex()
    assert "bond more" in bond_error(f"would revert: ('{sel}', '{sel}')")
    sel = "0x" + keccak(text="NotInIndex()")[:4].hex()
    assert "not in the value index" in bond_error(f"execution reverted {sel}")
    assert bond_error("out of gas") == "out of gas"


async def test_bounty_payout_uses_the_rounds_token(monkeypatch):
    """A round's bounty parts are owed in roundBounty's token (the bond's
    token when the round started finalizing), and paid by payOut."""
    from atn.runtime.review_batch import ReviewBondClient
    house, token = "0x" + "ab" * 20, "0x" + "c6" * 20

    class Fn:
        def __init__(self, value):
            self.value = value

        def call(self):
            return self.value

    class Bond:
        class functions:
            @staticmethod
            def householdOf(epoch, agent):
                return Fn(house)

            @staticmethod
            def roundBounty(digest, epoch):
                return Fn((5, 6, 7, 8, token))

            @staticmethod
            def owed(h, t):
                assert (h, t) == (house, token)
                return Fn(42)

            @staticmethod
            def payOut(h, t):
                return ("payOut", h, t)

    client = ReviewBondClient(None)
    sent = []

    async def read(fn):
        return fn(Bond, None)

    async def write(key, build, gas_floor=300_000):
        sent.append(build(Bond, None, None))
        return {"success": True, "tx_hash": "0xpay"}
    monkeypatch.setattr(client, "_read", read)
    monkeypatch.setattr(client, "_write", write)
    out = await client.pay_out_bounty("0x" + "11" * 32, "ab" * 32, 1, "0x" + "cd" * 20)
    assert out["success"] and out["amount"] == 42 and out["token"] == token
    assert sent == [("payOut", house, token)]

    # The claim normally pays the part itself: nothing owed, no payOut
    # (it would revert NothingToClaim).
    monkeypatch.setattr(Bond.functions, "owed", staticmethod(lambda h, t: Fn(0)))
    sent.clear()
    out = await client.pay_out_bounty("0x" + "11" * 32, "ab" * 32, 1, "0x" + "cd" * 20)
    assert out == {"success": True, "amount": 0} and sent == []


def test_bond_amount_covers_the_minimum_value():
    """The fewest base units whose value (amount x parity / 1e30) reaches
    a target: 100 value units in a 6-decimal token at 1e30 is 100 base
    units; in an 18-decimal token at parity 2e18, just enough and no more."""
    from atn.runtime.tool_bond import _amount_for_value
    assert _amount_for_value(100_000_000, 10 ** 30) == 100_000_000
    a = _amount_for_value(100_000_000, 2 * 10 ** 18)
    assert a * 2 * 10 ** 18 // 10 ** 30 >= 100_000_000
    assert (a - 1) * 2 * 10 ** 18 // 10 ** 30 < 100_000_000
    assert _amount_for_value(7, 3 * 10 ** 29) == 24   # ceil(70 / 3)
