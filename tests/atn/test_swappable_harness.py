"""Swappable harness parts (docs/tool_substrate.md, Decision 2026-10-12).

Three properties:

  1. POLICY. The loop's hardcoded policies are one declarative value carried
     by the distro blob. The reference policy reproduces today's prompts and
     thresholds byte for byte; a forked distro's overlay changes behavior.
  2. LOADOUT. An agent's distro is owner-bound (never an agent tool) and
     attestations carry the agent's own loadout digest.
  3. CODE SWAPS. A core-replacing provider swaps in only with the owner's
     core-replacement ack, a ToolBond bond (fail closed), and a per-agent
     loadout binding; every swapped call leaves signed evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atn import delegate_prompts as dp
from atn.agent_tools import execute_tool
from atn.events import EventBus
from atn.harness_policy import (
    DISTRO_KIND, HarnessPolicy, distro_config, encode_distro_config,
    parse_distro_config, policy_for_agent, policy_from_overlay,
    reference_policy,
)
from atn.models import AgentDefinition, AgentMode
from atn.providers import base as b
from atn.providers.base import Provider, ProviderResponse, ToolCall, Usage
from atn.runtime import shell_provider
from atn.runtime.tool_bond import DEV_OVERRIDE_ENV, ToolBondGate

# ---------------------------------------------------------------------------
# helpers
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


async def _register_agent(rt, agent_id, parent_id=None):
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5", parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


def _register_fork(store, overlay: dict, deps: list[str] | None = None,
                   name: str = "my_fork") -> str:
    ref = store.get(store.active_loadout)
    modules = deps if deps is not None else list(ref.manifest["dependencies"])
    cfg = distro_config(name, "fork-1", modules)
    cfg["policy"] = overlay
    res = store.register(
        name=name, description="a forked harness",
        input_schema={"type": "object"}, author="user",
        code=encode_distro_config(cfg), dependencies=sorted(modules),
        publish=False)
    return res["digest"]


class ScriptedProvider(Provider):
    def __init__(self, script):
        self._script = list(script)
        self.seen: list[list[dict]] = []
        self._active_model = "test-model"

    @property
    def name(self) -> str:
        return "scripted"

    async def send(self, **kw) -> ProviderResponse:
        return ProviderResponse(text="SUMMARY", stop_reason="end_turn")

    async def send_stream(self, *, messages, **kw) -> ProviderResponse:
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self._script.pop(0)


def _resp(**kw):
    kw.setdefault("usage", Usage(input_tokens=5, output_tokens=2))
    return ProviderResponse(**kw)


def _loop_script():
    """A run that writes a code file, then repeats one call (loop warning),
    then ends: exercises verify injection and the loop guard text."""
    write = ToolCall(id="w1", name="write_file",
                     input={"path": "x.py", "content": "pass"})
    rep = lambda i: ToolCall(id=f"r{i}", name="read_file", input={"path": "x.py"})
    return [
        _resp(tool_calls=[write], stop_reason="tool_use"),
        _resp(tool_calls=[rep(1)], stop_reason="tool_use"),
        _resp(tool_calls=[rep(2)], stop_reason="tool_use"),
        _resp(tool_calls=[rep(3)], stop_reason="tool_use"),
        _resp(text="done", stop_reason="end_turn"),
        _resp(text="verified", stop_reason="end_turn"),
    ]


async def _run(policy=None, pass_policy=True):
    p = ScriptedProvider(_loop_script())

    async def ex(name, inp):
        return {"content": "ok"}
    kw = dict(message="task", system="sys", tools=[
        {"name": "write_file", "input_schema": {"type": "object"}},
        {"name": "read_file", "input_schema": {"type": "object"}}],
        max_turns=10, tool_executor=ex)
    if pass_policy:
        kw["harness_policy"] = policy
    out = await p.run_agent_loop(**kw)
    return p, out


# ---------------------------------------------------------------------------
# 1. Reference policy reproduces the current harness
# ---------------------------------------------------------------------------

def test_reference_policy_is_the_current_constants():
    ref = reference_policy()
    assert ref.common_base == dp._COMMON_BASE
    assert dict(ref.tool_category_notes) == dp._TOOL_CATEGORY_NOTES
    assert dict(ref.type_guidance) == dp._TYPE_GUIDANCE
    assert ref.review_step_prompt == dp.REVIEW_STEP_PROMPT
    assert ref.verify_step_prompt == dp.VERIFY_STEP_PROMPT
    assert ref.compaction_inline_prompt == b.COMPACTION_INLINE_PROMPT
    assert ref.truncated_tool_call_prompt == b.TRUNCATED_TOOL_CALL_PROMPT
    assert ref.parallel_safe_tools == b.PARALLEL_SAFE_TOOLS
    assert ref.headroom_levels == (0.70, 0.80, 0.90)
    assert (ref.repeat_call_limit, ref.oscillation_window,
            ref.per_turn_input_max) == (3, 6, 200_000)
    assert (ref.max_compactions_per_run, ref.compaction_min_reduction,
            ref.compact_retain_turns, ref.compact_retain_user_chars,
            ref.prune_protect_recent_chars, ref.prune_min_savings,
            ref.max_hard_resets_per_run, ref.compact_summary_max_tokens) == (
        2, 0.20, 2, 20_000, 40_000, 20_000, 2, 4096)
    assert (ref.verify_follow_up_max_turns, ref.review_follow_up_max_turns) == (8, 4)
    assert ref.tool_bundles is None


def test_reference_prompts_are_byte_identical():
    ref = reference_policy()
    cats = [[], ["shell"], ["shell", "web", "sdk_builtin"],
            sorted(dp._TOOL_CATEGORY_NOTES)]
    for agent_type in list(dp._TYPE_GUIDANCE) + ["unknown"]:
        for c in cats:
            assert dp.build_system_prompt(agent_type, "a", None, c) == \
                dp.build_system_prompt(agent_type, "a", None, c, policy=ref)
    for ctx in (0, 16_000, 200_000):
        assert dp.build_identity_header("a", "general", None, ctx) == \
            dp.build_identity_header("a", "general", None, ctx, policy=ref)
    assert dp.format_review_prompt(["t1"]) == dp.format_review_prompt(["t1"], ref)
    assert b._headroom_note(70, 100) == (
        "\n\n[Context: about 70% of the usable context is in use; roughly "
        "30 tokens remain before older history is summarized. Keep tool "
        "output small and prefer finishing or delegating over broad new "
        "exploration.]")


def test_reference_roundtrips_through_the_distro_blob():
    ref = reference_policy()
    blob = encode_distro_config(distro_config("x", "v", ["a" * 64]))
    cfg = parse_distro_config(blob.encode())
    assert cfg["kind"] == DISTRO_KIND
    assert policy_from_overlay(cfg["policy"]) == ref
    assert policy_from_overlay(json.loads(json.dumps(ref.to_dict()))) == ref


async def test_loop_with_reference_policy_is_byte_identical():
    p_none, out_none = await _run(pass_policy=False)
    p_ref, out_ref = await _run(policy=reference_policy())
    p_dict, _ = await _run(policy=reference_policy().to_dict())
    assert p_none.seen == p_ref.seen == p_dict.seen
    assert out_none.text == out_ref.text
    flat = json.dumps(p_none.seen)
    assert "You appear to be looping" in flat
    assert "Before you conclude" in flat


async def test_reference_distro_blob_carries_the_reference_policy(tmp_path):
    rt = _make_runtime(tmp_path)
    store = rt.tool_store
    rec = store.get(store.active_loadout)
    raw = store._blob_store().get_bytes(rec.manifest["code_digest"])
    cfg = parse_distro_config(raw)
    assert cfg is not None and cfg["name"] == "atn_reference_harness"
    assert policy_from_overlay(cfg["policy"]) == reference_policy()
    distros = store.list_distros()
    assert distros[0]["digest"] == store.active_loadout
    assert distros[0]["is_reference"] and distros[0]["policy_overrides"] == []


# ---------------------------------------------------------------------------
# 1b. Overlays
# ---------------------------------------------------------------------------

def test_overlay_validation():
    pol = policy_from_overlay({
        "review_step_prompt": "Review {tools} now.",
        "verify_step_prompt": "no placeholder",          # rejected
        "repeat_call_limit": 10_000,                     # clamped
        "compaction_min_reduction": "bad",               # rejected
        "headroom_levels": [0.5, 2.0, 0.9],              # 2.0 dropped
        "tool_bundles": ["shell"],
        "not_a_field": 1,                                # ignored
        "type_guidance": {"general": "G!"},
    })
    ref = reference_policy()
    assert pol.review_step_prompt == "Review {tools} now."
    assert pol.verify_step_prompt == ref.verify_step_prompt
    assert pol.repeat_call_limit == 100
    assert pol.compaction_min_reduction == ref.compaction_min_reduction
    assert pol.headroom_levels == (0.5, 0.9)
    assert pol.select_tools(["shell", "web"]) == ["shell"]
    assert pol.type_guidance["general"] == "G!"
    assert pol.type_guidance["explore"] == ref.type_guidance["explore"]


async def test_overlay_changes_loop_behavior():
    pol = policy_from_overlay({"loop_warning_prompt": "STOP LOOPING",
                               "verify_enabled": False})
    p, _ = await _run(policy=pol)
    flat = json.dumps(p.seen)
    assert "STOP LOOPING" in flat
    assert "You appear to be looping" not in flat
    assert "Before you conclude" not in flat


def test_overlay_drives_the_system_prompt():
    pol = policy_from_overlay({"common_base": "CUSTOM BASE\n"})
    assert dp.build_system_prompt("general", "a", policy=pol).startswith("CUSTOM BASE")


async def test_policy_for_agent_reads_the_bound_fork(tmp_path):
    rt = _make_runtime(tmp_path)
    store = rt.tool_store
    fork = _register_fork(store, {"loop_warning_prompt": "FORKED"})
    defn = SimpleNamespace(id="a", loadout=fork)
    assert policy_for_agent(rt, defn).loop_warning_prompt == "FORKED"
    assert policy_for_agent(rt, SimpleNamespace(id="a", loadout="")) == reference_policy()
    assert policy_for_agent(rt, SimpleNamespace(id="a", loadout="f" * 64)) == reference_policy()
    names = {d["digest"]: d for d in store.list_distros()}
    assert fork in names and names[fork]["policy_overrides"] == ["loop_warning_prompt"]


# ---------------------------------------------------------------------------
# 2. Per-agent loadout binding
# ---------------------------------------------------------------------------

async def test_loadout_is_owner_set_only(tmp_path):
    rt = _make_runtime(tmp_path)
    await _register_agent(rt, "boss")
    await _register_agent(rt, "kid", parent_id="boss")
    fork = _register_fork(rt.tool_store, {"loop_warning_prompt": "F"})

    for caller in ("kid", "boss"):
        res = await execute_tool("update_agent", {"agent_id": "kid", "loadout": fork},
                                 rt, caller_id=caller)
        assert "owner" in res.get("error", ""), res
    res = await execute_tool("update_agent", {"agent_id": "kid", "loadout": "e" * 64}, rt)
    assert "not a distro" in res["error"]
    res = await execute_tool("update_agent", {"agent_id": "kid", "loadout": fork}, rt)
    assert "error" not in res, res
    assert rt.get_agent("kid").loadout == fork
    got = await execute_tool("get_agent", {"agent_id": "kid"}, rt)
    assert got["loadout"] == fork
    # Binding the reference distro is the same as clearing.
    await execute_tool("update_agent", {"agent_id": "kid",
                                        "loadout": rt.tool_store.active_loadout}, rt)
    assert rt.get_agent("kid").loadout == ""


async def test_loadout_survives_save_and_load(tmp_path):
    from atn.loader import load_agent_file, save_agent
    defn = AgentDefinition(id="z", name="z", mode=AgentMode.COGNITIVE,
                           provider="p", cognitive_model="m", loadout="ab" * 32)
    path = save_agent(defn, tmp_path)
    back, errs = load_agent_file(path)
    assert not errs and back.loadout == "ab" * 32


async def test_attestation_carries_the_agents_own_loadout(tmp_path):
    rt = _make_runtime(tmp_path)
    store = rt.tool_store
    await _register_agent(rt, "a1")
    await _register_agent(rt, "a2")
    fork = _register_fork(store, {"loop_warning_prompt": "F"})
    rt.get_agent("a1").loadout = fork
    res = store.register(name="t", description="d",
                         input_schema={"type": "object"}, author="a1",
                         code="print(1)")
    store.get(res["digest"]).grants.add("a2")
    sunk: list[dict] = []
    store.event_sink = sunk.append
    store.attest_usage("a1", [{"tool": res["digest"], "ok": True}], "ctx")
    store.attest_usage("a2", [{"tool": res["digest"], "ok": True}], "ctx")
    loadouts = [e.get("loadout") for e in sunk]
    assert loadouts[0] == fork
    assert loadouts[-1] == store.active_loadout


# ---------------------------------------------------------------------------
# 3. Code swaps: gates + evidence
# ---------------------------------------------------------------------------

PROVIDER_CODE = (
    "import sys, json, subprocess\n"
    "env = json.load(sys.stdin)\n"
    "open('scratch.txt', 'w').write('x')\n"
    "subprocess.run([sys.executable, '-c', 'pass'])\n"
    "cmd = env['args'].get('command', '')\n"
    "print(json.dumps({'ok': True, 'result': {'output': 'swapped:' + cmd,"
    " 'exit_code': 0}}))\n"
)
CORE_CAPS = {"provides": ["bash"], "net": True, "fs": True, "spawn": True}


async def _adopted_core_provider(tmp_path):
    author = _make_runtime(tmp_path / "author")
    await _register_agent(author, "foreign")
    res = await execute_tool("register_tool", {
        "name": "fast_shell", "description": "A faster bash.",
        "input_schema": {"type": "object"}, "code": PROVIDER_CODE,
        "capabilities": CORE_CAPS}, author, caller_id="foreign")
    assert "error" not in res, res
    digest = res["digest"]
    src = author.tool_store._blob_store()

    rt = _make_runtime(tmp_path / "adopter")
    await _register_agent(rt, "worker")

    async def _fetch(d):
        return src.get_bytes(d)
    rt.tool_store.blob_fetcher = _fetch
    return rt, digest


async def test_core_replacement_needs_explicit_ack(tmp_path):
    rt, digest = await _adopted_core_provider(tmp_path)
    store = rt.tool_store
    prop = await store.propose_adoption("worker", digest, reason="speed")
    assert prop["core_replacement"] == ["bash"]
    assert store.list_adoption_proposals("pending")[0]["core_replacement"] == ["bash"]
    res = await store.approve_adoption(digest)
    assert "core_replacement_ack required" in res["error"]
    assert store.get(digest) is None
    res = await store.approve_adoption(digest, core_replacement_ack=True)
    assert res["core_replacement_ack"] is True
    assert store.get(digest).core_replacement_ack is True
    # Survives a reload.
    store._records.clear()
    store._load()
    assert store.get(digest).core_replacement_ack is True


async def _bound_swap(tmp_path, ack=True):
    rt, digest = await _adopted_core_provider(tmp_path)
    store = rt.tool_store
    await store.propose_adoption("worker", digest, reason="speed")
    await store.approve_adoption(digest, core_replacement_ack=ack) if ack else None
    if not ack:
        # Simulate a pre-gate install: adopted but never acknowledged.
        from atn.tool_store import ToolRecord
        payload = await store.fetch_payload(digest)
        store._records[digest] = ToolRecord(digest=digest, manifest=payload,
                                            local_author="worker", origin="adopted")
    ref = store.get(store.active_loadout)
    fork = _register_fork(store, {}, list(ref.manifest["dependencies"]) + [digest],
                          name="fast_harness")
    return rt, digest, fork


async def test_no_swap_without_per_agent_binding(tmp_path, monkeypatch):
    monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
    rt, digest, fork = await _bound_swap(tmp_path)
    assert await shell_provider.dispatch_shell(rt, "worker", "bash",
                                               {"command": "echo hi"}) is None
    rt.get_agent("worker").loadout = fork
    assert set(shell_provider.resolve_shell_provider(rt, "worker")) == {"bash"}
    # The reference distro never swaps.
    rt.get_agent("worker").loadout = rt.tool_store.active_loadout
    assert shell_provider.resolve_shell_provider(rt, "worker") == {}


async def test_no_swap_without_ack(tmp_path, monkeypatch):
    monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
    rt, digest, fork = await _bound_swap(tmp_path, ack=False)
    rt.get_agent("worker").loadout = fork
    assert shell_provider.resolve_shell_provider(rt, "worker") == {}


async def test_bond_gate_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("ATN_TOOL_BOND_ADDRESS", raising=False)
    rt, digest, fork = await _bound_swap(tmp_path)
    rt.get_agent("worker").loadout = fork
    rt.tool_bond_gate = ToolBondGate(rt)
    monkeypatch.setattr("atn.runtime.tool_bond._registry_seed", lambda r: {})
    ok, reason = await rt.tool_bond_gate.is_bonded(digest)
    assert not ok and "no ToolBond address" in reason
    assert await shell_provider.dispatch_shell(rt, "worker", "bash",
                                               {"command": "x"}) is None
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: False)
    assert await shell_provider.dispatch_shell(rt, "worker", "bash",
                                               {"command": "x"}) is None


async def test_bonded_swap_runs_and_leaves_signed_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    rt, digest, fork = await _bound_swap(tmp_path)
    rt.get_agent("worker").loadout = fork
    seen: list[str] = []
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: seen.append(d) or True)
    # A core swap needs its declared spawn: reviewed safe by the network.
    rt.tool_store.security_standing_provider = lambda d: {"harm_sum": 0, "harm_mass": 100, "supply": 1000}

    out = await rt.engine.route_tool_call("bash", {"command": "echo hi"}, "worker")
    assert out == {"output": "swapped:echo hi", "exit_code": 0}
    assert seen == [digest]
    # Other shell names are untouched by a provider that claims only bash.
    assert await shell_provider.dispatch_shell(rt, "worker", "read_file",
                                               {"path": "x"}) is None

    store = rt.tool_store
    ev_digest = store.get(digest).last_evidence_digest
    assert ev_digest
    rows = shell_provider.list_evidence(store, digest)
    assert rows[0]["evidence_digest"] == ev_digest
    ev = store._blob_store().get_json(ev_digest)
    body = ev["body"]
    assert body["kind"] == "core_replacement_trace"
    assert body["tool_digest"] == digest and body["replaces"] == "bash"
    assert body["input"] == {"command": "echo hi"}
    assert "swapped:echo hi" in body["output"]
    kinds = {o["kind"] for o in body["observations"]}
    assert {"file", "spawn"} <= kinds, body["observations"]
    assert any(o.get("path", "").endswith("scratch.txt")
               for o in body["observations"] if o["kind"] == "file")
    if ev["sig"]:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        signer = Account.recover_message(
            encode_defunct(shell_provider.canonical_evidence_bytes(body)),
            signature=bytes.fromhex(ev["sig"].removeprefix("0x")))
        assert signer.lower() == ev["signer"].lower()


async def test_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
    rt, digest, fork = await _bound_swap(tmp_path)
    rt.get_agent("worker").loadout = fork
    monkeypatch.setenv(shell_provider.KILL_SWITCH_ENV, "1")
    assert shell_provider.resolve_shell_provider(rt, "worker") == {}


async def test_tool_bond_gate_basics(monkeypatch):
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    g = ToolBondGate(None, checker=lambda d: d.startswith("a"))
    assert (await g.is_bonded("a" * 64))[0] is True
    assert (await g.is_bonded("b" * 64))[0] is False
    assert (await g.is_bonded("nothex"))[0] is False
    monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
    g2 = ToolBondGate(None)
    assert (await g2.is_bonded("c" * 64)) == (True, "dev override")


async def test_worker_routes_swapped_names_to_the_daemon():
    """In an isolated worker, shell tools run locally, except names the
    agent's loadout swaps: those go to the daemon (route_tool_call ->
    dispatch_shell), the one place the gates and evidence live."""
    from atn.runtime.worker_loop import _route_tool

    calls: list = []

    class _Client:
        async def rpc(self, kind, body):
            calls.append((kind, body["name"]))
            return {"output": "from daemon"}

    out = await _route_tool(_Client(), "bash", {"command": "echo hi"}, "w",
                            swapped=frozenset({"bash"}))
    assert out == {"output": "from daemon"}
    assert calls == [("framework_tool", "bash")]


async def test_policy_hook_names_are_declarable_but_not_core(tmp_path):
    rt = _make_runtime(tmp_path)
    res = rt.tool_store.register(
        name="smart_compactor", description="d", input_schema={"type": "object"},
        author="user", code="print(1)",
        capabilities={"provides": ["policy:compaction", "policy:invented"]})
    assert res["manifest"]["capabilities"]["provides"] == ["policy:compaction"]
    assert shell_provider.core_claims(res["manifest"]) == []


# ---------------------------------------------------------------------------
# 4. Fallback is surfaced to the owner (transition-only events)
# ---------------------------------------------------------------------------

def _swap_events(rt):
    seen: list[dict] = []

    async def _h(ev):
        if ev.type.value == "harness.swap_status":
            seen.append(dict(ev.data))
    rt.events.subscribe(None, _h)
    return seen


async def test_fallback_notice_fires_once_per_transition(tmp_path, monkeypatch):
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    rt, digest, fork = await _bound_swap(tmp_path)
    rt.get_agent("worker").loadout = fork
    bonded = {"v": False}
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: bonded["v"])
    seen = _swap_events(rt)

    for _ in range(3):  # per-call refusals: one notice, not three
        assert await shell_provider.dispatch_shell(
            rt, "worker", "bash", {"command": "x"}) is None
    assert [(e["state"], e["reason"]) for e in seen] == [("fallback", "not_bonded")]
    assert seen[0]["agent_id"] == "worker" and seen[0]["tool_digest"] == digest
    assert seen[0]["provides"] == ["bash"] and seen[0]["tool_name"] == "fast_shell"

    detail = await execute_tool("get_agent", {"agent_id": "worker"}, rt)
    assert [(r["state"], r["reason"]) for r in detail["core_swap"]] == [
        ("fallback", "not_bonded")]
    assert len(seen) == 1  # same state again: silent

    # Bond lands: active again, the notice clears.
    bonded["v"] = True
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: bonded["v"])
    detail = await execute_tool("get_agent", {"agent_id": "worker"}, rt)
    assert detail["core_swap"][0]["state"] == "active"
    assert [e["state"] for e in seen] == ["fallback", "active"]

    # Unbinding the loadout after a fallback emits "gone".
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: (_ for _ in ()).throw(
        RuntimeError("rpc down")))
    detail = await execute_tool("get_agent", {"agent_id": "worker"}, rt)
    assert detail["core_swap"][0]["reason"] == "bond_check_error"
    rt.get_agent("worker").loadout = ""
    detail = await execute_tool("get_agent", {"agent_id": "worker"}, rt)
    assert detail["core_swap"] == []
    assert [e["state"] for e in seen] == ["fallback", "active", "fallback", "gone"]


async def test_fallback_reasons_for_sync_gates(tmp_path, monkeypatch):
    monkeypatch.setenv(DEV_OVERRIDE_ENV, "1")
    rt, digest, fork = await _bound_swap(tmp_path, ack=False)
    rt.get_agent("worker").loadout = fork
    rows = await shell_provider.swap_status(rt, "worker")
    assert [(r["reason"], r["provides"]) for r in rows] == [("not_approved", ["bash"])]

    rt.tool_store.get(digest).core_replacement_ack = True
    assert (await shell_provider.swap_status(rt, "worker"))[0]["state"] == "active"

    monkeypatch.setattr(rt.tool_store, "allowed", lambda a, r: False)
    assert (await shell_provider.swap_status(rt, "worker"))[0]["reason"] == "not_in_loadout"
    monkeypatch.undo()
    monkeypatch.setenv(shell_provider.KILL_SWITCH_ENV, "1")
    assert (await shell_provider.swap_status(rt, "worker"))[0]["reason"] == "swaps_disabled"
    # An agent with no core-replacing tool in its loadout has no rows.
    await _register_agent(rt, "plain")
    assert await shell_provider.swap_status(rt, "plain") == []
