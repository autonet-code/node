"""Reference harness distro bootstrap (docs/tool_substrate.md —
"Resident tools, loadouts, distros").

Covers: bootstrap registers a distro whose deps all exist + are enabled;
digests are stable and non-duplicating across two bootstraps on the same
store (content-addressed idempotency); ``active_loadout`` is set; and a
cognitive attestation carries ``loadout`` == the distro digest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.events import EventBus
from atn.harness_distro import DISTRO_NAME, bootstrap_reference_distro
from atn.models import AgentDefinition, AgentMode
from atn.agent_tools import execute_tool


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


def _fake_embedder(text):
    return (0.1, 0.2, 0.3) if text else ()


ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


class TestHarnessDistroBootstrap:
    def test_runtime_init_bootstraps_distro(self, tmp_path):
        """Runtime construction stamps active_loadout with a real distro."""
        rt = _make_runtime(tmp_path)
        store = rt.tool_store
        assert store.active_loadout, "active_loadout should be set at init"
        distro = store.get(store.active_loadout)
        assert distro is not None
        assert distro.name == DISTRO_NAME
        assert distro.trust_class == "pinned"
        assert distro.author == "user"
        assert distro.published is False

    def test_distro_deps_exist_and_enabled(self, tmp_path):
        rt = _make_runtime(tmp_path)
        store = rt.tool_store
        distro = store.get(store.active_loadout)
        deps = distro.manifest.get("dependencies") or []
        assert deps, "distro must declare module deps"
        assert deps == sorted(deps), "deps must be sorted"
        assert len(deps) == len(set(deps)), "deps must be unique"
        for dep_digest in deps:
            mod = store.get(dep_digest)
            assert mod is not None, f"module {dep_digest[:12]} missing"
            assert mod.enabled
            assert mod.trust_class == "pinned"
            assert mod.name.startswith("atn_")

    def test_shell_module_present(self, tmp_path):
        """The shell bundle IS included, and unlike every other module its
        blob is a RUNNABLE program (shell_tools.py grew a sealed-protocol
        __main__), so its manifest declares what it provides.

        The other atn_* modules stay identity-only: their executors take the
        live Runtime, so their blobs are concatenated source, not programs.
        See tests/atn/test_shell_bundle.py for the runnability proof.
        """
        rt = _make_runtime(tmp_path)
        store = rt.tool_store
        deps = store.get(store.active_loadout).manifest.get("dependencies") or []
        names = {store.get(d).name for d in deps}
        assert "atn_shell" in names

        shell = next(store.get(d) for d in deps
                     if store.get(d).name == "atn_shell")
        assert shell.manifest.get("code_digest"), "shell module lost its blob"
        assert (shell.manifest.get("capabilities") or {}).get("provides"), \
            "shell module no longer declares what it provides"

    def test_idempotent_stable_digest_no_duplicates(self, tmp_path):
        """A second bootstrap over identical source yields the same distro
        digest and does not duplicate records (content-addressed)."""
        rt = _make_runtime(tmp_path)
        store = rt.tool_store
        first = store.active_loadout
        count_after_init = len(store)

        second = bootstrap_reference_distro(rt)
        assert second == first, "distro digest must be stable"
        assert store.active_loadout == first
        assert len(store) == count_after_init, "re-bootstrap must not add records"

    @pytest.mark.asyncio
    async def test_attestation_carries_loadout(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        store = rt.tool_store
        store._embedder = _fake_embedder

        # An agent-authored tool to attest against.
        res = await execute_tool(
            "register_tool",
            {"name": "echo_tool", "description": "Echo x back.",
             "input_schema": SCHEMA, "code": ECHO_CODE},
            rt, caller_id="child",
        )
        assert "error" not in res

        sunk: list[dict] = []
        store.event_sink = sunk.append

        out = await execute_tool(
            "attest_tools",
            {"judgments": [{"tool": "echo_tool", "ok": True, "score": 0.9}],
             "context": "wiring up an echo round-trip"},
            rt, caller_id="child",
        )
        assert out == {"attested": 1, "skipped": []}
        assert len(sunk) == 1
        assert sunk[0]["loadout"] == store.active_loadout

        # And the attestation jsonl row carries it too.
        rows = list(store._iter_attestations())
        assert rows and rows[-1]["loadout"] == store.active_loadout

    def test_no_loadout_stamp_when_empty(self, tmp_path):
        """When active_loadout is empty, attestation events/rows omit it."""
        rt = _make_runtime(tmp_path)
        store = rt.tool_store
        store.active_loadout = ""  # simulate a store with no distro
        store._embedder = _fake_embedder

        # register directly (no agent needed; owner author "user").
        res = store.register(
            name="plain_tool", description="Echo.",
            input_schema=SCHEMA, author="user", code=ECHO_CODE,
        )
        sunk: list[dict] = []
        store.event_sink = sunk.append
        out = store.attest_usage(
            "user", [{"tool": res["digest"], "ok": True}], "ctx")
        assert out["attested"] == 1
        assert "loadout" not in sunk[0]
        rows = list(store._iter_attestations())
        assert "loadout" not in rows[-1]
