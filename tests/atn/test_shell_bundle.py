"""The shell bundle is a runnable reference tool; the swap is gated off.

docs/tool_substrate.md — "Resident tools, loadouts, distros".

Two properties, deliberately separated:

  EXTRACTION (shipped) — atn/shell_tools.py is simultaneously the in-process
  fast path, a runnable sealed tool, and the code blob of the atn_shell module
  manifest. That makes an already-existing manifest honest: what its digest
  locks is now a program.

  RESOLUTION (gated off) — nothing may actually replace the built-in shell.
  The gate is a security boundary, not a feature flag waiting to be flipped:
  see atn/runtime/shell_provider.py for why containment does not exist for a
  net+fs+spawn tool.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from atn.harness_distro import bootstrap_reference_distro
from atn.runtime import shell_provider
from atn.shell_tools import SHELL_TOOL_EXECUTORS, SHELL_TOOLS, dispatch
from atn.tool_store import ToolStore

SHELL_MODULE = Path(__file__).resolve().parents[2] / "atn" / "shell_tools.py"


def _run_as_subprocess(envelope: dict, script: Path = SHELL_MODULE) -> dict:
    proc = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(envelope), capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, f"non-zero exit: {proc.stderr[:400]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def test_module_runs_as_a_sealed_tool():
    """PROPERTY: the file executes as a tool subprocess, speaking the sealed
    protocol (JSON envelope in, JSON result out)."""
    out = _run_as_subprocess(
        {"tool": "bash", "args": {"command": "echo bundle-ok"}})
    assert out["ok"] is True
    assert "bundle-ok" in out["result"]["output"]


def test_in_process_and_envelope_paths_agree():
    """PROPERTY: the envelope is a router, not a reimplementation. If these
    ever diverge, a swapped provider would behave differently from the
    built-in it replaced."""
    args = {"path": str(SHELL_MODULE), "limit": 3}
    direct = asyncio.run(SHELL_TOOL_EXECUTORS["read_file"](args))
    routed = asyncio.run(dispatch({"tool": "read_file", "args": args}))
    assert routed["ok"] is True
    assert routed["result"] == direct


@pytest.mark.parametrize("envelope,why", [
    ({"tool": "nope", "args": {}}, "unknown tool"),
    ({"args": {}}, "missing tool name"),
    ({"tool": ""}, "empty tool name"),
    ("not-an-object", "non-dict envelope"),
])
def test_bad_envelopes_are_errors_not_crashes(envelope, why):
    """PROPERTY: a subprocess boundary turns exceptions into 'tool exited N'
    with a traceback. Failures must ride the protocol instead."""
    out = asyncio.run(dispatch(envelope))
    assert out["ok"] is False, why
    assert out["error"]


def test_executor_faults_become_error_envelopes():
    """PROPERTY: even an executor raising must not escape dispatch."""
    out = asyncio.run(dispatch({"tool": "read_file", "args": {}}))
    assert out["ok"] is False or "error" in out.get("result", {})


def test_edit_file_is_surgical():
    """PROPERTY: edit_file changes exactly the matched text — ambiguous or
    absent matches are errors, never partial writes."""
    p = Path(tempfile.mkdtemp()) / "t.py"
    p.write_text("a = 1\nb = 1\nc = 2\n", encoding="utf-8")
    ex = SHELL_TOOL_EXECUTORS["edit_file"]
    # ambiguous without replace_all
    out = asyncio.run(ex({"path": str(p), "old_string": "= 1", "new_string": "= 9"}))
    assert "error" in out and "2 times" in out["error"]
    assert p.read_text(encoding="utf-8") == "a = 1\nb = 1\nc = 2\n"
    # absent
    out = asyncio.run(ex({"path": str(p), "old_string": "zzz", "new_string": "y"}))
    assert "error" in out
    # unique match
    out = asyncio.run(ex({"path": str(p), "old_string": "c = 2", "new_string": "c = 3"}))
    assert out.get("status") == "ok"
    assert p.read_text(encoding="utf-8") == "a = 1\nb = 1\nc = 3\n"
    # replace_all
    out = asyncio.run(ex({"path": str(p), "old_string": "= 1", "new_string": "= 7",
                          "replace_all": True}))
    assert out.get("replacements") == 2
    assert p.read_text(encoding="utf-8") == "a = 7\nb = 7\nc = 3\n"


def test_public_surface_is_unchanged():
    """PROPERTY: three modules import these by identity at import time
    (execution_engine, worker_loop, ws_server). The extraction must not
    perturb them."""
    assert len(SHELL_TOOLS) == 6
    assert set(SHELL_TOOL_EXECUTORS) == {
        "bash", "read_file", "write_file", "edit_file", "list_directory",
        "search_files"}


# --------------------------------------------------------------------------
# The manifest is honest
# --------------------------------------------------------------------------
def _bootstrap():
    d = Path(tempfile.mkdtemp())
    rt = SimpleNamespace(_broker_client=None, get_agent=lambda a: None)
    store = ToolStore(rt, d)
    rt.tool_store = store
    digest = bootstrap_reference_distro(rt)
    return store, digest


def test_atn_shell_manifest_declares_what_it_provides():
    """PROPERTY: the shell module's manifest describes a program — the tools
    it provides and the host access it genuinely needs."""
    store, _ = _bootstrap()
    rec = next(r for r in store._records.values() if r.name == "atn_shell")
    caps = rec.manifest.get("capabilities") or {}

    assert set(caps.get("provides") or []) == set(SHELL_TOOL_EXECUTORS)
    # A shell bundle needs all three by definition. Declaring honestly is the
    # point -- and is also exactly why tool_guard cannot contain it.
    assert caps.get("net") and caps.get("fs") and caps.get("spawn")


def test_atn_shell_blob_is_the_runnable_program():
    """PROPERTY: what the digest locks IS the program, not a description of
    one. This is the difference between the old identity manifest and now."""
    store, _ = _bootstrap()
    rec = next(r for r in store._records.values() if r.name == "atn_shell")
    raw = store._blob_store().get_bytes(rec.manifest["code_digest"])
    assert raw

    d = Path(tempfile.mkdtemp())
    script = d / "blob.py"
    script.write_bytes(raw)
    out = _run_as_subprocess(
        {"tool": "bash", "args": {"command": "echo from-blob"}}, script)
    assert out["ok"] is True
    assert "from-blob" in out["result"]["output"]


def test_daemon_coupled_bundles_do_not_claim_to_provide():
    """PROPERTY: only the shell bundle claims 'provides'. The other atn_*
    manifests are identity records over daemon internals; claiming otherwise
    would be a lie in the one place the network reads."""
    store, _ = _bootstrap()
    for rec in store._records.values():
        if not rec.name.startswith("atn_") or rec.name == "atn_shell":
            continue
        caps = rec.manifest.get("capabilities") or {}
        assert not caps.get("provides"), f"{rec.name} claims provides"


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
def test_swap_is_disabled():
    """PROPERTY: the gate is OFF. This test failing means someone enabled the
    swap -- read shell_provider.py's docstring before changing it, and do not
    'fix' this test to match."""
    assert shell_provider.SHELL_SWAP_ENABLED is False


def test_dispatch_shell_is_a_noop_while_gated():
    """PROPERTY: None means fall through to the built-in. Every shell call
    today takes this path."""
    rt = SimpleNamespace(tool_store=None)
    for name in SHELL_TOOL_EXECUTORS:
        got = asyncio.run(shell_provider.dispatch_shell(rt, "a1", name, {}))
        assert got is None, f"{name} was overridden while the gate is off"


def test_resolution_returns_nothing_while_gated():
    store, _ = _bootstrap()
    rt = SimpleNamespace(tool_store=store)
    assert shell_provider.resolve_shell_provider(rt, "a1") == {}


def test_provides_claim_is_clamped_at_registration():
    """PROPERTY: a manifest cannot invent a name to provide, nor claim a
    framework tool outside the overridable set."""
    store, _ = _bootstrap()
    res = store.register(
        name="my_shell_impl", description="d",
        input_schema={"type": "object"}, author="a1",
        code="import sys,json;json.loads(sys.stdin.read());print('{}')",
        capabilities={"provides": ["bash", "create_agent", "invented_name"]},
    )
    assert res["manifest"]["capabilities"]["provides"] == ["bash"]
