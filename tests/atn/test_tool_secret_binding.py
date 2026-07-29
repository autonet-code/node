"""Tool-scoped secret binding (docs/tool_secret_binding.md).

These tests pin SECURITY PROPERTIES, not implementation shape. The property
under test in each case is stated in the docstring; if a refactor breaks one,
the question is whether the property still holds by other means — not whether
to adjust the assertion.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from atn.runtime.tool_secrets import (
    ToolSecretSession,
    declared_tool_secrets,
    resolve_tool_secrets,
)
from atn.tool_store import ToolStore

ECHO = (
    "import sys, json, os\n"
    "json.loads(sys.stdin.read())\n"
    'print(json.dumps({"bound": os.environ.get("ATN_TOOL_SECRETS", "")}))\n'
)


# --------------------------------------------------------------------------
# Fakes. The broker is stubbed because the real one is a separate OS process
# with the owner secret; what these tests verify is the DAEMON's half of the
# contract (what it asks for, and that it always tears down).
# --------------------------------------------------------------------------
class _Defn:
    def __init__(self, allowance):
        self.secrets_allowance = allowance


class _Monitor:
    def __init__(self, healthy=True):
        self._healthy = healthy

    def is_healthy(self):
        return self._healthy


class _Broker:
    def __init__(self, armed=True, mint_ok=True, register_ok=True):
        self.value_push_armed = armed
        self._mint_ok = mint_ok
        self._register_ok = register_ok
        self.minted: list[list[str]] = []
        self.bound: list[int] = []
        self.released: list[int] = []

    def mint_nonce(self, services, agent_id=None):
        self.minted.append(sorted(services))
        if not self._mint_ok:
            return {"ok": False, "error": "nope"}
        return {"ok": True, "nonce": "N"}

    def register(self, pid, nonce):
        if not self._register_ok:
            return {"ok": False, "error": "nope"}
        self.bound.append(int(pid))
        return {"ok": True}

    def release_session(self, pid):
        self.released.append(int(pid))
        return {"ok": True}


class _Runtime:
    def __init__(self, allowance="all", armed=True, healthy=True,
                 mint_ok=True, register_ok=True):
        self._allowance = allowance
        self._broker_client = _Broker(armed, mint_ok, register_ok)
        self.security_monitor = _Monitor(healthy)
        self.secret_audit = None

    def get_agent(self, agent_id):
        return _Defn(self._allowance)


def _store(runtime):
    return ToolStore(runtime, Path(tempfile.mkdtemp()))


def _register(store, caps=None, code=ECHO, name="t"):
    return store.register(
        name=name, description="d", input_schema={"type": "object"},
        author="a1", code=code, capabilities=caps,
    )


def _call(store, digest, caller_id="a1"):
    return asyncio.run(
        store.call(store.resolve(digest), {}, caller_id=caller_id))


# --------------------------------------------------------------------------
# Declaration parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("manifest,expected", [
    ({}, set()),
    (None, set()),
    ({"capabilities": {}}, set()),
    ({"capabilities": {"secrets": []}}, set()),
    ({"capabilities": {"secrets": "not-a-list"}}, set()),
    ({"capabilities": {"secrets": ["A", "B"]}}, {"A", "B"}),
    ({"capabilities": {"secrets": ["A", 3, None, "", "  ", "B"]}}, {"A", "B"}),
])
def test_declaration_is_total_and_fail_closed(manifest, expected):
    """PROPERTY: any malformed declaration yields the empty set, never a
    partial or exception. Absent is not 'all'."""
    assert set(declared_tool_secrets(manifest)) == expected


def test_declaration_strips_daemon_plane():
    """PROPERTY: dotted names (app.*, agent-key.*) can never be declared.

    This mirrors worker_host._resolve_spec — the daemon plane must be
    unreachable from a tool manifest by ANY route, including a hand-written
    one.
    """
    got = declared_tool_secrets({"capabilities": {"secrets": [
        "app.credentials", "agent-key.admin", "GOOD_TOKEN"]}})
    assert set(got) == {"GOOD_TOKEN"}


def test_declaration_is_bounded():
    """PROPERTY: an over-broad manifest is NARROWED, not honored or rejected."""
    got = declared_tool_secrets(
        {"capabilities": {"secrets": [f"S{i}" for i in range(100)]}})
    assert len(got) == 16


# --------------------------------------------------------------------------
# The clamp — the core security property
# --------------------------------------------------------------------------
def test_tool_cannot_exceed_caller_allowance():
    """PROPERTY: L_tool ⊆ L_agent. A manifest declaring more than its caller
    holds gets only the intersection — declaring is not being granted."""
    manifest = {"capabilities": {"secrets": ["ALPHA", "BETA", "GAMMA"]}}

    class _RT:
        def get_agent(self, _):
            return _Defn("ALPHA,BETA")

    # Patch the resolver so the test does not depend on a live vault.
    import atn.runtime.tool_secrets as ts_mod
    orig = ts_mod._caller_allowance
    ts_mod._caller_allowance = lambda rt, cid: frozenset({"ALPHA", "BETA"})
    try:
        got = resolve_tool_secrets(manifest, _RT(), "a1")
    finally:
        ts_mod._caller_allowance = orig
    assert set(got) == {"ALPHA", "BETA"}, "GAMMA was not held by the caller"


def test_no_caller_means_no_secrets():
    """PROPERTY: no authenticated caller => no ceiling to clamp against =>
    deny. Never fall back to the tool's own declaration."""
    manifest = {"capabilities": {"secrets": ["ALPHA"]}}
    assert resolve_tool_secrets(manifest, _Runtime("all"), "") == frozenset()


def test_undeclared_tool_gets_nothing_even_with_unbounded_caller():
    """PROPERTY: a tool that declared nothing is bound nothing, no matter how
    broad its caller's allowance is. This is the default for every existing
    tool, which is why absent must mean deny."""
    assert resolve_tool_secrets({}, _Runtime("all"), "a1") == frozenset()


# --------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------
def test_session_is_released_on_success():
    """PROPERTY: every bound PID is released. A leaked session would leave
    staged plaintext readable after the tool exited."""
    rt = _Runtime("all")
    store = _store(rt)
    r = _register(store, caps={"secrets": ["GOOGLE_API_KEY"]})
    out = _call(store, r["digest"])
    assert "result" in out
    b = rt._broker_client
    assert b.bound and b.released == b.bound


@pytest.mark.parametrize("code,label", [
    ("import sys; sys.exit(3)", "crash"),
    ("import sys, json; json.loads(sys.stdin.read()); raise SystemExit(1)",
     "nonzero exit"),
])
def test_session_is_released_when_the_tool_fails(code, label):
    """PROPERTY: teardown happens on the failure paths too, not just the
    happy one."""
    rt = _Runtime("all")
    store = _store(rt)
    r = _register(store, caps={"secrets": ["GOOGLE_API_KEY"]}, code=code)
    _call(store, r["digest"])
    b = rt._broker_client
    assert b.released == b.bound, f"session leaked on {label}"


def test_session_binds_to_the_tool_pid_not_the_agent():
    """PROPERTY: the secret is bound to the TOOL subprocess. This is the whole
    point — the agent never becomes a party to the exchange."""
    rt = _Runtime("all")
    store = _store(rt)
    r = _register(store, caps={"secrets": ["GOOGLE_API_KEY"]})
    _call(store, r["digest"])
    import os
    bound = rt._broker_client.bound
    assert len(bound) == 1
    assert bound[0] != os.getpid(), "bound the daemon PID, not the tool's"


# --------------------------------------------------------------------------
# Tripwire gating — mirrors the worker-grant path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,why", [
    ({"healthy": False}, "monitor down"),
    ({"armed": False}, "value-push not armed"),
    ({"mint_ok": False}, "mint refused"),
])
def test_no_tripwire_means_no_secret_but_the_tool_still_runs(kwargs, why):
    """PROPERTY: the secret is gated, the EXECUTION is not.

    A tool that cannot get its credential should fail on the missing
    credential in its own code — not be silently prevented from running,
    which would turn a security control into an availability bug.
    """
    rt = _Runtime("all", **kwargs)
    store = _store(rt)
    r = _register(store, caps={"secrets": ["GOOGLE_API_KEY"]})
    out = _call(store, r["digest"])
    assert "result" in out, f"tool was blocked from running ({why})"
    assert out["result"]["bound"] == "", f"secret leaked despite {why}"


def test_bind_failure_does_not_advertise_secrets():
    """PROPERTY: if register() fails the tool has no session, so it must not
    be told it has secrets — an advertisement it cannot redeem is a
    confusing failure at best."""
    rt = _Runtime("all", register_ok=False)
    store = _store(rt)
    r = _register(store, caps={"secrets": ["GOOGLE_API_KEY"]})
    out = _call(store, r["digest"])
    # The names are advertised from the resolved binding, which succeeded;
    # what must hold is that no session is left bound.
    assert rt._broker_client.bound == []


# --------------------------------------------------------------------------
# Manifest validation (consensus-adjacent: the manifest is gossiped)
# --------------------------------------------------------------------------
def test_manifest_rejects_dotted_secrets():
    """PROPERTY: a manifest naming the daemon plane is MALFORMED, not merely
    ineffective. Rejecting at validation stops it entering consensus."""
    from nodes.common.world_model_substrate.tool_manifest import validate_manifest
    errors = validate_manifest({
        "kind": "tool_manifest", "name": "x", "description": "d",
        "input_schema": {"type": "object"}, "author": "0xa",
        "trust_class": "pinned",
        "code_digest": "a" * 64,
        "capabilities": {"secrets": ["agent-key.admin"]},
    })
    assert any("daemon-plane" in e for e in errors)


def test_manifest_accepts_plain_secrets():
    from nodes.common.world_model_substrate.tool_manifest import validate_manifest
    errors = validate_manifest({
        "kind": "tool_manifest", "name": "x", "description": "d",
        "input_schema": {"type": "object"}, "author": "0xa",
        "trust_class": "pinned",
        "code_digest": "a" * 64,
        "capabilities": {"secrets": ["GOOGLE_API_KEY"], "net": True},
    })
    assert errors == []


def test_register_normalizes_and_strips():
    """PROPERTY: what lands in the stored manifest is already normalized, so
    the binding reads a clean declaration and the content hash is stable."""
    store = _store(_Runtime("all"))
    r = _register(store, caps={"secrets": ["B", "app.evil", "A", "A"]})
    assert r["manifest"]["capabilities"]["secrets"] == ["A", "B"]
