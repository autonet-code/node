"""The adoption approval queue is owner-only (docs/tool_substrate.md — Adoption).

REGRESSION. These handlers previously sat behind Gate 2 (authenticated) and
nothing else: no scope check, absent from OWNER_ONLY_TOOLS, absent from
KEY_LOCAL_ONLY_MESSAGES. A SCOPED session — a delegated remote link, which by
design does not act for the owner — could call approve_adoption and install
foreign pinned code on the host. That was a strictly weaker gate than reading
secret NAMES, which _SECRETS_MESSAGES already refuses to scoped sessions.

The property under test: approving an adoption installs code that then RUNS on
this machine, so it must be at least as gated as every other owner act.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from atn.ws_server import WebSocketBridge


class _Store:
    """Records whether the store was reached at all — the gate must refuse
    BEFORE any store call, not sanitize afterwards."""

    def __init__(self):
        self.approve_calls = []
        self.list_calls = []
        self.reject_calls = []

    def list_adoption_proposals(self, status=None):
        self.list_calls.append(status)
        return []

    async def approve_adoption(self, digest):
        self.approve_calls.append(digest)
        return {"status": "approved", "digest": digest}

    def reject_adoption(self, digest, reason=""):
        self.reject_calls.append(digest)
        return {"status": "rejected"}


def _server():
    store = _Store()
    runtime = SimpleNamespace(tool_store=store)
    srv = WebSocketBridge.__new__(WebSocketBridge)
    srv.runtime = runtime
    return srv, store


def _session(*, scoped: bool):
    """A session that has already passed Gates 1 and 2 (authenticated).

    scoped=True  -> a delegated link (scope_ids names a subtree)
    scoped=False -> the owner's full-fleet session
    """
    return SimpleNamespace(
        scope_ids={"agent-1"} if scoped else None,
        local=not scoped,
        owner="0xowner",
        authed=True,
    )


def _call(srv, msg_type, session, **extra):
    msg = {"type": msg_type, "msg_id": "1", **extra}
    return asyncio.run(srv._handle_message(msg, session))


@pytest.mark.parametrize("msg_type,extra", [
    ("approve_adoption", {"digest": "a" * 64}),
    ("reject_adoption", {"digest": "a" * 64}),
    ("list_adoption_proposals", {}),
])
def test_scoped_session_cannot_touch_the_adoption_queue(msg_type, extra):
    """PROPERTY: a scoped session is refused, and the store is never reached."""
    srv, store = _server()
    resp = _call(srv, msg_type, _session(scoped=True), **extra)

    assert resp["ok"] is False
    assert "owner-only" in resp["error"]
    assert store.approve_calls == [], "store was reached despite refusal"
    assert store.reject_calls == []
    assert store.list_calls == []


def test_owner_session_can_still_approve():
    """PROPERTY: the fix must not break the legitimate path."""
    srv, store = _server()
    resp = _call(srv, "approve_adoption", _session(scoped=False),
                 digest="b" * 64)

    assert resp["ok"] is True
    assert store.approve_calls == ["b" * 64]


def test_owner_session_can_still_list():
    srv, store = _server()
    resp = _call(srv, "list_adoption_proposals", _session(scoped=False))
    assert resp["ok"] is True
    assert store.list_calls == [None]


def test_adoption_gate_is_at_least_as_strict_as_the_secrets_gate():
    """PROPERTY: installing executable code is never easier than reading a
    secret NAME. Pins the relative ordering, not a specific mechanism — if
    someone later loosens adoption, this fails even if the wording changed.
    """
    srv, _ = _server()
    scoped = _session(scoped=True)

    adoption = _call(srv, "approve_adoption", scoped, digest="c" * 64)
    secrets = _call(srv, "secrets_status", scoped)

    assert secrets["ok"] is False, "precondition: secrets refuse scoped sessions"
    assert adoption["ok"] is False, (
        "adoption is more permissive than secrets — installing foreign code "
        "must never be easier than reading a secret name")
