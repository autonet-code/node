"""Tests for the agent directory publisher (daemon -> Firestore reachability).

Stubs the Firestore client so no network/creds are needed. Verifies the merge
payload shape, doc-id = address, the indexer-fields-untouched contract (merge),
and that publishing is skipped when no endpoint is configured."""
from __future__ import annotations

import pytest

from atn.agent_directory import AgentDirectoryPublisher


class _FakeDoc:
    def __init__(self, store, addr):
        self._store, self._addr = store, addr

    def set(self, payload, merge=False):
        self._store.writes.append((self._addr, dict(payload), merge))


class _FakeCollection:
    def __init__(self, store):
        self._store = store

    def document(self, addr):
        return _FakeDoc(self._store, addr)


class _FakeFirestore:
    def __init__(self):
        self.writes = []  # (addr, payload, merge)

    def collection(self, name):
        assert name == "agents"
        return _FakeCollection(self)


class _FakeRegistry:
    def __init__(self, ads):
        self._ads = ads

    def build_agent_advertisements(self):
        return self._ads


class _FakeRuntime:
    def __init__(self, ads):
        self.registry = _FakeRegistry(ads)


def _pub(ads, endpoint="wss://autonet.computer/ws"):
    pub = AgentDirectoryPublisher(_FakeRuntime(ads), endpoint=endpoint)
    pub._client = _FakeFirestore()       # inject stub; skip real _connect()
    return pub


@pytest.mark.asyncio
async def test_publishes_directory_entry_for_each_agent_with_identity():
    ads = [
        {"address": "0xAaa", "name": "Alpha", "agent_type": "research"},
        {"address": "0xBbb", "name": "Beta", "agent_type": ""},
        {"address": "", "name": "NoAddr"},   # no address -> skipped
    ]
    pub = _pub(ads)
    await pub._publish_once()
    writes = {w[0]: w for w in pub._client.writes}
    assert set(writes.keys()) == {"0xAaa", "0xBbb"}   # empty-address agent skipped
    for addr, (_a, payload, merge) in writes.items():
        assert payload["wsEndpoint"] == "wss://autonet.computer/ws"
        assert isinstance(payload["lastSeen"], int)
        assert merge is True   # must merge so indexer's chain fields survive
        # Daemon-owned fields only: reachability + display name/type. Never
        # address/peerId/registration (those are the indexer's).
        assert set(payload.keys()) == {"wsEndpoint", "lastSeen", "displayName", "agentType"}
    # Name + type carried through from the advertisement.
    assert writes["0xAaa"][1]["displayName"] == "Alpha"
    assert writes["0xAaa"][1]["agentType"] == "research"
    assert writes["0xBbb"][1]["displayName"] == "Beta"


@pytest.mark.asyncio
async def test_no_agents_writes_nothing():
    pub = _pub([])
    await pub._publish_once()
    assert pub._client.writes == []


@pytest.mark.asyncio
async def test_disabled_when_no_endpoint():
    # Empty endpoint => publisher is a no-op and never touches Firestore.
    pub = AgentDirectoryPublisher(_FakeRuntime([{"address": "0xAaa"}]), endpoint="")
    assert pub._enabled is False
    await pub.start()           # must not raise, must not connect
    assert pub._client is None


@pytest.mark.asyncio
async def test_publish_swallows_firestore_errors():
    class _Boom(_FakeFirestore):
        def collection(self, name):
            raise RuntimeError("firestore down")

    pub = _pub([{"address": "0xAaa"}])
    pub._client = _Boom()
    # A Firestore failure must never propagate (best-effort presence write).
    await pub._publish_once()
