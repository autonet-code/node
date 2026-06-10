"""Tests for the agent directory publisher (daemon -> on-chain wss endpoint).

The daemon publishes its browser-reachable wss endpoint by signing an
``updateEndpoint`` tx per hosted, registered agent — only when the endpoint
differs from what's already on-chain. It never touches Firestore (an indexer
mirrors the EndpointUpdated event off-chain). These tests stub OnChainService
so no chain/creds are needed."""
from __future__ import annotations

import pytest

from atn.agent_directory import AgentDirectoryPublisher


class _FakeIdentity:
    def __init__(self, address, registered=True):
        self.address = address
        self.registered_on_chain = registered


class _FakeDefn:
    def __init__(self, id, address, registered=True):
        self.id = id
        self.identity = _FakeIdentity(address, registered) if address else None


class _FakeRegistry:
    def __init__(self, defns, keys=None):
        self._defns = defns
        self._keys = keys or {d.id: f"key-{d.id}" for d in defns}

    def list_agents(self):
        return [(d, "active") for d in self._defns]

    def get_agent_key(self, agent_id):
        return self._keys.get(agent_id)


class _FakeSvc:
    """Stub OnChainService: records update_endpoint calls; returns a
    preset on-chain endpoint per address from get_agent_endpoint."""
    available = True

    def __init__(self, onchain=None):
        self._onchain = onchain or {}     # address -> current on-chain endpoint
        self.updates = []                  # (private_key, ws_endpoint)

    async def get_agent_endpoint(self, address):
        return self._onchain.get(address, "")

    async def update_endpoint(self, private_key, ws_endpoint):
        self.updates.append((private_key, ws_endpoint))
        return {"success": True, "tx_hash": "0xabc"}


class _Cfg:
    class autonet:  # noqa: N801 — mimics config.autonet attr access
        pass


class _FakeRuntime:
    def __init__(self, registry):
        self.registry = registry
        self._config = _Cfg()


def _pub(defns, *, endpoint="wss://autonet.computer/ws", svc=None, keys=None):
    pub = AgentDirectoryPublisher(_FakeRuntime(_FakeRegistry(defns, keys)),
                                  endpoint=endpoint)
    pub._svc = lambda: svc          # inject stub service
    return pub


@pytest.mark.asyncio
async def test_publishes_for_registered_agents_when_changed():
    svc = _FakeSvc()  # nothing on-chain yet -> all differ
    defns = [_FakeDefn("a", "0xAaa"), _FakeDefn("b", "0xBbb")]
    pub = _pub(defns, svc=svc)
    await pub._publish_once()
    assert len(svc.updates) == 2
    assert {ws for _k, ws in svc.updates} == {"wss://autonet.computer/ws"}
    assert {k for k, _ws in svc.updates} == {"key-a", "key-b"}


@pytest.mark.asyncio
async def test_skips_when_endpoint_unchanged():
    # 'a' already has our endpoint on-chain -> skip; 'b' differs -> publish.
    svc = _FakeSvc(onchain={"0xAaa": "wss://autonet.computer/ws"})
    defns = [_FakeDefn("a", "0xAaa"), _FakeDefn("b", "0xBbb")]
    pub = _pub(defns, svc=svc)
    await pub._publish_once()
    assert len(svc.updates) == 1
    assert svc.updates[0][0] == "key-b"


@pytest.mark.asyncio
async def test_skips_unregistered_and_keyless_and_identityless():
    svc = _FakeSvc()
    defns = [
        _FakeDefn("reg", "0xReg", registered=True),
        _FakeDefn("unreg", "0xUnreg", registered=False),  # not on chain
        _FakeDefn("noid", None),                            # no identity
    ]
    keys = {"reg": "key-reg", "unreg": "key-unreg"}        # noid has no key
    pub = _pub(defns, svc=svc, keys=keys)
    await pub._publish_once()
    assert [k for k, _ in svc.updates] == ["key-reg"]


@pytest.mark.asyncio
async def test_disabled_when_no_endpoint():
    pub = AgentDirectoryPublisher(_FakeRuntime(_FakeRegistry([])), endpoint="")
    assert pub._enabled is False
    await pub.start()   # no-op, must not raise


@pytest.mark.asyncio
async def test_no_chain_configured_is_noop():
    pub = _pub([_FakeDefn("a", "0xAaa")], svc=None)  # _svc() returns None
    await pub._publish_once()   # must not raise


@pytest.mark.asyncio
async def test_update_failure_is_swallowed():
    class _Boom(_FakeSvc):
        async def update_endpoint(self, k, ws):
            return {"success": False, "error": "reverted"}
    pub = _pub([_FakeDefn("a", "0xAaa")], svc=_Boom())
    await pub._publish_once()   # logged, not raised
