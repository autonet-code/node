"""Tests for the runtime AgentRegistry scope/lookup helpers added for
wallet-authenticated scoped rendering: get_agent_id_by_address and
get_subtree_ids.

These back the WS server's auth root-resolution (signer address -> agent) and
connection scoping (root -> the set of agents that connection may see)."""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentIdentity, AgentMode
from atn.runtime import Runtime


def _make_runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id: str, parent_id: str | None, *, address: str | None = None,
           mode=AgentMode.COGNITIVE) -> AgentDefinition:
    identity = None
    if address is not None:
        identity = AgentIdentity(public_key=address, address=address)
    return AgentDefinition(
        id=agent_id, name=agent_id, mode=mode, parent_id=parent_id,
        identity=identity, budgets={},
    )


async def _fleet(tmp_path) -> Runtime:
    """orchestrator -> a -> {a.1, a.2}, plus a sibling b. Addresses on a, a.1, b."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("orchestrator", None))
    await rt.registry.register_agent(_agent("a", "orchestrator", address="0xAAaa1111111111111111111111111111111111aA"))
    await rt.registry.register_agent(_agent("a.1", "a", address="0xB1b1000000000000000000000000000000000001"))
    await rt.registry.register_agent(_agent("a.2", "a"))
    await rt.registry.register_agent(_agent("b", "orchestrator", address="0xCCcc222222222222222222222222222222222222"))
    return rt


# ---------------------------------------------------------------------------
# get_agent_id_by_address
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_address_lookup_exact(tmp_path):
    rt = await _fleet(tmp_path)
    assert rt.registry.get_agent_id_by_address("0xAAaa1111111111111111111111111111111111aA") == "a"
    assert rt.registry.get_agent_id_by_address("0xCCcc222222222222222222222222222222222222") == "b"


@pytest.mark.asyncio
async def test_address_lookup_case_insensitive(tmp_path):
    rt = await _fleet(tmp_path)
    # Different casing than stored must still resolve.
    assert rt.registry.get_agent_id_by_address("0xaaaa1111111111111111111111111111111111aa") == "a"
    assert rt.registry.get_agent_id_by_address("0xCCCC222222222222222222222222222222222222") == "b"


@pytest.mark.asyncio
async def test_address_lookup_unknown_and_empty(tmp_path):
    rt = await _fleet(tmp_path)
    assert rt.registry.get_agent_id_by_address("0xdeadbeef00000000000000000000000000000000") is None
    assert rt.registry.get_agent_id_by_address("") is None
    # An agent with no identity (a.2) is never matched.
    assert rt.registry.get_agent_id_by_address("0x0000000000000000000000000000000000000000") is None


# ---------------------------------------------------------------------------
# get_subtree_ids
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subtree_root_includes_self_and_all_descendants(tmp_path):
    rt = await _fleet(tmp_path)
    assert rt.registry.get_subtree_ids("a") == {"a", "a.1", "a.2"}


@pytest.mark.asyncio
async def test_subtree_orchestrator_is_full_fleet(tmp_path):
    rt = await _fleet(tmp_path)
    assert rt.registry.get_subtree_ids("orchestrator") == {
        "orchestrator", "a", "a.1", "a.2", "b",
    }


@pytest.mark.asyncio
async def test_subtree_leaf_is_just_self(tmp_path):
    rt = await _fleet(tmp_path)
    assert rt.registry.get_subtree_ids("a.1") == {"a.1"}
    assert rt.registry.get_subtree_ids("b") == {"b"}


@pytest.mark.asyncio
async def test_subtree_unknown_id_returns_just_that_id(tmp_path):
    rt = await _fleet(tmp_path)
    # Caller decides whether an unknown root is an error; the helper is total.
    assert rt.registry.get_subtree_ids("ghost") == {"ghost"}


@pytest.mark.asyncio
async def test_subtree_terminates_on_parent_cycle(tmp_path):
    """A malformed parent cycle must not loop forever."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("x", None))
    await rt.registry.register_agent(_agent("y", "x"))
    # Force a cycle: x's parent becomes y (x -> y -> x).
    rt.registry._agents["x"].parent_id = "y"
    ids = rt.registry.get_subtree_ids("x")
    assert ids == {"x", "y"}
