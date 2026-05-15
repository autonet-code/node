"""Phase 13.3: daemon-startup reconciliation of cached ``registered_on_chain``
flags against ``Substrate.areRegistered([...])``.

When the Substrate contract is redeployed between daemon runs, the cached
``registered_on_chain=True`` flag on each agent's ``identity.json`` is no
longer truthful — the new contract has no record of that agent. The
reconciliation pass exists to detect and clear that mismatch.

These tests exercise the logic in isolation by patching the chain client.
We don't need real chain access; we just need to assert that:

  * stale flags are flipped to ``False`` AND persisted
  * still-registered agents are untouched
  * unset chain config short-circuits cleanly
  * RPC failures don't corrupt local state
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test fixtures — minimal config + registry shims
# ---------------------------------------------------------------------------

class _MockIdentity:
    """Stand-in for ``atn.models.AgentIdentity`` — we only care about three fields."""
    def __init__(self, address: str, registered: bool, tx: str | None = "0xprior"):
        self.address = address
        self.registered_on_chain = registered
        self.registration_tx = tx


class _MockDefn:
    def __init__(self, agent_id: str, identity: _MockIdentity | None):
        self.id = agent_id
        self.identity = identity


def _make_registry_with_agents(agents: dict[str, _MockDefn], *,
                                substrate_address: str = "0xa38201B9290EBe5FEf989274Ae7Edc43Ac6531C3",
                                rpc_url: str = "https://rpc.example") -> object:
    """Build a stub registry that mimics the relevant slice of AgentRegistry.

    We can't instantiate the real ``AgentRegistry`` cheaply (it pulls in
    inbox/output/execution stores), so we lift the bound method onto a
    bare object that exposes only what ``reconcile_chain_registrations``
    actually touches: ``self._agents``, ``self._config``,
    ``self.persist_identity``.
    """
    from atn.runtime.agent_registry import AgentRegistry

    # Mirror the real shape: ``ATNConfig.rpb`` holds the substrate fields.
    config = SimpleNamespace(
        rpb=SimpleNamespace(
            substrate_address=substrate_address,
            rpc_url=rpc_url,
        ),
    )
    obj = SimpleNamespace(
        _agents=agents,
        _config=config,
        persist_identity=MagicMock(),
    )
    obj.reconcile_chain_registrations = (
        AgentRegistry.reconcile_chain_registrations.__get__(obj, AgentRegistry)
    )
    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flips_stale_flag_when_chain_says_unregistered() -> None:
    stale = _MockDefn("stale", _MockIdentity("0xAAA", registered=True))
    real = _MockDefn("real",  _MockIdentity("0xBBB", registered=True))
    reg = _make_registry_with_agents({"stale": stale, "real": real})

    fake_contract = MagicMock()
    # areRegistered returns [False, True] — first address no longer there.
    fake_contract.functions.areRegistered.return_value.call.return_value = [False, True]
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3") as mock_w3_cls:
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = fake_contract
        mock_w3_cls.return_value = mock_w3
        mock_w3_cls.to_checksum_address.side_effect = lambda a: a
        result = reg.reconcile_chain_registrations()

    assert result["checked"] == 2
    assert result["flipped"] == ["stale"]
    assert stale.identity.registered_on_chain is False
    assert stale.identity.registration_tx is None
    assert real.identity.registered_on_chain is True   # unchanged
    reg.persist_identity.assert_called_once_with("stale")


def test_skips_agents_without_identity() -> None:
    no_identity = _MockDefn("no_identity", identity=None)
    not_registered = _MockDefn(
        "not_registered", _MockIdentity("0xCCC", registered=False),
    )
    real = _MockDefn("real", _MockIdentity("0xDDD", registered=True))
    reg = _make_registry_with_agents({
        "no_identity": no_identity,
        "not_registered": not_registered,
        "real": real,
    })

    fake_contract = MagicMock()
    fake_contract.functions.areRegistered.return_value.call.return_value = [True]
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3") as mock_w3_cls:
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = fake_contract
        mock_w3_cls.return_value = mock_w3
        mock_w3_cls.to_checksum_address.side_effect = lambda a: a
        result = reg.reconcile_chain_registrations()

    # Only the one agent claiming registration should be checked.
    assert result["checked"] == 1
    assert result["flipped"] == []
    # Confirm we passed exactly the expected address.
    args = fake_contract.functions.areRegistered.call_args[0][0]
    assert args == ["0xDDD"]


def test_noop_when_chain_unset() -> None:
    agent = _MockDefn("a", _MockIdentity("0xAAA", registered=True))
    reg = _make_registry_with_agents(
        {"a": agent}, substrate_address="", rpc_url="",
    )

    result = reg.reconcile_chain_registrations()

    assert result == {"checked": 0, "flipped": [], "skipped_reason": "chain_unset"}
    # Critically: the agent's flag is NOT cleared just because chain is offline.
    assert agent.identity.registered_on_chain is True
    reg.persist_identity.assert_not_called()


def test_rpc_failure_leaves_flags_untouched() -> None:
    agent = _MockDefn("a", _MockIdentity("0xAAA", registered=True))
    reg = _make_registry_with_agents({"a": agent})

    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3") as mock_w3_cls:
        mock_w3 = MagicMock()
        mock_w3.eth.contract.side_effect = RuntimeError("rpc kaput")
        mock_w3_cls.return_value = mock_w3
        mock_w3_cls.to_checksum_address.side_effect = lambda a: a
        result = reg.reconcile_chain_registrations()

    assert result["error"] == "rpc_failed"
    assert result["checked"] == 1
    assert result["flipped"] == []
    # Critical: a flaky RPC must not falsify the local view.
    assert agent.identity.registered_on_chain is True
    reg.persist_identity.assert_not_called()


def test_empty_agent_set_is_a_clean_noop() -> None:
    reg = _make_registry_with_agents({})

    result = reg.reconcile_chain_registrations()

    assert result == {"checked": 0, "flipped": []}
    reg.persist_identity.assert_not_called()


def test_persist_failure_does_not_abort_remaining_flips() -> None:
    """If persisting one stale flip throws, we still process the rest."""
    bad = _MockDefn("bad", _MockIdentity("0xAAA", registered=True))
    good = _MockDefn("good", _MockIdentity("0xBBB", registered=True))
    reg = _make_registry_with_agents({"bad": bad, "good": good})

    # First persist call raises; second succeeds.
    def _persist(agent_id):
        if agent_id == "bad":
            raise RuntimeError("disk full")

    reg.persist_identity = MagicMock(side_effect=_persist)

    fake_contract = MagicMock()
    fake_contract.functions.areRegistered.return_value.call.return_value = [False, False]
    with patch("web3.Web3.HTTPProvider"), \
         patch("web3.Web3") as mock_w3_cls:
        mock_w3 = MagicMock()
        mock_w3.eth.contract.return_value = fake_contract
        mock_w3_cls.return_value = mock_w3
        mock_w3_cls.to_checksum_address.side_effect = lambda a: a
        result = reg.reconcile_chain_registrations()

    # Both flips happen in memory; both end up in the returned list.
    assert set(result["flipped"]) == {"bad", "good"}
    assert bad.identity.registered_on_chain is False
    assert good.identity.registered_on_chain is False
    assert reg.persist_identity.call_count == 2
