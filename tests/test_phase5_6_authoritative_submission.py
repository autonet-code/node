"""Authoritative claims: a household reads the anchored claims blob and
claims its leaf on chain via ``Substrate.claim``, then withdraws each
token it is owed.

Tested:

  1. End to end: close -> anchor -> the submitter reads the anchor, checks
     the blob against its cid, proves its leaf, claims (signed by an
     operator that is not the earner) and withdraws every token to the
     earner.
  2. Idempotency: a second pass and a restarted submitter claim nothing
     twice (``claimedEpoch``).
  3. Three households claim independently; the amounts on chain are the
     anchored ones, token by token.
  4. A tampered blob raises; a missing blob fails cleanly; an earner with
     no leaf sends nothing; an unanchored epoch fails cleanly.
  5. A token that cannot be withdrawn (its issuer paused it) blocks none of
     the others and is retried at the next pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.authoritative_submitter import (
    AuthoritativeChainSubmitter,
    AuthoritativeSubmitterConfig,
    epoch_id_hash,
)
from nodes.common.blob_resolver import BlobIntegrityError, InMemoryBlobResolver
from nodes.common.canonical_ordering import canonical_order
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from tests.substrate_chain_helpers import (
    _deploy,
    artifact,
    deploy_substrate,
    index_tokens,
    register_bound,
    review_snapshot,
    seed_pool,
)

SUBSTRATE_JSON = Path(__file__).resolve().parent.parent / "artifacts/contracts/core/Substrate.sol/Substrate.json"
REVIEWER = "0x" + "9e" * 20


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _fixture(*, registry=None, tokens=None) -> Dict[str, Any]:
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    accounts = w3.eth.accounts
    deployer = accounts[0]
    addr = deploy_substrate(w3, abi, bytecode, deployer, registry=registry)
    contract = w3.eth.contract(address=addr, abi=abi)
    agent_addrs = accounts[1:4]
    for i, a in enumerate(agent_addrs):
        assert register_bound(w3, contract, a, f"test-peer-{i}".encode()).status == 1
    tokens = tokens if tokens is not None else index_tokens(contract)
    window = seed_pool(w3, contract, tokens=tokens)
    return {"w3": w3, "abi": abi, "addr": addr, "contract": contract,
            "deployer": deployer, "agent_addrs": list(agent_addrs),
            "tokens": tokens, "window": window}


@pytest.fixture
def chain():
    return _fixture()


def _registration(rpb: str, agent_id: str) -> EventBatch:
    digest = hashlib.sha256(f"tool-{agent_id}".encode()).hexdigest()
    ev = {"kind": "tool_registered", "seq": 1, "author_agent": agent_id,
          "artifact_digest": digest,
          "manifest_meta": {"trust_class": "pinned", "author": agent_id}}
    return EventBatch(rpb_address=rpb, sender_pubkey=Keypair.generate().public_key,
                      batch_seq=1, events=[ev], prev_batch_hash=b"",
                      timestamp=1_700_000_000.0)


def _close_and_anchor(fx, agent_ids: List[str], epoch_id: str):
    """A close in which every author's tool concludes over the seeded
    window, anchored; returns (close result, resolver holding the blob)."""
    canonical = canonical_order([_registration("rpb_5_6", a) for a in agent_ids])
    regs = federated_epoch_close(canonical)["tool_registrations"]
    result = federated_epoch_close(
        canonical, rep_shares={REVIEWER: 0.5},
        review_snapshot=review_snapshot(list(regs), REVIEWER),
        tool_share_window=dict(fx["window"]),
        parity={t.address.lower(): 10 ** (36 - int(t.functions.decimals().call()))
                for t in fx["tokens"]})
    result["epoch_id"] = epoch_id
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=fx["addr"]),
        web3=fx["w3"], contract_abi=fx["abi"])
    a = anchorer.anchor_close_result(result, from_address=fx["deployer"])
    assert a.success, a.error
    resolver = InMemoryBlobResolver()
    assert resolver.put(a.agent_mint_blob) == a.agent_mint_cid
    return result, resolver


def _submitter(fx, earner: str, resolver, claimant_id=None):
    return AuthoritativeChainSubmitter(
        claimant_id=claimant_id or earner.lower(), claimant_address=earner,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=fx["addr"]),
        sender_address=fx["deployer"], web3=fx["w3"], substrate_abi=fx["abi"])


def _balances(fx, who):
    return {t.address.lower(): t.functions.balanceOf(who).call() for t in fx["tokens"]}


def test_household_claims_its_leaf_and_withdraws_every_token(chain):
    result, resolver = _close_and_anchor(chain, chain["agent_addrs"], "e_5_6")
    earner = chain["agent_addrs"][0]
    expected = result["claims"][earner.lower()]["tokens"]
    assert len(expected) == 2
    eid = epoch_id_hash("e_5_6")
    assert not chain["contract"].functions.claimedEpoch(earner, eid).call()
    op_before = _balances(chain, chain["deployer"])

    r = _submitter(chain, earner, resolver).submit_for_epoch("e_5_6")
    assert r.success, r.error
    assert r.cid_verified and r.tokens == expected
    assert r.value == result["claims"][earner.lower()]["value"]
    assert sorted(r.withdrawn) == sorted(expected) and not r.withdraw_errors
    assert chain["contract"].functions.claimedEpoch(earner, eid).call()
    # Substrate values the claim at the parity it stored at the anchor; with
    # the index unchanged since the snapshot it is the close's value.
    assert chain["contract"].functions.earnedValue(earner).call() == r.value > 0
    # The tokens land on the earner, not the operator that sent the txs.
    assert _balances(chain, earner) == expected
    assert _balances(chain, chain["deployer"]) == op_before


def test_double_submit_and_restart_claim_once(chain):
    result, resolver = _close_and_anchor(chain, chain["agent_addrs"][:2], "e_idem")
    earner = chain["agent_addrs"][0]
    s1 = _submitter(chain, earner, resolver)
    assert s1.submit_for_epoch("e_idem").success
    after = _balances(chain, earner)
    again = s1.submit_for_epoch("e_idem")
    assert again.success and again.tx_hash == "" and not again.withdrawn
    restarted = _submitter(chain, earner, resolver).submit_for_epoch("e_idem")
    assert restarted.success and restarted.tx_hash == ""
    assert "already claimed" in restarted.error
    assert _balances(chain, earner) == after


def test_three_households_claim_independently(chain):
    result, resolver = _close_and_anchor(chain, chain["agent_addrs"], "e_three")
    c = chain["contract"]
    for earner in chain["agent_addrs"]:
        r = _submitter(chain, earner, resolver).submit_for_epoch("e_three")
        assert r.success, r.error
        assert _balances(chain, earner) == result["claims"][earner.lower()]["tokens"]
    for t in chain["tokens"]:
        k = t.address.lower()
        paid = sum(result["claims"][a.lower()]["tokens"][k] for a in chain["agent_addrs"])
        # The reviewer's own leaf stays unclaimed here.
        assert c.functions.claimedTotal(t.address).call() == paid
        assert paid + result["claims"][REVIEWER]["tokens"][k] <= chain["window"][k]


def test_tampered_blob_raises_integrity_error(chain):
    _, resolver = _close_and_anchor(chain, chain["agent_addrs"][:1], "e_tamper")
    real_cid = list(resolver._store.keys())[0]
    resolver._store[real_cid] = b'{"schema":9,"claims":{}}'
    with pytest.raises(BlobIntegrityError):
        _submitter(chain, chain["agent_addrs"][0], resolver).submit_for_epoch("e_tamper")


def test_missing_blob_returns_clean_failure(chain):
    _, resolver = _close_and_anchor(chain, chain["agent_addrs"][:1], "e_missing")
    resolver._store.clear()
    r = _submitter(chain, chain["agent_addrs"][0], resolver).submit_for_epoch("e_missing")
    assert not r.success and "blob not found" in r.error.lower()


def test_earner_without_a_leaf_sends_nothing(chain):
    _, resolver = _close_and_anchor(chain, chain["agent_addrs"][:1], "e_zero")
    other = chain["agent_addrs"][2]
    r = _submitter(chain, other, resolver).submit_for_epoch("e_zero")
    assert r.success and r.tx_hash == "" and r.tokens == {}
    assert not chain["contract"].functions.claimedEpoch(other, epoch_id_hash("e_zero")).call()


def test_unanchored_epoch_fails_cleanly(chain):
    resolver = InMemoryBlobResolver()
    resolver.put(b'{"schema":9,"claims":{}}')
    r = _submitter(chain, chain["agent_addrs"][0], resolver).submit_for_epoch("e_never")
    assert not r.success and r.error


def test_a_paused_token_blocks_nothing_and_is_retried():
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    reg = _deploy(w3, artifact("test/IndexTokenMocks.sol/MockRegistry"), deployer)
    paused = _deploy(w3, artifact("test/IndexTokenMocks.sol/PausableToken"), deployer)
    dai = _deploy(w3, artifact("test/IndexTokenMocks.sol/MockIndexToken"), deployer,
                  "D", "D", 18)
    for t, p in ((paused, 10 ** 30), (dai, 10 ** 18)):
        assert w3.eth.wait_for_transaction_receipt(reg.functions.setParity(
            t.address, p).transact({"from": deployer})).status == 1
    # A fresh chain per fixture: build it on this provider.
    abi, bytecode = _load_substrate()
    addr = deploy_substrate(w3, abi, bytecode, deployer, registry=reg.address)
    contract = w3.eth.contract(address=addr, abi=abi)
    agents = w3.eth.accounts[1:3]
    for i, a in enumerate(agents):
        assert register_bound(w3, contract, a, f"p-{i}".encode()).status == 1
    window = seed_pool(w3, contract, tokens=[paused, dai])
    fx = {"w3": w3, "abi": abi, "addr": addr, "contract": contract,
          "deployer": deployer, "agent_addrs": list(agents),
          "tokens": [paused, dai], "window": window}
    result, resolver = _close_and_anchor(fx, agents, "e_pause")
    earner = agents[0]
    owed = result["claims"][earner.lower()]["tokens"]

    w3.eth.wait_for_transaction_receipt(paused.functions.setPaused(True).transact(
        {"from": deployer}))
    sub = _submitter(fx, earner, resolver)
    r = sub.submit_for_epoch("e_pause")
    assert r.success, r.error                       # the claim itself landed
    assert list(r.withdrawn) == [dai.address.lower()]
    assert list(r.withdraw_errors) == [paused.address.lower()]
    assert dai.functions.balanceOf(earner).call() == owed[dai.address.lower()]
    assert contract.functions.owed(earner, paused.address).call() == owed[paused.address.lower()]

    # Unpaused: the next pass withdraws what is left, and nothing twice.
    w3.eth.wait_for_transaction_receipt(paused.functions.setPaused(False).transact(
        {"from": deployer}))
    r2 = sub.submit_for_epoch("e_pause")
    assert r2.success and list(r2.withdrawn) == [paused.address.lower()]
    assert paused.functions.balanceOf(earner).call() == owed[paused.address.lower()]
    assert contract.functions.owed(earner, paused.address).call() == 0
