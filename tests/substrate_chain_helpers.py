"""Shared eth_tester helpers for Substrate (governed anchors, 2026-10-16).

``Substrate.anchor`` is governor-only (production: the DAO timelock). The
Python chain tests deploy with an EOA governor (the deployer, eth_tester
account 0, by default) and anchor directly from it. Every claim must also
be covered by the on-chain pools, so:

  - ``deploy_index``: a MockRegistry holding the value index and two
    MockIndexTokens in it (6 decimals at parity 1e30, 18 decimals at parity
    1e18, so one whole token of either is worth 1e6 value units);
  - ``deploy_substrate``: constructor (treasury, governor, anchorer,
    registry, serviceFeeBps, feeTreasuryBps); deploys an index when no
    registry is given (its tokens: ``index_tokens``);
  - ``anchor_direct``: chain onto the current head and anchor from the
    governor;
  - ``seed_pool``: pay real service fees in every index token and anchor a
    seed epoch closing that window, so the NEXT anchor's pool cap covers
    the test's claims (lag one);
  - ``reviewed_close``: a close whose review rounds conclude for every
    registered tool, so the test has something to claim.

Accounts 6..8 are reserved here; tests use 0..5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from web3 import Web3

ZERO = "0x0000000000000000000000000000000000000000"
SLOT_POOL_RECIPIENT = 6
SLOT_POOL_PAYER = 7
SLOT_INDEX = 8      # deploys the index, so the deployer's nonces stay predictable
DEFAULT_SEED_GROSS = 10 ** 12

_ART = Path(__file__).resolve().parent.parent / "artifacts" / "contracts"
# substrate address (lowercase) -> {"registry": contract, "tokens": [contracts]}
INDEX: Dict[str, Dict[str, object]] = {}


def artifact(rel: str) -> dict:
    """A compiled artifact under artifacts/contracts, e.g.
    ``test/IndexTokenMocks.sol/MockRegistry``."""
    return json.loads((_ART / f"{rel}.json").read_text(encoding="utf-8"))


def claim_takes_value(abi: list) -> bool:
    """True while the compiled Substrate's ``claim`` still takes a REP value
    (the leaf without it is the current spec)."""
    for e in abi:
        if e.get("type") == "function" and e.get("name") == "claim":
            return len(e.get("inputs") or []) == 6
    return False


def account_key(w3: Web3, address: str) -> str:
    """Private key (0x hex) of an eth_tester managed account."""
    backend = w3.provider.ethereum_tester.backend
    from eth_account import Account
    for k in backend.account_keys:
        kb = k.to_bytes() if hasattr(k, "to_bytes") else bytes(k)
        if Account.from_key(kb).address.lower() == address.lower():
            return "0x" + kb.hex()
    raise KeyError(address)


def _deploy(w3: Web3, art: dict, deployer: str, *args):
    tx = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"]).constructor(
        *args).transact({"from": deployer, "gas": 8_000_000})
    r = w3.eth.wait_for_transaction_receipt(tx)
    assert r.status == 1
    return w3.eth.contract(address=r.contractAddress, abi=art["abi"])


def deploy_index(w3: Web3, deployer: str, *, decimals=(6, 18)) -> Dict[str, object]:
    """A MockRegistry and one MockIndexToken per entry of ``decimals``, each
    in the index at parity 10**(36 - decimals) (one whole token = 1e6 value
    units). Returns {"registry", "tokens"}."""
    reg = _deploy(w3, artifact("test/IndexTokenMocks.sol/MockRegistry"), deployer)
    tok_art = artifact("test/IndexTokenMocks.sol/MockIndexToken")
    tokens = []
    for d in decimals:
        t = _deploy(w3, tok_art, deployer, f"T{d}", f"T{d}", int(d))
        tx = reg.functions.setParity(t.address, 10 ** (36 - int(d))).transact(
            {"from": deployer, "gas": 300_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        tokens.append(t)
    return {"registry": reg, "tokens": tokens}


def deploy_substrate(
    w3: Web3,
    abi: list,
    bytecode: str,
    deployer: str,
    *,
    treasury: Optional[str] = None,
    governor: Optional[str] = None,
    anchorer: str = ZERO,
    registry: Optional[str] = None,
    service_fee_bps: int = 250,
    fee_treasury_bps: int = 5000,
) -> str:
    """Deploy Substrate. ``governor`` defaults to ``deployer``: the account
    the tests anchor from. Without ``registry`` an index is deployed for it
    (``index_tokens``, ``index_registry``)."""
    treasury = deployer if treasury is None else treasury
    governor = deployer if governor is None else governor
    index = None
    if registry is None:
        index = deploy_index(w3, w3.eth.accounts[SLOT_INDEX])
        registry = index["registry"].address
    factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = factory.constructor(treasury, governor, anchorer, registry,
                             service_fee_bps, fee_treasury_bps).transact(
        {"from": deployer, "gas": 12_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    if index is not None:
        INDEX[receipt.contractAddress.lower()] = index
    # Substrate starts its cutoffs at its deployment time; a first anchor
    # needs a cutoff after it and before its own block, so the chain moves
    # on two blocks.
    mine = getattr(getattr(w3.provider, "ethereum_tester", None), "mine_blocks", None)
    if mine is not None:
        mine(2)
    return receipt.contractAddress


def index_tokens(contract) -> list:
    """The index tokens deployed with this Substrate (deploy_substrate)."""
    return list(INDEX[contract.address.lower()]["tokens"])


def index_registry(contract):
    """The MockRegistry deployed with this Substrate (deploy_substrate)."""
    return INDEX[contract.address.lower()]["registry"]


def anchor_direct(
    w3: Web3,
    contract,
    epoch_id: str,
    epoch_root: bytes,
    agent_mint_cid: str,
    payload_hash: bytes,
    agent_mint_root: bytes,
    *,
    sender: Optional[str] = None,
    gas: int = 3_000_000,
    cutoff: Optional[int] = None,
):
    """Chain onto the current head and anchor from ``sender`` (default:
    Substrate's governor). ``cutoff`` defaults to the head block's
    timestamp (the anchoring block is later). Returns the receipt."""
    sender = sender or contract.functions.governor().call()
    if cutoff is None:
        cutoff = max(int(w3.eth.get_block("latest")["timestamp"]),
                     int(contract.functions.latestCutoff().call()) + 1)
    tx = contract.functions.anchor(
        epoch_id,
        bytes(epoch_root),
        bytes(contract.functions.latestEpochRoot().call()),
        bytes(contract.functions.latestAnchorHash().call()),
        agent_mint_cid,
        bytes(payload_hash),
        bytes(agent_mint_root),
        int(cutoff),
    ).transact({"from": sender, "gas": gas})
    return w3.eth.wait_for_transaction_receipt(tx)


def pay_fees(w3: Web3, contract, gross: int = DEFAULT_SEED_GROSS,
             tokens=None) -> Dict[str, int]:
    """Pay ``gross`` of each index token for a service (payer and recipient
    are the reserved accounts). Returns {token lowercase: tool share}."""
    accts = w3.eth.accounts
    payer, recipient = accts[SLOT_POOL_PAYER], accts[SLOT_POOL_RECIPIENT]
    out: Dict[str, int] = {}
    for t in (tokens if tokens is not None else index_tokens(contract)):
        for fn in (t.functions.mint(payer, gross),
                   t.functions.approve(contract.address, gross)):
            assert w3.eth.wait_for_transaction_receipt(
                fn.transact({"from": payer, "gas": 300_000})).status == 1
        before = int(contract.functions.toolShareTotal(t.address).call())
        tx = contract.functions.payForService(
            t.address, recipient, gross, bytes(32)).transact(
            {"from": payer, "gas": 800_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        out[t.address.lower()] = (
            int(contract.functions.toolShareTotal(t.address).call()) - before)
    return out


_seed_counter = [0]


def seed_pool(w3: Web3, contract, gross: int = DEFAULT_SEED_GROSS,
              tokens=None) -> Dict[str, int]:
    """Pay real fees in each index token and anchor a seed epoch closing
    that window. Returns {token: tool share}: what the NEXT anchor's pool
    cap adds."""
    shares = pay_fees(w3, contract, gross, tokens)
    _seed_counter[0] += 1
    n = _seed_counter[0]
    r = anchor_direct(
        w3, contract, f"seed-{n}",
        Web3.keccak(text=f"seed-root-{n}"), f"seed-cid-{n}",
        Web3.keccak(text=f"seed-payload-{n}"), bytes(32))
    assert r.status == 1, r
    return shares


def sign_owner_binding(w3: Web3, substrate: str, signer_key: str, agent: str,
                       parent: str = ZERO, nonce: int = 0) -> bytes:
    """EIP-712 OwnerBinding{agent, parent, nonce} signature bytes."""
    from eth_account import Account
    from eth_account.messages import encode_typed_data
    msg = encode_typed_data(full_message={
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "OwnerBinding": [
                {"name": "agent", "type": "address"},
                {"name": "parent", "type": "address"},
                {"name": "nonce", "type": "uint256"},
            ],
        },
        "primaryType": "OwnerBinding",
        "domain": {"name": "AutonetSubstrate", "version": "1",
                   "chainId": int(w3.eth.chain_id),
                   "verifyingContract": Web3.to_checksum_address(substrate)},
        "message": {"agent": Web3.to_checksum_address(agent),
                    "parent": Web3.to_checksum_address(parent),
                    "nonce": int(nonce)},
    })
    return bytes(Account.from_key(signer_key).sign_message(msg).signature)


def register_bound(w3: Web3, contract, agent: str, peer_id: bytes, *,
                   owner: Optional[str] = None, parent: str = ZERO,
                   gas: int = 500_000):
    """registerAgentBound for an eth_tester account. ``owner`` defaults to the
    agent itself (self-owned). Returns the receipt."""
    owner = agent if owner is None else owner
    sig = sign_owner_binding(w3, contract.address, account_key(w3, owner),
                             agent, parent)
    tx = contract.functions.registerAgentBound(
        peer_id, owner, parent, sig).transact({"from": agent, "gas": gas})
    return w3.eth.wait_for_transaction_receipt(tx)


# --- review pay (Decision 2026-11-07) --------------------------------------

REVIEWER = "0x" + "9e" * 20


def tool_registration_batch(rpb: str, author: str, digest: str, *,
                            sender: Optional[bytes] = None):
    """One gossiped batch carrying a pinned tool's ``tool_registered``
    event, authored by ``author``."""
    from nodes.common.event_gossip import EventBatch, Keypair
    ev = {
        "kind": "tool_registered",
        "seq": 1,
        "author_agent": author,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author},
    }
    return EventBatch(
        rpb_address=rpb,
        sender_pubkey=sender or Keypair.generate().public_key,
        batch_seq=1, events=[ev], prev_batch_hash=b"",
        timestamp=1_700_000_000.0,
    )


def review_snapshot(digests, reviewer: str = REVIEWER, *,
                    supply: int = 10 ** 21) -> dict:
    """A ToolBond review snapshot with one concluding round per tool: one
    reviewer rates every field and charter axis at the top, no malware,
    accuracy 1 (Decision 2026-11-09 shape)."""
    from nodes.common.library.charter import active_charter
    e18 = 10 ** 18

    def cell(v: int) -> dict:
        return {"sum": v * e18, "mass": e18}
    entry = {"fields": {"correctness": cell(100), "simplicity": cell(100),
                        "novelty": cell(100), "malware": cell(0)},
             "axes": {a: cell(100) for a in active_charter().ids}}
    return {
        "supply": supply, "reviewer_share_bps": 4000,
        "tools": {d: entry for d in digests},
        "rounds": {d: [{"epoch": 1, "supply": supply,
                        "reviews": [{"reviewer": reviewer, "accuracy": e18}]}]
                   for d in digests},
    }


def reviewed_close(canonical, *, reviewer: str = REVIEWER,
                   rep_share: float = 0.5, **kw) -> dict:
    """``federated_epoch_close`` in which every registered tool's review
    round concludes, so its author (and the reviewer) are paid. The only
    earning rail since Decision 2026-11-07."""
    from nodes.common.federated_reconcile import federated_epoch_close
    regs = federated_epoch_close(canonical)["tool_registrations"]
    return federated_epoch_close(
        canonical, rep_shares={reviewer: rep_share},
        review_snapshot=review_snapshot(regs, reviewer), **kw)


def deploy_homebase(w3: Web3, hb_art: dict, clone_maker_art: dict, deployer: str,
                    token: str, timelock: str, *, voting_delay_min: int = 1,
                    voting_period_min: int = 3, threshold: int = 1,
                    quorum_pct: int = 4, supermajority_bps: int = 6667):
    """A HomebaseDAO as a jurisdiction gets it (trustless-contracts): an
    EIP-1167 clone of the implementation, set up by initialize, with
    ``deployer`` as the launcher that approves option sources."""
    def _deploy_impl(art):
        tx = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"]).constructor(
            ).transact({"from": deployer, "gas": 15_000_000})
        return w3.eth.wait_for_transaction_receipt(tx).contractAddress

    impl = _deploy_impl(hb_art)
    maker = w3.eth.contract(address=_deploy_impl(clone_maker_art),
                            abi=clone_maker_art["abi"])
    clone = maker.functions.clone(impl).call({"from": deployer})
    r = w3.eth.wait_for_transaction_receipt(
        maker.functions.clone(impl).transact({"from": deployer, "gas": 1_000_000}))
    assert r.status == 1
    gov = w3.eth.contract(address=clone, abi=hb_art["abi"])
    r = w3.eth.wait_for_transaction_receipt(gov.functions.initialize(
        token, timelock, "Test DAO", voting_delay_min, voting_period_min,
        threshold, quorum_pct, deployer, supermajority_bps,
    ).transact({"from": deployer, "gas": 3_000_000}))
    assert r.status == 1
    return gov
