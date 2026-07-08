"""Owner-binding daemon-side round-trip against eth_tester + Substrate.

Verifies the DAEMON's EIP-712 owner-binding path end-to-end against the
compiled contract:

  1. ``OnChainService.register_agent_bound`` binds an agent to an owner
     wallet with a signature produced by ``eth_account.sign_typed_data``
     over the frontend-dictated domain (name "AutonetSubstrate", version
     "1", chainId, verifyingContract) and primary type ``OwnerBinding``
     {agent, parent, nonce} — asserts ``agentOwner`` on-chain and that
     the binding nonce advanced.
  2. ``rotate_owner`` rotates the agent to a SECOND wallet WITHOUT the old
     owner's participation (key-loss recovery) — asserts the new owner +
     the nonce increment.
  3. ``read_voice_state`` (the close-side owner_map derivation) picks up
     the binding from the OwnerBound event stream after an anchor.

The digest the daemon signs must be byte-identical to what the contract
recovers against — if it weren't, registerAgentBound would revert with
BadOwnerBindingSignature and step 1 would fail loudly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest

eth_tester = pytest.importorskip("eth_tester")

from eth_account import Account
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from atn.on_chain import OnChainService
from nodes.common.canonical_ordering import canonical_order
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from nodes.common.voice_state import read_voice_state

SUBSTRATE_JSON = Path(
    "C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str) -> str:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor(deployer, "0x0000000000000000000000000000000000000000").transact(
        {"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return receipt.contractAddress


_ZERO = "0x0000000000000000000000000000000000000000"

# EIP-712 types for the OwnerBinding struct (frontend contract).
_TYPES = {
    "OwnerBinding": [
        {"name": "agent", "type": "address"},
        {"name": "parent", "type": "address"},
        {"name": "nonce", "type": "uint256"},
    ],
}


def _sign_binding(w3, substrate_addr, chain_id, owner_acct, agent_addr,
                  parent_addr, nonce) -> str:
    """Sign OwnerBinding via eth_account.sign_typed_data; return 0x-hex sig."""
    full = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            **_TYPES,
        },
        "domain": {
            "name": "AutonetSubstrate",
            "version": "1",
            "chainId": chain_id,
            "verifyingContract": Web3.to_checksum_address(substrate_addr),
        },
        "primaryType": "OwnerBinding",
        "message": {
            "agent": Web3.to_checksum_address(agent_addr),
            "parent": Web3.to_checksum_address(parent_addr),
            "nonce": int(nonce),
        },
    }
    signed = Account.sign_typed_data(owner_acct.key, full_message=full)
    return signed.signature.hex()


def _svc_for(w3, abi, substrate_addr, chain_id) -> OnChainService:
    """Build an OnChainService bound to the eth_tester web3/contract."""
    config = SimpleNamespace(
        substrate_address=substrate_addr,
        rpc_url="http://eth-tester.invalid",
        chain_id=chain_id,
    )
    svc = OnChainService(config)  # type: ignore[arg-type]
    # Inject the in-memory web3/contract instead of an HTTP provider.
    svc._get_web3 = lambda: w3  # type: ignore[method-assign]
    svc._get_contract = lambda w=None: w3.eth.contract(  # type: ignore[method-assign]
        address=Web3.to_checksum_address(substrate_addr), abi=abi)
    return svc


def _identity(agent_addr: str) -> SimpleNamespace:
    # lineage_hash: any nonzero 32-byte hex; must be unique per agent.
    lineage = Web3.keccak(text="lineage-" + agent_addr).hex()
    return SimpleNamespace(address=agent_addr, lineage_hash=lineage)


@pytest.fixture
def fx():
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    accounts = w3.eth.accounts
    deployer = accounts[0]
    addr = _deploy(w3, deployer, abi, bytecode)
    chain_id = w3.eth.chain_id
    return {
        "w3": w3, "abi": abi, "addr": addr, "deployer": deployer,
        "chain_id": chain_id, "accounts": accounts,
        "contract": w3.eth.contract(address=addr, abi=abi),
    }


def _fund(w3, frm, to, wei=10**18):
    w3.eth.send_transaction({"from": frm, "to": to, "value": wei})


# --- eth_tester funded keys ------------------------------------------------
# eth_tester's default accounts have known private keys; we need the agent's
# key to sign the registration tx and the owner's key to sign the binding.
def _tester_keys(w3):
    # web3's EthereumTesterProvider exposes the backend account keys.
    backend = w3.provider.ethereum_tester.backend
    keys = backend.account_keys
    # map checksum address -> hex private key
    out = {}
    for k in keys:
        acct = Account.from_key(k.to_bytes())
        out[acct.address.lower()] = acct.key.hex()
    return out


class TestOwnerBindingRoundTrip:
    def test_register_bound_then_rotate(self, fx):
        w3, abi, addr, chain_id = fx["w3"], fx["abi"], fx["addr"], fx["chain_id"]
        accounts = fx["accounts"]
        keys = _tester_keys(w3)

        agent_addr = accounts[1]
        owner_addr = accounts[2]
        new_owner_addr = accounts[3]

        agent_key = keys[agent_addr.lower()]
        owner_acct = Account.from_key(keys[owner_addr.lower()])
        new_owner_acct = Account.from_key(keys[new_owner_addr.lower()])

        svc = _svc_for(w3, abi, addr, chain_id)
        contract = fx["contract"]

        # --- 1. bound registration --------------------------------------
        nonce0 = asyncio.run(svc.binding_nonce(agent_addr))
        assert nonce0 == 0
        sig = _sign_binding(w3, addr, chain_id, owner_acct, agent_addr,
                            _ZERO, nonce0)
        res = asyncio.run(svc.register_agent_bound(
            identity=_identity(agent_addr),
            private_key=agent_key,
            peer_id=b"peer-A",
            owner=owner_addr,
            owner_sig=sig,
            parent="",
        ))
        assert res["success"], res
        assert (contract.functions.agentOwner(agent_addr).call().lower()
                == owner_addr.lower())
        assert contract.functions.bindingNonce(agent_addr).call() == 1
        # read helpers agree
        assert (asyncio.run(svc.get_agent_owner(agent_addr)).lower()
                == owner_addr.lower())
        assert asyncio.run(svc.binding_nonce(agent_addr)) == 1

        # --- 2. rotate to a second wallet (no old-owner sig) ------------
        nonce1 = asyncio.run(svc.binding_nonce(agent_addr))
        assert nonce1 == 1
        sig2 = _sign_binding(w3, addr, chain_id, new_owner_acct, agent_addr,
                             _ZERO, nonce1)
        res2 = asyncio.run(svc.rotate_owner(
            private_key=agent_key,
            new_owner=new_owner_addr,
            new_owner_sig=sig2,
            new_parent="",
        ))
        assert res2["success"], res2
        assert (contract.functions.agentOwner(agent_addr).call().lower()
                == new_owner_addr.lower())
        assert contract.functions.bindingNonce(agent_addr).call() == 2

    def test_bad_signature_reverts(self, fx):
        """A signature from the WRONG wallet must not bind (contract-verified)."""
        w3, abi, addr, chain_id = fx["w3"], fx["abi"], fx["addr"], fx["chain_id"]
        accounts = fx["accounts"]
        keys = _tester_keys(w3)
        agent_addr = accounts[4]
        owner_addr = accounts[5]
        wrong_acct = Account.from_key(keys[accounts[6].lower()])
        svc = _svc_for(w3, abi, addr, chain_id)
        # wrong_acct signs, but we claim owner_addr as owner → BadSig revert.
        sig = _sign_binding(w3, addr, chain_id, wrong_acct, agent_addr,
                            _ZERO, 0)
        res = asyncio.run(svc.register_agent_bound(
            identity=_identity(agent_addr),
            private_key=keys[agent_addr.lower()],
            peer_id=b"peer-bad",
            owner=owner_addr,
            owner_sig=sig,
        ))
        assert not res["success"]


# --- voice_state owner_map derivation --------------------------------------

_DIGEST = "ef" * 32


def _coords6() -> List[float]:
    out = [0.0] * (6 + 1024)
    out[4] = 0.8
    out[6 + 3] = 0.5
    return out


def _one_batch(rpb: str, ev: dict, kp: Keypair, seq: int) -> EventBatch:
    return EventBatch(
        rpb_address=rpb, sender_pubkey=kp.public_key, batch_seq=seq,
        events=[ev], prev_batch_hash=b"",
        timestamp=1_700_000_000.0 + seq,
    )


def _tool_batches(rpb: str, author_addr: str) -> List[EventBatch]:
    reg = {
        "kind": "sub_claim_sprouted", "seq": 1,
        "author_agent": author_addr, "tendency_id": "correctness",
        "parent_id": "solver_root", "node_id": f"tm_{_DIGEST[:12]}",
        "position": "pro", "coords": _coords6(),
        "polarity_axis": _coords6(),
        "content": f"tool {_DIGEST[:8]}", "author_post": True,
        "artifact_digest": _DIGEST,
        "manifest_meta": {"trust_class": "pinned", "author": author_addr},
    }

    def _vet(vetter, seq):
        return {"kind": "tool_used", "seq": seq, "author_agent": vetter,
                "manifest_digest": _DIGEST, "tool_author": author_addr,
                "receipt_digest": f"v{seq:02d}" * 8, "ok": True,
                "fee_atn": 0.0, "vet": True}

    def _receipt(caller, seq):
        return {"kind": "tool_used", "seq": seq, "author_agent": caller,
                "manifest_digest": _DIGEST, "tool_author": author_addr,
                "receipt_digest": f"r{seq:02d}" * 8, "ok": True,
                "fee_atn": 0.0, "attested": True, "score": 0.8}

    return [
        _one_batch(rpb, reg, Keypair.generate(), 1),
        _one_batch(rpb, _vet("vetter-1", 1), Keypair.generate(), 1),
        _one_batch(rpb, _vet("vetter-2", 1), Keypair.generate(), 1),
        _one_batch(rpb, _receipt("caller-1", 1), Keypair.generate(), 1),
        _one_batch(rpb, _receipt("caller-2", 2), Keypair.generate(), 1),
    ]


class TestVoiceStateOwnerMap:
    def test_owner_map_picks_up_binding(self, fx):
        """After a bound registration + an anchor, read_voice_state's
        owner_map maps the agent -> its owner wallet (OwnerBound event)."""
        w3, abi, addr, chain_id = fx["w3"], fx["abi"], fx["addr"], fx["chain_id"]
        accounts = fx["accounts"]
        keys = _tester_keys(w3)

        agent_addr = accounts[1]
        owner_addr = accounts[2]
        owner_acct = Account.from_key(keys[owner_addr.lower()])
        svc = _svc_for(w3, abi, addr, chain_id)

        sig = _sign_binding(w3, addr, chain_id, owner_acct, agent_addr,
                            _ZERO, 0)
        res = asyncio.run(svc.register_agent_bound(
            identity=_identity(agent_addr),
            private_key=keys[agent_addr.lower()],
            peer_id=b"peer-A",
            owner=owner_addr,
            owner_sig=sig,
        ))
        assert res["success"], res

        # Close + anchor an epoch so read_voice_state has a snapshot block.
        batches = _tool_batches("rpb_ob", agent_addr)
        result = federated_epoch_close(canonical_order(batches))
        result["epoch_id"] = "e_ob"
        anchorer = EpochAnchorer(
            config=EpochAnchorerConfig(epoch_anchor_address=addr),
            web3=w3, contract_abi=abi)
        anchor_result = anchorer.anchor_close_result(
            result, from_address=fx["deployer"])
        assert anchor_result.success

        state = read_voice_state(addr, web3=w3)
        assert state["owner_map"].get(agent_addr.lower()) == owner_addr.lower()
