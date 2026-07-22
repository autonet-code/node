"""Pure-part unit tests for the daemon-side ServiceMarket chain client.

Stage A of the services wiring (atn/on_chain.py ServiceMarketClient +
atn/config.py plumbing). NO live-chain tests: these exercise only the
deterministic, offline surface —

  1. EIP-712 voucher digest construction (build_voucher_digest) matches an
     independently derived fixture (OZ _hashTypedDataV4 recomputed with
     eth_utils/eth_abi), and sign→recover round-trips.
  2. ServicePayment event decoding from a CRAFTED receipt (topics/data built
     from the ABI with eth_utils/eth_abi), via OnChainService.verify_service_payment
     with a monkeypatched web3 — no network.
  3. Config plumbing: payment_channel_address + service_registry_address resolve
     from a fake registry dict, and registry_address keeps its back-compat value.

Run: pytest tests/test_service_market_client.py
"""
from __future__ import annotations

import asyncio

import pytest

from eth_abi import encode as abi_encode
from eth_account import Account
from eth_utils import keccak, to_checksum_address
from web3 import Web3

from atn import config as cfg
from atn import on_chain as oc


_CHANNEL_ADDR = "0x97956A322dC5585157c91840d39292c396032508"
_SUBSTRATE_ADDR = "0x4C4dAEd19B98ddaEc6cC421b6a781Bd3fBB7af25"
_CHAIN_ID = 127823


# ---------------------------------------------------------------------------
# 1. EIP-712 voucher digest
# ---------------------------------------------------------------------------

def _manual_voucher_digest(channel_id, cumulative, channel_addr, chain_id):
    """Independently recompute PaymentChannel.voucherHash via OZ's
    EIP-712 _hashTypedDataV4 formula, from the typehash strings directly."""
    domain_type_hash = keccak(
        text="EIP712Domain(string name,string version,"
             "uint256 chainId,address verifyingContract)")
    domain_sep = keccak(abi_encode(
        ["bytes32", "bytes32", "bytes32", "uint256", "address"],
        [domain_type_hash,
         keccak(text="AutonetPaymentChannel"),
         keccak(text="1"),
         chain_id,
         to_checksum_address(channel_addr)],
    ))
    voucher_type_hash = keccak(
        text="Voucher(uint256 channelId,uint256 cumulativeAmount)")
    struct_hash = keccak(abi_encode(
        ["bytes32", "uint256", "uint256"],
        [voucher_type_hash, channel_id, cumulative]))
    return keccak(b"\x19\x01" + domain_sep + struct_hash)


def test_voucher_digest_matches_fixture():
    """build_voucher_digest reproduces the contract's typed-data digest."""
    for channel_id, cumulative in [(1, 100), (7, 123456), (42, 10**18)]:
        built = oc.build_voucher_digest(
            channel_id, cumulative, _CHANNEL_ADDR, _CHAIN_ID)
        expected = _manual_voucher_digest(
            channel_id, cumulative, _CHANNEL_ADDR, _CHAIN_ID)
        assert built == expected, (channel_id, cumulative)


def test_voucher_digest_is_domain_bound():
    """Digest changes with chainId and verifyingContract (replay safety)."""
    base = oc.build_voucher_digest(1, 100, _CHANNEL_ADDR, _CHAIN_ID)
    other_chain = oc.build_voucher_digest(1, 100, _CHANNEL_ADDR, _CHAIN_ID + 1)
    other_contract = oc.build_voucher_digest(
        1, 100, "0x0000000000000000000000000000000000001234", _CHAIN_ID)
    assert base != other_chain
    assert base != other_contract


def test_sign_voucher_recovers_to_client():
    """sign_voucher signs the digest the contract recovers against."""
    client = Account.create()
    conf = cfg.RPBConfig(
        payment_channel_address=_CHANNEL_ADDR, chain_id=_CHAIN_ID)
    smc = oc.ServiceMarketClient(conf)

    channel_id, cumulative = 5, 777
    sig_hex = smc.sign_voucher(client.key.hex(), channel_id, cumulative)
    sig = bytes.fromhex(sig_hex[2:] if sig_hex.startswith("0x") else sig_hex)

    digest = oc.build_voucher_digest(
        channel_id, cumulative, _CHANNEL_ADDR, _CHAIN_ID)
    recovered = (Account._recover_hash(digest, signature=sig)
                 if hasattr(Account, "_recover_hash")
                 else Account.recoverHash(digest, signature=sig))
    assert to_checksum_address(recovered) == client.address


# ---------------------------------------------------------------------------
# 2. ServicePayment event decode from a crafted receipt
# ---------------------------------------------------------------------------

class _FakeContractFns:
    """Minimal stand-in exposing only .functions/.events used by
    verify_service_payment (none of the contract functions are called in the
    happy path — decoding uses .events)."""


class _FakeEth:
    def __init__(self, receipt):
        self._receipt = receipt

    def get_transaction_receipt(self, tx_hash):
        return self._receipt

    def contract(self, address=None, abi=None):
        # Return a REAL web3 contract bound to the SUBSTRATE ABI so the
        # genuine event-decoder runs against our crafted logs.
        return Web3().eth.contract(address=address, abi=abi)


class _FakeWeb3:
    def __init__(self, receipt):
        self.eth = _FakeEth(receipt)

    def is_connected(self):
        return True

    @staticmethod
    def to_checksum_address(a):
        return to_checksum_address(a)


def _craft_service_payment_receipt(payer, recipient, amount, request_id_b):
    """Build a receipt dict with ONE ServicePayment log, ABI-encoded exactly
    as Substrate.sol would emit it (3 indexed topics + non-indexed amount)."""
    # event ServicePayment(address indexed payer, address indexed recipient,
    #                       uint256 amount, bytes32 indexed requestId)
    sig_topic = keccak(text="ServicePayment(address,address,uint256,bytes32)")
    topic_payer = bytes.fromhex("%064x" % int(payer, 16))
    topic_recipient = bytes.fromhex("%064x" % int(recipient, 16))
    topic_request = request_id_b
    data = abi_encode(["uint256"], [amount])
    log = {
        "address": to_checksum_address(_SUBSTRATE_ADDR),
        "topics": [sig_topic, topic_payer, topic_recipient, topic_request],
        "data": data,
        "logIndex": 0,
        "transactionIndex": 0,
        "transactionHash": b"\x00" * 32,
        "blockHash": b"\x00" * 32,
        "blockNumber": 1,
        "removed": False,
    }
    return {"status": 1, "logs": [log]}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _client_with_receipt(receipt):
    conf = cfg.RPBConfig(substrate_address=_SUBSTRATE_ADDR, chain_id=_CHAIN_ID)
    svc = oc.OnChainService(conf)
    fake = _FakeWeb3(receipt)
    svc._get_web3 = lambda: fake  # type: ignore[assignment]
    return svc


def test_verify_service_payment_decodes_and_matches():
    payer = Account.create().address
    recipient = Account.create().address
    amount = 5000
    req = keccak(text="request-body-abc")
    receipt = _craft_service_payment_receipt(payer, recipient, amount, req)
    svc = _client_with_receipt(receipt)

    out = _run(svc.verify_service_payment(
        request_id=req, recipient=recipient, min_amount=1000,
        payer=payer, tx_hash="0x" + "11" * 32))

    assert out["verified"] is True, out
    assert out["amount"] == amount
    assert to_checksum_address(out["payer"]) == to_checksum_address(payer)
    assert to_checksum_address(out["recipient"]) == to_checksum_address(recipient)
    assert out["request_id"] == req.hex()


def test_verify_service_payment_rejects_low_amount():
    payer = Account.create().address
    recipient = Account.create().address
    req = keccak(text="req")
    receipt = _craft_service_payment_receipt(payer, recipient, 100, req)
    svc = _client_with_receipt(receipt)

    out = _run(svc.verify_service_payment(
        request_id=req, recipient=recipient, min_amount=1000, tx_hash="0xabc"))
    assert out["verified"] is False
    assert "min_amount" in out["reason"]
    assert out["amount"] == 100  # decoded even though it failed the check


def test_verify_service_payment_rejects_recipient_mismatch():
    payer = Account.create().address
    recipient = Account.create().address
    other = Account.create().address
    req = keccak(text="req")
    receipt = _craft_service_payment_receipt(payer, recipient, 5000, req)
    svc = _client_with_receipt(receipt)

    out = _run(svc.verify_service_payment(
        request_id=req, recipient=other, min_amount=1, tx_hash="0xabc"))
    assert out["verified"] is False
    assert "matched" in out["reason"] or "recipient" in out["reason"]


def test_verify_service_payment_rejects_payer_mismatch():
    payer = Account.create().address
    recipient = Account.create().address
    wrong_payer = Account.create().address
    req = keccak(text="req")
    receipt = _craft_service_payment_receipt(payer, recipient, 5000, req)
    svc = _client_with_receipt(receipt)

    out = _run(svc.verify_service_payment(
        request_id=req, recipient=recipient, min_amount=1,
        payer=wrong_payer, tx_hash="0xabc"))
    assert out["verified"] is False
    assert out["reason"] == "payer mismatch"


def test_verify_service_payment_requires_tx_hash():
    svc = _client_with_receipt({"status": 1, "logs": []})
    out = _run(svc.verify_service_payment(
        request_id=keccak(text="x"), recipient=Account.create().address,
        min_amount=1, tx_hash=None))
    assert out["verified"] is False
    assert "tx_hash required" in out["reason"]


def test_verify_service_payment_reverted_tx():
    payer = Account.create().address
    recipient = Account.create().address
    req = keccak(text="req")
    receipt = _craft_service_payment_receipt(payer, recipient, 5000, req)
    receipt["status"] = 0
    svc = _client_with_receipt(receipt)
    out = _run(svc.verify_service_payment(
        request_id=req, recipient=recipient, min_amount=1, tx_hash="0xabc"))
    assert out["verified"] is False
    assert out["reason"] == "transaction reverted"


def test_verify_service_payment_no_matching_event():
    """A receipt with no ServicePayment log (unrelated tx) is not verified."""
    receipt = {"status": 1, "logs": []}
    svc = _client_with_receipt(receipt)
    out = _run(svc.verify_service_payment(
        request_id=keccak(text="x"), recipient=Account.create().address,
        min_amount=1, tx_hash="0xabc"))
    assert out["verified"] is False
    assert "no ServicePayment event" in out["reason"]


# ---------------------------------------------------------------------------
# 3. Config plumbing
# ---------------------------------------------------------------------------

_FAKE_REGISTRY = {
    "version": 1,
    "jurisdictions": {
        "autonet": {
            "name": "Autonet",
            "network": {"rpc_url": "https://rpc.test", "chain_id": 1},
            "contracts": {
                "dao": "0xDA0000000000000000000000000000000000dead",
                "substrate": "0x5UB000000000000000000000000000000000beef",
                "service_registry": "0x5E4000000000000000000000000000000000cafe",
                "payment_channel": "0xC4A44E10000000000000000000000000000face0",
            },
        }
    },
}


def test_registry_seed_resolves_service_market_addresses():
    """service_registry_address + payment_channel_address land from the
    registry seed, and registry_address keeps the back-compat value."""
    seed = cfg._parse_registry_data(_FAKE_REGISTRY, "autonet")
    assert seed["service_registry_address"] == \
        "0x5E4000000000000000000000000000000000cafe"
    assert seed["payment_channel_address"] == \
        "0xC4A44E10000000000000000000000000000face0"
    # Back-compat: the legacy shared field still carries the ServiceRegistry.
    assert seed["registry_address"] == \
        "0x5E4000000000000000000000000000000000cafe"


def test_registry_seed_no_payment_channel_absent():
    """A registry without payment_channel simply omits the field."""
    data = {
        "version": 1,
        "jurisdictions": {"autonet": {"contracts": {
            "substrate": "0xbeef", "service_registry": "0xcafe"}}},
    }
    seed = cfg._parse_registry_data(data, "autonet")
    assert "payment_channel_address" not in seed
    assert seed["service_registry_address"] == "0xcafe"


def test_rpbconfig_carries_new_fields():
    """RPBConfig exposes the new fields and they default empty."""
    conf = cfg.RPBConfig()
    assert conf.service_registry_address == ""
    assert conf.payment_channel_address == ""
    conf2 = cfg.RPBConfig(
        service_registry_address="0xaaa", payment_channel_address="0xbbb")
    assert conf2.service_registry_address == "0xaaa"
    assert conf2.payment_channel_address == "0xbbb"


def test_service_market_client_resolves_addresses():
    """ServiceMarketClient resolves configured addresses, with the legacy
    registry_address fallback for the ServiceRegistry."""
    conf = cfg.RPBConfig(
        service_registry_address="0xREG",
        payment_channel_address="0xCHAN",
        substrate_address="0xSUB",
        rpc_url="https://rpc.test",
    )
    smc = oc.ServiceMarketClient(conf)
    assert smc._service_registry_address() == "0xREG"
    assert smc._payment_channel_address() == "0xCHAN"
    assert smc._substrate_address() == "0xSUB"
    assert smc.registry_available is True
    assert smc.channel_available is True

    # Legacy fallback: only registry_address set (old seed shape).
    conf_legacy = cfg.RPBConfig(
        registry_address="0xLEGACY", rpc_url="https://rpc.test")
    smc_legacy = oc.ServiceMarketClient(conf_legacy)
    assert smc_legacy._service_registry_address() == "0xLEGACY"
