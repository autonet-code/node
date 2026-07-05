#!/usr/bin/env python3
"""Charter version hash + anchor verification.

Covers:
  - charter_hash determinism (same constant -> same hash; key order irrelevant)
  - charter_payload includes all 6 roots
  - the federated close authoritative payload carries charter_hash
  - verify_charter_against_anchor against a fake on-chain reader (match / drift)
"""

from __future__ import annotations

import hashlib
import json

from nodes.common.world_model_substrate.adapter import CHARTER
from nodes.common.world_model_substrate.charter_version import (
    charter_bytes,
    charter_hash,
    charter_payload,
)


# ---------------------------------------------------------------------------
# charter_payload / charter_hash
# ---------------------------------------------------------------------------


def test_payload_has_all_six_roots():
    payload = charter_payload()
    assert payload["n_dims"] == 6
    assert len(payload["roots"]) == 6
    ids = {r["id"] for r in payload["roots"]}
    assert ids == {
        "life_precious",
        "self_preservation",
        "promotion_of_intelligence",
        "evolution",
        "correctness",
        "simplicity",
    }
    # roots are ordered by axis_index
    axes = [r["axis_index"] for r in payload["roots"]]
    assert axes == sorted(axes) == [0, 1, 2, 3, 4, 5]


def test_hash_matches_sha256_of_bytes():
    assert charter_hash() == hashlib.sha256(charter_bytes()).hexdigest()


def test_hash_is_deterministic():
    assert charter_hash() == charter_hash()


def test_hash_is_key_order_independent():
    """Reshuffling dict key order in the source constant must not change the
    hash — the serialization sorts keys, so the hash is a pure function of the
    values, not declaration order."""
    payload = charter_payload()

    # Rebuild the same payload with dict keys inserted in a scrambled order
    # (list order preserved — the payload deliberately fixes root order by
    # axis_index). sort_keys must collapse the scrambled keys to identical bytes.
    scrambled_roots = [
        {
            "veto_floor": r["veto_floor"],
            "thesis": r["thesis"],
            "id": r["id"],
            "axis_index": r["axis_index"],
        }
        for r in payload["roots"]
    ]
    scrambled = {
        "roots": scrambled_roots,
        "n_dims": payload["n_dims"],
        "schema": payload["schema"],
    }
    scrambled_bytes = json.dumps(
        scrambled, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(scrambled_bytes).hexdigest() == charter_hash()


def test_hash_derived_from_same_constant():
    """The payload roots are drawn from adapter.CHARTER, not a duplicate."""
    payload_ids = [r["id"] for r in charter_payload()["roots"]]
    charter_ids = sorted((e["id"] for e in CHARTER))
    assert sorted(payload_ids) == charter_ids


# ---------------------------------------------------------------------------
# close payload carries the hash
# ---------------------------------------------------------------------------


def test_close_payload_carries_charter_hash():
    from nodes.common.canonical_ordering import CanonicalOrder
    from nodes.common.federated_reconcile import federated_epoch_close

    result = federated_epoch_close(CanonicalOrder())
    payload = result["authoritative_payload"]
    assert "charter_hash" in payload
    assert payload["charter_hash"] == charter_hash()


# ---------------------------------------------------------------------------
# verify_charter_against_anchor against a fake reader
# ---------------------------------------------------------------------------


class _FakeFunction:
    def __init__(self, value):
        self._value = value

    def call(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class _FakeFunctions:
    def __init__(self, current):
        self._current = current

    def currentCharter(self):
        return _FakeFunction(self._current)


class _FakeContract:
    def __init__(self, current):
        self.functions = _FakeFunctions(current)


class _FakeEth:
    def __init__(self, current):
        self._current = current

    def contract(self, address=None, abi=None):
        return _FakeContract(self._current)


class _FakeW3:
    """Minimal stand-in for a connected Web3 instance."""

    def __init__(self, current):
        self.eth = _FakeEth(current)


def _bytes32_from_hex(hexstr: str) -> bytes:
    return bytes.fromhex(hexstr)


# A valid EIP-55 checksummed address (Web3.to_checksum_address accepts it).
_ANCHOR_ADDR = "0x000000000000000000000000000000000000dEaD"


def test_verify_match():
    from atn.on_chain import verify_charter_against_anchor

    local = charter_hash()
    # currentCharter returns (version, hash, uri, prevHash, timestamp)
    current = (3, _bytes32_from_hex(local), "ipfs://v3", b"\x00" * 32, 123)
    w3 = _FakeW3(current)
    out = verify_charter_against_anchor(w3, _ANCHOR_ADDR)
    assert out["match"] is True
    assert out["chain_version"] == 3
    assert out["local_hash"] == local
    assert out["chain_hash"] == local


def test_verify_mismatch(caplog):
    import logging

    from atn.on_chain import verify_charter_against_anchor

    wrong = hashlib.sha256(b"a different charter").hexdigest()
    current = (5, _bytes32_from_hex(wrong), "ipfs://v5", b"\x00" * 32, 999)
    w3 = _FakeW3(current)
    with caplog.at_level(logging.WARNING):
        out = verify_charter_against_anchor(w3, _ANCHOR_ADDR)
    assert out["match"] is False
    assert out["chain_hash"] == wrong
    assert out["local_hash"] == charter_hash()
    assert any("CHARTER DIVERGENCE" in r.message for r in caplog.records)


def test_verify_no_anchor():
    from atn.on_chain import verify_charter_against_anchor

    # A revert (NoCharter) surfaces as an exception from .call()
    w3 = _FakeW3(RuntimeError("execution reverted: NoCharter"))
    out = verify_charter_against_anchor(w3, _ANCHOR_ADDR)
    assert out["match"] is None
    assert out["chain_hash"] is None
    assert out["local_hash"] == charter_hash()
