#!/usr/bin/env python3
"""Charter version hash + anchor verification.

Covers:
  - charter_hash determinism (sha256 of the active charter's registry text)
  - charter_payload includes all 6 roots
  - the federated close authoritative payload carries charter_hash
  - verify_charter against the registry value (match / drift / none)
"""

from __future__ import annotations

import hashlib
import json

from nodes.common.library.charter import DEFAULT_CHARTER
from nodes.common.library.charter_version import (
    charter_bytes,
    charter_hash,
    charter_payload,
)


# ---------------------------------------------------------------------------
# charter_payload / charter_hash
# ---------------------------------------------------------------------------


def test_payload_has_the_four_value_roots():
    payload = charter_payload()
    assert payload["n_dims"] == 4
    assert len(payload["roots"]) == 4
    ids = {r["id"] for r in payload["roots"]}
    assert ids == {
        "life_precious",
        "self_preservation",
        "promotion_of_intelligence",
        "evolution",
    }
    # roots are ordered by axis_index
    axes = [r["axis_index"] for r in payload["roots"]]
    assert axes == sorted(axes) == [0, 1, 2, 3]


def test_hash_matches_sha256_of_bytes():
    assert charter_hash() == hashlib.sha256(charter_bytes()).hexdigest()


def test_hash_is_deterministic():
    assert charter_hash() == charter_hash()


def test_hash_is_sha256_of_registry_text():
    """The hash is sha256 of the active charter's exact registry bytes
    (compact JSON), so it equals the Registry's charter hash."""
    from nodes.common.library.charter import active_charter

    text = active_charter().text
    assert charter_bytes() == text.encode("utf-8")
    assert charter_hash() == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert "0x" + charter_hash() == active_charter().hash
    assert json.loads(text)["axes"][0]["id"] == "life_precious"


def test_hash_derived_from_same_constant():
    """The payload roots are drawn from the active charter, not a duplicate."""
    payload_ids = [r["id"] for r in charter_payload()["roots"]]
    charter_ids = sorted((e["id"] for e in DEFAULT_CHARTER.entries()))
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
# verify_charter against the jurisdiction's registry value
# ---------------------------------------------------------------------------


def test_verify_match():
    from atn.on_chain import verify_charter

    local = charter_hash()
    # Registry values are 0x-prefixed; case must not matter.
    out = verify_charter("0x" + local.upper(), "sha256://v3")
    assert out["match"] is True
    assert out["local_hash"] == local
    assert out["chain_hash"] == local
    assert out["uri"] == "sha256://v3"


def test_verify_mismatch(caplog):
    import logging

    from atn.on_chain import verify_charter

    wrong = hashlib.sha256(b"a different charter").hexdigest()
    with caplog.at_level(logging.WARNING):
        out = verify_charter("0x" + wrong)
    assert out["match"] is False
    assert out["chain_hash"] == wrong
    assert out["local_hash"] == charter_hash()
    assert any("CHARTER DIVERGENCE" in r.message for r in caplog.records)


def test_verify_no_charter_published():
    from atn.on_chain import verify_charter

    out = verify_charter("")
    assert out["match"] is None
    assert out["chain_hash"] is None
    assert out["local_hash"] == charter_hash()
