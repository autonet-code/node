"""Canonical serialization + hash of the substrate charter.

The charter (the 6-root alignment vocabulary in ``adapter.CHARTER``) is a
hardcoded Python constant. Governance changing it is a forward-only fork
boundary (re-basing history breaks bit-identical replay — every historical
claim is embedded in the charter's coordinate basis). ``CharterAnchor.sol``
anchors WHICH version is canonical by committing the sha256 of the blob this
module produces; daemons compare their local hash against the on-chain anchor
and warn loudly on divergence.

This module is the single source of truth for that hash on the daemon side. It
derives the blob from the SAME constant ``build_charter_world`` uses — it does
not duplicate the charter values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from .adapter import CHARTER


def charter_payload() -> Dict[str, Any]:
    """The canonical charter structure, derived from ``adapter.CHARTER``.

    Each root is reduced to the consensus-relevant fields (id, axis_index,
    thesis, veto_floor). The result is a plain dict with a schema tag and a
    key-sorted ``roots`` list ordered by ``axis_index`` so the serialization
    is stable regardless of source declaration order.
    """
    roots = [
        {
            "id": entry["id"],
            "axis_index": entry["axis_index"],
            "thesis": entry["thesis"],
            "veto_floor": entry.get("veto_floor"),
        }
        for entry in CHARTER
    ]
    roots.sort(key=lambda r: r["axis_index"])
    return {
        "schema": 1,
        "n_dims": len(roots),
        "roots": roots,
    }


def charter_bytes() -> bytes:
    """Canonical UTF-8 bytes: JSON with sorted keys, no incidental whitespace.

    ``sort_keys=True`` + fixed separators make the byte string a pure function
    of the charter values — declaration order and dict insertion order are
    irrelevant, so every daemon computes identical bytes.
    """
    return json.dumps(
        charter_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def charter_hash() -> str:
    """sha256 hex digest of ``charter_bytes()`` — the anchor's ``charterHash``."""
    return hashlib.sha256(charter_bytes()).hexdigest()
