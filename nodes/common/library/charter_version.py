"""Canonical serialization + hash of the substrate charter.

The charter is data (docs/multi_jurisdiction.md decision 5): the
jurisdiction's DAO Registry holds its text under ``jurisdiction.<id>.charter``
as compact JSON ``{"axes":[{"id","thesis"},...]}``, and this process runs the
active charter installed at join (``charter.active_charter()``). The charter
hash is the sha256 of those exact registry bytes, so a daemon's local hash
equals the Registry's hash exactly when it runs the jurisdiction's charter.

Changing the charter is a forward-only boundary: every claim is embedded in
the charter's coordinate basis. A DAO vote takes effect at the next epoch
(daemons read it pinned to the previous anchor and restart onto it).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from .charter import active_charter


def charter_payload() -> Dict[str, Any]:
    """The active charter's structure: a schema tag plus its roots ordered
    by ``axis_index`` (id, axis_index, thesis, veto_floor)."""
    roots = [
        {
            "id": entry["id"],
            "axis_index": entry["axis_index"],
            "thesis": entry["thesis"],
            "veto_floor": entry.get("veto_floor"),
        }
        for entry in active_charter().entries()
    ]
    return {
        "schema": 1,
        "n_dims": len(roots),
        "roots": roots,
    }


def charter_bytes() -> bytes:
    """The active charter's registry bytes (UTF-8 compact JSON)."""
    return active_charter().text.encode("utf-8")


def charter_hash() -> str:
    """sha256 hex digest of ``charter_bytes()`` (no 0x prefix): the anchor's
    ``charterHash`` and the Registry charter hash without its prefix."""
    return hashlib.sha256(charter_bytes()).hexdigest()
