"""Tool standing — verdict-layer standing for manifests.

Design: ``docs/tool_substrate.md`` v2. A manifest's standing is the
same quantity artifact inference uses (Σ net_score of PRO claims minus
CON claims carrying the manifest's digest, ledger pricing).

v1's attested-decay functions were RETIRED with the Tools/Services
split: decay existed to compensate for endpoint-backed tools'
unverifiability, and those offerings are now Services
(``docs/services_market.md``) with no substrate standing at all.
Connector-backed (``attested``) tools remain publishable and debatable
but draw no mint, so nothing here needs a time dimension.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .infer import _artifact_standing, _standing_of

_STANDING_DECIMALS = 10


def manifest_standing(world: Any, digest: str) -> float:
    """Raw verdict-layer standing for one manifest digest.

    Works on non-equilibrated (ledger-mode) worlds: ``net_score`` is
    the plain tree recursion.
    """
    nodes = _artifact_standing(world).get(digest, [])
    return round(_standing_of(nodes), _STANDING_DECIMALS)


def all_manifest_standings(world: Any, digests: Iterable[str]) -> Dict[str, float]:
    """Standing for many digests with ONE world walk, canonically ordered."""
    by_digest = _artifact_standing(world)
    out: Dict[str, float] = {}
    for digest in sorted(set(digests)):
        out[digest] = round(_standing_of(by_digest.get(digest, [])),
                            _STANDING_DECIMALS)
    return out
