"""Tool standing — verdict-layer standing for manifests, with attested decay.

Design: ``docs/tool_substrate.md``. A manifest's raw standing is the
same quantity artifact inference uses (Σ net_score of PRO claims minus
CON claims carrying the manifest's digest, ledger pricing). What the
trust classes change is standing's *durability*:

  - ``pinned``   — code hash-locked, verdicts permanent: no decay.
  - ``attested`` — behavior lives outside the network: standing decays
    geometrically per epoch without a fresh usage receipt (rug-pull
    pricing).

Receipt history is derived exclusively from canonical ``tool_used``
events (``tool_usage.py``), so every honest daemon computes identical
decay — no wall clocks, no local state in the consensus math.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .infer import _artifact_standing, _standing_of
from .tool_manifest import TRUST_ATTESTED, TRUST_PINNED

# Proposed default (docs/tool_substrate.md open knob 3): five quiet
# epochs cut attested standing to ~a third. User-tunable; must be the
# same value at every daemon for a given epoch (config-gated like
# emission), NOT renegotiated per call.
DEFAULT_ATTESTED_DECAY = 0.8

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


def effective_standing(
    standing: float,
    trust_class: str,
    epochs_since_last_receipt: int,
    *,
    decay_rate: float = DEFAULT_ATTESTED_DECAY,
) -> float:
    """Apply trust-class durability to a raw standing.

    Pinned standing passes through untouched. Attested standing decays
    ``decay_rate ** epochs_since_last_receipt`` — an epoch WITH a
    receipt has distance 0 (no decay). Unknown trust classes are
    treated as attested: the conservative default for anything whose
    behavior the network cannot pin.
    """
    if trust_class == TRUST_PINNED:
        return round(standing, _STANDING_DECIMALS)
    if trust_class != TRUST_ATTESTED and trust_class:
        # Fall through to attested semantics deliberately.
        pass
    epochs = max(0, int(epochs_since_last_receipt))
    return round(standing * (decay_rate ** epochs), _STANDING_DECIMALS)


def update_receipt_history(
    history: Dict[str, int],
    used_digests: Iterable[str],
    known_digests: Iterable[str],
) -> Dict[str, int]:
    """Advance the epochs-since-last-receipt counter by one epoch close.

    ``used_digests``: manifests with ≥1 ``tool_used`` receipt in the
    closing epoch (keys of ``tool_usage_from_events``). ``known_digests``:
    every manifest the daemon tracks. A used manifest resets to 0; a
    quiet one increments (starting from the prior history, default 0
    for newly-seen manifests). Returns a new key-sorted dict — pure
    function of canonical inputs, bit-identical everywhere.
    """
    used = set(used_digests)
    out: Dict[str, int] = {}
    for digest in sorted(set(known_digests) | used):
        if digest in used:
            out[digest] = 0
        else:
            out[digest] = int(history.get(digest, 0)) + 1
    return out
