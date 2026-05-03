"""Event stream: the substrate's protocol-layer output.

A solver's contribution per task is a sequence of events. The
aggregator replays them on the global world to derive the new state;
the engine's equilibration handles score propagation and lineage.

Two event types
---------------

  - SubClaimSprouted: a new sub-claim node was added under a parent.
    Carries content-addressed node_id (or solver-local id, if the
    engine hasn't switched to content-addressing yet), parent_id,
    position, coords, polarity_axis, content, and the tendency_id
    the sprout lives under.

  - ObservationAdded: an observation was attached to the world. Carries
    obs_id and coords. The act of adding may absorb the observation
    into one or more tendencies' frames depending on equilibration.

Each event carries the author (agent_id) and a sequence index. The
sequence index is unique within the solver's submission and lets us
replay deterministically.

Why events instead of stake-deltas
----------------------------------

In the architecture as understood:
  - Score on a node is the equilibrium consequence of the events
    that landed on/under it, not an independently-tracked quantity.
  - Mint is computed from score-change at epoch close, attributed to
    the events that caused it.
  - There is no per-node stake bookkeeping that needs to be carried
    in the protocol; events are the unit of contribution.

So the solver's output is its event stream. The aggregator merges
streams, replays on the global world, and reads scores after.

Single-parent vs multi-parent shape
-----------------------------------

The protocol-layer event carries a primary (parent_id, position,
tendency_id) tuple. There's an optional `parents` list field for
forward-compatibility with explicit multi-parent emission, but in
practice the engine handles co-parenting at replay time: when
multiple events land on the same content-addressed coordinate
hash, the engine's sprout_child collision detection + cross-
tendency edge discovery accumulates parent edges automatically.

So solvers can emit single-parent events (the natural shape) and
federation merge still produces correctly co-parented nodes on
the live world. The `parents` list is there if a solver has
explicit reason to emit a multi-parented event in one shot, but
it's not required for the federation story to work.

See world-model docs/substrate-architecture.md for the
post-and-coparent + dedup mechanics on the engine side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SubClaimSprouted:
    """A sub-claim node was sprouted under one or more parents.

    Single-parent events (legacy) populate `parent_id` + `tendency_id`
    + `position` directly. Multi-parent events (post-and-coparent
    refactor) populate the optional `parents` list with one entry per
    edge: each entry is {"parent_id": ..., "position": "pro"|"con",
    "tendency_id": ...}. The aggregator treats single-parent events as
    a length-1 parents list at apply time.
    """
    kind: str = "sub_claim_sprouted"
    seq: int = 0
    author_agent: str = ""
    tendency_id: str = ""
    parent_id: str = ""           # solver-side; aggregator will remap to live id
    node_id: str = ""             # solver-side; aggregator may remap or accept
    position: str = "pro"         # "pro" | "con"
    coords: List[float] = field(default_factory=list)
    polarity_axis: List[float] = field(default_factory=list)
    content: str = ""
    # Optional reference to an Observation this sub-claim was sprouted
    # for. Carried through replay so locate can map back to the
    # underlying work unit.
    observation_id: str = ""
    # Multi-parent edge list (post-and-coparent refactor). When empty,
    # the aggregator falls back to (parent_id, position, tendency_id).
    # When non-empty, takes precedence; each entry has keys
    # parent_id, position, tendency_id.
    parents: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationAdded:
    """An observation was added to the world during training."""
    kind: str = "observation_added"
    seq: int = 0
    author_agent: str = ""
    obs_id: str = ""
    coords: List[float] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Event = Any   # SubClaimSprouted | ObservationAdded


def event_from_dict(d: Dict[str, Any]) -> Event:
    kind = d.get("kind")
    if kind == "sub_claim_sprouted":
        return SubClaimSprouted(**d)
    if kind == "observation_added":
        return ObservationAdded(**d)
    raise ValueError(f"unknown event kind: {kind!r}")


# ---------------------------------------------------------------------------
# Score snapshot at epoch boundary
# ---------------------------------------------------------------------------


def snapshot_node_scores(world) -> Dict[str, float]:  # type: ignore[no-untyped-def]
    """Map node_id -> net_score across all tendencies' trees."""
    out: Dict[str, float] = {}
    for tendency in world.tendencies.values():
        for node in tendency.tree.all_nodes():
            out[node.id] = node.net_score
    return out
