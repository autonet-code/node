"""Aggregator-side: replay solvers' event streams onto the global world.

Same protocol slot as ``aggregate_weight_deltas``: takes a list of
contributions plus optional weights, returns a merged payload that can
be applied to the global world to produce the next epoch's state.

Merge semantics
---------------

Solvers' contributions are event streams plus their own post-training
score snapshots. The aggregator:

  1. Concatenates all events across solvers, sorted by (author, seq) so
     ordering is deterministic across replays.
  2. Replays each event on the live world. Sprouts under the same
     content-addressed id from different solvers consolidate naturally
     (the engine's _ensure_obs_child / sprout_child treats them as the
     same node).
  3. Equilibrates after the merged replay.
  4. Returns the merged event list plus the post-replay score snapshot.

This is associative (concatenation is associative) and weight-aware:
the optional weights downstream affect mint computation per agent
but not the act of replaying events. Every event lands; weighting
applies only when computing per-agent rewards from the resulting
score-changes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from world_model.generalized import World, equilibrate
from world_model.models.tree import Position

from .events import Event, ObservationAdded, SubClaimSprouted, event_from_dict


logger = logging.getLogger(__name__)


def aggregate_contributions(
    contributions: List[Dict[str, Any]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Merge a list of solver contributions into a single payload.

    Returns:
      {
        "events": [...],            # concatenated, deterministically ordered
        "agent_weights": {agent_id: weight},
      }
    """
    all_events: List[Dict[str, Any]] = []
    for contrib in contributions:
        for ev in contrib.get("events", []):
            all_events.append(ev)
    # Deterministic order: by (author_agent, seq) so replays match
    all_events.sort(key=lambda e: (e.get("author_agent", ""), e.get("seq", 0)))

    agent_weights: Dict[str, float] = {}
    if weights is None:
        for contrib in contributions:
            agent_weights[contrib.get("agent_id", "")] = 1.0
    else:
        for contrib, w in zip(contributions, weights):
            agent_weights[contrib.get("agent_id", "")] = float(w)

    logger.info(
        "aggregated %d contributions: %d events, %d agents",
        len(contributions), len(all_events), len(agent_weights),
    )
    return {
        "events": all_events,
        "agent_weights": agent_weights,
    }


def apply_events(world: World, events: List[Dict[str, Any]]) -> World:
    """Replay an event stream onto a world. Mutates the world in place.

    Two event kinds:
      - observation_added: add an Observation by coords/id.
      - sub_claim_sprouted: sprout a child node under a parent.

    Sprouts target nodes by their solver-side ids. Because the engine
    uses content-addressed ids, two solvers proposing the same sub-claim
    end up with the same id; their replays consolidate naturally.
    """
    from world_model.generalized import Observation

    # parent-id remap: solver-side root id -> live root id, by tendency.
    # We learn solver root ids from sprout events whose parent is the
    # tendency's root.
    remap: Dict[str, str] = {}

    # Index sprouts by their solver-side node_id so we can resolve
    # "parent is itself a sprout" cases via remap as we go.
    sprouts_by_id: Dict[str, Dict[str, Any]] = {
        e["node_id"]: e for e in events
        if e.get("kind") == "sub_claim_sprouted"
    }

    def resolve_parent(ev: Dict[str, Any]) -> Optional[str]:
        parent_id = ev.get("parent_id", "")
        if parent_id in remap:
            return remap[parent_id]
        tendency = world.tendencies.get(ev.get("tendency_id", ""))
        if tendency is None:
            return None
        if parent_id not in sprouts_by_id:
            # Solver's parent_id refers to the tendency's root.
            live_root_id = tendency.tree.root_node.id
            remap[parent_id] = live_root_id
            return live_root_id
        return remap.get(parent_id)

    pending_sprouts: List[Dict[str, Any]] = []

    for ev in events:
        kind = ev.get("kind")
        if kind == "observation_added":
            world.add_observation(Observation(
                id=ev.get("obs_id", ""),
                coords=tuple(ev.get("coords", [])),
                label=ev.get("label", ""),
            ))
        elif kind == "sub_claim_sprouted":
            pending_sprouts.append(ev)
        else:
            logger.warning("unknown event kind: %s", kind)

    def parent_links_of(ev: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return the list of parent links for this sprout event.

        Multi-parent events carry a `parents` list explicitly. Legacy
        single-parent events get coerced to a length-1 list.
        """
        parents_list = ev.get("parents") or []
        if parents_list:
            return parents_list
        return [{
            "parent_id": ev.get("parent_id", ""),
            "position": ev.get("position", "pro"),
            "tendency_id": ev.get("tendency_id", ""),
        }]

    # Sprout in waves until none remain or no progress is made
    safety = 0
    while pending_sprouts and safety < 10:
        safety += 1
        next_round: List[Dict[str, Any]] = []
        for ev in pending_sprouts:
            links = parent_links_of(ev)
            # Resolve every link to (live_tendency, live_parent_id) up
            # front; if any link references an unknown tendency or has
            # an unresolved parent, defer the whole event.
            resolved_links: List[tuple] = []
            stranded = False
            for link in links:
                tend = world.tendencies.get(link.get("tendency_id", ""))
                if tend is None:
                    logger.warning(
                        "sprout targets unknown tendency %s",
                        link.get("tendency_id"),
                    )
                    stranded = True
                    break
                # Reuse resolve_parent's logic per-link by patching the
                # event-shaped dict it expects.
                proxy = {
                    "tendency_id": link.get("tendency_id", ""),
                    "parent_id": link.get("parent_id", ""),
                }
                live_parent = resolve_parent(proxy)
                if live_parent is None:
                    stranded = True
                    break
                resolved_links.append((tend, live_parent, link.get("position", "pro")))
            if stranded:
                next_round.append(ev)
                continue
            if not resolved_links:
                continue
            try:
                # Apply the first parent link via sprout_child to
                # create (or find by content hash) the node.
                first_tend, first_parent, first_pos = resolved_links[0]
                new_node = first_tend.sprout_child(
                    parent_node_id=first_parent,
                    position=Position(first_pos),
                    anchor=tuple(ev.get("coords") or ()),
                    polarity_axis=tuple(ev.get("polarity_axis") or ()),
                    content=ev.get("content", ""),
                    world=world,
                )
                # Apply remaining parent links via each tendency's
                # sprout_child; the existing-node branch will append
                # parent_links and update child lists.
                for tend, parent, pos in resolved_links[1:]:
                    tend.sprout_child(
                        parent_node_id=parent,
                        position=Position(pos),
                        anchor=tuple(ev.get("coords") or ()),
                        polarity_axis=tuple(ev.get("polarity_axis") or ()),
                        content=ev.get("content", ""),
                        world=world,
                    )
                # Carry observation_id through replay if present.
                obs_id = ev.get("observation_id", "")
                if obs_id:
                    new_node.observation_id = obs_id
                remap[ev["node_id"]] = new_node.id
            except Exception as e:
                logger.warning("sprout failed: %s", e)
        if len(next_round) == len(pending_sprouts):
            for ev in next_round:
                logger.warning("stranded sprout %s (parent %s unresolved)",
                               ev["node_id"], ev.get("parent_id"))
            break
        pending_sprouts = next_round

    # Equilibrate so post-replay scores reflect the merged state
    equilibrate(world, max_rounds=8, tolerance=1e-3)
    return world


# ---------------------------------------------------------------------------
# Backwards-compatible aliases for old code paths
# ---------------------------------------------------------------------------


def aggregate_stake_deltas(
    deltas: List[Dict[str, Any]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compatibility shim so older callers keep working. Treats stake
    deltas as event-bearing contributions when possible.
    """
    return aggregate_contributions(deltas, weights)


def apply_stake_delta(world: World, delta: Dict[str, Any]) -> World:
    """Compatibility shim: if delta has 'events', replay them.
    Otherwise, no-op (legacy stake-delta path is removed)."""
    events = delta.get("events", [])
    return apply_events(world, events)
