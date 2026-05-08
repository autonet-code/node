"""Novelty decay: settled regions stop drawing the engine's attention.

Once a node's score has been stable across N consecutive checkpoints,
the engine marks it decayed. Decayed nodes:

  - Don't generate fresh novelty when the probe encounters them
    (they're 'taken for granted').
  - Are deprioritized during equilibration: the engine still
    propagates scores through them but doesn't waste cycles
    re-evaluating their position.

Reactivation: any direct event (a fresh stake, a new child sprouted)
on a decayed node resets its stability counter and reactivates it.
A node that *was* settled but suddenly has fresh activity is exactly
where the engine should pay attention.

This lets the engine concentrate compute on the active growth
frontier as the tree grows. Without decay, every round revisits every
node forever.

Public API
----------

  StabilityTracker: maintains per-node consecutive-stable counters
    and the decay flag. Updated by .observe(world, history).

  is_decayed(world, node_id, tracker) -> bool
  reactivate(tracker, node_id)  -- call when fresh activity hits a node

Integration
-----------

The engine's equilibrate loop and probe call into the tracker as a
soft hint -- decayed nodes are skipped where cheap to do so. The
tracker doesn't mutate the world; it's a separate index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .prune import ScoreHistory
from .world import World


DEFAULT_STABILITY_WINDOW = 3       # consecutive stable checkpoints to decay
DEFAULT_STABILITY_THRESHOLD = 0.02 # max abs change between adjacent snapshots


@dataclass
class StabilityTracker:
    """Per-node stability state.

    Holds:
      - consecutive_stable[node_id]: number of recent adjacent
        score-snapshot pairs whose delta was below threshold.
      - decayed: set of node ids currently flagged decayed.

    Stability resets when an observe() pass finds the node moved
    above threshold, or when reactivate(node_id) is called.
    """

    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD
    stability_window: int = DEFAULT_STABILITY_WINDOW
    consecutive_stable: Dict[str, int] = field(default_factory=dict)
    decayed: set[str] = field(default_factory=set)

    def observe(self, world: World, history: ScoreHistory) -> None:
        """Walk every node; update its consecutive-stable count from
        the most recent two snapshots in history. Promote to decayed
        when the count crosses stability_window. Demote (and remove
        from decayed) when fresh movement is observed.
        """
        # Need at least 2 snapshots to compute a delta
        if len(history.snapshots) < 2:
            return
        # Each entry is (round_idx, score_dict). Pull the score dicts.
        _, last = history.snapshots[-1]
        _, prev = history.snapshots[-2]
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                # Skip the root -- never decayed (it's load-bearing
                # for inference even when stable).
                if node.parent_id is None:
                    continue
                if node.id not in last or node.id not in prev:
                    continue
                delta = abs(last[node.id] - prev[node.id])
                if delta < self.stability_threshold:
                    self.consecutive_stable[node.id] = (
                        self.consecutive_stable.get(node.id, 0) + 1
                    )
                    if self.consecutive_stable[node.id] >= self.stability_window:
                        self.decayed.add(node.id)
                else:
                    # Fresh movement: reset and reactivate.
                    self.consecutive_stable[node.id] = 0
                    self.decayed.discard(node.id)

    def observe_novelty(self, world: World, novelty_threshold: float = 0.1) -> None:
        """Persistent-novelty-based decay. A node whose persistent n has
        stayed below `novelty_threshold` for `stability_window` rounds
        is marked decayed. A node whose n bumps above threshold (e.g.
        due to CON-driven re-surprise) gets reactivated.

        Use this in place of observe(world, history) when the substrate
        is using the persistent-novelty refactor (see
        lindblad/NOVELTY_REFACTOR.md).
        """
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                if node.parent_id is None:
                    continue
                if node.n < novelty_threshold:
                    self.consecutive_stable[node.id] = (
                        self.consecutive_stable.get(node.id, 0) + 1
                    )
                    if self.consecutive_stable[node.id] >= self.stability_window:
                        self.decayed.add(node.id)
                else:
                    # n popped back up -- re-surprise. Reactivate.
                    self.consecutive_stable[node.id] = 0
                    self.decayed.discard(node.id)

    def is_decayed(self, node_id: str) -> bool:
        return node_id in self.decayed

    def reactivate(self, node_id: str) -> None:
        """Force-reactivate a node. Called when a fresh stake event,
        new sub-claim, or other direct activity hits the node.
        """
        self.consecutive_stable[node_id] = 0
        self.decayed.discard(node_id)

    def stats(self) -> Dict[str, int]:
        return {
            "n_tracked": len(self.consecutive_stable),
            "n_decayed": len(self.decayed),
        }


def is_decayed(node_id: str, tracker: Optional[StabilityTracker]) -> bool:
    """Convenience wrapper. Treats no-tracker as 'nothing is decayed'."""
    if tracker is None:
        return False
    return tracker.is_decayed(node_id)
