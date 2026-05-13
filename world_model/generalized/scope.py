"""Locality-based scope computation for the discrete equilibrate kernel.

Given an incoming observation (or coordinate), returns the set of
tendency IDs whose bandwidth the coordinate falls within. This is the
operational fractal gate that lets scoped equilibrate skip
not-locally-affected tendencies on the hot path.

Lives at the same locality test the architecture already declares for
cross-tendency edge discovery in `_maybe_add_cross_tendency_edges`
(tendency.py): `distance(coord, anchor) < bandwidth * slack`. We keep
slack as a knob so callers (production hot path vs. exploration mode
vs. testing) can widen or narrow it explicitly.
"""

from __future__ import annotations

import math
from typing import Set, Tuple

from .world import World


def scope_for_coords(
    world: World,
    coords: Tuple[float, ...],
    slack: float = 1.5,
) -> Set[str]:
    """Tendencies whose anchor falls within `bandwidth * slack` of coords.

    The slack default of 1.5 matches the cross-tendency edge discovery
    gate already used elsewhere in the substrate, so scoped equilibrate
    is consistent with the existing locality declaration.
    """
    out: Set[str] = set()
    if not coords:
        return out
    for tid, t in world.tendencies.items():
        if not t.anchor:
            continue
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(coords, t.anchor)))
        if d < t.bandwidth * slack:
            out.add(tid)
    return out


def scope_for_observation(
    world: World,
    obs,
    slack: float = 1.5,
) -> Set[str]:
    """Convenience wrapper accepting an Observation directly."""
    return scope_for_coords(world, tuple(obs.coords or ()), slack=slack)
