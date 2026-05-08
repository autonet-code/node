"""Generalized world model.

Provides a non-LLM, coordinate-space concrete implementation of the
existing novelty/reference-frame abstractions, plus a tendency type
that owns its own reference frame, has a budget, and acts via the
formalized novelty loop.

This subpackage doesn't replace the existing abstractions in
``world_model.novelty.core`` -- it implements them for domain-agnostic
fundamental-data settings where there is no language and no LLM.

Core types:

  CoordinateClaim    -- a claim positioned at a point in observation
                        space, with a "polarity axis" (a unit vector
                        defining which direction is PRO).
  CoordinateFrame    -- ReferenceFrame implementation. Similarity is
                        inverse Euclidean distance; stance is sign of
                        the dot product of (content - claim_point)
                        onto the polarity axis, gated by topical
                        proximity.
  CoordinateProbe    -- NoveltyProbe implementation. Walks the claim
                        hierarchy by adjacency in coordinate space.
  GeneralizedTendency -- thesis (root claim) + tree + budget + frame +
                         probe. The ``act`` method runs novelty
                         measurement on incoming observations and on
                         other tendencies' nodes, then redistributes
                         budget accordingly.
  World              -- coalition of tendencies. Sum of theses =
                        modeled world. Cross-stakes between tendencies
                        populate child nodes in each other's trees.
  equilibrate        -- run tendencies' actions until budgets stabilize.
"""

from .coordinate_frame import (
    CoordinateClaim,
    CoordinateFrame,
    CoordinateProbe,
)
from .tendency import GeneralizedTendency
from .world import World, Observation
from .equilibrate import equilibrate, equilibrate_with_growth, equilibrate_continuous
from .grow import propose_growth
from .prune import ScoreHistory, prune_settled_negatives, snapshot_scores
from .locate import (
    Region,
    RegionMember,
    Locator,
    CoordinateLocator,
    KeywordLocator,
    ChainLocator,
    default_locator,
)
from .render import render
from .decay import StabilityTracker, is_decayed

__all__ = [
    "CoordinateClaim",
    "CoordinateFrame",
    "CoordinateProbe",
    "GeneralizedTendency",
    "World",
    "Observation",
    "equilibrate",
    "equilibrate_with_growth",
    "equilibrate_continuous",
    "propose_growth",
    "ScoreHistory",
    "prune_settled_negatives",
    "snapshot_scores",
    "Region",
    "RegionMember",
    "Locator",
    "CoordinateLocator",
    "KeywordLocator",
    "ChainLocator",
    "default_locator",
    "render",
    "StabilityTracker",
    "is_decayed",
]
