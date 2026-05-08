"""Locate primitive: find the graph region near a query.

A single function used by three different consumers in the system:

  - Training agents call locate(world, content) to find the region of
    the graph where a new judgment node should be posted.
  - The inference adapter calls locate(world, query) to find the
    region whose structure constitutes the response to a query.
  - The engine itself calls locate(world, sub_claim) to determine
    which neighboring nodes a sub-claim influences (subsumes the
    locality rule).

Same primitive in three roles. The implementation is swappable so
nodes that have richer notions of distance (embeddings, etc.) can
plug them in without touching the substrate.

Core types
----------

  RegionMember = (tendency_id: str, node_id: str, distance: float)
  Region       = list[RegionMember], sorted by distance ascending.
  Locator      = callable: (world, content) -> Region

Baselines
---------

  CoordinateLocator: uses the world's existing coordinate-space.
    Distance = euclidean distance from content's coords to each
    node's claim anchor. content can be:
      - tuple of floats (raw coords)
      - Observation
      - dict with a "coords" field
      - any object with a .coords attribute

  KeywordLocator: substring-match on node.content. Distance =
    1 - (matched_keywords / total_keywords). Crude but works for
    text without an embedding layer.

  ChainLocator: tries multiple locators in order, returns the first
    non-empty region. Useful as the default when content type isn't
    known up front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from .observation import Observation
from .world import World


RegionMember = Tuple[str, str, float]   # (tendency_id, node_id, distance)
Region = List[RegionMember]
Locator = Callable[[World, Any], Region]


# ---------------------------------------------------------------------------
# Helpers: extract coords or text from heterogeneous content
# ---------------------------------------------------------------------------


def _coords_of(content: Any) -> Optional[Tuple[float, ...]]:
    if content is None:
        return None
    if isinstance(content, tuple):
        if all(isinstance(x, (int, float)) for x in content):
            return tuple(float(x) for x in content)
    if isinstance(content, list):
        if all(isinstance(x, (int, float)) for x in content):
            return tuple(float(x) for x in content)
    if isinstance(content, Observation):
        return content.coords
    if isinstance(content, dict):
        if "coords" in content:
            c = content["coords"]
            if isinstance(c, (list, tuple)):
                return tuple(float(x) for x in c)
    coords = getattr(content, "coords", None)
    if isinstance(coords, (list, tuple)):
        return tuple(float(x) for x in coords)
    anchor = getattr(content, "anchor", None)
    if isinstance(anchor, (list, tuple)):
        return tuple(float(x) for x in anchor)
    return None


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Observation):
        return content.label or ""
    if isinstance(content, dict):
        for key in ("text", "content", "label", "query", "request"):
            v = content.get(key)
            if isinstance(v, str):
                return v
    label = getattr(content, "label", None)
    if isinstance(label, str):
        return label
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _euclid(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


# ---------------------------------------------------------------------------
# Baseline 1: coordinate proximity
# ---------------------------------------------------------------------------


@dataclass
class CoordinateLocator:
    """Distance = euclidean distance between content's coords and each
    node's claim anchor.

    A max_distance threshold filters out nodes that aren't really in
    the region; defaults to None (all nodes returned, sorted).
    """

    max_distance: Optional[float] = None
    max_results: Optional[int] = None

    def __call__(self, world: World, content: Any) -> Region:
        coords = _coords_of(content)
        if coords is None:
            return []
        out: Region = []
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                claim = tendency._node_to_claim.get(node.id)
                if claim is None or not claim.anchor:
                    continue
                d = _euclid(coords, claim.anchor)
                if self.max_distance is not None and d > self.max_distance:
                    continue
                out.append((tendency.id, node.id, d))
        out.sort(key=lambda r: r[2])
        if self.max_results is not None:
            out = out[: self.max_results]
        return out


# ---------------------------------------------------------------------------
# Baseline 2: keyword overlap
# ---------------------------------------------------------------------------


@dataclass
class KeywordLocator:
    """Distance = 1 - (overlapping_words / total_query_words).

    Tokenizes both the query text and node.content into lowercase
    word sets; computes Jaccard-style proximity. Cheap, deterministic,
    no embedding layer.
    """

    min_overlap: float = 0.05   # nodes below this are not in region
    max_results: Optional[int] = None

    def __call__(self, world: World, content: Any) -> Region:
        text = _text_of(content)
        if not text:
            return []
        query_words = _tokenize(text)
        if not query_words:
            return []
        out: Region = []
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                node_text = node.content or ""
                node_words = _tokenize(node_text)
                if not node_words:
                    continue
                overlap = len(query_words & node_words) / len(query_words)
                if overlap < self.min_overlap:
                    continue
                d = 1.0 - overlap
                out.append((tendency.id, node.id, d))
        out.sort(key=lambda r: r[2])
        if self.max_results is not None:
            out = out[: self.max_results]
        return out


def _tokenize(text: str) -> set[str]:
    """Cheap word tokenization. Lowercases, splits on non-alpha, drops
    short stopword-like tokens."""
    if not text:
        return set()
    out: set[str] = set()
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                w = "".join(current)
                if len(w) >= 3:
                    out.add(w)
                current = []
    if current:
        w = "".join(current)
        if len(w) >= 3:
            out.add(w)
    return out


# ---------------------------------------------------------------------------
# Composite: try locators in order, return first non-empty region
# ---------------------------------------------------------------------------


@dataclass
class ChainLocator:
    """Try a sequence of locators; return the first non-empty region.

    Useful when content type isn't known up front: try coordinate
    lookup first (cheap, exact), fall back to keyword lookup if no
    coords present.
    """

    locators: List[Locator]

    def __call__(self, world: World, content: Any) -> Region:
        for loc in self.locators:
            r = loc(world, content)
            if r:
                return r
        return []


# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------


def default_locator() -> Locator:
    """Coordinate-first, keyword-fallback. The most useful default for
    callers that don't know what kind of content they have.
    """
    return ChainLocator(locators=[
        CoordinateLocator(max_results=64),
        KeywordLocator(max_results=64),
    ])
