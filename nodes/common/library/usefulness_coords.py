"""Usefulness coordinate space.

Maps (problem, resolution) text into a coordinate vector that the
substrate uses to locate work in the graph.

Backends:

  The local search model (``minilm.py``)
    all-MiniLM-L6-v2 over onnxruntime, fetched and verified by the daemon
    (atn/search_model.py). Its 384-dim output is reduced to the world's
    embedding width by a fixed random projection (``ProjectedEmbedder``).

  HashingEmbedder
    Deterministic word-trigram hashing. For tests (``ATN_USEFULNESS_EMBEDDER
    =hashing``), and for world coords authored while the model is still
    being fetched.

Tool SEARCH does not use this module: it embeds with the full 384-dim
model in the daemon's own index (atn/tool_search.py, Decision 2026-11-08).
What remains here authors the embedding tail of the world's coordinates.

Reduction
---------

The raw embedding (384 dims) is reduced to the world's embedding width by
random projection (Johnson-Lindenstrauss preserves distances
approximately).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Phase 2.2 (dim_sweep.py) measured categorical separation across
# dims on the work_units corpus and found 64 retains 95% of native-384
# separation while 32 collapses to 84% and 16 to 73%. Default raised
# from 16 → 64 to match. The original 16-dim system was conservative
# for early bootstrapping; with PCA/JL reduction at 64 the contest
# has substantially more room to differentiate work units.
DEFAULT_DIM = 64


# ---------------------------------------------------------------------------
# Hashing fallback embedder
# ---------------------------------------------------------------------------


@dataclass
class HashingEmbedder:
    """Deterministic word-trigram hashing embedder.

    For each input, tokenize into trigrams of words, hash each trigram
    into one of `dim` buckets, count frequencies. L2-normalize the
    result. Distances reflect rough topical overlap.

    Cheap, dependency-free. For tests, and for world coords authored
    before the search model is loaded.
    """

    dim: int = DEFAULT_DIM
    seed: int = 42

    def __call__(self, text: str) -> Tuple[float, ...]:
        if self.dim <= 0:
            # Pure-charter mode (embedding_dim=0): there is no tail to fill.
            return ()
        if not text:
            return tuple([0.0] * self.dim)
        tokens = _tokenize(text)
        if len(tokens) < 1:
            return tuple([0.0] * self.dim)
        vec = [0.0] * self.dim
        # Word trigrams; for very short texts fall back to bigrams or unigrams
        if len(tokens) >= 3:
            ngrams = _ngrams(tokens, 3)
        elif len(tokens) >= 2:
            ngrams = _ngrams(tokens, 2)
        else:
            ngrams = [(tokens[0],)]
        for ng in ngrams:
            h = int(hashlib.sha256(("_".join(ng) + str(self.seed)).encode()).hexdigest()[:16], 16)
            vec[h % self.dim] += 1.0
        return _l2_normalize(vec)


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{1,}\b", text)]


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _l2_normalize(vec: List[float]) -> Tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return tuple(vec)
    return tuple(v / norm for v in vec)


# ---------------------------------------------------------------------------
# The local search model, projected to the world's embedding width
# ---------------------------------------------------------------------------


class ProjectedEmbedder:
    """The loaded search model (``minilm.shared()``) reduced to ``dim``
    dimensions by a fixed Gaussian projection (seeded, so every daemon uses
    the same matrix), then L2-normalized."""

    def __init__(self, model: Any, dim: int = DEFAULT_DIM, seed: int = 42):
        import numpy as np
        self.model = model
        self.dim = dim
        rng = random.Random(seed)
        width = int(getattr(model, "dim", 384))
        self._projection = np.asarray(
            [[rng.gauss(0.0, 1.0 / math.sqrt(max(dim, 1))) for _ in range(width)]
             for _ in range(dim)], dtype=np.float64)

    def __call__(self, text: str) -> Tuple[float, ...]:
        if self.dim <= 0:
            return ()
        if not text:
            return tuple([0.0] * self.dim)
        import numpy as np
        emb = np.asarray(self.model(text), dtype=np.float64)
        return _l2_normalize([float(v) for v in self._projection @ emb])


_projected: dict = {}


def _projected_embedder(dim: int):
    """The shared projected embedder for ``dim``, or None while the search
    model is not loaded."""
    from . import minilm
    model = minilm.shared()
    if model is None:
        return None
    hit = _projected.get(dim)
    if hit is None or hit.model is not model:
        hit = ProjectedEmbedder(model, dim=dim)
        _projected[dim] = hit
    return hit


def _hashing_forced() -> bool:
    import os
    return os.environ.get("ATN_USEFULNESS_EMBEDDER", "").lower() == "hashing"


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def default_usefulness_embedder(dim: int = DEFAULT_DIM):
    """The embedder for world coords at width ``dim``. Never blocks.

    The local search model projected to ``dim`` once it is loaded; until
    then (the first run downloads it) the hashing embedder. Tests force
    hashing with ``ATN_USEFULNESS_EMBEDDER=hashing``.

    Embedder choice is NOT consensus-relevant: coords are computed by
    the authoring daemon and serialized into the event payload, so
    replays reproduce them regardless of the replayer's embedder.
    """
    if _hashing_forced():
        return HashingEmbedder(dim=dim)
    return _projected_embedder(dim) or HashingEmbedder(dim=dim)


def usefulness_embedder_if_ready(dim: int = DEFAULT_DIM, wait: float = 0.0):
    """The SEMANTIC embedder when the search model is loaded within
    ``wait`` seconds, else None: for interactive callers that would rather
    answer "not yet" than cluster on hashing coords."""
    import time
    if _hashing_forced():
        return HashingEmbedder(dim=dim)
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        emb = _projected_embedder(dim)
        if emb is not None or time.monotonic() >= deadline:
            return emb
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


# ---------------------------------------------------------------------------
# Coords for a (problem, resolution) pair
# ---------------------------------------------------------------------------


def coords_for_problem_resolution(
    problem: str,
    resolution: str,
    embedder: Optional[Callable[[str], Tuple[float, ...]]] = None,
) -> Tuple[float, ...]:
    """Combine the problem and resolution strings into a single
    coord vector. The embedder runs on a concatenation that gives
    the problem more weight (it's what locate queries against).
    """
    if embedder is None:
        embedder = default_usefulness_embedder()
    text = f"PROBLEM: {problem}\nRESOLUTION: {resolution}"
    return embedder(text)


def coords_for_query(
    query: str,
    embedder: Optional[Callable[[str], Tuple[float, ...]]] = None,
) -> Tuple[float, ...]:
    """Coord vector for a query/problem string at inference time."""
    if embedder is None:
        embedder = default_usefulness_embedder()
    return embedder(f"PROBLEM: {query}")


# ---------------------------------------------------------------------------
# Claim-coords verification
# ---------------------------------------------------------------------------


def _cosine(a, b) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    ma = sum(x * x for x in a[:n]) ** 0.5
    mb = sum(x * x for x in b[:n]) ** 0.5
    if ma == 0.0 or mb == 0.0:
        return 1.0 if ma == mb else 0.0
    return dot / (ma * mb)


def verify_claim_coords(
    coords,
    label: str,
    *,
    head_dims: Optional[int] = None,
    cosine_min: float = 0.995,
    dim: Optional[int] = None,
) -> dict:
    """Check that an event's coordinate tail really embeds its claim text.

    The claim (event ``label``) is the embedder input by protocol, so
    any peer can recompute the tail and compare. A post whose tail
    matches neither backend is carrying fabricated coordinates —
    text parked at an address its content doesn't embed to.

    Both backends are tried because authoring daemons legitimately
    fall back to the hashing embedder while the semantic model loads
    (hashing is bit-deterministic; the semantic model is pinned but
    floats may jitter across platforms, hence the cosine threshold).

    Returns {valid, cosine, backend} — backend is which embedder
    matched best.
    """
    if head_dims is None:
        from .charter import active_charter
        head_dims = active_charter().n_dims
    tail = tuple(coords)[head_dims:]
    if dim is None:
        dim = len(tail) or DEFAULT_DIM

    candidates = [("hashing", HashingEmbedder(dim=dim))]
    semantic = _projected_embedder(dim)
    if semantic is not None:
        candidates.append(("semantic", semantic))

    best_backend, best_cos = "", -1.0
    for name, embedder in candidates:
        try:
            recomputed = tuple(coords_for_query(label, embedder=embedder))
        except Exception:
            continue
        cos = _cosine(tail, recomputed)
        if cos > best_cos:
            best_backend, best_cos = name, cos

    return {
        "valid": best_cos >= cosine_min,
        "cosine": best_cos,
        "backend": best_backend,
    }
