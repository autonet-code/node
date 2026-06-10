"""Usefulness coordinate space.

Maps (problem, resolution) text into a coordinate vector that the
substrate uses to locate work in the graph. The choice of embedder
is pluggable: sentence-transformers is the simplest concrete starter
but the same interface accepts any model that produces a fixed-size
vector for a string.

Two backends provided:

  SentenceTransformersEmbedder
    Uses sentence-transformers/all-MiniLM-L6-v2 (384 dim) by default.
    Soft import: if sentence-transformers isn't installed, raises a
    clear error instead of crashing at module load.

  HashingEmbedder
    Deterministic fallback. Hashes word n-grams into a fixed-size
    sparse vector. Cheap, no dependencies, much weaker semantics
    but works for tests and bootstrapping.

A factory `default_usefulness_embedder()` returns the best available
(prefers sentence-transformers, falls back to hashing).

Reduction
---------

The raw embedding (~384 dims) is too high to use directly as graph
coordinates. We reduce to a small number of dims via random
projection (Johnson-Lindenstrauss preserves distances approximately).
The substrate uses 16D by default.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from dataclasses import dataclass, field
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

    Cheap, dependency-free. Good enough for testing and for early
    bootstrapping when sentence-transformers isn't available.
    """

    dim: int = DEFAULT_DIM
    seed: int = 42

    def __call__(self, text: str) -> Tuple[float, ...]:
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
# Sentence-transformers embedder (soft dependency)
# ---------------------------------------------------------------------------


@dataclass
class SentenceTransformersEmbedder:
    """Embedder backed by sentence-transformers.

    Lazy-imports the model on first call. Reduces output via random
    projection to `dim`-dimensional coords.
    """

    dim: int = DEFAULT_DIM
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 42
    _model: Any = field(default=None, init=False, repr=False)
    _projection: Any = field(default=None, init=False, repr=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. Install it or use "
                "HashingEmbedder as a dependency-free fallback."
            ) from e
        self._model = SentenceTransformer(self.model_name)
        embed_dim = self._model.get_sentence_embedding_dimension()
        # Random projection from embed_dim -> self.dim
        rng = random.Random(self.seed)
        self._projection = [
            [rng.gauss(0.0, 1.0 / math.sqrt(self.dim)) for _ in range(embed_dim)]
            for _ in range(self.dim)
        ]

    def __call__(self, text: str) -> Tuple[float, ...]:
        if not text:
            return tuple([0.0] * self.dim)
        self._ensure_model()
        emb = self._model.encode(text, convert_to_numpy=False)
        emb_list = list(emb)
        # Project
        out = []
        for row in self._projection:
            s = 0.0
            for r, v in zip(row, emb_list):
                s += r * float(v)
            out.append(s)
        return _l2_normalize(out)


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


_ST_IMPORT_TIMEOUT_SECONDS = 30.0
_st_import_thread = None


def _import_st_async() -> "threading.Thread":
    """Start (once) a background import of sentence_transformers.

    The import is heavy (~30s cold: torch + transformers) and has been
    observed to WEDGE indefinitely when first triggered from a daemon
    worker thread (loader-lock interaction). Running it on a dedicated
    thread lets callers wait with a timeout instead of blocking the
    feed cycle forever; once it lands in sys.modules every later call
    is instant.
    """
    import threading
    global _st_import_thread
    if _st_import_thread is None:
        def _do_import() -> None:
            try:
                import sentence_transformers  # type: ignore  # noqa: F401
            except Exception:
                pass
        _st_import_thread = threading.Thread(
            target=_do_import, name="st-embedder-import", daemon=True,
        )
        _st_import_thread.start()
    return _st_import_thread


def default_usefulness_embedder(dim: int = DEFAULT_DIM):
    """Return the best available embedder.

    Prefers SentenceTransformersEmbedder; falls back to HashingEmbedder
    if sentence-transformers isn't importable — or doesn't finish
    importing within the timeout (it keeps loading in the background,
    so later calls upgrade to the real embedder automatically).

    Embedder choice is NOT consensus-relevant: coords are computed by
    the authoring daemon and serialized into the event payload, so
    replays reproduce them regardless of the replayer's embedder.
    """
    import os
    import sys
    # Hard opt-out for deployments where the torch import deadlocks
    # in-process (observed on Windows daemons: the import wedges and
    # poisons the import lock for every other thread).
    if os.environ.get("ATN_USEFULNESS_EMBEDDER", "").lower() == "hashing":
        return HashingEmbedder(dim=dim)
    if "sentence_transformers" in sys.modules:
        return SentenceTransformersEmbedder(dim=dim)
    thread = _import_st_async()
    thread.join(timeout=_ST_IMPORT_TIMEOUT_SECONDS)
    if "sentence_transformers" in sys.modules:
        return SentenceTransformersEmbedder(dim=dim)
    logger.warning(
        "sentence_transformers not ready after %.0fs — using hashing "
        "embedder for this batch (will upgrade when the import lands)",
        _ST_IMPORT_TIMEOUT_SECONDS,
    )
    return HashingEmbedder(dim=dim)


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
