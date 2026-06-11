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
# Subprocess embedder (default): torch never enters the daemon process
# ---------------------------------------------------------------------------


class SubprocessEmbedder:
    """Embeds via an out-of-process worker (embed_worker.py).

    Rationale: importing sentence_transformers/torch from a daemon
    worker thread has WEDGED on Windows — the import deadlocks during
    extension-DLL initialization (loader-lock interaction) and poisons
    the import machinery for every other thread. A repro matrix
    (2026-06-11) shows the same import is reliable on the main thread
    of a fresh process across asyncio/trio/thread contexts, so the
    structural fix is to keep torch out of the daemon process
    entirely.

    Behavior: the worker is spawned lazily; ``ready_within`` reports
    whether the model finished loading. Calls have a per-request
    timeout; a dead or timed-out worker raises and is restarted at
    most once per call. Thread-safe via a lock (the substrate feed is
    single-threaded; the lock is cheap insurance).
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str = "st",
        request_timeout: float = 60.0,
    ):
        self.dim = dim
        self.model_name = model_name
        self.backend = backend
        self.request_timeout = request_timeout
        self._proc = None
        self._ready = False
        self._lock = None  # created lazily (threading import kept local)

    def _ensure_proc(self):
        import subprocess
        import sys as _sys
        import threading
        from pathlib import Path
        if self._lock is None:
            self._lock = threading.RLock()
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        worker = Path(__file__).resolve().parent / "embed_worker.py"
        self._proc = subprocess.Popen(
            [_sys.executable, "-u", str(worker), str(self.dim),
             self.model_name, self.backend],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        self._ready = False
        return self._proc

    def ready_within(self, timeout: float) -> bool:
        """Spawn (if needed) and wait up to ``timeout`` for the READY
        line. Non-blocking beyond the timeout: the worker keeps
        loading and a later call can find it ready."""
        import json as _json
        import threading
        proc = self._ensure_proc()
        if self._ready:
            return True

        result: list = []

        def _read():
            line = proc.stdout.readline()
            result.append(line)

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=timeout)
        if not result or not result[0]:
            return False
        try:
            self._ready = bool(_json.loads(result[0]).get("ready"))
        except Exception:
            self._ready = False
        return self._ready

    def __call__(self, text: str) -> Tuple[float, ...]:
        import json as _json
        import threading
        if not text:
            return tuple([0.0] * self.dim)
        if self._lock is None:
            self._lock = threading.RLock()
        with self._lock:
            proc = self._ensure_proc()
            if not self._ready and not self.ready_within(self.request_timeout):
                raise RuntimeError("embed worker not ready")
            proc.stdin.write(_json.dumps({"text": text}) + "\n")
            proc.stdin.flush()

            result: list = []

            def _read():
                result.append(proc.stdout.readline())

            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            reader.join(timeout=self.request_timeout)
            if not result or not result[0]:
                # Wedged or dead worker: kill so the next call respawns.
                try:
                    proc.kill()
                except Exception:
                    pass
                self._proc = None
                self._ready = False
                raise RuntimeError("embed worker timed out")
            resp = _json.loads(result[0])
            if "error" in resp:
                raise RuntimeError(f"embed worker error: {resp['error']}")
            return tuple(float(v) for v in resp["coords"])

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
            self._ready = False


_subprocess_embedder: Optional[SubprocessEmbedder] = None


def _shared_subprocess_embedder(dim: int) -> SubprocessEmbedder:
    """One worker per daemon process (model load is ~20s; per-batch
    spawning would dominate)."""
    global _subprocess_embedder
    if _subprocess_embedder is None or _subprocess_embedder.dim != dim:
        if _subprocess_embedder is not None:
            _subprocess_embedder.close()
        _subprocess_embedder = SubprocessEmbedder(dim=dim)
    return _subprocess_embedder


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

    Default: SubprocessEmbedder — semantic embeddings from an
    out-of-process worker, so torch never enters the daemon process
    (in-process imports have deadlocked on Windows worker threads).
    Falls back to HashingEmbedder for the current batch while the
    worker's model is still loading; later batches upgrade.

    ``ATN_USEFULNESS_EMBEDDER`` overrides:
      hashing     dependency-free hashing embedder, no subprocess
      inprocess   legacy in-process sentence-transformers (background
                  import + timeout fallback) — for platforms where
                  the in-process import is known safe
      subprocess  explicit default

    Embedder choice is NOT consensus-relevant: coords are computed by
    the authoring daemon and serialized into the event payload, so
    replays reproduce them regardless of the replayer's embedder.
    """
    import os
    import sys
    choice = os.environ.get("ATN_USEFULNESS_EMBEDDER", "subprocess").lower()

    if choice == "hashing":
        return HashingEmbedder(dim=dim)

    if choice == "inprocess":
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

    # Default: out-of-process worker.
    embedder = _shared_subprocess_embedder(dim)
    if embedder.ready_within(_ST_IMPORT_TIMEOUT_SECONDS):
        return embedder
    logger.warning(
        "embed worker not ready after %.0fs — using hashing embedder "
        "for this batch (worker keeps loading; later batches upgrade)",
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
        from .adapter import N_DIMS
        head_dims = N_DIMS
    tail = tuple(coords)[head_dims:]
    if dim is None:
        dim = len(tail) or DEFAULT_DIM

    candidates = [("hashing", HashingEmbedder(dim=dim))]
    try:
        semantic = default_usefulness_embedder(dim=dim)
        if not isinstance(semantic, HashingEmbedder):
            candidates.append(("semantic", semantic))
    except Exception:
        pass

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
