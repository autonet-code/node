"""Service topic vectors: the embedding cache behind ``service_topics``.

One process-wide cache of semantic coords per service spec digest, shared
by the WS handler (interactive clustering) and the boot-time warm-up. Specs
are content-addressed, so a digest's vector never changes and the cache
needs no invalidation, only growth.

Why a warm-up exists: embedding a spec is a round trip to the out-of-process
embed worker, ~0.5s each, and a catalogue can hold hundreds of specs. Paid
at request time that is a 30-40s first answer, far past the frontend's
timeout, and the Services page settled on "Everything else" until someone
tried again. Pre-embedding at boot makes the first real answer instant.

Only SEMANTIC vectors are cached. The hashing fallback embedder never
reaches this module: callers obtain the embedder through
``usefulness_embedder_if_ready`` and answer "pending" when it is not up.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

log = logging.getLogger(__name__)

# digest -> np.ndarray (float64) of semantic coords.
VEC_CACHE: dict[str, Any] = {}

# Held while a batch is being embedded, so the boot warm-up and an
# interactive request do not both push the whole catalogue through the
# worker. The request path polls it non-blocking and answers "pending"
# while the warm-up holds it.
EMBED_LOCK = threading.Lock()

# The coord dimension the service-topics rail embeds at (matches the
# substrate's DEFAULT_DIM; kept explicit here so the warm-up and the
# handler cannot drift apart).
DIM = 64


def embed_services(records: Iterable[Any], embedder) -> tuple[list[Any], list[str]]:
    """Coords for every record with embeddable text, cached by digest.

    Blocking: runs on a worker thread. Returns parallel lists
    ``(vectors, digests)`` in input order.
    """
    import numpy as np
    from .service_spec import service_embedding_text
    from nodes.common.world_model_substrate.usefulness_coords import coords_for_query

    vecs: list[Any] = []
    digests: list[str] = []
    for rec in records:
        vec = VEC_CACHE.get(rec.digest)
        if vec is None:
            text = service_embedding_text(rec.spec)
            if not text.strip():
                continue
            vec = np.asarray(coords_for_query(text, embedder), dtype=np.float64)
            VEC_CACHE[rec.digest] = vec
        vecs.append(vec)
        digests.append(rec.digest)
    return vecs, digests


def warm_service_vectors(store, *, ready_wait: float = 600.0) -> None:
    """Pre-embed the whole live catalogue on a background thread.

    Waits (off the event loop) for the embed worker to come up, then pushes
    every uncached spec through it under EMBED_LOCK. Any failure is logged
    at debug level: the request path degrades on its own.
    """
    def _warm() -> None:
        try:
            from nodes.common.world_model_substrate.usefulness_coords import (
                usefulness_embedder_if_ready,
            )
            embedder = usefulness_embedder_if_ready(DIM, ready_wait)
            if embedder is None:
                log.debug("service vector warm-up: embedder not ready")
                return
            records = list(store.list(include_retired=False))
            with EMBED_LOCK:
                vecs, _ = embed_services(records, embedder)
            log.info("service vector warm-up: %d spec(s) embedded", len(vecs))
        except Exception:                                  # noqa: BLE001
            log.debug("service vector warm-up failed", exc_info=True)

    threading.Thread(target=_warm, name="service-vec-warmup",
                     daemon=True).start()
