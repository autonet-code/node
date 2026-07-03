"""Artifact embedding index — the data plane of the two-plane substrate.

Decision (2026-07-03, ``docs/two_plane_inference.md``): the claim graph is
the *verdict* layer (debated judgments about artifacts), while the full
work-unit payloads live in the blob store and are retrieved by embedding
similarity. This module is that retrieval index.

The index is **derived, daemon-local state** — NOT consensus state:

  - It is fully rebuildable from the blob store (``rebuild()``).
  - It is never gossiped, never anchored, and never participates in node
    ids, coords, equilibration, or epoch close. Determinism of federated
    epoch close is unaffected by anything here.

Persistence is an append-only jsonl file at ``index_path`` (one row per
artifact: ``{"digest", "embedding", "label"}``), loaded eagerly on
construction. If ``index_path`` is None the index is in-memory only.

Digests are sha256 hex strings produced by ``blob_store.py`` — NOT IPFS
CIDs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .usefulness_coords import DEFAULT_DIM, default_usefulness_embedder

logger = logging.getLogger(__name__)


class ArtifactIndex:
    """Daemon-local embedding index over blob-store artifact payloads.

    Maps sha256 digests -> embedding vectors, backed by an append-only
    jsonl file. Search is cosine similarity over an in-memory matrix
    rebuilt lazily from the digest -> vector map.
    """

    def __init__(
        self,
        blob_store: Any,
        embedder: Optional[Callable[[str], Any]] = None,
        index_path: Optional[Any] = None,
        dim: int = DEFAULT_DIM,
    ):
        self.blob_store = blob_store
        self.dim = dim
        self.embedder = embedder if embedder is not None else default_usefulness_embedder(dim=dim)
        self.index_path: Optional[Path] = Path(index_path) if index_path is not None else None

        # digest -> np.array(embedding); insertion order preserved (py3.7+).
        self._vectors: Dict[str, np.ndarray] = {}
        # Row labels aligned with jsonl rows / _vectors, for persistence.
        self._labels: Dict[str, str] = {}

        # Lazily-built stacked matrix for search + the digest order it maps to.
        self._matrix: Optional[np.ndarray] = None
        self._matrix_digests: List[str] = []

        self._load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_artifact(self, payload: Dict[str, Any]) -> str:
        """Store ``payload`` in the blob store, embed its text, and index it.

        Idempotent per digest: re-adding an identical payload neither
        duplicates the jsonl row nor the in-memory entry. Returns the
        sha256 hex digest.
        """
        digest = self.blob_store.add_json(payload)
        if digest is None:
            raise RuntimeError("blob_store.add_json returned no digest")

        if digest in self._vectors:
            return digest

        label = self._embedding_text(payload)
        vec = self._embed(label)
        self._vectors[digest] = vec
        self._labels[digest] = label
        self._matrix = None  # invalidate cached search matrix
        self._append_row(digest, vec, label)
        return digest

    def search(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Top-``k`` ``(digest, cosine)`` by embedding similarity.

        Cosine similarity against the query embedding, sorted by score
        descending with a deterministic tie-break on digest ascending.
        """
        if not self._vectors or k <= 0:
            return []

        query_vec = self._embed(query_text)
        matrix, digests = self._ensure_matrix()

        # Cosine: rows and query are L2-normalized, then dot product.
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0.0:
            sims = np.zeros(len(digests), dtype=np.float64)
        else:
            q_unit = query_vec / q_norm
            row_norms = np.linalg.norm(matrix, axis=1)
            safe = np.where(row_norms == 0.0, 1.0, row_norms)
            unit_rows = matrix / safe[:, None]
            sims = unit_rows @ q_unit
            sims = np.where(row_norms == 0.0, 0.0, sims)

        # Deterministic: sort by (-cosine, digest). Digests are unique.
        ranked = sorted(
            zip(digests, (float(s) for s in sims)),
            key=lambda dc: (-dc[1], dc[0]),
        )
        return ranked[:k]

    def has(self, digest: str) -> bool:
        """True if ``digest`` is present in the in-memory index."""
        return digest in self._vectors

    def get(self, digest: str) -> Optional[np.ndarray]:
        """Return the stored embedding for ``digest``, or None."""
        return self._vectors.get(digest)

    def rebuild(self) -> int:
        """Re-embed every artifact-shaped blob and rewrite the jsonl.

        Iterates ``blob_store.list_hashes()``, loads each JSON payload,
        and re-indexes those that look like artifacts (a ``problem`` or
        ``text`` key). The jsonl is rewritten atomically (temp file then
        replace). Returns the number of artifacts indexed.
        """
        vectors: Dict[str, np.ndarray] = {}
        labels: Dict[str, str] = {}

        for digest in self.blob_store.list_hashes():
            payload = self.blob_store.get_json(digest)
            if not isinstance(payload, dict):
                continue
            if "problem" not in payload and "text" not in payload:
                continue
            label = self._embedding_text(payload)
            vectors[digest] = self._embed(label)
            labels[digest] = label

        self._vectors = vectors
        self._labels = labels
        self._matrix = None
        self._rewrite_all()
        return len(self._vectors)

    def __len__(self) -> int:
        return len(self._vectors)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _embedding_text(payload: Dict[str, Any]) -> str:
        """Derive the text to embed from an artifact payload.

        Work-unit payloads (``problem`` present) embed
        ``f"{problem}\\n\\n{resolution}"``; generic payloads embed the
        ``text`` field.
        """
        if "problem" in payload:
            problem = payload.get("problem") or ""
            resolution = payload.get("resolution") or ""
            return f"{problem}\n\n{resolution}"
        return str(payload.get("text") or "")

    def _embed(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder(text), dtype=np.float64)

    def _ensure_matrix(self) -> Tuple[np.ndarray, List[str]]:
        if self._matrix is None:
            digests = list(self._vectors.keys())
            if digests:
                self._matrix = np.vstack([self._vectors[d] for d in digests])
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float64)
            self._matrix_digests = digests
        return self._matrix, self._matrix_digests

    # ---- persistence -------------------------------------------------- #

    def _load(self) -> None:
        """Eagerly load the jsonl, tolerating missing/corrupt lines."""
        if self.index_path is None or not self.index_path.exists():
            return
        try:
            lines = self.index_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning("ArtifactIndex could not read %s: %s", self.index_path, e)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                digest = row["digest"]
                embedding = np.asarray(row["embedding"], dtype=np.float64)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.debug("ArtifactIndex skipping corrupt index row")
                continue
            self._vectors[digest] = embedding
            self._labels[digest] = row.get("label", "")
        self._matrix = None

    def _row_json(self, digest: str, vec: np.ndarray, label: str) -> str:
        return json.dumps(
            {"digest": digest, "embedding": [float(x) for x in vec], "label": label}
        )

    def _append_row(self, digest: str, vec: np.ndarray, label: str) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(self._row_json(digest, vec, label) + "\n")

    def _rewrite_all(self) -> None:
        """Atomically rewrite the jsonl from current in-memory state."""
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for digest, vec in self._vectors.items():
                fh.write(self._row_json(digest, vec, self._labels.get(digest, "")) + "\n")
        os.replace(tmp_path, self.index_path)
