"""Demonstrated-coverage atlas — the density plane of the tool substrate.

Design: ``docs/tool_substrate.md`` ("Retrieval: density, not centroid" +
"Attestation"). Cognitive attestations (``tool_used`` with ``attested``
truthy) carry ``problem_coords``: an embedding of the work-context the
caller was actually solving. Accumulated per manifest, these points form
a *demonstrated-coverage cloud* — collectively an atlas of what the
network can do and where.

Retrieval ranks a tool by the LOCAL DENSITY of its cloud near the query
(mean cosine of the top-k nearest stored points), NOT by distance to the
cloud's centroid. Centroid ranking subsidizes narrow tools and invites
fragmentation spam; density lets genuine breadth compete everywhere it
has actually served.

This module is **derived, daemon-local state** — same doctrine as
``ArtifactIndex``:

  - Fully rebuildable from the canonical event log / blob store.
  - NEVER gossiped, NEVER anchored, NEVER part of node ids, coords,
    equilibration, or epoch close. Determinism of federated epoch close
    is unaffected by anything here. Retrieval formulas are advisory and
    daemon-local (``docs/two_plane_inference.md`` constraints) — iterate
    freely.

Only COGNITIVE ATTESTATIONS enter the atlas. Mechanical receipts
(``attested`` falsey) are per-call exhaust carrying no judgment of
usefulness, so they are dropped: an exit code must never move retrieval.

Persistence is an append-only jsonl file at ``index_path`` (one row per
point: ``{"digest", "coords", "ts"}``), loaded eagerly on construction.
If ``index_path`` is None the atlas is in-memory only.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# A generous per-digest cap so a hot tool can't grow the atlas without
# bound. FIFO: the oldest attested points age out once the cap is hit.
# 2048 attested WORK-ITEM attestations for a single tool is already a
# very dense local cloud; density(top-k) saturates long before this.
DEFAULT_MAX_POINTS_PER_DIGEST = 2048

# tool_used events whose kind matches AND carry a truthy ``attested``
# flag AND a non-empty ``problem_coords`` list are the only ingest-able
# receipts.
_TOOL_USED_KIND = "tool_used"


def _unit(vec: np.ndarray) -> np.ndarray:
    """L2-normalize; a zero vector maps to itself (stays zero)."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class CoverageIndex:
    """Daemon-local demonstrated-coverage atlas: manifest digest ->
    cloud of unit-normalized attested ``problem_coords`` vectors.

    Density retrieval (``density``) ranks by mean cosine of the top-k
    nearest stored points, deliberately NOT by centroid distance. See
    module docstring / ``docs/tool_substrate.md``.
    """

    def __init__(
        self,
        index_path: Optional[Any] = None,
        max_points_per_digest: int = DEFAULT_MAX_POINTS_PER_DIGEST,
    ):
        self.index_path: Optional[Path] = (
            Path(index_path) if index_path is not None else None
        )
        self.max_points_per_digest = int(max_points_per_digest)
        # digest -> FIFO deque of unit-normalized np vectors.
        self._points: Dict[str, Deque[np.ndarray]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_point(
        self,
        manifest_digest: str,
        coords: Any,
        attested: bool,
        ts: int = 0,
    ) -> bool:
        """Add one attested coverage point for ``manifest_digest``.

        ONLY attested points enter the atlas (mechanical receipts carry
        no judgment). The vector is unit-normalized on the way in so
        ``density`` is a pure cosine. Per-digest FIFO cap keeps a hot
        tool's cloud bounded.

        Returns True if the point was stored, False if it was rejected
        (unattested, empty coords, or missing digest).
        """
        if not attested or not manifest_digest:
            return False
        vec = np.asarray(coords, dtype=np.float64).ravel()
        if vec.size == 0:
            return False
        vec = _unit(vec)
        bucket = self._points.get(manifest_digest)
        if bucket is None:
            bucket = self._points[manifest_digest] = deque(
                maxlen=self.max_points_per_digest
            )
        bucket.append(vec)
        self._append_row(manifest_digest, vec, ts)
        return True

    def density(self, query_vec: Any, digest: str, k: int = 5) -> float:
        """Local density of ``digest``'s cloud near ``query_vec``.

        Mean cosine of the top-``k`` nearest stored points for that
        digest. 0.0 if the digest has no points. This is LOCAL density,
        deliberately not centroid distance: breadth competes everywhere
        it has actually served (``docs/tool_substrate.md``).
        """
        bucket = self._points.get(digest)
        if not bucket or k <= 0:
            return 0.0
        q = np.asarray(query_vec, dtype=np.float64).ravel()
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return 0.0
        q = q / qn
        sims: List[float] = []
        for vec in bucket:
            n = min(vec.size, q.size)
            if n == 0:
                continue
            # Stored vectors are already unit; the query is unit here.
            # Cosine over the overlapping dims (handles head/tail-length
            # differences gracefully, mirroring usefulness_coords._cosine).
            sims.append(float(np.dot(vec[:n], q[:n])))
        if not sims:
            return 0.0
        sims.sort(reverse=True)
        top = sims[: min(k, len(sims))]
        return sum(top) / len(top)

    def coverage_gaps(self, query_vecs: Any, k: int = 5) -> List[float]:
        """Per-query MAX density across ALL known digests.

        The atlas-gap surface: for each query vector, how well-covered
        is that region by *any* tool. Low values are capability gaps —
        the future input to capability-gap mint multipliers (network
        pays more for uncovered regions; designed, not yet built).
        Simple by intent.
        """
        out: List[float] = []
        digests = list(self._points.keys())
        for qv in query_vecs:
            best = 0.0
            for digest in digests:
                d = self.density(qv, digest, k=k)
                if d > best:
                    best = d
            out.append(best)
        return out

    def ingest_event(self, event: Dict[str, Any]) -> bool:
        """Feed a ``tool_used`` event dict; store it when it qualifies.

        Qualifies iff: kind == "tool_used", ``attested`` truthy, and
        ``problem_coords`` non-empty. Returns True when a point was
        stored. Safe to call on every event (local emission AND remote
        gossip ingest) — this is daemon-local derived state, so it can
        never affect epoch close.
        """
        if not isinstance(event, dict):
            return False
        if event.get("kind") != _TOOL_USED_KIND:
            return False
        if not event.get("attested"):
            return False
        coords = event.get("problem_coords")
        if not coords:
            return False
        digest = event.get("manifest_digest") or ""
        ts = event.get("ts") or 0
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            ts = 0
        return self.add_point(digest, coords, attested=True, ts=ts)

    def has(self, digest: str) -> bool:
        """True if the atlas holds any coverage point for ``digest``."""
        bucket = self._points.get(digest)
        return bool(bucket)

    def n_points(self, digest: str) -> int:
        """Number of stored coverage points for ``digest``."""
        bucket = self._points.get(digest)
        return len(bucket) if bucket else 0

    def __len__(self) -> int:
        return sum(len(b) for b in self._points.values())

    # ------------------------------------------------------------------ #
    # Persistence (mirrors ArtifactIndex conventions)
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Eagerly load the jsonl, tolerating missing/corrupt lines."""
        if self.index_path is None or not self.index_path.exists():
            return
        try:
            lines = self.index_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning("CoverageIndex could not read %s: %s", self.index_path, e)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                digest = row["digest"]
                coords = np.asarray(row["coords"], dtype=np.float64).ravel()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.debug("CoverageIndex skipping corrupt row")
                continue
            if not digest or coords.size == 0:
                continue
            bucket = self._points.get(digest)
            if bucket is None:
                bucket = self._points[digest] = deque(
                    maxlen=self.max_points_per_digest
                )
            # Rows on disk are already unit-normalized (written by
            # add_point); re-normalize defensively in case of hand edits.
            bucket.append(_unit(coords))

    def _row_json(self, digest: str, vec: np.ndarray, ts: int) -> str:
        return json.dumps(
            {"digest": digest, "coords": [float(x) for x in vec], "ts": int(ts)}
        )

    def _append_row(self, digest: str, vec: np.ndarray, ts: int) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(self._row_json(digest, vec, ts) + "\n")
