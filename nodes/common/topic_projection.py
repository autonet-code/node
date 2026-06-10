"""2D topic-space projection for the substrate visualization.

The substrate stores each node's full coordinates as
``[charter_6 | embedding_M]``. The charter axes are visualized via
the constellation hexagon; the embedding tail (``M`` dimensions, up
to 1024) drives parent-child clustering and content similarity.

This module projects the *embedding tail* (or the full coords when
the embedding tail is empty) into 2D so the topic-space
visualization can render nodes by content similarity rather than
alignment.

Approach
--------

PCA via ``numpy.linalg.svd``. Cheap, deterministic, fits the
"sister to the alignment hexagon" design — the user wants relative
positions to be readable without per-run noise.

Stability across epochs is handled at the call site by passing
``previous_xy`` for known node ids: we sign-align each new principal
component to the previous-epoch axis (PCA's eigenvectors are unique
up to a sign flip, which would otherwise mirror the layout between
epochs and break the user's mental map).

UMAP/t-SNE would produce nicer clusters but introduce a heavy
dependency, run-to-run randomness, and an order-N^2 cost we don't
want to pay yet. PCA is the conservative choice; we can swap later
if clusters look too smeared.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple


logger = logging.getLogger(__name__)


def project_topic_2d(
    items: Iterable[Tuple[str, Iterable[float]]],
    *,
    charter_dim: int = 6,
    previous_xy: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Project a list of nodes' full coords to 2D topic space.

    ``items`` is an iterable of ``(node_id, coords)`` pairs; each
    node's coords must be the full ``[charter | embedding]`` vector.
    The first ``charter_dim`` columns are dropped before projection
    so the layout reflects topic similarity, not charter alignment.

    Returns ``{node_id: (x, y)}`` with values in roughly ``[-1, 1]``
    (component-normalized — outliers may exceed slightly).

    When fewer than 3 nodes are present, the result places everything
    at the origin (PCA isn't meaningful with so few points). When the
    embedding tail is zero-width, the charter columns are used
    instead so the call doesn't fail in tests with embedding_dim=0.
    """
    try:
        import numpy as np
    except ImportError:
        logger.debug("numpy unavailable; topic projection skipped")
        return {}

    node_ids: list[str] = []
    rows: list[list[float]] = []
    for node_id, coords in items:
        try:
            row = [float(c) for c in coords]
        except (TypeError, ValueError):
            continue
        if not row:
            continue
        node_ids.append(node_id)
        rows.append(row)

    if len(rows) < 3:
        return {nid: (0.0, 0.0) for nid in node_ids}

    # The world legitimately mixes coord widths: charter-only nodes (6D)
    # coexist with [charter | embedding] nodes (6+M). Zero-pad short rows
    # to the widest — the same convention the adapter uses when it pads
    # charter anchors into the concatenated space. Without this, one
    # mismatched node makes np.asarray raise ("inhomogeneous shape") and
    # the entire visualization endpoint 500s.
    width = max(len(r) for r in rows)
    rows = [r + [0.0] * (width - len(r)) if len(r) < width else r for r in rows]

    arr = np.asarray(rows, dtype=np.float64)
    # Drop the charter dimensions so the projection reflects topic.
    if arr.shape[1] > charter_dim:
        tail = arr[:, charter_dim:]
    else:
        tail = arr  # embedding_dim==0 fallback
    if tail.shape[1] == 0:
        return {nid: (0.0, 0.0) for nid in node_ids}

    # Center.
    centered = tail - tail.mean(axis=0, keepdims=True)
    # SVD-based PCA. Take the top-2 right singular vectors as the axes.
    try:
        # full_matrices=False is much cheaper for tall+thin matrices.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {nid: (0.0, 0.0) for nid in node_ids}
    if vt.shape[0] < 2:
        return {nid: (0.0, 0.0) for nid in node_ids}
    axes = vt[:2]  # shape (2, M)

    # Project each row onto the two axes.
    coords_2d = centered @ axes.T  # shape (N, 2)

    # Stability anchor: PCA eigenvectors are unique up to sign. If we
    # don't fix that, the whole layout flips between epochs. Use the
    # previous-epoch coords (when available) to choose the sign of
    # each axis.
    if previous_xy:
        for axis_index in (0, 1):
            agree = 0.0
            for i, nid in enumerate(node_ids):
                prev = previous_xy.get(nid)
                if prev is None:
                    continue
                agree += prev[axis_index] * coords_2d[i, axis_index]
            if agree < 0:
                # Flip this axis — both the projection and the basis
                # vector — so it lines up with the previous run.
                coords_2d[:, axis_index] = -coords_2d[:, axis_index]
                axes[axis_index] = -axes[axis_index]

    # Component-normalize so x and y both fall in roughly [-1, 1].
    # We divide by max absolute value per axis (robust enough for the
    # visualization; not a statistical scaler).
    scale = np.maximum(np.abs(coords_2d).max(axis=0), 1e-12)
    coords_2d = coords_2d / scale

    out: Dict[str, Tuple[float, float]] = {}
    for i, nid in enumerate(node_ids):
        out[nid] = (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
    return out


def assign_topic_clusters(
    topic_xy: Dict[str, Tuple[float, float]],
    *,
    merge_radius: float = 0.28,
    previous_clusters: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Single-link proximity merge in topic-xy space.

    Each node falls into the existing cluster whose center is within
    ``merge_radius`` (Euclidean, in normalized topic-xy units), or
    seeds a new one. Iteration order is sorted by node_id so the
    result is deterministic across calls with the same input.

    ``previous_clusters`` keeps cluster ids stable across recomputes:
    when a freshly seeded cluster contains nodes that previously
    shared a cluster id, that id is reused instead of minting a new
    one. Falls back to monotonically-increasing ``c0``, ``c1``, ...

    Returns ``{node_id: cluster_id}``.
    """
    if not topic_xy:
        return {}

    sorted_ids = sorted(topic_xy.keys())
    centers: Dict[str, Tuple[float, float]] = {}
    members: Dict[str, list[str]] = {}
    assignments: Dict[str, str] = {}
    used_prev_ids: set[str] = set()
    next_id = 0
    r2 = merge_radius * merge_radius

    def _new_cluster_id(seed_node: str) -> str:
        """Pick a stable id — either the seed's prior cluster (if not
        yet reused) or a fresh ``c{N}``."""
        nonlocal next_id
        if previous_clusters is not None:
            prev = previous_clusters.get(seed_node)
            if prev is not None and prev not in used_prev_ids:
                used_prev_ids.add(prev)
                return prev
        while True:
            cid = f"c{next_id}"
            next_id += 1
            if cid not in used_prev_ids:
                return cid

    for nid in sorted_ids:
        x, y = topic_xy[nid]
        best_cid = None
        best_d2 = float("inf")
        for cid, (cx, cy) in centers.items():
            dx = x - cx
            dy = y - cy
            d2 = dx * dx + dy * dy
            if d2 <= r2 and d2 < best_d2:
                best_cid = cid
                best_d2 = d2
        if best_cid is None:
            best_cid = _new_cluster_id(nid)
            members[best_cid] = []
            centers[best_cid] = (x, y)
        members[best_cid].append(nid)
        # Online mean update.
        m = members[best_cid]
        n = len(m)
        cx, cy = centers[best_cid]
        cx = cx + (x - cx) / n
        cy = cy + (y - cy) / n
        centers[best_cid] = (cx, cy)
        assignments[nid] = best_cid

    return assignments
