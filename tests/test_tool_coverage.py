"""Demonstrated-coverage atlas + density-blend retrieval.

Covers ``CoverageIndex`` (add/density/persistence/cap/ingest filtering)
and the density blend threaded through ``WorldService.infer_artifacts``.

THE KEY BEHAVIOR: retrieval ranks by LOCAL DENSITY, not centroid — a
"broad" tool with genuine attested usage in two distant regions must
out-rank a narrow tool for a query in either region; cold-start
manifests rank purely by claimed text. Design: docs/tool_substrate.md
("Retrieval: density, not centroid").

Uses HashingEmbedder (deterministic, dependency-free).
"""

from __future__ import annotations

import numpy as np

from nodes.common.world_service import WorldService
from nodes.common.world_model_substrate.coverage import CoverageIndex
from nodes.common.world_model_substrate.tool_manifest import build_tool_manifest
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder

EMBED_DIM = 64


def _svc(tmp_path, label="0xCoverageSvc"):
    return WorldService(
        label,
        data_root=tmp_path,
        embedding_dim=EMBED_DIM,
        snapshot_every_n_events=10_000,
        snapshot_every_seconds=10_000.0,
    )


def _manifest(**over):
    kwargs = dict(
        name="summarize_csv",
        description="Summarize a CSV file into per-column statistics.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        author="agent-abc",
        trust_class="pinned",
        code_digest="c" * 64,
        created_ts=1780531200,
    )
    kwargs.update(over)
    return build_tool_manifest(**kwargs)


def _receipt(digest, coords, *, attested=True, caller="caller-1", seq=1):
    """A tool_used event dict as it arrives at submit_events."""
    from nodes.common.world_model_substrate.events import ToolUsed
    return ToolUsed(
        seq=seq,
        author_agent=caller,
        manifest_digest=digest,
        tool_author="agent-abc",
        receipt_digest=f"r{seq:02d}" * 8,
        ok=True,
        problem_coords=list(coords),
        attested=attested,
    ).to_dict()


# --------------------------------------------------------------------------- #
# CoverageIndex unit behavior
# --------------------------------------------------------------------------- #


def test_add_and_density_basic():
    cov = CoverageIndex()
    d = "a" * 64
    # A cloud around a direction.
    v = np.zeros(8)
    v[0] = 1.0
    assert cov.add_point(d, v, attested=True) is True
    # Query aligned with the cloud -> density ~1.
    assert cov.density(v, d, k=5) > 0.99
    # Orthogonal query -> density ~0.
    q = np.zeros(8)
    q[1] = 1.0
    assert abs(cov.density(q, d, k=5)) < 1e-9
    # Unknown digest -> 0.0.
    assert cov.density(v, "z" * 64, k=5) == 0.0


def test_density_is_topk_mean_not_centroid():
    """Density averages the top-k NEAREST points, so a query near one
    dense sub-cluster scores high even if the cloud's centroid is far."""
    cov = CoverageIndex()
    d = "a" * 64
    # Two distant sub-clusters: +x region and +y region.
    for _ in range(5):
        cov.add_point(d, [1.0, 0.0, 0.0], attested=True)
    for _ in range(5):
        cov.add_point(d, [0.0, 1.0, 0.0], attested=True)
    # Centroid sits at ~(0.5, 0.5); a pure +x query is far from centroid
    # but sits right on top of a dense sub-cluster -> high density.
    dens = cov.density([1.0, 0.0, 0.0], d, k=5)
    assert dens > 0.99


def test_unattested_and_empty_rejected():
    cov = CoverageIndex()
    d = "a" * 64
    assert cov.add_point(d, [1.0, 0.0], attested=False) is False
    assert cov.add_point(d, [], attested=True) is False
    assert cov.add_point("", [1.0, 0.0], attested=True) is False
    assert len(cov) == 0


def test_ingest_event_filters():
    cov = CoverageIndex()
    d = "a" * 64
    coords = [1.0, 0.0, 0.0]
    # Attested + coords -> stored.
    assert cov.ingest_event(_receipt(d, coords, attested=True)) is True
    # Unattested -> rejected (mechanical receipt, no judgment).
    assert cov.ingest_event(_receipt(d, coords, attested=False)) is False
    # Missing coords -> rejected.
    assert cov.ingest_event(_receipt(d, [], attested=True)) is False
    # Wrong kind -> rejected.
    assert cov.ingest_event({"kind": "observation_added", "attested": True,
                             "problem_coords": coords}) is False
    assert cov.n_points(d) == 1


def test_per_digest_cap_fifo():
    cov = CoverageIndex(max_points_per_digest=4)
    d = "a" * 64
    for i in range(10):
        cov.add_point(d, [float(i), 1.0], attested=True)
    assert cov.n_points(d) == 4  # capped
    assert len(cov) == 4


def test_persistence_reload(tmp_path):
    path = tmp_path / "coverage_index.jsonl"
    cov = CoverageIndex(index_path=path)
    d1, d2 = "a" * 64, "b" * 64
    cov.add_point(d1, [1.0, 0.0, 0.0], attested=True, ts=100)
    cov.add_point(d1, [0.0, 1.0, 0.0], attested=True, ts=101)
    cov.add_point(d2, [0.0, 0.0, 1.0], attested=True, ts=102)

    # Reload from disk: same clouds, same density readings. d1 has two
    # points (one aligned with the query, one orthogonal); its single
    # aligned point is the nearest, so density(k=1) recovers ~1.0 while
    # the k=5 mean is pulled down by the orthogonal point.
    cov2 = CoverageIndex(index_path=path)
    assert cov2.n_points(d1) == 2
    assert cov2.n_points(d2) == 1
    assert cov2.density([1.0, 0.0, 0.0], d1, k=1) > 0.99
    assert cov2.density([0.0, 0.0, 1.0], d2, k=5) > 0.99


def test_coverage_gaps():
    cov = CoverageIndex()
    d = "a" * 64
    cov.add_point(d, [1.0, 0.0, 0.0], attested=True)
    # Query 1 is covered (max density high); query 2 is a gap (~0).
    gaps = cov.coverage_gaps([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], k=5)
    assert gaps[0] > 0.99
    assert abs(gaps[1]) < 1e-9


# --------------------------------------------------------------------------- #
# THE KEY BEHAVIOR: density beats centroid in ranking
# --------------------------------------------------------------------------- #


def test_broad_tool_beats_narrow_via_density():
    """A broad tool with genuine attested usage in two distant regions
    must rank ABOVE a narrow tool (usage only in region A) for a region-B
    query — because we rank by local density, not centroid distance. If
    we ranked by centroid, the broad tool's centroid would sit between
    the two regions and lose to the narrow tool in each region."""
    cov = CoverageIndex()
    broad, narrow = "b" * 64, "n" * 64

    # Region A and region B: two distant unit directions.
    region_a = [1.0, 0.0, 0.0, 0.0]
    region_b = [0.0, 0.0, 1.0, 0.0]

    # Broad tool: real usage in BOTH regions.
    for _ in range(4):
        cov.add_point(broad, region_a, attested=True)
        cov.add_point(broad, region_b, attested=True)
    # Narrow tool: usage only in region A.
    for _ in range(8):
        cov.add_point(narrow, region_a, attested=True)

    # Query in region B: broad tool has served here, narrow has not.
    # broad has 4 region-B points; top-k=5 mean = [1,1,1,1,0]/5 = 0.8.
    # narrow has zero region-B coverage -> ~0. The broad tool wins.
    dens_broad = cov.density(region_b, broad, k=5)
    dens_narrow = cov.density(region_b, narrow, k=5)
    assert dens_broad > dens_narrow
    assert dens_broad > 0.5
    assert dens_narrow < 1e-9
    # With k=4 (exactly the region-B sub-cluster size) density is a
    # clean 1.0 — the broad tool genuinely covers region B.
    assert cov.density(region_b, broad, k=4) > 0.99

    # Crux: density credits the broad tool for ACTUALLY serving region
    # B. A centroid ranker would blend the broad tool's two regions into
    # a midpoint and lose this signal; density preserves it.
    assert dens_broad == max(dens_broad, dens_narrow)


# --------------------------------------------------------------------------- #
# End-to-end through WorldService: covered manifest out-ranks uncovered
# --------------------------------------------------------------------------- #


def test_cold_start_ranks_by_claimed_text(tmp_path):
    """With no coverage yet, manifests rank purely by claimed cosine —
    byte-for-byte the pre-coverage behavior."""
    svc = _svc(tmp_path)
    embedder = HashingEmbedder(dim=EMBED_DIM)
    d_match = svc.submit_tool_manifest(
        _manifest(name="parse_dates",
                  description="parse a date string into a datetime object"),
        agent_id="a", embedder=embedder,
    )["manifest_digest"]
    d_other = svc.submit_tool_manifest(
        _manifest(name="resize_image",
                  description="resize an image to a thumbnail size"),
        agent_id="a", embedder=embedder,
    )["manifest_digest"]

    result = svc.infer_artifacts(
        "parse a date string into a datetime object", k=5
    )
    order = [a["digest"] for a in result["artifacts"]]
    assert order.index(d_match) < order.index(d_other)


def test_coverage_lifts_manifest_over_similar_uncovered(tmp_path):
    """End-to-end: two manifests with SIMILAR descriptions. One gets
    attested tool_used receipts whose problem_coords sit near a query;
    the other gets none. The covered manifest must out-rank the
    uncovered one for that query."""
    svc = _svc(tmp_path)
    embedder = HashingEmbedder(dim=EMBED_DIM)

    # Two tools with near-identical claimed text so claimed cosine alone
    # would rank them ~equally; coverage is the tiebreaker.
    d_covered = svc.submit_tool_manifest(
        _manifest(name="tool_alpha",
                  description="general data transformation utility"),
        agent_id="a", embedder=embedder,
    )["manifest_digest"]
    d_bare = svc.submit_tool_manifest(
        _manifest(name="tool_beta",
                  description="general data transformation helper"),
        agent_id="a", embedder=embedder,
    )["manifest_digest"]

    # The work-context the caller actually solved with tool_alpha.
    problem_text = "convert a messy CSV export into clean tidy rows"
    problem_coords = svc.coords_for_text(problem_text, embedder=embedder)

    # Attested receipts land tool_alpha's coverage right on the query.
    svc.submit_events([
        _receipt(d_covered, problem_coords, seq=i)
        for i in range(1, 6)
    ])

    # Sanity: the atlas holds tool_alpha's coverage, not tool_beta's.
    assert svc._coverage_index.has(d_covered)
    assert not svc._coverage_index.has(d_bare)

    result = svc.infer_artifacts(problem_text, k=5)
    order = [a["digest"] for a in result["artifacts"]]
    assert d_covered in order and d_bare in order
    assert order.index(d_covered) < order.index(d_bare), (
        f"covered manifest should out-rank uncovered, got {order}"
    )


def test_unattested_receipts_do_not_move_ranking(tmp_path):
    """Mechanical (unattested) receipts carry no judgment: they must NOT
    enter the atlas and must NOT change retrieval."""
    svc = _svc(tmp_path)
    embedder = HashingEmbedder(dim=EMBED_DIM)
    d = svc.submit_tool_manifest(
        _manifest(name="tool_x", description="do a thing with data"),
        agent_id="a", embedder=embedder,
    )["manifest_digest"]

    problem_coords = svc.coords_for_text("some work context", embedder=embedder)
    svc.submit_events([
        _receipt(d, problem_coords, attested=False, seq=i) for i in range(1, 6)
    ])
    assert not svc._coverage_index.has(d)


def test_remote_origin_receipts_still_feed_coverage(tmp_path):
    """Coverage is daemon-local derived state, so remote-origin gossip
    ingest feeds it too (cannot affect epoch close)."""
    svc = _svc(tmp_path)
    embedder = HashingEmbedder(dim=EMBED_DIM)
    d = svc.submit_tool_manifest(
        _manifest(name="tool_y", description="another data tool"),
        agent_id="a", embedder=embedder,
    )["manifest_digest"]

    problem_coords = svc.coords_for_text("remote work context", embedder=embedder)
    svc.submit_events(
        [_receipt(d, problem_coords, seq=1)],
        origin="remote",
    )
    assert svc._coverage_index.has(d)
