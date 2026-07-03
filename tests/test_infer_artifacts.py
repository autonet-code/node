"""Artifact-retrieval inference (two-plane data plane).

Covers ``_infer_artifacts`` and ``infer_with_world_model`` dispatch:
retrieval by embedding cosine, graph-standing re-rank (a well-supported
CON claim demotes an artifact), payload passthrough, missing-blob
tolerance, determinism, mode dispatch, and back-compat of the
alignment/general modes. Also covers ``render_context`` (arm-B-style
prompt rendering, docs/ledger_pricing.md Change 3) and the
``WorldService.infer_artifacts(render=True)`` wiring.

Uses HashingEmbedder (deterministic, dependency-free) so no torch and
stable rankings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nodes.common.blob_store import BlobStore
from nodes.common.world_model_substrate.artifact_index import ArtifactIndex
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder
from nodes.common.world_model_substrate.infer import (
    _infer_artifacts,
    infer_with_world_model,
    render_context,
)
from nodes.common.world_model_substrate.adapter import build_charter_world

from world_model.generalized.world import World
from world_model.generalized.tendency import GeneralizedTendency
from world_model.models.tree import Node, Position


DIM = 64


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(data_dir=str(tmp_path / "blobs"))


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


def _work_unit(problem: str, resolution: str) -> dict:
    return {"problem": problem, "resolution": resolution, "outcome": {"ok": True}}


def _single_tendency_world() -> World:
    """A minimal world with one tendency to hang claim nodes off."""
    world = World()
    world.add_tendency(
        GeneralizedTendency(
            id="T",
            thesis="root thesis",
            anchor=(1.0, 0.0),
            polarity_axis=(1.0, 0.0),
            budget=1.0,
            bandwidth=1.0,
        )
    )
    return world


def _add_claim(
    world: World,
    tendency_id: str,
    node_id: str,
    position: Position,
    posts: int,
    digest: str,
    content: str,
) -> Node:
    """Attach a leaf claim node under a tendency root, give it ``posts``
    unit-weight stakes (so net_score == posts for a leaf), and set its
    artifact_digest attribute the way the ingestion agent would.
    """
    tendency = world.tendencies[tendency_id]
    root = tendency.tree.root_node
    node = Node(id=node_id, tree_id=tendency.tree.id, content=content)
    tendency.tree.add_node(root.id, node, position, tendency_id=tendency_id)
    for _ in range(posts):
        node.add_post(agent_id="stub")
    node.artifact_digest = digest
    return node


# --------------------------------------------------------------------------- #
# (a) ranking follows cosine when standings are equal
# --------------------------------------------------------------------------- #


def test_ranking_follows_cosine_when_standings_equal(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    d_hit = idx.add_artifact(
        _work_unit(
            "parse a date string into a datetime object",
            "call datetime.strptime with the matching format code",
        )
    )
    d_miss = idx.add_artifact(
        _work_unit(
            "resize an image to a thumbnail",
            "scale the pixel dimensions down",
        )
    )

    # No claim nodes -> both standings are 0 -> pure cosine order.
    world = _single_tendency_world()

    result = _infer_artifacts(
        {"text": "parse a date string into a datetime object"},
        world,
        idx,
        blob_store,
        k=5,
    )
    arts = result["artifacts"]
    order = [a["digest"] for a in arts]
    assert order[0] == d_hit
    assert d_miss in order
    assert order.index(d_hit) < order.index(d_miss)
    assert all(a["standing"] == 0.0 for a in arts)


# --------------------------------------------------------------------------- #
# (b) a well-supported CON claim demotes an artifact below a lower-cosine rival
# --------------------------------------------------------------------------- #


def test_con_claim_flips_ranking(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    query = "parse a date string into a datetime object"

    d_high = idx.add_artifact(
        _work_unit(query, "call datetime.strptime with the matching format code")
    )
    # A rival that is semantically close but slightly lower cosine to the
    # query. Sharing most of the query words keeps it near the top.
    d_low = idx.add_artifact(
        _work_unit(
            "parse a date string into a struct",
            "call an alternative date parser routine",
        )
    )

    world = _single_tendency_world()

    # Baseline: no claims -> both standings 0 -> cosine order must put
    # d_high above d_low (otherwise the flip test is vacuous).
    base = _infer_artifacts({"text": query}, world, idx, blob_store, k=5)
    base_order = [a["digest"] for a in base["artifacts"]]
    assert base_order.index(d_high) < base_order.index(d_low), (
        "precondition: d_high must out-cosine d_low before re-rank"
    )
    base_by_digest = {a["digest"]: a for a in base["artifacts"]}
    assert base_by_digest[d_high]["cosine"] > base_by_digest[d_low]["cosine"]

    # Now post a strongly-supported CON claim against d_high. A CON node
    # with positive net_score drives standing negative, and
    # final = cosine * (1 + tanh(standing)) drops below the rival.
    _add_claim(world, "T", "n_con_high", Position.CON, posts=5,
               digest=d_high, content="this artifact is wrong")

    reranked = _infer_artifacts({"text": query}, world, idx, blob_store, k=5)
    r_order = [a["digest"] for a in reranked["artifacts"]]
    r_by_digest = {a["digest"]: a for a in reranked["artifacts"]}

    # Standing recorded and negative for d_high.
    assert r_by_digest[d_high]["standing"] == -5.0
    assert r_by_digest[d_high]["claims"] == [
        {"content": "this artifact is wrong", "position": "con", "net_score": 5.0}
    ]
    # The ordering actually flipped: the lower-cosine rival now ranks above.
    assert r_order.index(d_low) < r_order.index(d_high), (
        f"expected d_low above d_high after CON re-rank, got {r_order}"
    )
    # And final for d_high indeed fell below d_low's final.
    assert r_by_digest[d_high]["final"] < r_by_digest[d_low]["final"]


# --------------------------------------------------------------------------- #
# (c) payloads from blob store; missing blob -> payload None but digest present
# --------------------------------------------------------------------------- #


def test_payload_passthrough_and_missing_blob(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    payload = _work_unit("compute a checksum", "hash the bytes with sha256")
    d_real = idx.add_artifact(payload)

    world = _single_tendency_world()

    result = _infer_artifacts(
        {"text": "compute a checksum"}, world, idx, blob_store, k=5
    )
    real = next(a for a in result["artifacts"] if a["digest"] == d_real)
    assert real["payload"] == payload

    # A claim referencing a digest with no blob (never added to the store).
    missing_digest = "f" * 64
    _add_claim(world, "T", "n_missing", Position.PRO, posts=1,
               digest=missing_digest, content="claim about a ghost artifact")
    # Force the missing digest into candidates by embedding a matching
    # payload's text but deleting the blob is fiddly; instead assert the
    # helper handles a missing blob directly: search won't surface it
    # (not indexed), so verify blob_store returns None for it and that a
    # digest with no blob yields payload None when it IS a candidate.
    # Simulate by adding it as an index-only entry pointing at no blob:
    idx._vectors[missing_digest] = idx._embed("claim about a ghost artifact")
    idx._matrix = None
    result2 = _infer_artifacts(
        {"text": "claim about a ghost artifact"}, world, idx, blob_store, k=10
    )
    ghost = next(a for a in result2["artifacts"] if a["digest"] == missing_digest)
    assert ghost["payload"] is None
    assert ghost["digest"] == missing_digest


# --------------------------------------------------------------------------- #
# (d) deterministic output across two calls
# --------------------------------------------------------------------------- #


def test_deterministic_across_calls(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    for i in range(6):
        idx.add_artifact(_work_unit(f"problem {i}", f"resolution {i}"))

    world = _single_tendency_world()
    q = {"text": "problem 3"}
    r1 = _infer_artifacts(q, world, idx, blob_store, k=5)
    r2 = _infer_artifacts(q, world, idx, blob_store, k=5)

    assert [a["digest"] for a in r1["artifacts"]] == [a["digest"] for a in r2["artifacts"]]
    assert [a["final"] for a in r1["artifacts"]] == [a["final"] for a in r2["artifacts"]]


# --------------------------------------------------------------------------- #
# (e) infer_with_world_model dispatches to artifacts mode
# --------------------------------------------------------------------------- #


def test_dispatch_to_artifacts_mode(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    idx.add_artifact(_work_unit("build a widget", "assemble the parts"))
    world = _single_tendency_world()

    result = infer_with_world_model(
        {"text": "build a widget"},
        world=world,
        artifact_index=idx,
        blob_store=blob_store,
    )
    assert result["mode"] == "artifacts"
    assert result["artifacts"]
    assert result["predictions"] == []
    assert result["probabilities"] == []


def test_explicit_artifacts_mode_requires_deps(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    world = _single_tendency_world()
    # Missing blob_store -> clear error.
    with pytest.raises(ValueError):
        infer_with_world_model(
            {"text": "x"}, mode="artifacts", world=world, artifact_index=idx
        )
    # Missing world -> clear error.
    with pytest.raises(ValueError):
        infer_with_world_model(
            {"text": "x"}, mode="artifacts", artifact_index=idx, blob_store=blob_store
        )


def test_query_key_variants_and_missing(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    idx.add_artifact(_work_unit("alpha topic", "beta resolution"))
    world = _single_tendency_world()

    for key in ("text", "query", "question"):
        r = _infer_artifacts({key: "alpha topic"}, world, idx, blob_store, k=1)
        assert r["mode"] == "artifacts"

    with pytest.raises(ValueError):
        _infer_artifacts({"nope": "x"}, world, idx, blob_store, k=1)


# --------------------------------------------------------------------------- #
# (f) alignment / general modes still work the old way (no new kwargs)
# --------------------------------------------------------------------------- #


def test_alignment_mode_backcompat_old_call():
    result = infer_with_world_model(
        {"turn": {"intelligence_impact": 0.7, "evolution_impact": 0.3}}
    )
    assert result["mode"] == "alignment"
    assert "aligned_principle" in result
    assert isinstance(result["predictions"], list) and result["predictions"]


def test_general_mode_backcompat_old_call():
    result = infer_with_world_model({"coords": [0.0, 0.0, 0.9, 0.3]})
    assert result["mode"] == "general"
    assert "structure" in result
    assert result["predictions"] == []


# --------------------------------------------------------------------------- #
# (g) passing world= to general mode uses that world
# --------------------------------------------------------------------------- #


def test_general_mode_uses_passed_world():
    # A fresh charter world (default path) yields a baseline root score.
    baseline = infer_with_world_model({"coords": [0.0, 0.0, 0.9, 0.3]}, mode="general")
    base_score = baseline["root_scores"]["promotion_of_intelligence"]

    # A hand-seeded world: post extra unit stakes on one tendency's root.
    world = build_charter_world()
    root = world.tendencies["promotion_of_intelligence"].tree.root_node
    for _ in range(3):
        root.add_post(agent_id="stub")

    result = infer_with_world_model(
        {"coords": [0.0, 0.0, 0.9, 0.3]}, mode="general", world=world
    )
    assert result["mode"] == "general"
    # The passed world was used, not a fresh charter world: its extra
    # root posts lift promotion_of_intelligence above the baseline.
    assert result["root_scores"]["promotion_of_intelligence"] > base_score
    assert result["root_scores"]["promotion_of_intelligence"] == pytest.approx(
        base_score + 3.0
    )


# --------------------------------------------------------------------------- #
# (h) standing works WITHOUT equilibration — ledger-mode contract
#     (docs/ledger_pricing.md Change 3)
# --------------------------------------------------------------------------- #


def test_standing_works_without_equilibration(blob_store, embedder):
    """net_score is a pure tree recursion over posts + signed children —
    it requires no equilibrate() rounds. This test builds a world, sprouts
    claim nodes with artifact_digest and posts, and NEVER calls
    equilibrate(). It pins that _infer_artifacts standings are nonzero and
    that ranking still responds to a CON node with posts, as a contract for
    ledger-mode worlds (Change 1: replay without equilibrate rounds).
    """
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    query = "parse a date string into a datetime object"

    d_high = idx.add_artifact(
        _work_unit(query, "call datetime.strptime with the matching format code")
    )
    d_low = idx.add_artifact(
        _work_unit(
            "parse a date string into a struct",
            "call an alternative date parser routine",
        )
    )

    # A fresh world — no equilibrate() call anywhere in this test.
    world = _single_tendency_world()

    # PRO support for d_high: nonzero standing without equilibration.
    _add_claim(world, "T", "n_pro_high", Position.PRO, posts=3,
               digest=d_high, content="this artifact is correct")

    result = _infer_artifacts({"text": query}, world, idx, blob_store, k=5)
    by_digest = {a["digest"]: a for a in result["artifacts"]}
    assert by_digest[d_high]["standing"] == 3.0
    assert by_digest[d_high]["standing"] != 0.0

    # Now add a strongly-supported CON claim on top — ranking must still
    # respond (standing swings negative, final drops) purely from net_score
    # tree recursion, no equilibrate() involved.
    _add_claim(world, "T", "n_con_high", Position.CON, posts=10,
               digest=d_high, content="actually this artifact is wrong")

    reranked = _infer_artifacts({"text": query}, world, idx, blob_store, k=5)
    r_by_digest = {a["digest"]: a for a in reranked["artifacts"]}
    r_order = [a["digest"] for a in reranked["artifacts"]]

    # 3 (PRO) - 10 (CON) = -7.
    assert r_by_digest[d_high]["standing"] == -7.0
    assert r_order.index(d_low) < r_order.index(d_high), (
        f"expected CON-flipped ranking without equilibration, got {r_order}"
    )


# --------------------------------------------------------------------------- #
# render_context: arm-B-style prompt rendering (docs/ledger_pricing.md
# Change 3, docs/phase8_results.md C-B = -0.28)
# --------------------------------------------------------------------------- #


def _result_with_artifacts(payloads):
    """Build a minimal _infer_artifacts-shaped result dict from a list of
    (digest, payload_or_None) pairs, already in standing-ranked order.
    """
    artifacts = []
    for i, (digest, payload) in enumerate(payloads):
        artifacts.append({
            "digest": digest,
            "cosine": 1.0 - i * 0.01,
            "standing": 0.0,
            "final": 1.0 - i * 0.01,
            "payload": payload,
            "claims": [
                {"content": "some claim text", "position": "pro", "net_score": 5.0}
            ] if payload is not None else [],
        })
    return {"mode": "artifacts", "artifacts": artifacts, "predictions": [], "probabilities": []}


def test_render_context_respects_char_budget():
    payloads = [
        (f"digest{i}", {"problem": "P" * 500, "resolution": "R" * 500})
        for i in range(5)
    ]
    result = _result_with_artifacts(payloads)
    ctx = render_context(result, char_budget=1000)
    assert len(ctx) <= 1000


def test_render_context_excludes_claims_digests_and_standing():
    result = _result_with_artifacts([
        ("deadbeef" * 8, {"problem": "parse a date", "resolution": "use strptime"}),
    ])
    ctx = render_context(result)
    assert "deadbeef" not in ctx
    assert "some claim text" not in ctx
    assert "standing" not in ctx.lower()
    assert "net_score" not in ctx
    assert "cosine" not in ctx
    assert "[PRO" not in ctx and "[CON" not in ctx
    # The payload content IS present.
    assert "parse a date" in ctx
    assert "use strptime" in ctx


def test_render_context_skips_none_payloads():
    result = _result_with_artifacts([
        ("d1", {"problem": "problem one", "resolution": "resolution one"}),
        ("d2", None),
        ("d3", {"problem": "problem three", "resolution": "resolution three"}),
    ])
    ctx = render_context(result)
    assert "problem one" in ctx
    assert "problem three" in ctx
    # None payload contributed no block at all.
    assert ctx.count("--- artifact [") == 2


def test_render_context_all_none_payloads_returns_empty():
    result = _result_with_artifacts([("d1", None), ("d2", None)])
    assert render_context(result) == ""


def test_render_context_deterministic():
    payloads = [
        (f"digest{i}", {"problem": f"problem {i}", "resolution": f"resolution {i}"})
        for i in range(4)
    ]
    result = _result_with_artifacts(payloads)
    ctx1 = render_context(result)
    ctx2 = render_context(result)
    assert ctx1 == ctx2


def test_render_context_preserves_standing_ranked_order():
    # Artifacts arrive pre-ranked by standing (as _infer_artifacts sorts
    # them); render_context must not re-sort.
    payloads = [
        ("d_first", {"problem": "first problem alpha", "resolution": "first resolution"}),
        ("d_second", {"problem": "second problem beta", "resolution": "second resolution"}),
    ]
    result = _result_with_artifacts(payloads)
    ctx = render_context(result)
    assert ctx.index("first problem alpha") < ctx.index("second problem beta")


def test_render_context_proportional_truncation_across_artifacts():
    # Two artifacts sharing a fixed budget: each should get roughly half,
    # so a very long payload gets truncated while a short one is untouched.
    long_payload = {"problem": "P" * 5000, "resolution": "R" * 5000}
    short_payload = {"problem": "short problem", "resolution": "short resolution"}
    result = _result_with_artifacts([("d_long", long_payload), ("d_short", short_payload)])
    ctx = render_context(result, char_budget=1000)
    assert len(ctx) <= 1000
    # The short payload's full text survives; the long one is cut down.
    assert "short problem" in ctx
    assert "short resolution" in ctx
    assert "P" * 5000 not in ctx


# --------------------------------------------------------------------------- #
# WorldService.infer_artifacts(render=True) — integration
# --------------------------------------------------------------------------- #


def test_world_service_infer_artifacts_render_true(tmp_path: Path):
    from nodes.common.world_service import WorldService

    service = WorldService("test-rpb", data_root=tmp_path)
    service.submit_work_units([
        (
            "parse a date string into a datetime object",
            "call datetime.strptime with the matching format code",
            {"ok": True},
        ),
        (
            "resize an image to a thumbnail",
            "scale the pixel dimensions down",
            {"ok": True},
        ),
    ])

    out = service.infer_artifacts(
        "parse a date string into a datetime object", k=5, render=True
    )
    assert set(out.keys()) == {"result", "context"}
    assert out["result"]["mode"] == "artifacts"
    assert isinstance(out["context"], str)
    assert "strptime" in out["context"]
    # Verdict/digest text must not leak into the rendered context.
    for art in out["result"]["artifacts"]:
        assert art["digest"] not in out["context"]

    # render=False (default) stays back-compat: bare result dict.
    out_default = service.infer_artifacts("parse a date string into a datetime object", k=5)
    assert out_default["mode"] == "artifacts"
    assert "context" not in out_default
