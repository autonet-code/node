"""Economy-graph read surfaces (visualization backend).

Covers the three additive surfaces added for the economy/constellation
visualizations:

  * ``WorldService.read_economy_graph`` — registrations + declared-dep
    DAG + vetting + recent per-digest mint.
  * ``read_substrate_distribution``'s per-kind split (``root_kinds`` /
    ``kind_totals``): tool mint at manifest anchors, a vetting slice
    while the royalty window is open, everything else claim. Per-root
    conservation against ``root_mint_recent``.
  * ``kind`` on ``list_nodes_for_visualization`` entries (tool anchors
    vs plain claims).

Drives the REAL local close path: open_epoch → submit_tool_manifest
(registration sprout, author_post standing) → attested receipts + vets
via submit_events → close_epoch → read.
"""

from __future__ import annotations

import pytest

from nodes.common.federated_reconcile import (
    VET_ROYALTY_EPOCHS,
    VET_ROYALTY_SHARE,
)
from nodes.common.world_model_substrate.tool_manifest import build_tool_manifest
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder
from nodes.common.world_service import WorldService

EMBED_DIM = 64
AUTHOR = "author-1"
CALLER = "caller-1"


def _svc(tmp_path):
    return WorldService(
        "0xEconomyTest",
        data_root=tmp_path,
        embedding_dim=EMBED_DIM,
        snapshot_every_n_events=10_000,
        snapshot_every_seconds=10_000.0,
    )


def _manifest(name, *, dependencies=None):
    return build_tool_manifest(
        name=name,
        description=f"{name}: does a thing for the economy test.",
        input_schema={"type": "object",
                      "properties": {"x": {"type": "string"}}},
        author=AUTHOR,
        trust_class="pinned",
        code_digest="c" * 64,
        dependencies=dependencies,
        created_ts=1780531200,
    )


def _receipt(seq, digest, *, caller=CALLER):
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": digest,
        "tool_author": AUTHOR,
        "receipt_digest": f"r{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "attested": True,   # cognitive tier — the only mint input
        "score": 0.8,
    }


def _vet(seq, digest, vetter):
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": vetter,
        "manifest_digest": digest,
        "tool_author": AUTHOR,
        "receipt_digest": f"v{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "vet": True,
    }


@pytest.fixture(scope="module")
def closed_economy(tmp_path_factory):
    """One closed epoch with a composite tool (dep declared), both
    greenlit in the same close, attested usage on the composite.
    Module-scoped: every consumer reads, none mutates."""
    svc = _svc(tmp_path_factory.mktemp("economy"))
    embedder = HashingEmbedder(dim=EMBED_DIM)
    svc.open_epoch("ep-econ-1")

    dep_digest = svc.submit_tool_manifest(
        _manifest("dep_tool"), agent_id=AUTHOR, embedder=embedder,
    )["manifest_digest"]
    comp_digest = svc.submit_tool_manifest(
        _manifest("composite_tool", dependencies=[dep_digest]),
        agent_id=AUTHOR, embedder=embedder,
    )["manifest_digest"]

    events = [
        _receipt(10, comp_digest),
        _receipt(11, comp_digest),
        # Distinct vetters greenlight BOTH digests in this close
        # (only greenlit manifests mint; the dep needs it to receive
        # its composition share).
        _vet(20, comp_digest, "vetter-1"),
        _vet(21, comp_digest, "vetter-2"),
        _vet(22, dep_digest, "vetter-1"),
        _vet(23, dep_digest, "vetter-2"),
    ]
    svc.submit_events(events, equilibrate_after=False)
    close = svc.close_epoch()
    return svc, dep_digest, comp_digest, close


class TestReadEconomyGraph:
    def test_registrations_carry_dep_dag(self, closed_economy):
        svc, dep, comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert eg["epochs_considered"] == 1
        assert eg["registrations"][comp]["author"] == AUTHOR
        assert eg["registrations"][comp]["deps"] == [dep]
        assert eg["registrations"][dep]["deps"] == []

    def test_vetting_state(self, closed_economy):
        svc, dep, comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        for digest in (dep, comp):
            v = eg["vetting"][digest]
            assert v["greenlit"] is True
            assert v["validators"] == 2
            assert v["busted"] is False
        # Carried state is post-close (royalty window already ticked
        # once); the epoch record itself keeps the pre-decrement value.
        assert eg["vetting"][comp]["royalty_left"] == VET_ROYALTY_EPOCHS - 1

    def test_recent_mint_splits_over_composition(self, closed_economy):
        svc, dep, comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert eg["recent_tool_mint"][comp] > 0.0
        # The dep earns its composition share of the composite's usage.
        assert eg["recent_tool_mint"][dep] > 0.0
        assert eg["recent_tool_mint"][comp] > eg["recent_tool_mint"][dep]

    def test_last_epoch_summary(self, closed_economy):
        svc, dep, comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        last = eg["last_epoch"]
        assert last["epoch_id"] == "ep-econ-1"
        assert comp in last["tool_mint"]
        entry = last["tool_mint"][comp]
        # Record keeps the PRE-decrement royalty window.
        assert entry["royalty_left"] == VET_ROYALTY_EPOCHS
        assert entry["validators"] == 2

    def test_empty_service(self, tmp_path):
        svc = _svc(tmp_path)
        eg = svc.read_economy_graph()
        assert eg == {
            "registrations": {},
            "vetting": {},
            "recent_tool_mint": {},
            "positions": {},
            "last_epoch": None,
            "epochs_considered": 0,
        }

    def test_positions_expose_drifted_head_and_rating(self, closed_economy):
        svc, _dep, comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        pos = eg["positions"][comp]
        assert len(pos["head"]) == 6 and len(pos["mass"]) == 6
        # No axis reviews in this fixture → neutral head, prior mass.
        assert pos["head"] == [0.0] * 6
        assert pos["rating"] == 0.0


class TestDistributionKinds:
    def test_root_kinds_conserve_root_mint(self, closed_economy):
        svc, _dep, _comp, _ = closed_economy
        dist = svc.read_substrate_distribution(last_n_epochs=5)
        assert set(dist["root_kinds"]) == set(dist["root_scores"])
        for root_id, kinds in dist["root_kinds"].items():
            assert set(kinds) == {"claim", "tool", "vetting"}
            assert sum(kinds.values()) == pytest.approx(
                dist["root_mint_recent"][root_id], abs=1e-9)
        assert sum(dist["kind_totals"].values()) == pytest.approx(
            dist["total_mint_recent"], abs=1e-9)

    def test_tool_and_vetting_slices_present(self, closed_economy):
        svc, _dep, _comp, _ = closed_economy
        dist = svc.read_substrate_distribution(last_n_epochs=5)
        totals = dist["kind_totals"]
        assert totals["tool"] > 0.0
        # Royalty window open (validators=2, royalty_left=8 in the
        # record) → the vetting slice is VET_ROYALTY_SHARE of anchored
        # tool mint.
        assert totals["vetting"] > 0.0
        anchored = totals["tool"] + totals["vetting"]
        assert totals["vetting"] == pytest.approx(
            anchored * VET_ROYALTY_SHARE, rel=1e-6)

    def test_no_kinds_regression_when_empty(self, tmp_path):
        svc = _svc(tmp_path)
        dist = svc.read_substrate_distribution()
        assert dist["root_kinds"] == {
            rid: {"claim": 0.0, "tool": 0.0, "vetting": 0.0}
            for rid in dist["root_scores"]
        }
        assert dist["kind_totals"] == {
            "claim": 0.0, "tool": 0.0, "vetting": 0.0}


class TestVizNodeKind:
    def test_manifest_anchors_are_tool_kind(self, closed_economy):
        svc, dep, comp, _ = closed_economy
        nodes = svc.list_nodes_for_visualization(max_nodes=100)
        tool_entries = [e for e in nodes["items"] if e["kind"] == "tool"]
        # Both manifests anchor claim nodes; node ids are n_<hash> (the
        # tm_ prefix is the OBSERVATION id, not the node id).
        assert tool_entries, "manifest anchors should be kind=tool"
        assert all(e["label"].startswith("tool ") for e in tool_entries)
        assert all(e["kind"] in ("tool", "claim") for e in nodes["items"])

    def test_subtree_projection_carries_kind(self, closed_economy):
        svc, _dep, _comp, _ = closed_economy
        nodes = svc.list_nodes_for_visualization(max_nodes=100)
        anchor = next(e for e in nodes["items"] if e["kind"] == "tool")
        proj = svc.compute_subtree_projection(anchor["node_id"])
        assert all("kind" in e for e in proj["items"])
