"""Economy-graph read surfaces (visualization backend).

Covers the additive surfaces added for the economy/constellation
visualizations:

  * ``WorldService.read_economy_graph``: registrations + declared-dep
    DAG + positions + recent per-digest mint.
  * ``read_substrate_distribution``'s per-kind split (``root_kinds`` /
    ``kind_totals``): tool mint at manifest anchors, everything else
    claim. Per-root conservation against ``root_mint_recent``.
  * ``kind`` on ``list_nodes_for_visualization`` entries (tool anchors
    vs plain claims).

Drives the REAL local close path: open_epoch -> submit_tool_manifest
(registration sprout, author_post standing) -> close_epoch -> read. The
local projection close has no review snapshot, so it mints nothing for
tools (a tool is paid once from staked ToolBond review rounds, see
compute_tool_mint). Tests that need nonzero tool mint feed it through
an epoch_history record carrying tool_mint/node_mint directly.
"""

from __future__ import annotations

import pytest

from nodes.common.library.tool_manifest import build_tool_manifest
from nodes.common.library.usefulness_coords import HashingEmbedder
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


@pytest.fixture(scope="module")
def closed_economy(tmp_path_factory):
    """One closed epoch with a composite tool (dep declared).
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

    close = svc.close_epoch()
    return svc, dep_digest, comp_digest, close


def _inject_tool_mint(svc, dep, comp):
    """Append a history record carrying tool mint (as a federated close
    would), anchored at the tools' real claim nodes."""
    nodes = {e["label"]: e["node_id"]
             for e in svc.list_nodes_for_visualization(max_nodes=100)["items"]
             if e["kind"] == "tool"}
    assert len(nodes) == 2
    node_ids = list(nodes.values())
    svc._epoch_history.append({
        "epoch_id": "ep-econ-2",
        "closed_at": 1.0,
        "total_mint": 3.0,
        "tool_mint": {
            comp: {"mint": 2.0, "node_id": node_ids[0]},
            dep: {"mint": 1.0, "node_id": node_ids[1]},
        },
        "node_mint": {node_ids[0]: 2.0, node_ids[1]: 1.0},
    })


@pytest.fixture
def economy_with_mint(tmp_path):
    svc = _svc(tmp_path)
    embedder = HashingEmbedder(dim=EMBED_DIM)
    svc.open_epoch("ep-econ-1")
    dep = svc.submit_tool_manifest(
        _manifest("dep_tool"), agent_id=AUTHOR, embedder=embedder,
    )["manifest_digest"]
    comp = svc.submit_tool_manifest(
        _manifest("composite_tool", dependencies=[dep]),
        agent_id=AUTHOR, embedder=embedder,
    )["manifest_digest"]
    svc.close_epoch()
    _inject_tool_mint(svc, dep, comp)
    return svc, dep, comp


class TestReadEconomyGraph:
    def test_registrations_carry_dep_dag(self, closed_economy):
        svc, dep, comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert eg["epochs_considered"] == 1
        assert eg["registrations"][comp]["author"] == AUTHOR
        assert eg["registrations"][comp]["deps"] == [dep]
        assert eg["registrations"][dep]["deps"] == []

    def test_local_close_mints_nothing_for_tools(self, closed_economy):
        svc, _dep, _comp, _ = closed_economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert "vetting" not in eg
        assert not any(eg["recent_tool_mint"].values())
        assert not any(eg["last_epoch"]["tool_mint"].values())

    def test_recent_mint_from_history(self, economy_with_mint):
        svc, dep, comp = economy_with_mint
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert eg["recent_tool_mint"][comp] == pytest.approx(2.0)
        assert eg["recent_tool_mint"][dep] == pytest.approx(1.0)

    def test_last_epoch_summary(self, economy_with_mint):
        svc, dep, comp = economy_with_mint
        eg = svc.read_economy_graph(last_n_epochs=5)
        last = eg["last_epoch"]
        assert last["epoch_id"] == "ep-econ-2"
        assert last["total_mint"] == 3.0
        assert last["tool_mint"][comp]["mint"] == 2.0

    def test_empty_service(self, tmp_path):
        svc = _svc(tmp_path)
        eg = svc.read_economy_graph()
        assert eg == {
            "registrations": {},
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
    def test_root_kinds_conserve_root_mint(self, economy_with_mint):
        svc, _dep, _comp = economy_with_mint
        dist = svc.read_substrate_distribution(last_n_epochs=5)
        assert set(dist["root_kinds"]) == set(dist["root_scores"])
        for root_id, kinds in dist["root_kinds"].items():
            assert set(kinds) == {"claim", "tool"}
            assert sum(kinds.values()) == pytest.approx(
                dist["root_mint_recent"][root_id], abs=1e-9)
        assert sum(dist["kind_totals"].values()) == pytest.approx(
            dist["total_mint_recent"], abs=1e-9)

    def test_tool_slice_from_history(self, economy_with_mint):
        svc, _dep, _comp = economy_with_mint
        totals = svc.read_substrate_distribution(
            last_n_epochs=5)["kind_totals"]
        assert set(totals) == {"claim", "tool"}
        assert totals["tool"] == pytest.approx(3.0)

    def test_local_close_has_no_tool_slice(self, closed_economy):
        svc, _dep, _comp, _ = closed_economy
        totals = svc.read_substrate_distribution(
            last_n_epochs=5)["kind_totals"]
        assert totals["tool"] == 0.0

    def test_no_kinds_regression_when_empty(self, tmp_path):
        svc = _svc(tmp_path)
        dist = svc.read_substrate_distribution()
        assert dist["root_kinds"] == {
            rid: {"claim": 0.0, "tool": 0.0}
            for rid in dist["root_scores"]
        }
        assert dist["kind_totals"] == {"claim": 0.0, "tool": 0.0}


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
