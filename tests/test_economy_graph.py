"""Economy-graph read surface (the Substrate view and the Tools page).

``read_economy_graph``: registrations + declared-dep DAG + positions +
recent per-digest mint. Drives open_epoch -> submit_tool_manifest
(a tool_registered event) -> close_epoch -> read. The local close has no
review snapshot, so it mints nothing for tools (a tool is paid once from
staked ToolBond review rounds, see compute_tool_mint). Tests that need
nonzero tool mint feed it through an epoch_history record.
"""

from __future__ import annotations

import pytest

from nodes.common.library.tool_manifest import build_tool_manifest
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
    svc.open_epoch("ep-econ-1")

    dep_digest = svc.submit_tool_manifest(
        _manifest("dep_tool"), agent_id=AUTHOR, )["manifest_digest"]
    comp_digest = svc.submit_tool_manifest(
        _manifest("composite_tool", dependencies=[dep_digest]),
        agent_id=AUTHOR, )["manifest_digest"]

    close = svc.close_epoch()
    return svc, dep_digest, comp_digest, close


def _inject_tool_mint(svc, dep, comp):
    """Append a history record carrying tool mint (as a federated close
    would)."""
    svc._epoch_history.append({
        "epoch_id": "ep-econ-2",
        "closed_at": 1.0,
        "total_mint": 3.0,
        "tool_mint": {comp: {"mint": 2.0}, dep: {"mint": 1.0}},
    })


@pytest.fixture
def economy_with_mint(tmp_path):
    svc = _svc(tmp_path)
    svc.open_epoch("ep-econ-1")
    dep = svc.submit_tool_manifest(
        _manifest("dep_tool"), agent_id=AUTHOR, )["manifest_digest"]
    comp = svc.submit_tool_manifest(
        _manifest("composite_tool", dependencies=[dep]),
        agent_id=AUTHOR, )["manifest_digest"]
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
