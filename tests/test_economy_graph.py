"""Economy-graph read surface (the Substrate view and the Tools page).

``LibraryService.read_economy_graph`` reads the latest federated closes:
the anchored registrations with their declared-dep DAG, the ToolBond
ratings as positions, and the recent per-digest mint. Drives the real
path: two published tools -> a federated close over their
tool_registered events -> ``note_federated_close`` -> read.
"""

from __future__ import annotations

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from nodes.common.library.tool_manifest import build_tool_manifest
from nodes.common.library_service import LibraryService

AUTHOR = "author-1"
E18 = 10 ** 18


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


def _close_over(events, **kw):
    kp = Keypair.generate()
    batch = EventBatch(rpb_address="0xEconomyTest", sender_pubkey=kp.public_key,
                       batch_seq=1, events=events, prev_batch_hash=b"",
                       timestamp=1_700_000_000.0)
    return federated_epoch_close(canonical_order([batch]), **kw)


@pytest.fixture
def economy(tmp_path):
    """Two published tools (a composite and its dependency), one
    federated close with a ToolBond rating on the composite."""
    svc = LibraryService("0xEconomyTest", data_root=tmp_path)
    events = []
    svc.subscribe_local_events(events.extend)
    dep = svc.submit_tool_manifest(_manifest("dep_tool"), agent_id=AUTHOR)["manifest_digest"]
    comp = svc.submit_tool_manifest(
        _manifest("composite_tool", dependencies=[dep]), agent_id=AUTHOR)["manifest_digest"]
    snap = {"supply": 100 * E18, "tools": {comp: {
        "axes": {"correctness": {"sum": 80 * 10 * E18, "mass": 10 * E18},
                 "simplicity": {"sum": 40 * 10 * E18, "mass": 10 * E18}},
        "harm": {"sum": 0, "mass": 0}}}}
    close = _close_over(events, review_snapshot=snap)
    svc.note_federated_close(close, epoch_id="ep-econ-1")
    return svc, dep, comp


class TestReadEconomyGraph:
    def test_registrations_carry_dep_dag(self, economy):
        svc, dep, comp = economy
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert eg["epochs_considered"] == 1
        assert eg["registrations"][comp]["author"] == AUTHOR
        assert eg["registrations"][comp]["deps"] == [dep]
        assert eg["registrations"][dep]["deps"] == []

    def test_positions_are_the_toolbond_ratings(self, economy):
        svc, dep, comp = economy
        pos = svc.read_economy_graph()["positions"]
        assert len(pos[comp]["head"]) == 6 and len(pos[comp]["mass"]) == 6
        assert pos[comp]["rating"] == pytest.approx((0.8 + 0.4) / 2)
        assert pos[dep]["head"] == [0.0] * 6 and pos[dep]["rating"] == 0.0

    def test_recent_mint_and_last_epoch(self, economy):
        svc, dep, comp = economy
        svc.note_federated_close({"tool_mint": {comp: {"mint": 2.0}, dep: {"mint": 1.0}},
                                  "total_mint": 3.0}, epoch_id="ep-econ-2")
        eg = svc.read_economy_graph(last_n_epochs=5)
        assert eg["recent_tool_mint"][comp] == pytest.approx(2.0)
        assert eg["recent_tool_mint"][dep] == pytest.approx(1.0)
        assert eg["last_epoch"]["epoch_id"] == "ep-econ-2"
        assert eg["last_epoch"]["total_mint"] == 3.0
        # The later close carried no registrations: the earlier ones stand.
        assert set(eg["registrations"]) == {dep, comp}

    def test_survives_restart(self, economy, tmp_path):
        svc, dep, comp = economy
        again = LibraryService("0xEconomyTest", data_root=tmp_path)
        assert again.read_economy_graph() == svc.read_economy_graph()

    def test_empty_service(self, tmp_path):
        svc = LibraryService("0xEmpty", data_root=tmp_path)
        assert svc.read_economy_graph() == {
            "registrations": {},
            "recent_tool_mint": {},
            "positions": {},
            "last_epoch": None,
            "epochs_considered": 0,
        }
