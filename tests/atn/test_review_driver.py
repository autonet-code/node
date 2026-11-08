"""Staked reviews, daemon side (docs/multi_jurisdiction.md decision 10):
the review step queues ToolBond batch entries (attest_tools, vet_tool), the
tool descriptions list the active charter's axes, and the review driver's
commit / reveal / finalize / claim decisions against a mock ToolBond.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from atn.agent_tools import execute_tool, get_core_tool_def
from atn.review_driver import ReviewDriver, batch_rows
from atn.runtime.review_batch import (
    STAGE_COOLING, STAGE_DONE, STAGE_NONE, STAGE_OPEN, ReviewBatchStore,
    ReviewEntry, commitment_of, store_for,
)
from nodes.common.world_model_substrate.charter import (
    Charter, DEFAULT_CHARTER, active_charter, set_active_charter,
)
from tests.atn.test_tool_store import (
    _family, _make_runtime, _register_echo,
)

E18 = 10 ** 18
D1, D2 = "a1" * 32, "b2" * 32


class FakeBond:
    """A ToolBond with just enough state for the driver's decisions."""

    def __init__(self) -> None:
        self.epoch = 3
        self.now = 1_000
        self.anchored: Dict[int, int] = {}       # epoch -> revealEnds
        self.commits: Dict[tuple, Dict[str, Any]] = {}
        self.rounds: Dict[tuple, Dict[str, Any]] = {}
        self.claimed: set = set()
        self.calls: List[tuple] = []
        self.fail_commit = False

    def available(self) -> bool:
        return True

    async def current_epoch(self) -> int:
        return self.epoch

    async def chain_time(self) -> int:
        return self.now

    async def reveal_ends(self, epoch: int) -> int:
        return self.anchored.get(int(epoch), 0)

    async def commit(self, key: str, batch) -> Dict[str, Any]:
        self.calls.append(("commit", batch.agent, batch.epoch))
        if self.fail_commit:
            return {"success": False, "error": "would revert: NoWeight"}
        c = commitment_of(batch.epoch, batch.agent, batch.inputs(), bytes.fromhex(batch.salt))
        assert c.hex() == batch.commitment  # the driver persisted it first
        self.commits[(batch.epoch, batch.agent.lower())] = {
            "commitment": c.hex(), "revealed": False, "stake": batch.total_stake}
        batch.committed_tx = "0xc"
        return {"success": True, "tx_hash": "0xc"}

    async def commit_of(self, epoch: int, agent: str) -> Dict[str, Any]:
        return self.commits.get((int(epoch), agent.lower()),
                                {"commitment": "00" * 32, "revealed": False})

    async def reveal(self, key: str, batch) -> Dict[str, Any]:
        self.calls.append(("reveal", batch.agent, batch.epoch))
        self.commits[(batch.epoch, batch.agent.lower())]["revealed"] = True
        for r in batch.reviews:
            self.rounds.setdefault((r.digest, batch.epoch),
                                   {"stage": STAGE_OPEN, "overruled": False, "appeal_ends": 0})
        batch.revealed_tx = "0xr"
        return {"success": True, "tx_hash": "0xr"}

    async def round_of(self, digest: str, epoch: int) -> Dict[str, Any]:
        return dict(self.rounds.get((digest, int(epoch)),
                                    {"stage": STAGE_NONE, "overruled": False, "appeal_ends": 0}))

    async def finalize(self, key: str, digest: str, epoch: int, n: int) -> Dict[str, Any]:
        self.calls.append(("finalize", digest, epoch))
        r = self.rounds[(digest, int(epoch))]
        if r["stage"] == STAGE_OPEN and r.get("harm"):
            r["stage"], r["appeal_ends"] = STAGE_COOLING, self.now + 100
        elif r["stage"] == STAGE_COOLING and self.now <= r["appeal_ends"]:
            return {"success": False, "error": "would revert: TooEarly"}
        else:
            r["stage"] = STAGE_DONE
        return {"success": True}

    async def review_of(self, digest: str, epoch: int, reviewer: str) -> Dict[str, Any]:
        return {"claimed": (digest, int(epoch), reviewer.lower()) in self.claimed}

    async def claim(self, key: str, digest: str, epoch: int, reviewer: str) -> Dict[str, Any]:
        self.calls.append(("claim", digest, epoch))
        self.claimed.add((digest, int(epoch), reviewer.lower()))
        return {"success": True}


async def _reviewer(tmp_path, *, on_chain=True):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    res = await _register_echo(rt)          # author = child
    defn = rt.get_agent("parent")
    defn.identity.registered_on_chain = on_chain
    rt.review_bond_client = bond = FakeBond()
    return rt, res["digest"], bond, defn.identity.address


# ---------------------------------------------------------------------------
# Review step -> batch
# ---------------------------------------------------------------------------

class TestReviewQueueing:
    async def test_attest_tools_queues_axes_harm_and_stake(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", "ok": True,
                           "axes": {"correctness": 0.7, "not_an_axis": 1.0},
                           "harm": 0.05, "stake": 3 * E18}],
            "context": "echoing"}, rt, caller_id="parent")
        assert out["attested"] == 1 and "_reviews" not in out
        assert out["reviews"]["queued"] == 1 and out["reviews"]["epoch"] == 3
        batch = store_for(rt).load(addr, 3, list(active_charter().ids))
        (entry,) = batch.reviews
        assert entry.digest == digest
        assert entry.axes == {"correctness": 0.7}
        assert entry.harm == 0.05 and entry.stake == 3 * E18
        # The review note's blob digest would be the evidence; none here.
        assert entry.evidence_hash == "00" * 32
        # The local attestation row keeps the review for history and UI.
        row = rt.tool_store.recent_attestations(digest)[0]
        assert row["axes"] == {"correctness": 0.7, "not_an_axis": 1.0}
        assert row["harm"] == 0.05 and row["stake"] == 3 * E18

    async def test_usage_only_judgment_queues_nothing(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", "ok": True}], "context": "x"},
            rt, caller_id="parent")
        assert out["attested"] == 1 and "reviews" not in out
        assert store_for(rt).all() == []

    async def test_unregistered_reviewer_queues_nothing(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path, on_chain=False)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", "ok": True, "harm": 0.0}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["queued"] == 0 and "registered" in out["reviews"]["note"]

    async def test_own_tool_is_skipped(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        rec = rt.tool_store.get(digest)
        rec.manifest["author"] = addr       # simulate the reviewer's own tool
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", "ok": True, "harm": 0.0}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["queued"] == 0
        assert "own tool" in out["reviews"]["skipped"][0]["error"]

    async def test_vet_tool_queues_a_staked_review(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        out = await execute_tool("vet_tool", {
            "digest": digest, "verdict": "fail", "report": "pipes curl into sh",
            "harm": 0.9, "stake": E18, "axes": {"correctness": -0.5}},
            rt, caller_id="parent")
        assert out["review"]["queued"] == 1
        (entry,) = store_for(rt).load(addr, 3, []).reviews
        assert entry.harm == 0.9 and entry.stake == E18
        assert entry.evidence_hash == out["review_digest"]

    async def test_committed_batch_sends_new_reviews_to_next_epoch(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        store = store_for(rt)
        store.add(addr, 3, list(DEFAULT_CHARTER.ids), ReviewEntry(digest=D1, harm=0.1))
        b = store.load(addr, 3, [])
        b.salt, b.commitment = "11" * 32, "22" * 32
        store.save(b)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", "ok": True, "harm": 0.0}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["epoch"] == 4


class TestDynamicDescriptions:
    def test_review_tools_list_the_active_charter(self):
        seven = Charter(axes=DEFAULT_CHARTER.axes + (("kindness", "Be kind."),))
        set_active_charter(seven)
        try:
            for name in ("attest_tools", "vet_tool"):
                desc = get_core_tool_def(name)["description"]
                assert "kindness (Be kind.)" in desc
                assert "'harm': 0 safe .. 1 malicious, optional" in desc
                assert "'stake'" in desc
        finally:
            set_active_charter(DEFAULT_CHARTER)
        assert "kindness" not in get_core_tool_def("attest_tools")["description"]

    def test_schemas_carry_harm_and_stake(self):
        item = get_core_tool_def("attest_tools")["input_schema"]["properties"][
            "judgments"]["items"]["properties"]
        assert {"axes", "harm", "stake", "own_share"} <= set(item)
        vet = get_core_tool_def("vet_tool")["input_schema"]["properties"]
        assert {"axes", "harm", "stake"} <= set(vet)

    def test_flag_tool_is_retired(self):
        assert get_core_tool_def("flag_tool") is None


# ---------------------------------------------------------------------------
# Driver decisions
# ---------------------------------------------------------------------------

async def _driver_setup(tmp_path):
    rt, digest, bond, addr = await _reviewer(tmp_path)
    store = store_for(rt)
    axes = list(DEFAULT_CHARTER.ids)
    store.add(addr, 3, axes, ReviewEntry(digest=D1, axes={"correctness": 1.0}, stake=5))
    store.add(addr, 3, axes, ReviewEntry(digest=D2, harm=0.8, stake=2))
    return rt, bond, addr, store, ReviewDriver(rt)


class TestDriver:
    async def test_commits_only_after_the_local_close(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        assert (await drv.tick())["committed"] == 0
        drv.on_local_close()
        assert (await drv.tick())["committed"] == 1
        b = store.load(addr, 3, [])
        assert b.status == "committed" and b.commitment and b.salt
        assert bond.commits[(3, addr.lower())]["stake"] == 7
        # Committed once per epoch.
        drv.on_local_close()
        await drv.tick()
        assert [c for c in bond.calls if c[0] == "commit"] == [("commit", addr, 3)]

    async def test_failed_commit_keeps_reviews_pending_and_retries(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        bond.fail_commit = True
        drv.on_local_close()
        await drv.tick()
        b = store.load(addr, 3, [])
        assert not b.commitment and "NoWeight" in b.last_error and len(b.reviews) == 2
        bond.fail_commit = False
        assert (await drv.tick())["committed"] == 1  # same epoch, retried

    async def test_pending_batch_rolls_forward_when_its_epoch_closed(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        bond.epoch = 4                     # epoch 3 anchored before a commit
        assert (await drv.tick())["rolled"] == 1
        assert store.load(addr, 3, []).status == "rolled"
        assert len(store.load(addr, 4, []).reviews) == 2

    async def test_reveals_after_the_anchor_within_the_window(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        assert (await drv.tick())["revealed"] == 0      # not anchored yet
        bond.epoch, bond.anchored[3] = 4, bond.now + 50
        assert (await drv.tick())["revealed"] == 1
        assert store.load(addr, 3, []).status == "revealed"
        assert store.unrevealed() == []

    async def test_missed_window_marks_forfeited(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        bond.epoch, bond.anchored[3] = 4, bond.now - 1
        assert (await drv.tick())["forfeited"] == 1
        assert store.load(addr, 3, []).status == "forfeited"
        assert not [c for c in bond.calls if c[0] == "reveal"]

    async def test_relayed_reveal_is_recognized(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        bond.commits[(3, addr.lower())]["revealed"] = True   # someone relayed it
        bond.epoch, bond.anchored[3] = 4, bond.now - 1
        await drv.tick()
        assert store.load(addr, 3, []).status == "revealed"

    async def test_finalize_waits_out_windows_then_claims(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        bond.epoch, bond.anchored[3] = 4, bond.now + 10
        await drv.tick()                                   # reveal
        bond.rounds[(D2, 3)]["harm"] = True
        got = await drv.tick()                             # still inside reveal window
        assert got["finalized"] == 0
        bond.now += 11
        got = await drv.tick()
        # D1: no harm, Done at once and claimed; D2: harm, now Cooling.
        assert got["claimed"] == 1
        b = store.load(addr, 3, [])
        assert b.rounds == {D1: "claimed", D2: "cooling"} and b.status == "revealed"
        await drv.tick()                                   # appeal window: no-op
        assert store.load(addr, 3, []).rounds[D2] == "cooling"
        bond.now += 101
        got = await drv.tick()
        assert got["claimed"] == 1
        b = store.load(addr, 3, [])
        assert b.status == "settled" and b.rounds[D2] == "claimed"
        assert not [c for c in bond.calls if c[0] == "forfeit"]

    async def test_review_that_did_not_count_is_skipped(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        bond.epoch, bond.anchored[3] = 4, bond.now + 1
        await drv.tick()
        del bond.rounds[(D2, 3)]                           # refunded at reveal
        bond.now += 5
        await drv.tick()
        b = store.load(addr, 3, [])
        assert b.rounds[D2] == "none" and b.status == "settled"

    async def test_restart_resumes_from_disk(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        bond.epoch, bond.anchored[3] = 4, bond.now + 10
        fresh = ReviewDriver(rt, store=ReviewBatchStore(
            rt._config.data_dir, rt.jurisdiction))
        assert (await fresh.tick())["revealed"] == 1

    async def test_crash_after_commit_is_confirmed_on_chain(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        b = store.load(addr, 3, [])
        b.committed_tx, b.status = "", "committing"        # receipt never seen
        store.save(b)
        await drv.tick()
        b = store.load(addr, 3, [])
        assert b.status == "committed" and b.committed_tx

    async def test_ws_rows_carry_status_and_sealed_reveal(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        (row,) = batch_rows(rt)
        assert row["status"] == "committed" and row["reviews"] == 2
        assert row["sealed_reveal"]["salt"] == store.load(addr, 3, []).salt


async def test_review_batches_ws_is_owner_only(tmp_path):
    from types import SimpleNamespace
    from atn.ws_server import WebSocketBridge
    rt, bond, addr, store, drv = await _driver_setup(tmp_path)
    srv = WebSocketBridge.__new__(WebSocketBridge)
    srv.runtime = rt
    owner = SimpleNamespace(scope_ids=None, local=True, owner="0xo", authed=True)
    scoped = SimpleNamespace(scope_ids={"x"}, local=False, owner="0xo", authed=True)
    r = await srv._handle_message({"type": "review_batches", "msg_id": "1"}, owner)
    assert r["ok"] and r["result"]["batches"][0]["status"] == "pending"
    r = await srv._handle_message({"type": "review_batches", "msg_id": "2"}, scoped)
    assert r["ok"] is False and "owner-only" in r["error"]


async def test_unregistered_reviews_are_held_then_submitted(tmp_path):
    from atn.runtime.review_batch import HELD_NOTE, held_reviews
    rt, digest, bond, addr = await _reviewer(tmp_path, on_chain=False)
    out = await execute_tool("attest_tools", {
        "judgments": [{"tool": "echo_tool", "ok": True,
                       "axes": {"correctness": 0.6}, "stake": E18}],
        "context": "echoing"}, rt, caller_id="parent")
    assert out["reviews"]["queued"] == 0 and out["reviews"]["held"] == 1
    assert out["reviews"]["note"] == HELD_NOTE
    assert len(held_reviews(rt, "parent")) == 1

    # Still unregistered: the driver leaves them held.
    driver = ReviewDriver(rt)
    await driver.tick()
    assert len(held_reviews(rt, "parent")) == 1

    # The household registers: the next tick submits them to the open round.
    rt.get_agent("parent").identity.registered_on_chain = True
    done = await driver.tick()
    assert done.get("held_submitted") == 1
    assert held_reviews(rt, "parent") == []
    batch = store_for(rt).load(addr, 3, list(active_charter().ids))
    assert [e.digest for e in batch.reviews] == [digest]
