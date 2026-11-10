"""Reviews, daemon side (Decision 2026-11-09): the review step queues
ToolBond batch entries (attest_tools, vet_tool) with the four fixed fields,
a malware flag carries stored evidence, a household without REP sends
nothing, the tool descriptions list the active charter's axes, and the
review driver's commit / reveal / finalize / claim decisions against a mock
ToolBond.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from atn.agent_tools import execute_tool, get_core_tool_def
from atn.review_driver import ReviewDriver, batch_rows
from atn.runtime.review_batch import (
    STAGE_DONE, STAGE_NONE, STAGE_OPEN, ReviewBatchStore,
    ReviewEntry, commitment_of, store_for,
)
from nodes.common.library.charter import (
    Charter, DEFAULT_CHARTER, active_charter, set_active_charter,
)
from tests.atn.test_tool_store import (
    _family, _make_runtime, _register_echo,
)

E18 = 10 ** 18
D1, D2 = "a1" * 32, "b2" * 32
FIELDS = {"correctness": 0.7, "simplicity": 0.4, "novelty": 0.6, "malware": False}


def _entry(digest, **kw):
    return ReviewEntry(digest=digest, **{**FIELDS, **kw})


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
        self.rep: Optional[int] = 10 * E18
        self.rep_epochs: List[Optional[int]] = []

    def available(self) -> bool:
        return True

    async def current_epoch(self) -> int:
        return self.epoch

    async def chain_time(self) -> int:
        return self.now

    async def reveal_ends(self, epoch: int) -> int:
        return self.anchored.get(int(epoch), 0)

    async def household_rep(self, agent: str, epoch: Optional[int] = None) -> Optional[int]:
        self.rep_epochs.append(epoch)
        return self.rep

    async def commit(self, key: str, batch) -> Dict[str, Any]:
        self.calls.append(("commit", batch.agent, batch.epoch))
        if self.fail_commit:
            return {"success": False, "error": "would revert: NoWeight"}
        c = commitment_of(batch.epoch, batch.agent, batch.inputs(), bytes.fromhex(batch.salt))
        assert c.hex() == batch.commitment  # the driver persisted it first
        self.commits[(batch.epoch, batch.agent.lower())] = {
            "commitment": c.hex(), "revealed": False, "reviews": len(batch.reviews)}
        batch.committed_tx = "0xc"
        return {"success": True, "tx_hash": "0xc"}

    async def commit_of(self, epoch: int, agent: str) -> Dict[str, Any]:
        return self.commits.get((int(epoch), agent.lower()),
                                {"commitment": "00" * 32, "revealed": False})

    async def reveal(self, key: str, batch) -> Dict[str, Any]:
        self.calls.append(("reveal", batch.agent, batch.epoch))
        self.commits[(batch.epoch, batch.agent.lower())]["revealed"] = True
        for r in batch.reviews:
            self.rounds.setdefault((r.digest, batch.epoch), {"stage": STAGE_OPEN})
        batch.revealed_tx = "0xr"
        return {"success": True, "tx_hash": "0xr"}

    async def round_of(self, digest: str, epoch: int) -> Dict[str, Any]:
        return dict(self.rounds.get((digest, int(epoch)), {"stage": STAGE_NONE}))

    async def finalize(self, key: str, digest: str, epoch: int, n: int) -> Dict[str, Any]:
        self.calls.append(("finalize", digest, epoch))
        self.rounds[(digest, int(epoch))]["stage"] = STAGE_DONE
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
    async def test_attest_tools_queues_the_four_fields_and_axes(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **FIELDS,
                           "axes": {"evolution": 0.7, "not_an_axis": 1.0}}],
            "context": "echoing"}, rt, caller_id="parent")
        assert out["attested"] == 1 and "_reviews" not in out
        assert out["reviews"]["queued"] == 1 and out["reviews"]["epoch"] == 3
        batch = store_for(rt).load(addr, 3, list(active_charter().ids))
        (entry,) = batch.reviews
        assert entry.digest == digest
        assert (entry.correctness, entry.simplicity, entry.novelty,
                entry.malware) == (0.7, 0.4, 0.6, False)
        assert entry.axes == {"evolution": 0.7}
        # No note, no malware flag: no evidence.
        assert entry.evidence_hash == "00" * 32
        # The local attestation row keeps the review for history and UI.
        row = rt.tool_store.recent_attestations(digest)[0]
        assert row["axes"] == {"evolution": 0.7, "not_an_axis": 1.0}
        assert row["correctness"] == 0.7 and row["malware"] is False

    async def test_a_review_missing_a_field_is_refused(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", "ok": True, "correctness": 0.5}],
            "context": "x"}, rt, caller_id="parent")
        assert out["attested"] == 0 and "reviews" not in out
        assert "simplicity" in out["skipped"][0]["error"]
        assert store_for(rt).all() == []

    async def test_a_malware_flag_stores_its_evidence(self, tmp_path):
        import hashlib
        rt, digest, bond, addr = await _reviewer(tmp_path)
        flagged = {**FIELDS, "correctness": -1.0, "malware": True}
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **flagged}],
            "context": "x"}, rt, caller_id="parent")
        assert out["attested"] == 0 and "evidence" in out["skipped"][0]["error"]
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **flagged,
                           "evidence": "tool.py line 3 posts ~/.ssh/id_rsa to evil.example"}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["queued"] == 1
        (entry,) = store_for(rt).load(addr, 3, []).reviews
        assert entry.malware is True and entry.evidence_hash != "00" * 32
        raw = rt.tool_store._blob_store().get_bytes(entry.evidence_hash)
        assert hashlib.sha256(raw).hexdigest() == entry.evidence_hash
        got = await execute_tool("fetch_evidence",
                                 {"evidence_hash": entry.evidence_hash}, rt,
                                 caller_id="parent")
        assert got["found"] is True
        assert "line 3" in got["evidence"]["evidence"]
        assert got["evidence"]["manifest_digest"] == digest

    async def test_fetch_evidence_asks_peers_for_what_it_lacks(self, tmp_path):
        import hashlib
        rt, digest, bond, addr = await _reviewer(tmp_path)
        blob = b'{"evidence": "line 9 deletes the home dir"}'
        h = hashlib.sha256(blob).hexdigest()

        async def _fetch(d):
            return blob if d == h else None
        rt.tool_store.blob_fetcher = _fetch
        got = await execute_tool("fetch_evidence", {"evidence_hash": h}, rt,
                                 caller_id="parent")
        assert got["found"] and got["evidence"]["evidence"].startswith("line 9")
        missing = await execute_tool("fetch_evidence", {"evidence_hash": "cd" * 32},
                                     rt, caller_id="parent")
        assert missing["found"] is False

    async def test_household_without_rep_sends_nothing(self, tmp_path):
        from atn.agent_tools import review_due_names
        from atn.runtime.review_batch import NO_REP_NOTE
        rt, digest, bond, addr = await _reviewer(tmp_path)
        bond.rep = 0
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **FIELDS}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["queued"] == 0 and out["reviews"]["note"] == NO_REP_NOTE
        # Read at the open epoch's snapshot, as ToolBond.commit weighs it.
        assert bond.rep_epochs == [bond.epoch]
        assert store_for(rt).all() == []
        # And the daemon stops reminding it.
        rt.tool_store.adopted_by = lambda agent_id: [rt.tool_store.get(digest)]
        assert review_due_names(rt, "parent") == []

    async def test_unregistered_reviewer_queues_nothing(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path, on_chain=False)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **FIELDS}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["queued"] == 0 and "registered" in out["reviews"]["note"]

    async def test_own_tool_is_skipped(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        rec = rt.tool_store.get(digest)
        rec.manifest["author"] = addr       # simulate the reviewer's own tool
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **FIELDS}],
            "context": "x"}, rt, caller_id="parent")
        assert out["reviews"]["queued"] == 0
        assert "own tool" in out["reviews"]["skipped"][0]["error"]

    async def test_vet_tool_queues_a_review(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        inspected = await execute_tool("vet_tool", {"digest": digest}, rt,
                                       caller_id="parent")
        assert "code" in inspected and "review" not in inspected
        out = await execute_tool("vet_tool", {
            "digest": digest, "report": "pipes curl into sh",
            **FIELDS, "correctness": -0.5, "malware": True,
            "evidence": "main.py line 2: curl | sh",
            "axes": {"evolution": -0.5}}, rt, caller_id="parent")
        assert out["review"]["queued"] == 1
        (entry,) = store_for(rt).load(addr, 3, []).reviews
        assert entry.malware is True and entry.correctness == -0.5
        assert entry.evidence_hash not in ("00" * 32, out["review_digest"])

    async def test_vet_tool_without_a_report_is_refused(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        out = await execute_tool("vet_tool", {"digest": digest, **FIELDS}, rt,
                                 caller_id="parent")
        assert "report" in out["error"]

    async def test_committed_batch_sends_new_reviews_to_next_epoch(self, tmp_path):
        rt, digest, bond, addr = await _reviewer(tmp_path)
        store = store_for(rt)
        store.add(addr, 3, list(DEFAULT_CHARTER.ids), _entry(D1))
        b = store.load(addr, 3, [])
        b.salt, b.commitment = "11" * 32, "22" * 32
        store.save(b)
        out = await execute_tool("attest_tools", {
            "judgments": [{"tool": "echo_tool", **FIELDS}],
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
                assert "NIST SP 800-53" in desc and "0.5" in desc
                assert "stake" not in desc.replace("no stake", "")
        finally:
            set_active_charter(DEFAULT_CHARTER)
        assert "kindness" not in get_core_tool_def("attest_tools")["description"]

    def test_schemas_carry_the_four_fields(self):
        item = get_core_tool_def("attest_tools")["input_schema"]["properties"][
            "judgments"]["items"]
        assert {"correctness", "simplicity", "novelty", "malware",
                "evidence", "axes"} <= set(item["properties"])
        assert {"correctness", "simplicity", "novelty", "malware"} <= set(item["required"])
        assert not {"harm", "stake", "own_share"} & set(item["properties"])
        vet = get_core_tool_def("vet_tool")["input_schema"]["properties"]
        assert {"correctness", "malware", "evidence", "report"} <= set(vet)
        assert not {"harm", "stake", "verdict"} & set(vet)

    def test_flag_tool_is_retired(self):
        assert get_core_tool_def("flag_tool") is None


# ---------------------------------------------------------------------------
# Driver decisions
# ---------------------------------------------------------------------------

async def _driver_setup(tmp_path):
    rt, digest, bond, addr = await _reviewer(tmp_path)
    store = store_for(rt)
    axes = list(DEFAULT_CHARTER.ids)
    store.add(addr, 3, axes, _entry(D1, axes={"evolution": 1.0}))
    store.add(addr, 3, axes, _entry(D2, malware=True, evidence_hash="ee" * 32))
    return rt, bond, addr, store, ReviewDriver(rt)


class TestDriver:
    async def test_commits_only_after_the_local_close(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        assert (await drv.tick())["committed"] == 0
        drv.on_local_close()
        assert (await drv.tick())["committed"] == 1
        b = store.load(addr, 3, [])
        assert b.status == "committed" and b.commitment and b.salt
        assert bond.commits[(3, addr.lower())]["reviews"] == 2
        # Committed once per epoch.
        drv.on_local_close()
        await drv.tick()
        assert [c for c in bond.calls if c[0] == "commit"] == [("commit", addr, 3)]

    async def test_a_household_without_rep_is_refused(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        bond.fail_commit = True              # ToolBond: NoWeight
        drv.on_local_close()
        await drv.tick()
        b = store.load(addr, 3, [])
        assert not b.commitment and b.status == "refused" and "REP" in b.last_error
        bond.fail_commit = False
        assert (await drv.tick())["committed"] == 0  # refused for good

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

    async def test_finalize_waits_for_the_reveal_window_then_claims(self, tmp_path):
        rt, bond, addr, store, drv = await _driver_setup(tmp_path)
        drv.on_local_close()
        await drv.tick()
        bond.epoch, bond.anchored[3] = 4, bond.now + 10
        await drv.tick()                                   # reveal
        got = await drv.tick()                             # still inside reveal window
        assert got["finalized"] == 0
        bond.now += 11
        got = await drv.tick()
        # Both rounds Done and claimed; a malware yes waits for nothing,
        # since a slashed stake goes to the treasury (Decision 2026-11-10).
        assert got["claimed"] == 2
        b = store.load(addr, 3, [])
        assert b.rounds == {D1: "claimed", D2: "claimed"} and b.status == "settled"
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
        "judgments": [{"tool": "echo_tool", **FIELDS, "axes": {"evolution": 0.6}}],
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
