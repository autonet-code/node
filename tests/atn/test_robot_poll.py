"""The poll rail (docs/robots.md, Decision 2026-10-25, Phase A)."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from eth_account import Account

from atn.robot import crypto as C
from atn.robot import poll as P
from atn.robot import worklog as W
from atn.robot.node import LocalNetwork, PollNode, PollParams

SEED = "ab" * 32


def _acct():
    a = Account.create()
    return a.address.lower(), a.key.hex()


def _members(n_houses: int = 6, agents_per_house: int = 2):
    """Households h0..h(n-1), REP rising with the index."""
    keys, members = {}, []
    for h in range(n_houses):
        house, _ = _acct()
        for _ in range(agents_per_house):
            addr, key = _acct()
            keys[addr] = key
            members.append(P.Member(addr, house, rep=(h + 1) * 100))
    return members, keys


def _round(members, index=5, start=1000.0, period=100.0, **kw):
    return P.Round(index=index, start=start, seed=SEED, period_s=period,
                   members=tuple(members), **kw)


# -- the draw ---------------------------------------------------------------

def test_draw_is_deterministic_and_order_free():
    members, _ = _members()
    robot, _ = _acct()
    a = P.draw_pollers(_round(members), robot, k=3, percentile=50)
    b = P.draw_pollers(_round(list(reversed(members))), robot, k=3, percentile=50)
    assert a == b and len(a) == 3


def test_one_poller_per_household_and_never_the_robots_own():
    members, _ = _members(n_houses=4)
    robot = members[-1].agent                     # robot belongs to the top household
    rnd = _round(members)
    drawn = P.draw_pollers(rnd, robot, k=3, percentile=0)
    houses = [rnd.household_of(a) for a in drawn]
    assert len(set(houses)) == 3
    assert rnd.household_of(robot) not in houses


def test_percentile_floor_widens_to_k():
    members, _ = _members(n_houses=10)
    # Top 10% of 10 households is 1, so the floor widens to the top 3.
    houses = P.eligible_households(members, percentile=90, k=3)
    top3 = sorted({m.household for m in members}, key=lambda h: -next(
        m.rep for m in members if m.household == h))[:3]
    assert houses == set(top3)


def test_sanctioned_and_zero_rep_households_never_poll():
    members, _ = _members(n_houses=5)
    top = members[-1].household
    zero = P.Member(_acct()[0], "0x" + "11" * 20, 0)
    rnd = _round(members + [zero], sanctioned=frozenset({top}))
    for robot in (_acct()[0] for _ in range(20)):
        drawn = P.draw_pollers(rnd, robot, k=3, percentile=0)
        assert top not in {rnd.household_of(a) for a in drawn}
        assert zero.agent not in drawn


def test_draw_varies_by_robot():
    members, _ = _members(n_houses=12)
    rnd = _round(members)
    draws = {tuple(P.draw_pollers(rnd, _acct()[0], k=3, percentile=0)) for _ in range(12)}
    assert len(draws) > 1


# -- the poll ----------------------------------------------------------------

def test_poll_signature_fixes_time_and_challenge():
    members, keys = _members()
    robot, _ = _acct()
    rnd = _round(members)
    poller = P.draw_pollers(rnd, robot, k=3, percentile=50)[0]
    p1 = P.make_poll(rnd, robot, 0, poller=poller, private_key=keys[poller])
    p2 = P.make_poll(rnd, robot, 0, poller=poller, private_key=keys[poller])
    assert p1["sig"] == p2["sig"]                 # deterministic
    at, challenge = P.poll_timing(p1, rnd)
    assert rnd.start <= at < rnd.start + rnd.period_s and len(challenge) == 64
    assert P.verify_poll(p1, rnd, k=3, percentile=50, now=at) == []
    assert any(e.startswith("early") for e in
               P.verify_poll(p1, rnd, k=3, percentile=50, now=at - 10))


def test_poll_from_an_undrawn_member_is_refused():
    members, keys = _members()
    robot, _ = _acct()
    rnd = _round(members)
    drawn = P.draw_pollers(rnd, robot, k=3, percentile=50)
    outsider = next(m.agent for m in members if m.agent not in drawn)
    poll = P.make_poll(rnd, robot, 0, poller=outsider, private_key=keys[outsider])
    at, _ = P.poll_timing(poll, rnd)
    assert "poller was not drawn for this robot and slot" in \
        P.verify_poll(poll, rnd, k=3, percentile=50, now=at)


def test_high_s_signature_is_rejected():
    addr, key = _acct()
    sig = bytes.fromhex(C.sign_bytes(b"x", key)[2:])
    r, s, v = sig[:32], int.from_bytes(sig[32:64], "big"), sig[64]
    n = C._N
    flipped = r + (n - s).to_bytes(32, "big") + bytes([v ^ 1])
    with pytest.raises(ValueError, match="high-s"):
        C.recover_bytes(b"x", "0x" + flipped.hex())
    assert C.recover_bytes(b"x", "0x" + sig.hex()) == addr


# -- the worklog chain ---------------------------------------------------------

def _chunk(bag: Path, stream: str, i: int, data: bytes, t: float) -> Path:
    p = bag / stream / "worklog" / f"worklog_{i}.mcap"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data + W.MCAP_MAGIC)
    os.utime(p, (t, t))
    return p


def test_chain_commits_closed_chunks_per_stream(tmp_path):
    bag = tmp_path / "bags"
    addr, key = _acct()
    wl = W.Worklog(bag, tmp_path / "state", addr, key)
    _chunk(bag, "pose", 0, b"p0", 1)
    _chunk(bag, "pose", 1, b"p1", 2)              # still being written
    _chunk(bag, "scan", 0, b"s0", 1)
    _chunk(bag, "scan", 1, b"s1", 2)
    l0 = wl.commit(packs=["d"], now=10)
    assert {lf["name"] for lf in l0["leaves"]} == {"pose/worklog/worklog_0.mcap",
                                                   "scan/worklog/worklog_0.mcap"}
    assert wl.commit(packs=["d"], now=11) is None          # nothing new closed
    _chunk(bag, "pose", 2, b"p2", 3)
    l1 = wl.commit(packs=["d"], now=12)
    assert [lf["name"] for lf in l1["leaves"]] == ["pose/worklog/worklog_1.mcap"]
    assert W.verify_chain(wl.links, robot=addr) == []
    # Reload from disk: same chain, nothing re-committed.
    again = W.Worklog(bag, tmp_path / "state", addr, key)
    assert again.head == l1 and again.pending_leaves() == []


def test_chain_detects_tampering_and_dropped_links(tmp_path):
    bag = tmp_path / "bags"
    addr, key = _acct()
    wl = W.Worklog(bag, tmp_path / "state", addr, key)
    for i in range(4):
        _chunk(bag, "pose", i, f"p{i}".encode(), i + 1)
        wl.commit(packs=[], force=True, now=10 + i)
    links = wl.links
    assert W.verify_chain([links[0], links[2]], robot=addr)          # gap
    files = wl.produce(lf["name"] for lf in links[1]["leaves"])
    assert W.verify_chunks(links[1], files) == []
    bad = {k: b"!" + v for k, v in files.items()}
    assert W.verify_chunks(links[1], bad)
    forged = C.sign({**links[1], "leaves": []}, key)
    assert "root does not match the leaves" in W.verify_link(forged, robot=addr)


def test_thinning_keeps_the_chain_verifiable(tmp_path):
    bag = tmp_path / "bags"
    addr, key = _acct()
    wl = W.Worklog(bag, tmp_path / "state", addr, key)
    for i in range(3):
        _chunk(bag, "pose", i, b"p%d" % i, i + 1)
        _chunk(bag, "scan", i, b"s%d" % i, i + 1)
    link = wl.commit(packs=[], now=100)
    removed = wl.prune({"scan": 50}, now=200)
    assert removed and all(n.startswith("scan/") for n in removed)
    kept = wl.produce(lf["name"] for lf in link["leaves"])
    assert kept and all(n.startswith("pose/") for n in kept)
    assert W.verify_chunks(link, kept) == []
    assert W.verify_chain(wl.links, robot=addr) == []
    # A kept leaf proves against the signed root by itself.
    hashes = [W.leaf_hash(lf) for lf in link["leaves"]]
    i = next(j for j, lf in enumerate(link["leaves"]) if lf["stream"] == "pose")
    assert C.verify_proof(hashes[i], C.merkle_proof(hashes, i), link["root"])


def test_merkle_proofs_all_sizes():
    for n in range(1, 9):
        leaves = [C.sha256_hex(bytes([i])) for i in range(n)]
        root = C.merkle_root(leaves)
        for i in range(n):
            assert C.verify_proof(leaves[i], C.merkle_proof(leaves, i), root)


# -- a whole round between nodes -------------------------------------------------

class World:
    """Three robots in one territory (each witnesses the other two) and five
    member households that poll. One fake clock."""

    def __init__(self, tmp_path: Path, period: float = 100.0, standards=("pack",)):
        self.standards = tuple(standards)
        self.packs = {}
        self.now = [1000.0]
        self.net = LocalNetwork()
        members, self.keys = _members(n_houses=5, agents_per_house=1)
        self.robots = []
        for i in range(3):
            addr, key = _acct()
            self.keys[addr] = key
            self.robots.append(addr)
            members.append(P.Member(addr, f"0xrobot{i}", 0))
        self.members = members
        self.rnd = _round(members, start=1000.0, period=period)
        self.challenges: dict[str, list[str]] = {r: [] for r in self.robots}
        self.params = PollParams(k=3, percentile=0, deadline_s=2.0, commit_gap_s=30.0)
        self.nodes: dict[str, PollNode] = {}
        self.bags: dict[str, Path] = {}
        for m in members:
            is_robot = m.agent in self.robots
            bag = tmp_path / m.agent / "bags"
            self.bags[m.agent] = bag
            node = PollNode(
                address=m.agent, private_key=self.keys[m.agent],
                transport=self.net.transport(m.agent), params=self.params,
                round_at=self.round_at, roster=self.roster, robots=lambda: self.robots,
                state_dir=tmp_path / m.agent / "state",
                worklog=W.Worklog(bag, tmp_path / m.agent / "wl", m.agent,
                                  self.keys[m.agent]) if is_robot else None,
                packs=lambda a=m.agent: self.packs.get(a, ["pack"]),
                write_challenge=(lambda c, a=m.agent: self.challenges[a].append(c))
                if is_robot else None,
                clock=lambda: self.now[0])
            self.nodes[m.agent] = node
            self.net.add(node)
        self.chunk_i = 0

    def round_at(self, t):
        if t < self.rnd.start:
            return None
        k = int((t - self.rnd.start) // self.rnd.period_s)
        return P.Round(index=self.rnd.index + k, start=self.rnd.start + k * self.rnd.period_s,
                       seed=SEED, period_s=self.rnd.period_s, members=self.rnd.members,
                       standards=self.standards)

    def roster(self, robot):
        return [r for r in self.robots if r != robot.lower()] if robot.lower() in self.robots else []

    def record(self):
        """Every robot closes one chunk."""
        self.chunk_i += 1
        for r in self.robots:
            if r not in self.net.down:
                _chunk(self.bags[r], "pose", self.chunk_i, os.urandom(16), self.chunk_i)

    async def run_until(self, t, step=1.0):
        while self.now[0] < t:
            self.now[0] += step
            if int(self.now[0]) % 5 == 0:
                self.record()
            for n in list(self.nodes.values()):
                if n.address not in self.net.down:
                    await n.step()


def _run(coro):
    return asyncio.run(coro)


def test_full_round_receipts_aggregates_and_witnesses(tmp_path):
    w = World(tmp_path)
    _run(w.run_until(1000 + 100 + 5))
    rnd = w.rnd
    subject = w.robots[0]
    robot_node = w.nodes[subject]
    drawn = P.draw_pollers(rnd, subject, k=3, percentile=0)
    # Every drawn poller polled; the robot holds a receipt as subject from each.
    as_subject = [r for r in robot_node.receipts if r["role"] == "subject" and r["round"] == rnd.index]
    assert sorted(r["poller"] for r in as_subject) == sorted(drawn)
    # It also answered as a witness of the other two robots.
    assert any(r["role"] == "witness" for r in robot_node.receipts)
    # Each poll's challenge reached the subject AND both witnesses.
    for r in as_subject:
        poll_challenges = [c for c in w.challenges[subject]]
        assert len(poll_challenges) >= 3
    # Aggregates: one per poller for the round, and every receipt is in one.
    aggs = [g for g in w.net.gossip_log if g.get("type") == P.AGGREGATE_TYPE
            and g["round"] == rnd.index]
    assert {a["poller"] for a in aggs} >= set(drawn)
    for receipt in robot_node.receipts:
        if receipt["round"] == rnd.index:
            assert any(P.receipt_in_aggregate(receipt, a) for a in aggs)
    # Every witness holds each robot's newest link, poll answers included.
    for r in w.robots:
        for wit in w.roster(r):
            assert w.nodes[wit].heads[r][0]["seq"] == w.nodes[r].worklog.head["seq"]
            assert w.nodes[wit].missed.get(r, 0) == 0
    # The chain each robot built is intact, and no reports were filed.
    for r in w.robots:
        assert W.verify_chain(w.nodes[r].worklog.links, robot=r) == []
    reports = [g for g in w.net.gossip_log if g.get("type") == P.REPORT_TYPE]
    assert reports == [], reports


def test_absent_poller_is_reported_by_the_robot(tmp_path):
    w = World(tmp_path)
    subject = w.robots[0]
    lazy = P.draw_pollers(w.rnd, subject, k=3, percentile=0)[0]
    w.net.down.add(lazy)
    _run(w.run_until(1000 + 100 + 5))
    absent = [g for g in w.net.gossip_log if g.get("type") == P.REPORT_TYPE
              and g["kind"] == "poller_absent" and g["robot"] == subject]
    assert [a["poller"] for a in absent] == [lazy]
    assert C.signer(absent[0]) == subject


def test_silent_robot_is_reported_missing_and_unanswered(tmp_path):
    w = World(tmp_path)
    _run(w.run_until(1010))
    gone = w.robots[1]
    w.net.down.add(gone)
    _run(w.run_until(1000 + 100 + 5))
    kinds = {(g["kind"], g["robot"]) for g in w.net.gossip_log
             if g.get("type") == P.REPORT_TYPE}
    assert ("missing_commit", gone) in kinds
    assert ("no_answer", gone) in kinds
    assert all(robot == gone for _, robot in kinds)


def test_late_answer_is_reported(tmp_path):
    w = World(tmp_path)
    subject = w.robots[2]
    node = w.nodes[subject]
    real = node.handle_request

    async def slow(msg):
        w.now[0] += 5                       # five seconds pass on the poller's clock
        return await real(msg)
    node.handle_request = slow
    _run(w.run_until(1000 + 100 + 5))
    late = [g for g in w.net.gossip_log if g.get("type") == P.REPORT_TYPE
            and g["kind"] == "late" and g["robot"] == subject]
    assert late


# -- rounds from the chain, rosters from the Registry --------------------------

def test_chain_rounds_pin_to_the_anchor_in_force():
    from atn.robot.rail import ChainRounds
    anchors = [(1000, "0x" + "01" * 32), (1050, "0x" + "02" * 32),
               (1120, "0x" + "03" * 32), (1310, "0x" + "04" * 32)]
    asked = []

    def voice(i):
        asked.append(i)
        return {"agent_rep": {"0xa": 10 * (i + 1), "0xb": 5},
                "owner_map": {"0xa": "0xhouse"}, "sanctioned_households": ["0xbad"],
                "robots": ["0xr1", "0xr2"], "robot_standards": ["d" * 64]}
    rounds = ChainRounds(period_s=100, anchors=lambda: anchors, voice=voice)
    rounds.refresh(now=1250)
    # Round 1 starts at 1100: anchors 0 and 1 precede it, too few for a seed.
    assert rounds.round_at(1150) is None
    # Round 2 starts at 1200, anchor 2 (ts 1120) in force, seed = anchor 0.
    rnd = rounds.round_at(1250)
    assert rnd.index == 2 and rnd.start == 1200 and rnd.seed == "01" * 32
    assert P.Member("0xa", "0xhouse", 30) in rnd.members
    assert rnd.sanctioned == frozenset({"0xbad"})
    assert rounds.fleet(1250) == ["0xr1", "0xr2"]
    assert rnd.standards == ("d" * 64,)
    # The next anchor (ts 1310) lands inside round 3, so only round 4 uses it.
    rounds.refresh(now=1350)
    assert rounds.round_at(1350).seed == "01" * 32
    rounds.refresh(now=1420)
    assert rounds.round_at(1420).seed == "02" * 32


def _reg(block, **keys):
    return [{"block": block, "index": i, "key": k, "value": v}
            for i, (k, v) in enumerate(keys.items())]


def test_robot_policy_cascades_standards_up_the_nesting():
    from nodes.common.voice_state import robot_policy
    A, B, C = "a" * 64, "b" * 64, "c" * 64
    regs = {
        "0xdistrict": _reg(1, **{
            "jurisdiction.district.robots": '["0xR1", "0xr2"]',
            "jurisdiction.district.robot_standards": f'["{A}"]',
            "jurisdiction.district.parent": '{"registry": "0xCity", "id": "city"}'}),
        "0xcity": _reg(1, **{
            "jurisdiction.city.robots": '["0xcityrobot"]',
            "jurisdiction.city.robot_standards": f'{{"packs": ["0x{B}"]}}',
            "jurisdiction.city.parent": '{"registry": "0xstate", "id": "state"}'}),
        "0xstate": _reg(1, **{"jurisdiction.state.robot_standards": f'["{C}", "{A}"]'}),
    }
    pol = robot_policy(lambda r: regs[r], "0xDistrict", 5, "district")
    assert pol["robots"] == ["0xr1", "0xr2"]                 # own fleet only
    assert pol["robot_standards"] == sorted({A, B, C})
    assert pol["jurisdiction_lineage"] == ["district", "city", "state"]
    # Pinned: a parent's later write is invisible at the snapshot block.
    regs["0xcity"] += _reg(9, **{"jurisdiction.city.robot_standards": "[]"})
    assert B in robot_policy(lambda r: regs[r], "0xdistrict", 5, "district")["robot_standards"]
    assert B not in robot_policy(lambda r: regs[r], "0xdistrict", 9, "district")["robot_standards"]


def test_robot_policy_stops_at_a_loop():
    from nodes.common.voice_state import robot_policy
    regs = {"0xa": _reg(1, **{"jurisdiction.a.parent": '{"registry": "0xb", "id": "b"}'}),
            "0xb": _reg(1, **{"jurisdiction.b.parent": '{"registry": "0xa", "id": "a"}'})}
    assert robot_policy(lambda r: regs[r], "0xa", 2, "a")["jurisdiction_lineage"] == ["a", "b"]


def test_robot_protocol_round_trip_between_two_hosts():
    trio = pytest.importorskip("trio")
    from libp2p.peer.peerinfo import PeerInfo
    from nodes.common.p2p import AutonetHost

    async def main():
        a = AutonetHost(node_id="rail-a", listen_host="127.0.0.1")
        b = AutonetHost(node_id="rail-b", listen_host="127.0.0.1")
        seen = []

        async def handler(msg):
            seen.append(msg)
            return {"echo": msg["n"] + 1}
        b.set_robot_handler(handler)
        async with a.run():
            async with b.run():
                await a.connect_to_info(PeerInfo(b._host.get_id(), b._host.get_addrs()))
                out = await a.request_robot(b._host.get_id(), {"n": 41})
                none = await b.request_robot(a._host.get_id(), {"n": 1})
        return out, none, seen
    out, none, seen = trio.run(main)
    assert out == {"echo": 42} and seen == [{"n": 41}]
    assert "error" in none          # a host with no handler answers with an error


def test_wrong_packs_are_reported(tmp_path):
    w = World(tmp_path)
    stale = w.robots[1]
    w.packs[stale] = ["an-old-pack"]
    _run(w.run_until(1000 + 100 + 5))
    kinds = {(g["kind"], g["robot"]) for g in w.net.gossip_log
             if g.get("type") == P.REPORT_TYPE}
    assert kinds == {("wrong_packs", stale)}


def test_unfinished_chunk_is_not_committed(tmp_path):
    """The recorder finishes the old chunk after opening the next: until the
    closing magic lands, the chunk (and any after it) waits."""
    bag = tmp_path / "bags"
    addr, key = _acct()
    wl = W.Worklog(bag, tmp_path / "state", addr, key)
    first = _chunk(bag, "pose", 0, b"p0", 1)
    first.write_bytes(b"p0-half-written")
    os.utime(first, (1, 1))
    _chunk(bag, "pose", 1, b"p1", 2)
    _chunk(bag, "pose", 2, b"p2", 3)
    assert wl.commit(packs=[], now=10) is None
    first.write_bytes(b"p0" + W.MCAP_MAGIC)
    os.utime(first, (1, 1))
    link = wl.commit(packs=[], now=11)
    assert [lf["name"] for lf in link["leaves"]] == ["pose/worklog/worklog_0.mcap",
                                                     "pose/worklog/worklog_1.mcap"]


def test_link_pushed_while_a_poll_is_in_flight_is_not_a_break(tmp_path):
    """A robot commits and pushes a link between receiving a poll and
    answering it: the poller must judge the answer against the head it held
    when it asked."""
    w = World(tmp_path)
    subject = w.robots[0]
    node = w.nodes[subject]
    real = node.handle_request

    async def push_then_answer(msg):
        link = node.worklog.commit(packs=["pack"], force=True, now=w.now[0])
        for m in w.members:
            await w.nodes[m.agent].handle_push(json.loads(json.dumps(link)))
        return await real(msg)
    node.handle_request = push_then_answer
    _run(w.run_until(1000 + 100 + 5))
    reps = [g for g in w.net.gossip_log if g.get("type") == P.REPORT_TYPE]
    assert reps == [], reps


def test_poll_landing_in_the_slack_after_the_round_is_not_absent(tmp_path):
    """A poller held up until just after the round ends still polls within
    the slack; the robot waits for that before calling it absent."""
    w = World(tmp_path)
    subject = w.robots[0]
    late = P.draw_pollers(w.rnd, subject, k=3, percentile=0)[0]

    async def run():
        while w.now[0] < 1000 + 100 + 6:
            w.now[0] += 1
            if int(w.now[0]) % 5 == 0:
                w.record()
            for n in list(w.nodes.values()):
                if n.address == late and w.now[0] < 1100 + 1:
                    continue                      # busy until 1 s after the end
                await n.step()
    _run(run())
    absent = [g for g in w.net.gossip_log if g.get("type") == P.REPORT_TYPE
              and g["kind"] == "poller_absent"]
    assert absent == [], absent
    assert any(r["poller"] == late and r["round"] == w.rnd.index
               for r in w.nodes[subject].receipts)
