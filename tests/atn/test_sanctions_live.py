"""The daemon's live sanction list (atn/sanctions.py) and the agent rails it
closes: paying, buying from, and adopting the tools of a sanctioned
household."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from atn.sanctions import SanctionList, refusal

HOUSE = "0x" + "aa" * 20
AGENT = "0x" + "a1" * 20
OTHER = "0x" + "b2" * 20


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _list(values, owners=None, clock=None, reads=None):
    def read():
        if reads is not None:
            reads.append(1)
        return dict(values)
    return SanctionList(read, (owners or {}).get, clock=clock or Clock())


def test_entries_skip_other_keys_lifted_and_expired():
    clock = Clock(1000)
    sl = _list({
        f"sanction.{HOUSE}": json.dumps({"until": 2000, "reason": "trial-1"}),
        f"sanction.{OTHER}": "",
        "sanction.0x" + "cc" * 20: json.dumps({"until": 500}),
        "jurisdiction.autonet.substrate": "0x1",
    }, clock=clock)
    assert sl.entries() == {HOUSE: {"until": 2000, "reason": "trial-1"}}
    clock.t = 2000
    assert sl.entries() == {}


def test_agent_is_covered_by_its_owners_sanction():
    sl = _list({f"sanction.{HOUSE}": "{}"}, owners={AGENT: HOUSE})
    assert sl.check(AGENT.upper().replace("0X", "0x"))["household"] == HOUSE
    assert sl.check(HOUSE)["household"] == HOUSE
    assert sl.check(OTHER) is None


def test_reads_at_most_once_a_minute_and_keeps_last_list_on_failure():
    clock, reads = Clock(), []
    values = {f"sanction.{HOUSE}": "{}"}
    sl = _list(values, clock=clock, reads=reads)
    sl.entries()
    sl.entries()
    assert len(reads) == 1
    clock.t += 61

    def broken():
        raise RuntimeError("rpc down")
    sl._read = broken
    assert HOUSE in sl.entries()


def test_an_unbound_address_is_looked_up_again_after_a_minute():
    """An address with no owner yet may be bound later: its owner is kept
    only for the refresh period, so the owner's sanction comes into view.
    A bound owner is read again after a minute too (it may be rotated)."""
    clock, owners, lookups = Clock(), {}, []

    def owner_of(address):
        lookups.append(address)
        return owners.get(address, "0x" + "00" * 20)
    sl = SanctionList(lambda: {f"sanction.{HOUSE}": "{}"}, owner_of, clock=clock)
    assert sl.check(AGENT) is None
    owners[AGENT] = HOUSE                       # bound after the first look
    clock.t += 30
    assert sl.check(AGENT) is None and len(lookups) == 1
    clock.t += 31
    assert sl.check(AGENT)["household"] == HOUSE and len(lookups) == 2
    clock.t += 3600
    assert sl.check(AGENT)["household"] == HOUSE and len(lookups) == 3


def test_a_rotated_agent_is_checked_against_its_new_owner():
    """Agent X moves from household A to household B, then the DAO
    sanctions B: within a minute the check names B (Decision 2026-11-10)."""
    clock = Clock()
    owners = {AGENT: OTHER}
    sanctions = {}
    sl = SanctionList(lambda: dict(sanctions), owners.get, clock=clock)
    assert sl.check(AGENT) is None
    owners[AGENT] = HOUSE                       # rotated to HOUSE
    sanctions[f"sanction.{HOUSE}"] = "{}"       # then HOUSE is sanctioned
    clock.t += 61
    assert sl.check(AGENT)["household"] == HOUSE


def _binding(agent, owner, prev, block, index=0):
    return {"agent": agent, "owner": owner, "prev_owner": prev,
            "block": block, "index": index}


def test_owner_map_follows_bindings_incrementally():
    """One scan from the deploy block, then only the blocks since: a
    rotation shows the new owner and keeps the old one in the history."""
    from atn.sanctions import OwnerMap
    zero = "0x" + "00" * 20
    clock = Clock()
    reads = []
    chain = [_binding(AGENT, OTHER, zero, 10)]

    def read(from_block):
        reads.append(from_block)
        lo = from_block if from_block is not None else 0
        return [b for b in chain if b["block"] >= lo], 20 if len(reads) == 1 else 40

    om = OwnerMap(read, clock=clock)
    sl = SanctionList(lambda: {f"sanction.{OTHER}": "{}"}, owners=om, clock=clock)
    assert sl.check(AGENT)["household"] == OTHER
    chain.append(_binding(AGENT, HOUSE, OTHER, 30))
    clock.t += 30
    assert om.owner(AGENT) == OTHER             # not read again yet
    clock.t += 31
    assert om.owner(AGENT) == HOUSE
    assert om.past(AGENT) == [OTHER]
    assert reads == [None, 21]                  # deploy block, then from 21
    # Rotating away from a sanctioned household does not lift it.
    assert sl.check(AGENT)["household"] == OTHER


def test_owner_map_keeps_its_map_when_a_read_fails():
    from atn.sanctions import OwnerMap
    zero = "0x" + "00" * 20
    clock = Clock()
    state = {"fail": False}

    def read(from_block):
        if state["fail"]:
            raise ConnectionError("rpc down")
        return [_binding(AGENT, HOUSE, zero, 5)], 9

    om = OwnerMap(read, clock=clock)
    assert om.owner(AGENT) == HOUSE
    state["fail"] = True
    clock.t += 61
    assert om.owner(AGENT) == HOUSE


def test_no_chain_means_nothing_is_sanctioned():
    sl = SanctionList.from_config(SimpleNamespace(rpc_url="", dao_registry_address=""))
    assert sl.entries() == {} and sl.check(HOUSE) is None


def test_refusal_names_the_household_end_and_reason():
    out = refusal(AGENT, {"household": HOUSE, "until": 86400 * 365, "reason": "trial-1"},
                  "buy services from")
    assert HOUSE in out["error"] and "1971-01-01" in out["error"]
    assert "trial-1" in out["error"] and "buy services from" in out["error"]


def _runtime(sl):
    return SimpleNamespace(autonet=SimpleNamespace(sanctions=sl))


def test_agent_cannot_buy_from_a_sanctioned_household(monkeypatch):
    import atn.agent_tools as at
    from atn.agent_tools import _open_credits, _open_session, _request_service
    monkeypatch.setattr(at, "_buyer", lambda rt, cid: ("0x" + "b0" * 20, "key"))
    sl = _list({f"sanction.{HOUSE}": "{}"}, owners={AGENT: HOUSE})
    rt = _runtime(sl)
    ref = {"provider_address": AGENT, "spec_digest": "ab" * 32}
    bought = asyncio.run(_request_service(rt, {**ref, "payload": {}}))
    assert "sanctioned" in bought["error"]
    session = asyncio.run(_open_session(rt, ref))
    assert "sanctioned" in session["error"]
    credits = asyncio.run(_open_credits(rt, {"provider_address": AGENT, "deposit": 10}))
    assert "sanctioned" in credits["error"]


def test_adoption_refused_for_agents_and_flagged_for_the_owner(tmp_path):
    from atn.tool_store import OWNER_AUTHOR, ToolStore

    digest = "ab" * 32
    manifest = {"kind": "tool_manifest", "name": "t", "trust_class": "pinned",
                "author": AGENT, "description": "d"}

    store = ToolStore.__new__(ToolStore)
    store._records, store._proposals = {}, {}
    store.sanction_check = _list({f"sanction.{HOUSE}": "{}"},
                                 owners={AGENT: HOUSE}).check
    store._save_proposals = lambda: None
    store._verify_manifest_sig = lambda payload: False

    async def fetch(_d):
        return manifest
    store.fetch_payload = fetch

    import nodes.common.library.tool_manifest as tm
    orig = tm.is_tool_manifest
    tm.is_tool_manifest = lambda p: True
    try:
        refused = asyncio.run(store.propose_adoption("agent-1", digest, reason="r"))
        assert "sanctioned" in refused["error"]
        assert digest not in store._proposals
        owned = asyncio.run(store.propose_adoption(None, digest, reason="r"))
        assert owned["status"] == "pending"
        assert store._proposals[digest]["sanction"]["household"] == HOUSE
        assert store._proposals[digest]["proposed_by"] == OWNER_AUTHOR
    finally:
        tm.is_tool_manifest = orig


def test_rotating_an_agent_away_does_not_lift_its_sanction():
    """Survey 2026-10-26: an agent rotated from a sanctioned household to a
    fresh wallet is still sanctioned."""
    fresh = "0x" + "dd" * 20
    sl = SanctionList(lambda: {f"sanction.{HOUSE}": "trial-1"},
                      {AGENT: fresh}.get, clock=Clock(),
                      past_owners=lambda a: [HOUSE] if a == AGENT else [])
    entry = sl.check(AGENT)
    assert entry is not None and entry["household"] == HOUSE
    assert sl.check(OTHER) is None


def test_the_service_inference_provider_refuses_a_sanctioned_provider():
    import pytest

    from atn.providers.service import ProviderError, ServiceProvider

    prov = ServiceProvider.__new__(ServiceProvider)
    prov._provider_address = AGENT
    prov._sanctions = SanctionList(lambda: {f"sanction.{HOUSE}": "x"},
                                   {AGENT: HOUSE}.get, clock=Clock())
    with pytest.raises(ProviderError, match="sanctioned"):
        asyncio.run(prov._refuse_if_sanctioned())
    prov._provider_address = OTHER
    asyncio.run(prov._refuse_if_sanctioned())


def test_owner_logs_scan_from_the_deploy_block_and_split_capped_ranges(monkeypatch):
    """The owner map's one scan of OwnerBound logs starts at the configured
    deploy block and splits the range when the RPC caps eth_getLogs spans,
    like the close does; later reads start where the last one ended."""
    import web3

    from atn.sanctions import _chain_owner_logs

    calls = []

    class _Event:
        @staticmethod
        def get_logs(from_block=None, to_block=None, **_kw):
            calls.append((from_block, to_block))
            if to_block - from_block > 1000:
                raise ValueError("block range too large")
            if from_block <= 4321 <= to_block:
                return [{"args": {"agent": AGENT, "owner": OTHER,
                                  "prevOwner": HOUSE},
                         "blockNumber": 4321, "logIndex": 2}]
            return []

    class _W3:
        def __init__(self, _provider):
            self.eth = SimpleNamespace(
                block_number=5000,
                contract=lambda **kw: SimpleNamespace(
                    events=SimpleNamespace(OwnerBound=_Event)))

        HTTPProvider = staticmethod(lambda *a, **kw: object())
        to_checksum_address = staticmethod(lambda a: a)

    monkeypatch.setattr(web3, "Web3", _W3)
    read = _chain_owner_logs("http://rpc", OTHER, deploy_block=1000)
    bindings, last = read(None)
    assert bindings == [{"agent": AGENT, "owner": OTHER, "prev_owner": HOUSE,
                         "block": 4321, "index": 2}]
    assert last == 5000
    assert min(lo for lo, _ in calls) == 1000
    assert any(hi - lo <= 1000 for lo, hi in calls)
    calls.clear()
    assert read(5001) == ([], 5000)       # nothing new: no log read at all
    assert calls == []


def test_a_failed_owner_history_scan_is_not_retried_on_every_check():
    clock = Clock(1000)
    scans = []

    def past(_a):
        scans.append(1)
        raise RuntimeError("rpc down")

    sl = SanctionList(lambda: {f"sanction.{HOUSE}": "x"}, {}.get,
                      clock=clock, past_owners=past)
    assert sl.check(AGENT) is None
    assert sl.check(AGENT) is None
    assert len(scans) == 1
    clock.t += 61
    sl.check(AGENT)
    assert len(scans) == 2
