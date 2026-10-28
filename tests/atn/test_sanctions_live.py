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


def test_agent_cannot_pay_or_buy_from_a_sanctioned_household(monkeypatch):
    import atn.agent_tools as at
    from atn.agent_tools import _pay_for_service, _request_service
    monkeypatch.setattr(at, "_caller_identity", lambda rt, cid: (None, None, "key"))
    sl = _list({f"sanction.{HOUSE}": "{}"}, owners={AGENT: HOUSE})
    rt = _runtime(sl)
    paid = asyncio.run(_pay_for_service(rt, {"recipient": AGENT, "amount": 1}))
    assert "sanctioned" in paid["error"]
    bought = asyncio.run(_request_service(rt, {
        "provider_address": AGENT, "service_id": "s", "payload": {},
        "tx_hash": "0x1", "request_id": "r"}))
    assert "sanctioned" in bought["error"]


def test_adoption_refused_for_agents_and_flagged_for_the_owner(tmp_path):
    from atn.tool_store import OWNER_AUTHOR, ToolStore

    digest = "ab" * 32
    manifest = {"kind": "tool_manifest", "name": "t", "trust_class": "pinned",
                "author": AGENT, "description": "d"}

    store = ToolStore.__new__(ToolStore)
    store._records, store._proposals = {}, {}
    store.vet_status_provider = None
    store.sanction_check = _list({f"sanction.{HOUSE}": "{}"},
                                 owners={AGENT: HOUSE}).check
    store._save_proposals = lambda: None
    store._verify_manifest_sig = lambda payload: False

    async def fetch(_d):
        return manifest
    store.fetch_payload = fetch

    import nodes.common.world_model_substrate.tool_manifest as tm
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


def test_owner_history_scan_splits_capped_log_ranges(monkeypatch):
    """Issue #40: the OwnerBound scan starts at the deploy block and splits
    the range when the RPC caps eth_getLogs spans, like the close does."""
    import web3

    from atn.sanctions import _chain_past_owners

    calls = []

    class _Event:
        @staticmethod
        def get_logs(argument_filters=None, from_block=None, to_block=None):
            calls.append((from_block, to_block, dict(argument_filters or {})))
            if to_block - from_block > 1000:
                raise ValueError("block range too large")
            if from_block <= 4321 <= to_block:
                return [{"args": {"prevOwner": HOUSE}}]
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
    past = _chain_past_owners("http://rpc", OTHER, deploy_block=1000)
    assert past(AGENT) == [HOUSE]
    assert min(lo for lo, _, _ in calls) == 1000
    assert all(f == {"agent": AGENT} for _, _, f in calls)
    assert any(hi - lo <= 1000 for lo, hi, _ in calls)


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
