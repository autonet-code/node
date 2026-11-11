"""Provider-side channel settlement (atn/channel_settlement.py): a channel
is settled up to what was served (vouchers pay ahead, and the unserved
rest stays the buyer's); an open one (it stays open) once its unpaid
amount reaches the threshold, or once it has had unpaid value for the
settle period since its last settle; nothing is sent when there is nothing
new to pay; a channel whose client started a close is settled inside the
window (so the client's refund keeps its deadline); a closed or settled one
is forgotten, and so is one whose client-close window has passed (closing
it would only revert)."""
from __future__ import annotations

import asyncio
import time

from atn.channel_settlement import (
    CLIENT_CLOSING, CLOSING, OPEN, SETTLED, SettlementLog, settle_channels,
)

PROV = "0x" + "aa" * 20
NOW = 1_000_000.0
ATN = 1_000_000


class Store:
    def __init__(self, vouchers, pending=()):
        self.v = dict(vouchers)
        self.pending = set(pending)

    def channel_vouchers(self):
        return {k: dict(x) for k, x in self.v.items()}

    def forget_channel_voucher(self, cid):
        self.v.pop(str(cid), None)

    def has_pending_voucher(self, cid):
        return str(cid) in self.pending


class Client:
    """A PaymentChannel stand-in: settle and close move ``claimed`` like the
    contract (min(cumulative, deposit)), and settle with nothing new fails
    as the estimate would."""

    def __init__(self, channels):
        self.channels = {}
        for cid, ch in channels.items():
            if isinstance(ch, int):
                ch = {"status": ch}
            self.channels[cid] = {"provider": PROV, "deposit": str(10 * ATN),
                                  "claimed": "0", "closes_at": NOW + 60, **ch}
        self.settled, self.closed, self.up_to = [], [], []

    async def get_channel(self, cid):
        ch = self.channels.get(cid)
        return None if ch is None else dict(ch)

    def _capped(self, cid, cumulative):
        return min(int(cumulative), int(self.channels[cid]["deposit"]))

    async def settle_channel(self, key, cid, cumulative, signature, up_to=None):
        capped = self._capped(cid, cumulative)
        if up_to is not None:
            capped = min(capped, int(up_to))
        if capped <= int(self.channels[cid]["claimed"]):
            return {"success": False,
                    "error": "not sent, it would revert: NothingToSettle",
                    "reverted": True}
        self.settled.append((key, cid, cumulative, signature))
        self.up_to.append(up_to)
        self.channels[cid]["claimed"] = str(capped)
        return {"success": True}

    async def close_channel(self, key, cid, cumulative, signature):
        self.closed.append((key, cid, cumulative, signature))
        ch = self.channels[cid]
        ch["claimed"] = str(max(int(ch["claimed"]), self._capped(cid, cumulative)))
        ch["status"] = CLOSING
        return {"success": True}


def _key(p):
    return "0xkey" if p == PROV else None


def _sweep(store, client, ledger=None, **kw):
    kw.setdefault("now", NOW)
    return asyncio.run(settle_channels(store, client, _key, ledger=ledger, **kw))


def _v(cumulative, sig="0xs1", ts=NOW, consumed=None):
    """A held voucher; everything it covers served unless ``consumed``
    says otherwise."""
    return {"cumulative": cumulative, "signature": sig, "ts": ts,
            "consumed": cumulative if consumed is None else consumed}


def test_settles_at_the_threshold_and_the_channel_stays_open():
    store = Store({"1": _v(ATN)})
    client = Client({1: OPEN})
    out = _sweep(store, client)
    assert client.settled == [("0xkey", 1, ATN, "0xs1")] and client.closed == []
    assert out == [{"channel_id": "1", "action": "settled", "reason": "threshold",
                    "amount": ATN, "ok": True, "error": None}]
    # The voucher is kept (the gate's floor) and the channel is still open.
    assert "1" in store.v and client.channels[1]["status"] == OPEN


def test_below_the_threshold_waits_for_the_period():
    ledger = SettlementLog()
    store = Store({"1": _v(ATN - 1)})
    client = Client({1: OPEN})
    assert _sweep(store, client, ledger, settle_period_s=3600) == []
    # First seen at NOW: the period runs from there.
    assert ledger.get("1") == {"paid": 0, "since": NOW}
    assert _sweep(store, client, ledger, settle_period_s=3600,
                  now=NOW + 3599) == []
    out = _sweep(store, client, ledger, settle_period_s=3600, now=NOW + 3600)
    assert out[0]["reason"] == "period" and out[0]["amount"] == ATN - 1
    assert client.settled == [("0xkey", 1, ATN - 1, "0xs1")]
    assert ledger.get("1") == {"paid": ATN - 1, "since": NOW + 3600}


def test_the_period_restarts_at_each_settle():
    ledger = SettlementLog()
    store = Store({"1": _v(100)})
    client = Client({1: OPEN})
    _sweep(store, client, ledger, settle_period_s=10)
    assert _sweep(store, client, ledger, settle_period_s=10, now=NOW + 10)[0]["ok"]
    # More served since: the next settle waits a whole period again.
    store.v["1"] = _v(250, "0xs2")
    assert _sweep(store, client, ledger, settle_period_s=10, now=NOW + 19) == []
    out = _sweep(store, client, ledger, settle_period_s=10, now=NOW + 20)
    assert out[0]["amount"] == 150 and client.settled[-1] == ("0xkey", 1, 250, "0xs2")


def test_nothing_new_sends_nothing():
    """What the channel paid is read from chain (claimed): a voucher at or
    below it is not sent again, whatever the period."""
    store = Store({"1": _v(3 * ATN)})
    client = Client({1: {"status": OPEN, "claimed": str(3 * ATN)}})
    assert _sweep(store, client, settle_period_s=1, now=NOW + 10 ** 6) == []
    assert client.settled == [] and client.closed == []
    # Two sweeps in a row after one settle: the second sends nothing.
    store = Store({"2": _v(2 * ATN)})
    client = Client({2: OPEN})
    ledger = SettlementLog()
    assert len(_sweep(store, client, ledger)) == 1
    assert _sweep(store, client, ledger) == [] and len(client.settled) == 1


def test_a_lagging_read_of_claimed_does_not_resend():
    """The chain read still shows the old claimed right after a settle: the
    ledger's record of what was paid wins."""
    ledger = SettlementLog()
    store = Store({"1": _v(2 * ATN)})
    client = Client({1: OPEN})
    _sweep(store, client, ledger)
    client.channels[1]["claimed"] = "0"           # a stale RPC
    assert _sweep(store, client, ledger) == [] and len(client.settled) == 1


def test_unsettled_amount_is_capped_at_the_deposit():
    store = Store({"1": _v(50 * ATN)})
    client = Client({1: {"status": OPEN, "deposit": str(5 * ATN),
                         "claimed": str(4 * ATN)}})
    out = _sweep(store, client)
    assert out[0]["amount"] == ATN and client.channels[1]["claimed"] == str(5 * ATN)
    assert _sweep(store, client, settle_period_s=1, now=NOW + 100) == []


def test_only_what_was_served_is_claimed():
    """Vouchers pay ahead: a voucher for 5 ATN with 2 ATN served settles up
    to 2 ATN; the rest stays the buyer's until more is served."""
    ledger = SettlementLog()
    store = Store({"1": _v(5 * ATN, consumed=2 * ATN)})
    client = Client({1: OPEN})
    out = _sweep(store, client, ledger)
    assert out[0]["amount"] == 2 * ATN and client.up_to == [2 * ATN]
    assert client.channels[1]["claimed"] == str(2 * ATN)
    assert _sweep(store, client, ledger, settle_period_s=1, now=NOW + 100) == []
    # Served past the voucher (one reply's overshoot): capped at the voucher.
    store.v["1"] = _v(5 * ATN, consumed=6 * ATN)
    out = _sweep(store, client, ledger)
    assert out[0]["amount"] == 3 * ATN and client.up_to[-1] == 5 * ATN


def test_a_client_close_is_settled_up_to_what_was_served():
    store = Store({"1": _v(1000, consumed=300)})
    client = Client({1: CLIENT_CLOSING})
    out = _sweep(store, client)
    assert out[0]["amount"] == 300 and client.up_to == [300]
    assert client.channels[1]["claimed"] == "300" and store.v == {}
    # Nothing served: nothing to collect, the whole deposit goes back.
    store = Store({"2": _v(1000, consumed=0)})
    client = Client({2: CLIENT_CLOSING})
    assert _sweep(store, client) == [{"channel_id": "2", "action": "forgot"}]
    assert client.settled == []


def test_a_spent_deposit_no_longer_closes_the_channel():
    """The old rule closed a channel whose deposit no longer covered the
    price; the buyer tops up instead, so the provider only settles."""
    store = Store({"1": _v(10 * ATN)})
    client = Client({1: OPEN})                    # deposit 10 ATN, all used
    out = _sweep(store, client)
    assert out[0]["action"] == "settled" and client.closed == []
    assert client.channels[1]["status"] == OPEN


def test_a_payout_not_sent_by_this_sweep_restarts_the_period():
    ledger = SettlementLog()
    ledger.note("1", 0, NOW - 10 ** 6)
    store = Store({"1": _v(500)})
    client = Client({1: {"status": OPEN, "claimed": "200"}})
    assert _sweep(store, client, ledger, settle_period_s=3600) == []
    assert ledger.get("1") == {"paid": 200, "since": NOW}


def test_client_close_is_answered_with_a_settle_inside_the_window():
    store = Store({"1": _v(300), "3": _v(70, "0xs3")})
    client = Client({1: CLIENT_CLOSING, 3: CLOSING})
    out = _sweep(store, client)
    assert client.settled == [("0xkey", 1, 300, "0xs1")] and client.closed == []
    assert store.v == {}
    assert {(o["channel_id"], o["action"]) for o in out} == {
        ("1", "settled"), ("3", "forgot")}
    assert out[0]["reason"] == "client_closing"


def test_client_close_with_everything_paid_is_forgotten_without_a_send():
    store = Store({"1": _v(300)})
    client = Client({1: {"status": CLIENT_CLOSING, "claimed": "300"}})
    out = _sweep(store, client)
    assert client.closed == [] and client.settled == [] and store.v == {}
    assert out == [{"channel_id": "1", "action": "forgot"}]


def test_closed_and_settled_channels_are_forgotten():
    ledger = SettlementLog()
    ledger.note("2", 5, NOW)
    store = Store({"2": _v(5), "4": _v(9)})
    client = Client({2: CLOSING, 4: SETTLED})
    out = _sweep(store, client, ledger)
    assert store.v == {} and ledger.all() == {}
    assert [o["action"] for o in out] == ["forgot", "forgot"]


def test_no_key_leaves_the_voucher_for_a_later_sweep():
    store = Store({"1": _v(300)})
    client = Client({1: CLIENT_CLOSING})
    asyncio.run(settle_channels(store, client, lambda p: None, now=NOW))
    assert client.closed == [] and "1" in store.v
    store = Store({"2": _v(2 * ATN)})
    client = Client({2: OPEN})
    asyncio.run(settle_channels(store, client, lambda p: None, now=NOW))
    assert client.settled == []


def test_a_channel_serving_a_request_is_left_for_the_next_sweep():
    store = Store({"1": _v(5 * ATN)}, pending={"1"})
    client = Client({1: OPEN})
    assert _sweep(store, client, settle_period_s=1, now=NOW + 10 ** 6) == []
    assert client.settled == []


def test_client_close_window_passed_is_forgotten_without_a_send():
    store = Store({"1": _v(300)})
    client = Client({1: {"status": CLIENT_CLOSING, "closes_at": NOW - 1}})
    out = _sweep(store, client)
    assert client.closed == [] and store.v == {}
    assert out == [{"channel_id": "1", "action": "window_passed"}]


def test_a_failed_settle_keeps_the_voucher_and_the_period():
    ledger = SettlementLog()
    store = Store({"1": _v(2 * ATN)})
    client = Client({1: OPEN})

    async def fail(key, cid, cumulative, signature, up_to=None):
        return {"success": False, "error": "not sent, it would revert: x"}
    client.settle_channel = fail
    out = _sweep(store, client, ledger)
    assert out[0]["ok"] is False and "1" in store.v
    assert ledger.get("1") == {"paid": 0, "since": NOW}


def test_a_failed_settle_of_a_closing_channel_keeps_the_voucher():
    store = Store({"1": _v(300)})
    client = Client({1: CLIENT_CLOSING})

    async def fail(key, cid, cumulative, signature, up_to=None):
        return {"success": False, "error": "not sent, it would revert: x"}
    client.settle_channel = fail
    out = _sweep(store, client)
    assert out[0]["ok"] is False and "1" in store.v


def test_settlement_log_survives_a_restart_and_drops_forgotten_channels(tmp_path):
    path = tmp_path / "services" / "channel_settlements.json"
    ledger = SettlementLog(path)
    store = Store({"1": _v(100)})
    client = Client({1: OPEN})
    _sweep(store, client, ledger)
    assert SettlementLog(path).get("1") == {"paid": 0, "since": NOW}
    # The period kept across the restart.
    out = _sweep(store, client, SettlementLog(path), settle_period_s=50,
                 now=NOW + 50)
    assert out[0]["reason"] == "period"
    store.v.clear()
    _sweep(store, client, SettlementLog(path))
    assert SettlementLog(path).all() == {}


def _runtime(tmp_path):
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime

    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir(parents=True)
    agents_dir.mkdir(parents=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.rpc_url = "http://127.0.0.1:8545"
    config.autonet.payment_channel_address = "0x" + "97" * 20
    config.autonet.private_key = ""            # a normal install
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def test_daemon_sweep_settles_with_the_operator_key(tmp_path, monkeypatch):
    """A normal install (no configured daemon key): the provider of an owner
    or external-agent service is the operator, and the daemon's sweep
    settles its channel with the vault-held operator key: below the
    configured threshold only once the configured period has passed since
    the sweep first saw the channel, at the threshold at once."""
    from eth_account import Account

    import atn.on_chain as oc

    rt = _runtime(tmp_path)
    rt._config.rpb.channel_idle_close_s = 60
    rt._config.rpb.channel_settle_min = 5000
    key = rt.registry.daemon_operator_key()
    operator = Account.from_key(key).address
    assert rt._channel_key_for(operator) == key
    assert rt._channel_key_for("0x" + "99" * 20) is None
    settled = []
    claimed = {"3": 0}

    class Market:
        channel_available = True

        def __init__(self, cfg):
            pass

        def _channel_contract(self):
            raise ConnectionError("no window read in this test")

        async def get_channel(self, cid):
            return {"status": OPEN, "provider": operator, "deposit": "10000",
                    "claimed": str(claimed[str(cid)]), "closes_at": 0}

        async def settle_channel(self, k, cid, cumulative, signature, up_to=None):
            settled.append((k, cid, cumulative, up_to))
            claimed[str(cid)] = min(cumulative, up_to)
            return {"success": True}

    monkeypatch.setattr(oc, "ServiceMarketClient", Market)
    store = rt.service_store
    client = "0x" + "12" * 20
    store.accept_voucher(3, 5000, "0xs", client=client, provider=operator)
    store.consume(3, 1000)
    out = asyncio.run(rt.settle_channels_once())
    assert settled == [] and out["settled"] == []          # just seen
    ledger_file = tmp_path / "data" / "services" / "channel_settlements.json"
    assert ledger_file.exists()
    rt._channel_settlements.note("3", 0, time.time() - 61)
    out = asyncio.run(rt.settle_channels_once())
    # Up to what was served, not the whole voucher paid ahead.
    assert settled == [(key, 3, 5000, 1000)] and out["settled"][0]["reason"] == "period"
    # Kept: the channel stays open and its voucher is the gate's floor.
    assert "3" in store.channel_vouchers()
    # 5000 more served on a new voucher: at the threshold, settled at once.
    store.accept_voucher(3, 10000, "0xs2", client=client, provider=operator)
    store.consume(3, 5000)
    out = asyncio.run(rt.settle_channels_once())
    assert settled[-1] == (key, 3, 10000, 6000) and out["settled"][0]["reason"] == "threshold"
    assert asyncio.run(rt.settle_channels_once())["settled"] == []
