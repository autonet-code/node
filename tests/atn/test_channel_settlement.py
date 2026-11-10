"""Provider-side channel settlement (atn/channel_settlement.py): a channel
whose client started a close is settled with the stored best voucher, and so
is an open one whose deposit no longer covers the price or that has gone
idle; a closed or settled one is forgotten, and so is one whose client-close
window has passed (closing it would only revert)."""
from __future__ import annotations

import asyncio

from atn.channel_settlement import (
    CLIENT_CLOSING, CLOSING, OPEN, settle_channels,
)

PROV = "0x" + "aa" * 20
NOW = 1_000_000.0


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
    def __init__(self, channels, prices=None):
        self.channels = channels
        self.prices = prices or {}
        self.closed = []

    async def get_channel(self, cid):
        ch = self.channels.get(cid)
        if ch is None:
            return None
        if isinstance(ch, int):
            ch = {"status": ch}
        return {"provider": PROV, "deposit": "1000", "closes_at": NOW + 60, **ch}

    async def get_service(self, sid):
        price = self.prices.get(sid)
        return None if price is None else {"ask_amount": str(price)}

    async def close_channel(self, key, cid, cumulative, signature):
        self.closed.append((key, cid, cumulative, signature))
        return {"success": True}


def _key(p):
    return "0xkey" if p == PROV else None


def _sweep(store, client, **kw):
    kw.setdefault("now", NOW)
    return asyncio.run(settle_channels(store, client, _key, **kw))


def test_settles_client_closes_and_forgets_finished_channels():
    store = Store({"1": {"cumulative": 300, "signature": "0xs1", "ts": NOW},
                   "2": {"cumulative": 50, "signature": "0xs2", "ts": NOW},
                   "3": {"cumulative": 70, "signature": "0xs3", "ts": NOW}})
    client = Client({1: CLIENT_CLOSING, 2: OPEN, 3: CLOSING})
    out = _sweep(store, client)
    assert client.closed == [("0xkey", 1, 300, "0xs1")]
    assert set(store.v) == {"2"}
    assert {o["channel_id"] for o in out} == {"1", "3"}


def test_no_key_leaves_the_voucher_for_a_later_sweep():
    store = Store({"1": {"cumulative": 300, "signature": "0xs1"}})
    client = Client({1: CLIENT_CLOSING})
    asyncio.run(settle_channels(store, client, lambda p: None, now=NOW))
    assert client.closed == [] and "1" in store.v


def test_provider_closes_when_the_deposit_no_longer_covers_the_price():
    # Deposit 1000, best voucher 900: 100 left, the service costs 150 on
    # chain now (the voucher was taken at 100).
    store = Store({"1": {"cumulative": 900, "signature": "0xs1", "ts": NOW,
                         "service_id": 4, "ask": 100}})
    client = Client({1: OPEN}, prices={4: 150})
    out = _sweep(store, client)
    assert client.closed == [("0xkey", 1, 900, "0xs1")]
    assert out[0]["reason"] == "deposit_spent" and store.v == {}


def test_open_channel_with_deposit_left_and_recent_voucher_stays_open():
    store = Store({"1": {"cumulative": 800, "signature": "0xs1", "ts": NOW - 10,
                         "service_id": 4, "ask": 100}})
    client = Client({1: OPEN}, prices={4: 100})
    assert _sweep(store, client) == [] and client.closed == []


def test_provider_closes_an_idle_channel():
    store = Store({"1": {"cumulative": 200, "signature": "0xs1",
                         "ts": NOW - 3601, "service_id": 4, "ask": 100}})
    client = Client({1: OPEN}, prices={4: 100})
    assert _sweep(store, client, idle_close_s=7200) == []
    out = _sweep(store, client, idle_close_s=3600)
    assert out[0]["reason"] == "idle"
    assert client.closed == [("0xkey", 1, 200, "0xs1")]


def test_unreadable_price_falls_back_to_the_price_charged():
    store = Store({"1": {"cumulative": 950, "signature": "0xs1", "ts": NOW,
                         "service_id": 9, "ask": 100}})
    client = Client({1: OPEN})                 # service 9 unreadable
    assert _sweep(store, client)[0]["reason"] == "deposit_spent"


def test_a_channel_serving_a_request_is_left_for_the_next_sweep():
    store = Store({"1": {"cumulative": 950, "signature": "0xs1",
                         "ts": NOW - 10 ** 6, "ask": 100}}, pending={"1"})
    client = Client({1: OPEN})
    assert _sweep(store, client) == [] and client.closed == []


def test_client_close_window_passed_is_forgotten_without_a_send():
    store = Store({"1": {"cumulative": 300, "signature": "0xs1", "ts": NOW}})
    client = Client({1: {"status": CLIENT_CLOSING, "closes_at": NOW - 1}})
    out = _sweep(store, client)
    assert client.closed == [] and store.v == {}
    assert out == [{"channel_id": "1", "action": "window_passed"}]


def test_a_failed_close_keeps_the_voucher():
    store = Store({"1": {"cumulative": 300, "signature": "0xs1", "ts": NOW}})
    client = Client({1: CLIENT_CLOSING})

    async def fail(key, cid, cumulative, signature):
        return {"success": False, "error": "not sent, it would revert: x"}
    client.close_channel = fail
    out = _sweep(store, client)
    assert out[0]["ok"] is False and "1" in store.v


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


def test_daemon_sweep_closes_with_the_operator_key(tmp_path, monkeypatch):
    """A normal install (no configured daemon key): the provider of an owner
    or external-agent service is the operator, and the daemon's sweep closes
    its idle channel with the vault-held operator key, after the configured
    idle time."""
    from eth_account import Account

    import atn.on_chain as oc

    rt = _runtime(tmp_path)
    rt._config.rpb.channel_idle_close_s = 60
    key = rt.registry.daemon_operator_key()
    operator = Account.from_key(key).address
    assert rt._channel_key_for(operator) == key
    assert rt._channel_key_for("0x" + "99" * 20) is None
    closed = []

    class Market:
        channel_available = True

        def __init__(self, cfg):
            pass

        def _channel_contract(self):
            raise ConnectionError("no window read in this test")

        async def get_channel(self, cid):
            return {"status": OPEN, "provider": operator, "deposit": "10000",
                    "closes_at": 0}

        async def get_service(self, sid):
            return {"ask_amount": "1000"}

        async def close_channel(self, k, cid, cumulative, signature):
            closed.append((k, cid, cumulative))
            return {"success": True}

    monkeypatch.setattr(oc, "ServiceMarketClient", Market)
    rt.service_store.record_channel_voucher(3, 1000, "0xs", service_id=7, ask=1000)
    out = asyncio.run(rt.settle_channels_once())
    assert closed == [] and out["settled"] == []          # used just now
    rt.service_store._channel_vouchers["3"]["ts"] -= 61
    out = asyncio.run(rt.settle_channels_once())
    assert closed == [(key, 3, 1000)] and out["settled"][0]["reason"] == "idle"
    assert rt.service_store.channel_vouchers() == {}
