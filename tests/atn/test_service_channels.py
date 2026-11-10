"""Buyer side of the payment channel (issue #45): one channel per provider,
a voucher per request, committed only when the request may have reached the
provider, and the unspent deposit withdrawn after the window."""

from __future__ import annotations

import pytest

from atn.service_channels import (
    CLIENT_CLOSING,
    CLOSING,
    SETTLED,
    ChannelBook,
    close_channel,
    settle_voucher,
    sweep_refunds,
    voucher_for,
)

PAYER, PROVIDER = "0xPayer", "0xProvider"


class FakeMarket:
    def __init__(self):
        self.opened, self.closed, self.withdrawn = [], [], []
        self.state = {}

    async def open_channel(self, key, provider, deposit):
        cid = len(self.opened) + 1
        self.opened.append((provider, deposit))
        self.state[cid] = {"status": 1, "closes_at": 0}
        return {"success": True, "channel_id": cid}

    async def start_close(self, key, cid):
        self.closed.append(cid)
        self.state[cid] = {"status": CLIENT_CLOSING, "closes_at": 0}
        return {"success": True, "tx_hash": "0xc"}

    async def withdraw_remainder(self, key, cid):
        self.withdrawn.append(cid)
        self.state[cid] = {"status": SETTLED, "closes_at": 0}
        return {"success": True}

    async def get_channel(self, cid):
        return self.state.get(cid)

    def sign_voucher(self, key, cid, cumulative):
        return f"sig:{cid}:{cumulative}"


@pytest.mark.asyncio
async def test_one_channel_many_requests(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v1 = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    assert m.opened == [(PROVIDER, 1000)]          # 10 requests' worth
    assert v1 == {"channel_id": 1, "cumulative_amount": 100, "signature": "sig:1:100"}
    book.commit(PAYER, PROVIDER, 1, 100)
    v2 = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    assert v2["cumulative_amount"] == 200 and len(m.opened) == 1
    # Survives a restart.
    assert ChannelBook(tmp_path / "ch.json").get(PAYER, PROVIDER)["cumulative"] == 100


@pytest.mark.asyncio
async def test_a_voucher_signs_for_n_units_in_one_step(tmp_path):
    """Metered work (Decision 2026-11-10): the running total grows by
    units x price in one signature, for a stream that meters several work
    items inside one interaction."""
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100, units=3)
    assert v["cumulative_amount"] == 300 and m.opened == [(PROVIDER, 1000)]
    book.commit(PAYER, PROVIDER, 1, 300)
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100, units=7)
    assert v == {"channel_id": 1, "cumulative_amount": 1000, "signature": "sig:1:1000"}
    book.commit(PAYER, PROVIDER, 1, 1000)
    # Spent: the next step opens a new channel.
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100, units=2)
    assert v["channel_id"] == 2 and v["cumulative_amount"] == 200
    assert m.closed == [1]
    assert "error" in await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100,
                                        units=0)


@pytest.mark.asyncio
async def test_a_new_channel_covers_a_step_larger_than_the_default(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100, units=25)
    assert m.opened == [(PROVIDER, 2500)] and v["cumulative_amount"] == 2500
    # A named deposit is used when it covers the step.
    book2, m2 = ChannelBook(tmp_path / "ch2.json"), FakeMarket()
    await voucher_for(book2, m2, "k", PAYER, PROVIDER, ask=100, deposit=400,
                      units=4)
    assert m2.opened == [(PROVIDER, 400)]


@pytest.mark.asyncio
async def test_undelivered_request_spends_nothing(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    # Never delivered: not committed, so the next voucher reuses the amount.
    again = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    assert again["cumulative_amount"] == v["cumulative_amount"] == 100


@pytest.mark.asyncio
async def test_spent_channel_is_closed_and_replaced(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100, deposit=150)
    book.commit(PAYER, PROVIDER, 1, 100)
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    assert m.closed == [1] and v["channel_id"] == 2 and v["cumulative_amount"] == 100
    assert "1" in book.closing()
    out = await sweep_refunds(book, m, lambda payer: "k")
    assert m.withdrawn == [1] and out[0]["action"] == "withdrew"
    assert book.closing() == {}


@pytest.mark.asyncio
async def test_close_channel(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    assert "error" in await close_channel(book, m, "k", PAYER, PROVIDER)
    await voucher_for(book, m, "k", PAYER, PROVIDER, ask=5)
    res = await close_channel(book, m, "k", PAYER, PROVIDER)
    assert res["ok"] and book.get(PAYER, PROVIDER) is None and "1" in book.closing()


@pytest.mark.asyncio
async def test_open_failure_is_reported(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()

    async def broke(key, provider, deposit):
        return {"success": False, "error": "insufficient balance"}
    m.open_channel = broke
    out = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=5)
    assert "insufficient balance" in out["error"] and book.get(PAYER, PROVIDER) is None


# ---------------------------------------------------------------------------
# Counting a voucher: only what the provider may hold (never behind it)
# ---------------------------------------------------------------------------

async def _sent(book, m, reply):
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    book.reserve(PAYER, PROVIDER, v["channel_id"], v["cumulative_amount"])
    return settle_voucher(book, PAYER, PROVIDER, v, reply)


@pytest.mark.asyncio
async def test_served_request_counts(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    assert await _sent(book, m, {"delivered": True, "ok": True, "charged": True})
    assert book.get(PAYER, PROVIDER)["cumulative"] == 100


@pytest.mark.asyncio
async def test_refused_or_failed_request_does_not_count(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    assert not await _sent(book, m, {"delivered": True, "ok": False, "charged": False})
    ch = book.get(PAYER, PROVIDER)
    assert ch["cumulative"] == 0 and "pending" not in ch
    # The next request signs the same total again.
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    assert v["cumulative_amount"] == 100


@pytest.mark.asyncio
async def test_unanswered_request_that_may_have_arrived_counts(tmp_path):
    # A timeout after sending, or an error from a relay (no word from the
    # provider): it may have served it and kept the voucher.
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    assert await _sent(book, m, {"delivered": True, "error": "no reply within 300s"})
    assert await _sent(book, m, {"delivered": True, "ok": False,
                                 "error": "the provider did not reply through the relay"})
    assert book.get(PAYER, PROVIDER)["cumulative"] == 200


@pytest.mark.asyncio
async def test_undelivered_request_does_not_count(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    assert not await _sent(book, m, {"error": "no path", "delivered": False})
    assert book.get(PAYER, PROVIDER)["cumulative"] == 0


@pytest.mark.asyncio
async def test_a_voucher_out_when_the_daemon_stopped_counts_at_start(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    book.reserve(PAYER, PROVIDER, v["channel_id"], v["cumulative_amount"])
    # Restart before the reply: the provider may hold it.
    again = ChannelBook(tmp_path / "ch.json")
    ch = again.get(PAYER, PROVIDER)
    assert ch["cumulative"] == 100 and "pending" not in ch


@pytest.mark.asyncio
async def test_a_channel_the_provider_closed_is_replaced(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    book.commit(PAYER, PROVIDER, v["channel_id"], 100)
    m.state[1] = {"status": CLOSING, "closes_at": 0}      # provider collected
    v2 = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    assert v2["channel_id"] == 2 and v2["cumulative_amount"] == 100
    assert m.closed == []                       # no startClose on a closed channel
    assert "1" in book.closing()
    out = await sweep_refunds(book, m, lambda payer: "k")
    assert m.withdrawn == [1] and out[0]["action"] == "withdrew"


@pytest.mark.asyncio
async def test_sweep_notices_a_provider_close_and_refunds_after_the_window(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)
    book.commit(PAYER, PROVIDER, v["channel_id"], 100)
    m.state[1] = {"status": CLOSING, "closes_at": 500}
    out = await sweep_refunds(book, m, lambda payer: "k", now=100)
    assert {"channel_id": "1", "action": "closed_by_provider"} in out
    assert book.get(PAYER, PROVIDER) is None and m.withdrawn == []
    out = await sweep_refunds(book, m, lambda payer: "k", now=600)
    assert m.withdrawn == [1] and book.closing() == {}


@pytest.mark.asyncio
async def test_failed_start_close_is_sent_again_by_the_sweep(tmp_path):
    book, m = ChannelBook(tmp_path / "ch.json"), FakeMarket()
    await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100, deposit=150)
    book.commit(PAYER, PROVIDER, 1, 100)
    real_close = m.start_close

    async def fail(key, cid):
        return {"success": False, "error": "rpc down"}
    m.start_close = fail
    v = await voucher_for(book, m, "k", PAYER, PROVIDER, ask=100)   # spent: replace
    assert v["channel_id"] == 2 and "1" in book.closing()
    assert m.state[1]["status"] == 1                   # still Open on chain
    m.start_close = real_close
    out = await sweep_refunds(book, m, lambda payer: "k", now=0)
    assert {"channel_id": "1", "action": "close_started", "error": None} in out
    assert m.closed == [1]
    out = await sweep_refunds(book, m, lambda payer: "k", now=0)
    assert m.withdrawn == [1] and book.closing() == {}


# ---------------------------------------------------------------------------
# The request_service agent tool: paid only for what was served
# ---------------------------------------------------------------------------

DIGEST = "ab" * 32


def _buyer(tmp_path, monkeypatch, replies, entry=None):
    from types import SimpleNamespace

    from atn import agent_tools, on_chain, service_client

    market = FakeMarket()

    class Market(FakeMarket):
        channel_available = True

        def __init__(self, cfg):
            self.__dict__ = market.__dict__

    async def ask(cfg, provider, digest):
        return {"active": True, "ask_amount": 100}

    async def read(cfg, number):
        return dict(entry) if entry else {"error": f"no on-chain service #{number}"}

    sent = []

    async def request(cfg, provider, digest, payload, **kw):
        sent.append({"provider": provider, "digest": digest, **kw})
        return dict(replies.pop(0))

    async def clear(runtime, address, what):
        return None

    monkeypatch.setattr(on_chain, "ServiceMarketClient", Market)
    monkeypatch.setattr(service_client, "lookup_service_ask", ask)
    monkeypatch.setattr(service_client, "read_service", read)
    monkeypatch.setattr(service_client, "request_service", request)
    monkeypatch.setattr(agent_tools, "_sanctioned", clear)
    identity = SimpleNamespace(address=PAYER)
    rt = SimpleNamespace(
        _config=SimpleNamespace(rpb=SimpleNamespace()),
        get_agent=lambda aid: SimpleNamespace(identity=identity),
        registry=SimpleNamespace(get_agent_key=lambda aid: "k"),
        channel_book=ChannelBook(tmp_path / "ch.json"))
    return rt, sent


async def _call(rt, **args):
    from atn.agent_tools import _request_service
    return await _request_service(rt, {"payload": {}, "_caller_id": "agent-1", **args})


@pytest.mark.asyncio
async def test_request_refused_by_the_provider_is_not_paid(tmp_path, monkeypatch):
    rt, sent = _buyer(tmp_path, monkeypatch, [
        {"delivered": True, "ok": False, "charged": False, "error": "backing failed"},
        {"delivered": True, "ok": True, "charged": True, "result": {}}])
    failed = await _call(rt, provider_address=PROVIDER, spec_digest=DIGEST)
    assert failed["paid"] == 0
    assert rt.channel_book.get(PAYER, PROVIDER)["cumulative"] == 0
    served = await _call(rt, provider_address=PROVIDER, spec_digest=DIGEST)
    assert served["paid"] == 100
    # The failed request's total was signed again for the served one.
    assert [s["voucher"]["cumulative_amount"] for s in sent] == [100, 100]
    assert rt.channel_book.get(PAYER, PROVIDER)["cumulative"] == 100


@pytest.mark.asyncio
async def test_unanswered_request_is_counted_as_paid(tmp_path, monkeypatch):
    rt, _ = _buyer(tmp_path, monkeypatch, [
        {"delivered": True, "error": "Provider did not reply within 300s."}])
    out = await _call(rt, provider_address=PROVIDER, spec_digest=DIGEST)
    assert out["paid"] == 100
    assert rt.channel_book.get(PAYER, PROVIDER)["cumulative"] == 100


@pytest.mark.asyncio
async def test_a_numeric_service_id_is_read_from_the_chain(tmp_path, monkeypatch):
    entry = {"service_id": 4, "provider": PROVIDER, "spec_digest": "0x" + DIGEST.upper(),
             "ask_amount": 250, "active": True}
    rt, sent = _buyer(tmp_path, monkeypatch,
                      [{"delivered": True, "ok": True, "charged": True}], entry=entry)
    out = await _call(rt, service_id=4)
    assert out["paid"] == 250
    assert sent[0]["provider"] == PROVIDER and sent[0]["digest"] == DIGEST
    assert sent[0]["voucher"]["cumulative_amount"] == 250


@pytest.mark.asyncio
async def test_service_id_naming_errors(tmp_path, monkeypatch):
    entry = {"service_id": 4, "provider": PROVIDER, "spec_digest": DIGEST,
             "ask_amount": 250, "active": True}
    rt, sent = _buyer(tmp_path, monkeypatch, [], entry=entry)
    assert "spec_digest" in (await _call(rt, service_id=DIGEST))["error"]
    assert "provided by" in (await _call(rt, service_id=4,
                                         provider_address="0x" + "99" * 20))["error"]
    assert "another spec" in (await _call(rt, service_id="4",
                                          spec_digest="cd" * 32))["error"]
    assert "Name the service" in (await _call(rt, provider_address=PROVIDER))["error"]
    assert "provider_address" in (await _call(rt, spec_digest=DIGEST))["error"]
    assert sent == []


def test_market_client_methods_run_off_the_event_loop():
    """Every async ServiceMarketClient method runs on a worker thread
    (on_chain._off_the_loop), so a chain call never blocks the loop."""
    import asyncio
    import threading

    from atn.config import RPBConfig
    from atn.on_chain import ServiceMarketClient

    seen = []

    def fake_web3(self):
        seen.append(threading.get_ident())
        raise ConnectionError("no chain in this test")

    smc = ServiceMarketClient(RPBConfig(rpc_url="http://x", payment_channel_address="0x1"))
    smc._get_web3 = fake_web3.__get__(smc)

    async def main():
        assert await smc.get_channel(1) is None
        return threading.get_ident()
    loop_thread = asyncio.run(main())
    assert seen and seen[0] != loop_thread


@pytest.mark.asyncio
async def test_request_on_channel_counts_only_what_the_provider_kept(tmp_path, monkeypatch):
    """The shared buyer path (agents, the `service` model provider, the
    owner's purchase in the app): a refused request costs nothing, a served
    one costs its price, and the next voucher follows the book."""
    from atn import service_client
    from atn.service_channels import request_on_channel

    book = ChannelBook(tmp_path / "channels.json")
    m = FakeMarket()
    answers = [{"ok": False, "delivered": True, "charged": False},
               {"ok": True, "delivered": True, "charged": True}]
    sent = []

    async def fake(config, provider, digest, args, **kw):
        sent.append(kw["voucher"]["cumulative_amount"])
        return dict(answers.pop(0))
    monkeypatch.setattr(service_client, "request_service", fake)

    refused = await request_on_channel(book, m, None, "k", PAYER, PROVIDER,
                                       "d" * 64, {}, 100)
    assert refused["paid"] == 0 and refused["request_id"]
    served = await request_on_channel(book, m, None, "k", PAYER, PROVIDER,
                                      "d" * 64, {}, 100)
    assert served["paid"] == 100
    # The refused voucher was dropped, so the served one signed the same total.
    assert sent == [100, 100]
    assert int(book.get(PAYER, PROVIDER)["cumulative"]) == 100
