"""The buyer's credits with service providers (atn/service_channels.py,
Decision 2026-11-12): one payment channel per (buyer, provider), opened and
closed by the Agent; vouchers pay ahead by the provider's voucher size,
never past the deposit; a request carries a voucher only when the credit
the provider holds does not cover its price; the remainder of closed
credits comes back after the challenge window. Top-ups are tested in
test_service_channel_top_up.py."""

from __future__ import annotations

import pytest

import atn.service_client as service_client
from atn.service_channels import (
    CLIENT_CLOSING,
    CLOSING,
    OPEN,
    SETTLED,
    ChannelBook,
    close_credits,
    credits_view,
    next_voucher,
    open_credits,
    request_on_credits,
    sweep_refunds,
)

PAYER, PROVIDER, OTHER = "0xPayer", "0xProvider", "0xOther"
DIGEST = "ab" * 32
PRICE, VOUCHER, DEPOSIT = 1000, 3000, 10_000


class Market:
    """PaymentChannel as the buyer's daemon sees it."""

    def __init__(self):
        self.state: dict[int, dict] = {}
        self.opened, self.closed, self.withdrawn = [], [], []

    async def open_channel(self, key, provider, deposit):
        cid = len(self.opened) + 1
        self.opened.append((provider, deposit))
        self.state[cid] = {"status": OPEN, "closes_at": 0}
        return {"success": True, "channel_id": cid, "tx_hash": "0xo"}

    async def get_channel(self, cid):
        st = self.state.get(cid)
        return None if st is None else dict(st)

    async def start_close(self, key, cid):
        self.closed.append(cid)
        self.state[cid] = {"status": CLIENT_CLOSING, "closes_at": 100}
        return {"success": True, "tx_hash": "0xc"}

    async def withdraw_remainder(self, key, cid):
        self.withdrawn.append(cid)
        self.state[cid] = {"status": SETTLED, "closes_at": 0}
        return {"success": True}

    def sign_voucher(self, key, cid, cumulative):
        return f"sig:{cid}:{cumulative}"


async def _opened(tmp_path, deposit=DEPOSIT):
    book, m = ChannelBook(tmp_path / "ch.json"), Market()
    assert (await open_credits(book, m, "k", PAYER, PROVIDER, deposit))["ok"]
    return book, m


@pytest.mark.asyncio
async def test_open_credits_once_per_provider(tmp_path):
    book, m = await _opened(tmp_path)
    assert book.get(PAYER, PROVIDER)["deposit"] == DEPOSIT
    again = await open_credits(book, m, "k", PAYER, PROVIDER, 500)
    assert "already open" in again["error"] and again["credits"]["deposit"] == DEPOSIT
    assert "deposit above 0" in (await open_credits(book, m, "k", PAYER, OTHER, 0))["error"]
    assert (await open_credits(book, m, "k", PAYER, OTHER, 500))["ok"]
    assert m.opened == [(PROVIDER, DEPOSIT), (OTHER, 500)]
    # Kept across a restart.
    assert ChannelBook(tmp_path / "ch.json").get(PAYER, PROVIDER)["channel_id"] == 1
    # The provider closed them: new credits replace them, the old channel
    # waits for its refund.
    m.state[1] = {"status": CLOSING, "closes_at": 100}
    res = await open_credits(book, m, "k", PAYER, PROVIDER, 700)
    assert res["ok"] and res["credits"]["channel_id"] == 3
    assert "1" in book.closing()


@pytest.mark.asyncio
async def test_a_voucher_pays_ahead_by_the_voucher_size(tmp_path):
    book, m = await _opened(tmp_path)
    v = await next_voucher(book, m, "k", PAYER, PROVIDER, need=PRICE, voucher=VOUCHER)
    assert v == {"channel_id": 1, "cumulative_amount": VOUCHER,
                 "signature": f"sig:1:{VOUCHER}"}
    # Counted as soon as signed: the provider may hold it.
    assert book.get(PAYER, PROVIDER)["cumulative"] == VOUCHER
    # The provider served it all: the next one pays ahead again.
    v = await next_voucher(book, m, "k", PAYER, PROVIDER, need=PRICE, voucher=VOUCHER,
                           consumed=VOUCHER)
    assert v["cumulative_amount"] == 2 * VOUCHER
    # A price above the voucher size is covered whole.
    v = await next_voucher(book, m, "k", PAYER, PROVIDER, need=3500, voucher=VOUCHER,
                           consumed=2 * VOUCHER)
    assert v["cumulative_amount"] == 2 * VOUCHER + 3500


@pytest.mark.asyncio
async def test_a_voucher_stops_at_the_deposit_and_spent_credits_refuse(tmp_path):
    book, m = await _opened(tmp_path)
    book.set_signed(PAYER, PROVIDER, 1, 9000)
    v = await next_voucher(book, m, "k", PAYER, PROVIDER, need=PRICE, voucher=VOUCHER,
                           consumed=9000)
    assert v["cumulative_amount"] == DEPOSIT
    out = await next_voucher(book, m, "k", PAYER, PROVIDER, need=PRICE, voucher=VOUCHER,
                             consumed=DEPOSIT)
    assert "spent" in out["error"] and "top them up" in out["error"]
    assert out["credits"] == {"channel_id": 1, "deposit": DEPOSIT, "used": DEPOSIT,
                              "paid_ahead": 0, "left": 0}
    # Not enough left for the price: refused too, nothing signed.
    book2, m2 = await _opened(tmp_path / "b", deposit=1500)
    book2.set_signed(PAYER, PROVIDER, 1, 1000)
    out = await next_voucher(book2, m2, "k", PAYER, PROVIDER, need=PRICE, voucher=VOUCHER,
                             consumed=1000)
    assert "spent" in out["error"] and book2.get(PAYER, PROVIDER)["cumulative"] == 1000


@pytest.mark.asyncio
async def test_closed_credits_leave_the_book(tmp_path):
    book, m = await _opened(tmp_path)
    m.state[1]["status"] = CLOSING
    out = await next_voucher(book, m, "k", PAYER, PROVIDER, need=PRICE, voucher=VOUCHER)
    assert "closed" in out["error"]
    assert book.get(PAYER, PROVIDER) is None and "1" in book.closing()


class Provider:
    """The provider's gate as the buyer meets it: holds the best voucher and
    what it served, and asks for a voucher when the credit falls short."""

    def __init__(self):
        self.cumulative = self.consumed = 0
        self.short = False          # refuse whatever the voucher
        self.sent: list[dict] = []

    async def request_service(self, config, provider, digest, args, *, request_id,
                              client_address, payer_key, voucher=None, channel_id=None,
                              **kw):
        self.sent.append({"request_id": request_id, "voucher": voucher,
                          "channel_id": channel_id, "client": client_address})
        if voucher:
            self.cumulative = max(self.cumulative, voucher["cumulative_amount"])
        cid = (voucher or {}).get("channel_id", channel_id)
        if self.short or self.cumulative - self.consumed < PRICE:
            return {"ok": False, "delivered": True, "authenticated": True,
                    "error": "Payment refused: voucher needed: short",
                    "credit": {"channel_id": cid, "consumed": self.consumed,
                               "cumulative": self.cumulative},
                    "request_id": request_id}
        self.consumed += PRICE
        return {"ok": True, "delivered": True, "authenticated": True,
                "result": {"request_id": request_id, "result": "done"},
                "credit": {"channel_id": cid, "consumed": self.consumed,
                           "cumulative": self.cumulative},
                "request_id": request_id}


async def _request(book, m):
    return await request_on_credits(book, m, None, "k", PAYER, PROVIDER, DIGEST, {},
                                    price=PRICE, voucher=VOUCHER)


@pytest.mark.asyncio
async def test_a_voucher_goes_only_when_the_held_credit_falls_short(tmp_path, monkeypatch):
    book, m = await _opened(tmp_path)
    prov = Provider()
    monkeypatch.setattr(service_client, "request_service", prov.request_service)
    out = await _request(book, m)
    assert out["ok"] and prov.sent[0]["voucher"]["cumulative_amount"] == VOUCHER
    assert prov.sent[0]["client"] == PAYER
    # The provider's count is kept: three requests per voucher.
    assert out["credits"] == {"channel_id": 1, "deposit": DEPOSIT, "used": PRICE,
                              "paid_ahead": VOUCHER - PRICE, "left": DEPOSIT - PRICE}
    for _ in range(2):
        assert (await _request(book, m))["ok"]
    assert [s["voucher"] is None for s in prov.sent] == [False, True, True]
    assert [s["channel_id"] for s in prov.sent[1:]] == [1, 1]
    out = await _request(book, m)
    assert out["ok"] and prov.sent[-1]["voucher"]["cumulative_amount"] == 2 * VOUCHER
    assert len({s["request_id"] for s in prov.sent}) == 4


@pytest.mark.asyncio
async def test_one_retry_with_a_voucher_when_the_provider_asks(tmp_path, monkeypatch):
    book, m = await _opened(tmp_path)
    prov = Provider()
    monkeypatch.setattr(service_client, "request_service", prov.request_service)
    assert (await _request(book, m))["ok"]
    # Served elsewhere (a session on the same credits): the book still thinks
    # two requests are paid ahead.
    prov.consumed = VOUCHER
    out = await _request(book, m)
    assert out["ok"], out
    assert prov.sent[1]["voucher"] is None and prov.sent[2]["voucher"] is not None
    assert prov.sent[1]["request_id"] != prov.sent[2]["request_id"]
    assert out["request_id"] == prov.sent[2]["request_id"]
    assert book.get(PAYER, PROVIDER)["consumed"] == VOUCHER + PRICE
    # A refusal of the retry, voucher and all, is final: no third send.
    prov.short = True
    out = await _request(book, m)
    assert out["ok"] is False and "voucher needed" in out["error"]
    assert len(prov.sent) == 5 and prov.sent[4]["voucher"] is not None
    assert out["request_id"] == prov.sent[4]["request_id"]


@pytest.mark.asyncio
async def test_without_credits_nothing_is_sent(tmp_path, monkeypatch):
    prov = Provider()
    monkeypatch.setattr(service_client, "request_service", prov.request_service)
    out = await _request(ChannelBook(tmp_path / "ch.json"), Market())
    assert out["delivered"] is False and "no credits" in out["error"]
    book, m = await _opened(tmp_path, deposit=500)
    out = await _request(book, m)
    assert out["delivered"] is False and "spent" in out["error"]
    assert prov.sent == []


@pytest.mark.asyncio
async def test_close_credits_and_sweep_the_refund(tmp_path):
    book, m = await _opened(tmp_path)
    assert "no credits" in (await close_credits(book, m, "k", PAYER, OTHER))["error"]
    res = await close_credits(book, m, "k", PAYER, PROVIDER)
    assert res == {"ok": True, "channel_id": 1, "tx_hash": "0xc"} and m.closed == [1]
    assert book.get(PAYER, PROVIDER) is None and "1" in book.closing()

    def key_for(addr):
        return "k" if addr == PAYER else None
    # Inside the challenge window: nothing yet.
    assert await sweep_refunds(book, m, key_for, now=50) == []
    out = await sweep_refunds(book, m, key_for, now=100)
    assert out == [{"channel_id": "1", "action": "withdrew", "error": None}]
    assert m.withdrawn == [1] and book.closing() == {}


@pytest.mark.asyncio
async def test_the_sweep_follows_the_chain(tmp_path):
    book, m = await _opened(tmp_path)
    assert (await open_credits(book, m, "k", PAYER, OTHER, 500))["ok"]
    key_for = lambda addr: "k"  # noqa: E731
    # The provider closed channel 1; channel 2's close was never sent.
    m.state[1] = {"status": CLOSING, "closes_at": 100}
    book.retire(PAYER, OTHER)
    out = await sweep_refunds(book, m, key_for, now=0)
    assert {"channel_id": "1", "action": "closed_by_provider"} in out
    assert {"channel_id": "2", "action": "close_started", "error": None} in out
    assert m.closed == [2] and book.all_open() == []
    # A channel settled elsewhere is forgotten.
    m.state[1] = {"status": SETTLED, "closes_at": 0}
    out = await sweep_refunds(book, m, key_for, now=0)
    assert {"channel_id": "1", "action": "forgot"} in out and "1" not in book.closing()


def test_credits_view():
    assert credits_view(None) == {}
    ch = {"channel_id": 4, "deposit": DEPOSIT, "cumulative": VOUCHER, "consumed": PRICE}
    assert credits_view(ch) == {"channel_id": 4, "deposit": DEPOSIT, "used": PRICE,
                                "paid_ahead": VOUCHER - PRICE, "left": DEPOSIT - PRICE}
    # One reply past the voucher: nothing paid ahead, never negative.
    ch["consumed"] = VOUCHER + PRICE
    assert credits_view(ch)["paid_ahead"] == 0
