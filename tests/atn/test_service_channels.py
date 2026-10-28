"""Buyer side of the payment channel (issue #45): one channel per provider,
a voucher per request, committed only when the request may have reached the
provider, and the unspent deposit withdrawn after the window."""

from __future__ import annotations

import pytest

from atn.service_channels import (
    CLIENT_CLOSING,
    SETTLED,
    ChannelBook,
    close_channel,
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
