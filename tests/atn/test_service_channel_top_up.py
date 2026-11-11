"""Buyer side: topping up the credits with a provider
(``service_channels.top_up``) adds to the channel's deposit on chain and in
the book; only an open channel takes it (a closed one leaves the book, so
new credits can be opened), and a failed top-up changes nothing."""

from __future__ import annotations

import pytest

from atn.service_channels import CLIENT_CLOSING, OPEN, ChannelBook, top_up

PAYER, PROVIDER = "0xPayer", "0xProvider"


class Market:
    def __init__(self, status=OPEN, deposit=1000, decode=True):
        self.state = {7: {"status": status, "deposit": str(deposit)}}
        self.topped = []
        self.decode = decode

    async def get_channel(self, cid):
        st = self.state.get(cid)
        return None if st is None else dict(st)

    async def top_up_channel(self, key, cid, amount):
        self.topped.append((key, cid, amount))
        st = self.state[cid]
        st["deposit"] = str(int(st["deposit"]) + amount)
        return {"success": True, "tx_hash": "0xt",
                "deposit": int(st["deposit"]) if self.decode else None}


def _book(tmp_path, deposit=1000):
    book = ChannelBook(tmp_path / "ch.json")
    book.set_open(PAYER, PROVIDER, 7, deposit)
    return book


@pytest.mark.asyncio
async def test_top_up_grows_the_deposit_on_chain_and_in_the_book(tmp_path):
    book, m = _book(tmp_path), Market()
    res = await top_up(book, m, "k", PAYER, PROVIDER, 500)
    assert {k: res[k] for k in ("ok", "channel_id", "deposit", "tx_hash")} == {
        "ok": True, "channel_id": 7, "deposit": 1500, "tx_hash": "0xt"}
    assert res["credits"]["deposit"] == 1500 and res["credits"]["left"] == 1500
    assert m.topped == [("k", 7, 500)]
    assert book.get(PAYER, PROVIDER)["deposit"] == 1500
    # Kept across a restart.
    assert ChannelBook(tmp_path / "ch.json").get(PAYER, PROVIDER)["deposit"] == 1500


@pytest.mark.asyncio
async def test_top_up_without_a_decoded_event_uses_the_chain_deposit(tmp_path):
    # The book says 1000, the chain 1200 (a top-up from elsewhere).
    book, m = _book(tmp_path), Market(deposit=1200, decode=False)
    res = await top_up(book, m, "k", PAYER, PROVIDER, 300)
    assert res["deposit"] == 1500 and book.get(PAYER, PROVIDER)["deposit"] == 1500


@pytest.mark.asyncio
async def test_top_up_refuses_without_an_open_channel(tmp_path):
    m = Market()
    empty = ChannelBook(tmp_path / "none.json")
    assert "open credits first" in (await top_up(empty, m, "k", PAYER, PROVIDER, 5))["error"]
    book = _book(tmp_path)
    assert "positive" in (await top_up(book, m, "k", PAYER, PROVIDER, 0))["error"]
    closing = Market(status=CLIENT_CLOSING)
    res = await top_up(book, closing, "k", PAYER, PROVIDER, 5)
    assert "closed" in res["error"] and closing.topped == []
    # The closed channel leaves the book for its refund to be swept.
    assert book.get(PAYER, PROVIDER) is None and "7" in book.closing()


@pytest.mark.asyncio
async def test_a_failed_top_up_leaves_the_book(tmp_path):
    book, m = _book(tmp_path), Market()

    async def fail(key, cid, amount):
        return {"success": False, "error": "approve failed: insufficient ATN"}
    m.top_up_channel = fail
    res = await top_up(book, m, "k", PAYER, PROVIDER, 500)
    assert "insufficient ATN" in res["error"]
    assert book.get(PAYER, PROVIDER)["deposit"] == 1000


def test_set_deposit_only_touches_the_named_open_channel(tmp_path):
    book = _book(tmp_path)
    assert book.set_deposit(PAYER, PROVIDER, 8, 9999) is False
    assert book.get(PAYER, PROVIDER)["deposit"] == 1000
    assert book.set_deposit(PAYER, PROVIDER, 7, 2000) is True
    assert book.get(PAYER, PROVIDER)["deposit"] == 2000
