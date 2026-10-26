"""Provider-side channel settlement (atn/channel_settlement.py): a channel
whose client started a close is settled with the stored best voucher; a
closed or settled one is forgotten; an open one is left alone."""
from __future__ import annotations

import asyncio

from atn.channel_settlement import (
    CLIENT_CLOSING, CLOSING, OPEN, settle_closing_channels,
)

PROV = "0x" + "aa" * 20


class Store:
    def __init__(self, vouchers):
        self.v = dict(vouchers)

    def channel_vouchers(self):
        return {k: dict(x) for k, x in self.v.items()}

    def forget_channel_voucher(self, cid):
        self.v.pop(str(cid), None)


class Client:
    def __init__(self, statuses):
        self.statuses = statuses
        self.closed = []

    async def get_channel(self, cid):
        st = self.statuses.get(cid)
        return None if st is None else {"status": st, "provider": PROV}

    async def close_channel(self, key, cid, cumulative, signature):
        self.closed.append((key, cid, cumulative, signature))
        return {"success": True}


def test_settles_client_closes_and_forgets_finished_channels():
    store = Store({"1": {"cumulative": 300, "signature": "0xs1"},
                   "2": {"cumulative": 50, "signature": "0xs2"},
                   "3": {"cumulative": 70, "signature": "0xs3"}})
    client = Client({1: CLIENT_CLOSING, 2: OPEN, 3: CLOSING})
    out = asyncio.run(settle_closing_channels(
        store, client, lambda p: "0xkey" if p == PROV else None))
    assert client.closed == [("0xkey", 1, 300, "0xs1")]
    assert set(store.v) == {"2"}
    assert {o["channel_id"] for o in out} == {"1", "3"}


def test_no_key_leaves_the_voucher_for_a_later_sweep():
    store = Store({"1": {"cumulative": 300, "signature": "0xs1"}})
    client = Client({1: CLIENT_CLOSING})
    asyncio.run(settle_closing_channels(store, client, lambda p: None))
    assert client.closed == [] and "1" in store.v
