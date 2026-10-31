"""On-chain calls run off the daemon's event loop (issues #40, #49).

OnChainService / ServiceMarketClient methods are async but make blocking
web3 calls. Each now runs on a worker thread, so a slow RPC no longer
freezes the loop the WS server and gossip ingest share."""
import asyncio
import time

import atn.on_chain as oc
from atn.config import RPBConfig


def _slow_rpc(self, *a, **k):
    time.sleep(0.5)          # a blocking RPC round trip
    raise RuntimeError("no chain")


def test_a_slow_rpc_does_not_block_the_loop(monkeypatch):
    monkeypatch.setattr(oc.OnChainService, "_get_web3", _slow_rpc)
    svc = oc.OnChainService(RPBConfig(rpc_url="http://127.0.0.1:1",
                                      substrate_address="0x" + "11" * 20))

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        t = asyncio.create_task(ticker())
        assert await svc.is_registered("0x" + "22" * 20) is False
        t.cancel()
        return ticks

    # The loop kept ticking through the 0.5 s call.
    assert asyncio.run(main()) >= 5


def test_a_method_calling_a_sibling_runs_it_inline(monkeypatch):
    calls = []

    async def count(self):
        calls.append("count")
        return 2

    async def get(self, i):
        calls.append(i)
        return {"id": i}

    monkeypatch.setattr(oc.ServiceMarketClient, "service_count", oc._off_the_loop(
        type("X", (), {"f": count})).f)
    monkeypatch.setattr(oc.ServiceMarketClient, "get_service", get)

    async def listing(self):
        n = await self.service_count()
        return [await self.get_service(i) for i in range(n)]

    wrapped = oc._off_the_loop(type("Y", (), {"listing": listing})).listing
    client = object.__new__(oc.ServiceMarketClient)
    assert asyncio.run(wrapped(client)) == [{"id": 0}, {"id": 1}]
    assert calls == ["count", 0, 1]
