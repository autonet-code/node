"""The daemon's side of service sessions (atn/service_sessions.py): the
provider's handler (refusals before payment, the quoted prices, metering in
the pipe, the end of a session), the buyer's local port
(``open_local_session``) and reachability. The provider's tool is a real
session tool run by the real service runner (tests/atn/test_service_runner.py);
the WS bridge is a fake whose credit gate keeps the real credit ledger
(ServiceStore); buyer and provider are real libp2p hosts in a background
trio thread, the way the daemon runs its host beside its asyncio loop.

Run: python -m pytest tests/atn/test_service_sessions.py -v
"""
from __future__ import annotations

import asyncio
import threading
import types
import uuid

import pytest
import trio

import atn.on_chain as oc
from atn import service_channels, service_sessions
from atn.service_runner import runner_for
from atn.service_sessions import DaemonSessionHandler, open_local_session
from tests.atn.test_service_runner import TOOL, _make_runtime

PAYER_KEY = "0x" + "11" * 32
PROVIDER = "0x" + "a1" * 20
CID = "7"
SCHEMA = {"type": "object"}


def _payer() -> str:
    from eth_account import Account
    return Account.from_key(PAYER_KEY).address


PAYER = _payer().lower()


def _service(rt, *, measures, prices, voucher=100, process="shared", max_sessions=0,
             code=TOOL, name="echo"):
    tool = rt.tool_store.register(
        name=f"{name}_server", description="Echoes bytes.", input_schema=SCHEMA,
        author="user", code=code,
        session={"protocol": {"id": "echo/1", "spec": "ab" * 32},
                 "max_sessions": max_sessions, "process": process,
                 "reports_usage": ["output_tokens"]})
    res = rt.service_store.register(
        name=name, description="Bytes back.", mode="session", author="user",
        backing_tool=tool["digest"], prices=prices, voucher=voucher,
        measures=[{"measure": m, "quantity": q} for m, q in measures.items()],
        settings={"model": "echo-large"})
    return rt.service_store.get(res["digest"])


def _per_byte(rt, **kw):
    """1 ATN base unit per byte carried either way."""
    return _service(rt, measures={"gigabytes": 1}, prices={"gigabytes": 10 ** 9}, **kw)


class _Bridge:
    """What DaemonSessionHandler uses of the WS bridge. The credit gate
    holds a voucher sent with the opening and checks the credits are paid
    up, as WebSocketBridge._credit_gate does (it is tested with the bridge);
    a voucher is good when its signature starts with 0xok."""

    def __init__(self, rt, refuse=""):
        self.runtime, self.refuse = rt, refuse
        self.gate_calls, self.voucher_calls = [], []

    async def _credit_gate(self, msg, record, *, need):
        self.gate_calls.append((dict(msg), need))
        if self.refuse:
            return {"ok": False, "reason": self.refuse}
        store = self.runtime.service_store
        cid = str(int(msg["channel_id"]))
        if msg.get("cumulative_amount") is not None and msg.get("signature"):
            res = await self._accept_voucher(record, cid, msg["cumulative_amount"],
                                             msg["signature"], payer=PAYER)
            if not res["ok"]:
                return res
        if store.channel_ledger(cid) is None:
            return {"ok": False, "reason": "voucher needed: no credits held"}
        store.consume(cid, need)
        if not store.paid_up(cid):
            store.refund(cid, need)
            return {"ok": False, "reason": "voucher needed"}
        return {"ok": True, "payer": PAYER, "channel_id": cid}

    async def _accept_voucher(self, record, channel_id, cumulative, signature, *, payer=""):
        self.voucher_calls.append((str(channel_id), int(cumulative), payer))
        if not str(signature).startswith("0xok"):
            return {"ok": False, "reason": "not the channel's client"}
        self.runtime.service_store.accept_voucher(channel_id, int(cumulative), str(signature),
                                                  client=payer, provider=PROVIDER)
        return {"ok": True, "client": payer}


def _opening(record, *, rid=None, paid=True, **extra):
    msg = {"type": "service_session", "spec_digest": record.digest,
           "request_id": rid or "0x" + uuid.uuid4().hex * 2, "args": {}}
    if paid:
        msg["channel_id"] = int(CID)
    msg.update(extra)
    return msg


async def _in_trio(fn, *args):
    """Run a trio function in a thread of its own (the asyncio loop stays
    free for the handler's checks)."""
    return await asyncio.get_running_loop().run_in_executor(None, lambda: trio.run(fn, *args))


async def _until(cond, timeout=15.0):
    for _ in range(int(timeout / 0.05)):
        if cond():
            return True
        await asyncio.sleep(0.05)
    return cond()


@pytest.fixture
async def env(tmp_path):
    rt = _make_runtime(tmp_path)
    runner = runner_for(rt)
    await runner.start()
    bridge = _Bridge(rt)
    handler = DaemonSessionHandler(bridge, asyncio.get_running_loop())
    try:
        yield types.SimpleNamespace(rt=rt, store=rt.service_store, runner=runner,
                                    bridge=bridge, handler=handler)
    finally:
        await runner.stop()


def _hold(store, cumulative):
    """The buyer's credits as the provider holds them (a checked voucher)."""
    store.accept_voucher(CID, cumulative, "0xok", client=PAYER, provider=PROVIDER)


# ---------------------------------------------------------------------------
# The provider's handler: accepting
# ---------------------------------------------------------------------------

async def test_openings_are_refused_before_anything_is_paid(env):
    store, h, bridge = env.store, env.handler, env.bridge
    rec = _per_byte(env.rt)
    _hold(store, 1000)

    async def refused(msg):
        out = await h._accept(msg)
        assert out.reply["ok"] is False and out.ledger is None
        return out.reply["error"]
    assert "Unknown service" in await refused(_opening(rec, rid="0x" + "0" * 64,
                                                       spec_digest="ee" * 32))
    assert "Missing" in await refused({"spec_digest": rec.digest})
    assert "'args' must be an object" in await refused(_opening(rec, args=[1]))
    tool = env.rt.tool_store.register(
        name="once", description="One reply.", input_schema=SCHEMA, author="user",
        code="print('{}')\n")
    req = env.store.register(name="once", description="One reply.", mode="request",
                             author="user", backing_tool=tool["digest"],
                             prices={"requests": 5}, voucher=50)
    assert "request service" in await refused(_opening(store.get(req["digest"])))
    store.retire(req["digest"])
    assert "retired" in await refused(_opening(store.get(req["digest"])))
    assert bridge.gate_calls == []                       # all refused before the gate
    bridge.refuse = "a paid request must be signed by the channel's client"
    assert "signed by the channel's client" in await refused(_opening(rec))
    assert bridge.gate_calls[-1][1] == 0                 # an opening charges nothing
    assert env.runner._sessions.get(rec.digest, 0) == 0  # and took no lease
    assert rec.digest not in env.runner._servers
    assert store.balance(CID) == (0, 1000)
    assert store.summary()[rec.digest]["refused"] == 3      # counted, never logged


async def test_a_tool_down_or_full_is_refused_at_opening(env):
    store, h = env.store, env.handler
    _hold(store, 1000)
    broken = _per_byte(env.rt, code="import sys\nsys.exit(5)\n", name="broken")
    out = await h._accept(_opening(broken))
    assert out.reply["ok"] is False and "code 5" in out.reply["error"]
    assert env.runner.errors(broken.digest)
    one = _per_byte(env.rt, max_sessions=1, name="one")
    first = _opening(one)
    accepted = await h._accept(first)
    assert accepted.reply["ok"], accepted.reply
    full = await h._accept(_opening(one))
    assert full.reply["ok"] is False and "full" in full.reply["error"]
    # The session ends: its place comes back.
    await _in_trio(h.ended, types.SimpleNamespace(request_id=first["request_id"]))
    again = _opening(one)
    assert (await h._accept(again)).reply["ok"]
    await _in_trio(h.ended, types.SimpleNamespace(request_id=again["request_id"]))
    assert store.balance(CID) == (0, 1000)               # nothing was charged


async def test_accepting_quotes_the_prices_and_ending_gives_everything_back(env):
    store, h = env.store, env.handler
    rec = _service(env.rt, measures={"seconds": 60, "output_tokens": 1000},
                   prices={"seconds": 3, "output_tokens": 20}, voucher=500)
    msg = _opening(rec, cumulative_amount=500, signature="0xok")
    out = await h._accept(msg)
    assert out.reply["ok"], out.reply
    r = out.reply["result"]
    assert r["protocol"]["id"] == "echo/1" and r["settings"] == {"model": "echo-large"}
    assert r["prices"] == {"seconds": 3, "output_tokens": 20} and r["voucher"] == 500
    assert r["measures"] == [{"measure": "seconds", "quantity": 60},
                             {"measure": "output_tokens", "quantity": 1000}]
    assert (r["consumed"], r["cumulative"], r["channel_id"]) == (0, 500, 7)
    assert out.ledger.balance() == (0, 500) and out.ledger.paid_up()
    assert store.has_pending_voucher(CID)                # the sweep waits for it
    assert env.runner._sessions[rec.digest] == 1
    # The same request id cannot open a second session meanwhile.
    dup = await h._accept(dict(msg))
    assert dup.reply["ok"] is False and "already open" in dup.reply["error"]
    assert env.runner._sessions[rec.digest] == 1
    await _in_trio(h.ended, types.SimpleNamespace(request_id=msg["request_id"]))
    assert env.runner._sessions[rec.digest] == 0
    assert not store.has_pending_voucher(CID)
    logged = store.summary()[rec.digest]
    assert logged["ok_count"] == 1 and logged["unique_clients"] == 1


# ---------------------------------------------------------------------------
# Sessions on real hosts
# ---------------------------------------------------------------------------

class _Hosts:
    """Buyer and provider hosts in a background trio thread, directly
    connected."""

    def __init__(self):
        from nodes.common.p2p import AutonetHost
        self.buyer = AutonetHost(node_id="buyer", listen_host="127.0.0.1", mesh_interval=0.5)
        self.provider = AutonetHost(node_id="seller", listen_host="127.0.0.1",
                                    mesh_interval=0.5)
        self.ready, self.stop = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=lambda: trio.run(self._main), daemon=True)

    async def _main(self):
        async with self.buyer.run(), self.provider.run():
            await self.buyer.connect_to_peer(self.provider.addrs[0])
            self.ready.set()
            while not self.stop.is_set():
                await trio.sleep(0.05)

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(30)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(15)

    async def trio(self, fn):
        token = self.buyer._trio_token
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: trio.from_thread.run(fn, trio_token=token))


@pytest.fixture(scope="module")
def hosts():
    with _Hosts() as h:
        yield h


@pytest.fixture
def serving(env, hosts):
    hosts.provider.set_service_session_handler(env.handler)
    yield hosts
    hosts.provider.set_service_session_handler(None)


class _Pays:
    """The buyer: each voucher pays ``size`` ahead of what is used, or
    none at all."""

    def __init__(self, size=100, pay=True, signature="0xok"):
        self.size, self.pay, self.signature, self.asked = size, pay, signature, []

    async def __call__(self, consumed, cumulative):
        self.asked.append((consumed, cumulative))
        if not self.pay:
            return None
        return {"channel_id": int(CID), "signature": self.signature,
                "cumulative_amount": max(consumed, cumulative) + self.size}


async def _run(hosts, msg, body, voucher_source=None):
    """Open a session with ``msg`` and run ``body(session)`` in it."""
    async def go():
        async with hosts.buyer.service_session(hosts.provider.peer_id, msg,
                                               voucher_source=voucher_source,
                                               timeout=60) as s:
            return await body(s)
    return await hosts.trio(go)


async def _echo(s, data: bytes) -> bytes:
    await s.send(data)
    back = bytearray()
    while len(back) < len(data):
        chunk = await s.receive()
        if not chunk:
            break
        back.extend(chunk)
    return bytes(back)


async def _session_over(env, rec, n=1):
    """The provider has ended ``n`` sessions of ``rec`` and holds none."""
    return await _until(lambda: env.runner._sessions.get(rec.digest, 0) == 0
                        and not env.store.has_pending_voucher(CID)
                        and env.store.summary().get(rec.digest, {}).get("ok_count", 0) >= n)


async def test_bytes_carried_either_way_are_charged(env, serving):
    rec = _per_byte(env.rt)
    _hold(env.store, 10_000)

    async def body(s):
        return await _echo(s, b"x" * 300)
    assert await _run(serving, _opening(rec), body) == b"x" * 300
    assert await _session_over(env, rec)
    # 300 bytes up and 300 back; the tool's token reports are not priced.
    assert env.store.balance(CID) == (600, 10_000)
    rows = (env.store._requests_path).read_text(encoding="utf-8").splitlines()
    assert '"amount": "600"' in rows[-1] and PAYER in rows[-1]


async def test_seconds_are_charged_while_the_session_is_open(env, serving):
    rec = _service(env.rt, measures={"seconds": 1}, prices={"seconds": 7}, voucher=1000)
    _hold(env.store, 10_000)

    async def body(s):
        await trio.sleep(2.3)
        return await _echo(s, b"tick")
    assert await _run(serving, _opening(rec), body) == b"tick"
    assert await _session_over(env, rec)
    consumed = env.store.balance(CID)[0]
    assert 21 <= consumed <= 28 and consumed % 7 == 0      # 3 or 4 seconds begun


async def test_tokens_are_charged_from_the_tools_reports(env, serving):
    """Output tokens priced 50 per 100: the tool reports 300 for a 300-byte
    reply, after it (so a reply can overshoot the credits by one)."""
    rec = _service(env.rt, measures={"output_tokens": 100},
                   prices={"output_tokens": 50}, voucher=1000)
    _hold(env.store, 10_000)
    store = env.store

    async def body(s):
        back = await _echo(s, b"t" * 300)
        with trio.fail_after(10):
            while store.balance(CID)[0] < 150:
                await trio.sleep(0.05)
        return back
    assert await _run(serving, _opening(rec), body) == b"t" * 300
    assert await _session_over(env, rec)
    assert store.balance(CID) == (150, 10_000)


async def test_nothing_is_forwarded_while_not_paid_up(env, serving):
    """100 paid ahead, 1 per byte: 60 bytes reach the tool, but its 60-byte
    echo would make 120; the buyer pays no more, so the echo is never
    forwarded and the session ends."""
    rec = _per_byte(env.rt)
    _hold(env.store, 100)
    pays = _Pays(pay=False)

    async def body(s):
        await s.send(b"y" * 60)
        got = await s.receive()
        return got, s.close_reason
    got, reason = await _run(serving, _opening(rec), body, voucher_source=pays)
    assert got == b"" and "stopped paying" in reason
    assert pays.asked                                    # it was asked ahead
    assert await _session_over(env, rec)
    assert env.store.balance(CID) == (120, 100)          # one step over, not served


async def test_a_voucher_in_the_session_lets_it_go_on(env, serving):
    rec = _per_byte(env.rt)
    _hold(env.store, 100)
    pays = _Pays(size=100)

    async def body(s):
        return await _echo(s, b"z" * 60)
    assert await _run(serving, _opening(rec), body, voucher_source=pays) == b"z" * 60
    assert await _session_over(env, rec)
    consumed, cumulative = env.store.balance(CID)
    assert consumed == 120 and cumulative - consumed <= 100
    # Checked on the session's channel, for the payer the gate verified.
    assert env.bridge.voucher_calls
    assert all(c == (CID, c[1], PAYER) for c in env.bridge.voucher_calls)
    # A voucher the chain check refuses ends the session.
    bad = _Pays(size=100, signature="0xforged")

    async def body2(s):
        await s.send(b"w" * 200)
        return await s.receive(), s.close_reason
    got, reason = await _run(serving, _opening(rec), body2, voucher_source=bad)
    assert got == b"" and "voucher refused" in reason
    assert await _session_over(env, rec, 2)


async def test_a_per_session_tool_stops_when_the_session_ends(env, serving):
    rec = _per_byte(env.rt, process="per_session")
    _hold(env.store, 10_000)
    seen = {}

    async def body(s):
        seen["procs"] = [srv.proc for srv in env.runner._leased.values()]
        return await _echo(s, b"hi")
    assert await _run(serving, _opening(rec), body) == b"hi"
    assert await _session_over(env, rec)
    assert len(seen["procs"]) == 1
    assert await _until(lambda: seen["procs"][0].returncode is not None)
    assert env.runner._leased == {}


# ---------------------------------------------------------------------------
# The buyer: open_local_session
# ---------------------------------------------------------------------------

@pytest.fixture
def chain(monkeypatch):
    """The channel as the chain reads it (open) and vouchers as the payer
    signs them (good for the fake gate); every voucher made is recorded."""
    made = []

    async def get_channel(self, channel_id):
        return {"status": service_channels.OPEN}

    def sign_voucher(self, private_key, channel_id, cumulative_amount):
        return f"0xok{cumulative_amount}"
    monkeypatch.setattr(oc.ServiceMarketClient, "get_channel", get_channel)
    monkeypatch.setattr(oc.ServiceMarketClient, "sign_voucher", sign_voucher)
    real = service_channels.next_voucher

    async def next_voucher(*a, **kw):
        out = await real(*a, **kw)
        made.append((kw.get("consumed"), out.get("cumulative_amount")))
        return out
    monkeypatch.setattr(service_channels, "next_voucher", next_voucher)
    return made


async def _open(env, hosts, rec, **kw):
    return await open_local_session(
        env.rt, payer_key=kw.pop("payer_key", PAYER_KEY), payer=PAYER,
        provider_address=PROVIDER, spec_digest=rec.digest,
        p2p_host=hosts.buyer, peer_id=hosts.provider.peer_id, **kw)


async def _through(out, data: bytes) -> bytes:
    reader, writer = await asyncio.open_connection(out["host"], out["port"])
    writer.write(data)
    await writer.drain()
    back = await asyncio.wait_for(reader.readexactly(len(data)), 30)
    writer.close()
    return back


async def test_a_session_needs_credits_with_the_provider(env, serving, chain):
    rec = _per_byte(env.rt)
    out = await _open(env, serving, rec)
    assert "open credits first" in out["error"]
    env.rt.channel_book.set_open(PAYER, PROVIDER, int(CID), 1000)
    out = await _open(env, serving, rec, payer_key="")
    assert "needs a key" in out["error"]
    assert env.bridge.gate_calls == [] and chain == []   # nothing was sent or signed


async def test_a_local_port_reaches_the_tool_and_pays_ahead_once(env, serving, chain):
    """The opening carries a voucher only when nothing is paid ahead; the
    local program's bytes reach the tool and come back."""
    rec = _per_byte(env.rt)
    book = env.rt.channel_book
    book.set_open(PAYER, PROVIDER, int(CID), 1000)
    out = await _open(env, serving, rec)
    assert out.get("ok"), out
    assert out["path"] == "direct" and out["protocol"]["id"] == "echo/1"
    assert out["prices"] == {"gigabytes": 10 ** 9} and out["voucher"] == 100
    assert out["credits"]["paid_ahead"] == 100
    first = env.bridge.gate_calls[-1][0]
    assert first["cumulative_amount"] == 100 and first["payer_signature"]
    assert await _through(out, b"hello") == b"hello"
    assert await _session_over(env, rec)
    # Paid ahead already: the next opening carries no voucher.
    out = await _open(env, serving, rec)
    assert out.get("ok"), out
    assert "cumulative_amount" not in env.bridge.gate_calls[-1][0]
    assert await _through(out, b"again") == b"again"
    assert await _session_over(env, rec, 2)
    assert chain == [(None, 100)]


async def test_vouchers_in_the_session_are_the_quoted_size(env, serving, chain, monkeypatch):
    """The catalog's list says 50; the provider quotes 100 at the opening,
    and the quote holds for the session: each voucher in it pays 100
    ahead of what the provider has counted."""
    from atn.service_catalog import ServiceCatalog
    rec = _per_byte(env.rt)
    listed = dict(rec.price_list, voucher=50)
    monkeypatch.setattr(ServiceCatalog, "price_list", lambda self, d, p="": dict(listed))
    book = env.rt.channel_book
    book.set_open(PAYER, PROVIDER, int(CID), 5000)
    out = await _open(env, serving, rec)
    assert out.get("ok"), out
    assert await _through(out, b"q" * 300) == b"q" * 300
    assert await _session_over(env, rec)
    assert chain[0] == (None, 50)                       # the opening: the listed size
    assert len(chain) >= 2
    prev = chain[0][1]
    for consumed, total in chain[1:]:
        assert total == max(consumed, prev) + 100
        prev = total
    consumed, cumulative = env.store.balance(CID)
    assert consumed == 600 and 0 <= cumulative - consumed <= 100
    assert book.get(PAYER, PROVIDER)["cumulative"] == cumulative


async def test_a_refused_session_is_an_error(env, serving, chain):
    rec = _per_byte(env.rt)
    env.rt.channel_book.set_open(PAYER, PROVIDER, int(CID), 1000)
    env.store.retire(rec.digest)
    out = await _open(env, serving, rec)
    assert "refused the session" in out["error"] and "retired" in out["error"]


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

class _FakeHost:
    def __init__(self, reachable, peers=None):
        self._trio_token = object()
        self._capability = types.SimpleNamespace(reachable=reachable)
        self.addrs = ["/ip4/127.0.0.1/tcp/1/p2p/me"]
        self.peer_id = "me"
        self._peer_attribution = types.SimpleNamespace(
            peer_for_agent=lambda a: (types.SimpleNamespace(peer_id=(peers or {})[a.lower()])
                                      if a.lower() in (peers or {}) else None))

    def peer_reachable(self, pid):
        return {"p-open": True, "p-nat": False}.get(pid)


def test_reachability_for_the_app_and_the_catalog():
    out = service_sessions.own_reachability(_FakeHost(False))
    assert out["reachable"] is False and "relay" in out["advice"]
    out = service_sessions.own_reachability(_FakeHost(True))
    assert out["reachable"] is True and "advice" not in out
    assert service_sessions.own_reachability(
        types.SimpleNamespace(_trio_token=None))["reachable"] is None
    host = _FakeHost(True, {"0xaa": "p-open", "0xbb": "p-nat", "0xcc": "me"})
    assert service_sessions.provider_reachable("0xAA", host) is True
    assert service_sessions.provider_reachable("0xbb", host) is False
    assert service_sessions.provider_reachable("0xcc", host) is True    # this daemon
    assert service_sessions.provider_reachable("0xdd", host) is None    # not known
