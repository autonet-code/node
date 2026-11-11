"""The daemon's side of service sessions (docs/services_market.md, Decision
2026-11-12). The network layer and the wire are
``nodes/common/service_session.py``.

Provider (``DaemonSessionHandler``): an opening is checked before anything
is paid: the service exists, is a session service and is not retired; its
tool is up and not full (``atn/service_runner.py`` leases it); the buyer's
credits (a payment channel, with any new voucher verified on chain) belong
to the payer who signed the opening and are paid up. The quoted prices (the
current price list) hold for the whole session. The session is then piped
to the tool's port as its own local connection and metered: seconds while
it is open and gigabytes carried either way are counted by the daemon,
tokens come from the tool's usage reports. Every charge goes to the
buyer's credits (``ServiceStore.consume``); nothing more is forwarded until
they are paid up, a voucher is asked for ahead of time, and with none in
time the session ends.

Buyer (``open_local_session``): opens a session on the household's credits
with the provider and hands it to a local program on a fresh 127.0.0.1
port (one connection). Vouchers are signed as the provider asks
(``service_channels.next_voucher``), each by the provider's voucher size.

Reachability: ``own_reachability()`` tells the app whether this daemon can
be dialled directly; ``provider_reachable(address)`` is what catalog rows
show for a provider.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from typing import Any, Optional

from .service_prices import Meter, voucher_size
from .service_spec import BYTES_PER_GIGABYTE, MODE_SESSION, session_protocol

log = logging.getLogger(__name__)

GATE_TIMEOUT = 180.0        # the check reads the chain and may start a tool
STORE_TIMEOUT = 30.0
PIPE_READ = 65536
TICK = 1.0

try:
    from nodes.common.service_session import (
        Accept,
        Ledger,
        SessionClosed,
        SessionHandler,
        VoucherCheck,
    )
except Exception:  # pragma: no cover (nodes ships with the daemon)
    SessionHandler = object  # type: ignore[assignment,misc]
    Ledger = object  # type: ignore[assignment,misc]


class _ChannelLedger(Ledger):  # type: ignore[misc,valid-type]
    """One session's view of the buyer's credits, kept in the store's
    (thread-safe) ledger."""

    def __init__(self, store: Any, channel_id: Any) -> None:
        self.store, self.channel_id = store, str(channel_id)

    def charge(self, amount: int) -> None:
        self.store.consume(self.channel_id, int(amount))

    def paid_up(self) -> bool:
        return self.store.paid_up(self.channel_id)

    def balance(self) -> tuple:
        return self.store.balance(self.channel_id)


class _SessionMeter:
    """A session's running charge at its quoted prices, fed from two
    threads (the pipe, and the tool's usage reports)."""

    def __init__(self, spec: dict, prices: dict, ledger: Optional[_ChannelLedger]) -> None:
        scaled = dict(spec)
        measures = []
        for m in spec.get("measures") or []:
            q = int(m["quantity"])
            if m["measure"] == "gigabytes":
                q *= BYTES_PER_GIGABYTE   # the daemon counts bytes
            measures.append({"measure": m["measure"], "quantity": q})
        scaled["measures"] = measures
        self.meter = Meter(scaled, prices)
        self.ledger = ledger
        self._lock = threading.Lock()

    def charges(self, measure: str) -> bool:
        return self.ledger is not None and self.meter.charges(measure)

    def add(self, measure: str, amount: int) -> int:
        if self.ledger is None:
            return 0
        with self._lock:
            delta = self.meter.add(measure, amount)
        if delta:
            self.ledger.charge(delta)
        return delta

    @property
    def charged(self) -> int:
        return self.meter.charged


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

def session_handler_for(bridge: Any, loop: Any) -> Any:
    """The provider's ``SessionHandler`` for ``bridge`` (the WS bridge,
    which holds the gate's keys and chain access), or None."""
    if bridge is None or loop is None:
        return None
    return DaemonSessionHandler(bridge, loop)


class DaemonSessionHandler(SessionHandler):  # type: ignore[misc,valid-type]
    """Serves this daemon's session services (trio side); the checks run on
    the daemon's asyncio loop."""

    def __init__(self, bridge: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.bridge = bridge
        self.loop = loop
        self._open: dict[str, dict[str, Any]] = {}   # request id -> session state

    async def _on_loop(self, factory: Any, timeout: float) -> Any:
        import trio
        loop = self.loop

        def run() -> Any:
            return asyncio.run_coroutine_threadsafe(factory(), loop).result(timeout)
        return await trio.to_thread.run_sync(run)

    @property
    def _store(self) -> Any:
        return self.bridge.runtime.service_store

    async def accept(self, open_msg: dict) -> Any:
        return await self._on_loop(lambda: self._accept(open_msg), GATE_TIMEOUT)

    async def _accept(self, msg: dict) -> Any:
        from .service_runner import Lease, runner_for

        bridge, store = self.bridge, self._store
        digest = str(msg.get("spec_digest") or "")
        rid = str(msg.get("request_id") or "")

        def refuse(error: str) -> Any:
            if digest and store.get(digest) is not None:
                store.count_refused(digest)
            return Accept(reply={"ok": False, "error": error})
        if not digest or not rid:
            return refuse("Missing 'spec_digest' or 'request_id' field")
        if rid in self._open:
            # Its state would replace the open one's, whose lease would
            # then never be given back.
            return refuse("this request_id names a session already open")
        record = store.get(digest)
        if record is None:
            return refuse(f"Unknown service digest: {digest[:16]}")
        if record.retired:
            return refuse("Service retired")
        if record.mode != MODE_SESSION:
            return refuse("this is a request service: send a service_request")
        if not isinstance(msg.get("args", {}), dict):
            return refuse("'args' must be an object")
        prices = dict(record.prices)
        priced = any(int(v or 0) > 0 for v in prices.values())
        credit = None
        if priced:
            credit = await bridge._credit_gate(msg, record, need=0)
            if not credit.get("ok"):
                return refuse(credit.get("reason") or "payment refused")
        lease = await runner_for(bridge.runtime).lease(record)
        if not isinstance(lease, Lease):
            return refuse(str(lease))
        if rid in self._open:   # opened meanwhile under the same id
            await lease.release()
            return refuse("this request_id names a session already open")
        ledger = None
        cid = None
        if credit is not None:
            cid = str(credit["channel_id"])
            ledger = _ChannelLedger(store, cid)
            store.begin_use(cid)
        meter = _SessionMeter(record.spec, prices, ledger)
        self._open[rid] = {"record": record, "lease": lease, "meter": meter,
                           "channel_id": cid, "payer": (credit or {}).get("payer", ""),
                           "opened": time.time()}
        consumed, cumulative = store.balance(cid) if cid is not None else (0, 0)
        reply = {"ok": True, "result": {
            "request_id": rid, "protocol": session_protocol(record.spec),
            "prices": prices, "measures": record.measures,
            "voucher": record.voucher, "settings": record.settings,
            "consumed": consumed, "cumulative": cumulative,
            "channel_id": int(cid) if cid is not None else None}}
        return Accept(reply=reply, ledger=ledger)

    async def check_voucher(self, open_msg: dict, frame: dict) -> Any:
        return await self._on_loop(lambda: self._check(open_msg, frame), GATE_TIMEOUT)

    async def _check(self, open_msg: dict, frame: dict) -> Any:
        state = self._open.get(str(open_msg.get("request_id") or ""))
        if state is None or state.get("channel_id") is None:
            return VoucherCheck(ok=False, error="this session is not paid by credits")
        if str(frame.get("channel_id")) != str(state["channel_id"]):
            return VoucherCheck(ok=False, error="a voucher on another channel")
        res = await self.bridge._accept_voucher(
            state["record"], frame.get("channel_id"), frame.get("cumulative_amount"),
            frame.get("signature"), payer=state.get("payer", ""))
        if not res.get("ok"):
            return VoucherCheck(ok=False, error=str(res.get("reason") or "invalid"))
        return VoucherCheck(ok=True)

    async def serve(self, session: Any) -> None:
        import trio

        state = self._open.get(session.request_id)
        if state is None:
            await session.aclose("the session was not accepted")
            return
        lease, meter = state["lease"], state["meter"]
        try:
            tcp = await trio.open_tcp_stream(lease.host, lease.port)
        except OSError as e:
            self._error(state, f"the tool is not serving: {e}")
            await session.aclose(f"the provider's tool is not serving: {e}")
            return
        local_port = tcp.socket.getsockname()[1]
        from .service_runner import runner_for
        runner = runner_for(self.bridge.runtime)
        self.loop.call_soon_threadsafe(runner.watch, lease.token, local_port, meter.add)
        state["local_port"] = local_port
        try:
            await pipe(session, tcp, meter=meter, voucher=state["record"].voucher)
        finally:
            with trio.CancelScope(shield=True), trio.move_on_after(3):
                try:
                    await tcp.aclose()
                except Exception:  # noqa: BLE001
                    pass

    async def ended(self, session: Any) -> None:
        state = self._open.pop(session.request_id, None)
        if state is None:
            return

        async def run() -> None:
            from .service_runner import runner_for
            runner = runner_for(self.bridge.runtime)
            if state.get("local_port"):
                runner.unwatch(state["lease"].token, state["local_port"])
            try:
                await state["lease"].release()
            except Exception as e:  # noqa: BLE001
                log.debug("releasing a service tool lease failed: %s", e)
            store = self._store
            if state.get("channel_id") is not None:
                store.end_use(state["channel_id"])
            store.record_request(state["record"].digest, session.request_id,
                                 state.get("payer", ""), ok=True,
                                 amount=str(state["meter"].charged))
        await self._on_loop(run, STORE_TIMEOUT)

    def _error(self, state: dict, error: str) -> None:
        from .service_runner import runner_for
        self.loop.call_soon_threadsafe(
            runner_for(self.bridge.runtime).record_error, state["record"].digest, error)


async def pipe(session: Any, local: Any, *, meter: Optional[_SessionMeter] = None,
               voucher: int = 0) -> None:
    """Copy bytes between a session and a local trio stream until either
    ends. On the provider (``meter``) every chunk and every second is
    charged first, and nothing is forwarded while the credits are not paid
    up; a voucher is asked for once less than half a voucher is left."""
    import trio

    charging = meter is not None and meter.ledger is not None
    ahead = max(1, int(voucher) // 2)

    async def paid() -> None:
        if charging:
            await session.paid_up()
            await session.low(ahead)

    async def carried(nbytes: int) -> None:
        if charging and meter.charges("gigabytes"):
            meter.add("gigabytes", nbytes)
        await paid()

    async def up(done: Any) -> None:
        try:
            while True:
                data = await session.receive()
                if not data:
                    break
                await carried(len(data))
                await local.send_all(data)
        except (SessionClosed, trio.BrokenResourceError, trio.ClosedResourceError):
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("session to local pipe ended: %s", e)
        done.cancel()

    async def down(done: Any) -> None:
        try:
            while True:
                data = await local.receive_some(PIPE_READ)
                if not data:
                    break
                await carried(len(data))
                await session.send(data)
        except (SessionClosed, trio.BrokenResourceError, trio.ClosedResourceError):
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("local to session pipe ended: %s", e)
        done.cancel()

    async def clock(done: Any) -> None:
        try:
            while True:
                meter.add("seconds", 1)
                await paid()
                await trio.sleep(TICK)
        except SessionClosed:
            pass
        done.cancel()

    try:
        if charging:
            await paid()
        async with trio.open_nursery() as nursery:
            if charging and meter.charges("seconds"):
                nursery.start_soon(clock, nursery.cancel_scope)
            nursery.start_soon(up, nursery.cancel_scope)
            nursery.start_soon(down, nursery.cancel_scope)
    except SessionClosed:
        pass


# ---------------------------------------------------------------------------
# Buyer
# ---------------------------------------------------------------------------

async def open_local_session(runtime: Any, *, payer_key: str, payer: str,
                             provider_address: str, spec_digest: str,
                             args: Optional[dict] = None,
                             p2p_host: Any = None, peer_id: str = "",
                             timeout: float = 60.0,
                             accept_timeout: float = 300.0) -> dict:
    """Buyer: open a session to ``provider_address``'s service on the
    household's credits with it, and expose it on 127.0.0.1 for one local
    connection. Returns ``{ok, host, port, path, relay, request_id,
    protocol, prices, measures, voucher, credits}`` once the provider
    accepted, or ``{"error": ...}``. The session lasts while the local
    connection is open; a port nobody connects to within ``accept_timeout``
    closes it."""
    from . import service_client
    from .on_chain import ServiceMarketClient
    from .service_catalog import catalog_for
    from .service_channels import channel_lock, credits_view, next_voucher

    args = dict(args or {})
    host = service_client._running_host(p2p_host)
    if host is None:
        return {"error": "libp2p is not running on this daemon, so it cannot "
                         "open a session"}
    config = runtime._config.rpb
    peer = peer_id or await service_client.resolve_provider_peer_id(
        config, provider_address, host)
    if not peer:
        return {"error": "no peer id on chain for the provider"}
    price_list = catalog_for(runtime).price_list(spec_digest, provider_address)
    priced = bool(price_list) and any(
        int(v or 0) > 0 for v in (price_list.get("prices") or {}).values())
    book = getattr(runtime, "channel_book", None)
    smc = ServiceMarketClient(config)
    loop = asyncio.get_running_loop()
    open_rid = service_client.new_request_id()
    msg: dict[str, Any] = {"type": "service_session", "msg_id": uuid.uuid4().hex[:12],
                           "spec_digest": spec_digest, "request_id": open_rid,
                           "args": args, "client": payer}
    ch = book.get(payer, provider_address) if book is not None else None
    if priced or ch is not None:
        if ch is None:
            return {"error": "no credits with this provider: open credits first "
                             "(open_credits with a deposit)"}
        if not payer_key:
            return {"error": "paying for a session needs a key (call as an agent)"}
        voucher = voucher_size(price_list) if price_list else 0
        async with channel_lock(book, payer, provider_address):
            ch = book.get(payer, provider_address) or ch
            msg["channel_id"] = int(ch["channel_id"])
            if int(ch["cumulative"]) - int(ch.get("consumed") or 0) <= 0:
                fields = await next_voucher(book, smc, payer_key, payer, provider_address,
                                            need=1, voucher=voucher)
                if fields.get("error"):
                    return {"error": str(fields["error"]), "credits": fields.get("credits")}
                msg.update({k: fields[k] for k in ("cumulative_amount", "signature")})
        msg["payer_signature"] = service_client.sign_request(
            payer_key, open_rid, spec_digest, args, provider_address)
    cid = msg.get("channel_id")
    # Vouchers in the session are the size the provider quotes at its
    # opening (the quote holds for the session), else its price list's.
    quoted = {"voucher": voucher_size(price_list) if price_list else 0}

    def on_balance(consumed: int, cumulative: int) -> None:
        if book is not None and cid is not None:
            loop.call_soon_threadsafe(book.note_balance, payer, provider_address,
                                      int(cid), int(consumed), int(cumulative))

    async def make_voucher(consumed: int, cumulative: int) -> dict:
        async with channel_lock(book, payer, provider_address):
            return await next_voucher(book, smc, payer_key, payer, provider_address,
                                      need=1, voucher=quoted["voucher"], consumed=consumed)

    async def voucher_source(consumed: int, cumulative: int) -> Optional[dict]:
        import trio
        if cid is None:
            return None

        def run() -> dict:
            fut = asyncio.run_coroutine_threadsafe(make_voucher(consumed, cumulative), loop)
            try:
                return fut.result(120)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                return {"error": "no voucher could be made in time"}
        return await trio.to_thread.run_sync(run)

    ready: concurrent.futures.Future = concurrent.futures.Future()

    async def session() -> None:
        import trio
        from nodes.common.service_session import SessionRefused
        try:
            async with host.service_session(
                    peer, msg, voucher_source=voucher_source, on_balance=on_balance,
                    timeout=timeout) as sess:
                result = sess.accept_reply.get("result") or {}
                try:   # before the first checkpoint: the reader may ask next
                    if int(result.get("voucher") or 0) > 0:
                        quoted["voucher"] = int(result["voucher"])
                except (TypeError, ValueError):
                    pass
                listeners = await trio.open_tcp_listeners(0, host="127.0.0.1")
                port = listeners[0].socket.getsockname()[1]
                ready.set_result({"ok": True, "host": "127.0.0.1", "port": port,
                                  "path": sess.path, "relay": sess.relay,
                                  "request_id": open_rid,
                                  "protocol": result.get("protocol"),
                                  "prices": result.get("prices"),
                                  "measures": result.get("measures"),
                                  "voucher": result.get("voucher")})
                conn = None
                with trio.move_on_after(accept_timeout):
                    conn = await listeners[0].accept()
                for listener in listeners:
                    await listener.aclose()
                if conn is not None:
                    async with conn:
                        await pipe(sess, conn)
        except SessionRefused as e:
            if not ready.done():
                ready.set_result({"error": f"the provider refused the session: {e}"})
        except ConnectionError as e:
            if not ready.done():
                ready.set_result({"error": f"no path to the provider: {e}"})
        except TimeoutError as e:
            if not ready.done():
                ready.set_result({"error": str(e)})
        except Exception as e:  # noqa: BLE001
            log.warning("service session failed: %s", e)
            if not ready.done():
                ready.set_result({"error": f"session failed: {e}"})

    token, nursery = host._trio_token, host._nursery
    if token is None or nursery is None:
        return {"error": "libp2p host not running"}
    token.run_sync_soon(nursery.start_soon, session)
    try:
        out = await asyncio.wait_for(asyncio.wrap_future(ready), timeout + 60)
    except asyncio.TimeoutError:
        return {"error": "the session did not open in time"}
    if out.get("ok") and book is not None:
        out["credits"] = credits_view(book.get(payer, provider_address))
    return out


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

UNREACHABLE_ADVICE = (
    "Buyers reach this daemon only through a relay: its sessions go through "
    "someone else's bandwidth and are slower. Make it directly reachable: "
    "forward its p2p port on the router (or turn UPnP on there), or set "
    "p2p.announce_addrs / AUTONET_P2P_ANNOUNCE to a public address or name "
    "that reaches it.")


def own_reachability(p2p_host: Any = None) -> dict:
    """Whether this daemon can be dialled directly, for the app (a provider
    is told when it is not): ``{reachable, addrs, peer_id, advice?}``;
    ``reachable`` is None when libp2p is not running."""
    from . import service_client
    host = service_client._running_host(p2p_host)
    if host is None:
        return {"reachable": None, "addrs": [], "peer_id": "",
                "advice": "libp2p is not running on this daemon"}
    cap = getattr(host, "_capability", None)
    reachable = bool(getattr(cap, "reachable", False))
    out = {"reachable": reachable, "addrs": list(host.addrs), "peer_id": host.peer_id}
    if not reachable:
        out["advice"] = UNREACHABLE_ADVICE
    return out


def provider_reachable(provider_address: str, p2p_host: Any = None) -> Optional[bool]:
    """Whether the provider at ``provider_address`` says its daemon is
    directly reachable (its capability, or a relay's account of it); None
    when not known yet (it is looked up in the background)."""
    from . import service_client
    host = service_client._running_host(p2p_host)
    if host is None or not provider_address:
        return None
    attribution = getattr(host, "_peer_attribution", None)
    try:
        rec = attribution.peer_for_agent(provider_address) if attribution else None
    except Exception:  # noqa: BLE001
        rec = None
    if rec is None or not getattr(rec, "peer_id", ""):
        return None
    if rec.peer_id == host.peer_id:
        return bool(getattr(getattr(host, "_capability", None), "reachable", False))
    return host.peer_reachable(rec.peer_id)
