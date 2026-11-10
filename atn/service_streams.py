"""The daemon's side of service streams (docs/services_market.md, Decision
2026-11-10 "a stream between buyer and seller"). The network layer, the
wire and the voucher interface are ``nodes/common/service_stream.py``.

Provider (``DaemonStreamHandler``): the stream's opening and every voucher
sent on it go through the gate the request/reply rail runs
(``WebSocketBridge._validate_service_payment``: on-chain price and payee,
the channel voucher, the payer's signature, the replay guard). A voucher is
reserved when accepted and kept (``ServiceStore.keep_channel_voucher``)
only once usage enters the units it covers; the rest are released. An
accepted stream is piped to the service's backend: for a stream service
(its spec names a ``protocol``) the ``backing_tool`` is a local address,
``tcp://127.0.0.1:<port>``, where the owner runs the application (an
inference server, a signaling server). The pipe meters by the spec's
``unit``: a time unit (second, minute, hour) is consumed once per period
while the stream is open, a size unit (byte, KB, MB, GB, KiB, MiB, GiB)
per started unit of bytes carried either way, and anything else (per
item, or none) once for the whole stream.

Buyer (``open_local_stream``): opens a paid stream to a provider and hands
it to a local program on a fresh 127.0.0.1 port (one connection). Vouchers
come from the household's payment channel (``service_channels``), one at a
time per channel, as the provider asks for them.

Reachability (decision 3): ``own_reachability()`` tells the app whether
this daemon can be dialled directly; ``provider_reachable(address)`` is
what catalog rows show for a provider.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid
from typing import Any, Optional

log = logging.getLogger(__name__)

STREAM_BACKEND_SCHEME = "tcp://"
LOCAL_BACKEND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
GATE_TIMEOUT = 180.0        # the gate reads the chain
STORE_TIMEOUT = 30.0
PIPE_READ = 65536

_TIME_UNITS = {"second": 1, "seconds": 1, "sec": 1, "s": 1,
               "minute": 60, "minutes": 60, "min": 60,
               "hour": 3600, "hours": 3600, "h": 3600}
_SIZE_UNITS = {"byte": 1, "bytes": 1, "b": 1,
               "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3,
               "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}


def stream_backend(record: Any) -> Optional[tuple]:
    """``(host, port)`` of a stream service's local backend, or None.
    Only loopback: the daemon pipes paid streams to what its owner runs on
    this machine, never to another host."""
    backing = str(getattr(record, "backing_tool", "") or "").strip()
    if not backing.lower().startswith(STREAM_BACKEND_SCHEME):
        return None
    rest = backing[len(STREAM_BACKEND_SCHEME):].rstrip("/")
    host, _, port = rest.rpartition(":")
    host = host.strip("[]").lower()
    if host not in LOCAL_BACKEND_HOSTS:
        return None
    try:
        port_n = int(port)
    except ValueError:
        return None
    return (host, port_n) if 0 < port_n < 65536 else None


def meter_for(spec: Any) -> tuple:
    """How the backend pipe meters a stream: ("time", seconds per unit),
    ("size", bytes per unit) or ("stream", 0): one unit for the stream."""
    unit = spec.get("unit") if isinstance(spec, dict) else None
    name = str(unit or "").strip().lower().replace(" ", "_")
    for prefix in ("per_", "/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name in _TIME_UNITS:
        return ("time", _TIME_UNITS[name])
    if name in _SIZE_UNITS:
        return ("size", _SIZE_UNITS[name])
    return ("stream", 0)


def _previous_cumulative(store: Any, channel_id: Any, token: Any) -> int:
    """The channel's cumulative before the voucher the gate just reserved
    under ``token``: the best kept voucher and the other pending ones.
    Read right after the gate returns, with no await in between, so it is
    what the gate compared the voucher with."""
    cid = str(channel_id)
    pending = (getattr(store, "_pending_vouchers", {}) or {}).get(cid) or {}
    others = [int(p.get("cumulative", 0)) for t, p in pending.items() if t != token]
    return max([int(store.kept_channel_cumulative(cid))] + others)


def _voucher_units(store: Any, gate: dict, msg: dict) -> int:
    """Units a voucher pays for: the gate's own count when it gives one,
    else the voucher's increment over the channel divided by the price."""
    units = gate.get("units")
    if isinstance(units, int) and not isinstance(units, bool) and units > 0:
        return units
    ask = int(gate.get("ask_amount") or 0)
    if ask <= 0:
        return 1
    res = gate.get("reservation") or {}
    prev = _previous_cumulative(store, msg.get("channel_id"), res.get("token"))
    return max(1, (int(msg.get("cumulative_amount") or 0) - prev) // ask)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

def stream_handler_for(serve: Any, loop: Any) -> Any:
    """The provider's ``StreamHandler`` for the WS bridge behind ``serve``
    (its bound ``_serve_p2p_service_request``), or None."""
    bridge = getattr(serve, "__self__", None)
    if bridge is None or loop is None:
        return None
    return DaemonStreamHandler(bridge, loop)


try:
    from nodes.common.service_stream import (
        Accept,
        StreamClosed,
        StreamHandler,
        VoucherCheck,
    )
except Exception:  # pragma: no cover (nodes ships with the daemon)
    StreamHandler = object  # type: ignore[assignment,misc]


class DaemonStreamHandler(StreamHandler):  # type: ignore[misc,valid-type]
    """Serves this daemon's stream services (trio side), with the WS
    bridge's payment gate run on the daemon's asyncio loop."""

    def __init__(self, bridge: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.bridge = bridge
        self.loop = loop
        self._payers: dict = {}   # opening request id -> payer the gate verified

    async def _on_loop(self, factory: Any, timeout: float) -> Any:
        import trio
        loop = self.loop

        def run() -> Any:
            return asyncio.run_coroutine_threadsafe(factory(), loop).result(timeout)
        # Not abandoned on cancel: a reservation the gate made must reach
        # the stream, or it would never be released.
        return await trio.to_thread.run_sync(run)

    @property
    def _store(self) -> Any:
        return self.bridge.runtime.service_store

    # -- gate ---------------------------------------------------------------

    async def accept(self, open_msg: dict) -> Any:
        return await self._on_loop(lambda: self._accept(open_msg), GATE_TIMEOUT)

    async def _accept(self, msg: dict) -> Any:
        from .service_spec import stream_protocol

        bridge, store = self.bridge, self._store
        digest = str(msg.get("spec_digest") or "")
        rid = str(msg.get("request_id") or "")

        def refuse(error: str) -> Any:
            return Accept(reply=bridge._sign_service_reply(msg, {
                "msg_id": msg.get("msg_id"), "ok": False, "charged": False,
                "error": error}))
        if not digest or not rid:
            return refuse("Missing 'spec_digest' or 'request_id' field")
        record = store.get(digest)
        if record is None:
            return refuse(f"Unknown service digest: {digest[:16]}")
        if record.retired:
            store.count_refused(digest)
            return refuse("Service retired")
        protocol = stream_protocol(record.spec)
        if not protocol:
            store.count_refused(digest)
            return refuse("this service is not a stream (its spec names no "
                          "protocol): send a service_request")
        if stream_backend(record) is None:
            store.count_refused(digest)
            return refuse("the service has no stream backend (its backing must "
                          "be a local address, tcp://127.0.0.1:<port>)")
        if not isinstance(msg.get("args", {}), dict):
            store.count_refused(digest)
            return refuse("'args' must be an object")
        if msg.get("tx_hash"):
            store.count_refused(digest)
            return refuse("a stream is paid with payment-channel vouchers, "
                          "not a payForService transfer; the payment was not "
                          "taken")
        gate_msg = {**msg, "type": "service_request"}
        gate = await bridge._validate_service_payment(gate_msg, record, local=False)
        if not gate.get("ok"):
            store.count_refused(digest)
            return refuse(f"Payment validation failed: {gate.get('reason', 'unknown')}")
        proof = gate.get("proof")
        units: Optional[int] = None
        token: Any = None
        if proof == "voucher":
            units = _voucher_units(store, gate, msg)
            payer = bridge._verified_payer(gate_msg, record, gate)
            self._payers[rid] = payer
            token = {"reservation": gate.get("reservation"), "payer": payer,
                     "request_id": rid,
                     "amount": units * int(gate.get("ask_amount") or 0)}
        elif proof != "free":
            if gate.get("reservation") is not None:
                store.release_channel_voucher(gate["reservation"])
            return refuse(f"a stream cannot be paid that way ({proof})")
        reply = {"msg_id": msg.get("msg_id"), "ok": True, "charged": None,
                 "result": {"request_id": rid, "units": units,
                            "unit": record.spec.get("unit") or "per_item",
                            "protocol": protocol}}
        return Accept(reply=bridge._sign_service_reply(msg, reply),
                      units=units, token=token)

    async def check_voucher(self, open_msg: dict, frame: dict) -> Any:
        return await self._on_loop(lambda: self._check(open_msg, frame), GATE_TIMEOUT)

    async def _check(self, open_msg: dict, frame: dict) -> Any:
        bridge, store = self.bridge, self._store
        open_rid = str(open_msg.get("request_id") or "")
        args = frame.get("args")
        if not isinstance(args, dict) or str(args.get("stream") or "") != open_rid:
            return VoucherCheck(ok=False, error="a stream voucher must name its "
                                                "stream (args.stream)")
        if frame.get("channel_id") is None:
            return VoucherCheck(ok=False, error="a stream voucher must be a "
                                                "channel voucher")
        digest = str(open_msg.get("spec_digest") or "")
        record = store.get(digest)
        if record is None or record.retired:
            return VoucherCheck(ok=False, error="the service is retired")
        msg = {"type": "service_request", "spec_digest": digest,
               "request_id": str(frame.get("request_id") or ""), "args": args,
               "channel_id": frame.get("channel_id"),
               "cumulative_amount": frame.get("cumulative_amount"),
               "signature": frame.get("signature"),
               "payer_signature": frame.get("payer_signature")}
        gate = await bridge._validate_service_payment(msg, record, local=False)
        if not gate.get("ok"):
            store.count_refused(digest)
            return VoucherCheck(ok=False, error=f"Payment validation failed: "
                                                f"{gate.get('reason', 'unknown')}")
        reservation = gate.get("reservation")
        if gate.get("proof") != "voucher" or reservation is None:
            return VoucherCheck(ok=False, error="a stream voucher must be a "
                                                "channel voucher")
        units = _voucher_units(store, gate, msg)
        payer = bridge._verified_payer(msg, record, gate)
        first = self._payers.get(open_rid, "")
        if first and payer.lower() != first.lower():
            store.release_channel_voucher(reservation)
            return VoucherCheck(ok=False, error="the voucher is not from the "
                                                "stream's payer")
        return VoucherCheck(ok=True, units=units, token={
            "reservation": reservation, "payer": payer,
            "request_id": msg["request_id"],
            "amount": units * int(gate.get("ask_amount") or 0)})

    async def keep(self, open_msg: dict, token: Any) -> None:
        if not token or token.get("reservation") is None:
            return
        digest = str(open_msg.get("spec_digest") or "")

        async def run() -> None:
            store = self._store
            store.keep_channel_voucher(token["reservation"])
            store.record_request(digest, token["request_id"], token.get("payer", ""),
                                 ok=True, amount=str(token.get("amount") or 0))
        await self._on_loop(run, STORE_TIMEOUT)

    async def release(self, open_msg: dict, token: Any) -> None:
        if not token or token.get("reservation") is None:
            return

        async def run() -> None:
            self._store.release_channel_voucher(token["reservation"])
        await self._on_loop(run, STORE_TIMEOUT)

    async def sign(self, open_msg: dict, request_id: str, reply: dict) -> dict:
        msg = {"spec_digest": open_msg.get("spec_digest"), "request_id": request_id}

        async def run() -> dict:
            return self.bridge._sign_service_reply(msg, reply)
        return await self._on_loop(run, STORE_TIMEOUT)

    # -- application: the local backend ---------------------------------------

    async def serve(self, stream: Any) -> None:
        import trio

        record = self._store.get(stream.spec_digest)
        backend = stream_backend(record) if record is not None else None
        if backend is None:
            await stream.aclose("the service has no stream backend")
            return
        kind, size = meter_for(record.spec)
        try:
            tcp = await trio.open_tcp_stream(*backend)
        except OSError as e:
            await stream.aclose(f"the provider's backend is not running: {e}")
            return
        try:
            await pipe(stream, tcp, kind=kind, size=size)
        finally:
            self._payers.pop(stream.request_id, None)
            with trio.CancelScope(shield=True), trio.move_on_after(3):
                try:
                    await tcp.aclose()
                except Exception:  # noqa: BLE001
                    pass


async def pipe(stream: Any, local: Any, *, kind: str = "stream", size: int = 0,
               meter: bool = True) -> None:
    """Copy bytes between a service stream and a local trio stream until
    either ends. On the provider (``meter``) usage is consumed as
    ``meter_for`` says before it is served; a usage no voucher covers ends
    the stream."""
    import trio

    pending = [0]

    async def take(nbytes: int) -> None:
        if not meter or kind != "size":
            return
        pending[0] += nbytes
        while pending[0] > 0:
            await stream.consume(1)
            pending[0] -= size

    async def up(done: Any) -> None:
        try:
            while True:
                data = await stream.receive()
                if not data:
                    break
                await take(len(data))
                await local.send_all(data)
        except (StreamClosed, trio.BrokenResourceError, trio.ClosedResourceError):
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("stream to local pipe ended: %s", e)
        done.cancel()

    async def down(done: Any) -> None:
        try:
            while True:
                data = await local.receive_some(PIPE_READ)
                if not data:
                    break
                await take(len(data))
                await stream.send(data)
        except (StreamClosed, trio.BrokenResourceError, trio.ClosedResourceError):
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("local to stream pipe ended: %s", e)
        done.cancel()

    async def clock(done: Any) -> None:
        try:
            while True:
                await stream.consume(1)
                await trio.sleep(size)
        except StreamClosed:
            pass
        done.cancel()

    try:
        if meter and kind == "stream":
            await stream.consume(1)
        async with trio.open_nursery() as nursery:
            if meter and kind == "time":
                nursery.start_soon(clock, nursery.cancel_scope)
            nursery.start_soon(up, nursery.cancel_scope)
            nursery.start_soon(down, nursery.cancel_scope)
    except StreamClosed:
        pass


# ---------------------------------------------------------------------------
# Buyer
# ---------------------------------------------------------------------------

class _Payer:
    """The buyer's vouchers for one stream: issued one at a time on the
    household's channel to the provider, written down before they go out,
    counted or dropped as the provider's receipts say."""

    def __init__(self, runtime: Any, *, payer_key: str, payer: str, provider: str,
                 spec_digest: str, ask: int, deposit: int,
                 loop: asyncio.AbstractEventLoop) -> None:
        self.runtime, self.payer_key, self.payer = runtime, payer_key, payer
        self.provider, self.spec_digest = provider, spec_digest
        self.ask, self.deposit, self.loop = int(ask), int(deposit or 0), loop
        self.outstanding: dict = {}   # request id -> voucher
        self.paid = 0

    def _channel_lock(self) -> asyncio.Lock:
        from .agent_tools import _channel_lock
        return _channel_lock(self.runtime, self.payer, self.provider)

    async def voucher(self, units: int, args: dict, request_id: str = "") -> dict:
        """The payment fields of the next voucher (asyncio), signed under
        ``request_id`` (a fresh one by default). Holds the channel until the
        provider's receipt for it comes back."""
        from .on_chain import ServiceMarketClient
        from .service_channels import voucher_for
        from .service_client import new_request_id, sign_request

        book = getattr(self.runtime, "channel_book", None)
        smc = ServiceMarketClient(self.runtime._config.rpb)
        if book is None or not smc.channel_available:
            return {"error": "PaymentChannel not configured on this daemon"}
        lock = self._channel_lock()
        await lock.acquire()
        try:
            v = await voucher_for(book, smc, self.payer_key, self.payer,
                                  self.provider, self.ask * max(1, int(units)),
                                  self.deposit)
            if v.get("error"):
                lock.release()
                return {"error": str(v["error"])}
            book.reserve(self.payer, self.provider, v["channel_id"],
                         v["cumulative_amount"])
        except BaseException:
            lock.release()
            raise
        rid = request_id or new_request_id()
        self.outstanding[rid] = {**v, "units": int(units), "lock": lock}
        return {"request_id": rid, "args": args,
                "channel_id": v["channel_id"],
                "cumulative_amount": v["cumulative_amount"],
                "signature": v["signature"],
                "payer_signature": sign_request(self.payer_key, rid, self.spec_digest,
                                                args, self.provider)}

    def settle(self, request_id: str, charged: Optional[bool]) -> None:
        """A voucher's outcome (asyncio thread): counted when kept or
        unknown, dropped when the provider says it kept nothing."""
        v = self.outstanding.pop(request_id, None)
        if v is None:
            return
        book = getattr(self.runtime, "channel_book", None)
        try:
            if book is not None:
                if charged is False:
                    book.release(self.payer, self.provider, v["channel_id"])
                else:
                    book.commit(self.payer, self.provider, v["channel_id"],
                                v["cumulative_amount"])
                    self.paid += self.ask * int(v.get("units") or 1)
        finally:
            lock = v.get("lock")
            if lock is not None and lock.locked():
                lock.release()

    def settle_threadsafe(self, request_id: str, charged: Optional[bool]) -> None:
        self.loop.call_soon_threadsafe(self.settle, request_id, charged)

    def finish(self, delivered: bool) -> None:
        """Vouchers still outstanding when the stream is over (asyncio)."""
        for rid in list(self.outstanding):
            self.settle(rid, None if delivered else False)


async def open_local_stream(runtime: Any, *, payer_key: str, payer: str,
                            provider_address: str, spec_digest: str,
                            args: Optional[dict] = None, units: int = 1,
                            deposit: int = 0, ask: Optional[int] = None,
                            p2p_host: Any = None, peer_id: str = "",
                            timeout: float = 60.0,
                            accept_timeout: float = 300.0) -> dict:
    """Buyer: open a paid stream to ``provider_address``'s service and
    expose it on 127.0.0.1 for one local connection.

    ``units`` is how many units each voucher pays for (the opening and
    every later one). Returns ``{ok, host, port, path, relay, request_id,
    protocol, units}`` once the provider accepted, or ``{"error": ...}``.
    The stream lasts while the local connection is open; a port nobody
    connects to within ``accept_timeout`` closes it unused (nothing kept)."""
    from . import service_client

    args = dict(args or {})
    host = service_client._running_host(p2p_host)
    if host is None:
        return {"error": "libp2p is not running on this daemon, so it cannot "
                         "open a stream"}
    config = runtime._config.rpb
    peer = peer_id or await service_client.resolve_provider_peer_id(
        config, provider_address, host)
    if not peer:
        return {"error": "no peer id on chain for the provider"}
    if ask is None:
        found = await service_client.lookup_service_ask(config, provider_address,
                                                        spec_digest)
        if found.get("error"):
            return {"error": str(found["error"])}
        if not found.get("active"):
            return {"error": "that service is retired"}
        ask = int(found.get("ask_amount") or 0)
    loop = asyncio.get_running_loop()
    pay = _Payer(runtime, payer_key=payer_key, payer=payer, provider=provider_address,
                 spec_digest=spec_digest, ask=int(ask), deposit=deposit, loop=loop)
    units = max(1, int(units or 1))
    open_rid = service_client.new_request_id()
    msg = {"type": "service_request", "msg_id": uuid.uuid4().hex[:12],
           "spec_digest": spec_digest, "request_id": open_rid, "args": args,
           "client": payer, "client_address": payer, "units": units}
    if int(ask) > 0:
        if not payer_key or not payer:
            return {"error": "paying for a stream needs a key (call as an agent)"}
        fields = await pay.voucher(units, args, request_id=open_rid)
        if fields.get("error"):
            return {"error": str(fields["error"])}
        msg.update({k: fields[k] for k in ("channel_id", "cumulative_amount",
                                           "signature", "payer_signature")})

    def verify(reply: dict, request_id: str) -> bool:
        return service_client.reply_from_provider(reply, request_id, spec_digest,
                                                  provider_address)

    ready: concurrent.futures.Future = concurrent.futures.Future()
    delivered = [False]

    async def voucher_source(n: int) -> Optional[dict]:
        import trio
        if int(ask) <= 0:
            return None

        def run() -> dict:
            fut = asyncio.run_coroutine_threadsafe(
                pay.voucher(n, {"stream": open_rid, "units": n}), loop)
            try:
                return fut.result(120)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                return {"error": "no voucher could be made in time"}
        return await trio.to_thread.run_sync(run)

    async def session() -> None:
        import trio
        from nodes.common.service_stream import StreamRefused
        try:
            async with host.service_stream(
                    peer, msg, verify=verify, voucher_source=voucher_source,
                    on_receipt=pay.settle_threadsafe, timeout=timeout) as stream:
                delivered[0] = True
                listeners = await trio.open_tcp_listeners(0, host="127.0.0.1")
                port = listeners[0].socket.getsockname()[1]
                result = stream.accept_reply.get("result") or {}
                ready.set_result({"ok": True, "host": "127.0.0.1", "port": port,
                                  "path": stream.path, "relay": stream.relay,
                                  "request_id": open_rid,
                                  "protocol": result.get("protocol"),
                                  "units": result.get("units"),
                                  "unit": result.get("unit")})
                conn = None
                with trio.move_on_after(accept_timeout):
                    conn = await listeners[0].accept()
                for listener in listeners:
                    await listener.aclose()
                if conn is not None:
                    async with conn:
                        await pipe(stream, conn, meter=False)
        except StreamRefused as e:
            delivered[0] = True
            if not ready.done():
                ready.set_result({"error": f"the provider refused the stream: {e}",
                                  "authenticated": verify(e.reply, open_rid)})
        except ConnectionError as e:
            if not ready.done():
                ready.set_result({"error": f"no path to the provider: {e}"})
        except TimeoutError as e:
            delivered[0] = True
            if not ready.done():
                ready.set_result({"error": str(e)})
        except Exception as e:  # noqa: BLE001
            log.warning("service stream session failed: %s", e)
            if not ready.done():
                ready.set_result({"error": f"stream failed: {e}"})
        finally:
            done = delivered[0]
            loop.call_soon_threadsafe(pay.finish, done)

    token, nursery = host._trio_token, host._nursery
    if token is None or nursery is None:
        pay.finish(False)
        return {"error": "libp2p host not running"}
    token.run_sync_soon(nursery.start_soon, session)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(ready), timeout + 60)
    except asyncio.TimeoutError:
        return {"error": "the stream did not open in time"}


# ---------------------------------------------------------------------------
# Reachability (decision 3)
# ---------------------------------------------------------------------------

UNREACHABLE_ADVICE = (
    "Buyers reach this daemon only through a relay: its streams go through "
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
