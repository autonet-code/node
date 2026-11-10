"""A two-way stream between a buyer's and a seller's daemons for a service
(docs/services_market.md, Decision 2026-11-10 "a stream between buyer and
seller"; docs/p2p_networking.md, "Service streams").

The network gives the two daemons a stream; what runs on it is the
parties' business (the spec's ``protocol`` names it: "openai-chat/1",
"webrtc-sdp/1", anything). Payment travels on the same stream, beside the
application's bytes.

Protocol ``/autonet/service-stream/1.0.0``. Every frame is
``[kind: 1 byte][length: 4 bytes big-endian][payload]``: kind ``C`` is a
JSON control message, kind ``D`` is application bytes.

1. The buyer's first frame (C) is the ``service_request`` frame the
   request/reply rail sends, with ``"type": "service_stream"``:
   ``spec_digest``, ``request_id``, ``args``, the payment proof (a channel
   voucher ``channel_id``, ``cumulative_amount``, ``signature``) and
   ``payer_signature`` over the request id, digest, args and provider.
2. The provider runs its payment gate on it before anything else and
   answers with one control frame, a reply signed like a service reply
   (``ok``, ``error``, ``result``, ``charged`` and ``receipt_sig``).
   ``result.units`` is how many units the opening payment covers (None:
   the service is free and unmetered). A refused stream ends there.
3. Application bytes flow both ways in D frames. The provider meters what
   it serves (``ServiceStream.consume``) and serves only while vouchers
   cover the usage: when coverage runs low it sends
   ``{"type": "voucher_needed", "units": n}``, the buyer answers with
   ``{"type": "voucher", ...}`` (another payment proof under a fresh
   request id, its ``args`` naming this stream), and the provider checks it
   with the same gate. Usage that no voucher covers within
   ``voucher_wait`` seconds closes the stream. A buyer that will not pay
   more says ``{"type": "no_voucher"}`` and still gets what it paid for:
   the provider closes once usage reaches the coverage.
4. Not served = not paid: a voucher is kept only once usage enters the
   units it covers. The provider says what it did with each voucher in a
   signed ``{"type": "receipt", ...}`` (``charged`` true: kept; false:
   released or refused). A voucher with no receipt when the stream ends
   may have been kept, so the buyer counts it.
5. Either side ends the stream with ``{"type": "close", "reason": ...}``.

The buyer's side calls back for payment through one interface:
``voucher_source(units) -> dict | None`` returns the payment fields of the
next voucher (``request_id``, ``args``, ``channel_id``,
``cumulative_amount``, ``signature``, ``payer_signature``), or None to stop
paying; ``on_receipt(request_id, charged)`` reports each voucher's outcome
(None: unknown); ``verify(reply, request_id)`` says whether a reply or
receipt is signed by the provider paid.

The provider's side is a ``StreamHandler`` (accept, check_voucher, keep,
release, sign, serve), which the daemon backs with its payment gate.
"""

from __future__ import annotations

import json
import logging
import struct
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import trio

try:
    from libp2p.custom_types import TProtocol
    SERVICE_STREAM_PROTOCOL = TProtocol("/autonet/service-stream/1.0.0")
except ImportError:  # pragma: no cover
    SERVICE_STREAM_PROTOCOL = "/autonet/service-stream/1.0.0"

log = logging.getLogger(__name__)

KIND_CONTROL = 0x43  # "C"
KIND_DATA = 0x44     # "D"
MAX_CONTROL_BYTES = 64 * 1024
MAX_DATA_BYTES = 1024 * 1024
WRITE_CHUNK = 65000  # Noise's 65535-byte limit with yamux's header on top
OPEN_TIMEOUT = 60.0          # the provider's gate reads the chain
VOUCHER_WAIT = 30.0          # unpaid usage this long ends the stream
DATA_BUFFER_FRAMES = 64


class StreamRefused(Exception):
    """The provider did not accept the stream (its reply is ``reply``)."""

    def __init__(self, reply: dict, message: str = "") -> None:
        super().__init__(message or str(reply.get("error") or "refused"))
        self.reply = reply


class StreamClosed(Exception):
    """The stream is closed (``reason`` says why)."""


class StreamUnpaid(StreamClosed):
    """Usage was not covered by a voucher in time; the stream is closed."""


@dataclass
class Accept:
    """The provider's answer to an open: ``reply`` goes to the buyer as is
    (signed); ``units`` is what the opening payment covers (None:
    unmetered); ``token`` is handed back to keep/release."""
    reply: dict
    units: Optional[int] = None
    token: Any = None


@dataclass
class VoucherCheck:
    ok: bool
    units: int = 0
    token: Any = None
    error: str = ""


class StreamHandler:
    """The provider's side of service streams. Every method runs in the
    host's trio loop. The default refuses everything."""

    async def accept(self, open_msg: dict) -> Accept:
        return Accept(reply={"ok": False, "charged": False,
                             "error": "this node serves no service streams"})

    async def check_voucher(self, open_msg: dict, frame: dict) -> VoucherCheck:
        return VoucherCheck(ok=False, error="no voucher gate")

    async def keep(self, open_msg: dict, token: Any) -> None:
        return None

    async def release(self, open_msg: dict, token: Any) -> None:
        return None

    def sign(self, open_msg: dict, request_id: str, reply: dict) -> dict:
        return reply

    async def serve(self, stream: "ServiceStream") -> None:
        return None


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

async def _read_exactly(raw: Any, n: int) -> bytes:
    parts: list = []
    remaining = n
    while remaining > 0:
        chunk = await raw.read(min(remaining, WRITE_CHUNK))
        if not chunk:
            raise EOFError("stream ended mid-frame")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


async def read_frame(raw: Any) -> tuple:
    """(kind, payload); raises EOFError at a clean end of stream."""
    header = b""
    while len(header) < 5:
        chunk = await raw.read(5 - len(header))
        if not chunk:
            if header:
                raise EOFError("stream ended mid-header")
            raise EOFError("end of stream")
        header += chunk
    kind, length = header[0], struct.unpack(">I", header[1:5])[0]
    limit = MAX_CONTROL_BYTES if kind == KIND_CONTROL else MAX_DATA_BYTES
    if kind not in (KIND_CONTROL, KIND_DATA):
        raise ValueError(f"unknown frame kind {kind}")
    if length > limit:
        raise ValueError(f"frame of {length} bytes exceeds {limit}")
    return kind, (await _read_exactly(raw, length) if length else b"")


async def write_frame(raw: Any, kind: int, payload: bytes) -> None:
    data = bytes([kind]) + struct.pack(">I", len(payload)) + payload
    for off in range(0, len(data), WRITE_CHUNK):
        await raw.write(data[off:off + WRITE_CHUNK])


def _control(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":"), default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------

@dataclass
class _Voucher:
    start: int            # usage at which this voucher's units begin
    units: int
    request_id: str
    token: Any = None
    kept: bool = False


class ServiceStream:
    """One side of an accepted service stream: ``send`` and ``receive``
    application bytes; the provider meters with ``consume``."""

    def __init__(self, raw: Any, role: str, open_msg: dict, *,
                 path: str = "", peer: str = "", relay: str = "",
                 accept_reply: Optional[dict] = None) -> None:
        self.raw = raw
        self.role = role
        self.open_msg = open_msg
        self.request_id = str(open_msg.get("request_id") or "")
        self.spec_digest = str(open_msg.get("spec_digest") or "")
        self.path = path          # "direct" or "relayed"
        self.relay = relay        # the relay's peer id on a relayed path
        self.peer = peer
        self.accept_reply = accept_reply or {}
        self.close_reason = ""
        self.closed = False
        self._send_lock = trio.Lock()
        self._data_tx, self._data_rx = trio.open_memory_channel(DATA_BUFFER_FRAMES)
        self._cancel: Optional[trio.CancelScope] = None
        # Provider-side metering.
        self.handler: Optional[StreamHandler] = None
        self.unmetered = True
        self.covered = 0
        self.used = 0
        self.increment = 1
        self.voucher_wait = VOUCHER_WAIT
        self._vouchers: deque = deque()
        self._changed = trio.Event()
        self._asked = False
        self._buyer_done = ""       # why the buyer pays no more
        # Buyer-side payment.
        self.voucher_source: Optional[Callable[[int], Awaitable[Optional[dict]]]] = None
        self.on_receipt: Optional[Callable[[str, Optional[bool]], Any]] = None
        self.verify: Optional[Callable[[dict, str], bool]] = None
        self.unsettled: dict = {}   # request id -> units, no receipt yet
        self.receipts: dict = {}    # request id -> charged

    # -- application side ---------------------------------------------------

    async def send(self, data: bytes) -> None:
        if self.closed:
            raise StreamClosed(self.close_reason or "stream closed")
        view = memoryview(bytes(data))
        for off in range(0, len(view), MAX_DATA_BYTES):
            await self._write(KIND_DATA, bytes(view[off:off + MAX_DATA_BYTES]))

    async def receive(self) -> bytes:
        """The next application bytes; b"" once the stream has ended."""
        try:
            return await self._data_rx.receive()
        except (trio.EndOfChannel, trio.ClosedResourceError):
            return b""

    async def consume(self, units: int = 1) -> None:
        """Provider: account for ``units`` served. Waits for a voucher when
        the usage would pass what vouchers cover; raises ``StreamUnpaid``
        (and closes the stream) when none comes within ``voucher_wait``."""
        units = int(units)
        if units <= 0:
            return
        if self.closed:
            raise StreamClosed(self.close_reason or "stream closed")
        if self.unmetered:
            self.used += units
            return
        with trio.move_on_after(self.voucher_wait):
            while (self.used + units > self.covered and not self.closed
                   and not self._buyer_done):
                await self._ask()
                changed = self._changed
                await changed.wait()
        if self.closed:
            raise StreamClosed(self.close_reason or "stream closed")
        if self.used + units > self.covered:
            reason = ("the buyer stopped paying: " + self._buyer_done
                      if self._buyer_done else "usage not covered by vouchers")
            await self.aclose(reason)
            raise StreamUnpaid(reason)
        self.used += units
        await self._keep_entered()
        if self.covered - self.used < max(1, self.increment // 2):
            await self._ask()

    async def aclose(self, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        self.close_reason = reason or "closed"
        if self.role == "provider":
            await self._release_unentered()
        with trio.CancelScope(shield=True), trio.move_on_after(3):
            try:
                await self._write(KIND_CONTROL, _control(
                    {"type": "close", "reason": self.close_reason}), force=True)
            except Exception:  # noqa: BLE001 (the far end is gone)
                pass
            try:
                await self.raw.close()
            except Exception:  # noqa: BLE001
                pass
        self._data_tx.close()
        self._signal()
        # The provider's application runs inside the stream's nursery, so it
        # stops with the stream. The buyer's body is the caller's own code:
        # it sees the end as receive() returning b"" (open_on cancels the
        # reader when the body exits).
        if self.role == "provider" and self._cancel is not None:
            self._cancel.cancel()

    # -- internals ------------------------------------------------------------

    def _signal(self) -> None:
        self._changed.set()
        self._changed = trio.Event()

    async def _write(self, kind: int, payload: bytes, force: bool = False) -> None:
        if self.closed and not force:
            raise StreamClosed(self.close_reason or "stream closed")
        async with self._send_lock:
            await write_frame(self.raw, kind, payload)

    async def _send_control(self, msg: dict) -> None:
        await self._write(KIND_CONTROL, _control(msg))

    async def _ask(self) -> None:
        if self._asked or self.closed or self._buyer_done:
            return
        self._asked = True
        await self._send_control({"type": "voucher_needed",
                                  "units": self.increment,
                                  "used": self.used, "covered": self.covered})

    async def _receipt(self, request_id: str, ok: bool, charged: bool,
                       error: str = "", units: int = 0) -> None:
        reply = {"ok": ok, "charged": charged, "error": error or None,
                 "result": {"request_id": request_id, "units": units,
                            "covered": self.covered, "used": self.used}}
        if self.handler is not None:
            reply = self.handler.sign(self.open_msg, request_id, reply)
        reply["type"] = "receipt"
        reply["request_id"] = request_id
        try:
            await self._write(KIND_CONTROL, _control(reply), force=True)
        except Exception:  # noqa: BLE001
            pass

    async def _keep_entered(self) -> None:
        for v in self._vouchers:
            if not v.kept and self.used > v.start:
                v.kept = True
                if self.handler is not None:
                    await self.handler.keep(self.open_msg, v.token)
                await self._receipt(v.request_id, True, True, units=v.units)

    async def _release_unentered(self) -> None:
        for v in self._vouchers:
            if not v.kept:
                v.kept = True  # settled either way
                if self.handler is not None:
                    try:
                        await self.handler.release(self.open_msg, v.token)
                    except Exception as e:  # noqa: BLE001
                        log.warning("releasing a stream voucher failed: %s", e)
                await self._receipt(v.request_id, True, False, units=v.units)

    async def _on_voucher(self, frame: dict) -> None:
        rid = str(frame.get("request_id") or "")
        self._asked = False
        check = await self.handler.check_voucher(self.open_msg, frame) \
            if self.handler is not None else VoucherCheck(ok=False, error="no gate")
        if not check.ok or check.units <= 0:
            await self._receipt(rid, False, False, error=check.error or "voucher refused")
            await self.aclose(f"voucher refused: {check.error or 'invalid'}")
            return
        self._vouchers.append(_Voucher(start=self.covered, units=int(check.units),
                                       request_id=rid, token=check.token))
        self.covered += int(check.units)
        self._signal()

    async def _on_voucher_needed(self, msg: dict) -> None:
        try:
            units = max(1, int(msg.get("units") or 1))
        except (TypeError, ValueError):
            units = 1
        fields = None
        if self.voucher_source is not None:
            try:
                fields = await self.voucher_source(units)
            except Exception as e:  # noqa: BLE001
                log.warning("the stream's voucher source failed: %s", e)
                fields = None
        if not fields or fields.get("error"):
            # What is paid for is still served; the provider ends the
            # stream when usage reaches it.
            reason = str((fields or {}).get("error") or "no more vouchers")
            await self._send_control({"type": "no_voucher", "reason": reason})
            return
        frame = {**fields, "type": "voucher", "units": units}
        rid = str(frame.get("request_id") or "")
        self.unsettled[rid] = units
        await self._send_control(frame)

    def _on_receipt(self, msg: dict) -> None:
        rid = str(msg.get("request_id") or "")
        if rid not in self.unsettled and rid != self.request_id:
            return
        if self.verify is not None and not self.verify(msg, rid):
            log.warning("a stream receipt for %s is not signed by the provider",
                        rid[:18])
            return
        charged = msg.get("charged") if isinstance(msg.get("charged"), bool) else None
        self.unsettled.pop(rid, None)
        self.receipts[rid] = charged
        if self.on_receipt is not None:
            try:
                self.on_receipt(rid, charged)
            except Exception as e:  # noqa: BLE001
                log.warning("on_receipt failed: %s", e)

    async def _reader(self) -> None:
        try:
            while True:
                kind, payload = await read_frame(self.raw)
                if kind == KIND_DATA:
                    await self._data_tx.send(payload)
                    continue
                msg = json.loads(payload.decode("utf-8"))
                if not isinstance(msg, dict):
                    continue
                kind_ = msg.get("type")
                if kind_ == "close":
                    self.close_reason = str(msg.get("reason") or "closed by the peer")
                    break
                if self.role == "provider" and kind_ == "voucher":
                    await self._on_voucher(msg)
                elif self.role == "provider" and kind_ == "no_voucher":
                    self._buyer_done = str(msg.get("reason") or "no more vouchers")
                    self._asked = False
                    self._signal()
                elif self.role == "buyer" and kind_ == "voucher_needed":
                    await self._on_voucher_needed(msg)
                elif self.role == "buyer" and kind_ == "receipt":
                    self._on_receipt(msg)
        except (EOFError, trio.ClosedResourceError, trio.BrokenResourceError):
            pass
        except Exception as e:  # noqa: BLE001 (a broken or reset stream)
            log.debug("service stream reader ended: %s", e)
            if not self.close_reason:
                self.close_reason = f"stream broke: {e}"
        await self.aclose(self.close_reason or "closed by the peer")

    def _settle_unknown(self) -> None:
        """Buyer: vouchers with no receipt may have been kept."""
        for rid in list(self.unsettled):
            self.unsettled.pop(rid, None)
            self.receipts[rid] = None
            if self.on_receipt is not None:
                try:
                    self.on_receipt(rid, None)
                except Exception as e:  # noqa: BLE001
                    log.warning("on_receipt failed: %s", e)


# ---------------------------------------------------------------------------
# Opening (buyer) and serving (provider)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def open_on(raw: Any, open_msg: dict, *,
                  verify: Optional[Callable[[dict, str], bool]] = None,
                  voucher_source: Optional[Callable[[int], Awaitable[Optional[dict]]]] = None,
                  on_receipt: Optional[Callable[[str, Optional[bool]], Any]] = None,
                  timeout: float = OPEN_TIMEOUT, path: str = "", peer: str = "",
                  relay: str = "") -> AsyncIterator[ServiceStream]:
    """Buyer: send the opening frame on ``raw`` and yield the accepted
    stream. Raises ``StreamRefused`` when the provider refuses (or its
    reply is not signed by the provider), ``TimeoutError`` when no reply
    comes. ``on_receipt`` hears the opening payment's outcome too."""
    msg = dict(open_msg)
    msg["type"] = "service_stream"
    rid = str(msg.get("request_id") or "")
    paid = bool(msg.get("channel_id") is not None or msg.get("tx_hash"))
    stream = ServiceStream(raw, "buyer", msg, path=path, peer=peer, relay=relay)
    stream.verify, stream.voucher_source, stream.on_receipt = verify, voucher_source, on_receipt
    try:
        await write_frame(raw, KIND_CONTROL, _control(msg))
    except Exception as e:  # noqa: BLE001
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise ConnectionError(f"could not send the stream's opening: {e}") from e
    if paid:
        stream.unsettled[rid] = None
    try:
        with trio.fail_after(timeout):
            while True:
                kind, payload = await read_frame(raw)
                if kind == KIND_CONTROL:
                    break
        reply = json.loads(payload.decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("the reply is not an object")
    except trio.TooSlowError:
        stream._settle_unknown()
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise TimeoutError(f"the provider did not answer the stream within {int(timeout)}s") from None
    except Exception as e:  # noqa: BLE001
        stream._settle_unknown()
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise TimeoutError(f"the stream broke before the provider answered: {e}") from e
    authentic = verify is None or verify(reply, rid)
    if not authentic or not reply.get("ok"):
        if authentic and paid:
            charged = reply.get("charged") if isinstance(reply.get("charged"), bool) else None
            stream.unsettled.pop(rid, None)
            if on_receipt is not None:
                on_receipt(rid, charged)
        else:
            stream._settle_unknown()
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise StreamRefused(reply, "the reply is not signed by the provider"
                            if not authentic else "")
    stream.accept_reply = reply
    if isinstance(reply.get("charged"), bool) and paid:
        stream._on_receipt({**reply, "request_id": rid})
    try:
        async with trio.open_nursery() as nursery:
            stream._cancel = nursery.cancel_scope
            nursery.start_soon(stream._reader)
            try:
                yield stream
            finally:
                await stream.aclose(stream.close_reason or "closed by the buyer")
                nursery.cancel_scope.cancel()
    finally:
        stream._settle_unknown()


async def serve_incoming(raw: Any, handler: Optional[StreamHandler], *,
                         path: str = "", peer: str = "") -> None:
    """Provider: read the opening, run the gate, and serve an accepted
    stream until either side ends it."""
    handler = handler or StreamHandler()
    try:
        with trio.fail_after(OPEN_TIMEOUT):
            kind, payload = await read_frame(raw)
        if kind != KIND_CONTROL:
            raise ValueError("the first frame must be the opening")
        open_msg = json.loads(payload.decode("utf-8"))
        if not isinstance(open_msg, dict) or open_msg.get("type") != "service_stream":
            raise ValueError("the first frame must be a service_stream opening")
    except Exception as e:  # noqa: BLE001
        log.debug("bad service stream opening: %s", e)
        try:
            await write_frame(raw, KIND_CONTROL, _control(
                {"ok": False, "charged": False, "error": f"bad opening: {e}"}))
        except Exception:  # noqa: BLE001
            pass
        await _quiet_close(raw)
        return
    try:
        accepted = await handler.accept(open_msg)
    except Exception as e:  # noqa: BLE001
        log.warning("service stream gate failed: %s", e)
        accepted = Accept(reply={"ok": False, "charged": False,
                                 "error": f"provider failed to check the stream: {e}"})
    reply = dict(accepted.reply)
    reply["type"] = "service_stream_reply"
    stream = ServiceStream(raw, "provider", open_msg, path=path, peer=peer,
                           accept_reply=reply)
    stream.handler = handler
    ok = bool(reply.get("ok"))
    if ok and accepted.units is not None:
        stream.unmetered = False
        stream.increment = max(1, int(accepted.units))
        stream._vouchers.append(_Voucher(start=0, units=int(accepted.units),
                                         request_id=stream.request_id,
                                         token=accepted.token))
        stream.covered = int(accepted.units)
    try:
        await write_frame(raw, KIND_CONTROL, _control(reply))
    except Exception as e:  # noqa: BLE001
        log.debug("could not answer a service stream: %s", e)
        ok = False
    if not ok:
        if accepted.units is not None and accepted.token is not None:
            await handler.release(open_msg, accepted.token)
        await _quiet_close(raw)
        return
    try:
        async with trio.open_nursery() as nursery:
            stream._cancel = nursery.cancel_scope
            nursery.start_soon(stream._reader)
            try:
                await handler.serve(stream)
            except StreamClosed:
                pass
            except Exception as e:  # noqa: BLE001
                log.warning("service stream application failed: %s", e)
                stream.close_reason = f"provider failed: {e}"
            finally:
                await stream.aclose(stream.close_reason or "closed by the provider")
    finally:
        if not stream.closed:
            await stream.aclose("ended")
        else:
            # Released inside aclose; a cancellation may have skipped it.
            with trio.CancelScope(shield=True):
                await stream._release_unentered()


async def _quiet_close(raw: Any) -> None:
    with trio.move_on_after(3):
        try:
            await raw.close()
        except Exception:  # noqa: BLE001
            pass
