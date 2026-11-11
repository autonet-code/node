"""A session between a buyer's and a provider's daemons for a service the
buyer keeps open (docs/services_market.md, Decision 2026-11-12;
docs/p2p_networking.md, "Service sessions").

The network gives the two daemons a two-way connection; what runs on it is
the parties' business: the tool behind the service names the protocol
("openai-chat/1", "ssh/2", anything) in its manifest. Payment travels on
the same connection, beside the application's bytes.

Protocol ``/autonet/service-session/1.0.0``. Every frame is
``[kind: 1 byte][length: 4 bytes big-endian][payload]``: kind ``C`` is a
JSON control message, kind ``D`` is application bytes.

1. The buyer's first frame (C) is the opening, ``"type": "service_session"``:
   ``spec_digest``, ``request_id``, ``args`` (the opening arguments),
   ``channel_id`` (the buyer's credits with this provider: a payment
   channel), optionally a new voucher (``cumulative_amount``,
   ``signature``), and ``payer_signature`` over the request id, digest,
   args and provider.
2. The provider checks it before anything else and answers with one
   control frame (``"type": "service_session_reply"``): ``ok``, ``error``,
   and ``result`` with the quoted prices (they hold for the whole session),
   the voucher size, and the channel's ``consumed`` and ``cumulative`` as
   the provider counts them. A refused session ends there.
3. Application bytes flow both ways in D frames. The provider charges what
   it serves against the buyer's credits (``ServiceSession.charge``) and
   forwards only while the credits are paid up (``paid_up``). When they
   run low it sends ``{"type": "voucher_needed", "consumed", "cumulative"}``;
   the buyer answers ``{"type": "voucher", "channel_id",
   "cumulative_amount", "signature"}``, a new running total for the
   channel. A buyer that cannot pay more says ``{"type": "no_voucher"}``.
   With no voucher within ``voucher_wait`` seconds, the provider ends the
   session. The provider reports the credits it counts after each voucher
   (``{"type": "balance", "consumed", "cumulative"}``).
4. Either side ends the session with ``{"type": "close", "reason": ...}``.

Vouchers pay ahead: each raises the channel's running total by the
provider's voucher size, so the most a buyer has at risk is one voucher.
There are no per-voucher receipts: the provider cashes the running total
in on chain (``PaymentChannel.settle``), and that payment is what the
epoch-close counts for the buyer's reviews.

The buyer's side calls back through ``voucher_source(consumed, cumulative)
-> dict | None`` (the next voucher's fields, or None to stop paying) and
``on_balance(consumed, cumulative)``. The provider's side is a
``SessionHandler`` (accept, check_voucher, serve), which the daemon backs
with its credit ledger.
"""

from __future__ import annotations

import json
import logging
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import trio

try:
    from libp2p.custom_types import TProtocol
    SERVICE_SESSION_PROTOCOL = TProtocol("/autonet/service-session/1.0.0")
except ImportError:  # pragma: no cover
    SERVICE_SESSION_PROTOCOL = "/autonet/service-session/1.0.0"

log = logging.getLogger(__name__)

KIND_CONTROL = 0x43  # "C"
KIND_DATA = 0x44     # "D"
MAX_CONTROL_BYTES = 64 * 1024
MAX_DATA_BYTES = 1024 * 1024
WRITE_CHUNK = 65000  # Noise's 65535-byte limit with yamux's header on top
OPEN_TIMEOUT = 60.0          # the provider's check reads the chain
VOUCHER_WAIT = 30.0          # unpaid usage this long ends the session
CLOSE_WAIT = 5.0             # a closing side waits this long for the other's close
DATA_BUFFER_FRAMES = 64


class SessionRefused(Exception):
    """The provider did not accept the session (its reply is ``reply``)."""

    def __init__(self, reply: dict, message: str = "") -> None:
        super().__init__(message or str(reply.get("error") or "refused"))
        self.reply = reply


class SessionClosed(Exception):
    """The session is closed (``reason`` says why)."""


class SessionUnpaid(SessionClosed):
    """The credits ran out and no voucher came in time; the session is
    closed."""


class Ledger:
    """The provider's view of one buyer's credits, as a session sees it.
    The daemon backs it with its credit ledger; the default is free."""

    def charge(self, amount: int) -> None:
        return None

    def paid_up(self) -> bool:
        return True

    def balance(self) -> tuple:
        """``(consumed, cumulative)`` in ATN base units."""
        return (0, 0)


@dataclass
class Accept:
    """The provider's answer to an opening: ``reply`` goes to the buyer as
    is; ``ledger`` is what the session charges (None: free)."""
    reply: dict
    ledger: Optional[Ledger] = None


@dataclass
class VoucherCheck:
    ok: bool
    error: str = ""


class SessionHandler:
    """The provider's side of service sessions. Every method runs in the
    host's trio loop. The default refuses everything."""

    async def accept(self, open_msg: dict) -> Accept:
        return Accept(reply={"ok": False,
                             "error": "this node serves no service sessions"})

    async def check_voucher(self, open_msg: dict, frame: dict) -> VoucherCheck:
        return VoucherCheck(ok=False, error="no voucher check")

    async def serve(self, session: "ServiceSession") -> None:
        return None

    async def ended(self, session: "ServiceSession") -> None:
        """Called once when an accepted session is over, whatever ended it."""
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
            raise EOFError("connection ended mid-frame")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


async def read_frame(raw: Any) -> tuple:
    """(kind, payload); raises EOFError at a clean end of the connection."""
    header = b""
    while len(header) < 5:
        chunk = await raw.read(5 - len(header))
        if not chunk:
            if header:
                raise EOFError("connection ended mid-header")
            raise EOFError("end of connection")
        header += chunk
    kind, length = header[0], struct.unpack(">I", header[1:5])[0]
    if kind not in (KIND_CONTROL, KIND_DATA):
        raise ValueError(f"unknown frame kind {kind}")
    limit = MAX_CONTROL_BYTES if kind == KIND_CONTROL else MAX_DATA_BYTES
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
# The session
# ---------------------------------------------------------------------------

class ServiceSession:
    """One side of an accepted session: ``send`` and ``receive``
    application bytes; the provider charges with ``charge`` and waits for
    payment with ``paid_up``."""

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
        # Vouchers are made (buyer) and checked (provider) in tasks of their
        # own here, so the application's bytes keep flowing meanwhile.
        self._nursery: Optional[trio.Nursery] = None
        self._reader_done = trio.Event()
        # Provider side.
        self.handler: Optional[SessionHandler] = None
        self.ledger: Optional[Ledger] = None
        self.voucher_wait = VOUCHER_WAIT
        self._changed = trio.Event()
        self._asked = False
        self._buyer_done = ""        # why the buyer pays no more
        # Buyer side.
        self.voucher_source: Optional[Callable[[int, int], Awaitable[Optional[dict]]]] = None
        self.on_balance: Optional[Callable[[int, int], Any]] = None

    # -- application side ---------------------------------------------------

    async def send(self, data: bytes) -> None:
        if self.closed:
            raise SessionClosed(self.close_reason or "session closed")
        view = memoryview(bytes(data))
        for off in range(0, len(view), MAX_DATA_BYTES):
            await self._write(KIND_DATA, bytes(view[off:off + MAX_DATA_BYTES]))

    async def receive(self) -> bytes:
        """The next application bytes; b"" once the session has ended."""
        try:
            return await self._data_rx.receive()
        except (trio.EndOfChannel, trio.ClosedResourceError):
            return b""

    def charge(self, amount: int) -> None:
        """Provider: record ``amount`` (ATN base units) served. Never
        blocks; ``paid_up`` decides whether serving goes on."""
        if self.ledger is not None and int(amount) > 0:
            self.ledger.charge(int(amount))

    async def paid_up(self) -> None:
        """Provider: return once the buyer's credits cover what was charged.
        Asks for a voucher when they do not, and when they run low; raises
        ``SessionUnpaid`` (and closes the session) when none comes within
        ``voucher_wait`` or the buyer pays no more."""
        if self.closed:
            raise SessionClosed(self.close_reason or "session closed")
        ledger = self.ledger
        if ledger is None:
            return
        with trio.move_on_after(self.voucher_wait):
            while not ledger.paid_up() and not self.closed and not self._buyer_done:
                changed = self._changed
                await self._ask()
                await changed.wait()
        if self.closed:
            raise SessionClosed(self.close_reason or "session closed")
        if not ledger.paid_up():
            reason = ("the buyer stopped paying: " + self._buyer_done
                      if self._buyer_done else
                      "the buyer's credits ran out and no voucher came")
            await self.aclose(reason)
            raise SessionUnpaid(reason)

    async def low(self, threshold: int) -> None:
        """Provider: ask ahead for a voucher when fewer than ``threshold``
        base units of credit are left (so serving never has to pause)."""
        if self.ledger is None or self.closed:
            return
        consumed, cumulative = self.ledger.balance()
        if cumulative - consumed < max(1, int(threshold)):
            await self._ask()

    async def aclose(self, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        self.close_reason = reason or "closed"
        with trio.CancelScope(shield=True), trio.move_on_after(3):
            try:
                await self._write(KIND_CONTROL, _control(
                    {"type": "close", "reason": self.close_reason}), force=True)
            except Exception:  # noqa: BLE001 (the far end is gone)
                pass
            # Half-close: the peer's last frames and its close can still be
            # read. The connection is closed fully when both sides are done.
            try:
                closer = getattr(self.raw, "close_write", None)
                await (closer() if closer is not None else self.raw.close())
            except Exception:  # noqa: BLE001
                pass
        self._data_tx.close()
        self._signal()
        # The provider's application runs inside the session's nursery, so
        # it stops with the session. The buyer's body is the caller's own
        # code: it sees the end as receive() returning b"".
        if self.role == "provider" and self._cancel is not None:
            self._cancel.cancel()

    # -- internals ------------------------------------------------------------

    def _signal(self) -> None:
        self._changed.set()
        self._changed = trio.Event()

    async def _write(self, kind: int, payload: bytes, force: bool = False) -> None:
        if self.closed and not force:
            raise SessionClosed(self.close_reason or "session closed")
        async with self._send_lock:
            await write_frame(self.raw, kind, payload)

    async def _send_control(self, msg: dict) -> None:
        await self._write(KIND_CONTROL, _control(msg))

    async def _ask(self) -> None:
        if self._asked or self.closed or self._buyer_done or self.ledger is None:
            return
        self._asked = True
        consumed, cumulative = self.ledger.balance()
        try:
            await self._send_control({"type": "voucher_needed",
                                      "consumed": int(consumed),
                                      "cumulative": int(cumulative)})
        except SessionClosed:
            pass

    async def _on_voucher(self, frame: dict) -> None:
        # The ask stays outstanding until the voucher is checked: asking
        # again meanwhile would have the buyer pay a second voucher ahead.
        try:
            check = (await self.handler.check_voucher(self.open_msg, frame)
                     if self.handler is not None
                     else VoucherCheck(ok=False, error="no check"))
        except Exception as e:  # noqa: BLE001 (the check could not be made)
            check = VoucherCheck(ok=False, error=f"the check failed: {e}")
        self._asked = False
        if not check.ok:
            await self.aclose(f"voucher refused: {check.error or 'invalid'}")
            return
        if self.ledger is not None:
            consumed, cumulative = self.ledger.balance()
            try:
                await self._send_control({"type": "balance", "consumed": int(consumed),
                                          "cumulative": int(cumulative)})
            except SessionClosed:
                pass
        self._signal()

    async def _on_voucher_needed(self, msg: dict) -> None:
        try:
            consumed = int(msg.get("consumed") or 0)
            cumulative = int(msg.get("cumulative") or 0)
        except (TypeError, ValueError):
            consumed = cumulative = 0
        self._note_balance(consumed, cumulative)
        fields = None
        if self.voucher_source is not None:
            try:
                fields = await self.voucher_source(consumed, cumulative)
            except Exception as e:  # noqa: BLE001
                log.warning("the session's voucher source failed: %s", e)
                fields = None
        if not fields or fields.get("error"):
            # What is paid for is still served; the provider ends the
            # session when the credits are used up.
            reason = str((fields or {}).get("error") or "no more credits")
            await self._send_control({"type": "no_voucher", "reason": reason})
            return
        await self._send_control({"type": "voucher",
                                  "channel_id": fields.get("channel_id"),
                                  "cumulative_amount": fields.get("cumulative_amount"),
                                  "signature": fields.get("signature")})

    async def _dispatch(self, fn: Callable[[dict], Awaitable[None]], msg: dict) -> None:
        """Run ``fn(msg)`` beside the reader when the session runs in a
        nursery, else inline."""
        if self._nursery is None:
            await fn(msg)
            return
        self._nursery.start_soon(self._guarded, fn, msg)

    async def _guarded(self, fn: Callable[[dict], Awaitable[None]], msg: dict) -> None:
        try:
            await fn(msg)
        except (SessionClosed, trio.ClosedResourceError, trio.BrokenResourceError):
            pass
        except Exception as e:  # noqa: BLE001 (a broken or reset connection)
            log.debug("service session control task ended: %s", e)
            await self.aclose(f"connection broke: {e}")

    def _note_balance(self, consumed: int, cumulative: int) -> None:
        if self.on_balance is not None:
            try:
                self.on_balance(int(consumed), int(cumulative))
            except Exception as e:  # noqa: BLE001
                log.warning("on_balance failed: %s", e)

    async def _reader(self) -> None:
        try:
            while True:
                kind, payload = await read_frame(self.raw)
                if kind == KIND_DATA:
                    # After this side closed, data is dropped but the reader
                    # goes on: the peer's close follows.
                    if not self.closed:
                        try:
                            await self._data_tx.send(payload)
                        except (trio.ClosedResourceError, trio.BrokenResourceError):
                            pass
                    continue
                msg = json.loads(payload.decode("utf-8"))
                if not isinstance(msg, dict):
                    continue
                kind_ = msg.get("type")
                if kind_ == "close":
                    self.close_reason = str(msg.get("reason") or "closed by the peer")
                    break
                if self.role == "provider" and kind_ == "voucher":
                    await self._dispatch(self._on_voucher, msg)
                elif self.role == "provider" and kind_ == "no_voucher":
                    self._buyer_done = str(msg.get("reason") or "no more credits")
                    self._asked = False
                    self._signal()
                elif self.role == "buyer" and kind_ == "voucher_needed":
                    await self._dispatch(self._on_voucher_needed, msg)
                elif self.role == "buyer" and kind_ == "balance":
                    try:
                        self._note_balance(int(msg.get("consumed") or 0),
                                           int(msg.get("cumulative") or 0))
                    except (TypeError, ValueError):
                        pass
        except (EOFError, trio.ClosedResourceError, trio.BrokenResourceError):
            pass
        except Exception as e:  # noqa: BLE001 (a broken or reset connection)
            log.debug("service session reader ended: %s", e)
            if not self.close_reason:
                self.close_reason = f"connection broke: {e}"
        finally:
            self._reader_done.set()
        await self.aclose(self.close_reason or "closed by the peer")


# ---------------------------------------------------------------------------
# Opening (buyer) and serving (provider)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def open_on(raw: Any, open_msg: dict, *,
                  voucher_source: Optional[Callable[[int, int], Awaitable[Optional[dict]]]] = None,
                  on_balance: Optional[Callable[[int, int], Any]] = None,
                  timeout: float = OPEN_TIMEOUT, path: str = "", peer: str = "",
                  relay: str = "") -> AsyncIterator[ServiceSession]:
    """Buyer: send the opening on ``raw`` and yield the accepted session.
    Raises ``SessionRefused`` when the provider refuses, ``TimeoutError``
    when no answer comes."""
    msg = dict(open_msg)
    msg["type"] = "service_session"
    session = ServiceSession(raw, "buyer", msg, path=path, peer=peer, relay=relay)
    session.voucher_source, session.on_balance = voucher_source, on_balance
    try:
        await write_frame(raw, KIND_CONTROL, _control(msg))
    except Exception as e:  # noqa: BLE001
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise ConnectionError(f"could not send the session's opening: {e}") from e
    try:
        with trio.fail_after(timeout):
            while True:
                kind, payload = await read_frame(raw)
                if kind == KIND_CONTROL:
                    break
        reply = json.loads(payload.decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("the answer is not an object")
    except trio.TooSlowError:
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise TimeoutError(f"the provider did not answer within {int(timeout)}s") from None
    except Exception as e:  # noqa: BLE001
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise TimeoutError(f"the connection broke before the provider answered: {e}") from e
    if not reply.get("ok"):
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)
        raise SessionRefused(reply)
    session.accept_reply = reply
    result = reply.get("result") or {}
    try:
        session._note_balance(int(result.get("consumed") or 0),
                              int(result.get("cumulative") or 0))
    except (TypeError, ValueError):
        pass
    try:
        async with trio.open_nursery() as nursery:
            session._cancel, session._nursery = nursery.cancel_scope, nursery
            nursery.start_soon(session._reader)
            try:
                yield session
            finally:
                await session.aclose(session.close_reason or "closed by the buyer")
                with trio.move_on_after(CLOSE_WAIT):
                    await session._reader_done.wait()
                nursery.cancel_scope.cancel()
    finally:
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)


async def serve_incoming(raw: Any, handler: Optional[SessionHandler], *,
                         path: str = "", peer: str = "") -> None:
    """Provider: read the opening, check it, and serve an accepted session
    until either side ends it."""
    handler = handler or SessionHandler()
    try:
        with trio.fail_after(OPEN_TIMEOUT):
            kind, payload = await read_frame(raw)
        if kind != KIND_CONTROL:
            raise ValueError("the first frame must be the opening")
        open_msg = json.loads(payload.decode("utf-8"))
        if not isinstance(open_msg, dict) or open_msg.get("type") != "service_session":
            raise ValueError("the first frame must be a service_session opening")
    except Exception as e:  # noqa: BLE001
        log.debug("bad service session opening: %s", e)
        try:
            await write_frame(raw, KIND_CONTROL, _control(
                {"type": "service_session_reply", "ok": False,
                 "error": f"bad opening: {e}"}))
        except Exception:  # noqa: BLE001
            pass
        await _quiet_close(raw)
        return
    try:
        accepted = await handler.accept(open_msg)
    except Exception as e:  # noqa: BLE001
        log.warning("service session check failed: %s", e)
        accepted = Accept(reply={"ok": False,
                                 "error": f"provider failed to check the session: {e}"})
    reply = dict(accepted.reply)
    reply["type"] = "service_session_reply"
    session = ServiceSession(raw, "provider", open_msg, path=path, peer=peer,
                             accept_reply=reply)
    session.handler = handler
    session.ledger = accepted.ledger
    accepted_ok = ok = bool(reply.get("ok"))
    try:
        await write_frame(raw, KIND_CONTROL, _control(reply))
    except Exception as e:  # noqa: BLE001
        log.debug("could not answer a service session: %s", e)
        ok = False
    if not ok:
        # The handler took what an accepted session holds (a tool lease, a
        # credit in use): it is ended even though the buyer never heard.
        if accepted_ok:
            session.close_reason = "the acceptance could not be sent"
            with trio.CancelScope(shield=True), trio.move_on_after(15):
                try:
                    await handler.ended(session)
                except Exception:  # noqa: BLE001
                    pass
        await _quiet_close(raw)
        return
    try:
        async with trio.open_nursery() as nursery:
            session._cancel, session._nursery = nursery.cancel_scope, nursery
            nursery.start_soon(session._reader)
            try:
                await handler.serve(session)
            except SessionClosed:
                pass
            except Exception as e:  # noqa: BLE001
                log.warning("service session application failed: %s", e)
                session.close_reason = f"provider failed: {e}"
            finally:
                await session.aclose(session.close_reason or "closed by the provider")
    finally:
        if not session.closed:
            await session.aclose("ended")
        with trio.CancelScope(shield=True), trio.move_on_after(15):
            try:
                await handler.ended(session)
            except Exception as e:  # noqa: BLE001
                log.warning("ending a service session failed: %s", e)
        with trio.CancelScope(shield=True):
            await _quiet_close(raw)


async def _quiet_close(raw: Any) -> None:
    with trio.move_on_after(3):
        try:
            await raw.close()
        except Exception:  # noqa: BLE001
            pass
