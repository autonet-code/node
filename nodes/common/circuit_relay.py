"""Relayed connections between daemons that cannot dial each other
(circuit relay v2, docs/p2p_networking.md "Relayed connections").

A daemon behind NAT is reachable only over the connections it opened. Two
such daemons connected to the same relay get a libp2p connection to each
other through it: the dialer sends the relay a HOP CONNECT naming the
target, the relay opens a STOP stream to the target, and from then on it
copies bytes between the two streams. The two ends run a full libp2p
connection over them (Noise, then yamux), so the relay sees only
ciphertext and each end has authenticated the other's peer id.

What runs where:

- every host installs the STOP side (``install_stop``), py-libp2p's own
  handler, so it accepts relayed connections;
- a relay (``atn relay``) installs ``CircuitRelay``, the HOP side;
- the dialer side is ``dial_via_relay``.

Wire format: py-libp2p 0.6's circuit v2 messages (``HopMessage`` and
``StopMessage`` protobufs written without a length prefix, which is how
that version reads them), so the STOP side is the library's handler. It
does not interoperate with go-libp2p or rust-libp2p relays (they
length-prefix every message), nor with py-libp2p 0.8 (whose target must
hold a reservation and whose ``peer`` field is a message).

Why the HOP side is our own: py-libp2p 0.6's relay writes each read of up
to 65535 bytes to the other stream in one write, which Noise refuses (its
message limit is 65535 bytes and yamux adds a 12-byte header), so any
relayed transfer above about 64 KiB breaks the circuit. It also wants the
dialer to hold a reservation, accepts four reservations in all by default,
caps each reservation at 1 GiB, and keys its circuits by peer, so closing
one buyer's circuit to a provider closes another's.

Caps (``RelayCaps``): none by default. A relay gives its bandwidth the way a
torrent seed does; its operator may cap a circuit's duration and bytes and
the number of circuits at once (``atn relay --circuit-max-duration``,
``--circuit-max-bytes``, ``--max-circuits``).
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import trio

try:
    from libp2p.network.connection.raw_connection import RawConnection
    from libp2p.network.stream.net_stream import NetStream
    from libp2p.peer.id import ID as PeerID
    from libp2p.relay.circuit_v2.pb.circuit_pb2 import (
        HopMessage,
        StopMessage,
    )
    from libp2p.relay.circuit_v2.protocol import (
        PROTOCOL_ID as HOP_PROTOCOL,
        STOP_PROTOCOL_ID as STOP_PROTOCOL,
        CircuitV2Protocol,
    )
    from libp2p.relay.circuit_v2.protocol_buffer import StatusCode

    CIRCUIT_AVAILABLE = True
except ImportError:  # pragma: no cover (libp2p ships with the core install)
    CIRCUIT_AVAILABLE = False

log = logging.getLogger(__name__)

# Hop and stop messages are a few hundred bytes; one read takes one.
MAX_MESSAGE = 4096
HANDSHAKE_TIMEOUT = 15.0
# Under Noise's 65535-byte message limit with yamux's header on top (the
# same bound as p2p._MAX_WRITE_CHUNK).
PIPE_CHUNK = 65000
PIPE_READ = 65535
# A write that cannot make progress this long means the far end stopped
# reading: the circuit is dropped instead of holding the relay's buffers.
PIPE_WRITE_TIMEOUT = 60.0
# Reservations are not needed by our dialer; a RESERVE (sent by py-libp2p's
# own client transport) is answered OK for this long.
RESERVATION_TTL = 3600


@dataclass
class RelayCaps:
    """A relay operator's limits on relayed connections. 0 = no limit."""
    max_duration: float = 0.0   # seconds one circuit may stay open
    max_bytes: int = 0          # bytes one circuit may carry, both ways together
    max_circuits: int = 0       # circuits open at once

    def describe(self) -> str:
        parts = []
        if self.max_duration:
            parts.append(f"{int(self.max_duration)} s per circuit")
        if self.max_bytes:
            parts.append(f"{int(self.max_bytes)} bytes per circuit")
        if self.max_circuits:
            parts.append(f"{int(self.max_circuits)} circuits at once")
        return ", ".join(parts) or "unlimited"


def _status_message(kind: str, code: int, text: str) -> bytes:
    cls = HopMessage if kind == "hop" else StopMessage
    msg = cls(type=cls.STATUS)
    msg.status.code = int(code)
    msg.status.message = text
    return msg.SerializeToString()


async def _close(stream: Any, reset: bool = False) -> None:
    if stream is None:
        return
    with trio.CancelScope(shield=True), trio.move_on_after(3):
        try:
            if reset:
                await stream.reset()
            else:
                await stream.close()
        except Exception:  # noqa: BLE001 (already closed or broken)
            pass


class CircuitRelay:
    """The HOP side: forwards a CONNECT to a peer this relay is connected
    to, then copies bytes both ways until either end closes or a cap ends
    the circuit. A relay never dials a target for a CONNECT."""

    def __init__(self, host: Any, caps: Optional[RelayCaps] = None) -> None:
        self.host = host
        self.caps = caps or RelayCaps()
        self.active = 0
        self.total = 0
        self.refused = 0
        self.bytes_relayed = 0

    def install(self) -> None:
        self.host.set_stream_handler(HOP_PROTOCOL, self._handle_hop)

    def stats(self) -> dict:
        return {"active": self.active, "total": self.total,
                "refused": self.refused, "bytes": self.bytes_relayed,
                "caps": self.caps.describe()}

    async def _status(self, stream: Any, code: int, text: str,
                      reservation: bool = False) -> None:
        msg = HopMessage(type=HopMessage.STATUS)
        msg.status.code = int(code)
        msg.status.message = text
        if reservation:
            msg.reservation.expire = int(time.time()) + RESERVATION_TTL
        if self.caps.max_duration:
            msg.limit.duration = int(self.caps.max_duration)
        if self.caps.max_bytes:
            msg.limit.data = int(self.caps.max_bytes)
        with trio.move_on_after(HANDSHAKE_TIMEOUT):
            await stream.write(msg.SerializeToString())

    async def _handle_hop(self, stream: Any) -> None:
        src = stream.muxed_conn.peer_id
        try:
            while True:
                with trio.fail_after(HANDSHAKE_TIMEOUT):
                    data = await stream.read(MAX_MESSAGE)
                if not data:
                    return
                msg = HopMessage()
                msg.ParseFromString(data)
                if msg.type == HopMessage.RESERVE:
                    await self._status(stream, StatusCode.OK,
                                       "reservation accepted", reservation=True)
                    continue
                if msg.type == HopMessage.CONNECT:
                    await self._connect(stream, src, msg)
                    return
                await self._status(stream, StatusCode.MALFORMED_MESSAGE,
                                   "expected RESERVE or CONNECT")
                await _close(stream, reset=True)
                return
        except Exception as e:  # noqa: BLE001 (a bad peer ends its own stream)
            log.debug("hop stream from %s failed: %s", str(src)[:16], e)
            await _close(stream, reset=True)

    async def _connect(self, stream: Any, src: Any, msg: Any) -> None:
        try:
            dst = PeerID(msg.peer)
        except Exception:  # noqa: BLE001
            dst = None
        if dst is None or dst == self.host.get_id() or dst == src:
            self.refused += 1
            await self._status(stream, StatusCode.MALFORMED_MESSAGE,
                               "CONNECT names no valid target")
            await _close(stream, reset=True)
            return
        if not self.host.get_network().get_connections(dst):
            # Only peers that dialled this relay are reached through it; a
            # relay is not a dialler for whoever asks.
            self.refused += 1
            await self._status(stream, StatusCode.CONNECTION_FAILED,
                               "the relay is not connected to the target")
            await _close(stream, reset=True)
            return
        if self.caps.max_circuits and self.active >= self.caps.max_circuits:
            self.refused += 1
            await self._status(stream, StatusCode.RESOURCE_LIMIT_EXCEEDED,
                               f"the relay carries at most "
                               f"{self.caps.max_circuits} circuits at once")
            await _close(stream, reset=True)
            return
        self.active += 1
        self.total += 1
        dst_stream = None
        try:
            try:
                with trio.fail_after(HANDSHAKE_TIMEOUT):
                    dst_stream = await self.host.new_stream(dst, [STOP_PROTOCOL])
                    await dst_stream.write(StopMessage(
                        type=StopMessage.CONNECT,
                        peer=src.to_bytes()).SerializeToString())
                    resp = StopMessage()
                    resp.ParseFromString(await dst_stream.read(MAX_MESSAGE))
                if (resp.type != StopMessage.STATUS or not resp.HasField("status")
                        or resp.status.code != StatusCode.OK):
                    raise ConnectionError(resp.status.message or "the target refused")
            except Exception as e:  # noqa: BLE001
                self.refused += 1
                await self._status(stream, StatusCode.CONNECTION_FAILED,
                                   f"could not reach the target: {e}")
                await _close(stream, reset=True)
                await _close(dst_stream, reset=True)
                return
            await self._status(stream, StatusCode.OK, "connected")
            log.debug("circuit %s -> %s open", str(src)[:12], str(dst)[:12])
            await self._pipe(stream, dst_stream)
        finally:
            self.active -= 1

    async def _pipe(self, a: Any, b: Any) -> None:
        """Copy bytes both ways; either side closing, or a cap, ends both."""
        budget = [int(self.caps.max_bytes) or None]
        ended: list = []

        async def one_way(src: Any, dst: Any, done: trio.CancelScope) -> None:
            try:
                while True:
                    data = await src.read(PIPE_READ)
                    if not data:
                        break
                    if budget[0] is not None:
                        if len(data) > budget[0]:
                            ended.append("byte cap")
                            break
                        budget[0] -= len(data)
                    for off in range(0, len(data), PIPE_CHUNK):
                        with trio.fail_after(PIPE_WRITE_TIMEOUT):
                            await dst.write(data[off:off + PIPE_CHUNK])
                    self.bytes_relayed += len(data)
            except Exception as e:  # noqa: BLE001 (a closed or reset end)
                log.debug("circuit pipe ended: %s", e)
            done.cancel()

        deadline = (trio.move_on_after(float(self.caps.max_duration))
                    if self.caps.max_duration else nullcontext())
        try:
            with deadline as scope:
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(one_way, a, b, nursery.cancel_scope)
                    nursery.start_soon(one_way, b, a, nursery.cancel_scope)
            if scope is not None and getattr(scope, "cancelled_caught", False):
                ended.append("duration cap")
        finally:
            if ended:
                log.info("circuit closed by the relay's %s", ended[0])
            await _close(a, reset=bool(ended))
            await _close(b, reset=bool(ended))


def install_stop(host: Any) -> Any:
    """Accept relayed connections on ``host`` (py-libp2p's STOP handler:
    it answers the relay and upgrades the stream as an inbound
    connection). Returns the protocol object that holds the handler."""
    proto = CircuitV2Protocol(host, allow_hop=False)
    host.set_stream_handler(STOP_PROTOCOL, proto._handle_stop_stream)
    return proto


async def dial_via_relay(host: Any, relay_id: Any, target_id: Any,
                         timeout: float = HANDSHAKE_TIMEOUT) -> Any:
    """A libp2p connection to ``target_id`` through ``relay_id`` (both
    connected to the relay). Returns the swarm connection; raises
    ``ConnectionError`` when the relay or the target refuses."""
    if isinstance(relay_id, str):
        relay_id = PeerID.from_base58(relay_id)
    if isinstance(target_id, str):
        target_id = PeerID.from_base58(target_id)
    try:
        with trio.fail_after(timeout):
            stream = await host.new_stream(relay_id, [HOP_PROTOCOL])
    except Exception as e:  # noqa: BLE001
        raise ConnectionError(f"relay {str(relay_id)[:16]} offers no circuit: {e}") from e
    try:
        with trio.fail_after(timeout):
            await stream.write(HopMessage(
                type=HopMessage.CONNECT,
                peer=target_id.to_bytes()).SerializeToString())
            resp = HopMessage()
            resp.ParseFromString(await stream.read(MAX_MESSAGE))
        if resp.type != HopMessage.STATUS or not resp.HasField("status"):
            raise ConnectionError("the relay sent no status")
        if resp.status.code != StatusCode.OK:
            raise ConnectionError(f"the relay refused: {resp.status.message}")
        raw = RawConnection(stream=stream, initiator=True)
        with trio.fail_after(timeout):
            # Noise over the relayed stream: fails unless the far end holds
            # target_id's key.
            return await host.upgrade_outbound_connection(raw, target_id)
    except BaseException as e:
        await _close(stream, reset=True)
        if isinstance(e, (ConnectionError, trio.Cancelled, KeyboardInterrupt)):
            raise
        if isinstance(e, trio.TooSlowError):
            raise ConnectionError("the relayed connection timed out") from e
        if isinstance(e, Exception):
            raise ConnectionError(f"the relayed connection failed: {e}") from e
        raise


def is_relayed(conn: Any) -> bool:
    """Whether a swarm connection runs over a relayed stream (its raw
    transport is a libp2p stream, not a socket)."""
    if not CIRCUIT_AVAILABLE:
        return False
    queue = [getattr(conn, "muxed_conn", conn)]
    for _ in range(12):
        if not queue:
            break
        obj = queue.pop()
        if isinstance(obj, NetStream):
            return True
        # Yamux -> SecureSession -> Noise reader/writers -> RawConnection
        # -> its stream (a socket, or a NetStream when relayed).
        for name in ("secured_conn", "conn", "read_writer", "read_write_closer",
                     "stream", "_wrapped"):
            try:
                nxt = obj.__dict__.get(name) if hasattr(obj, "__dict__") else None
            except Exception:  # noqa: BLE001
                nxt = None
            if nxt is not None and nxt is not obj:
                queue.append(nxt)
    return False
