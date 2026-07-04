"""IPC contract between the daemon and a per-agent worker process (Phase 1).

This is the *dark library* for agent per-PID process isolation. Nothing in the
daemon references it yet; the actual spawn + wiring lands in P4. What is built
here is the wire contract itself, so the framing/demux can be tested and frozen
in isolation before anything depends on it.

SECURITY — READ BEFORE TOUCHING THE FRAMING
-------------------------------------------
The daemon<->worker pipe carries data from a process whose whole purpose is to
be *less* trusted than the daemon (the threat model is the agents themselves).
Therefore the frame codec is length-prefixed JSON and NOTHING ELSE:

    struct.pack(">I", len(body)) + body        # body = utf-8 json bytes

sent/received exclusively via ``Connection.send_bytes`` / ``recv_bytes``.

``Connection.send`` / ``Connection.recv`` (the pickle path) MUST NEVER be used
on this pipe. Pickle-over-pipe means a malicious worker can execute arbitrary
code inside the daemon on ``recv`` — a total defeat of the isolation model. The
codec below asserts this at the type level (it only ever calls the *_bytes
methods) and ``_assert_no_pickle`` documents/enforces the invariant for anyone
who tries to hand a raw pipe object around.

IDENTITY
--------
Agent identity is NOT carried in the message body. It is a property of *which
pipe* a frame arrived on. The daemon constructs one ``WorkerChannel`` per spawned
worker with the authoritative ``agent_id`` (from its own spawn record, which also
holds the PID). Any ``agent_id`` / ``caller_id`` appearing in an inbound body is
IGNORED. Outbound events have their source OVERWRITTEN by the daemon with the
pipe-bound agent_id — a worker cannot spoof another agent.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger(__name__)

# Protocol version. Bumped only on a breaking envelope change.
PROTO_VERSION = 1

# Max frame body size (bytes). A worker sending a larger frame is treated as
# hostile/broken and the channel is torn down rather than allocating unbounded
# memory in the daemon on its behalf. 64 MiB is comfortably above any legitimate
# tool result / ProviderResponse; tune in P4 if a real payload ever approaches it.
MAX_FRAME_BYTES = 64 * 1024 * 1024

# Bounded event queue depth (worker->daemon events are droppable; see below).
EVENT_QUEUE_MAXSIZE = 1024

# Per-channel cap on concurrently in-flight rpc_req handlers (H3). The reader
# blocks on this semaphore once the cap is reached, which stops it draining the
# pipe and backpressures the worker's send_bytes — so a malicious worker cannot
# flood the daemon with unbounded tasks (each spawn_child = a real child process)
# and starve the shared event loop serving every other agent. Generous vs any
# honest worker (tool calls are near-serial); it only bites a flood.
MAX_INFLIGHT_RPC = 64

# Consecutive undecodable inbound frames before a channel is torn down (M1). A
# single bad frame is skipped (see _recv_loop), but a worker spewing garbage in a
# tight loop would otherwise spin a core; past this many in a row with no valid
# frame between, treat the worker as hostile/broken and end the channel.
MAX_CONSECUTIVE_BAD_FRAMES = 64


# ---------------------------------------------------------------------------
# Envelope kinds
# ---------------------------------------------------------------------------
# cmd     daemon -> worker : run, send_user_message, interrupt(cooperative), shutdown
# rpc_req worker -> daemon : framework_tool, spawn_child, budget_check,
#                            budget_record, inbox_drain, inbox_post, surface_tool, mcp_tool
# rpc_res daemon -> worker : result/error for a prior rpc_req (matched by id)
# event   worker -> daemon : step.output, agent.tool_use_* (fire-and-forget, droppable)
# status  worker -> daemon : ready, execution_done, heartbeat
#
# NOTE: "kill" is deliberately NOT a kind. A wedged worker won't read its pipe,
# so kill is out-of-band PID termination by the supervisor, never a message.

KIND_CMD = "cmd"
KIND_CMD_RES = "cmd_res"
KIND_RPC_REQ = "rpc_req"
KIND_RPC_RES = "rpc_res"
KIND_EVENT = "event"
KIND_STATUS = "status"

# cmd_res  worker -> daemon : reply to a daemon cmd that carried an id
#                            (expect_reply=True). This is the SYMMETRIC twin of
#                            rpc_res: it lets a host->worker cmd carry a result
#                            back (e.g. send_user_message consumption ack,
#                            request_compaction accepted flag) WITHOUT a second
#                            channel. Fire-and-forget cmds (id=None) never get one.
_VALID_KINDS = frozenset({KIND_CMD, KIND_CMD_RES, KIND_RPC_REQ, KIND_RPC_RES,
                          KIND_EVENT, KIND_STATUS})

# rpc names the worker may ask the daemon to perform (in the daemon's shared
# context). Wiring of the actual callables is P4; here they are just the
# accepted vocabulary the WorkerChannel dispatch keys on.
RPC_NAMES = frozenset({
    "framework_tool",
    "spawn_child",
    "budget_check",
    "budget_record",
    "inbox_drain",
    "inbox_post",
    "surface_tool",
    "mcp_tool",
})


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

@dataclass
class Envelope:
    """A single IPC message.

    Wire form is the JSON object ``{"v":1,"kind":...,"id":...,"name":...,
    "payload":...}``. ``id`` correlates rpc_req<->rpc_res (uuid str) and is None
    for fire-and-forget kinds (event/status) and cmds that don't expect a reply.
    ``name`` is the sub-selector (the cmd name, the rpc name, the event type, or
    the status name). ``payload`` is the arbitrary JSON body.
    """
    kind: str
    name: str = ""
    id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    v: int = PROTO_VERSION

    def to_bytes(self) -> bytes:
        return _encode(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "kind": self.kind,
            "id": self.id,
            "name": self.name,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Envelope":
        kind = d.get("kind")
        if kind not in _VALID_KINDS:
            raise ValueError(f"invalid envelope kind: {kind!r}")
        return cls(
            kind=kind,
            name=d.get("name", "") or "",
            id=d.get("id"),
            payload=d.get("payload") or {},
            v=d.get("v", PROTO_VERSION),
        )

    # -- convenience constructors ------------------------------------------
    @classmethod
    def cmd(cls, name: str, payload: dict | None = None, *, expect_reply: bool = False) -> "Envelope":
        return cls(kind=KIND_CMD, name=name, id=(_new_id() if expect_reply else None),
                   payload=payload or {})

    @classmethod
    def rpc_req(cls, name: str, payload: dict | None = None) -> "Envelope":
        return cls(kind=KIND_RPC_REQ, name=name, id=_new_id(), payload=payload or {})

    @classmethod
    def rpc_res(cls, req_id: str, payload: dict | None = None, *, error: str | None = None) -> "Envelope":
        body = dict(payload or {})
        if error is not None:
            body = {"error": error}
        return cls(kind=KIND_RPC_RES, name="", id=req_id, payload=body)

    @classmethod
    def cmd_res(cls, cmd_id: str, payload: dict | None = None, *, error: str | None = None) -> "Envelope":
        body = dict(payload or {})
        if error is not None:
            body = {"error": error}
        return cls(kind=KIND_CMD_RES, name="", id=cmd_id, payload=body)

    @classmethod
    def event(cls, name: str, payload: dict | None = None) -> "Envelope":
        return cls(kind=KIND_EVENT, name=name, id=None, payload=payload or {})

    @classmethod
    def status(cls, name: str, payload: dict | None = None) -> "Envelope":
        return cls(kind=KIND_STATUS, name=name, id=None, payload=payload or {})


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Framing — length-prefixed JSON, NEVER pickle
# ---------------------------------------------------------------------------

_LEN = struct.Struct(">I")  # 4-byte big-endian unsigned length prefix


def _encode(obj: dict[str, Any]) -> bytes:
    """Serialize an envelope dict to length-prefixed JSON bytes."""
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {len(body)} > {MAX_FRAME_BYTES}")
    return _LEN.pack(len(body)) + body


def _decode(frame: bytes) -> dict[str, Any]:
    """Parse a length-prefixed JSON frame back to a dict.

    Used by the socket-pair test harness; the live pipe path uses recv_bytes
    which already delivers exactly one framed body (see recv_envelope). This
    helper validates the prefix for the raw-stream case.
    """
    if len(frame) < _LEN.size:
        raise ValueError("short frame: missing length prefix")
    (n,) = _LEN.unpack(frame[:_LEN.size])
    if n > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {n} > {MAX_FRAME_BYTES}")
    body = frame[_LEN.size:_LEN.size + n]
    if len(body) != n:
        raise ValueError(f"truncated frame: have {len(body)}, want {n}")
    return json.loads(body.decode("utf-8"))


def _assert_no_pickle(conn: Any) -> None:
    """Sanity guard: we only ever use send_bytes/recv_bytes on the pipe.

    This does not (cannot) prevent someone calling ``conn.send`` elsewhere, but
    it makes the intent explicit and fails loudly if a non-Connection object is
    threaded in by mistake. The real enforcement is that *this module never
    calls .send/.recv* — grep the file: there are zero occurrences.
    """
    if not (hasattr(conn, "send_bytes") and hasattr(conn, "recv_bytes")):
        raise TypeError(
            "worker IPC requires a multiprocessing Connection with "
            "send_bytes/recv_bytes; pickle-based send/recv is FORBIDDEN"
        )


# ---------------------------------------------------------------------------
# Pipe transport — a thin async wrapper over a multiprocessing.Connection
# ---------------------------------------------------------------------------

class PipeTransport:
    """Async send/recv of Envelopes over a duplex Connection using ONLY
    send_bytes/recv_bytes.

    ``multiprocessing.Connection.send_bytes`` already prepends its own length
    header and ``recv_bytes`` returns exactly one whole message, so the JSON
    payload we hand it is one atomic frame per envelope — no manual prefixing on
    the pipe path. (The 4-byte prefix in _encode/_decode is for the raw
    socket-pair used by the unit test, which is a byte stream with no framing.)

    Blocking Connection I/O is dispatched to a thread executor so it never
    stalls the event loop.
    """

    def __init__(self, conn: Any) -> None:
        _assert_no_pickle(conn)
        self._conn = conn
        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()

    async def send(self, env: Envelope) -> None:
        body = json.dumps(env.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_FRAME_BYTES:
            raise ValueError(f"frame too large: {len(body)} > {MAX_FRAME_BYTES}")
        async with self._send_lock:
            # send_bytes, never send. send() would pickle -> RCE surface.
            await asyncio.to_thread(self._conn.send_bytes, body)

    async def recv(self) -> Envelope:
        async with self._recv_lock:
            # recv_bytes, never recv. recv() would unpickle -> RCE in the daemon.
            # Pass maxlength so an oversize frame is rejected by multiprocessing
            # BEFORE it is allocated (on POSIX it also restores the size-prefix
            # guard); an over-cap message raises OSError, which _recv_loop treats
            # as terminal for that (hostile) worker.
            body = await asyncio.to_thread(self._conn.recv_bytes, MAX_FRAME_BYTES)
        if len(body) > MAX_FRAME_BYTES:
            raise ValueError(f"inbound frame too large: {len(body)}")
        d = json.loads(body.decode("utf-8"))
        return Envelope.from_dict(d)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker-side: DaemonClient
# ---------------------------------------------------------------------------

class DaemonClient:
    """Worker-side stub for talking to the daemon over the inherited pipe.

    - ``rpc(name, payload)`` -> awaits the matching rpc_res by id.
    - ``emit(event)`` -> fire-and-forget event (never blocks the loop meaningfully).
    - ``send_status(name, payload)`` -> ready/execution_done/heartbeat.

    A single reader task demuxes inbound frames by kind:
      - rpc_res  -> resolve the per-id future
      - cmd      -> hand to the injected cmd handler (run/send_user_message/
                    interrupt/shutdown)
    The worker never receives event/status/rpc_req (those flow the other way).
    """

    def __init__(
        self,
        transport: PipeTransport,
        on_cmd: Optional[Callable[[Envelope], Awaitable[None]]] = None,
    ) -> None:
        self._t = transport
        self._on_cmd = on_cmd
        self._pending: dict[str, asyncio.Future] = {}
        self._reader: Optional[asyncio.Task] = None
        self._closed = False

    def set_cmd_handler(self, on_cmd: Callable[[Envelope], Awaitable[None]]) -> None:
        """Bind the inbound-cmd handler after construction.

        Needed because the worker and the client reference each other: the
        client is built first (to wrap the transport), then the worker (which
        needs the client), then the handler is wired back here.
        """
        self._on_cmd = on_cmd

    def start(self) -> None:
        if self._reader is None:
            self._reader = asyncio.create_task(self._recv_loop(), name="daemon-client-reader")

    async def _recv_loop(self) -> None:
        try:
            while not self._closed:
                env = await self._t.recv()
                if env.kind == KIND_RPC_RES:
                    fut = self._pending.pop(env.id, None)
                    if fut is not None and not fut.done():
                        err = env.payload.get("error") if isinstance(env.payload, dict) else None
                        if err is not None:
                            fut.set_exception(RuntimeError(str(err)))
                        else:
                            fut.set_result(env.payload)
                    else:
                        log.warning("rpc_res for unknown id %s", env.id)
                elif env.kind == KIND_CMD:
                    if self._on_cmd is not None:
                        # A cmd carrying an id expects a cmd_res reply (the
                        # symmetric twin of rpc_req/rpc_res). The handler's
                        # return value (a dict, or None) becomes the reply body;
                        # a raised exception becomes a cmd_res error. Handlers
                        # for fire-and-forget cmds (id=None) return None and no
                        # reply is sent — unchanged behavior.
                        if env.id is not None:
                            try:
                                result = await self._on_cmd(env)
                                body = result if isinstance(result, dict) else {}
                                await self._t.send(Envelope.cmd_res(env.id, body))
                            except Exception as e:
                                try:
                                    await self._t.send(Envelope.cmd_res(
                                        env.id, error=f"{type(e).__name__}: {e}"))
                                except Exception:
                                    log.warning("failed to send cmd_res error")
                        else:
                            await self._on_cmd(env)
                    else:
                        log.warning("no cmd handler for cmd %r", env.name)
                else:
                    # Worker should not receive event/status/rpc_req.
                    log.warning("worker got unexpected inbound kind %r", env.kind)
        except (EOFError, OSError):
            log.info("daemon pipe closed; worker reader exiting")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker recv loop crashed")
        finally:
            # Fail any outstanding RPCs so awaiters don't hang forever.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("daemon pipe closed"))
            self._pending.clear()

    async def rpc(self, name: str, payload: dict | None = None) -> dict:
        if name not in RPC_NAMES:
            raise ValueError(f"unknown rpc name: {name!r}")
        env = Envelope.rpc_req(name, payload)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[env.id] = fut
        try:
            await self._t.send(env)
        except Exception:
            self._pending.pop(env.id, None)
            raise
        return await fut

    async def emit(self, name: str, payload: dict | None = None) -> None:
        """Fire-and-forget event to the daemon. Never awaits an ack."""
        try:
            await self._t.send(Envelope.event(name, payload))
        except Exception:
            log.debug("dropped event %r (pipe send failed)", name)

    async def send_status(self, name: str, payload: dict | None = None) -> None:
        await self._t.send(Envelope.status(name, payload))

    async def close(self) -> None:
        self._closed = True
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
        self._t.close()


# ---------------------------------------------------------------------------
# Daemon-side: WorkerChannel
# ---------------------------------------------------------------------------

# The daemon injects one callable per rpc name. Each takes (agent_id, payload)
# and returns a dict (or raises). agent_id is the AUTHORITATIVE pipe-bound id —
# the injected callable receives it from the channel, never from the worker body.
RpcHandler = Callable[[str, dict], Awaitable[dict]]

# Injected sink for worker->daemon events. Signature (agent_id, name, payload).
# The daemon OVERWRITES the event source with agent_id before publishing on its
# EventBus, so a worker cannot forge another agent's events.
EventSink = Callable[[str, str, dict], Awaitable[None]]

# Injected sink for status frames (ready/execution_done/heartbeat).
StatusSink = Callable[[str, str, dict], Awaitable[None]]


class WorkerChannel:
    """Daemon-side handler for ONE worker's pipe.

    Constructed with the authoritative ``agent_id`` (from the daemon's own spawn
    record). It re-authorizes every inbound frame by *pipe binding*: identity is
    this channel's agent_id, and any agent_id/caller_id in the body is ignored.

    Responsibilities:
      - one reader task demuxing inbound frames by kind
      - rpc_req  -> dispatch to the injected handler for that rpc name, reply
                    with rpc_res(id) (RPC requests are NEVER dropped)
      - event    -> push onto a bounded queue drained to the event sink; on
                    overflow drop the OLDEST and warn (events are droppable)
      - status   -> forward to the status sink
      - send()   -> daemon->worker cmds (run/send_user_message/interrupt/shutdown)

    Wiring of the concrete handlers/sinks is P4. Here they are injected deps;
    unset handlers reply with a NotImplemented error rather than crashing.
    """

    def __init__(
        self,
        agent_id: str,
        transport: PipeTransport,
        *,
        rpc_handlers: Optional[dict[str, RpcHandler]] = None,
        event_sink: Optional[EventSink] = None,
        status_sink: Optional[StatusSink] = None,
        event_queue_maxsize: int = EVENT_QUEUE_MAXSIZE,
    ) -> None:
        # The AUTHORITATIVE identity. Never read from the wire.
        self.agent_id = agent_id
        self._t = transport
        self._rpc_handlers = dict(rpc_handlers or {})
        self._event_sink = event_sink
        self._status_sink = status_sink
        self._events: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=event_queue_maxsize)
        self._reader: Optional[asyncio.Task] = None
        self._event_worker: Optional[asyncio.Task] = None
        self._closed = False
        # Futures for daemon->worker cmds that carried an id (expect_reply). The
        # worker replies with a cmd_res correlated by id — symmetric to the
        # DaemonClient rpc_req/rpc_res mechanism, on the SAME channel.
        self._pending_cmds: dict[str, asyncio.Future] = {}
        # Bounds concurrent rpc handlers (H3); acquired by the reader before each
        # rpc task is spawned, released in _handle_rpc's finally.
        self._rpc_sem = asyncio.Semaphore(MAX_INFLIGHT_RPC)

    def start(self) -> None:
        if self._reader is None:
            self._reader = asyncio.create_task(
                self._recv_loop(), name=f"worker-channel-{self.agent_id}")
        if self._event_worker is None:
            self._event_worker = asyncio.create_task(
                self._drain_events(), name=f"worker-events-{self.agent_id}")

    # -- daemon -> worker --------------------------------------------------
    async def send_cmd(self, name: str, payload: dict | None = None) -> None:
        await self._t.send(Envelope.cmd(name, payload))

    async def send_cmd_await(
        self, name: str, payload: dict | None = None, *, timeout: float = 5.0,
    ) -> dict:
        """Send a daemon->worker cmd and await the worker's cmd_res reply.

        Symmetric to ``DaemonClient.rpc`` but in the daemon->worker direction:
        used for cmds whose result the daemon needs (send_user_message's
        consumption ack, request_compaction's accepted flag). The reply body is
        returned as a dict; a worker-side error is raised as RuntimeError; a
        missing reply within ``timeout`` raises asyncio.TimeoutError. Reuses the
        one existing pipe — NOT a second channel."""
        env = Envelope.cmd(name, payload, expect_reply=True)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_cmds[env.id] = fut
        try:
            await self._t.send(env)
        except Exception:
            self._pending_cmds.pop(env.id, None)
            raise
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_cmds.pop(env.id, None)

    # -- inbound demux -----------------------------------------------------
    async def _recv_loop(self) -> None:
        bad_frames = 0
        try:
            while not self._closed:
                # Decode is isolated so a SINGLE malformed frame (non-object JSON,
                # bad kind, non-utf8) is skipped rather than killing the channel.
                # Previously any decode error escaped to the outer `except
                # Exception` and exited the loop, silently wedging the worker's
                # whole channel while its process kept running — the supervisor
                # never learned it died and any parent awaiting its status hung.
                try:
                    env = await self._t.recv()
                except (EOFError, OSError):
                    raise                      # pipe closed / oversize -> terminal
                except asyncio.CancelledError:
                    raise
                except Exception:
                    bad_frames += 1
                    if bad_frames >= MAX_CONSECUTIVE_BAD_FRAMES:
                        log.warning("worker %s: %d consecutive undecodable frames"
                                    " — tearing down channel", self.agent_id,
                                    bad_frames)
                        break
                    log.warning("worker %s sent an undecodable frame; skipping",
                                self.agent_id)
                    continue
                bad_frames = 0
                if env.kind == KIND_RPC_REQ:
                    # NEVER dropped, but bounded: block here (=> pipe backpressure)
                    # once MAX_INFLIGHT_RPC handlers are outstanding. Each runs in
                    # its own task so a slow handler doesn't head-of-line-block.
                    await self._rpc_sem.acquire()
                    asyncio.create_task(self._handle_rpc(env))
                elif env.kind == KIND_CMD_RES:
                    fut = self._pending_cmds.pop(env.id, None)
                    if fut is not None and not fut.done():
                        err = env.payload.get("error") if isinstance(env.payload, dict) else None
                        if err is not None:
                            fut.set_exception(RuntimeError(str(err)))
                        else:
                            fut.set_result(dict(env.payload or {}))
                    else:
                        log.warning("worker %s: cmd_res for unknown id %s",
                                    self.agent_id, env.id)
                elif env.kind == KIND_EVENT:
                    self._offer_event(env)
                elif env.kind == KIND_STATUS:
                    await self._handle_status(env)
                else:
                    # A worker sending cmd/rpc_res to the daemon is malformed.
                    log.warning("worker %s sent unexpected kind %r", self.agent_id, env.kind)
        except (EOFError, OSError):
            log.info("worker %s pipe closed", self.agent_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker %s channel reader crashed", self.agent_id)

    async def _handle_rpc(self, env: Envelope) -> None:
        name = env.name
        handler = self._rpc_handlers.get(name)
        try:
            try:
                if name not in RPC_NAMES:
                    raise ValueError(f"unknown rpc name: {name!r}")
                if handler is None:
                    # P4 wires these; until then answer honestly.
                    raise NotImplementedError(f"rpc handler not wired: {name}")
                # AUTHORITATIVE agent_id — the worker's body cannot override it.
                result = await handler(self.agent_id, dict(env.payload or {}))
                await self._t.send(Envelope.rpc_res(env.id, result if isinstance(result, dict) else {"result": result}))
            except Exception as e:
                log.debug("rpc %r for %s failed: %s", name, self.agent_id, e)
                try:
                    await self._t.send(Envelope.rpc_res(env.id, error=f"{type(e).__name__}: {e}"))
                except Exception:
                    log.warning("failed to send rpc error for %s", self.agent_id)
        finally:
            # Release the H3 in-flight slot the reader acquired for this rpc.
            self._rpc_sem.release()

    def _offer_event(self, env: Envelope) -> None:
        """Enqueue an event, dropping the OLDEST on overflow (events droppable)."""
        try:
            self._events.put_nowait(env)
        except asyncio.QueueFull:
            try:
                dropped = self._events.get_nowait()
                log.warning("worker %s event queue full; dropped oldest event %r",
                            self.agent_id, dropped.name)
                self._events.put_nowait(env)
            except asyncio.QueueEmpty:
                pass

    async def _drain_events(self) -> None:
        try:
            while not self._closed:
                env = await self._events.get()
                if self._event_sink is not None:
                    try:
                        # OVERWRITE source with the authoritative agent_id.
                        await self._event_sink(self.agent_id, env.name, dict(env.payload or {}))
                    except Exception:
                        log.exception("event sink failed for %s", self.agent_id)
        except asyncio.CancelledError:
            raise

    async def _handle_status(self, env: Envelope) -> None:
        if self._status_sink is not None:
            try:
                await self._status_sink(self.agent_id, env.name, dict(env.payload or {}))
            except Exception:
                log.exception("status sink failed for %s", self.agent_id)

    async def close(self) -> None:
        self._closed = True
        # Fail any outstanding cmd replies so a caller in send_cmd_await doesn't
        # hang past channel teardown (mirrors DaemonClient's pending-rpc drain).
        for fut in self._pending_cmds.values():
            if not fut.done():
                fut.set_exception(ConnectionError("worker channel closed"))
        self._pending_cmds.clear()
        for task in (self._reader, self._event_worker):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._t.close()
