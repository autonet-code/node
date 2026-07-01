"""Per-agent worker process entry point (Phase 1 — dark library).

Each cognitive agent will (P4) run in its OWN OS process spawned as::

    subprocess.Popen([sys.executable, "-u", "-m", "atn.agent_worker", ...])

so its kernel PID becomes a real security identity. THIS FILE IS NOT SPAWNED BY
THE DAEMON YET. It exists now so the connect + manifest + handshake + recv-loop
scaffolding is real and importable, and so ``python -m atn.agent_worker`` can be
driven by a smoke harness. The actual provider/cognitive loop is a TODO(P4) — for
now the worker connects, reads its manifest, waits for "go", and answers with a
ready status + heartbeats, then shuts down cleanly on the shutdown cmd.

HANDSHAKE (P1 subset)
---------------------
1. connect the inherited duplex pipe end (handle passed on argv; scrubbed of any
   secret — see the security note below).
2. read the MANIFEST frame: a ``cmd`` named ``manifest`` carrying non-secret spawn
   context (agent_id label for logging only, protocol version). Identity is
   pipe-bound on the daemon side; the manifest is informational.
3. send ``status: ready``.
4. block on the ``go`` field of a ``run`` cmd (P4 runs the loop here; P1 just
   acknowledges and idles emitting heartbeats).
5. on ``shutdown`` cmd, stop cleanly.

SECURITY
--------
- The worker NEVER receives a secret in its env or argv. The pipe handle and a
  logging label are the only things passed. Secret access (P4+) is mediated by
  the daemon over RPC (``request_secret`` becomes an authority RPC), gated by the
  PID the daemon recorded at spawn.
- All IPC uses length-prefixed JSON via send_bytes/recv_bytes (see worker_rpc).
  Never pickle.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

from .runtime.worker_rpc import (
    DaemonClient,
    Envelope,
    PipeTransport,
    PROTO_VERSION,
)

log = logging.getLogger("atn.agent_worker")


class AgentWorker:
    """The worker-side runtime. P1: handshake + idle heartbeat + clean shutdown.

    P4 will replace ``_run_execution`` with the real provider/cognitive loop
    (recreated locally in this process; the daemon graph is NOT shared).
    """

    def __init__(self, client: DaemonClient, *, heartbeat_secs: float = 30.0) -> None:
        self._client = client
        self._heartbeat_secs = heartbeat_secs
        self._agent_label = "?"
        self._ready = False
        self._shutdown = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None

    # -- inbound cmd handling ---------------------------------------------
    async def on_cmd(self, env: Envelope) -> None:
        name = env.name
        if name == "manifest":
            await self._on_manifest(env.payload)
        elif name == "run":
            await self._on_run(env.payload)
        elif name == "send_user_message":
            # P4: inject into the running loop. P1: no-op ack.
            log.debug("[%s] send_user_message (P1 no-op)", self._agent_label)
        elif name == "interrupt":
            # Cooperative interrupt. P4: signal the loop. P1: no-op.
            log.debug("[%s] interrupt (P1 no-op)", self._agent_label)
        elif name == "shutdown":
            log.info("[%s] shutdown requested", self._agent_label)
            self._shutdown.set()
        else:
            log.warning("[%s] unknown cmd %r", self._agent_label, name)

    async def _on_manifest(self, payload: dict) -> None:
        # Manifest is informational only; identity is pipe-bound on the daemon.
        self._agent_label = str(payload.get("agent_label", "?"))
        peer_v = payload.get("v", PROTO_VERSION)
        if peer_v != PROTO_VERSION:
            log.warning("[%s] manifest protocol v%s != worker v%s",
                        self._agent_label, peer_v, PROTO_VERSION)
        log.info("[%s] manifest received (pid=%s)", self._agent_label, os.getpid())
        self._ready = True
        await self._client.send_status("ready", {"pid": os.getpid(), "v": PROTO_VERSION})
        # Start heartbeats once we've announced readiness.
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _on_run(self, payload: dict) -> None:
        # The daemon fires the manifest -> ready, then (after _on_pid_bound on
        # its side) sends run with go=True. P1 just acknowledges.
        if not payload.get("go", False):
            log.debug("[%s] run without go — holding", self._agent_label)
            return
        result = await self._run_execution(payload)
        # execution_done carries the final ProviderResponse (P4). P1: minimal.
        await self._client.send_status("execution_done", result)

    async def _run_execution(self, payload: dict) -> dict:
        # TODO(P4): recreate the provider locally and run the real cognitive
        # loop here (route_tool_call splits LOCAL in-worker vs AUTHORITY RPC).
        # P1 stand-in: echo that we would have run.
        log.info("[%s] (P1) run acknowledged; provider loop is TODO(P4)", self._agent_label)
        return {"status": "noop", "detail": "P1 worker: provider loop not implemented", "output": None}

    async def _heartbeat_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=self._heartbeat_secs)
                except asyncio.TimeoutError:
                    await self._client.send_status("heartbeat", {"pid": os.getpid()})
        except asyncio.CancelledError:
            raise

    # -- lifecycle ---------------------------------------------------------
    async def run(self) -> None:
        self._client.start()
        await self._shutdown.wait()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._client.close()
        log.info("[%s] worker exited cleanly", self._agent_label)


# ---------------------------------------------------------------------------
# Pipe reconstruction from an inherited handle
# ---------------------------------------------------------------------------

def _connection_from_handle(handle: int) -> object:
    """Rebuild a duplex multiprocessing Connection from an inherited OS handle.

    The daemon (P4) creates the pipe with ``multiprocessing.Pipe(duplex=True)``,
    makes the child's end inheritable, and passes its integer handle on argv
    (on Windows via the STARTUPINFO handle-inherit list; on POSIX the fd is
    inherited directly).

    The wrapper class differs by platform: ``multiprocessing.Pipe`` yields a
    Windows *named pipe* (``PipeConnection``, overlapped I/O) but a POSIX
    *socketpair-backed* ``Connection``. Using the wrong class raises WinError
    10038 ("not a socket") — so pick per platform.
    """
    if sys.platform == "win32":
        from multiprocessing.connection import PipeConnection
        return PipeConnection(handle)
    from multiprocessing.connection import Connection
    return Connection(handle)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m atn.agent_worker")
    p.add_argument(
        "--pipe-handle", type=int, default=None,
        help="inherited OS handle/fd for the daemon<->worker duplex pipe",
    )
    p.add_argument(
        "--agent-label", type=str, default="?",
        help="human-readable label for logging ONLY (never an identity/secret)",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


async def _amain(ns: argparse.Namespace) -> int:
    if ns.pipe_handle is None:
        # Smoke mode: no pipe was handed in. A real spawn always passes one.
        # This lets ``python -m atn.agent_worker`` exit cleanly instead of
        # hanging, which is what the import/handshake smoke check exercises.
        log.info("no --pipe-handle: nothing to connect (P1 smoke exit)")
        return 0

    conn = _connection_from_handle(ns.pipe_handle)
    transport = PipeTransport(conn)
    # DaemonClient and AgentWorker reference each other: build the client first,
    # then the worker, then wire the client's cmd handler to the worker.
    client = DaemonClient(transport)
    worker = AgentWorker(client)
    client.set_cmd_handler(worker.on_cmd)

    await worker.run()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ns = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(ns.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return asyncio.run(_amain(ns))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
