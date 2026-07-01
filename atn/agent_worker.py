"""Per-agent worker process entry point (Phase 4 — cognitive loop in-worker).

Each cognitive agent that uses an API provider (Anthropic / OpenAI-compatible)
runs in its OWN OS process, spawned by the daemon as::

    subprocess.Popen([sys.executable, "-u", "-m", "atn.agent_worker", ...])

so its kernel PID becomes a real security identity. On "go" the worker rebuilds
its provider LOCALLY from the run manifest and drives the SAME cognitive loop the
in-process path drives (``Provider.send_orchestrate`` + the streaming/tool
cycle), with one split:

  - LOCAL tools (the 5 sandboxed shell executors) execute IN this process.
  - AUTHORITY tools (``is_authority_tool``) become an RPC back to the daemon
    (``framework_tool`` / ``surface_tool`` / ``mcp_tool``). A spawn attempt
    returns a clean "not supported under isolation" tool-result (P6), never a
    crash.
  - budget_check before the run, budget_record as usage accrues — over RPC, so
    the DAEMON ledger stays the single source of truth (P0 risk #1).
  - step.output + agent.tool_use_* stream to the daemon via ``emit``.
  - on completion/error the worker sends ``execution_done`` carrying the final
    ProviderResponse-derived dict so the daemon can populate ``record.output``
    BEFORE it finalizes.

HANDSHAKE
---------
1. connect the inherited duplex pipe end (handle on argv; env scrubbed of
   secrets by the daemon).
2. read the MANIFEST ``cmd`` (protocol version + logging label). Identity is
   pipe-bound on the daemon side; the manifest is informational for identity.
3. send ``status: ready``.
4. block on the ``go`` field of a ``run`` cmd, which carries the full run
   manifest (provider_config, defn, prebuilt message/system/tools/history).
5. run the loop, send ``execution_done``.
6. on ``shutdown`` cmd (or pipe EOF), stop cleanly.

SECURITY
--------
- The worker NEVER receives a secret in its env or argv. The pipe handle and a
  logging label are the only things on argv. The provider API key travels in
  ``provider_config`` on the (local, private, non-pickled) pipe — isolated in
  one place so the vault track can later replace it with a PID-gated RPC fetch
  (see worker_provider.py SECRETS note). Secret allowance is explicitly NOT
  implemented here.
- All IPC uses length-prefixed JSON via send_bytes/recv_bytes (see worker_rpc).
  Never pickle.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Optional

from .runtime.worker_rpc import (
    DaemonClient,
    Envelope,
    PipeTransport,
    PROTO_VERSION,
)

log = logging.getLogger("atn.agent_worker")


class AgentWorker:
    """The worker-side runtime: handshake + in-worker cognitive loop + clean
    shutdown.

    The provider is recreated LOCALLY in this process from the run manifest; the
    daemon object graph is NOT shared. Authority tools cross the pipe as RPCs.
    """

    def __init__(self, client: DaemonClient, *, heartbeat_secs: float = 30.0) -> None:
        self._client = client
        self._heartbeat_secs = heartbeat_secs
        self._agent_label = "?"
        self._ready = False
        self._shutdown = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None
        # The background task running the current execution (spawned from the
        # "run" cmd so the recv loop stays free to resolve RPC replies).
        self._run_task: Optional[asyncio.Task] = None
        # Live provider for the current run (set inside _run_execution). Held so
        # send_user_message / interrupt cmds can reach the running loop.
        self._provider: Any = None
        # Extra user messages injected mid-run (delegate_message live-inject).
        self._injected_messages: list[str] = []

    # -- inbound cmd handling ---------------------------------------------
    async def on_cmd(self, env: Envelope) -> None:
        name = env.name
        if name == "manifest":
            await self._on_manifest(env.payload)
        elif name == "run":
            # CRITICAL: run in a BACKGROUND task, not inline. The cognitive loop
            # issues RPCs (budget_check/budget_record/authority tools) whose
            # rpc_res frames are read by THIS SAME recv loop — running the loop
            # inline here would block the reader and deadlock every RPC. A
            # background task keeps the reader free to resolve rpc futures and to
            # service concurrent send_user_message/interrupt/shutdown cmds.
            payload = env.payload
            if not payload.get("go", False):
                log.debug("[%s] run without go — holding", self._agent_label)
            elif self._run_task is not None and not self._run_task.done():
                log.warning("[%s] run already in progress — ignoring duplicate",
                            self._agent_label)
            else:
                self._run_task = asyncio.create_task(
                    self._on_run(payload), name=f"worker-run-{self._agent_label}")
        elif name == "send_user_message":
            # Live-inject a user message into the running loop. The generic
            # base-class loop doesn't expose a mid-flight injection point, so we
            # stage it: a provider that supports send_user_message gets it now;
            # otherwise it's queued for the daemon to re-post to the inbox on the
            # next run (the daemon's send_delegate_message handles that fallback).
            content = str((env.payload or {}).get("content", ""))
            if content:
                self._injected_messages.append(content)
                prov = self._provider
                inject = getattr(prov, "send_user_message", None) if prov else None
                if inject is not None:
                    try:
                        await inject(content)
                    except Exception:
                        log.debug("[%s] live send_user_message failed", self._agent_label,
                                  exc_info=True)
        elif name == "interrupt":
            # Cooperative interrupt: signal the provider loop to stop after the
            # current turn (matches ExecutionControl.interrupt semantics).
            prov = self._provider
            if prov is not None and hasattr(prov, "interrupt"):
                try:
                    await prov.interrupt()
                except Exception:
                    log.debug("[%s] interrupt failed", self._agent_label, exc_info=True)
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
        # its side) sends run with go=True carrying the full run manifest.
        if not payload.get("go", False):
            log.debug("[%s] run without go — holding", self._agent_label)
            return
        try:
            result = await self._run_execution(payload)
        except Exception as exc:
            log.exception("[%s] run failed", self._agent_label)
            result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "output": None,
                "token_usage": {},
            }
        # execution_done carries the ProviderResponse-derived dict so the daemon
        # can populate record.output BEFORE finalize. Always sent (even on
        # error) so the daemon's clean path runs instead of the kill path.
        await self._client.send_status("execution_done", result)

    async def _run_execution(self, payload: dict) -> dict:
        """Run the cognitive loop IN this worker process.

        ``payload`` is the run manifest (see agent_worker module docstring for
        the schema). Returns the ProviderResponse-derived dict the daemon uses
        to populate ``record.output`` + reconcile the budget.

        The split:
          - provider is rebuilt locally from ``payload["provider_config"]``;
          - LOCAL shell tools run here; AUTHORITY tools RPC to the daemon;
          - budget_check pre-flight + budget_record per turn, both over RPC;
          - step.output / tool events stream to the daemon via ``emit``.
        """
        from .runtime.worker_provider import build_provider_from_manifest
        from .runtime.worker_loop import run_cognitive_loop

        provider = build_provider_from_manifest(payload.get("provider_config") or {})
        self._provider = provider
        self._injected_messages = []
        try:
            result = await run_cognitive_loop(
                client=self._client,
                provider=provider,
                manifest=payload,
                agent_label=self._agent_label,
            )
            return result
        finally:
            self._provider = None
            try:
                await provider.close()
            except Exception:
                pass

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
        # Cancel any in-flight run so a cooperative shutdown mid-execution
        # doesn't leave the loop dangling (the daemon finalizes on-behalf).
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
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
