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
    async def on_cmd(self, env: Envelope) -> Optional[dict]:
        """Handle one inbound daemon cmd.

        Returns a dict for cmds that carried an id (expect_reply): the
        DaemonClient reader sends it back as the cmd_res body. Fire-and-forget
        cmds (id=None) return None and no reply is sent — unchanged behavior.
        """
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
            # Live-inject a user message into the running loop and poll for the
            # consumption ack, mirroring ExecutionControl.send_delegate_message's
            # in-process path (the provider lives HERE under isolation, so the
            # poll runs worker-side). Returns the ack to the daemon over cmd_res:
            #   {"status": "injected"}                   — consumed by the loop
            #   {"status": "inbox_fallback",
            #    "content": <reclaimed>}                 — raced turn end / dropped;
            #                                              the DAEMON owns the inbox
            #                                              re-post (it holds the
            #                                              InboxManager, not us).
            return await self._handle_send_user_message(env.payload or {})
        elif name == "request_compaction":
            # §15 manual compaction of a RUNNING worker: set the provider's
            # compact-requested flag (honored at the next generic-loop iteration
            # boundary). The provider lives in THIS process, so the daemon-side
            # compact_agent tool forwards the request here. Returns:
            #   {"accepted": True}      — a generic loop is active and took the flag
            #   {"accepted": False}     — no active loop (race: just ended) → daemon
            #                             falls through to the idle-store path
            #   {"unsupported": True}   — a bridge/SDK worker: its loop lives in the
            #                             node subprocess with no compact control,
            #                             so the daemon answers unsupported_while_running
            #                             (matches the in-process bridge branch).
            prov = self._provider
            payload = env.payload or {}
            requested_by = str(payload.get("requested_by", "") or "")
            if prov is not None and self._provider_is_bridge(prov):
                return {"unsupported": True}
            accepted = False
            fn = getattr(prov, "request_compaction", None) if prov else None
            if callable(fn):
                try:
                    accepted = bool(fn(requested_by=requested_by))
                except Exception:
                    log.debug("[%s] request_compaction failed", self._agent_label,
                              exc_info=True)
            return {"accepted": accepted}
        elif name == "get_context":
            # Context-breakdown probe of a RUNNING worker: measure the live
            # provider's context snapshot (system / tools / message stack) in
            # THIS process, where the provider actually lives. Returns:
            #   {"breakdown": {...}}   — live measurement succeeded
            #   {"unsupported": True}  — bridge/SDK worker or no live snapshot;
            #                            the daemon falls back to the
            #                            reconstructed (store-based) path
            prov = self._provider
            if prov is None or self._provider_is_bridge(prov):
                return {"unsupported": True}
            try:
                from .context_inspect import breakdown_from_provider
                bd = breakdown_from_provider(prov, "live_worker")
            except Exception:
                log.debug("[%s] get_context breakdown failed", self._agent_label,
                          exc_info=True)
                bd = None
            if bd is None:
                return {"unsupported": True}
            return {"breakdown": bd}
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
        return None

    async def _handle_send_user_message(self, payload: dict) -> dict:
        """Live-inject a steering message and poll for its consumption ack.

        Mirrors ExecutionControl.send_delegate_message's in-process branch, run
        WORKER-SIDE because the provider is local here:
          - no provider yet / no send_user_message contract -> stage for the
            daemon inbox fallback (the next run picks it up);
          - provider reports an injection id -> poll injection_consumed briefly;
            on timeout reclaim via take_unconsumed_injection and ask the daemon
            to re-post it to the inbox;
          - provider with no id contract (returns None/True) -> treat as injected
            (prior behavior).
        """
        content = str(payload.get("content", ""))
        if not content:
            return {"status": "injected"}
        # Always stage in the fallback list too, so a shutdown-cancelled run
        # doesn't silently drop it (belt-and-suspenders with the inbox re-post).
        self._injected_messages.append(content)
        prov = self._provider
        inject = getattr(prov, "send_user_message", None) if prov else None
        if inject is None:
            # No live loop to consume it — the daemon must inbox it.
            return {"status": "inbox_fallback", "content": content}
        try:
            inj_id = await inject(content)
        except Exception:
            log.debug("[%s] live send_user_message failed", self._agent_label,
                      exc_info=True)
            return {"status": "inbox_fallback", "content": content}

        # Providers whose send_user_message doesn't report an id (older / the
        # base generic queue returning True) can't be verified — injected.
        if not isinstance(inj_id, str) or not inj_id:
            return {"status": "injected"}

        consumed = await self._await_injection_consumed(prov, inj_id)
        if consumed:
            return {"status": "injected"}

        take = getattr(prov, "take_unconsumed_injection", None)
        pending = take(inj_id) if callable(take) else content
        if pending is None:
            # Consumed in the race between the poll and the take — good.
            return {"status": "injected"}
        log.info("[%s] injection %s not consumed — daemon inbox fallback",
                 self._agent_label, inj_id)
        return {"status": "inbox_fallback", "content": pending}

    async def _await_injection_consumed(
        self, provider: Any, inj_id: str, timeout: float = 2.0, interval: float = 0.05,
    ) -> bool:
        """Poll ``provider.injection_consumed(inj_id)`` up to ``timeout`` s.

        Worker-side copy of ExecutionControl._await_injection_consumed (same
        window/interval) so the injected/inbox_fallback boundary matches the
        in-process path byte-for-byte."""
        checker = getattr(provider, "injection_consumed", None)
        if not callable(checker):
            return True  # can't verify — assume delivered
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if checker(inj_id):
                return True
            await asyncio.sleep(interval)
        return checker(inj_id)

    @staticmethod
    def _provider_is_bridge(prov: Any) -> bool:
        """True if the live provider drives its loop in a node/SDK subprocess
        (BridgeProvider / CodexBridgeProvider) rather than the generic base
        loop. Bridge has no compact-control input, so manual compaction of a
        running bridge is unsupported — mirrors the in-process tool branch.
        Import lazily + guard so a missing optional module never crashes the
        cmd handler."""
        try:
            from .providers.bridge import BridgeProvider
            if isinstance(prov, BridgeProvider):
                return True
        except Exception:
            pass
        try:
            from .providers.codex_bridge import CodexBridgeProvider
            if isinstance(prov, CodexBridgeProvider):
                return True
        except Exception:
            pass
        return False

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

    # Mark this process as a worker so the secret_* tool handlers (P5) will run
    # (they are fail-closed refused outside a worker). Set only here — a real
    # spawn always reaches this line; the smoke/no-pipe path above returns first.
    # The env-scrub that builds a grandchild's environment runs from the DAEMON's
    # env (not this worker's), so this marker never leaks down a spawn chain.
    os.environ["ATN_IS_WORKER"] = "1"

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
