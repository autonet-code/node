"""Running the tools behind session services (docs/services_market.md,
Decision 2026-11-12: once a service is published, the daemon serves it and
takes payment with no further decisions or actions).

A session service's tool is a server. The daemon starts it with three
variables in its environment and nothing else beyond what its manifest
declares:

- ``AUTONET_SERVICE_PORT``: the port on 127.0.0.1 to serve buyers on. Each
  buyer's session reaches the tool as its own TCP connection from the
  daemon.
- ``AUTONET_SERVICE_SETTINGS``: the service's settings, as JSON (the model,
  for example): the tool serves exactly what the spec promises.
- ``AUTONET_USAGE_URL``: where a tool that reports usage (its manifest's
  ``session.reports_usage``) posts what each connection used, after each
  reply: ``POST`` with ``{"connection": <the connection's remote port, as
  the tool sees it>, "input_tokens": n, "output_tokens": m}``. The daemon
  charges it and cuts the connection when the buyer's credits do not cover
  it.

The tool's manifest says whether one process serves every session
(``process: "shared"``, a model server: started when the daemon starts and
kept running) or each buyer gets a fresh one (``"per_session"``, a rented
machine: started when a session opens, stopped when it ends), and how many
sessions it takes at once (``max_sessions``, 0 = no cap). A buyer is refused
at opening, before paying, when the tool is down or full.

A tool that will not start or keeps dying is restarted with a doubling
backoff; every failure is recorded (``errors``, and
``<data_dir>/services/service_errors.jsonl``) for the notifications that
will report them to the Agent or the owner. Nothing here notifies yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

READY_TIMEOUT = 30.0          # a started tool must accept connections by then
SUPERVISE_INTERVAL = 5.0
RESTART_BASE = 5.0
RESTART_CAP = 300.0
MAX_ERRORS_KEPT = 200
MAX_REPORT_BYTES = 16 * 1024


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _backoff(failures: int) -> float:
    return min(RESTART_BASE * (2 ** max(0, failures - 1)), RESTART_CAP)


async def _wait_ready(port: int, proc: Any, timeout: float) -> Optional[str]:
    """None once ``port`` accepts a connection; else why not."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            return f"the tool exited with code {proc.returncode} before serving"
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), 1.0)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            return None
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(0.2)
    return f"the tool did not serve on port {port} within {int(timeout)}s"


@dataclass
class _Server:
    digest: str
    port: int
    token: str
    proc: Any = None
    secret_session: Any = None
    sessions: int = 0
    failures: int = 0
    next_try: float = 0.0
    starting: Optional[asyncio.Task] = None
    problem: str = ""          # why the last start failed


@dataclass
class Lease:
    """One session's use of a tool server: where to connect, and how to
    give it back."""
    host: str
    port: int
    token: str
    release: Callable[[], Any] = field(repr=False, default=lambda: None)


class ServiceRunner:
    """Starts, supervises and stops the tools behind session services."""

    def __init__(self, runtime: Any, data_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(data_dir) / "services"
        self._servers: dict[str, _Server] = {}       # shared: digest -> server
        self._leased: dict[str, _Server] = {}        # per_session: token -> server
        self._sessions: dict[str, int] = {}          # digest -> open sessions
        self._watch: dict[tuple, Callable[[str, int], Any]] = {}
        self._errors: list[dict[str, Any]] = []
        self._usage_server: Optional[asyncio.base_events.Server] = None
        self._usage_port = 0
        self._supervisor: Optional[asyncio.Task] = None
        self._tokens: dict[str, str] = {}            # token -> digest

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        if self._usage_server is None:
            self._usage_server = await asyncio.start_server(
                self._on_usage, "127.0.0.1", 0)
            self._usage_port = int(self._usage_server.sockets[0].getsockname()[1])
        if self._supervisor is None:
            self._supervisor = asyncio.get_running_loop().create_task(self._supervise())

    async def stop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.cancel()
            self._supervisor = None
        for server in list(self._servers.values()) + list(self._leased.values()):
            await self._stop_server(server)
        self._servers.clear()
        self._leased.clear()
        if self._usage_server is not None:
            self._usage_server.close()
            self._usage_server = None

    def _tool_store(self) -> Any:
        return getattr(self._runtime, "tool_store", None)

    def _service_store(self) -> Any:
        return getattr(self._runtime, "service_store", None)

    def _session_block(self, record: Any) -> dict[str, Any]:
        tools = self._tool_store()
        tool = tools.get(record.backing_tool) if tools is not None else None
        block = tool.manifest.get("session") if tool is not None else None
        return block if isinstance(block, dict) else {}

    def _shared_services(self) -> list[Any]:
        store = self._service_store()
        if store is None:
            return []
        out = []
        for rec in store.list(include_retired=False):
            if rec.mode != "session":
                continue
            if self._session_block(rec).get("process", "shared") == "shared":
                out.append(rec)
        return out

    async def _supervise(self) -> None:
        while True:
            try:
                wanted = {rec.digest: rec for rec in self._shared_services()}
                for digest, server in list(self._servers.items()):
                    if digest not in wanted:
                        await self._stop_server(server)
                        self._servers.pop(digest, None)
                for digest, rec in wanted.items():
                    server = self._servers.get(digest)
                    if server is not None:
                        if server.starting is not None or self._running(server):
                            continue
                        await self._reap(server)
                        if server.next_try > time.monotonic():
                            continue
                    await self._ensure_shared(rec)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the supervisor never dies
                log.warning("service runner: supervision failed: %s", exc)
            await asyncio.sleep(SUPERVISE_INTERVAL)

    # -- servers ---------------------------------------------------------------

    @staticmethod
    def _running(server: _Server) -> bool:
        return server.proc is not None and server.proc.returncode is None

    async def _reap(self, server: _Server) -> None:
        """A shared tool whose process exited: record it and schedule the
        restart (a doubling backoff)."""
        proc = server.proc
        if proc is None or proc.returncode is None:
            return
        self.record_error(server.digest, f"the tool exited with code {proc.returncode}")
        await self._stop_server(server)
        server.proc = None
        server.failures += 1
        server.next_try = time.monotonic() + _backoff(server.failures)

    def _env(self, record: Any, port: int, token: str) -> dict[str, str]:
        return {
            "AUTONET_SERVICE_PORT": str(port),
            "AUTONET_SERVICE_SETTINGS": json.dumps(record.settings, sort_keys=True),
            "AUTONET_USAGE_URL": f"http://127.0.0.1:{self._usage_port}/usage/{token}",
        }

    async def _launch(self, record: Any, server: _Server) -> Optional[str]:
        """Start ``server``'s process; None when it serves, else why not."""
        tools = self._tool_store()
        tool = tools.get(record.backing_tool) if tools is not None else None
        if tool is None:
            return "the service's tool is no longer registered on this daemon"
        log_path = self._dir / "logs" / f"{record.digest[:16]}.log"
        try:
            proc, secret_session = await tools.start_server(
                tool, extra_env=self._env(record, server.port, server.token),
                log_path=log_path)
        except Exception as exc:  # noqa: BLE001
            return f"the tool could not start: {exc}"
        server.proc, server.secret_session = proc, secret_session
        problem = await _wait_ready(server.port, proc, READY_TIMEOUT)
        if problem is not None:
            await self._stop_server(server)
            server.proc = None
            return problem
        return None

    async def _ensure_shared(self, record: Any) -> Optional[str]:
        server = self._servers.get(record.digest)
        if server is None:
            token = secrets.token_hex(16)
            server = _Server(digest=record.digest, port=_free_port(), token=token)
            self._servers[record.digest] = server
            self._tokens[token] = record.digest
        if self._running(server):
            return None
        if server.starting is not None:
            await asyncio.shield(server.starting)
            return None if server.proc is not None else (
                server.problem or "the tool is not running")
        await self._reap(server)
        if server.next_try > time.monotonic():
            # Restarted by the supervisor once the backoff has passed, not
            # by every buyer that asks meanwhile.
            return "the tool failed and is restarting"

        async def go() -> None:
            server.port = _free_port()
            problem = await self._launch(record, server)
            server.problem = problem or ""
            if problem is None:
                server.failures = 0
                log.info("service %s: tool serving on port %d", record.name, server.port)
            else:
                server.failures += 1
                server.next_try = time.monotonic() + _backoff(server.failures)
                self.record_error(record.digest, problem)
        server.starting = asyncio.get_running_loop().create_task(go())
        try:
            await asyncio.shield(server.starting)
        finally:
            server.starting = None
        return None if server.proc is not None else (server.problem or "the tool is not running")

    async def _stop_server(self, server: _Server) -> None:
        if server.proc is None:
            return
        tools = self._tool_store()
        try:
            await tools.stop_server(server.proc, server.secret_session)
        except Exception as exc:  # noqa: BLE001
            log.debug("stopping a service tool failed: %s", exc)

    async def lease(self, record: Any) -> Any:
        """A connection target for one new session of ``record``, or a
        string saying why there is none (refused before payment)."""
        block = self._session_block(record)
        if not block:
            return "the service's tool does not serve sessions"
        digest = record.digest
        cap = int(block.get("max_sessions") or 0)
        open_now = self._sessions.get(digest, 0)
        if cap and open_now >= cap:
            return f"the service is full ({cap} sessions at once); try again later"
        # The place is held before anything is awaited, so sessions opening
        # at once cannot pass the cap together; a lease is given back once.
        self._sessions[digest] = open_now + 1
        released = False

        def unreserve() -> bool:
            nonlocal released
            if released:
                return False
            released = True
            self._sessions[digest] = max(0, self._sessions.get(digest, 1) - 1)
            return True

        if block.get("process", "shared") == "per_session":
            token = secrets.token_hex(16)
            server = _Server(digest=digest, port=_free_port(), token=token)
            self._tokens[token] = digest
            problem = await self._launch(record, server)
            if problem is not None:
                unreserve()
                self._tokens.pop(token, None)
                self.record_error(digest, problem)
                return f"the service's tool is not available: {problem}"
            self._leased[token] = server

            async def release() -> None:
                if unreserve():
                    self._tokens.pop(token, None)
                    self._leased.pop(token, None)
                    await self._stop_server(server)
            return Lease("127.0.0.1", server.port, token, release)
        problem = await self._ensure_shared(record)
        if problem is not None:
            unreserve()
            return f"the service's tool is not available: {problem}"
        shared = self._servers[digest]

        async def release_shared() -> None:
            unreserve()
        return Lease("127.0.0.1", shared.port, shared.token, release_shared)

    # -- usage reports -----------------------------------------------------------

    def watch(self, token: str, local_port: int, callback: Callable[[str, int], Any]) -> None:
        """Send the usage the tool reports for the connection whose port on
        the daemon's side is ``local_port`` to ``callback(measure, amount)``."""
        self._watch[(token, int(local_port))] = callback

    def unwatch(self, token: str, local_port: int) -> None:
        self._watch.pop((token, int(local_port)), None)

    async def _on_usage(self, reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> None:
        status = "204 No Content"
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = (lines[0].split(" ", 2) + ["", ""])[:3]
            length = 0
            for line in lines[1:]:
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip() or 0)
            if method != "POST" or not path.startswith("/usage/") or length > MAX_REPORT_BYTES:
                status = "400 Bad Request"
            else:
                body = await asyncio.wait_for(reader.readexactly(length), 10) if length else b""
                token = path[len("/usage/"):].strip("/")
                if token not in self._tokens:
                    status = "404 Not Found"
                else:
                    data = json.loads(body.decode("utf-8") or "{}")
                    for report in (data if isinstance(data, list) else [data]):
                        self._apply_report(token, report)
        except Exception as exc:  # noqa: BLE001
            log.debug("bad usage report: %s", exc)
            status = "400 Bad Request"
        try:
            writer.write(f"HTTP/1.1 {status}\r\nContent-Length: 0\r\n"
                         "Connection: close\r\n\r\n".encode("ascii"))
            await writer.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            writer.close()

    def _apply_report(self, token: str, report: Any) -> None:
        if not isinstance(report, dict):
            return
        try:
            port = int(report.get("connection"))
        except (TypeError, ValueError):
            return
        callback = self._watch.get((token, port))
        if callback is None:
            return
        for measure in ("input_tokens", "output_tokens"):
            try:
                amount = int(report.get(measure) or 0)
            except (TypeError, ValueError):
                continue
            if amount > 0:
                try:
                    callback(measure, amount)
                except Exception as exc:  # noqa: BLE001
                    log.warning("charging reported usage failed: %s", exc)

    # -- errors ------------------------------------------------------------------

    def record_error(self, digest: str, error: str) -> None:
        """Record a failure of a service's tool, for later notifications."""
        row = {"ts": int(time.time()), "digest": digest, "error": str(error)[:2000]}
        self._errors.append(row)
        del self._errors[:-MAX_ERRORS_KEPT]
        log.warning("service %s: %s", digest[:16], error)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with (self._dir / "service_errors.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def errors(self, digest: str = "") -> list[dict[str, Any]]:
        return [dict(r) for r in self._errors if not digest or r["digest"] == digest]


def runner_for(runtime: Any) -> ServiceRunner:
    """The runtime's service runner (created on first use)."""
    existing = runtime.__dict__.get("service_runner")
    if isinstance(existing, ServiceRunner):
        return existing
    try:
        data_dir = Path(runtime._config.data_dir)
    except Exception:  # noqa: BLE001
        data_dir = Path(".")
    runner = ServiceRunner(runtime, data_dir)
    runtime.__dict__["service_runner"] = runner
    return runner
