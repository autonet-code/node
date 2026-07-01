"""CLI entry point — event stream + interactive commands.

Starts the Runtime and renders events to the terminal in real time.
Accepts commands on stdin for agent management and kill switches.

Usage:
    python -m atn
    atn              (if installed via pip)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import ATNConfig, load_config
from .events import Event, EventBus, EventType
from .loader import load_agents_dir
from .runtime import Runtime
from .ws_server import WebSocketBridge, DEFAULT_PORT

console = Console()
log = logging.getLogger(__name__)

# Maps event types to (colour, symbol) for the log stream.
_EVENT_STYLE: dict[EventType, tuple[str, str]] = {
    EventType.RUNTIME_STARTED:      ("bold green",   "+"),
    EventType.RUNTIME_STOPPED:      ("bold red",     "-"),
    EventType.AGENT_REGISTERED:     ("cyan",         "+"),
    EventType.AGENT_UNREGISTERED:   ("cyan",         "-"),
    EventType.AGENT_ACTIVATED:      ("green",        ">"),
    EventType.AGENT_DEACTIVATED:    ("yellow",       "#"),
    EventType.EXECUTION_QUEUED:     ("dim",          "~"),
    EventType.EXECUTION_STARTED:    ("bold yellow",  ">"),
    EventType.EXECUTION_COMPLETED:  ("bold green",   "v"),
    EventType.EXECUTION_FAILED:     ("bold red",     "x"),
    EventType.EXECUTION_KILLED:     ("bold magenta", "K"),
    EventType.STEP_STARTED:         ("blue",         "."),
    EventType.STEP_COMPLETED:       ("green",        "v"),
    EventType.STEP_FAILED:          ("red",          "x"),
    EventType.STEP_KILLED:          ("magenta",      "K"),
    EventType.MESSAGE_POSTED:       ("dim cyan",     "m"),
    EventType.MESSAGE_DRAINED:      ("dim",          "d"),
    EventType.SCHEDULE_TRIGGERED:   ("dim yellow",   "t"),
}


def _format_event(event: Event) -> str:
    """Format a single event as a coloured line for the terminal."""
    style, symbol = _EVENT_STYLE.get(event.type, ("white", "?"))
    ts = event.timestamp.strftime("%H:%M:%S")
    etype = event.type.value

    parts: list[str] = []
    for key in ("agent_id", "execution_id", "step_name", "step_type",
                "status", "trigger_source", "output_preview", "error"):
        val = event.data.get(key)
        if val:
            if key == "execution_id":
                val = val[:8]
            if key == "error":
                val = str(val)[:60]
            if key == "output_preview":
                val = str(val)[:60]
            parts.append(f"{key}={val}")

    detail = "  ".join(parts)
    return f"[dim]{ts}[/] [{style}]{symbol} {etype:<26}[/]  {detail}"


async def _print_event(event: Event) -> None:
    """Event handler that prints each event to the console."""
    console.print(_format_event(event))


def _print_status(runtime: Runtime) -> None:
    """Print a snapshot of all agents and running executions."""
    snap = runtime.snapshot()

    # Agents table
    t = Table(title="Agents", expand=False)
    t.add_column("ID", style="cyan", width=10)
    t.add_column("Name")
    t.add_column("Status", width=12)
    t.add_column("Schedule", width=10)
    t.add_column("Running", justify="center", width=8)
    t.add_column("Inbox", justify="center", width=6)

    for aid, info in snap["agents"].items():
        st = info["status"]
        scolor = {"active": "green", "running": "bold yellow",
                  "error": "bold red", "stopped": "red"}.get(st, "dim")
        t.add_row(
            aid, info["name"], f"[{scolor}]{st}[/]",
            info["schedule"] or "-",
            str(info["running"]) if info["running"] else "-",
            str(info["inbox"]) if info["inbox"] else "-",
        )

    if not snap["agents"]:
        t.add_row("-", "[dim]no agents[/]", "", "", "", "")
    console.print(t)

    # Executions table
    if snap["executions"]:
        t2 = Table(title="Running Executions", expand=False)
        t2.add_column("Execution", style="cyan", width=14)
        t2.add_column("Agent")
        t2.add_column("Step")
        t2.add_column("Source")
        t2.add_column("Duration", width=10)

        now = datetime.now(timezone.utc)
        for eid, info in snap["executions"].items():
            started = datetime.fromisoformat(info["started_at"])
            dur = (now - started).total_seconds()
            t2.add_row(eid[:12], info["agent_id"], info["step"],
                       info["trigger"], f"{dur:.1f}s")
        console.print(t2)
    else:
        console.print("[dim]No active executions.[/]")


# Module-level flag set by the `restart` command so the outer driver knows
# to re-exec the same Python process after cleanup completes (picks up
# code changes). Plain `quit` leaves this False and the process exits.
_restart_requested = False


def _print_help() -> None:
    console.print(
        "\n[bold]Commands:[/]\n"
        "  [cyan]status[/]              Show agents and running executions\n"
        "  [cyan]run[/] <agent_id>      Trigger an agent run\n"
        "  [cyan]kill[/] <agent_id>     Kill all executions of an agent\n"
        "  [cyan]kill[/] <exec_id>      Kill a specific execution\n"
        "  [cyan]killall[/]             Kill everything\n"
        "  [cyan]activate[/] <id>       Activate an agent for scheduling\n"
        "  [cyan]deactivate[/] <id>     Deactivate an agent\n"
        "  [cyan]agents[/]              List registered agents\n"
        "  [cyan]reload[/]              Reload agent definitions from YAML files\n"
        "  [cyan]reconcile[/]           Re-verify each agent's on-chain registration (run after a contract redeploy)\n"
        "  [cyan]history[/] <id>        Show execution history for an agent\n"
        "  [cyan]logs[/]                Show on-disk execution log summary\n"
        "  [cyan]clear[/] <id>          Clear execution history for an agent\n"
        "  [cyan]clearall[/]            Clear all execution history\n"
        "  [cyan]usage[/]               Show subscription utilization (Claude Max /usage equivalent)\n"
        "  [cyan]restart[/]             Stop and re-launch the daemon (picks up code changes)\n"
        "  [cyan]help[/]                Show this message\n"
        "  [cyan]quit[/] / [cyan]q[/]            Shutdown\n"
    )


async def _handle_command(line: str, runtime: Runtime) -> bool:
    """Process one command.  Returns False to signal shutdown."""
    parts = line.strip().split()
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("quit", "q", "exit"):
        return False

    elif cmd == "restart":
        global _restart_requested
        _restart_requested = True
        console.print("  [yellow]Restart requested — shutting down to re-exec...[/]")
        return False

    elif cmd == "help":
        _print_help()

    elif cmd == "status":
        _print_status(runtime)

    elif cmd == "agents":
        for defn, status in runtime.list_agents():
            scolor = {"active": "green", "running": "bold yellow",
                      "error": "bold red"}.get(status.value, "dim")
            console.print(
                f"  [cyan]{defn.id}[/]  {defn.name:<20}  [{scolor}]{status.value}[/]"
                f"  steps={len(defn.steps)}  schedule={defn.schedule or '-'}"
                f"  concurrency={defn.concurrency}"
            )
        if not runtime.list_agents():
            console.print("  [dim]No agents registered.[/]")

    elif cmd == "reload":
        await _load_agents(runtime, runtime._config)

    elif cmd == "reconcile":
        try:
            report = runtime.reconcile_chain_registrations()
        except Exception as exc:
            console.print(f"  [red]Reconcile failed: {exc}[/]")
            return True
        if report.get("skipped_reason") == "chain_unset":
            console.print("  [yellow]Chain not configured — nothing to reconcile.[/]")
            return True
        if report.get("error") == "rpc_failed":
            console.print(
                f"  [red]RPC failed (checked={report.get('checked', 0)}) — "
                f"local flags untouched. See logs for details.[/]"
            )
            return True
        checked = report.get("checked", 0)
        flipped = report.get("flipped", [])
        if checked == 0:
            console.print("  [dim]No agents currently marked as registered on chain.[/]")
        elif not flipped:
            console.print(f"  [green]All {checked} agent(s) still registered on chain.[/]")
        else:
            console.print(
                f"  [yellow]Flipped {len(flipped)} of {checked} agent(s) to unregistered: "
                f"{', '.join(flipped)}[/]"
            )

    elif cmd == "run" and args:
        try:
            eid = await runtime.trigger_run(args[0])
            if eid:
                console.print(f"  [green]Started execution {eid[:8]}[/]")
            else:
                console.print(f"  [yellow]At concurrency limit for {args[0]}[/]")
        except ValueError as e:
            console.print(f"  [red]{e}[/]")

    elif cmd == "kill" and args:
        target = args[0]
        # Try as execution_id first, then as agent_id
        if await runtime.kill_execution(target):
            console.print(f"  [magenta]Killed execution {target[:8]}[/]")
        else:
            n = await runtime.kill_agent(target)
            if n:
                console.print(f"  [magenta]Killed {n} execution(s) for {target}[/]")
            else:
                console.print(f"  [dim]Nothing running for {target}[/]")

    elif cmd == "killall":
        n = await runtime.kill_all()
        console.print(f"  [magenta]Killed {n} execution(s)[/]")

    elif cmd == "activate" and args:
        try:
            await runtime.activate_agent(args[0])
        except ValueError as e:
            console.print(f"  [red]{e}[/]")

    elif cmd == "deactivate" and args:
        try:
            await runtime.deactivate_agent(args[0])
        except ValueError as e:
            console.print(f"  [red]{e}[/]")

    elif cmd == "history" and args:
        records = runtime.execution_log.get_history(args[0])
        for rec in records:
            dur = ""
            if rec.completed_at and rec.started_at:
                dur = f"  {(rec.completed_at - rec.started_at).total_seconds():.1f}s"
            scolor = {"completed": "green", "failed": "red", "killed": "magenta"}.get(
                rec.status.value, "yellow")
            console.print(
                f"  {rec.execution_id[:8]}  [{scolor}]{rec.status.value:<10}[/]"
                f"  src={rec.trigger_source}  steps={len(rec.step_results)}{dur}"
            )
        if not records:
            console.print(f"  [dim]No history for {args[0]}[/]")

    elif cmd == "logs":
        summary = runtime.execution_log.disk_summary()
        if summary:
            t = Table(title="On-Disk Execution Logs", expand=False)
            t.add_column("Agent", style="cyan")
            t.add_column("Records", justify="right")
            total = 0
            for agent_id, count in summary.items():
                t.add_row(agent_id, str(count))
                total += count
            t.add_row("[bold]total[/]", f"[bold]{total}[/]")
            console.print(t)
        else:
            console.print("  [dim]No execution logs on disk.[/]")

    elif cmd == "clear" and args:
        removed = runtime.execution_log.clear_agent(args[0])
        if removed:
            console.print(f"  [green]Cleared history for {args[0]} (memory + disk)[/]")
        else:
            console.print(f"  [green]Cleared in-memory history for {args[0]}[/]")

    elif cmd == "usage":
        # Print subscription utilization captured from bridge providers'
        # SDKRateLimitEvent stream — Anthropic's view of "% of plan used,"
        # i.e. the same data Claude Code's /usage shows.
        any_data = False
        for pid, prov in runtime.providers._active_providers.items():
            rate_limits = getattr(prov, "_rate_limits", None)
            if not rate_limits:
                continue
            any_data = True
            console.print(f"\n  [bold cyan]{pid}[/]")
            for key, entry in rate_limits.items():
                util = entry.get("utilization")
                util_pct = f"{util * 100:.1f}%" if isinstance(util, (int, float)) else "?"
                status = entry.get("status", "?")
                resets = entry.get("resetsAt", "")
                console.print(
                    f"    {key:<20} util={util_pct:<8} status={status}"
                    + (f"  resets={resets}" if resets else "")
                )
        if not any_data:
            console.print(
                "  [dim]No rate-limit data yet. Run at least one turn on a "
                "subscription provider (claude_max, codex_max) first.[/]"
            )

    elif cmd == "clearall":
        n = runtime.execution_log.clear_all()
        console.print(f"  [green]Cleared all history ({n} file(s) deleted)[/]")

    else:
        console.print(f"  [dim]Unknown command: {cmd}.  Type 'help' for usage.[/]")

    return True


async def _input_loop(runtime: Runtime) -> None:
    """Read commands from stdin in a thread so we don't block the event loop."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Race between stdin input and shutdown signal
            read_task = asyncio.ensure_future(
                loop.run_in_executor(None, sys.stdin.readline)
            )
            shutdown_task = asyncio.ensure_future(runtime._shutdown_event.wait())
            done, pending = await asyncio.wait(
                [read_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            if shutdown_task in done:
                break
            line = read_task.result()
            if not line:
                break
            keep_going = await _handle_command(line, runtime)
            if not keep_going:
                break
        except EOFError:
            break


async def _load_agents(runtime: Runtime, config: ATNConfig) -> int:
    """Load agent definitions from YAML files and register them.

    Handles additions, updates, and removals compared to currently registered agents.
    Returns the number of agents loaded.
    """
    agents, errors = load_agents_dir(config.agents_dir)

    for err in errors:
        console.print(f"  [red]Load error: {err}[/]")

    # Determine what changed
    current_ids = {defn.id for defn, _ in runtime.list_agents()}
    new_ids = {a.id for a in agents}

    # Unregister agents whose files were removed (skip orchestrator)
    from .orchestrator import ORCHESTRATOR_ID
    for removed_id in current_ids - new_ids:
        if removed_id == ORCHESTRATOR_ID:
            continue
        try:
            await runtime.unregister_agent(removed_id)
            console.print(f"  [yellow]Removed agent: {removed_id}[/]")
        except ValueError:
            pass

    # Register new or update existing. legacy=True so on-disk agents from
    # before the mandatory-budget rule still hydrate; new creates via
    # create_agent / the frontend go through the strict path.
    for defn in agents:
        if defn.id in current_ids:
            await runtime.unregister_agent(defn.id)
        await runtime.register_agent(defn, legacy=True)

    if agents:
        console.print(f"  [green]Loaded {len(agents)} agent(s) from {config.agents_dir}[/]")
    elif not errors:
        console.print(f"  [dim]No agent files in {config.agents_dir}[/]")

    # Auto-activate agents that have schedules or heartbeats
    for defn in agents:
        if defn.schedule or defn.heartbeat:
            await runtime.activate_agent(defn.id)

    return len(agents)


def _try_reclaim_port(port: int) -> int | None:
    """If *port* is held by an ATN MCP server (our own child), kill it.

    Returns the PID that was killed, or None.
    """
    import psutil

    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.status == "LISTEN":
            try:
                proc = psutil.Process(conn.pid)
                cmdline = " ".join(proc.cmdline())
                if "atn.mcp_server" in cmdline or "atn/mcp_server" in cmdline:
                    proc.kill()
                    proc.wait(timeout=3)
                    return conn.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    return None


async def run_cli() -> None:
    """Main async entry point."""
    from .lock_manager import LockManager

    config = load_config()

    # Singleton lock — only one daemon per data_dir
    lock = LockManager()
    lock.set_data_dir(config.data_dir)

    existing = lock.is_daemon_running()
    if existing:
        console.print(
            f"[red]ATN daemon is already running![/]  "
            f"[dim]PID {existing['pid']}, started {existing.get('started_at', '?')}[/]"
        )
        sys.exit(1)

    if not lock.acquire_lock():
        console.print("[red]Failed to acquire daemon lock.[/]")
        sys.exit(1)

    # Warm heavy native imports (numpy / sounddevice via voice_service, and
    # the world_model embedding stack) on the MAIN thread before any
    # background asyncio executor thread can touch them. numpy's
    # _core.multiarray init is not safe to run first-time from a worker
    # thread while the main thread holds the import lock — doing so wedges
    # both threads permanently (observed as the WS initial-snapshot hang).
    # Importing here, single-threaded, makes every later import a cheap
    # sys.modules lookup.
    try:
        import numpy as _warm_numpy  # noqa: F401
    except Exception:
        pass

    event_bus = EventBus()
    runtime = Runtime(event_bus, data_dir=config.data_dir, config=config)

    # Subscribe the console printer to all events.
    event_bus.subscribe(None, _print_event)

    console.print(
        "\n[bold blue]ATN Runtime[/]  [dim]v0.1.0  |  Agent Orchestration Framework[/]\n"
    )
    console.print(f"  [dim]data_dir:   {config.data_dir}[/]")
    console.print(f"  [dim]agents_dir: {config.agents_dir}[/]")
    if config.orchestrator.provider:
        console.print(
            f"  [dim]orchestrator: provider={config.orchestrator.provider}"
            f"  model={config.orchestrator.model or '(provider default)'}[/]"
        )
    if config.providers:
        for name, prov in config.providers.items():
            key_hint = (prov.api_key[:4] + "...") if len(prov.api_key) > 4 else ("(not set)" if not prov.api_key else "***")
            console.print(f"  [dim]provider:   {name}  model={prov.default_model or '-'}  key={key_hint}[/]")

    await runtime.start()

    # Load agents from YAML files
    await _load_agents(runtime, config)

    # Register the orchestrator meta-agent
    try:
        await runtime.setup_orchestrator()
        console.print("  [green]Orchestrator registered[/]")
    except Exception as exc:
        console.print(f"  [yellow]Orchestrator not available: {exc}[/]")

    # Phase 12: if any loaded agent already has on-chain registration,
    # auto-start autonet — the user opted in earlier and the daemon
    # should reflect that on every boot, not require another click.
    try:
        started = await runtime.maybe_autostart_autonet_for_registered_agents()
        if started:
            console.print("  [green]Autonet auto-started for registered agent(s)[/]")
    except Exception as exc:
        console.print(f"  [yellow]Could not auto-start autonet: {exc}[/]")

    # Start WebSocket bridge so the Flutter frontend can connect.
    # If the port is held by a stale MCP server, attempt to reclaim it.
    ws_bridge: WebSocketBridge | None = None
    _an = getattr(runtime._config, "autonet", None)
    _remote_host = getattr(_an, "remote_ws_host", "") if _an else ""
    _remote_port = getattr(_an, "remote_ws_port", 0) if _an else 0
    _owner_wallet = getattr(_an, "owner_wallet", "") if _an else ""
    for _attempt in range(2):
        ws_bridge = WebSocketBridge(
            runtime, host="localhost", port=DEFAULT_PORT,
            remote_host=_remote_host, remote_port=_remote_port,
            owner_wallet=_owner_wallet,
        )
        try:
            await ws_bridge.start()
            # Register the bridge as a first-class Surface (Track 1): the
            # runtime now knows it as self.ws_bridge, wires its input-seam
            # policy, and includes it in active_surfaces(). Listeners + the
            # port-retry loop stay CLI-owned.
            runtime.register_surface(ws_bridge)
            console.print(f"  [green]WebSocket server listening on ws://localhost:{DEFAULT_PORT}[/]")
            if _remote_host:
                _modes = ("owner+agent-self" if _owner_wallet else "agent-self")
                console.print(
                    f"  [green]Remote (auth: {_modes}) WS on "
                    f"ws://{_remote_host}:{ws_bridge.remote_port}[/]")
            break
        except OSError as exc:
            ws_bridge = None
            if _attempt == 0:
                # Try to find and kill the process holding the port
                killed = _try_reclaim_port(DEFAULT_PORT)
                if killed:
                    console.print(f"  [yellow]Reclaimed port {DEFAULT_PORT} from stale process (PID {killed})[/]")
                    await asyncio.sleep(0.5)
                    continue
            console.print(
                f"  [red]WebSocket server failed to start: {exc}[/]\n"
                f"  [dim]Port {DEFAULT_PORT} is in use. Find the process: "
                f"netstat -ano | findstr {DEFAULT_PORT}[/]"
            )
            break

    # Auto-update: start the poll task (opt-in via config). It only *stages*
    # verified updates; the running daemon never restarts itself. Skipped
    # entirely on a developer's working tree / editable install — the running
    # code isn't the pip-installed package there.
    update_poll_task: asyncio.Task | None = None
    if getattr(config.auto_update, "enabled", False):
        try:
            from .update_boot import dev_environment_reason
            _dev_reason = dev_environment_reason()
        except Exception:
            _dev_reason = ""
        if _dev_reason:
            console.print(
                f"  [yellow]Auto-update: skipped[/] [dim]({_dev_reason} — "
                f"only pip-installed daemons auto-update)[/]"
            )
        else:
            update_poll_task = asyncio.create_task(_update_poll_loop(runtime, config))
            console.print(
                f"  [green]Auto-update enabled[/] [dim](source: "
                f"{config.auto_update.source}, every "
                f"{config.auto_update.check_interval_secs}s — staged, applied on restart)[/]"
            )

    _print_help()
    console.print("[dim]Events stream below.  Type commands at any time.\n[/]")

    try:
        await _input_loop(runtime)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[bold red]Shutting down...[/]")
        if update_poll_task is not None and not update_poll_task.done():
            update_poll_task.cancel()
            try:
                await update_poll_task
            except asyncio.CancelledError:
                pass
        if ws_bridge:
            await ws_bridge.stop()
        if not runtime._shutdown_event.is_set():
            await runtime.stop()
        lock.release_lock()
        console.print("[dim]Bye.[/]")


def _build_updater(runtime: Runtime, config: ATNConfig):
    """Construct an AutonetUpdater from the daemon's auto_update config.

    rpc_url / registry_address (for the advisory on-chain core-hash check)
    are sourced from the autonet/blockchain config when available; absent
    them, staging proceeds with wheel-hash verification only (logged).
    """
    from nodes.common.updater import AutonetUpdater
    from .runtime.snapshot import _daemon_version

    au = config.auto_update
    _an = getattr(config, "autonet", None)
    rpc_url = getattr(_an, "rpc_url", "") if _an else ""
    registry_addr = (
        getattr(_an, "registry_address", "")
        or getattr(_an, "rpb_contract_address", "")
        if _an else ""
    )
    return AutonetUpdater(
        current_version=_daemon_version(),
        source=au.source,
        package_name=au.package_name,
        pypi_index_url=au.pypi_index_url,
        data_dir=str(config.data_dir),
        check_interval=au.check_interval_secs,
        rpc_url=rpc_url or "",
        registry_address=registry_addr or "",
    )


async def _update_poll_loop(runtime: Runtime, config: ATNConfig) -> None:
    """Background poll: check the release source on a fixed interval, stage
    if a newer release is found.

    The interval is the *maximum* check frequency. The last-check time is
    persisted across restarts, so a daemon that's been down longer than the
    interval checks shortly after startup; one restarted within the interval
    waits out the remainder before its first check.

    Stage-only — never restarts the running daemon. Emits UPDATE_STATUS
    events on state transitions so the frontend can surface availability.
    Stored on ``runtime._updater`` for the ws ``update_status`` handler and
    the snapshot to read.
    """
    import asyncio as _asyncio

    # Never auto-update a developer's working tree / editable install — the
    # running code isn't the pip-installed package, so an upgrade would either
    # do nothing useful or clobber the dev's source. Same guard the boot-apply
    # path uses, applied here so a dev machine doesn't even download/stage.
    try:
        from .update_boot import dev_environment_reason
        reason = dev_environment_reason()
    except Exception:
        reason = ""
    if reason:
        log.info("Auto-update: disabled for this environment (%s)", reason)
        console.print(
            f"  [yellow]Auto-update: skipped[/] [dim]({reason} — "
            f"only pip-installed daemons auto-update)[/]"
        )
        return

    try:
        updater = _build_updater(runtime, config)
    except Exception as exc:
        log.warning("Auto-update: could not construct updater: %s", exc)
        return
    runtime._updater = updater  # type: ignore[attr-defined]

    interval = max(60, int(config.auto_update.check_interval_secs))
    last_state = ""

    # First check: if the persisted last-check is older than the interval
    # (or never happened), check soon after boot; otherwise wait out the
    # remaining time so we honor the interval as a max frequency. A small
    # floor keeps the check from competing with boot work.
    since = updater.seconds_since_last_check()
    if since >= interval:
        initial_wait = 15
    else:
        initial_wait = max(15, int(interval - since))
    log.info(
        "Auto-update: first check in %ds (%.0fs since last check, interval %ds)",
        initial_wait,
        since if since != float("inf") else -1,
        interval,
    )
    slept = 0
    while slept < initial_wait:
        try:
            await _asyncio.sleep(min(30, initial_wait - slept))
        except _asyncio.CancelledError:
            return
        slept += 30
    while True:
        try:
            info = await _asyncio.to_thread(updater.check_update)
            if info is not None and info.has_update:
                staged = await _asyncio.to_thread(updater.stage_update, info)
                if staged:
                    log.info(
                        "Auto-update: staged %s (applies on next restart)",
                        info.available_version,
                    )
            # Emit a status event only when the state changed.
            status = updater.get_status()
            if status.get("state") != last_state:
                last_state = status.get("state", "")
                try:
                    await runtime.events.emit(Event(
                        type=EventType.UPDATE_STATUS,
                        source="updater",
                        data=status,
                    ))
                except Exception:
                    pass
        except _asyncio.CancelledError:
            break
        except Exception as exc:
            log.warning("Auto-update poll error: %s", exc)
        # Sleep in short slices so cancellation is responsive.
        slept = 0
        while slept < interval:
            try:
                await _asyncio.sleep(min(60, interval - slept))
            except _asyncio.CancelledError:
                return
            slept += 60


def main() -> None:
    # Auto-update: apply any staged release BEFORE importing heavy runtime
    # modules, so a pip-install can cleanly replace them, then re-exec once.
    # Import-light and self-contained (stdlib only); no-ops when nothing is
    # staged or auto-update is off. See atn/update_boot.py.
    try:
        from .update_boot import apply_staged_update_if_any
        apply_staged_update_if_any()
    except Exception:
        # Never let the updater block the daemon from starting.
        pass

    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        pass
    if _restart_requested:
        console.print("[bold green]Re-launching daemon...[/]")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.execv(sys.executable, [sys.executable, *sys.argv])
    # Force-exit to avoid hanging on lingering non-daemon threads or asyncio
    # transports during interpreter shutdown (numpy/scipy Fortran runtime,
    # libp2p protobuf threads, etc.). All persistent state has already been
    # flushed by runtime.stop() / lock.release_lock().
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)
