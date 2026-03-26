"""CLI entry point — event stream + interactive commands.

Starts the Runtime and renders events to the terminal in real time.
Accepts commands on stdin for agent management and kill switches.

Usage:
    python -m atn
    atn              (if installed via pip)
"""
from __future__ import annotations

import asyncio
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
        "  [cyan]history[/] <id>        Show execution history for an agent\n"
        "  [cyan]logs[/]                Show on-disk execution log summary\n"
        "  [cyan]clear[/] <id>          Clear execution history for an agent\n"
        "  [cyan]clearall[/]            Clear all execution history\n"
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

    # Register new or update existing
    for defn in agents:
        if defn.id in current_ids:
            # Re-register: unregister old, register new
            await runtime.unregister_agent(defn.id)
        await runtime.register_agent(defn)

    if agents:
        console.print(f"  [green]Loaded {len(agents)} agent(s) from {config.agents_dir}[/]")
    elif not errors:
        console.print(f"  [dim]No agent files in {config.agents_dir}[/]")

    # Auto-activate agents that have schedules or heartbeats
    for defn in agents:
        if defn.schedule or defn.heartbeat:
            await runtime.activate_agent(defn.id)

    return len(agents)


async def run_cli() -> None:
    """Main async entry point."""
    from .ws_server import _acquire_lock

    config = load_config()

    # Singleton lock — only one daemon per data_dir / port
    _acquire_lock(config.data_dir, port=DEFAULT_PORT)

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

    # Start WebSocket bridge so the Flutter frontend can connect
    ws_bridge = WebSocketBridge(runtime, host="localhost", port=DEFAULT_PORT)
    try:
        await ws_bridge.start()
        console.print(f"  [green]WebSocket server listening on ws://localhost:{DEFAULT_PORT}[/]")
    except OSError as exc:
        console.print(f"  [red]WebSocket server failed to start: {exc}[/]")
        ws_bridge = None

    _print_help()
    console.print("[dim]Events stream below.  Type commands at any time.\n[/]")

    try:
        await _input_loop(runtime)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[bold red]Shutting down...[/]")
        if ws_bridge:
            await ws_bridge.stop()
        if not runtime._shutdown_event.is_set():
            await runtime.stop()
        console.print("[dim]Bye.[/]")


def main() -> None:
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        pass
