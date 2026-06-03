"""End-to-end self-test of the ChatService binding against the stub adapter.

No daemon, no Discord, no real runtime. Drives the ChatService with real
EventBus Event objects (the same shapes the runtime emits) plus a minimal fake
runtime that records send_agent_message calls, then asserts the stub transcript
reflects the design:

  - orchestrator output renders in the bound channel
  - each top-level delegate (depth 1) gets its own thread
  - nested sub-agents (depth >= 2) render as pinned tiles in that thread
  - a non-dotted create_agent agent (agent.registered with parent_id) gets a
    thread (the create_agent case, not just dotted delegates)
  - replying to a tile resolves to the right sub-agent
  - the operator gate allows the operator, declines others, and a relayed
    message reaches runtime.send_agent_message

Run:  python -m atn.chat._selftest      (from C:\\code\\autonet)
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ..events import Event, EventBus, EventType
from .adapters.stub import StubAdapter
from .chat_service import ChatService
from .policy import OperatorGate
from .protocol import Author, Channel, ChannelKind, Message

CHANNEL = "chan-dorg"
OPERATOR = "938049028757807135"


class _FakeRuntime:
    """Minimal runtime stand-in: records send_agent_message calls."""

    def __init__(self) -> None:
        self.relayed: list[tuple[str, str]] = []
        self.registry = None  # no seed

    async def send_agent_message(self, agent_id: str, content: str) -> bool:
        self.relayed.append((agent_id, content))
        return True


def _evt(t: EventType, source: str, **data) -> Event:
    return Event(type=t, source=source, data=data)


def _step(source: str, channel: str, content: str) -> Event:
    return _evt(EventType.STEP_OUTPUT, source,
                agent_id=source, channel=channel, content=content)


def _spawned(agent_id: str, parent_id: str, title: str) -> Event:
    return _evt(EventType.DELEGATE_SPAWNED, agent_id,
                agent_id=agent_id, parent_id=parent_id, title=title,
                status="running")


def _registered(agent_id: str, parent_id: str, name: str) -> Event:
    return _evt(EventType.AGENT_REGISTERED, "runtime",
                agent_id=agent_id, parent_id=parent_id, name=name)


def _inbound(content, channel_id, author_id, reply_to_id=None, parent_id=None) -> Message:
    kind = ChannelKind.THREAD if parent_id else ChannelKind.TEXT
    return Message(
        id="in-" + author_id,
        channel=Channel(id=channel_id, kind=kind, parent_id=parent_id),
        author=Author(id=author_id, display_name=author_id),
        content=content,
        ts=datetime.now(timezone.utc),
        reply_to_id=reply_to_id,
    )


async def main() -> int:
    bus = EventBus()
    rt = _FakeRuntime()
    adapter = StubAdapter()
    svc = ChatService(bus, rt, adapter, CHANNEL, policy=OperatorGate({OPERATOR}))
    await svc.start()

    async def emit(e: Event) -> None:
        # Deliver straight to the service's handler (EventBus.emit also works,
        # but calling the handler keeps the test deterministic/ordered).
        await svc._on_bus_event(e)

    # --- OUTPUT PATH ---
    await emit(_evt(EventType.EXECUTION_STARTED, "orchestrator", agent_id="orchestrator"))
    await emit(_step("orchestrator", "text", "On it — spinning up a researcher."))
    await emit(_spawned("orchestrator.1", "orchestrator", "Research competitors"))
    await emit(_evt(EventType.EXECUTION_STARTED, "orchestrator.1", agent_id="orchestrator.1"))
    await emit(_evt(EventType.AGENT_TOOL_USE_START, "orchestrator.1", tool_name="mcp__atn__web_search"))
    await emit(_step("orchestrator.1", "text", "Found three competitors."))
    await emit(_spawned("orchestrator.1.1", "orchestrator.1", "Summarize Acme"))
    await emit(_evt(EventType.EXECUTION_STARTED, "orchestrator.1.1", agent_id="orchestrator.1.1"))
    await emit(_step("orchestrator.1.1", "text", "Acme charges per-seat."))
    await emit(_evt(EventType.EXECUTION_COMPLETED, "orchestrator.1.1", agent_id="orchestrator.1.1"))
    await emit(_evt(EventType.EXECUTION_COMPLETED, "orchestrator.1", agent_id="orchestrator.1"))
    await emit(_evt(EventType.EXECUTION_COMPLETED, "orchestrator", agent_id="orchestrator"))

    print("=== STUB TRANSCRIPT ===")
    print(adapter.transcript())
    print("=======================\n")

    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    check("top-level delegate has a thread", "orchestrator.1" in svc._threads)
    check("no thread for nested orchestrator.1.1", "orchestrator.1.1" not in svc._threads)
    thread_id = svc._threads.get("orchestrator.1")
    tile = svc._tiles.get("orchestrator.1.1")
    check("nested agent has a tile", tile is not None)
    if tile and tile.message_id:
        m = adapter.store.messages.get(tile.message_id)
        check("tile lives in the delegate's thread", m and m["channel_id"] == thread_id)
        check("completed tile is unpinned", m and not m["pinned"])
    check("orchestrator rendered in the channel",
          any(m["channel_id"] == CHANNEL for m in adapter.store.messages.values()))

    # --- create_agent (non-dotted) path ---
    await emit(_registered("test-simple", "orchestrator", "Simple test agent"))
    await emit(_evt(EventType.EXECUTION_STARTED, "test-simple", agent_id="test-simple"))
    await emit(_step("test-simple", "text", "Simple test agent reporting in."))
    await emit(_evt(EventType.EXECUTION_COMPLETED, "test-simple", agent_id="test-simple"))
    check("non-dotted create_agent gets a thread", "test-simple" in svc._threads)
    ts_thread = svc._threads.get("test-simple")
    check("test-simple output renders in its own thread",
          any(m["channel_id"] == ts_thread and "reporting in" in (m.get("content") or "")
              for m in adapter.store.messages.values()))

    # --- INPUT PATH: addressing + policy + relay ---
    check("channel -> orchestrator",
          svc.resolve_target(_inbound("status?", CHANNEL, OPERATOR)) == "orchestrator")
    check("thread -> delegate",
          svc.resolve_target(_inbound("x", thread_id, OPERATOR, parent_id=CHANNEL)) == "orchestrator.1")
    if tile and tile.message_id:
        check("reply-to-tile -> sub-agent",
              svc.resolve_target(_inbound("x", thread_id, OPERATOR,
                                          reply_to_id=tile.message_id, parent_id=CHANNEL)) == "orchestrator.1.1")

    # operator relays through to runtime.send_agent_message
    relayed = await svc.handle_inbound(_inbound("do the thing", CHANNEL, OPERATOR))
    check("operator message relayed to runtime", relayed and rt.relayed[-1][0] == "orchestrator")
    # non-operator declined, not relayed
    before = len(rt.relayed)
    declined = await svc.handle_inbound(_inbound("hi", CHANNEL, "rando"))
    check("non-operator declined (not relayed)", (not declined) and len(rt.relayed) == before)

    await svc.stop()
    print("\n" + ("ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
