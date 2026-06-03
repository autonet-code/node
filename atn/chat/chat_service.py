"""ChatService -- the chat Surface: binds agent conversations to a chat platform.

A Surface (see atn/surface.py) is a long-lived bidirectional bridge from an
external real-time channel to agent sessions — the counterpart to a Connector
(outbound tools an agent uses). ChatService is the chat Surface; VoiceService
is the voice one, the same shape. Mounted at the EventBus seam: subscribe for
OUTPUT, call ``runtime.send_agent_message`` for INPUT (gated at the input seam
by an InputPolicy), keep a routing table binding chat surfaces to agents.

Interaction model, one dedicated channel:

  - The CHANNEL is bound to the orchestrator. Unprefixed channel messages go
    to the orchestrator; orchestrator output renders in the channel.
  - Each TOP-LEVEL delegate (depth 1) gets its own THREAD, created when its
    delegate.spawned / agent.registered event fires. The thread binds to it.
  - NESTED sub-agents (depth >= 2) render as pinned, edit-in-place TILES inside
    their top-level delegate's thread. Replying to a tile addresses that
    sub-agent.

Platform-neutral: it drives a ``MessagingClient`` and knows nothing about
discord.py. The platform binding (which channel is THE channel) is supplied at
construction.

DОrg-specific application logic (member-agent registration, credit metering)
is NOT here -- it lives at the input seam as the ``InputPolicy`` supplied by
the deployment.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from ..events import Event, EventBus, EventType
from .policy import AllowAll, InputPolicy
from .rendering import AgentExecutionRender, SubAgentTile
from .routing import (
    ORCHESTRATOR, agent_depth, is_top_level_delegate, top_level_delegate,
)

if TYPE_CHECKING:
    from .protocol import ChannelId, MessagingClient, Message

log = logging.getLogger("atn.chat")

# Agents never rendered (background/infra whose routine executions would spam
# the channel). Empty by default; the deployment names them via config.
_DEFAULT_EXCLUDED_AGENTS: set[str] = set()

# EventTypes that drive the chat surface. Subscribed in start().
_SUBSCRIBED = (
    EventType.AGENT_REGISTERED,
    EventType.AGENT_UNREGISTERED,
    EventType.DELEGATE_SPAWNED,
    EventType.DELEGATE_COMPLETED,
    EventType.DELEGATE_FAILED,
    EventType.EXECUTION_STARTED,
    EventType.EXECUTION_COMPLETED,
    EventType.EXECUTION_FAILED,
    EventType.STEP_OUTPUT,
    EventType.AGENT_TOOL_USE_START,
)


class ChatService:
    """Binds the agent fleet to a chat platform, at the EventBus seam.

    Parameters
    ----------
    event_bus:
        The runtime's EventBus (agent output flows through here).
    runtime:
        The autonet runtime (input goes via runtime.send_agent_message).
    client:
        The platform adapter (Discord, or the stub) implementing MessagingClient.
    channel_id:
        THE channel -- bound to the orchestrator. All orchestrator I/O happens
        here; delegate threads are created under it.
    policy:
        Input-seam policy (operator gate, credits, ...). Defaults to AllowAll.
    orchestrator_label:
        Display name for the orchestrator (dОrg presents it as "K3V|N").
    excluded_agents:
        Agent ids never rendered (background/infra). Empty by default.
    """

    def __init__(
        self,
        event_bus: EventBus,
        runtime: Any,
        client: "MessagingClient",
        channel_id: "ChannelId",
        *,
        bound_agent: str = ORCHESTRATOR,
        policy: InputPolicy | None = None,
        orchestrator_label: str = "K3V|N",
        excluded_agents: set[str] | None = None,
    ) -> None:
        self.events = event_bus
        self.runtime = runtime
        self.client = client
        self.channel_id = channel_id
        # The agent THIS channel is bound to — its conversation IS the channel.
        # Usually the orchestrator (fleet control), but can be any agent
        # (e.g. a support agent on a support channel). Treated as the channel's
        # root for routing: its output renders in the channel; the agents IT
        # spawns get threads. Defaults to the orchestrator.
        self.bound_agent = bound_agent
        self.policy = policy or AllowAll()
        self.orchestrator_label = orchestrator_label
        self._excluded = set(excluded_agents) if excluded_agents is not None \
            else set(_DEFAULT_EXCLUDED_AGENTS)

        # Routing / binding state.
        # agent_id -> the channel/thread id its output renders into.
        self._render_channel: dict[str, "ChannelId"] = {bound_agent: channel_id}
        # top-level delegate agent_id -> thread id (created lazily).
        self._threads: dict[str, "ChannelId"] = {}
        # in-flight thread creations (so concurrent events don't double-create).
        self._thread_locks: dict[str, asyncio.Lock] = {}
        # agent_id -> current execution render (orchestrator + top-level delegates).
        self._renders: dict[str, AgentExecutionRender] = {}
        # nested agent_id -> its pinned tile.
        self._tiles: dict[str, SubAgentTile] = {}
        # tile message_id -> agent_id (reverse map for reply-addressing).
        self._tile_msg_to_agent: dict[str, str] = {}
        # spawn/registration titles, for nicer labels.
        self._titles: dict[str, str] = {}
        # agent_id -> resolved parent_id. Seeded from the registry at start and
        # from agent.registered. Lets routing work for create_agent agents whose
        # ids are NOT dotted lineage (e.g. "test-simple"); the routing helpers
        # walk this instead of string-parsing the id.
        # The bound agent is the channel's root for routing (parent = "" =>
        # depth 0 => renders in the channel; its children get threads), even
        # if its real lineage has a parent (e.g. support's parent is the
        # orchestrator). Always include the orchestrator as a root too.
        self._parent: dict[str, str] = {ORCHESTRATOR: "", bound_agent: ""}
        # execution start times for duration footers.
        self._started_at: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._running = False

    def _parent_of(self, agent_id: str) -> str | None:
        """Resolver for the routing helpers: agent_id -> parent_id, or None if
        we don't know the lineage (helpers then fall back to dotted parsing)."""
        return self._parent.get(agent_id)

    def _in_scope(self, agent_id: str) -> bool:
        """True if this agent renders on THIS channel — i.e. it is the bound
        agent or a (transitive) descendant of it. A channel bound to `support`
        shows support + its sub-agents only, not the orchestrator's whole fleet.
        Walks the parent map up to a root; falls back to dotted-id descent."""
        if agent_id == self.bound_agent:
            return True
        cur = agent_id
        for _ in range(64):
            p = self._parent.get(cur)
            if p == self.bound_agent:
                return True
            if p is None or p == "" or p == cur:
                break
            cur = p
        # Dotted fallback (e.g. bound to orchestrator: orchestrator.1.2 descends).
        return agent_id.startswith(self.bound_agent + ".")

    def _forget_agent(self, agent_id: str) -> None:
        """Drop all state for an agent that was unregistered, so a long-running
        session doesn't accumulate stale lineage/threads/tiles. We leave the
        platform thread/tile messages in place (history is fine to keep); we
        just stop tracking the agent so its id can be cleanly reused."""
        if not agent_id:
            return
        self._parent.pop(agent_id, None)
        self._titles.pop(agent_id, None)
        self._threads.pop(agent_id, None)
        self._render_channel.pop(agent_id, None)
        self._renders.pop(agent_id, None)
        self._started_at.pop(agent_id, None)
        tile = self._tiles.pop(agent_id, None)
        if tile is not None and tile.message_id:
            self._tile_msg_to_agent.pop(tile.message_id, None)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._seed_from_registry()
        for et in _SUBSCRIBED:
            self.events.subscribe(et, self._on_bus_event)
        self._running = True
        log.info("ChatService started; channel=%s bound to orchestrator", self.channel_id)

    async def stop(self) -> None:
        if not self._running:
            return
        for et in _SUBSCRIBED:
            try:
                self.events.unsubscribe(et, self._on_bus_event)
            except Exception:
                pass
        self._running = False

    def _seed_from_registry(self) -> None:
        """Learn current lineage/titles from the runtime registry so routing
        works for agents that already exist (replaces the WS connect-snapshot).
        Best-effort: tolerate whatever shape the registry exposes."""
        reg = getattr(self.runtime, "registry", None)
        if reg is None:
            return
        try:
            defs = getattr(reg, "agents", None)
            items = defs.values() if isinstance(defs, dict) else (defs or [])
            for d in items:
                aid = getattr(d, "id", None) or (d.get("id") if isinstance(d, dict) else None)
                if not aid:
                    continue
                pid = getattr(d, "parent_id", None) if not isinstance(d, dict) else d.get("parent_id")
                name = getattr(d, "name", None) if not isinstance(d, dict) else d.get("name")
                if pid is not None and aid not in self._parent:
                    self._parent[aid] = pid
                if name:
                    self._titles.setdefault(aid, name)
        except Exception:
            log.debug("registry seed failed (non-fatal)", exc_info=True)

    # -- output path (EventBus -> platform) ----------------------------------

    async def _on_bus_event(self, event: Event) -> None:
        """Single EventBus handler. Normalizes the Event into the routing the
        rest of the service expects (etype string, source, data dict)."""
        etype = event.type.value if hasattr(event.type, "value") else str(event.type)
        agent_id = event.source or ""
        data = event.data or {}

        # agent.registered is sourced from "runtime"; the real agent id and its
        # parent_id are in the payload. Learn lineage and (for a top-level
        # agent) open its thread eagerly, before the first execution event.
        if etype == EventType.AGENT_REGISTERED.value:
            await self._on_agent_registered(data)
            return
        if etype == EventType.AGENT_UNREGISTERED.value:
            self._forget_agent(data.get("agent_id", ""))
            return

        if not agent_id or agent_id in self._excluded:
            return

        if etype == EventType.DELEGATE_SPAWNED.value:
            await self._on_delegate_spawned(data)
            return
        if etype in (EventType.DELEGATE_COMPLETED.value, EventType.DELEGATE_FAILED.value):
            # Sub-agent lifecycle is also covered by execution.* for the agent
            # itself; we use delegate.* to set the tile's final status reliably.
            sub = data.get("agent_id", agent_id)
            tile = self._tiles.get(sub)
            if tile is not None:
                await tile.finalize(
                    "completed" if etype == EventType.DELEGATE_COMPLETED.value else "failed")
            return

        # Only render agents on THIS channel's subtree (bound agent + its
        # descendants). A support channel ignores the orchestrator's fleet.
        if not self._in_scope(agent_id):
            return

        async with self._lock:
            if etype == EventType.EXECUTION_STARTED.value:
                await self._start_execution(agent_id)
            elif etype == EventType.STEP_OUTPUT.value:
                await self._on_step_output(agent_id, data)
            elif etype == EventType.AGENT_TOOL_USE_START.value:
                await self._on_tool(agent_id, data.get("tool_name") or "")
            elif etype in (EventType.EXECUTION_COMPLETED.value, EventType.EXECUTION_FAILED.value):
                await self._finalize_execution(
                    agent_id,
                    "completed" if etype == EventType.EXECUTION_COMPLETED.value else "failed")

    async def _on_agent_registered(self, data: dict) -> None:
        aid = data.get("agent_id", "")
        if not aid or aid in self._excluded or aid == ORCHESTRATOR:
            return
        # Always learn the lineage, even for out-of-scope agents — it keeps the
        # parent map complete so scope/depth checks for in-scope descendants
        # resolve correctly.
        pid = data.get("parent_id")
        if pid is not None and aid != self.bound_agent:
            self._parent[aid] = pid
        name = data.get("name") or data.get("title")
        if name:
            self._titles.setdefault(aid, name)
        await self._render_new_agent(aid)

    async def _on_delegate_spawned(self, data: dict) -> None:
        aid = data.get("agent_id", "")
        if not aid or aid in self._excluded:
            return
        pid = data.get("parent_id")
        if pid is not None and aid != self.bound_agent:
            self._parent[aid] = pid
        self._titles[aid] = data.get("title") or aid
        await self._render_new_agent(aid)

    async def _render_new_agent(self, aid: str) -> None:
        """Open the render target for a newly-seen agent, if it's on this
        channel's subtree. The bound agent renders in the channel (no thread);
        a depth-1 descendant gets a thread; deeper gets a tile."""
        if not self._in_scope(aid) or aid == self.bound_agent:
            return
        if is_top_level_delegate(aid, self._parent_of):
            await self._ensure_thread(aid)
        elif agent_depth(aid, self._parent_of) >= 2:
            await self._ensure_tile(aid)

    # -- render-target resolution -------------------------------------------

    async def _ensure_thread(self, delegate_id: str) -> "ChannelId | None":
        """Create (once) and return the thread for a top-level delegate."""
        if delegate_id in self._threads:
            return self._threads[delegate_id]
        lock = self._thread_locks.setdefault(delegate_id, asyncio.Lock())
        async with lock:
            if delegate_id in self._threads:  # re-check under lock
                return self._threads[delegate_id]
            title = self._titles.get(delegate_id, delegate_id)
            name = f"🤖 {title}"[:90]
            try:
                # Anchor the thread on a small announcement message in the channel.
                anchor = await self.client.send(
                    self.channel_id, content=f"🧵 **{title}** · `{delegate_id}`")
                post = await self.client.create_thread(
                    anchor.message_id, self.channel_id, name=name)
                thread_id = post.thread_id or post.channel_id
            except Exception:
                log.exception("thread create failed for %s", delegate_id)
                return None
            self._threads[delegate_id] = thread_id
            self._render_channel[delegate_id] = thread_id
            log.info("thread %s bound to delegate %s", thread_id, delegate_id)
            return thread_id

    async def _ensure_tile(self, sub_id: str) -> SubAgentTile | None:
        """Create (once) the pinned tile for a nested sub-agent."""
        if sub_id in self._tiles:
            return self._tiles[sub_id]
        top = top_level_delegate(sub_id, self._parent_of)
        if top is None:
            return None
        thread_id = await self._ensure_thread(top)
        if thread_id is None:
            return None
        tile = SubAgentTile(
            client=self.client,
            thread_id=thread_id,
            agent_id=sub_id,
            title=self._titles.get(sub_id, ""),
        )
        await tile.start()
        self._tiles[sub_id] = tile
        if tile.message_id:
            self._tile_msg_to_agent[tile.message_id] = sub_id
        return tile

    async def _render_for(self, agent_id: str) -> "ChannelId | None":
        """The channel/thread an agent's execution render writes into.

        - orchestrator -> the bound channel
        - top-level delegate -> its thread (created if needed)
        - nested -> handled as tiles, not execution renders (returns None)
        """
        depth = agent_depth(agent_id, self._parent_of)
        if depth == 0:
            return self.channel_id
        if depth == 1:
            return await self._ensure_thread(agent_id)
        return None  # nested: rendered via tiles

    # -- execution rendering -------------------------------------------------

    async def _start_execution(self, agent_id: str) -> None:
        self._started_at[agent_id] = time.monotonic()
        depth = agent_depth(agent_id, self._parent_of)
        if depth >= 2:
            # Nested: ensure a tile exists; it streams via step.output.
            await self._ensure_tile(agent_id)
            return
        # Finalize any prior render for this agent first.
        prev = self._renders.get(agent_id)
        if prev is not None:
            await prev.finalize()
        channel = await self._render_for(agent_id)
        if channel is None:
            return
        # Relabel only the real orchestrator (presentation: "K3V|N"). Any other
        # bound agent (e.g. support) keeps its own name.
        label = self.orchestrator_label if agent_id == ORCHESTRATOR else \
            self._titles.get(agent_id, agent_id)
        render = AgentExecutionRender(
            client=self.client, channel_id=channel, agent_id=agent_id, label=label)
        await render.start()
        self._renders[agent_id] = render

    async def _on_step_output(self, agent_id: str, data: dict) -> None:
        channel = data.get("channel")
        if channel == "usage":
            r = self._renders.get(agent_id)
            if r is not None:
                # Pass the full split so the render meters REAL SPEND
                # (cache_creation + output + $), not context size.
                r.on_usage(data.get("usage") or {}, data.get("cumulative") or {})
            return
        content = data.get("content") or ""
        if not content:
            return
        if agent_depth(agent_id, self._parent_of) >= 2:
            tile = await self._ensure_tile(agent_id)
            if tile is None:
                return
            if channel == "text":
                await tile.on_text(content)
            elif channel == "thinking":
                await tile.on_thinking(content)
            return
        r = self._renders.get(agent_id)
        if r is None:
            await self._start_execution(agent_id)
            r = self._renders.get(agent_id)
        if r is None:
            return
        if channel == "text":
            await r.on_text(content)
        elif channel == "thinking":
            await r.on_thinking(content)

    async def _on_tool(self, agent_id: str, tool_name: str) -> None:
        if agent_depth(agent_id, self._parent_of) >= 2:
            tile = await self._ensure_tile(agent_id)
            if tile is not None:
                await tile.on_tool(tool_name)
            return
        r = self._renders.get(agent_id)
        if r is None:
            await self._start_execution(agent_id)
            r = self._renders.get(agent_id)
        if r is not None:
            await r.on_tool(tool_name)

    async def _finalize_execution(self, agent_id: str, status: str) -> None:
        elapsed = None
        t0 = self._started_at.pop(agent_id, None)
        if t0 is not None:
            elapsed = time.monotonic() - t0
        if agent_depth(agent_id, self._parent_of) >= 2:
            tile = self._tiles.get(agent_id)
            if tile is not None:
                await tile.finalize(status)
            return
        r = self._renders.pop(agent_id, None)
        if r is not None:
            await r.finalize(status, elapsed_s=elapsed)

    # -- input path (platform -> agent) --------------------------------------

    def resolve_target(self, message: "Message") -> str | None:
        """Map an inbound message to the agent it should be delivered to.

        - Reply to a sub-agent tile -> that sub-agent.
        - Message in a delegate thread -> that delegate.
        - Message in the bound channel -> the bound agent.
        - Anything else -> None (not for us).
        """
        # Reply to a pinned tile addresses that sub-agent.
        if message.reply_to_id and message.reply_to_id in self._tile_msg_to_agent:
            return self._tile_msg_to_agent[message.reply_to_id]
        ch_id = message.channel.id
        if ch_id == self.channel_id:
            return self.bound_agent
        # A thread bound to a top-level delegate.
        for delegate_id, thread_id in self._threads.items():
            if thread_id == ch_id:
                return delegate_id
        return None

    async def _maybe_handle_command(self, message: "Message", target: str) -> bool:
        """Handle `kk …` surface commands. Returns True if handled (so the
        message is not relayed to the agent). Commands are about the surface,
        not prompts for the agent."""
        text = (message.content or "").strip()
        # Strip a leading bot @mention so "@K3V|N kk context" works too.
        low = text.lower()
        if "kk context" not in low and not low.startswith("kk "):
            return False
        if "context" in low:
            stats = {}
            try:
                stats = self.runtime.get_session_stats(target) or {}
            except Exception:
                log.debug("get_session_stats failed for %s", target, exc_info=True)
            pct = stats.get("context_used_pct")
            turns = stats.get("num_turns")
            ctx = stats.get("context_window")
            cost = stats.get("total_cost_usd")
            parts = []
            if pct is not None:
                parts.append(f"context {pct:.0f}% used" + (f" of {ctx:,}" if ctx else ""))
            if turns is not None:
                parts.append(f"{turns} turns")
            if isinstance(cost, (int, float)) and cost > 0:
                parts.append(f"${cost:.4f} so far")
            reply = f"📊 `{target}`: " + (" · ".join(parts) if parts else "no session yet")
            try:
                await self.client.reply(message.id, message.channel.id, content=reply)
            except Exception:
                log.debug("kk context reply failed", exc_info=True)
            return True
        return False

    async def handle_inbound(self, message: "Message") -> bool:
        """Route an inbound platform message into the right agent session.

        Returns True if the message was relayed. Applies the input policy at
        this seam (the dОrg delta -- access/credits/consensus -- lives here)."""
        if message.author.is_bot:
            return False
        target = self.resolve_target(message)
        if target is None:
            return False
        # Surface commands (kk …) are handled here, not relayed to the agent and
        # not credit-gated — they're meta, about the surface itself.
        if await self._maybe_handle_command(message, target):
            return True
        decision = self.policy.evaluate(message.author, target, message.content)
        if not decision.allow:
            if decision.reason:
                try:
                    await self.client.reply(
                        message.id, message.channel.id,
                        content=decision.reason, mention_author=True)
                except Exception:
                    log.debug("policy-deny reply failed", exc_info=True)
            return False
        # Stamp the sender as a TAGGABLE id, not a name. The agent only ever
        # learns people as ids it can ping — so any reference it makes is a tag
        # (a notification), never talking about someone behind their back.
        sender_id = str(getattr(message.author, "id", "") or "")
        stamped = f"[from <@{sender_id}>]: {message.content}" if sender_id else message.content
        ok = await self.runtime.send_agent_message(target, stamped)
        if ok:
            log.info("relayed inbound -> %s (%d chars)", target, len(message.content))
        return ok
