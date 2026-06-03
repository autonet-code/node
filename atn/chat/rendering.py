"""Rendering: turn an agent's event stream into platform messages.

Platform-neutral. Everything here talks to the `messaging.MessagingClient`
protocol, never to discord.py. The ChatService owns routing/lifecycle and
drives these renderers.

Two render shapes, both edit-in-place (so we never spam):

  - `AgentExecutionRender` -- one agent EXECUTION rendered as a single
    edit-in-place status message in a channel/thread. Used for the
    orchestrator (in the channel) and each top-level delegate (in its
    thread). Mirrors the AgentExecutionEmbed in fleet_view.py, but via the
    MessagingClient + StatusEmbedHandle protocol.

  - `SubAgentTile` -- a nested sub-agent (depth >= 2) rendered as a single
    PINNED message inside its top-level delegate's thread, showing a rolling
    window of its latest output, edited in place, lineage in the title.
    Replying to this message addresses the sub-agent.

Token/usage and tool-name formatting match the fleet_view conventions so the
two surfaces read the same.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import (
        ChannelId, EmbedSpec, MessagingClient, PostResult, StatusEmbedHandle,
    )

log = logging.getLogger("hackathon.chat.rendering")

# Embed/body limits (Discord embed description is 4096; stay well under).
_BODY_LIMIT = 3800
# Rolling window for nested sub-agent tiles (~200 words; chars as proxy).
_TILE_WINDOW = 1400

# Status colors (shared with fleet_view).
_COLOR_LIVE = 0x5865F2     # blurple -- in progress
_COLOR_DONE = 0x57F287     # green   -- completed
_COLOR_FAIL = 0xED4245     # red     -- failed
_COLOR_TILE = 0x4E5058     # grey    -- nested sub-agent tile (live)


def clean_tool_name(raw: str) -> str:
    """mcp__atn__get_snapshot -> get_snapshot."""
    return str(raw or "working").rsplit("__", 1)[-1]


def _truncate_tail(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return "…" + text[-limit:]
    return text


@dataclass
class AgentExecutionRender:
    """One execution of one agent, rendered as an edit-in-place message.

    Lives in a channel (orchestrator) or a thread (top-level delegate). The
    ChatService creates one per execution.started and finalizes it on
    execution.completed/failed.
    """

    client: "MessagingClient"
    channel_id: "ChannelId"
    agent_id: str
    label: str = ""                       # display label (defaults to agent_id)
    _handle: "StatusEmbedHandle | None" = field(default=None, init=False)
    _answer: str = field(default="", init=False)
    _thinking: str = field(default="", init=False)
    _tools: list[str] = field(default_factory=list, init=False)
    _current_tool: str | None = field(default=None, init=False)
    _tokens: int = field(default=0, init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.agent_id

    async def start(self) -> None:
        self._handle = self.client.make_status_handle(self.channel_id)
        try:
            await self._handle.send_initial()
        except Exception:
            log.exception("render start failed (%s)", self.agent_id)

    async def on_tool(self, tool_name: str) -> None:
        if self._handle is None or self._finalized:
            return
        name = clean_tool_name(tool_name)
        self._current_tool = name
        if name not in self._tools:
            self._tools.append(name)
        try:
            await self._handle.set_tool(name)
        except Exception:
            log.debug("render set_tool failed (%s)", self.agent_id, exc_info=True)

    async def on_text(self, delta: str) -> None:
        if not delta or self._handle is None or self._finalized:
            return
        self._answer += delta
        if self._current_tool is not None:
            self._current_tool = None
            try:
                await self._handle.clear_tool()
            except Exception:
                pass
        # The status handle streams thinking; the final answer lands at
        # finalize(). To keep the live view useful we surface the answer-so-far
        # as thinking-channel content (adapter renders it as the live body).
        try:
            await self._handle.append_thinking(delta)
        except Exception:
            log.debug("render on_text failed (%s)", self.agent_id, exc_info=True)

    async def on_thinking(self, delta: str) -> None:
        if not delta or self._handle is None or self._finalized:
            return
        # Only surface thinking before any answer text has arrived.
        if self._answer:
            return
        self._thinking = (self._thinking + delta)[-600:]
        try:
            await self._handle.append_thinking(delta)
        except Exception:
            log.debug("render on_thinking failed (%s)", self.agent_id, exc_info=True)

    def on_tokens(self, n: int) -> None:
        if n:
            self._tokens += n

    async def finalize(self, status: str = "completed", elapsed_s: float | None = None) -> None:
        if self._finalized or self._handle is None:
            self._finalized = True
            return
        self._finalized = True
        self._current_tool = None
        body = _truncate_tail(self._answer, _BODY_LIMIT) or "*(no text output)*"
        bits: list[str] = []
        if self._tools:
            bits.append("used " + ", ".join(self._tools))
        if elapsed_s is not None:
            bits.append(f"{elapsed_s:.0f}s")
        if self._tokens:
            bits.append(f"{self._tokens:,} tok")
        footer = ("-# " + " · ".join(bits)) if bits else ""
        final_text = f"{body}\n\n{footer}" if footer else body
        try:
            await self._handle.finalize(final_text)
        except Exception:
            log.debug("render finalize failed (%s)", self.agent_id, exc_info=True)


@dataclass
class SubAgentTile:
    """A nested sub-agent (depth >= 2) rendered as a pinned, edit-in-place
    tile inside its top-level delegate's thread.

    Shows a rolling window of the agent's latest output. The message id is the
    addressing handle: replying to it routes to this sub-agent.
    """

    client: "MessagingClient"
    thread_id: "ChannelId"
    agent_id: str
    title: str = ""                       # short label from delegate.spawned
    _message_id: str | None = field(default=None, init=False)
    _post: "PostResult | None" = field(default=None, init=False)
    _window: str = field(default="", init=False)
    _current_tool: str | None = field(default=None, init=False)
    _status: str = field(default="running", init=False)
    _pinned: bool = field(default=False, init=False)

    @property
    def message_id(self) -> str | None:
        return self._message_id

    def _embed(self) -> "EmbedSpec":
        from .protocol import EmbedSpec
        lineage = self.agent_id
        head = f"↳ {self.title or lineage}"
        if self._current_tool:
            head += f" · ⚙ {self._current_tool}"
        body = _truncate_tail(self._window, _TILE_WINDOW) or "*working…*"
        color = {
            "running": _COLOR_TILE,
            "completed": _COLOR_DONE,
            "failed": _COLOR_FAIL,
        }.get(self._status, _COLOR_TILE)
        return EmbedSpec(
            title=head,
            description=body,
            color=color,
            footer=f"-# {lineage}" + ("" if self._status == "running" else f" · {self._status}"),
        )

    async def start(self) -> None:
        try:
            self._post = await self.client.send(self.thread_id, embed=self._embed())
            self._message_id = self._post.message_id
        except Exception:
            log.exception("tile start failed (%s)", self.agent_id)
            return
        # Pin so it's both easy to find and a stable reply target.
        try:
            await self._pin()
        except Exception:
            log.debug("tile pin failed (%s)", self.agent_id, exc_info=True)

    async def _pin(self) -> None:
        if self._message_id is None or self._pinned:
            return
        pin = getattr(self.client, "pin", None)
        if pin is None:
            return
        await pin(self._message_id, self.thread_id)
        self._pinned = True

    async def _unpin(self) -> None:
        if self._message_id is None or not self._pinned:
            return
        unpin = getattr(self.client, "unpin", None)
        if unpin is None:
            return
        try:
            await unpin(self._message_id, self.thread_id)
            self._pinned = False
        except Exception:
            log.debug("tile unpin failed (%s)", self.agent_id, exc_info=True)

    async def _redraw(self) -> None:
        if self._message_id is None:
            return
        try:
            await self.client.edit(self._message_id, self.thread_id, embed=self._embed())
        except Exception:
            log.debug("tile redraw failed (%s)", self.agent_id, exc_info=True)

    async def on_tool(self, tool_name: str) -> None:
        self._current_tool = clean_tool_name(tool_name)
        await self._redraw()

    async def on_text(self, delta: str) -> None:
        if not delta:
            return
        self._current_tool = None
        self._window = (self._window + delta)[-_TILE_WINDOW:]
        await self._redraw()

    async def on_thinking(self, delta: str) -> None:
        if not delta:
            return
        self._window = (self._window + delta)[-_TILE_WINDOW:]
        await self._redraw()

    async def finalize(self, status: str = "completed") -> None:
        self._status = status
        self._current_tool = None
        await self._redraw()
        # Completed sub-agents free a pin slot (cap is 50/thread); the tile
        # stays in place as a reply target, just unpinned.
        if status in ("completed", "failed"):
            await self._unpin()
