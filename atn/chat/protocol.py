"""Platform-neutral chat messaging interface.

Covers the small set of primitives we actually use today:
    rooms/channels  ->  threads  ->  messages  ->  edits / replies / reactions
plus a live-updating "status" message (embed) that streams tool calls and
final response, used by Kevin during agentic turns.

Discord, Matrix/Element, Slack, MS Teams all converge on this shape. The
goal is that when we port from Discord to (say) Matrix, the only code that
changes is the adapter module; the rest of Kevin can stay put.

Phase 1 just defines the protocol. Adapters come later — for now, existing
code continues to use discord.py directly. New code is encouraged to depend
on these types where it makes sense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
#
# All IDs are strings even when the underlying platform uses ints. Discord
# uses snowflake ints; Matrix uses opaque strings; treating them as strings
# uniformly is friction-free at the type level. Adapters convert at the edge.

ChannelId = str
MessageId = str
UserId = str
ThreadId = str


# ---------------------------------------------------------------------------
# Channel kinds
# ---------------------------------------------------------------------------

class ChannelKind(str, Enum):
    """Coarse category of a chat surface."""

    TEXT = "text"          # A regular room/channel
    THREAD = "thread"      # A thread inside another channel
    DM = "dm"              # Private 1:1 with the bot
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Author / user
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Author:
    id: UserId
    display_name: str           # Server-nick / display name when available, else handle
    is_bot: bool = False
    handle: str = ""            # Platform handle / @mention key (without the @)


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    id: ChannelId
    kind: ChannelKind
    name: str = ""
    parent_id: ChannelId | None = None     # If kind == THREAD, the parent channel
    server_id: str | None = None           # Guild / workspace / space id
    topic: str = ""


# ---------------------------------------------------------------------------
# Inbound message
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Message:
    """An inbound chat message routed to the bot."""

    id: MessageId
    channel: Channel
    author: Author
    content: str
    ts: datetime
    mentions_bot: bool = False             # True if the bot was @-mentioned
    reply_to_id: MessageId | None = None   # If this message is a reply
    raw: Any = None                        # Adapter-specific original message object


# ---------------------------------------------------------------------------
# Outbound: embed spec (platform-neutral)
# ---------------------------------------------------------------------------

@dataclass
class EmbedField:
    name: str
    value: str
    inline: bool = False


@dataclass
class EmbedSpec:
    """A rich-content message description.

    Adapters render this to the native shape (Discord embed, Matrix HTML
    formatted body, Slack block kit, etc.). Keep the surface small — only
    what we actually use.
    """

    title: str = ""
    description: str = ""
    color: int | None = None               # 0xRRGGBB, optional
    fields: list[EmbedField] = field(default_factory=list)
    footer: str = ""
    url: str = ""                          # Title link, when supported


# ---------------------------------------------------------------------------
# Outbound: history (read-back)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryMessage:
    id: MessageId
    author: Author
    content: str
    ts: datetime


# ---------------------------------------------------------------------------
# Outbound: post result + status-embed handle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostResult:
    """Returned by send/reply/create_thread so callers can edit or react later."""

    message_id: MessageId
    channel_id: ChannelId
    thread_id: ThreadId | None = None      # Set when create_thread() was called


class StatusEmbedHandle(Protocol):
    """A live-updating "status" message used during agentic turns.

    Lifecycle:
        send_initial()   -> post "Thinking..."
        set_tool(name, detail=None)
                         -> update title/description to reflect a tool call
        append_thinking(text)
                         -> stream reasoning text (italic) into the body
        clear_tool()     -> clear the current tool indicator
        finalize(text)   -> replace the status with the final response

    Discord adapter implements this as an embed that gets edited in place,
    then replaced with the final message content. Matrix adapter would
    edit a message with HTML formatting; Slack would update a Block Kit
    message. Same lifecycle.
    """

    async def send_initial(self) -> None: ...
    async def set_tool(self, name: str, detail: str | None = None) -> None: ...
    async def clear_tool(self) -> None: ...
    async def append_thinking(self, text: str) -> None: ...
    async def finalize(self, text: str) -> None: ...


# ---------------------------------------------------------------------------
# MessagingClient -- the surface adapters implement
# ---------------------------------------------------------------------------

@runtime_checkable
class MessagingClient(Protocol):
    """Platform-agnostic chat client interface.

    Adapters implement this; callers depend on this Protocol. Keep it small.
    Add a method only when we actually need one.
    """

    @property
    def me(self) -> Author:
        """The bot's own identity on this platform."""
        ...

    async def send(
        self,
        channel_id: ChannelId,
        *,
        content: str = "",
        embed: EmbedSpec | None = None,
    ) -> PostResult:
        """Post a new message in the given channel/thread."""
        ...

    async def reply(
        self,
        message_id: MessageId,
        channel_id: ChannelId,
        *,
        content: str = "",
        embed: EmbedSpec | None = None,
        mention_author: bool = False,
    ) -> PostResult:
        """Reply to an existing message (as a Discord-style reply / Matrix
        in-reply-to / Slack thread reply, depending on platform)."""
        ...

    async def edit(
        self,
        message_id: MessageId,
        channel_id: ChannelId,
        *,
        content: str | None = None,
        embed: EmbedSpec | None = None,
    ) -> None:
        """Edit a message we previously posted."""
        ...

    async def delete(
        self,
        message_id: MessageId,
        channel_id: ChannelId,
    ) -> None:
        """Delete a message we previously posted."""
        ...

    async def create_thread(
        self,
        parent_message_id: MessageId,
        channel_id: ChannelId,
        *,
        name: str,
    ) -> PostResult:
        """Create a thread anchored to an existing message. Returns the
        thread channel id (in PostResult.thread_id) so subsequent send()
        calls can target it."""
        ...

    async def get_history(
        self,
        channel_id: ChannelId,
        *,
        limit: int = 30,
    ) -> list[HistoryMessage]:
        """Read recent messages from a channel/thread, chronological order."""
        ...

    async def add_reaction(
        self,
        message_id: MessageId,
        channel_id: ChannelId,
        emoji: str,
    ) -> None:
        """Add an emoji reaction to a message."""
        ...

    def make_status_handle(
        self,
        channel_id: ChannelId,
        *,
        reply_to_id: MessageId | None = None,
    ) -> StatusEmbedHandle:
        """Construct a live-updating status handle for an upcoming agentic
        turn. Adapter owns the rendering details."""
        ...


# ---------------------------------------------------------------------------
# Notes for future adapters (intentionally not implemented yet)
# ---------------------------------------------------------------------------
#
# Discord adapter mapping:
#   - Author.id = str(discord.User.id), Author.handle = User.name
#   - Channel.id = str(channel.id), Channel.parent_id = str(channel.parent_id)
#   - EmbedSpec -> discord.Embed.from_dict({...})
#   - StatusEmbedHandle = the existing _StatusEmbed in general_support.py
#   - mention_author flag maps to discord.Message.reply(mention_author=)
#
# Matrix adapter mapping (rough sketch — for when we get there):
#   - Author.id = MXID ("@user:server"), Author.handle = displayname or localpart
#   - Channel.id = room id; threads are root event ids inside a room.
#   - EmbedSpec -> rich HTML body via formatted_body (m.room.message).
#   - reply uses m.relates_to / m.in_reply_to.
#   - StatusEmbedHandle = a message we edit via m.replace.
#
# Slack adapter mapping (sketch):
#   - Channel = channel id, threads = thread_ts, replies = thread_ts on
#     chat.postMessage.
#   - EmbedSpec -> Block Kit blocks (header + section + context).
