"""In-memory stub adapter -- a MessagingClient that records instead of sending.

Used to prove the ChatService end-to-end without a real platform: every
send/edit/reply/create_thread/pin call is recorded in an in-memory store, and
a transcript can be printed or asserted on. Channels and threads are just ids;
messages are dicts. The StatusEmbedHandle is implemented as an editable
message so the live -> final lifecycle exercises the same code paths.

This is what the design doc §7 calls the "stub/echo adapter": the foundation
test target before the Discord adapter exists.
"""
from __future__ import annotations

import itertools
import logging

from ..protocol import (
    Author, ChannelId, EmbedSpec, HistoryMessage, MessageId, PostResult,
)

log = logging.getLogger("hackathon.chat.adapters.stub")


class _Store:
    """Shared mutable state for a stub adapter run."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        # message_id -> {channel_id, content, embed, pinned, replied_to, kind}
        self.messages: dict[str, dict] = {}
        # thread_id -> {parent_channel, name, anchor_message_id}
        self.threads: dict[str, dict] = {}

    def new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._ids)}"


class _StubStatusHandle:
    """StatusEmbedHandle backed by a single editable stub message."""

    def __init__(self, adapter: "StubAdapter", channel_id: ChannelId,
                 reply_to_id: MessageId | None = None) -> None:
        self._a = adapter
        self._channel_id = channel_id
        self._reply_to_id = reply_to_id
        self._message_id: str | None = None
        self._tool: str | None = None
        self._body = ""

    async def send_initial(self) -> None:
        post = await self._a.send(self._channel_id, content="*working…*")
        self._message_id = post.message_id

    async def set_tool(self, name: str, detail: str | None = None) -> None:
        self._tool = name
        await self._redraw()

    async def clear_tool(self) -> None:
        self._tool = None
        await self._redraw()

    async def append_thinking(self, text: str) -> None:
        self._body += text
        await self._redraw()

    async def finalize(self, text: str) -> None:
        self._tool = None
        self._body = text
        await self._redraw(final=True)

    async def _redraw(self, final: bool = False) -> None:
        if self._message_id is None:
            return
        prefix = "" if final else (f"[⚙ {self._tool}] " if self._tool else "[…] ")
        await self._a.edit(self._message_id, self._channel_id,
                           content=prefix + self._body)


class StubAdapter:
    """A recording MessagingClient. Implements the full protocol surface plus
    pin/unpin (used by sub-agent tiles via duck-typing)."""

    def __init__(self, store: _Store | None = None,
                 me: Author | None = None) -> None:
        self._store = store or _Store()
        self._me = me or Author(id="stub-bot", display_name="StubBot",
                                 is_bot=True, handle="stub")

    @property
    def me(self) -> Author:
        return self._me

    @property
    def store(self) -> _Store:
        return self._store

    async def send(self, channel_id: ChannelId, *, content: str = "",
                   embed: EmbedSpec | None = None) -> PostResult:
        mid = self._store.new_id("m")
        self._store.messages[mid] = {
            "channel_id": channel_id, "content": content, "embed": embed,
            "pinned": False, "replied_to": None, "kind": "message",
        }
        log.debug("send -> %s: %s", channel_id, _short(content, embed))
        return PostResult(message_id=mid, channel_id=channel_id)

    async def reply(self, message_id: MessageId, channel_id: ChannelId, *,
                    content: str = "", embed: EmbedSpec | None = None,
                    mention_author: bool = False) -> PostResult:
        mid = self._store.new_id("m")
        self._store.messages[mid] = {
            "channel_id": channel_id, "content": content, "embed": embed,
            "pinned": False, "replied_to": message_id, "kind": "reply",
        }
        return PostResult(message_id=mid, channel_id=channel_id)

    async def edit(self, message_id: MessageId, channel_id: ChannelId, *,
                   content: str | None = None, embed: EmbedSpec | None = None) -> None:
        m = self._store.messages.get(message_id)
        if m is None:
            return
        if content is not None:
            m["content"] = content
        if embed is not None:
            m["embed"] = embed

    async def delete(self, message_id: MessageId, channel_id: ChannelId) -> None:
        self._store.messages.pop(message_id, None)

    async def create_thread(self, parent_message_id: MessageId,
                            channel_id: ChannelId, *, name: str) -> PostResult:
        tid = self._store.new_id("t")
        self._store.threads[tid] = {
            "parent_channel": channel_id, "name": name,
            "anchor_message_id": parent_message_id,
        }
        log.debug("create_thread %s (%s) under %s", tid, name, channel_id)
        return PostResult(message_id=parent_message_id, channel_id=channel_id,
                          thread_id=tid)

    async def get_history(self, channel_id: ChannelId, *,
                          limit: int = 30) -> list[HistoryMessage]:
        return []

    async def add_reaction(self, message_id: MessageId, channel_id: ChannelId,
                           emoji: str) -> None:
        pass

    async def pin(self, message_id: MessageId, channel_id: ChannelId) -> None:
        m = self._store.messages.get(message_id)
        if m is not None:
            m["pinned"] = True

    async def unpin(self, message_id: MessageId, channel_id: ChannelId) -> None:
        m = self._store.messages.get(message_id)
        if m is not None:
            m["pinned"] = False

    def make_status_handle(self, channel_id: ChannelId, *,
                           reply_to_id: MessageId | None = None) -> _StubStatusHandle:
        return _StubStatusHandle(self, channel_id, reply_to_id)

    # -- test helpers --------------------------------------------------------

    def transcript(self) -> str:
        lines: list[str] = []
        for tid, t in self._store.threads.items():
            lines.append(f"THREAD {tid} [{t['name']}] under {t['parent_channel']}")
        for mid, m in self._store.messages.items():
            tag = "📌" if m["pinned"] else "  "
            body = _short(m["content"], m["embed"])
            rep = f" (reply->{m['replied_to']})" if m["replied_to"] else ""
            lines.append(f"{tag} {mid} @{m['channel_id']}{rep}: {body}")
        return "\n".join(lines)


def _short(content: str, embed: EmbedSpec | None, n: int = 80) -> str:
    if embed is not None:
        t = embed.title or ""
        d = (embed.description or "").replace("\n", " ")
        s = f"<embed {t!r} {d}>"
    else:
        s = (content or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
