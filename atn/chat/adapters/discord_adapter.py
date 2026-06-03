"""Discord adapter -- a MessagingClient backed by discord.py.

This is the first real platform adapter. It owns a `discord.Client` (this IS
the bot process), maps the platform-neutral protocol onto discord.py at the
edge, and converts inbound `discord.Message` objects into protocol `Message`
objects (carrying `reply_to_id`, which the ChatService uses for tile
addressing) handed to an `on_message` callback.

Mapping (per messaging/protocol.py notes):
  - Author.id      = str(user.id), Author.handle = user.name
  - Channel.id     = str(channel.id), parent_id = str(channel.parent_id)
  - EmbedSpec      -> discord.Embed
  - StatusEmbedHandle = throttled edit-in-place embed (general_support pattern)
  - reply / mention_author -> discord.Message.reply(mention_author=)

All ids are strings at the protocol boundary; we int() them when calling
discord.py. The adapter never reaches into autonet -- it only talks Discord.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

import discord

from ..protocol import (
    Author, Channel, ChannelId, ChannelKind, EmbedSpec, HistoryMessage,
    Message, MessageId, PostResult,
)

log = logging.getLogger("hackathon.chat.adapters.discord")

# At most one embed edit per this many seconds (Discord-safe; matches fleet_view).
_EDIT_INTERVAL = 1.5
# Discord message content hard cap is 2000; embed description 4096.
_CONTENT_LIMIT = 1990
_DESC_LIMIT = 4000


def _to_embed(spec: EmbedSpec) -> discord.Embed:
    embed = discord.Embed(
        title=spec.title or None,
        description=(spec.description or "")[:_DESC_LIMIT] or None,
        color=spec.color if spec.color is not None else None,
        url=spec.url or None,
    )
    for f in spec.fields:
        embed.add_field(name=f.name, value=f.value, inline=f.inline)
    if spec.footer:
        embed.set_footer(text=spec.footer[:2048])
    return embed


def _channel_kind(ch: object) -> ChannelKind:
    if isinstance(ch, discord.Thread):
        return ChannelKind.THREAD
    if isinstance(ch, discord.DMChannel):
        return ChannelKind.DM
    if isinstance(ch, (discord.TextChannel,)):
        return ChannelKind.TEXT
    return ChannelKind.UNKNOWN


class _DiscordStatusHandle:
    """Edit-in-place status embed (the general_support _StatusEmbed pattern)
    expressed through the adapter so the ChatService stays platform-neutral."""

    def __init__(self, adapter: "DiscordAdapter", channel_id: ChannelId,
                 reply_to_id: MessageId | None = None) -> None:
        self._a = adapter
        self._channel_id = channel_id
        self._reply_to_id = reply_to_id
        self._message: discord.Message | None = None
        self._tool: str | None = None
        self._body = ""
        self._last_edit = 0.0
        self._final = False

    async def send_initial(self) -> None:
        embed = discord.Embed(description="*working…*", color=0x5865F2)
        try:
            ch = await self._a._resolve(self._channel_id)
            if ch is None:
                return
            if self._reply_to_id is not None:
                ref = await self._a._fetch_message(ch, self._reply_to_id)
                if ref is not None:
                    self._message = await ref.reply(embed=embed, mention_author=False)
                    return
            self._message = await ch.send(embed=embed)
        except Exception:
            log.exception("status send_initial failed")

    async def set_tool(self, name: str, detail: str | None = None) -> None:
        self._tool = name
        await self._maybe_edit()

    async def clear_tool(self) -> None:
        self._tool = None
        await self._maybe_edit()

    async def append_thinking(self, text: str) -> None:
        if not text:
            return
        self._body = (self._body + text)[-_DESC_LIMIT:]
        await self._maybe_edit()

    async def finalize(self, text: str) -> None:
        self._final = True
        self._tool = None
        self._body = text
        await self._edit(force=True)

    async def _maybe_edit(self) -> None:
        if self._final:
            return
        now = time.monotonic()
        if now - self._last_edit < _EDIT_INTERVAL:
            return
        await self._edit()

    async def _edit(self, force: bool = False) -> None:
        if self._message is None:
            return
        self._last_edit = time.monotonic()
        body = self._body.strip()
        if len(body) > _DESC_LIMIT:
            body = "…" + body[-_DESC_LIMIT:]
        if self._final:
            color = 0x57F287
            title = None
        else:
            color = 0x5865F2
            title = f"⚙ {self._tool}" if self._tool else "working…"
        embed = discord.Embed(
            title=title, description=body or "*working…*", color=color)
        try:
            await self._message.edit(embed=embed)
        except Exception:
            log.debug("status edit failed", exc_info=True)


class DiscordAdapter:
    """MessagingClient over discord.py. Owns the discord.Client.

    Usage:
        adapter = DiscordAdapter(token, allowed_channel_ids={CHANNEL_ID})
        adapter.on_message = chat_service.handle_inbound
        await adapter.start()      # blocks running the client
    """

    def __init__(
        self,
        token: str,
        *,
        allowed_channel_ids: set[str] | None = None,
    ) -> None:
        self._token = token
        # Restrict inbound handling to these channels (+ their threads). Empty
        # = all. The dОrg deployment binds exactly ONE channel.
        self._allowed = set(allowed_channel_ids or set())
        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._me: Author | None = None
        self.on_message: Callable[[Message], Awaitable[bool]] | None = None
        self._ready = asyncio.Event()

        @self._client.event
        async def on_ready() -> None:  # noqa: F811
            u = self._client.user
            self._me = Author(id=str(u.id), display_name=u.display_name,
                              is_bot=True, handle=u.name)
            self._ready.set()
            log.info("discord adapter ready as %s (%s)", u, u.id)

        @self._client.event
        async def on_message(message: discord.Message) -> None:  # noqa: F811
            await self._handle_raw(message)

    # -- lifecycle -----------------------------------------------------------

    @property
    def client(self) -> discord.Client:
        return self._client

    async def start(self) -> None:
        await self._client.start(self._token)

    async def close(self) -> None:
        await self._client.close()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    @property
    def me(self) -> Author:
        if self._me is None:
            return Author(id="?", display_name="bot", is_bot=True)
        return self._me

    # -- inbound -------------------------------------------------------------

    async def _handle_raw(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        ch = message.channel
        # Channel allow-list: the channel itself, or a thread under it.
        if self._allowed:
            cid = str(ch.id)
            parent = str(getattr(ch, "parent_id", "") or "")
            if cid not in self._allowed and parent not in self._allowed:
                return
        if self.on_message is None:
            return
        msg = self._to_message(message)
        try:
            await self.on_message(msg)
        except Exception:
            log.exception("on_message handler failed")

    def _to_message(self, m: discord.Message) -> Message:
        ch = m.channel
        parent_id = None
        if isinstance(ch, discord.Thread) and ch.parent_id:
            parent_id = str(ch.parent_id)
        channel = Channel(
            id=str(ch.id),
            kind=_channel_kind(ch),
            name=getattr(ch, "name", "") or "",
            parent_id=parent_id,
            server_id=str(m.guild.id) if m.guild else None,
        )
        author = Author(
            id=str(m.author.id),
            display_name=m.author.display_name,
            is_bot=m.author.bot,
            handle=m.author.name,
        )
        reply_to_id = None
        reply_to_bot = False
        if m.reference is not None and m.reference.message_id is not None:
            reply_to_id = str(m.reference.message_id)
            # A reply to one of the bot's own messages counts as engaging the
            # bot even if the reply-ping was toggled off (so it won't show up
            # in m.mentions). resolved is the cached replied-to message.
            resolved = getattr(m.reference, "resolved", None)
            ra = getattr(resolved, "author", None)
            if ra is not None and self._client.user is not None and ra.id == self._client.user.id:
                reply_to_bot = True
        mentions_bot = bool(
            (self._client.user and self._client.user in m.mentions) or reply_to_bot)
        # Strip the bot's own @mention from the content — the agent shouldn't
        # see "<@984…> what's up", just "what's up". (Discord uses <@id> and the
        # legacy <@!id> nickname form.)
        content = m.content or ""
        if self._client.user:
            bid = self._client.user.id
            content = content.replace(f"<@{bid}>", "").replace(f"<@!{bid}>", "").strip()
        return Message(
            id=str(m.id),
            channel=channel,
            author=author,
            content=content,
            ts=m.created_at or datetime.now(timezone.utc),
            mentions_bot=mentions_bot,
            reply_to_id=reply_to_id,
            raw=m,
        )

    # -- channel/message resolution -----------------------------------------

    async def _resolve(self, channel_id: ChannelId):
        """Channel/thread object from a string id (cache, then fetch)."""
        try:
            cid = int(channel_id)
        except (TypeError, ValueError):
            return None
        ch = self._client.get_channel(cid)
        if ch is not None:
            return ch
        try:
            return await self._client.fetch_channel(cid)
        except Exception:
            log.debug("could not resolve channel %s", channel_id, exc_info=True)
            return None

    async def _fetch_message(self, channel, message_id: MessageId):
        try:
            return await channel.fetch_message(int(message_id))
        except Exception:
            return None

    # -- MessagingClient surface --------------------------------------------

    async def send(self, channel_id: ChannelId, *, content: str = "",
                   embed: EmbedSpec | None = None) -> PostResult:
        ch = await self._resolve(channel_id)
        if ch is None:
            raise RuntimeError(f"channel {channel_id} not resolvable")
        msg = await ch.send(
            content=(content or None) and content[:_CONTENT_LIMIT],
            embed=_to_embed(embed) if embed else None,
        )
        return PostResult(message_id=str(msg.id), channel_id=channel_id)

    async def reply(self, message_id: MessageId, channel_id: ChannelId, *,
                    content: str = "", embed: EmbedSpec | None = None,
                    mention_author: bool = False) -> PostResult:
        ch = await self._resolve(channel_id)
        if ch is None:
            raise RuntimeError(f"channel {channel_id} not resolvable")
        ref = await self._fetch_message(ch, message_id)
        kwargs = dict(
            content=(content or None) and content[:_CONTENT_LIMIT],
            embed=_to_embed(embed) if embed else None,
        )
        if ref is not None:
            msg = await ref.reply(mention_author=mention_author, **kwargs)
        else:
            msg = await ch.send(**kwargs)
        return PostResult(message_id=str(msg.id), channel_id=channel_id)

    async def edit(self, message_id: MessageId, channel_id: ChannelId, *,
                   content: str | None = None, embed: EmbedSpec | None = None) -> None:
        ch = await self._resolve(channel_id)
        if ch is None:
            return
        msg = await self._fetch_message(ch, message_id)
        if msg is None:
            return
        kwargs = {}
        if content is not None:
            kwargs["content"] = content[:_CONTENT_LIMIT]
        if embed is not None:
            kwargs["embed"] = _to_embed(embed)
        await msg.edit(**kwargs)

    async def delete(self, message_id: MessageId, channel_id: ChannelId) -> None:
        ch = await self._resolve(channel_id)
        if ch is None:
            return
        msg = await self._fetch_message(ch, message_id)
        if msg is not None:
            try:
                await msg.delete()
            except Exception:
                log.debug("delete failed", exc_info=True)

    async def create_thread(self, parent_message_id: MessageId,
                            channel_id: ChannelId, *, name: str) -> PostResult:
        ch = await self._resolve(channel_id)
        if ch is None:
            raise RuntimeError(f"channel {channel_id} not resolvable")
        anchor = await self._fetch_message(ch, parent_message_id)
        if anchor is not None:
            thread = await anchor.create_thread(name=name[:100],
                                                auto_archive_duration=10080)
        else:
            thread = await ch.create_thread(
                name=name[:100], type=discord.ChannelType.public_thread,
                auto_archive_duration=10080)
        return PostResult(message_id=parent_message_id, channel_id=channel_id,
                          thread_id=str(thread.id))

    async def get_history(self, channel_id: ChannelId, *,
                          limit: int = 30) -> list[HistoryMessage]:
        ch = await self._resolve(channel_id)
        if ch is None:
            return []
        out: list[HistoryMessage] = []
        async for m in ch.history(limit=limit, oldest_first=True):
            out.append(HistoryMessage(
                id=str(m.id),
                author=Author(id=str(m.author.id),
                              display_name=m.author.display_name,
                              is_bot=m.author.bot, handle=m.author.name),
                content=m.content or "",
                ts=m.created_at or datetime.now(timezone.utc),
            ))
        return out

    async def add_reaction(self, message_id: MessageId, channel_id: ChannelId,
                           emoji: str) -> None:
        ch = await self._resolve(channel_id)
        if ch is None:
            return
        msg = await self._fetch_message(ch, message_id)
        if msg is not None:
            try:
                await msg.add_reaction(emoji)
            except Exception:
                log.debug("add_reaction failed", exc_info=True)

    async def pin(self, message_id: MessageId, channel_id: ChannelId) -> None:
        ch = await self._resolve(channel_id)
        if ch is None:
            return
        msg = await self._fetch_message(ch, message_id)
        if msg is not None:
            try:
                await msg.pin()
            except Exception:
                log.debug("pin failed (cap reached?)", exc_info=True)

    async def unpin(self, message_id: MessageId, channel_id: ChannelId) -> None:
        ch = await self._resolve(channel_id)
        if ch is None:
            return
        msg = await self._fetch_message(ch, message_id)
        if msg is not None:
            try:
                await msg.unpin()
            except Exception:
                log.debug("unpin failed", exc_info=True)

    def make_status_handle(self, channel_id: ChannelId, *,
                           reply_to_id: MessageId | None = None) -> _DiscordStatusHandle:
        return _DiscordStatusHandle(self, channel_id, reply_to_id)
