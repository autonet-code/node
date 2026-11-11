"""The buyer's credits with service providers (docs/services_market.md,
Decision 2026-11-12).

A buyer's credits with one provider are a payment channel: the Agent opens
them with a deposit (``open_credits``) and tops them up itself
(``top_up``); nothing here opens, closes or tops up on its own. The same
credits pay for every service of that provider, requests and sessions
alike.

Vouchers pay ahead. Each one raises the channel's running total by the
provider's voucher size (its price list), never past the deposit; the
provider serves while what it has served stays within the running total,
and asks for the next voucher when it runs low. So the most a buyer has at
risk is one voucher. The provider reports its count (``consumed``,
``cumulative``) on every reply and session, and the book keeps it, so the
buyer's daemon knows when the next voucher is due.

When the credits are spent (the deposit cannot cover another voucher),
requests and sessions are refused with an error the Agent sees; it tops up
or closes. Closing (``close_credits``) starts the channel's close; the
remainder comes back after the challenge window (``sweep_refunds``).

A request to a provider the buyer holds no credits with is paid directly
instead (one ``payForService`` transaction), unless the Agent opens credits
with it.

``ChannelBook`` keeps, per (payer, provider), the open channel, its
deposit, the running total signed so far and the provider's last reported
count, in ``<data_dir>/service_channels.json``. A voucher is counted as
soon as it is signed: the provider may hold it, and a book behind the
provider would have every later voucher refused as adding nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

OPEN, CLOSING, SETTLED, CLIENT_CLOSING = 1, 2, 3, 4


class ChannelBook:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"open": {}, "closing": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data.update(loaded)
            except (OSError, ValueError):
                log.warning("service channels: %s unreadable, starting empty", self.path)

    @staticmethod
    def _key(payer: str, provider: str) -> str:
        return f"{payer.lower()}:{provider.lower()}"

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, payer: str, provider: str) -> Optional[dict[str, Any]]:
        with self._lock:
            ch = self._data["open"].get(self._key(payer, provider))
            return dict(ch) if ch else None

    def set_open(self, payer: str, provider: str, channel_id: int, deposit: int) -> None:
        with self._lock:
            self._data["open"][self._key(payer, provider)] = {
                "channel_id": int(channel_id), "deposit": int(deposit),
                "cumulative": 0, "consumed": 0, "payer": payer,
                "provider": provider, "opened": time.time()}
            self._save()

    def _open_entry(self, payer: str, provider: str, channel_id: int):
        ch = self._data["open"].get(self._key(payer, provider))
        if ch and int(ch["channel_id"]) == int(channel_id):
            return ch
        return None

    def set_deposit(self, payer: str, provider: str, channel_id: int,
                    deposit: int) -> bool:
        """Record the open channel's deposit after a top-up. Returns whether
        the open channel to ``provider`` is ``channel_id`` (else nothing is
        changed)."""
        with self._lock:
            ch = self._open_entry(payer, provider, channel_id)
            if ch is None:
                return False
            ch["deposit"] = int(deposit)
            self._save()
            return True

    def set_signed(self, payer: str, provider: str, channel_id: int, cumulative: int) -> None:
        """A voucher for ``cumulative`` was signed (it counts at once)."""
        with self._lock:
            ch = self._open_entry(payer, provider, channel_id)
            if ch is not None:
                ch["cumulative"] = max(int(ch["cumulative"]), int(cumulative))
                self._save()

    def note_balance(self, payer: str, provider: str, channel_id: int,
                     consumed: int, cumulative: int) -> None:
        """The provider's count for the channel, from a reply or a session.
        The book's running total never drops below what the provider holds."""
        with self._lock:
            ch = self._open_entry(payer, provider, channel_id)
            if ch is None:
                return
            changed = False
            if int(consumed) != int(ch.get("consumed") or 0):
                ch["consumed"] = max(0, int(consumed))
                changed = True
            if int(cumulative) > int(ch["cumulative"]):
                ch["cumulative"] = int(cumulative)
                changed = True
            if changed:
                self._save()

    def retire(self, payer: str, provider: str) -> Optional[dict[str, Any]]:
        """Move the open channel to the closing list (its remainder is to be
        withdrawn: at once if the provider closed it, after the challenge
        window if the buyer did)."""
        with self._lock:
            ch = self._data["open"].pop(self._key(payer, provider), None)
            if ch:
                self._data["closing"][str(ch["channel_id"])] = ch
                self._save()
            return ch

    def closing(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._data["closing"].items()}

    def forget_closing(self, channel_id: int) -> None:
        with self._lock:
            if self._data["closing"].pop(str(int(channel_id)), None) is not None:
                self._save()

    def all_open(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(v) for v in self._data["open"].values()]


_BOOKS: dict[str, ChannelBook] = {}
_BOOKS_LOCK = threading.Lock()


def book_at(path: Path) -> ChannelBook:
    """The one ChannelBook for ``path`` in this process."""
    key = str(Path(path).resolve())
    with _BOOKS_LOCK:
        if key not in _BOOKS:
            _BOOKS[key] = ChannelBook(Path(path))
        return _BOOKS[key]


def channel_lock(book: ChannelBook, payer: str, provider: str) -> asyncio.Lock:
    """The lock that serializes signing on one (payer, provider) channel:
    two vouchers signed at once would sign the same running total."""
    key = (str(payer or "").lower(), str(provider or "").lower())
    with _BOOKS_LOCK:
        locks = book.__dict__.setdefault("_async_locks", {})
        lock = locks.get(key)
        if lock is None:
            lock = locks[key] = asyncio.Lock()
        return lock


def credits_view(ch: Optional[dict[str, Any]]) -> dict[str, Any]:
    """What the Agent sees of its credits with one provider (base units)."""
    if not ch:
        return {}
    deposit, cumulative = int(ch.get("deposit") or 0), int(ch.get("cumulative") or 0)
    consumed = int(ch.get("consumed") or 0)
    return {"channel_id": int(ch["channel_id"]), "deposit": deposit,
            "used": consumed, "paid_ahead": max(0, cumulative - consumed),
            "left": max(0, deposit - consumed)}


async def open_credits(book: ChannelBook, smc: Any, payer_key: str, payer: str,
                       provider: str, deposit: int) -> dict[str, Any]:
    """Open credits with ``provider``: a payment channel escrowing
    ``deposit`` (ATN base units). One per provider: with credits already
    open, top them up instead."""
    if int(deposit or 0) <= 0:
        return {"error": "credits need a deposit above 0 (ATN base units)"}
    ch = book.get(payer, provider)
    if ch is not None:
        state = await smc.get_channel(int(ch["channel_id"]))
        if state is None or int(state.get("status", 0)) == OPEN:
            return {"error": "credits with this provider are already open: top them "
                             "up instead", "credits": credits_view(ch)}
        # The provider closed them: new credits replace them, and the old
        # channel's remainder comes back through sweep_refunds.
        book.retire(payer, provider)
    opened = await smc.open_channel(payer_key, provider, int(deposit))
    if not opened.get("success") or opened.get("channel_id") is None:
        return {"error": f"could not open credits with {provider[:10]}: "
                         f"{opened.get('error') or 'no channel id in the receipt'}"}
    book.set_open(payer, provider, int(opened["channel_id"]), int(deposit))
    return {"ok": True, "tx_hash": opened.get("tx_hash"),
            "credits": credits_view(book.get(payer, provider))}


async def next_voucher(book: ChannelBook, smc: Any, payer_key: str, payer: str,
                       provider: str, *, need: int, voucher: int,
                       consumed: Optional[int] = None) -> dict[str, Any]:
    """The next voucher on the payer's credits with ``provider``: the
    running total raised to cover ``need`` more than the provider has
    served, by at least ``voucher`` (the provider's voucher size), never
    past the deposit. Call under ``channel_lock``. Returns ``{channel_id,
    cumulative_amount, signature}`` (already counted) or ``{"error"}``."""
    ch = book.get(payer, provider)
    if ch is None:
        return {"error": "no credits with this provider: open credits first"}
    state = await smc.get_channel(int(ch["channel_id"]))
    if state is not None and int(state.get("status", 0)) != OPEN:
        book.retire(payer, provider)
        return {"error": "your credits with this provider are closed (the provider "
                         "or a close of yours ended them): open new credits"}
    if consumed is not None:
        book.note_balance(payer, provider, int(ch["channel_id"]), int(consumed),
                          int(ch["cumulative"]))
        ch = book.get(payer, provider) or ch
    deposit, cumulative = int(ch["deposit"]), int(ch["cumulative"])
    used = int(ch.get("consumed") or 0)
    target = max(cumulative, used + int(need), max(cumulative, used) + int(voucher or 0))
    target = min(target, deposit)
    if target <= cumulative or target - used < int(need):
        return {"error": f"your credits with this provider are spent ({used} of a "
                         f"{deposit} deposit used): top them up",
                "credits": credits_view(ch)}
    signature = smc.sign_voucher(payer_key, int(ch["channel_id"]), target)
    book.set_signed(payer, provider, int(ch["channel_id"]), target)
    return {"channel_id": int(ch["channel_id"]), "cumulative_amount": target,
            "signature": signature}


def _needs_voucher(reply: dict[str, Any]) -> bool:
    err = str(reply.get("error") or "")
    return "voucher needed" in err


async def request_on_credits(book: ChannelBook, smc: Any, config: Any,
                             payer_key: str, payer: str, provider: str,
                             spec_digest: str, args: dict[str, Any], *,
                             price: int, voucher: int,
                             **request_kwargs: Any) -> dict[str, Any]:
    """Send one request paid from the payer's credits with ``provider``.
    A voucher goes with it when the credits the provider holds do not
    cover ``price``; if the provider's count says otherwise, it asks, and
    the request is sent once more with one. The reply gains ``credits``
    (the book's view) and the ``request_id`` it was sent under."""
    from . import service_client

    ch = book.get(payer, provider)
    if ch is None:
        return {"error": "no credits with this provider", "delivered": False}
    cid = int(ch["channel_id"])
    reply: dict[str, Any] = {}
    for attempt in range(2):
        fields: dict[str, Any] = {}
        async with channel_lock(book, payer, provider):
            ch = book.get(payer, provider) or ch
            held = int(ch["cumulative"]) - int(ch.get("consumed") or 0)
            if attempt or held < int(price):
                fields = await next_voucher(book, smc, payer_key, payer, provider,
                                            need=int(price), voucher=int(voucher))
                if fields.get("error"):
                    return {**fields, "delivered": False}
        request_id = service_client.new_request_id()
        reply = await service_client.request_service(
            config, provider, spec_digest, args, request_id=request_id,
            client_address=payer, payer_key=payer_key,
            voucher=fields or None, channel_id=None if fields else cid,
            **request_kwargs)
        credit = reply.get("credit") if isinstance(reply.get("credit"), dict) else None
        if credit is not None:
            try:
                book.note_balance(payer, provider, cid, int(credit.get("consumed") or 0),
                                  int(credit.get("cumulative") or 0))
            except (TypeError, ValueError):
                pass
        if not (_needs_voucher(reply) and not fields):
            break
    reply.setdefault("request_id", request_id)
    reply["credits"] = credits_view(book.get(payer, provider))
    return reply


async def close_credits(book: ChannelBook, smc: Any, payer_key: str, payer: str,
                        provider: str) -> dict[str, Any]:
    """Start closing the credits with ``provider``; the remainder is
    withdrawn by ``sweep_refunds`` after the challenge window."""
    ch = book.get(payer, provider)
    if ch is None:
        return {"error": "no credits with that provider"}
    res = await smc.start_close(payer_key, int(ch["channel_id"]))
    if not res.get("success"):
        return {"error": f"startClose failed: {res.get('error')}"}
    book.retire(payer, provider)
    return {"ok": True, "channel_id": ch["channel_id"], "tx_hash": res.get("tx_hash")}


async def sweep_refunds(book: ChannelBook, smc: Any, key_for: Any,
                        now: Optional[float] = None) -> list[dict[str, Any]]:
    """Take back what closed channels did not pay.

    - An open channel the provider closed on chain moves to the closing
      list.
    - A closing channel still Open on chain (our startClose failed or was
      never sent) gets its startClose sent again.
    - A closing channel whose remainder is due (at once after the
      provider's close, after the challenge window after the buyer's) has
      it withdrawn; one settled is forgotten."""
    out: list[dict[str, Any]] = []
    for ch in book.all_open():
        state = await smc.get_channel(int(ch["channel_id"]))
        if state is not None and int(state.get("status", 0)) != OPEN:
            book.retire(str(ch["payer"]), str(ch["provider"]))
            out.append({"channel_id": str(ch["channel_id"]), "action": "closed_by_provider"})
    for cid, ch in sorted(book.closing().items()):
        state = await smc.get_channel(int(cid))
        if state is None:
            continue
        status = int(state.get("status", 0))
        if status == SETTLED:
            book.forget_closing(int(cid))
            out.append({"channel_id": cid, "action": "forgot"})
            continue
        key = key_for(str(ch.get("payer") or ""))
        if not key:
            continue
        if status == OPEN:
            res = await smc.start_close(key, int(cid))
            out.append({"channel_id": cid,
                        "action": "close_started" if res.get("success") else "failed",
                        "error": res.get("error")})
            continue
        if status not in (CLOSING, CLIENT_CLOSING):
            continue
        if (now if now is not None else time.time()) < float(state.get("closes_at") or 0):
            continue
        res = await smc.withdraw_remainder(key, int(cid))
        if res.get("success"):
            book.forget_closing(int(cid))
        out.append({"channel_id": cid, "action": "withdrew" if res.get("success") else "failed",
                    "error": res.get("error")})
    return out


async def top_up(book: ChannelBook, smc: Any, payer_key: str, payer: str,
                 provider: str, amount: int) -> dict[str, Any]:
    """Add ``amount`` ATN to the payer's open channel with ``provider`` (its
    credit with that provider, across all of the provider's services):
    ``topUp`` on chain (an ``approve`` first when the allowance falls
    short), then the book's deposit. Only an Open channel takes a top-up;
    a closing one is refused before anything is sent.

    Returns ``{ok, channel_id, deposit, tx_hash}`` or ``{"error": ...}``."""
    amount = int(amount)
    if amount <= 0:
        return {"error": "a top-up adds a positive amount"}
    ch = book.get(payer, provider)
    if ch is None:
        return {"error": "no credits with that provider: open credits first"}
    cid = int(ch["channel_id"])
    state = await smc.get_channel(cid)
    if state is not None and int(state.get("status", 0)) != OPEN:
        book.retire(payer, provider)
        return {"error": f"your credits with this provider (channel {cid}) are "
                         f"closed (status {state.get('status')}): open new credits"}
    res = await smc.top_up_channel(payer_key, cid, amount)
    if not res.get("success"):
        return {"error": f"topUp failed: {res.get('error')}"}
    deposit = res.get("deposit")
    if deposit is None:
        base = state.get("deposit") if state is not None else ch["deposit"]
        deposit = int(str(base or "0")) + amount
    book.set_deposit(payer, provider, cid, int(deposit))
    return {"ok": True, "channel_id": cid, "deposit": int(deposit),
            "tx_hash": res.get("tx_hash"),
            "credits": credits_view(book.get(payer, provider))}

