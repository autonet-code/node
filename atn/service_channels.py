"""Buyer side of the payment channel (docs/services_market.md, settlement:
"the prepaid payment channel, ONLY"; issue #45).

A buyer used to pay each request up front with ``payForService`` and then
send it, so a provider that was offline, or a request that never arrived,
kept the money with nothing served. With a channel the buyer escrows a
deposit once per provider and sends each request with a signed voucher for
its running total; the provider can collect only what vouchers cover, so the
most a buyer can lose to a provider who takes a voucher and does not serve is
one request's ask. The rest of the deposit comes back when the channel is
closed (``startClose``, then ``withdrawRemainder`` after the challenge
window), or when the provider closes it itself.

``ChannelBook`` keeps, per (payer, provider), the open channel, its deposit
and the cumulative signed so far, in ``<data_dir>/service_channels.json``.
A voucher is written down as pending before its request is sent
(``reserve``), then counted (``commit``) when the provider served it, or when
the request may have reached the provider and no answer says otherwise
(``settle_voucher``), and dropped (``release``) when the provider says it
kept nothing. A pending voucher found at start counts: the daemon stopped
while the request was out, so the provider may hold it. The book is never
behind the provider, or every later voucher would be refused as adding
nothing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# A new channel escrows this many requests' worth of the ask, unless the
# caller names a deposit. Escrow, not spend: the unused part is refunded.
DEFAULT_DEPOSIT_REQUESTS = 10

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
        # A voucher still pending was out when the daemon stopped: the
        # provider may have served it and kept it, so it counts.
        folded = False
        for ch in self._data["open"].values():
            if ch.get("pending") is not None:
                ch["cumulative"] = max(int(ch["cumulative"]), int(ch.pop("pending")))
                folded = True
        if folded:
            self._save()

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
                "cumulative": 0, "payer": payer, "provider": provider,
                "opened": time.time()}
            self._save()

    def _open_entry(self, payer: str, provider: str, channel_id: int):
        ch = self._data["open"].get(self._key(payer, provider))
        if ch and int(ch["channel_id"]) == int(channel_id):
            return ch
        return None

    def reserve(self, payer: str, provider: str, channel_id: int, cumulative: int) -> None:
        """Write the voucher down before its request goes out."""
        with self._lock:
            ch = self._open_entry(payer, provider, channel_id)
            if ch is not None:
                ch["pending"] = int(cumulative)
                self._save()

    def commit(self, payer: str, provider: str, channel_id: int, cumulative: int) -> None:
        with self._lock:
            ch = self._open_entry(payer, provider, channel_id)
            if ch is not None:
                ch["cumulative"] = max(int(ch["cumulative"]), int(cumulative))
                ch.pop("pending", None)
                self._save()

    def release(self, payer: str, provider: str, channel_id: int) -> None:
        """The provider kept nothing for the pending voucher: drop it."""
        with self._lock:
            ch = self._open_entry(payer, provider, channel_id)
            if ch is not None and ch.pop("pending", None) is not None:
                self._save()

    def retire(self, payer: str, provider: str) -> Optional[dict[str, Any]]:
        """Move the open channel to the closing list (its remainder is to be
        withdrawn once the challenge window passes)."""
        with self._lock:
            ch = self._data["open"].pop(self._key(payer, provider), None)
            if ch:
                if ch.get("pending") is not None:
                    ch["cumulative"] = max(int(ch["cumulative"]), int(ch.pop("pending")))
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
    """The one ChannelBook for ``path`` in this process (the agent tools and
    the service inference provider share it)."""
    key = str(Path(path).resolve())
    with _BOOKS_LOCK:
        if key not in _BOOKS:
            _BOOKS[key] = ChannelBook(Path(path))
        return _BOOKS[key]


async def voucher_for(book: ChannelBook, smc: Any, payer_key: str, payer: str,
                      provider: str, ask: int, deposit: int = 0) -> dict[str, Any]:
    """The voucher for one request of ``ask`` to ``provider``: opens (or
    replaces) the channel when the open one cannot cover another ask or is
    no longer open on chain (the provider closed it). Returns
    ``{channel_id, cumulative_amount, signature}`` (not yet counted: call
    ``book.reserve`` before sending and ``settle_voucher`` after), or
    ``{"error": ...}``."""
    ask = int(ask)
    ch = book.get(payer, provider)
    if ch is not None:
        state = await smc.get_channel(int(ch["channel_id"]))
        if state is not None and int(state.get("status", 0)) != OPEN:
            # Closed on chain (by the provider, or a close of ours that
            # landed): the refund sweep takes the remainder back.
            log.info("service channels: channel %s to %s is no longer open "
                     "(status %s); opening a new one", ch["channel_id"],
                     provider[:10], state.get("status"))
            book.retire(payer, provider)
            ch = None
    if ch is None or int(ch["cumulative"]) + ask > int(ch["deposit"]):
        if ch is not None:
            # Spent: close it so its remainder comes back, then open anew.
            # A failed close is sent again by the refund sweep.
            res = await smc.start_close(payer_key, int(ch["channel_id"]))
            if not res.get("success"):
                log.warning("service channels: closing spent channel %s failed: %s",
                            ch["channel_id"], res.get("error"))
            book.retire(payer, provider)
        want = int(deposit) if int(deposit or 0) >= ask else ask * DEFAULT_DEPOSIT_REQUESTS
        opened = await smc.open_channel(payer_key, provider, want)
        if not opened.get("success") or opened.get("channel_id") is None:
            return {"error": f"could not open a payment channel to {provider[:10]}: "
                             f"{opened.get('error') or 'no channel id in the receipt'}"}
        book.set_open(payer, provider, int(opened["channel_id"]), want)
        ch = book.get(payer, provider)
    cumulative = int(ch["cumulative"]) + ask
    signature = smc.sign_voucher(payer_key, int(ch["channel_id"]), cumulative)
    return {"channel_id": int(ch["channel_id"]), "cumulative_amount": cumulative,
            "signature": signature}


def settle_voucher(book: ChannelBook, payer: str, provider: str,
                   voucher: dict[str, Any], reply: dict[str, Any]) -> bool:
    """Count or drop a reserved voucher once its request is answered.
    Returns whether it counts (the buyer paid for that request).

    Counts when the provider served it (it says ``charged``), or when the
    request may have reached the provider and no answer from the provider
    says otherwise (a timeout, a broken stream, an error from a relay):
    the provider may have served it and kept the voucher, and a book behind
    the provider would have every later voucher refused. Dropped when the
    request never left, or the provider answered that it kept nothing (a
    refused or failed request is not paid)."""
    cid = int(voucher["channel_id"])
    charged = reply.get("charged")
    counts = bool(reply.get("delivered")) and (
        charged if isinstance(charged, bool) else True)
    if counts:
        book.commit(payer, provider, cid, int(voucher["cumulative_amount"]))
    else:
        book.release(payer, provider, cid)
    return counts


async def close_channel(book: ChannelBook, smc: Any, payer_key: str, payer: str,
                        provider: str) -> dict[str, Any]:
    """Start closing the open channel to ``provider``; its remainder is
    withdrawn by ``sweep_refunds`` after the challenge window."""
    ch = book.get(payer, provider)
    if ch is None:
        return {"error": "no open channel to that provider"}
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
    - A closing channel whose challenge window has passed has its
      remainder withdrawn; one settled is forgotten."""
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

