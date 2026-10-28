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
window).

``ChannelBook`` keeps, per (payer, provider), the open channel, its deposit
and the cumulative signed so far, in ``<data_dir>/service_channels.json``.
A voucher is committed only once its request may have reached the provider;
a request that was never delivered leaves the total unchanged.
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

    def commit(self, payer: str, provider: str, channel_id: int, cumulative: int) -> None:
        with self._lock:
            ch = self._data["open"].get(self._key(payer, provider))
            if ch and int(ch["channel_id"]) == int(channel_id):
                ch["cumulative"] = max(int(ch["cumulative"]), int(cumulative))
                self._save()

    def retire(self, payer: str, provider: str) -> Optional[dict[str, Any]]:
        """Move the open channel to the closing list (its remainder is to be
        withdrawn once the challenge window passes)."""
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
    replaces) the channel when the open one cannot cover another ask.
    Returns ``{channel_id, cumulative_amount, signature}`` (not yet
    committed: call ``book.commit`` once the request may have arrived), or
    ``{"error": ...}``."""
    ask = int(ask)
    ch = book.get(payer, provider)
    if ch is None or int(ch["cumulative"]) + ask > int(ch["deposit"]):
        if ch is not None:
            # Spent: close it so its remainder comes back, then open anew.
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


async def sweep_refunds(book: ChannelBook, smc: Any, key_for: Any) -> list[dict[str, Any]]:
    """Withdraw the remainder of every closing channel whose challenge
    window has passed; forget channels that are done."""
    out: list[dict[str, Any]] = []
    now = time.time()
    for cid, ch in sorted(book.closing().items()):
        state = await smc.get_channel(int(cid))
        if state is None:
            continue
        status = int(state.get("status", 0))
        if status == SETTLED:
            book.forget_closing(int(cid))
            out.append({"channel_id": cid, "action": "forgot"})
            continue
        if status not in (CLOSING, CLIENT_CLOSING) or now < float(state.get("closes_at") or 0):
            continue
        key = key_for(str(ch.get("payer") or ""))
        if not key:
            continue
        res = await smc.withdraw_remainder(key, int(cid))
        if res.get("success"):
            book.forget_closing(int(cid))
        out.append({"channel_id": cid, "action": "withdrew" if res.get("success") else "failed",
                    "error": res.get("error")})
    return out
