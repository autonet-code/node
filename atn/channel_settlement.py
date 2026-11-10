"""Provider-side settlement of payment channels (ServiceMarket.sol).

A provider daemon keeps each channel's best voucher as it serves
(``ServiceStore.keep_channel_voucher``) and collects it by closing the
channel (``closeChannel``, which pays it at once; the client's remainder is
refunded after the challenge window). The sweep closes a channel

- when its client has started a close (``startClose``): the provider has
  one challenge window to settle with its best voucher, after which the
  client takes the whole deposit back;
- when the deposit left after the best voucher no longer covers the
  service's price, read from its ServiceRegistry entry (the client cannot
  pay for another request on it);
- when it has had no new voucher for ``idle_close_s`` (default a day,
  ``autonet.channel_idle_close_s``), so earnings do not wait on a client
  that went quiet.

A channel already closed or settled is forgotten, and so is one whose
client-close window has passed (nothing can be collected any more; sending
``closeChannel`` then only burns gas on a revert). A channel with a request
still being served is left for the next sweep, so the close does not cut
off what that request pays.

Keys: the channel's provider is the operator (the owner surface's and the
external agents' services) or a native agent (its own key);
``key_for(provider)`` resolves it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

OPEN, CLOSING, SETTLED, CLIENT_CLOSING = 1, 2, 3, 4

DEFAULT_IDLE_CLOSE_S = 86400.0


async def _price(client: Any, voucher: dict[str, Any]) -> Optional[int]:
    """The price of the service the voucher last paid for, from its
    ServiceRegistry entry; the price the gate charged if unreadable."""
    sid = voucher.get("service_id")
    reader = getattr(client, "get_service", None)
    if sid is not None and reader is not None:
        try:
            entry = await reader(int(sid))
            if entry is not None:
                return int(str(entry.get("ask_amount") or "0") or "0")
        except Exception:  # noqa: BLE001
            log.debug("price read for service %s failed", sid, exc_info=True)
    ask = int(voucher.get("ask") or 0)
    return ask or None


async def settle_channels(
    store: Any, client: Any, key_for: Callable[[str], Optional[str]], *,
    idle_close_s: float = DEFAULT_IDLE_CLOSE_S, now: Optional[float] = None,
) -> list[dict[str, Any]]:
    """One sweep. ``client`` is a ServiceMarketClient; ``key_for(provider)``
    returns the provider's private key or None. Returns what was done."""
    out: list[dict[str, Any]] = []
    clock = time.time() if now is None else float(now)
    for cid, voucher in sorted(store.channel_vouchers().items()):
        ch = await client.get_channel(int(cid))
        if ch is None:
            continue
        status = int(ch["status"])
        if status in (CLOSING, SETTLED):
            store.forget_channel_voucher(cid)
            out.append({"channel_id": cid, "action": "forgot"})
            continue
        cumulative = int(voucher["cumulative"])
        if status == CLIENT_CLOSING:
            if clock >= float(ch.get("closes_at") or 0):
                # The window passed: the client takes the whole deposit and
                # closeChannel would revert. Nothing left to collect.
                log.warning("channel %s: the client's close window passed before "
                            "it was settled; voucher for %s lost", cid, cumulative)
                store.forget_channel_voucher(cid)
                out.append({"channel_id": cid, "action": "window_passed"})
                continue
            reason = "client_closing"
        elif status == OPEN:
            if getattr(store, "has_pending_voucher", lambda c: False)(cid):
                continue
            reason = ""
            price = await _price(client, voucher)
            remaining = int(str(ch.get("deposit") or "0")) - cumulative
            if price is not None and remaining < price:
                reason = "deposit_spent"
            elif idle_close_s > 0 and clock - float(voucher.get("ts") or clock) >= idle_close_s:
                reason = "idle"
            if not reason:
                continue
        else:
            continue
        key = key_for(str(ch["provider"]))
        if not key:
            log.warning("channel %s is due to close (%s) but no key is held for "
                        "its provider %s", cid, reason, ch["provider"])
            continue
        res = await client.close_channel(
            key, int(cid), cumulative, str(voucher["signature"]))
        if res.get("success"):
            store.forget_channel_voucher(cid)
            log.info("closed channel %s at %s (%s)", cid, cumulative, reason)
        else:
            log.warning("closing channel %s failed: %s", cid, res.get("error"))
        out.append({"channel_id": cid, "action": "settled", "reason": reason,
                    "ok": bool(res.get("success")), "error": res.get("error")})
    return out

