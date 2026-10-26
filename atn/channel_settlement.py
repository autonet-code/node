"""Provider-side settlement of payment channels (ServiceMarket.sol).

A provider daemon stores each channel's best voucher as it serves
(``ServiceStore.record_channel_voucher``) but nothing used to close the
channel, so the provider's earnings never landed and the client's deposit
stayed escrowed. This sweep settles every channel whose client has started a
close (``startClose``): the provider has one challenge window to settle with
its best voucher, after which the client takes the whole deposit back. A
channel already closed or settled is forgotten.

Keys: the channel's provider is the serving agent (its registered key) or
the owner wallet (the daemon's signing key, for owner-authored services).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

OPEN, CLOSING, SETTLED, CLIENT_CLOSING = 1, 2, 3, 4


async def settle_closing_channels(
    store: Any, client: Any, key_for: Callable[[str], Optional[str]],
) -> list[dict[str, Any]]:
    """One sweep. ``client`` is a ServiceMarketClient; ``key_for(provider)``
    returns the provider's private key or None. Returns what was done."""
    out: list[dict[str, Any]] = []
    for cid, voucher in sorted(store.channel_vouchers().items()):
        ch = await client.get_channel(int(cid))
        if ch is None:
            continue
        status = int(ch["status"])
        if status in (CLOSING, SETTLED):
            store.forget_channel_voucher(cid)
            out.append({"channel_id": cid, "action": "forgot"})
            continue
        if status != CLIENT_CLOSING:
            continue
        key = key_for(str(ch["provider"]))
        if not key:
            log.warning("channel %s is closing but no key for its provider %s",
                        cid, ch["provider"])
            continue
        res = await client.close_channel(
            key, int(cid), int(voucher["cumulative"]), str(voucher["signature"]))
        if res.get("success"):
            store.forget_channel_voucher(cid)
            log.info("settled channel %s at %s (client started a close)",
                     cid, voucher["cumulative"])
        else:
            log.warning("settling channel %s failed: %s", cid, res.get("error"))
        out.append({"channel_id": cid, "action": "settled",
                    "ok": bool(res.get("success"))})
    return out
