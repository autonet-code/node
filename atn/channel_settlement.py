"""Provider-side settlement of payment channels (ServiceMarket.sol).

A buyer's channel with a provider is its credit with that provider, used
across all of the provider's services. A provider daemon keeps each
channel's best voucher and what it has served against it (``consumed``)
as it serves (``ServiceStore.accept_voucher``, ``consume``) and cashes it
in with ``settle``, which pays up to what was served (``upTo``) over what
the channel has paid so far (the contract's ``claimed``) and leaves the
channel open, so the buyer's credit carries on and the service fee burns
as the provider earns. Vouchers pay ahead, so the best voucher can cover
more than was served: that part stays the buyer's, and comes back to it
when the channel closes. What a channel owes the provider (its unsettled
amount) is min(served, best voucher, deposit) minus ``claimed``. The sweep

- settles an open channel once its unsettled amount reaches
  ``settle_min`` (default 1 ATN, ``autonet.channel_settle_min``);
- settles an open channel with any unsettled amount once ``settle_period_s``
  has passed since it was last settled, or since the sweep first saw it
  (about when it opened: default a day, ``autonet.channel_idle_close_s``;
  <= 0 never settles on time alone), so small earnings do not wait forever;
- settles a channel whose client has started a close (``startClose``)
  up to what was served, inside the window: the provider has one challenge
  window to collect, after which the client takes back everything not yet
  paid. ``settle`` rather than ``closeChannel``: settle claims only what
  was served, while closeChannel would claim the whole voucher, including
  the part paid ahead, which is the buyer's.

Nothing is sent for a channel with nothing new to pay (``settle`` would
revert and burn gas): what has been paid is read from the channel
(``claimed``) at each sweep and kept, with the time of the last settle, in
the sweep's own ``SettlementLog``. A channel already closed or settled is
forgotten, and so is one whose client-close window has passed (nothing can
be collected any more; sending then only burns gas on a revert) or whose
client closes it with nothing served left unpaid. An open channel with a
request or session still being served is left for the next sweep: what it
has served is about to change.

Keys: the channel's provider is the operator (the owner surface's and the
external agents' services) or a native agent (its own key);
``key_for(provider)`` resolves it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

OPEN, CLOSING, SETTLED, CLIENT_CLOSING = 1, 2, 3, 4

# 1 ATN (6 decimals): settle once this much is unpaid on a channel.
DEFAULT_SETTLE_MIN = 1_000_000
# Settle whatever is unpaid at least this often (the default of
# ``autonet.channel_idle_close_s``).
DEFAULT_SETTLE_PERIOD_S = 86400.0


class SettlementLog:
    """Per channel, what the sweep has seen paid (``paid``, the contract's
    ``claimed``) and since when (``since``: the last settle, or when the
    sweep first saw the channel). In memory, or kept in ``path`` so a
    restart does not reset the settle period."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {str(k): dict(v) for k, v in loaded.items()
                                  if isinstance(v, dict)}
            except (OSError, ValueError):
                log.warning("channel settlements: %s unreadable, starting empty",
                            self.path)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def get(self, channel_id: str | int) -> Optional[dict[str, Any]]:
        with self._lock:
            entry = self._data.get(str(channel_id))
            return dict(entry) if entry else None

    def note(self, channel_id: str | int, paid: int, since: float) -> None:
        with self._lock:
            self._data[str(channel_id)] = {"paid": int(paid), "since": float(since)}
            self._save()

    def forget(self, channel_id: str | int) -> None:
        with self._lock:
            if self._data.pop(str(channel_id), None) is not None:
                self._save()

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}


def _paid(ch: dict[str, Any], logged: Optional[dict[str, Any]]) -> int:
    """What the channel has paid the provider: the chain's ``claimed``, or
    what the last settle here paid if the read lags behind it."""
    try:
        claimed = int(str(ch.get("claimed") or "0"))
    except (TypeError, ValueError):
        claimed = 0
    return max(claimed, int((logged or {}).get("paid") or 0))


async def settle_channels(
    store: Any, client: Any, key_for: Callable[[str], Optional[str]], *,
    settle_min: int = DEFAULT_SETTLE_MIN,
    settle_period_s: float = DEFAULT_SETTLE_PERIOD_S,
    ledger: Optional[SettlementLog] = None,
    now: Optional[float] = None,
) -> list[dict[str, Any]]:
    """One sweep. ``client`` is a ServiceMarketClient; ``key_for(provider)``
    returns the provider's private key or None; ``ledger`` keeps what was
    paid and when (one per daemon; in memory if not given). Returns what
    was done: per channel acted on, ``{channel_id, action, ...}`` with
    action ``settled`` (by settle, reason ``threshold``, ``period`` or
    ``client_closing``; ``amount`` what it owed), ``forgot`` or
    ``window_passed``."""
    out: list[dict[str, Any]] = []
    ledger = ledger if ledger is not None else SettlementLog()
    clock = time.time() if now is None else float(now)
    vouchers = store.channel_vouchers()
    for cid in list(ledger.all()):
        if cid not in vouchers:
            ledger.forget(cid)
    for cid, voucher in sorted(vouchers.items()):
        ch = await client.get_channel(int(cid))
        if ch is None:
            continue
        status = int(ch["status"])
        if status in (CLOSING, SETTLED):
            store.forget_channel_voucher(cid)
            ledger.forget(cid)
            out.append({"channel_id": cid, "action": "forgot"})
            continue
        cumulative = int(voucher["cumulative"])
        # Only what was served is claimed: the rest of a voucher paid ahead
        # stays the buyer's.
        served = min(int(voucher.get("consumed") or 0), cumulative)
        deposit = int(str(ch.get("deposit") or "0"))
        logged = ledger.get(cid)
        paid = _paid(ch, logged)
        unsettled = min(served, deposit) - paid
        if status == CLIENT_CLOSING:
            if clock >= float(ch.get("closes_at") or 0):
                # The window passed: the client takes back what was not
                # paid and closeChannel would revert. Nothing to collect.
                if unsettled > 0:
                    log.warning("channel %s: the client's close window passed "
                                "before it was settled; %s unpaid is lost",
                                cid, unsettled)
                store.forget_channel_voucher(cid)
                ledger.forget(cid)
                out.append({"channel_id": cid, "action": "window_passed"})
                continue
            if unsettled <= 0:
                # Everything served is paid: the rest is the client's.
                store.forget_channel_voucher(cid)
                ledger.forget(cid)
                out.append({"channel_id": cid, "action": "forgot"})
                continue
            action, reason = "settled", "client_closing"
        elif status == OPEN:
            if logged is None or int(logged.get("paid") or 0) != paid:
                # First sight, or a payout this sweep did not send: the
                # period starts now.
                ledger.note(cid, paid, clock)
                logged = ledger.get(cid)
            if getattr(store, "has_pending_voucher", lambda c: False)(cid):
                continue
            if unsettled <= 0:
                continue
            since = float((logged or {}).get("since") or clock)
            if settle_min > 0 and unsettled >= settle_min:
                reason = "threshold"
            elif settle_period_s > 0 and clock - since >= settle_period_s:
                reason = "period"
            else:
                continue
            action = "settled"
        else:
            continue
        key = key_for(str(ch["provider"]))
        if not key:
            log.warning("channel %s is due to be collected (%s) but no key is "
                        "held for its provider %s", cid, reason, ch["provider"])
            continue
        res = await client.settle_channel(
            key, int(cid), cumulative, str(voucher["signature"]), up_to=served)
        ok = bool(res.get("success"))
        if ok and reason == "client_closing":
            # Everything served is paid; the client withdraws the rest when
            # its window ends. Nothing more can come on this channel.
            store.forget_channel_voucher(cid)
            ledger.forget(cid)
            log.info("settled closing channel %s up to %s", cid, served)
        elif ok:
            ledger.note(cid, paid + unsettled, clock)
            log.info("settled channel %s: %s more, %s paid in all (%s)",
                     cid, unsettled, paid + unsettled, reason)
        else:
            log.warning("collecting channel %s failed: %s", cid, res.get("error"))
        out.append({"channel_id": cid, "action": action, "reason": reason,
                    "amount": unsettled, "ok": ok, "error": res.get("error")})
    return out
