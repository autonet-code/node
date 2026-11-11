"""A service's price list (docs/services_market.md, Decision 2026-11-12).

Prices are off chain. The provider signs a price list for each of its
services with the key that registered it, and gossips it with the spec;
search shows it, and the provider's daemon quotes it when a buyer opens a
session (the quoted prices hold for the whole session). A new list replaces
the old by its sequence number: changing prices takes no transaction and no
new spec.

A list holds:

- ``service``: the spec digest it prices;
- ``provider``: the address that signs it (the service's registered
  provider, the one paid);
- ``prices``: ``{measure: amount}`` in ATN base units (6 decimals), one per
  measure the spec charges for, each for the spec's quantity of that
  measure (a request service: ``{"requests": amount}`` per request);
- ``voucher``: how much a buyer pays ahead with each voucher, the most it
  has at risk;
- ``seq`` and ``ts``.

Charging rounds up per measure: a session that used ``u`` of a measure
priced ``p`` per ``q`` owes ``ceil(u * p / q)`` for it, so a running charge
only ever grows (``Meter``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .service_spec import MEASURE_REQUESTS, spec_measures

PRICE_LIST_KIND = "service_prices"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SIG_FIELD = "sig"
_FIELDS = frozenset({"kind", "service", "provider", "prices", "voucher", "seq",
                     "ts", _SIG_FIELD})


def _norm_digest(d: Any) -> str:
    return str(d or "").strip().lower().removeprefix("0x")


def build_price_list(*, service: str, provider: str, prices: Dict[str, Any],
                     voucher: int, seq: int, ts: int) -> Dict[str, Any]:
    return {
        "kind": PRICE_LIST_KIND,
        "service": _norm_digest(service),
        "provider": str(provider or ""),
        "prices": {str(k): int(v) for k, v in (prices or {}).items()},
        "voucher": int(voucher),
        "seq": int(seq),
        "ts": int(ts),
    }


def validate_price_list(pl: Any, spec: Optional[Dict[str, Any]] = None) -> List[str]:
    """Problems with a price list (empty = valid); against ``spec`` when
    given, the prices must name exactly the measures it charges for."""
    if not isinstance(pl, dict) or pl.get("kind") != PRICE_LIST_KIND:
        return [f"a price list has kind {PRICE_LIST_KIND!r}"]
    errors: List[str] = []
    extra = sorted(set(pl) - _FIELDS)
    if extra:
        errors.append(f"price list has unknown fields {extra}")
    if not _HEX64.match(_norm_digest(pl.get("service"))):
        errors.append("price list names its service by spec digest")
    prices = pl.get("prices")
    if not isinstance(prices, dict) or not prices:
        errors.append("price list needs prices")
    else:
        for k, v in prices.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                errors.append(f"price for {k!r} must be ATN base units (a whole number >= 0)")
    for field in ("voucher", "seq", "ts"):
        v = pl.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            errors.append(f"{field} must be a whole number >= 0")
    if spec is not None and isinstance(prices, dict):
        want = {m.get("measure") for m in spec_measures(spec)
                if isinstance(m, dict) and isinstance(m.get("measure"), str)}
        if set(prices) != want:
            errors.append(f"prices must name exactly {sorted(want)}")
    voucher = pl.get("voucher")
    priced = isinstance(prices, dict) and any(
        isinstance(v, int) and v > 0 for v in prices.values())
    if priced and isinstance(voucher, int) and voucher <= 0:
        errors.append("a priced service needs a voucher size above 0")
    return errors


def canonical_price_bytes(pl: Dict[str, Any]) -> bytes:
    body = {k: v for k, v in pl.items() if k != _SIG_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_price_list(pl: Dict[str, Any], key: str) -> Dict[str, Any]:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    signed = Account.sign_message(encode_defunct(canonical_price_bytes(pl)),
                                  private_key=key)
    out = dict(pl)
    out[_SIG_FIELD] = signed.signature.hex()
    return out


def price_list_signer(pl: Dict[str, Any]) -> str:
    sig = str(pl.get(_SIG_FIELD) or "")
    if not sig:
        return ""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        return str(Account.recover_message(
            encode_defunct(canonical_price_bytes(pl)),
            signature=sig if sig.startswith("0x") else "0x" + sig))
    except Exception:  # noqa: BLE001
        return ""


def signed_by_provider(pl: Dict[str, Any]) -> bool:
    provider = str(pl.get("provider") or "").lower()
    return bool(provider) and price_list_signer(pl).lower() == provider


def request_price(pl: Optional[Dict[str, Any]]) -> int:
    """What one request costs under ``pl`` (0 when unpriced)."""
    if not isinstance(pl, dict):
        return 0
    try:
        return int((pl.get("prices") or {}).get(MEASURE_REQUESTS) or 0)
    except (TypeError, ValueError):
        return 0


def voucher_size(pl: Optional[Dict[str, Any]]) -> int:
    if not isinstance(pl, dict):
        return 0
    try:
        return int(pl.get("voucher") or 0)
    except (TypeError, ValueError):
        return 0


def charge_for(usage: int, price: int, quantity: int) -> int:
    """What ``usage`` of a measure priced ``price`` per ``quantity`` costs,
    rounded up."""
    usage, price, quantity = int(usage), int(price), max(1, int(quantity))
    if usage <= 0 or price <= 0:
        return 0
    return -(-usage * price // quantity)


class Meter:
    """A session's running charge at its quoted prices: usage per measure
    in, the charge so far out (it only grows)."""

    def __init__(self, spec: Dict[str, Any], prices: Dict[str, Any]) -> None:
        self.quantity = {m["measure"]: int(m["quantity"]) for m in spec_measures(spec)}
        self.prices = {k: int(v) for k, v in (prices or {}).items()
                       if k in self.quantity}
        self.usage: Dict[str, int] = {k: 0 for k in self.quantity}
        self.charged = 0

    def charges(self, measure: str) -> bool:
        return self.prices.get(measure, 0) > 0

    def add(self, measure: str, amount: int) -> int:
        """Record ``amount`` more of ``measure``; returns how much the
        charge grew (ATN base units)."""
        if measure not in self.usage or int(amount) <= 0:
            return 0
        self.usage[measure] += int(amount)
        total = sum(charge_for(self.usage[m], self.prices.get(m, 0), self.quantity[m])
                    for m in self.usage)
        delta = total - self.charged
        self.charged = total
        return max(0, delta)
