"""Nonces for a daemon's own transactions.

One key often sends from two places at once (an agent pays for a service
while the chain-submission driver records its mint), and RPCs behind a proxy
or a load balancer report the pending count late. Reading the count at each
send then hands the same nonce out twice and one transaction fails with
"nonce too low". ``next_nonce`` serializes the read per address and never
hands out a nonce below the last one it gave, unless that one is stale (its
transaction never reached the chain), so a failed send cannot leave a gap
forever.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Tuple

# A nonce handed out but not yet visible in the pending count is trusted for
# this long, and only while the chain lags it by a few; otherwise the chain's
# count wins (the send failed, or the chain was reset).
STALE_SECS = 30.0
MAX_LAG = 8

_guard = threading.Lock()
_locks: Dict[str, threading.Lock] = {}
_last: Dict[str, Tuple[int, float]] = {}


def _lock_for(address: str) -> threading.Lock:
    with _guard:
        lock = _locks.get(address)
        if lock is None:
            lock = _locks[address] = threading.Lock()
        return lock


def next_nonce(w3: Any, address: str) -> int:
    """The nonce for ``address``'s next transaction."""
    key = str(address).lower()
    with _lock_for(key):
        chain = int(w3.eth.get_transaction_count(address, "pending"))
        last = _last.get(key)
        nonce = chain
        if (last is not None and time.monotonic() - last[1] < STALE_SECS
                and 0 <= last[0] - chain < MAX_LAG):
            nonce = last[0] + 1
        _last[key] = (nonce, time.monotonic())
        return nonce


def reset() -> None:
    """Forget every nonce handed out (tests that restart their chain)."""
    with _guard:
        _last.clear()
