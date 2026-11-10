"""DAO sanctions as the daemon sees them now (docs/accountability.md).

A DAO proposal writes ``sanction.<household 0x>`` into its Registry (JSON
``{"until": <unix seconds>, "reason": <uri>, ...}``; empty = lifted). The
federated close reads the list pinned to the previous anchor and withholds
mint (nodes/common/voice_state.py). This module is the live read for
everything else: agents may not buy from, adopt the tools of, or review on
behalf of a sanctioned household, and the app badges it.

The key names a household (an owner wallet). An agent address is checked
against its own entry and its owner's, so a fresh agent key under a
sanctioned owner is still sanctioned, and so is an agent the DAO named
directly.

Who owns which agent comes from ``OwnerMap``: Substrate's ``OwnerBound``
logs, followed incrementally (one scan from the jurisdiction's deploy
block, then only the new blocks, at most once a minute), so a rotation to
another owner is seen within a minute and no address costs a chain read of
its own (Decision 2026-11-10: search shows each listing's owner and
sanction from the same map).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

_REFRESH_S = 60.0
_ZERO = "0x" + "0" * 40

_REGISTRY_ABI = [
    {"inputs": [], "name": "getAllKeys",
     "outputs": [{"internalType": "string[]", "name": "", "type": "string[]"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getAllValues",
     "outputs": [{"internalType": "string[]", "name": "", "type": "string[]"}],
     "stateMutability": "view", "type": "function"},
]
def _chain_owner_logs(rpc_url: str, substrate: str,
                      deploy_block: int | None = None):
    """``read(from_block) -> (bindings, last_block)``: Substrate's
    ``OwnerBound`` logs from ``from_block`` (the deploy block, searched when
    not configured, on the first read) to the latest block, each as
    ``{agent, owner, prev_owner, block, index}``. Blocking: callers run it
    off the event loop."""
    def read(from_block: int | None) -> tuple[list[dict[str, Any]], int]:
        from web3 import Web3
        from nodes.common.voice_state import _VOICE_ABI, _get_logs
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        address = Web3.to_checksum_address(substrate)
        c = w3.eth.contract(address=address, abi=_VOICE_ABI)
        latest = int(w3.eth.block_number)
        start = from_block if from_block is not None else deploy_block
        if start is not None and start > latest:
            return [], latest
        logs = _get_logs(c.events.OwnerBound, w3, address, latest, start)
        return [{"agent": str(lg["args"]["agent"]).lower(),
                 "owner": str(lg["args"]["owner"]).lower(),
                 "prev_owner": str(lg["args"]["prevOwner"]).lower(),
                 "block": int(lg["blockNumber"]),
                 "index": int(lg["logIndex"])} for lg in logs], latest
    return read


class OwnerMap:
    """Every agent's current owner and the owners it was rotated away from,
    followed from ``OwnerBound`` logs: the first sync reads them all, later
    syncs only the blocks since, at most once per ``refresh_s``. A failed
    read keeps the map it has."""

    def __init__(self, read_logs: Callable[[int | None],
                                           tuple[list[dict[str, Any]], int]],
                 *, clock: Callable[[], float] = time.time,
                 refresh_s: float = _REFRESH_S) -> None:
        self._read = read_logs
        self._clock = clock
        self._refresh_s = refresh_s
        self._owner: dict[str, str] = {}
        self._past: dict[str, list[str]] = {}
        self._last_block: int | None = None
        self._synced_at: float | None = None
        self._lock = threading.Lock()

    def sync(self) -> None:
        with self._lock:
            now = self._clock()
            if self._synced_at is not None and now - self._synced_at < self._refresh_s:
                return
            self._synced_at = now
            start = None if self._last_block is None else self._last_block + 1
            try:
                bindings, last = self._read(start)
            except Exception as exc:  # noqa: BLE001
                log.warning("owner bindings read failed, keeping the last map: %s", exc)
                return
            for b in sorted(bindings, key=lambda b: (b["block"], b["index"])):
                agent = b["agent"]
                owner = "" if b["owner"] == _ZERO else b["owner"]
                self._owner[agent] = owner
                prev = b.get("prev_owner") or ""
                if prev and prev != _ZERO and prev not in self._past.setdefault(agent, []):
                    self._past[agent].append(prev)
            self._last_block = last

    def owner(self, agent: str) -> str:
        self.sync()
        return self._owner.get(str(agent or "").lower(), "")

    def past(self, agent: str) -> list[str]:
        self.sync()
        return list(self._past.get(str(agent or "").lower(), []))


def _chain_reader(rpc_url: str, dao_registry: str) -> Callable[[], dict[str, str]]:
    def read() -> dict[str, str]:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        reg = w3.eth.contract(address=Web3.to_checksum_address(dao_registry),
                              abi=_REGISTRY_ABI)
        keys = reg.functions.getAllKeys().call()
        values = reg.functions.getAllValues().call()
        return dict(zip(keys, values))
    return read


class SanctionList:
    """The jurisdiction's sanctions in force, refreshed at most once a
    minute. A failed read keeps the last good list (and logs once)."""

    def __init__(self, read_registry: Callable[[], dict[str, str]] | None,
                 owner_of: Callable[[str], str] | None = None, *,
                 past_owners: Callable[[str], list[str]] | None = None,
                 owners: OwnerMap | None = None,
                 clock: Callable[[], float] = time.time,
                 refresh_s: float = _REFRESH_S) -> None:
        self._read = read_registry
        # ``owners`` (the OwnerBound map) is what a daemon uses; the
        # per-address callables remain for tests and callers without logs.
        self._owners_map = owners
        self._owner_of = owner_of
        self._past_owners = past_owners
        self._history: dict[str, list[str]] = {}
        self._history_failed: dict[str, float] = {}
        self._clock = clock
        self._refresh_s = refresh_s
        self._raw: dict[str, dict[str, Any]] = {}
        self._read_at: float | None = None
        # Per-address answers (callable path only) are read again after
        # ``refresh_s``: an agent bound or rotated since must not keep the
        # old answer.
        self._owners: dict[str, str] = {}
        self._owner_read_at: dict[str, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, rpb: Any) -> "SanctionList":
        rpc = str(getattr(rpb, "rpc_url", "") or "")
        registry = str(getattr(rpb, "dao_registry_address", "") or "")
        substrate = str(getattr(rpb, "substrate_address", "") or "")
        deploy_block = int(getattr(rpb, "deploy_block", 0) or 0) or None
        if not rpc or not registry:
            return cls(None)
        return cls(_chain_reader(rpc, registry),
                   owners=(OwnerMap(_chain_owner_logs(rpc, substrate, deploy_block))
                           if substrate else None))

    def _refresh(self) -> None:
        if self._read is None:
            return
        now = self._clock()
        if self._read_at is not None and now - self._read_at < self._refresh_s:
            return
        from nodes.common.voice_state import SANCTION_KEY_PREFIX, parse_sanction
        try:
            values = self._read()
        except Exception as exc:
            if self._read_at is None or self._raw:
                log.warning("sanction list read failed, keeping the last one: %s", exc)
            self._read_at = now
            return
        raw: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            if not str(key).startswith(SANCTION_KEY_PREFIX):
                continue
            addr = str(key)[len(SANCTION_KEY_PREFIX):].strip().lower()
            entry = parse_sanction(value)
            if addr and entry is not None:
                raw[addr] = entry
        self._raw = raw
        self._read_at = now

    def entries(self) -> dict[str, dict[str, Any]]:
        """Sanctions in force now: address -> entry (with ``until`` when it
        has an end)."""
        from nodes.common.voice_state import sanction_in_force
        with self._lock:
            self._refresh()
            now = int(self._clock())
            return {a: dict(e) for a, e in sorted(self._raw.items())
                    if sanction_in_force(e, now)}

    def owner(self, address: str) -> str:
        """The owner wallet (household) ``address`` is bound to now, or ""."""
        return self._owner(str(address or "").strip().lower())

    def _owner(self, address: str) -> str:
        if self._owners_map is not None:
            return self._owners_map.owner(address)
        if self._owner_of is None:
            return ""
        read_at = self._owner_read_at.get(address)
        if read_at is None or self._clock() - read_at >= self._refresh_s:
            try:
                owner = self._owner_of(address) or ""
            except Exception:
                return self._owners.get(address, "")
            self._owners[address] = "" if owner.lower() == _ZERO else owner.lower()
            self._owner_read_at[address] = self._clock()
        return self._owners[address]

    def check(self, address: str) -> dict[str, Any] | None:
        """The sanction covering ``address`` (an agent or a household), with
        ``household`` set to the address it was recorded under; None when
        clear."""
        addr = str(address or "").strip().lower()
        if not addr:
            return None
        active = self.entries()
        if not active:
            return None
        if addr in active:
            return {**active[addr], "household": addr}
        owner = self._owner(addr)
        if owner and owner in active:
            return {**active[owner], "household": owner}
        # Rotating an agent to a fresh wallet does not lift its sanction.
        for prev in self._past(addr):
            if prev in active:
                return {**active[prev], "household": prev}
        return None

    def _past(self, address: str) -> list[str]:
        if self._owners_map is not None:
            return self._owners_map.past(address)
        if self._past_owners is None:
            return []
        if address not in self._history:
            # A failed scan is retried at most once per refresh period, not
            # on every check (each one is a full log scan).
            failed = self._history_failed.get(address)
            if failed is not None and self._clock() - failed < self._refresh_s:
                return []
            try:
                self._history[address] = [
                    str(p).lower() for p in self._past_owners(address)]
            except Exception as exc:
                log.warning("owner history read for %s failed: %s", address, exc)
                self._history_failed[address] = self._clock()
                return []
            self._history_failed.pop(address, None)
        return self._history[address]


def listing_context(runtime: Any, address: str) -> dict[str, Any]:
    """What a search row shows about a listing's author or provider: the
    household it belongs to now and the sanction covering it, if any
    (``{household, sanction}``). Read from the daemon's sanction list and
    its ``OwnerMap``, so a row costs no chain read of its own. Blocking on
    a refresh: callers run it off the event loop."""
    out: dict[str, Any] = {"household": "", "sanction": None}
    addr = str(address or "").strip().lower()
    bridge = getattr(runtime, "autonet", None)
    try:
        sl = getattr(bridge, "sanctions", None) if bridge is not None else None
    except Exception:  # noqa: BLE001 - no chain config
        sl = None
    if sl is None or not addr.startswith("0x"):
        return out
    try:
        out["household"] = sl.owner(addr) or ""
        out["sanction"] = sl.check(addr)
    except Exception:  # noqa: BLE001 - context is additive
        pass
    return out


def refusal(address: str, entry: dict[str, Any], what: str) -> dict[str, Any]:
    """The error an agent tool returns when ``what`` involves a sanctioned
    household."""
    msg = f"{address} belongs to household {entry['household']}, which the DAO has sanctioned"
    if entry.get("until"):
        msg += f" until {time.strftime('%Y-%m-%d', time.gmtime(int(entry['until'])))}"
    if entry.get("reason"):
        msg += f" (reason: {entry['reason']})"
    return {"error": f"{msg}. Agents may not {what} a sanctioned household.",
            "sanction": entry}
