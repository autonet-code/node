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
_OWNER_ABI = [
    {"inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
     "name": "getAgentOwner",
     "outputs": [{"internalType": "address", "name": "", "type": "address"}],
     "stateMutability": "view", "type": "function"},
]


def _chain_past_owners(rpc_url: str, substrate: str,
                       deploy_block: int | None = None) -> Callable[[str], list[str]]:
    """Every owner an agent was rotated away from (OwnerBound prevOwner).

    The scan starts at the jurisdiction's deploy block (searched when not
    configured) and splits the range when the RPC caps eth_getLogs spans,
    the same way the close reads its logs (voice_state._get_logs). Callers
    run it off the event loop (asyncio.to_thread)."""
    def past_owners(agent: str) -> list[str]:
        from web3 import Web3
        from nodes.common.voice_state import _VOICE_ABI, _get_logs
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        address = Web3.to_checksum_address(substrate)
        c = w3.eth.contract(address=address, abi=_VOICE_ABI)
        event = c.events.OwnerBound
        filters = {"agent": Web3.to_checksum_address(agent)}

        class _Filtered:
            @staticmethod
            def get_logs(**kw):
                return event.get_logs(argument_filters=filters, **kw)

        logs = _get_logs(_Filtered, w3, address, int(w3.eth.block_number),
                         start_block=deploy_block)
        return [str(lg["args"]["prevOwner"]).lower() for lg in logs
                if str(lg["args"]["prevOwner"]).lower() != _ZERO]
    return past_owners


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


def _chain_owner_lookup(rpc_url: str, substrate: str) -> Callable[[str], str]:
    def owner_of(agent: str) -> str:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        c = w3.eth.contract(address=Web3.to_checksum_address(substrate),
                            abi=_OWNER_ABI)
        return str(c.functions.getAgentOwner(
            Web3.to_checksum_address(agent)).call()).lower()
    return owner_of


class SanctionList:
    """The jurisdiction's sanctions in force, refreshed at most once a
    minute. A failed read keeps the last good list (and logs once)."""

    def __init__(self, read_registry: Callable[[], dict[str, str]] | None,
                 owner_of: Callable[[str], str] | None = None, *,
                 past_owners: Callable[[str], list[str]] | None = None,
                 clock: Callable[[], float] = time.time,
                 refresh_s: float = _REFRESH_S) -> None:
        self._read = read_registry
        self._owner_of = owner_of
        self._past_owners = past_owners
        self._history: dict[str, list[str]] = {}
        self._history_failed: dict[str, float] = {}
        self._clock = clock
        self._refresh_s = refresh_s
        self._raw: dict[str, dict[str, Any]] = {}
        self._read_at: float | None = None
        self._owners: dict[str, str] = {}
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
                   _chain_owner_lookup(rpc, substrate) if substrate else None,
                   past_owners=(_chain_past_owners(rpc, substrate, deploy_block)
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

    def _owner(self, address: str) -> str:
        if self._owner_of is None:
            return ""
        if address not in self._owners:
            try:
                owner = self._owner_of(address) or ""
            except Exception:
                return ""
            self._owners[address] = "" if owner == _ZERO else owner.lower()
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
