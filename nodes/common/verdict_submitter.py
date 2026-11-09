"""Open and settle ToolBond's malware verdicts from anchored closes
(Decision 2026-11-09, docs/tool_substrate.md).

Each close commits, in the anchor's agentMintRoot, one leaf per tool that
concluded in it with the tool's malware share in bps (the close is the one
definition of the share). After the anchor lands, someone has to show
ToolBond each leaf: ``ToolBond.conclude(digest, anchorIndex, shareBps,
proof)`` records the conclusion and, when more than half the reviewing REP
said malware, opens the appeal window. Once the window (and any registered
appeal) has passed, or the DAO has ruled, ``settleMalware(digest)``
applies the verdict. Both are permissionless and idempotent on chain; every
daemon that claims a household's mint does them too (the operator pays
the gas), so they never depend on one daemon.

``conclude_epoch`` runs once per anchored epoch, from the claim path;
``settle_due`` on every tick, for the verdicts this daemon opened or saw
open (kept in a small JSON file so a restart remembers them).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .authoritative_encoding import cid_for_blob, decode_agent_mint_blob, decode_concluded
from .mint_merkle import concluded_proof, verify_concluded_proof

logger = logging.getLogger(__name__)

_BPS = 10_000
# How far back from the latest anchor to look for an epoch's index.
_ANCHOR_SCAN = 64


def _fn(name, inputs, outputs=(), mutability="nonpayable"):
    return {"name": name, "type": "function", "stateMutability": mutability,
            "inputs": [{"name": n, "type": t} for n, t in inputs],
            "outputs": [{"name": n, "type": t} for n, t in outputs]}


TOOL_BOND_VERDICT_ABI = [
    _fn("conclude", [("digest", "bytes32"), ("anchorIndex", "uint256"),
                     ("shareBps", "uint256"), ("proof", "bytes32[]")]),
    _fn("settleMalware", [("digest", "bytes32")]),
    _fn("verdicts", [("", "bytes32")],
        [("concludedAt", "uint48"), ("appealEnds", "uint48"), ("heldUntil", "uint48"),
         ("ruling", "uint8"), ("settled", "bool"), ("share", "uint64"),
         ("yesMass", "uint256"), ("bondPool", "uint256")], "view"),
]

_ANCHOR_COMPONENTS = [
    {"name": "epochId", "type": "string"}, {"name": "epochRoot", "type": "bytes32"},
    {"name": "prevEpochRoot", "type": "bytes32"}, {"name": "prevAnchorHash", "type": "bytes32"},
    {"name": "agentMintCid", "type": "string"}, {"name": "payloadHash", "type": "bytes32"},
    {"name": "submitter", "type": "address"}, {"name": "blockNumber", "type": "uint256"},
    {"name": "timestamp", "type": "uint256"}, {"name": "agentMintRoot", "type": "bytes32"},
]


class VerdictSubmitter:
    """Concludes and settles verdicts with one sender key. ``send(fn)``
    signs and sends a contract call, returning ``{"success", "error"?}``;
    tests and the driver supply it."""

    def __init__(self, *, w3: Any, tool_bond_address: str, substrate: Any,
                 resolver: Any, send: Callable[[Any], Dict[str, Any]],
                 state_path: Optional[Path] = None) -> None:
        self._w3 = w3
        self._bond = w3.eth.contract(address=w3.to_checksum_address(tool_bond_address),
                                     abi=TOOL_BOND_VERDICT_ABI)
        self._substrate = substrate
        self._resolver = resolver
        self._send = send
        self._path = Path(state_path) if state_path else None
        self._lock = threading.Lock()
        self._open: Dict[str, int] = self._load()   # digest -> share bps

    # -- persistence ---------------------------------------------------------
    def _load(self) -> Dict[str, int]:
        if self._path is None or not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return {str(k): int(v) for k, v in (data or {}).items()}
        except (OSError, ValueError, TypeError):
            logger.warning("verdict state %s unreadable; starting empty", self._path)
            return {}

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(sorted(self._open.items()))), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as e:
            logger.warning("failed to persist verdict state: %s", e)

    @property
    def open_verdicts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._open)

    # -- conclude ------------------------------------------------------------
    def _anchor_index(self, epoch_id: str) -> Optional[int]:
        count = int(self._substrate.functions.anchorCount().call())
        for i in range(count - 1, max(-1, count - 1 - _ANCHOR_SCAN), -1):
            if str(self._substrate.functions.getAnchor(i).call()[0]) == str(epoch_id):
                return i
        return None

    def conclude_epoch(self, epoch_id: str) -> List[Dict[str, Any]]:
        """Show ToolBond every concluded leaf of ``epoch_id``'s anchor.
        Returns one row per tool. Never raises."""
        try:
            anchor = self._substrate.functions.getAnchorByEpochId(epoch_id).call()
            blob = self._resolver.get(anchor[4]) if anchor[4] else None
            if blob is None or cid_for_blob(blob) != anchor[4]:
                return []
            concluded = decode_concluded(blob)
            if not concluded:
                return []
            mint = decode_agent_mint_blob(blob)
            index = self._anchor_index(epoch_id)
            if index is None:
                logger.warning("epoch %s: anchor index not found; verdicts not opened",
                               epoch_id)
                return []
            root = bytes(anchor[9])
        except Exception as e:  # noqa: BLE001
            logger.warning("epoch %s: reading the concluded tools failed: %s", epoch_id, e)
            return []
        rows: List[Dict[str, Any]] = []
        for digest, bps in sorted(concluded.items()):
            row: Dict[str, Any] = {"digest": digest, "share_bps": bps}
            try:
                proof = concluded_proof(mint, concluded, digest) or []
                if not verify_concluded_proof(root, digest, bps, proof):
                    row["error"] = "the concluded leaf does not prove against the anchor"
                    rows.append(row)
                    continue
                d = bytes.fromhex(digest)
                v = self._bond.functions.verdicts(d).call()
                if int(v[0]) == 0:
                    res = self._send(self._bond.functions.conclude(
                        d, int(index), int(bps), [bytes(p) for p in proof]))
                    row["concluded"] = bool(res.get("success"))
                    if not res.get("success"):
                        if int(self._bond.functions.verdicts(d).call()[0]) != 0:
                            # Another daemon's conclude landed first: every
                            # operator shows ToolBond the same leaves.
                            row["concluded"] = "already"
                        else:
                            row["error"] = res.get("error", "conclude failed")
                else:
                    row["concluded"] = "already"
                if bps * 2 > _BPS:
                    with self._lock:
                        self._open[digest] = int(bps)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            rows.append(row)
        self._save()
        return rows

    # -- settle --------------------------------------------------------------
    def settle_due(self, now: Optional[int] = None) -> List[str]:
        """Settle every open malware verdict whose window and holds have
        passed (or that the DAO ruled). Returns the digests settled."""
        settled: List[str] = []
        if now is None:
            try:
                now = int(self._w3.eth.get_block("latest")["timestamp"])
            except Exception:  # noqa: BLE001
                return settled
        for digest in sorted(self.open_verdicts):
            try:
                d = bytes.fromhex(digest)
                concluded_at, appeal_ends, held_until, ruling, done = (
                    self._bond.functions.verdicts(d).call()[:5])
                if done or int(concluded_at) == 0:
                    if done:
                        with self._lock:
                            self._open.pop(digest, None)
                    continue
                if int(ruling) == 0 and (now <= int(appeal_ends) or now <= int(held_until)):
                    continue
                res = self._send(self._bond.functions.settleMalware(d))
                if res.get("success"):
                    settled.append(digest)
                    with self._lock:
                        self._open.pop(digest, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("settling the verdict on %s failed: %s", digest[:12], e)
        self._save()
        return settled
