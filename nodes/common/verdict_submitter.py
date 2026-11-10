"""Show ToolBond the malware verdicts of anchored closes (Decisions
2026-11-09 and 2026-11-10, docs/tool_substrate.md).

Each close commits, in the anchor's agentMintRoot, one leaf per tool that
concluded in it with the tool's malware share in bps (the close is the one
definition of the share). After the anchor lands, someone has to show
ToolBond each leaf: ``ToolBond.conclude(digest, anchorIndex, shareBps,
proof)`` records the conclusion and, when more than half the reviewing REP
said malware, settles the verdict at once (the stake's share to the
treasury, the household's REP burned). There is no appeal window: the DAO
clears or designates a verdict by proposal at any time. The call is
permissionless and idempotent on chain; every daemon that claims a
household's mint makes it too (the operator pays the gas), so it never
depends on one daemon.

``conclude_epoch`` runs once per anchored epoch, from the claim path.
"""

from __future__ import annotations

import logging
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
    _fn("verdicts", [("", "bytes32")],
        [("concludedAt", "uint48"), ("malware", "uint8"), ("cleared", "bool"),
         ("share", "uint64"), ("repBurned", "uint256")], "view"),
]

_ANCHOR_COMPONENTS = [
    {"name": "epochId", "type": "string"}, {"name": "epochRoot", "type": "bytes32"},
    {"name": "prevEpochRoot", "type": "bytes32"}, {"name": "prevAnchorHash", "type": "bytes32"},
    {"name": "agentMintCid", "type": "string"}, {"name": "payloadHash", "type": "bytes32"},
    {"name": "submitter", "type": "address"}, {"name": "blockNumber", "type": "uint256"},
    {"name": "timestamp", "type": "uint256"}, {"name": "agentMintRoot", "type": "bytes32"},
]


class VerdictSubmitter:
    """Concludes verdicts with one sender key. ``send(fn)`` signs and sends a
    contract call, returning ``{"success", "error"?}``; tests and the driver
    supply it."""

    def __init__(self, *, w3: Any, tool_bond_address: str, substrate: Any,
                 resolver: Any, send: Callable[[Any], Dict[str, Any]]) -> None:
        self._w3 = w3
        self._bond = w3.eth.contract(address=w3.to_checksum_address(tool_bond_address),
                                     abi=TOOL_BOND_VERDICT_ABI)
        self._substrate = substrate
        self._resolver = resolver
        self._send = send

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
                row["malware"] = bps * 2 > _BPS
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            rows.append(row)
        return rows
