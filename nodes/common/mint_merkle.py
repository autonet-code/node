"""Merkle tree over an epoch's agent mint map.

Counterpart of ``Substrate.recordTrainingForEpoch``'s proof check:
mint amounts stop being self-reported — the anchor commits a root over
(agent address, scaled amount) leaves, and each agent proves its own
leaf when claiming.

Encoding (must match the contract byte-for-byte):

  leaf  = keccak256(bytes.concat(keccak256(abi.encode(address, uint256))))
          (double-hashed, OpenZeppelin convention, so a 64-byte leaf
          preimage can never be confused with an interior node)
  node  = keccak256(sorted_pair(a, b))   — ascending byte order, so
          proofs carry no left/right index bits
  odd levels promote the unpaired node unchanged
  single-leaf tree: root == leaf, proof == []

Scaling: the same ``int(raw_float * mint_scale)`` truncation the
agent-side submitter applies (authoritative_submitter, default 1e6).
Both sides MUST route through ``scale_mint`` here — a one-ulp float
disagreement means an unprovable leaf.

Only keys that parse as 0x addresses enter the tree: the contract can
only verify msg.sender, and non-address agent ids could never claim
on-chain anyway. Zero shares are excluded (the contract treats zero
as a no-op without a proof).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from eth_abi import encode as abi_encode
from eth_utils import is_address, keccak, to_checksum_address

logger = logging.getLogger(__name__)

DEFAULT_MINT_SCALE = 1_000_000
EMPTY_ROOT = b"\x00" * 32


def scale_mint(raw: float, mint_scale: int = DEFAULT_MINT_SCALE) -> int:
    """The canonical float→int truncation. Identical to the submitter's
    ``int(raw_mint * mint_scale)``."""
    return int(float(raw) * mint_scale)


def mint_leaves(
    agent_mint: Dict[str, float],
    mint_scale: int = DEFAULT_MINT_SCALE,
) -> List[Tuple[str, int]]:
    """Address-keyed, nonzero, deterministically-ordered leaf entries.

    Returns [(checksum_address, scaled_amount)] sorted by lowercase
    address. Non-address keys (test agents, pubkey-hex authors) are
    skipped — they have no on-chain claim path.
    """
    entries: List[Tuple[str, int]] = []
    for key, raw in agent_mint.items():
        if not is_address(str(key)):
            continue
        scaled = scale_mint(float(raw), mint_scale)
        if scaled <= 0:
            continue
        entries.append((to_checksum_address(str(key)), scaled))
    entries.sort(key=lambda e: e[0].lower())
    return entries


def _leaf_hash(address: str, amount: int) -> bytes:
    return keccak(keccak(abi_encode(["address", "uint256"], [address, amount])))


def _hash_pair(a: bytes, b: bytes) -> bytes:
    return keccak(a + b) if a <= b else keccak(b + a)


def _build_levels(leaf_hashes: List[bytes]) -> List[List[bytes]]:
    """All tree levels, leaves first. Odd nodes promote unchanged."""
    levels = [list(leaf_hashes)]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        nxt: List[bytes] = []
        for i in range(0, len(prev) - 1, 2):
            nxt.append(_hash_pair(prev[i], prev[i + 1]))
        if len(prev) % 2 == 1:
            nxt.append(prev[-1])
        levels.append(nxt)
    return levels


def mint_merkle_root(
    agent_mint: Dict[str, float],
    mint_scale: int = DEFAULT_MINT_SCALE,
) -> bytes:
    """Root over the mint map. EMPTY_ROOT (32 zero bytes) when no
    claimable entries exist — the contract rejects claims against it.
    """
    entries = mint_leaves(agent_mint, mint_scale)
    if not entries:
        return EMPTY_ROOT
    levels = _build_levels([_leaf_hash(a, m) for a, m in entries])
    return levels[-1][0]


def mint_merkle_proof(
    agent_mint: Dict[str, float],
    address: str,
    mint_scale: int = DEFAULT_MINT_SCALE,
) -> Optional[List[bytes]]:
    """Proof for one agent's leaf, or None when the agent has no
    claimable share. A single-leaf tree yields an empty proof.
    """
    entries = mint_leaves(agent_mint, mint_scale)
    target = None
    for i, (addr, _amt) in enumerate(entries):
        if addr.lower() == str(address).lower():
            target = i
            break
    if target is None:
        return None

    levels = _build_levels([_leaf_hash(a, m) for a, m in entries])
    proof: List[bytes] = []
    idx = target
    for level in levels[:-1]:
        sibling = idx + 1 if idx % 2 == 0 else idx - 1
        if sibling < len(level):
            proof.append(level[sibling])
        # else: odd node promoted unchanged — no proof element.
        idx //= 2
    return proof


def verify_mint_proof(
    root: bytes,
    address: str,
    amount: int,
    proof: List[bytes],
) -> bool:
    """Python mirror of the contract's check, for tests and pre-flight
    validation before spending gas."""
    computed = _leaf_hash(to_checksum_address(address), amount)
    for p in proof:
        computed = _hash_pair(computed, p)
    return computed == root
