"""Merkle tree over an epoch's claims.

Counterpart of ``Substrate.claim``'s proof check: claims are never
self-reported. The anchor's ``claimRoot`` commits one leaf per earner (a
household's owner wallet): the amount it takes of each token's pool (the
same fraction of every pool, the basket). The REP value is not in the
leaf: Substrate computes it on chain from the capped amounts at each
token's parity stored when the epoch is anchored (the close's ``value`` is
informational). Anyone may submit the claim; the leaf fixes the earner.

Encoding (must match the contract byte for byte)::

  leaf  = keccak256(bytes.concat(keccak256(abi.encode(
              CLAIM_TAG, address earner,
              address[] tokens, uint256[] amounts))))
          CLAIM_TAG = keccak256("autonet.claim"); tokens strictly
          ascending, one amount each (double-hashed, OpenZeppelin
          convention, so a leaf preimage can never be confused with an
          interior node)
  node  = keccak256(sorted_pair(a, b)): ascending byte order, so proofs
          carry no left/right index bits
  odd levels promote the unpaired node unchanged
  single-leaf tree: root == leaf, proof == []

Claims are ``{earner: {"tokens": {token: int}, "value": int}}`` (the
close's ``claims``). Only earners that parse as 0x addresses enter the
tree (nothing else can claim on chain), and only tokens with a non-zero
amount; an earner with no token has no leaf.

Concluded tools (Decision 2026-11-09) share the tree: after the claim
leaves, sorted by digest, one leaf per tool that concluded in the close,

  leaf = keccak256(bytes.concat(keccak256(abi.encode(
             keccak256("autonet.tool_concluded"), bytes32 digest,
             uint256 shareBps))))

which ``ToolBond.conclude`` proves to open the tool's malware verdict.
Each leaf type starts with its own tag hash, so the two cannot collide.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from eth_abi import encode as abi_encode
from eth_utils import is_address, keccak, to_checksum_address

logger = logging.getLogger(__name__)

EMPTY_ROOT = b"\x00" * 32
CLAIM_TAG = keccak(text="autonet.claim")
CONCLUDED_TAG = keccak(text="autonet.tool_concluded")

ClaimLeaf = Tuple[str, List[str], List[int]]


def claim_leaves(claims: Dict[str, Any]) -> List[ClaimLeaf]:
    """The claim leaves in tree order: ``[(earner checksum, tokens
    (checksum, ascending), amounts)]`` sorted by lowercase earner.
    Non-address earners and zero amounts are skipped; an earner left with
    no token has no leaf."""
    out: List[ClaimLeaf] = []
    for key in sorted(claims, key=lambda k: str(k).lower()):
        if not is_address(str(key)):
            continue
        entry = claims[key] or {}
        raw_tokens = entry.get("tokens") or {}
        pairs = sorted(
            (str(t).lower(), int(a)) for t, a in raw_tokens.items()
            if is_address(str(t)) and int(a) > 0)
        if not pairs:
            continue
        out.append((to_checksum_address(str(key)),
                    [to_checksum_address(t) for t, _ in pairs],
                    [a for _, a in pairs]))
    return out


def claim_leaf(earner: str, tokens: List[str], amounts: List[int]) -> bytes:
    """One earner's leaf (``Substrate.claimLeaf``). ``tokens`` must already
    be strictly ascending."""
    return keccak(keccak(abi_encode(
        ["bytes32", "address", "address[]", "uint256[]"],
        [CLAIM_TAG, to_checksum_address(earner),
         [to_checksum_address(t) for t in tokens], [int(a) for a in amounts]])))


def concluded_leaf(digest: str, share_bps: int) -> bytes:
    """A concluded tool's leaf (ToolBond.conclude)."""
    d = bytes.fromhex(str(digest).lower().removeprefix("0x"))
    return keccak(keccak(abi_encode(
        ["bytes32", "bytes32", "uint256"], [CONCLUDED_TAG, d, int(share_bps)])))


def _concluded_order(concluded: Optional[Dict[str, int]]) -> List[Tuple[str, int]]:
    norm = {str(k).lower().removeprefix("0x"): int(v) for k, v in (concluded or {}).items()}
    return [(d, norm[d]) for d in sorted(norm)]


def _all_leaves(claims: Dict[str, Any], concluded: Optional[Dict[str, int]]) -> List[bytes]:
    """The tree's leaves: claim leaves (by earner), then concluded leaves
    (by digest)."""
    leaves = [claim_leaf(*leaf) for leaf in claim_leaves(claims)]
    leaves.extend(concluded_leaf(d, s) for d, s in _concluded_order(concluded))
    return leaves


def _proof_at(leaves: List[bytes], target: int) -> List[bytes]:
    levels = _build_levels(leaves)
    proof: List[bytes] = []
    idx = target
    for level in levels[:-1]:
        sibling = idx + 1 if idx % 2 == 0 else idx - 1
        if sibling < len(level):
            proof.append(level[sibling])
        # else: odd node promoted unchanged, no proof element.
        idx //= 2
    return proof


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


def claim_merkle_root(claims: Dict[str, Any],
                      concluded: Optional[Dict[str, int]] = None) -> bytes:
    """Root over the claims and the concluded tools. EMPTY_ROOT (32 zero
    bytes) when the tree has no leaf: the contract rejects claims against
    it."""
    leaves = _all_leaves(claims, concluded)
    if not leaves:
        return EMPTY_ROOT
    return _build_levels(leaves)[-1][0]


def claim_entry(claims: Dict[str, Any], earner: str) -> Optional[ClaimLeaf]:
    """The earner's leaf fields, or None when it has none."""
    for leaf in claim_leaves(claims):
        if leaf[0].lower() == str(earner).lower():
            return leaf
    return None


def claim_merkle_proof(claims: Dict[str, Any], earner: str,
                       concluded: Optional[Dict[str, int]] = None) -> Optional[List[bytes]]:
    """Proof for one earner's leaf, or None when it has none. A single-leaf
    tree yields an empty proof."""
    for i, leaf in enumerate(claim_leaves(claims)):
        if leaf[0].lower() == str(earner).lower():
            return _proof_at(_all_leaves(claims, concluded), i)
    return None


def concluded_proof(claims: Dict[str, Any], concluded: Dict[str, int],
                    digest: str) -> Optional[List[bytes]]:
    """Proof for one concluded tool's leaf, or None when the close did not
    conclude it."""
    d = str(digest).lower().removeprefix("0x")
    ordered = [x for x, _ in _concluded_order(concluded)]
    if d not in ordered:
        return None
    target = len(claim_leaves(claims)) + ordered.index(d)
    return _proof_at(_all_leaves(claims, concluded), target)


def _verify(leaf: bytes, root: bytes, proof: List[bytes]) -> bool:
    computed = leaf
    for p in proof:
        computed = _hash_pair(computed, bytes(p))
    return computed == bytes(root)


def verify_concluded_proof(root: bytes, digest: str, share_bps: int,
                           proof: List[bytes]) -> bool:
    """Python mirror of ToolBond.conclude's proof check."""
    return _verify(concluded_leaf(digest, share_bps), root, proof)


def verify_claim_proof(root: bytes, earner: str, tokens: List[str],
                       amounts: List[int], proof: List[bytes]) -> bool:
    """Python mirror of ``Substrate.claim``'s proof check, for tests and
    pre-flight validation before spending gas."""
    return _verify(claim_leaf(earner, tokens, amounts), root, proof)
