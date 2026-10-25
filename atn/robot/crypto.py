"""Signing and merkle helpers shared by the worklog and the poll rail.

Every signed object is a JSON dict whose ``sig`` field is an Ethereum
personal-sign signature over the canonical JSON of the rest. Ethereum
signatures are deterministic (RFC 6979) and we accept only the low-s form,
so each (key, object) pair has exactly one valid signature. The poll rail
depends on that: a poller's signature fixes its poll time and challenge.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
ZERO = "0" * 64


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: dict[str, Any]) -> bytes:
    body = {k: v for k, v in obj.items() if k != "sig"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def obj_hash(obj: dict[str, Any]) -> str:
    """Hash of the signed body (``sig`` excluded)."""
    return sha256_hex(canonical(obj))


def sign_bytes(data: bytes, private_key: str) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    return "0x" + Account.sign_message(encode_defunct(data),
                                       private_key=private_key).signature.hex().removeprefix("0x")


def recover_bytes(data: bytes, sig: str) -> str:
    """Signer address, lowercased. Raises on a malformed or high-s signature."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    raw = bytes.fromhex((sig or "").removeprefix("0x"))
    if len(raw) != 65:
        raise ValueError("signature must be 65 bytes")
    if int.from_bytes(raw[32:64], "big") > _N // 2:
        raise ValueError("non-canonical (high-s) signature")
    return Account.recover_message(encode_defunct(data), signature=raw).lower()


def sign(obj: dict[str, Any], private_key: str) -> dict[str, Any]:
    return {**obj, "sig": sign_bytes(canonical(obj), private_key)}


def signer(obj: dict[str, Any]) -> str:
    return recover_bytes(canonical(obj), obj.get("sig") or "")


# -- merkle -----------------------------------------------------------------
# Ordered binary tree over hex sha256 leaves; an odd node is carried up
# unchanged. A proof is a list of (sibling hex, sibling_is_left).

def _pair(left: str, right: str) -> str:
    return sha256_hex(bytes.fromhex(left) + bytes.fromhex(right))


def merkle_root(leaves: list[str]) -> str:
    level = list(leaves)
    if not level:
        return ZERO
    while len(level) > 1:
        nxt = [_pair(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def merkle_proof(leaves: list[str], index: int) -> list[tuple[str, bool]]:
    proof, level, i = [], list(leaves), index
    while len(level) > 1:
        sib = i ^ 1
        if sib < len(level):
            proof.append((level[sib], sib < i))
        nxt = [_pair(level[j], level[j + 1]) for j in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level, i = nxt, i // 2
    return proof


def verify_proof(leaf: str, proof: list[tuple[str, bool]], root: str) -> bool:
    h = leaf
    for sib, is_left in proof:
        h = _pair(sib, h) if is_left else _pair(h, sib)
    return h == root
