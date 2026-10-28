#!/usr/bin/env python3
"""Build the ATN genesis claim tree for a jurisdiction fork or split.

A new jurisdiction carries the ATN balances of an old Substrate (minus an
exclusion list) as a merkle root passed to the new Substrate's constructor
(``genesisRoot``). Each holder then calls ``claimGenesis(amount, proof)``
once. A fresh network passes a zero root and needs none of this.

Steps:
  1. Holders = every address seen in the old Substrate's ATNTransfer logs
     (from/to) between --from-block and --block (chunked: public RPCs cap
     log ranges).
  2. Balance = ``balanceOfAt(holder, --block)`` (Substrate checkpoints
     balances, so no archive node is needed).
  3. Drop address(0), zero balances and every address in --exclude.
  4. Leaves keccak256(bytes.concat(keccak256(abi.encode(address, amount))))
     sorted by lowercase address, sorted-pair tree (the claimFor
     convention, nodes/common/mint_merkle.py).

REP genesis needs no contract work: dao's deploy-autonet-jurisdiction.js
takes MEMBERS/AMOUNTS. With --rep-token the script reads the old
RepToken's balances (holders from its Transfer logs, balanceOf at --block;
a historical balanceOf needs an archive node unless --block is the head)
minus the same exclusions and emits those two env strings.

The exclusion file is one address per line (``#`` comments allowed) or a
JSON list of addresses.

Output (--out, JSON):
    {"genesis_root", "substrate", "block", "total", "count",
     "excluded", "claims": {address: {"amount", "proof"}},
     "rep": {"MEMBERS", "AMOUNTS", "total", "count"}}   # rep with --rep-token

Usage:
    python scripts/build_atn_genesis.py --rpc https://rpc.old \\
        --substrate 0xOldSubstrate --block 1234567 [--from-block N] \\
        [--exclude excluded.txt] [--rep-token 0xOldRep] --out genesis.json
Then deploy with GENESIS_ROOT=<genesis_root> (scripts/deploy_substrate.js).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eth_utils import is_address, keccak, to_checksum_address  # noqa: E402

from nodes.common.mint_merkle import (  # noqa: E402
    EMPTY_ROOT,
    _build_levels,
    _leaf_hash,
    verify_mint_proof,
)

ZERO = "0x" + "00" * 20
ATN_TRANSFER_TOPIC = "0x" + keccak(
    text="ATNTransfer(address,address,uint256)").hex()
ERC20_TRANSFER_TOPIC = "0x" + keccak(
    text="Transfer(address,address,uint256)").hex()

_SUBSTRATE_VIEW_ABI = [{
    "name": "balanceOfAt", "type": "function", "stateMutability": "view",
    "inputs": [{"name": "agent", "type": "address"},
               {"name": "blockNumber", "type": "uint256"}],
    "outputs": [{"name": "", "type": "uint256"}],
}]
_ERC20_VIEW_ABI = [{
    "name": "balanceOf", "type": "function", "stateMutability": "view",
    "inputs": [{"name": "account", "type": "address"}],
    "outputs": [{"name": "", "type": "uint256"}],
}]


def _hex(b: bytes) -> str:
    return "0x" + bytes(b).hex()


def load_exclusions(path: Optional[str]) -> Set[str]:
    """Lowercase addresses from a JSON list or one-per-line file."""
    if not path:
        return set()
    text = Path(path).read_text(encoding="utf-8")
    try:
        items = json.loads(text)
    except ValueError:
        items = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    out: Set[str] = set()
    for a in items:
        a = str(a).strip()
        if not a:
            continue
        if not is_address(a):
            raise ValueError(f"exclusion list: not an address: {a!r}")
        out.add(a.lower())
    return out


def _topic_address(topic) -> str:
    raw = bytes(topic) if not isinstance(topic, str) else bytes.fromhex(topic[2:])
    return to_checksum_address(raw[-20:])


def collect_holders(w3, address: str, topic: str, from_block: int,
                    to_block: int, chunk: int = 10_000) -> Set[str]:
    """Every from/to address in the contract's transfer logs (indexed
    topics 1 and 2), fetched in chunks."""
    holders: Set[str] = set()
    addr = to_checksum_address(address)
    start = from_block
    while start <= to_block:
        end = min(start + chunk - 1, to_block)
        logs = w3.eth.get_logs({"address": addr, "fromBlock": start,
                                "toBlock": end, "topics": [topic]})
        for lg in logs:
            for t in lg["topics"][1:3]:
                holders.add(_topic_address(t))
        start = end + 1
    holders.discard(to_checksum_address(ZERO))
    return holders


def atn_balances(w3, substrate: str, block: int, from_block: int = 0,
                 chunk: int = 10_000) -> Dict[str, int]:
    """ATN balance of every holder at the END of ``block``."""
    c = w3.eth.contract(address=to_checksum_address(substrate),
                        abi=_SUBSTRATE_VIEW_ABI)
    out: Dict[str, int] = {}
    for h in collect_holders(w3, substrate, ATN_TRANSFER_TOPIC,
                             from_block, block, chunk):
        out[h] = int(c.functions.balanceOfAt(h, block).call())
    return out


def rep_balances(w3, rep_token: str, block: int, from_block: int = 0,
                 chunk: int = 10_000) -> Dict[str, int]:
    """RepToken balance of every holder at ``block`` (historical
    balanceOf: archive node unless ``block`` is the head)."""
    c = w3.eth.contract(address=to_checksum_address(rep_token),
                        abi=_ERC20_VIEW_ABI)
    out: Dict[str, int] = {}
    for h in collect_holders(w3, rep_token, ERC20_TRANSFER_TOPIC,
                             from_block, block, chunk):
        out[h] = int(c.functions.balanceOf(h).call(block_identifier=block))
    return out


def filter_balances(balances: Dict[str, int],
                    excluded: Iterable[str]) -> List[Tuple[str, int]]:
    """Drop zero, address(0) and excluded; sort by lowercase address."""
    ex = {a.lower() for a in excluded} | {ZERO}
    entries = [(to_checksum_address(a), int(v)) for a, v in balances.items()
               if int(v) > 0 and a.lower() not in ex]
    entries.sort(key=lambda e: e[0].lower())
    return entries


def build_tree(entries: List[Tuple[str, int]]) -> Tuple[bytes, Dict[str, dict]]:
    """(root, {address: {"amount": str, "proof": [hex]}}). Empty input
    gives the zero root (= no genesis)."""
    if not entries:
        return EMPTY_ROOT, {}
    levels = _build_levels([_leaf_hash(a, m) for a, m in entries])
    root = levels[-1][0]
    claims: Dict[str, dict] = {}
    for i, (addr, amt) in enumerate(entries):
        proof: List[bytes] = []
        idx = i
        for level in levels[:-1]:
            sib = idx + 1 if idx % 2 == 0 else idx - 1
            if sib < len(level):
                proof.append(level[sib])
            idx //= 2
        assert verify_mint_proof(root, addr, amt, proof)
        claims[addr] = {"amount": str(amt), "proof": [_hex(p) for p in proof]}
    return root, claims


def rep_env(entries: List[Tuple[str, int]]) -> Dict[str, str]:
    return {"MEMBERS": ",".join(a for a, _ in entries),
            "AMOUNTS": ",".join(str(v) for _, v in entries)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--substrate", required=True, help="old Substrate address")
    ap.add_argument("--block", type=int, help="snapshot block (default: head)")
    ap.add_argument("--from-block", type=int, default=0,
                    help="first block to scan for transfers (the old deploy block)")
    ap.add_argument("--chunk", type=int, default=10_000, help="log range per call")
    ap.add_argument("--exclude", help="exclusion list file")
    ap.add_argument("--rep-token", help="old RepToken address (emit MEMBERS/AMOUNTS)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(args.rpc))
    block = args.block if args.block is not None else w3.eth.block_number
    excluded = load_exclusions(args.exclude)

    atn = filter_balances(
        atn_balances(w3, args.substrate, block, args.from_block, args.chunk),
        excluded)
    root, claims = build_tree(atn)
    out = {
        "genesis_root": _hex(root),
        "substrate": to_checksum_address(args.substrate),
        "block": block,
        "total": str(sum(v for _, v in atn)),
        "count": len(atn),
        "excluded": sorted(excluded),
        "claims": claims,
    }
    if args.rep_token:
        rep = filter_balances(
            rep_balances(w3, args.rep_token, block, args.from_block, args.chunk),
            excluded)
        out["rep"] = {**rep_env(rep), "total": str(sum(v for _, v in rep)),
                      "count": len(rep)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"GENESIS_ROOT={out['genesis_root']}  ({out['count']} holders, "
          f"{out['total']} ATN base units, block {block})")
    if "rep" in out:
        print(f"MEMBERS={out['rep']['MEMBERS']}")
        print(f"AMOUNTS={out['rep']['AMOUNTS']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
