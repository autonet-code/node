#!/usr/bin/env python3
"""Build a genesis seed snapshot for merging jurisdictions by genesis.

docs/multi_jurisdiction.md decision 9: a new DAO is deployed with an agreed
genesis REP list, and its FIRST anchor is a governed proposal that adopts a
posted state snapshot (hash on chain, blob served peer to peer). This script
builds that snapshot and the proposal arguments; it sends no transaction.

Sources, in priority order (on a conflict the source listed first wins):

    --source chain:<rpc_url>,<substrate_address>
        The latest anchored carry bundle of a live jurisdiction, fetched with
        fetch_carry_over_from_chain (payload sha256 == on-chain payloadHash,
        bundle sha256 == payload carry_cid). Blobs are read from the
        --source-blobs directories.
    --source bundle:<path>
        A local canonical carry bundle file (decoded strictly; NOT verified
        against any chain, the summary says so).

Outputs (in --out):
    <carry_cid>             the merged carry bundle blob (epoch_id "seed")
    <payload_cid>           the seed anchor payload blob ("seed": true)
    seed_anchor.json        Substrate.anchor(...) arguments, the proposal
                            description, calldata (when eth_abi is present),
                            per-source counts and conflicts
With --blob-store DIR both blobs are also written into that daemon blob store
(a daemon's store is <world data root>/<rpb>/blobs, by default
~/.autonet/world/<rpb>/blobs; pass it explicitly), from which the daemon
serves them to peers over /autonet/blob/1.0.0.

Usage:
    python scripts/build_genesis_seed.py --chain-id 127823 \\
        --source chain:https://rpc.a,0xSubstrateA --source bundle:b.json \\
        --source-blobs D:/blobs_a --out seed_out [--blob-store DIR]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes.common.authoritative_encoding import (  # noqa: E402
    CARRY_KEYS,
    cid_for_blob,
    decode_carry_bundle,
    encode_authoritative_payload,
    encode_carry_bundle,
)
from nodes.common.governed_anchor import anchor_description  # noqa: E402

SEED_EPOCH_ID = "seed"
ZERO32 = "00" * 32
SEED_ROOT_DOMAIN = b"autonet:genesis-seed:v1"
ANCHOR_SIGNATURE = "anchor(string,bytes32,bytes32,bytes32,string,bytes32,bytes32,uint256)"


@dataclass
class SourceCarry:
    label: str
    kind: str                     # "chain" | "bundle"
    carry: Dict[str, Any]
    verified: bool
    epoch_id: str = ""
    carry_cid: str = ""


@dataclass
class MergeResult:
    carry: Dict[str, Any]
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Merge (pure)
# ---------------------------------------------------------------------------


def _merge_keyed(name: str, maps: List[Tuple[str, Dict[str, Any]]],
                 conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Union of dicts keyed by digest (or agent); first source wins."""
    merged: Dict[str, Any] = {}
    kept_from: Dict[str, str] = {}
    for label, m in maps:
        for key in sorted(m):
            value = m[key]
            if key not in merged:
                merged[key] = copy.deepcopy(value)
                kept_from[key] = label
            elif merged[key] != value:
                conflicts.append({"map": name, "key": key,
                                  "kept": kept_from[key], "dropped": label})
    return {k: merged[k] for k in sorted(merged)}


def merge_carry(sources: List[SourceCarry]) -> MergeResult:
    """Deterministic union of carry bundles, first-listed source wins.

    tool_registrations and tool_positions are keyed by tool digest.
    tool_vetting is ``{"manifests": {digest: ...}, "busts": {agent: n}}``,
    so each of its sub-maps is unioned by key the same way.
    """
    if not sources:
        raise ValueError("at least one source is required")
    conflicts: List[Dict[str, Any]] = []
    warnings: List[str] = []
    out: Dict[str, Any] = {}
    for key in ("tool_registrations", "tool_positions"):
        out[key] = _merge_keyed(
            key, [(s.label, s.carry.get(key) or {}) for s in sources], conflicts)
    vet_subs = sorted({sub for s in sources
                       for sub in (s.carry.get("tool_vetting") or {})})
    vetting: Dict[str, Any] = {}
    for sub in vet_subs:
        vals = [(s.label, (s.carry.get("tool_vetting") or {}).get(sub))
                for s in sources
                if sub in (s.carry.get("tool_vetting") or {})]
        if all(isinstance(v, dict) for _, v in vals):
            vetting[sub] = _merge_keyed(f"tool_vetting.{sub}", vals, conflicts)
        else:
            vetting[sub] = copy.deepcopy(vals[0][1])
            for label, v in vals[1:]:
                if v != vals[0][1]:
                    conflicts.append({"map": "tool_vetting", "key": sub,
                                      "kept": vals[0][0], "dropped": label})
    out["tool_vetting"] = vetting
    # Position heads are per charter axis: a source on a different charter
    # has a different dimension count, which the members should see.
    dims = sorted({len(p.get("head") or []) for p in out["tool_positions"].values()
                   if isinstance(p, dict)})
    if len(dims) > 1:
        warnings.append(
            "tool_positions heads have mixed dimension counts "
            f"{dims}: sources ran different charters")
    return MergeResult(carry={k: out[k] for k in CARRY_KEYS},
                       conflicts=conflicts, warnings=warnings)


# ---------------------------------------------------------------------------
# Seed anchor (pure)
# ---------------------------------------------------------------------------


def seed_epoch_root(carry_cid: str, chain_id: int) -> str:
    """Deterministic epochRoot for the seed anchor: sha256 over a domain
    tag, the new DAO's chain id and the bundle cid (hex, no 0x)."""
    h = hashlib.sha256()
    h.update(SEED_ROOT_DOMAIN)
    h.update(b"|" + str(int(chain_id)).encode("ascii"))
    h.update(b"|" + carry_cid.lower().encode("ascii"))
    return h.hexdigest()


def build_seed(carry: Dict[str, Any], *, chain_id: int,
               cutoff: Optional[int] = None) -> Dict[str, Any]:
    """Encode the bundle + seed payload and derive the anchor arguments.

    ``cutoff`` (Decision 2026-10-26) must be after the new Substrate's
    deployedAt and before the seed proposal executes; default: now."""
    if cutoff is None:
        import time
        cutoff = int(time.time())
    bundle = encode_carry_bundle(carry, epoch_id=SEED_EPOCH_ID)
    carry_cid = cid_for_blob(bundle)
    root = seed_epoch_root(carry_cid, chain_id)
    payload = encode_authoritative_payload(
        {
            "agent_mint": {}, "agent_novelty": {},
            "total_mint": 0.0, "total_novelty": 0.0,
            "gate_applied": False, "n_batches": 0, "n_events": 0,
            "world_cid": "", "carry_cid": carry_cid, "seed": True,
        },
        epoch_id=SEED_EPOCH_ID, epoch_root_hex=root,
        prev_epoch_root_hex=ZERO32,
    )
    payload_cid = cid_for_blob(payload)
    args = {
        "epochId": SEED_EPOCH_ID,
        "epochRoot": "0x" + root,
        "prevEpochRoot": "0x" + ZERO32,
        "prevAnchorHash": "0x" + ZERO32,
        "agentMintCid": "",
        "payloadHash": "0x" + payload_cid,
        "agentMintRoot": "0x" + ZERO32,
        "cutoff": int(cutoff),
    }
    return {
        "bundle": bundle, "carry_cid": carry_cid,
        "payload": payload, "payload_cid": payload_cid,
        "anchor_args": args,
        "anchor_args_ordered": [args[k] for k in (
            "epochId", "epochRoot", "prevEpochRoot", "prevAnchorHash",
            "agentMintCid", "payloadHash", "agentMintRoot", "cutoff")],
        "description": anchor_description(SEED_EPOCH_ID, bytes.fromhex(root)),
        "calldata": _anchor_calldata(args),
    }


def _anchor_calldata(args: Dict[str, str]) -> Optional[str]:
    try:
        from eth_abi import encode
        from eth_utils import keccak
    except Exception:
        return None
    b = lambda h: bytes.fromhex(h[2:])  # noqa: E731
    body = encode(
        ["string", "bytes32", "bytes32", "bytes32", "string", "bytes32",
         "bytes32", "uint256"],
        [args["epochId"], b(args["epochRoot"]), b(args["prevEpochRoot"]),
         b(args["prevAnchorHash"]), args["agentMintCid"],
         b(args["payloadHash"]), b(args["agentMintRoot"]), int(args["cutoff"])])
    return "0x" + (keccak(text=ANCHOR_SIGNATURE)[:4] + body).hex()


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _dir_blob_getter(dirs: List[Path]):
    def get(cid: str) -> Optional[bytes]:
        for d in dirs:
            p = d / cid
            if p.is_file():
                return p.read_bytes()
        return None
    return get


def load_chain_source(contract: Any, blob_source: Any, label: str) -> SourceCarry:
    from nodes.common.state_sync import fetch_carry_over_from_chain

    fetch = fetch_carry_over_from_chain(contract, blob_source)
    if fetch.status != "ok":
        raise RuntimeError(
            f"{label}: no verified carry bundle (status {fetch.status!r}"
            + (f", epoch {fetch.epoch_id}" if fetch.epoch_id else "") + ")")
    return SourceCarry(label=label, kind="chain", carry=fetch.carry,
                       verified=True, epoch_id=fetch.epoch_id,
                       carry_cid=fetch.carry_cid)


def load_bundle_source(path: Path) -> SourceCarry:
    blob = path.read_bytes()
    bundle = decode_carry_bundle(blob)
    return SourceCarry(label=f"bundle:{path}", kind="bundle",
                       carry=bundle["carry"], verified=False,
                       epoch_id=bundle["epoch_id"], carry_cid=cid_for_blob(blob))


def _load_source(spec: str, blob_dirs: List[Path]) -> SourceCarry:
    kind, _, rest = spec.partition(":")
    if kind == "bundle":
        return load_bundle_source(Path(rest))
    if kind == "chain":
        rpc, _, addr = rest.rpartition(",")
        if not rpc or not addr:
            raise ValueError(f"chain source needs <rpc_url>,<substrate>: {spec}")
        from web3 import Web3
        from nodes.common.voice_state import _VOICE_ABI

        w3 = Web3(Web3.HTTPProvider(rpc))
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(addr), abi=_VOICE_ABI)
        return load_chain_source(contract, _dir_blob_getter(blob_dirs),
                                 f"chain:{rpc}@{addr}")
    raise ValueError(f"unknown source kind in {spec!r} (chain: or bundle:)")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _counts(carry: Dict[str, Any]) -> Dict[str, int]:
    vet = carry.get("tool_vetting") or {}
    return {
        "tool_registrations": len(carry.get("tool_registrations") or {}),
        "tool_positions": len(carry.get("tool_positions") or {}),
        "tool_vetting.manifests": len(vet.get("manifests") or {}),
    }


def write_outputs(out_dir: Path, seed: Dict[str, Any], *, chain_id: int,
                  sources: List[SourceCarry], merged: MergeResult,
                  substrate: str = "",
                  blob_store: Optional[Path] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / seed["carry_cid"]).write_bytes(seed["bundle"])
    (out_dir / seed["payload_cid"]).write_bytes(seed["payload"])
    if blob_store is not None:
        from nodes.common.blob_store import BlobStore

        store = BlobStore(data_dir=str(blob_store))
        for blob in (seed["bundle"], seed["payload"]):
            store.add_bytes(blob)
    report = {
        "chain_id": int(chain_id),
        "target_substrate": substrate,
        "carry_cid": seed["carry_cid"],
        "payload_cid": seed["payload_cid"],
        "anchor_args": seed["anchor_args"],
        "anchor_args_ordered": seed["anchor_args_ordered"],
        "description": seed["description"],
        "calldata": seed["calldata"],
        "sources": [{"source": s.label, "verified_against_chain": s.verified,
                     "epoch_id": s.epoch_id, "carry_cid": s.carry_cid,
                     "counts": _counts(s.carry)} for s in sources],
        "merged_counts": _counts(merged.carry),
        "conflicts": merged.conflicts,
        "warnings": merged.warnings,
    }
    (out_dir / "seed_anchor.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def summary(report: Dict[str, Any]) -> str:
    lines = [f"Genesis seed for chain {report['chain_id']}", "Sources (first wins):"]
    for s in report["sources"]:
        c = s["counts"]
        lines.append(
            f"  {s['source']}  [{'chain-verified' if s['verified_against_chain'] else 'UNVERIFIED local file'}]"
            f"  epoch={s['epoch_id']}  tools={c['tool_registrations']}"
            f"  positions={c['tool_positions']}  vetted={c['tool_vetting.manifests']}")
    m = report["merged_counts"]
    lines.append(f"Merged: tools={m['tool_registrations']}  "
                 f"positions={m['tool_positions']}  "
                 f"vetted={m['tool_vetting.manifests']}")
    lines.append(f"Conflicts: {len(report['conflicts'])}")
    for c in report["conflicts"]:
        lines.append(f"  {c['map']}[{c['key']}]: kept {c['kept']}, dropped {c['dropped']}")
    for w in report["warnings"]:
        lines.append(f"Warning: {w}")
    a = report["anchor_args"]
    lines += [
        f"carry_cid   {report['carry_cid']}",
        f"payload_cid {report['payload_cid']}  (= payloadHash)",
        f"epochRoot   {a['epochRoot']}",
        f"Proposal description: {report['description']}",
        "Members: fetch both blobs, check sha256 matches these cids, and check "
        "the payload's carry_cid before voting.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chain-id", type=int, required=True,
                    help="chain id of the NEW DAO (bound into epochRoot)")
    ap.add_argument("--source", action="append", required=True,
                    help="chain:<rpc_url>,<substrate> or bundle:<path>; "
                         "repeat, first listed wins conflicts")
    ap.add_argument("--source-blobs", action="append", default=[],
                    help="blob directory to read chain sources' payload and "
                         "carry blobs from (repeatable)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--blob-store", default=None,
                    help="also write both blobs into this daemon blob store dir")
    ap.add_argument("--substrate", default="",
                    help="the new DAO's Substrate address (recorded only)")
    ap.add_argument("--cutoff", type=int, default=None,
                    help="the seed epoch's cutoff (unix seconds): after the "
                         "new Substrate's deployedAt; default now")
    args = ap.parse_args(argv)

    blob_dirs = [Path(d).expanduser() for d in args.source_blobs]
    sources = [_load_source(s, blob_dirs) for s in args.source]
    merged = merge_carry(sources)
    seed = build_seed(merged.carry, chain_id=args.chain_id, cutoff=args.cutoff)
    report = write_outputs(
        Path(args.out), seed, chain_id=args.chain_id, sources=sources,
        merged=merged, substrate=args.substrate,
        blob_store=Path(args.blob_store).expanduser() if args.blob_store else None)
    print(summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
