#!/usr/bin/env python3
"""Archive every tool the WAN run publishes (issues #24, #26).

Tool code lives only with its author's daemon (and whoever adopted it), and
the run's VMs go away. Every few minutes this reads Substrate's
ToolRegistered events (each publish registers the manifest digest), asks
the local daemon to fetch each new tool (an adoption proposal pulls the
manifest and its code over the network; nothing is approved or run), then
copies the digest-verified blobs into the library:

  <library>/blobs/<sha256>      manifests and code, content-addressed
  <library>/index.jsonl         one line per tool: digest, name, author,
                                code digest, dependencies, publisher, block
  <library>/uploads/            tool stores VMs uploaded to the hub

    python scripts/s2/archive_tools.py --deploy D:/autonet_s2/deploy-<...> \\
        --daemon ws://127.0.0.1:<port> --data-dir D:/autonet_s3/vm-<...>/h11/.atn
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SSH_KEY = r"C:/code/engineer/hetzner_ed25519"
HOST = "ubuntu@65.21.186.126"


def find_blob(data_dir: Path, digest: str):
    for p in data_dir.rglob(digest):
        if p.is_file():
            return p
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", required=True)
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--data-dir", required=True, help="the local daemon's ~/.atn")
    ap.add_argument("--library", default=r"D:/autonet_s3/library")
    ap.add_argument("--interval", type=float, default=300)
    a = ap.parse_args()
    deploy = Path(a.deploy)
    j = json.loads((deploy / "jurisdiction.json").read_text(encoding="utf-8"))
    rpc = json.loads((deploy / "vm_env.json").read_text(encoding="utf-8"))["ATN_JURISDICTION_RPC"]
    lib = Path(a.library)
    (lib / "blobs").mkdir(parents=True, exist_ok=True)
    (lib / "uploads").mkdir(exist_ok=True)
    index = lib / "index.jsonl"
    done = set()
    if index.exists():
        done = {json.loads(l)["digest"] for l in index.read_text(encoding="utf-8").splitlines() if l}

    from web3 import Web3
    from nodes.common.contract_artifacts import load_abi
    import s0_local_run as s0
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30,
                                                     "headers": {"User-Agent": "autonet-wan/1"}}))
    sub = w3.eth.contract(address=w3.to_checksum_address(j["autonet"]["substrate"]),
                          abi=load_abi("Substrate"))
    client = s0.DaemonClient(a.daemon, "archive")
    await client.connect()
    data_dir = Path(a.data_dir)
    while True:
        try:
            logs = sub.events.ToolRegistered().get_logs(from_block=0)
            for ev in logs:
                digest = bytes(ev["args"]["manifestDigest"]).hex()
                if digest in done:
                    continue
                await client.call({"type": "adopt_tool", "digest": digest,
                                   "reason": "archive for the mainnet seed library"}, timeout=120)
                # Approval is what pulls the code: the tool is installed on
                # this archive daemon but never run.
                await client.call({"type": "approve_adoption", "digest": digest}, timeout=180)
                mpath = find_blob(data_dir, digest)
                if mpath is None:
                    print(f"{digest[:16]}: manifest not fetched yet", flush=True)
                    continue
                raw = mpath.read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest:
                    print(f"{digest[:16]}: manifest digest mismatch, skipped", flush=True)
                    continue
                manifest = json.loads(raw)
                code = manifest.get("code_digest") or ""
                cpath = find_blob(data_dir, code) if code else None
                if code and cpath is None:
                    print(f"{digest[:16]} {manifest.get('name')}: code not fetched yet", flush=True)
                    continue
                shutil.copy2(mpath, lib / "blobs" / digest)
                if cpath is not None:
                    shutil.copy2(cpath, lib / "blobs" / code)
                row = {"digest": digest, "name": manifest.get("name"),
                       "author": manifest.get("author"), "code_digest": code,
                       "deps": manifest.get("dependencies") or manifest.get("deps"),
                       "publisher": ev["args"]["agent"], "block": ev["blockNumber"]}
                with open(index, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                done.add(digest)
                print(f"archived {manifest.get('name')} {digest[:16]}", flush=True)
            subprocess.run(["scp", "-i", SSH_KEY, "-q", "-o", "BatchMode=yes",
                            f"{HOST}:/home/ubuntu/autonet-wan/obs/uploads/*", str(lib / "uploads")],
                           capture_output=True)
            # Everything that ever crossed the relay (it keeps relayed blobs),
            # so a tool whose author's VM was recycled is still kept.
            rb = lib / "relay_blobs"
            rb.mkdir(exist_ok=True)
            subprocess.run(["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", HOST,
                            "sudo tar czf /tmp/relay_blobs.tgz -C /home/ubuntu/autonet-wan/relay blobs"
                            " && sudo chown ubuntu /tmp/relay_blobs.tgz"], capture_output=True)
            subprocess.run(["scp", "-i", SSH_KEY, "-q", "-o", "BatchMode=yes",
                            f"{HOST}:/tmp/relay_blobs.tgz", str(rb / "relay_blobs.tgz")], capture_output=True)
            subprocess.run(["tar", "xzf", str(rb / "relay_blobs.tgz"), "-C", str(rb)], capture_output=True)
            print(f"[{time.strftime('%H:%M:%S')}] library: {len(done)} tools, "
                  f"{len(list((lib / 'uploads').iterdir()))} uploads", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"archive: {exc!r}", flush=True)
        await asyncio.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
