#!/usr/bin/env python3
"""WAN run collector (issue #23), run on the owner's machine.

Three views, appended every interval under <run dir>:
  anchors.jsonl  the chain (public RPC): each new anchor, its epoch length,
                 burned fees, ATN supply, tools registered on Substrate
  daemons.jsonl  the hub's telemetry ingest: newest report per daemon
                 (peers, epoch, errors, log lines)
  relay.jsonl    the relay's log since the last poll (connections, errors)
and one summary line per poll on stdout.

    python scripts/s2/collector.py --deploy D:/autonet_s2/deploy-<...>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Cloudflare refuses urllib's default user agent.
UA = "autonet-wan/1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SSH_KEY = r"C:/code/engineer/hetzner_ed25519"
HOST = "ubuntu@65.21.186.126"


def obs_latest(token: str) -> dict:
    req = urllib.request.Request("https://obs.autonet.computer/latest",
                                 headers={"User-Agent": UA, "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"_error": repr(exc)[:200]}


def relay_lines(since: str) -> list[str]:
    r = subprocess.run(["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", HOST,
                        f"docker logs --since {since} autonet-relay 2>&1 | tail -200"],
                       capture_output=True, text=True, errors="replace")
    return [ln[:300] for ln in r.stdout.splitlines()
            if any(k in ln for k in ("Connected", "disconnect", "ERROR", "WARNING", "Traceback"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", required=True, help="the s2_deploy.py run dir")
    ap.add_argument("--interval", type=float, default=30.0)
    a = ap.parse_args()
    deploy = Path(a.deploy)
    j = json.loads((deploy / "jurisdiction.json").read_text(encoding="utf-8"))
    rpc = json.loads((deploy / "vm_env.json").read_text(encoding="utf-8"))["ATN_JURISDICTION_RPC"]
    token = Path(r"C:/code/cloud/wan_obs_token.txt").read_text(encoding="utf-8").strip()
    out = deploy / time.strftime("collect-%Y%m%d-%H%M%S")
    out.mkdir()

    from web3 import Web3
    from nodes.common.contract_artifacts import load_abi
    from nodes.common.voice_state import _anchor_cutoff
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    sub = w3.eth.contract(address=w3.to_checksum_address(j["autonet"]["substrate"]),
                          abi=load_abi("Substrate"))
    seen, prev_cutoff = 0, int(sub.functions.deployedAt().call())
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    while True:
        try:
            n = int(sub.functions.anchorCount().call())
            with open(out / "anchors.jsonl", "a", encoding="utf-8") as fh:
                while seen < n:
                    an = sub.functions.getAnchor(seen).call()
                    cutoff = _anchor_cutoff(an)
                    row = {"index": seen, "epoch_id": an[0], "length": cutoff - prev_cutoff,
                           "burned": int(sub.functions.burnedBetween(prev_cutoff, cutoff).call()),
                           "supply": int(sub.functions.atnTotalSupply().call())}
                    fh.write(json.dumps(row) + "\n")
                    print(f"ANCHOR {seen + 1}: {an[0]} length {row['length']}s burned {row['burned']}",
                          flush=True)
                    prev_cutoff, seen = cutoff, seen + 1
            latest = obs_latest(token)
            with open(out / "daemons.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "latest": latest}) + "\n")
            now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            lines = relay_lines(since)
            since = now
            with open(out / "relay.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "lines": lines}) + "\n")
            live = [d for d, r in latest.items() if isinstance(r, dict)
                    and time.time() - float(r.get("received") or 0) < 180]
            print(f"[{time.strftime('%H:%M:%S')}] anchors {n}, daemons reporting {len(live)}"
                  f" ({', '.join(sorted(live))}), relay events {len(lines)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"collector: {exc!r}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
