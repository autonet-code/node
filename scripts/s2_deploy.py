#!/usr/bin/env python3
"""S2: deploy the WAN run's jurisdiction onto the hub's test chain
(wan_test_plan, "S2"; issue #23).

The hub (the Hetzner box) runs the test chain on its loopback, served to the
world only as https://rpc.autonet.computer/<token> behind Cloudflare, and the
relay behind https://relay.autonet.computer (wss). This script reaches the
chain through an SSH tunnel and does what S0 does for its local chain:
test stablecoin, owner funding, the packaged jurisdiction (trustless-
contracts deploy-autonet-jurisdiction.js) with the relay as the Registry's
only seed, by hostname. It then restarts the relay on the jurisdiction's key
and writes the env a VM daemon needs.

Run:
    python scripts/s2_deploy.py --relay-peer <12D3Koo...>

Env: S2_DIR (default D:/autonet_s2), S2_SSH_KEY, S2_HOST (ubuntu@65.21.186.126),
S2_TOKEN_FILE (C:/code/cloud/wan_rpc_token.txt), TRUSTLESS_CONTRACTS_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("S0_DIR", os.environ.get("S2_DIR", r"D:/autonet_s2"))
import s0_local_run as s0  # noqa: E402

SSH_KEY = os.environ.get("S2_SSH_KEY", r"C:/code/engineer/hetzner_ed25519")
HOST = os.environ.get("S2_HOST", "ubuntu@65.21.186.126")
TOKEN_FILE = Path(os.environ.get("S2_TOKEN_FILE", r"C:/code/cloud/wan_rpc_token.txt"))
RELAY_HOST = "relay.autonet.computer"
RPC_HOST = "rpc.autonet.computer"


def ssh(cmd: str) -> str:
    r = subprocess.run(["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", HOST, cmd],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed: {r.stderr.strip()[-800:]}")
    return r.stdout.strip()


def start_relay(jkey: str) -> None:
    """(Re)start the relay serving ``jkey`` plus every jurisdiction it already
    serves, so an earlier run's daemons keep their seed."""
    try:
        args = ssh("docker inspect autonet-relay --format '{{join .Args \" \"}}'")
    except RuntimeError:
        args = ""
    keys = [a for a in args.split() if a.startswith(f"{s0.CHAIN_ID}:0x")]
    keys = list(dict.fromkeys(keys + [jkey]))
    flags = " ".join(f"--jurisdiction {k}" for k in keys)
    ssh("docker rm -f autonet-relay >/dev/null 2>&1; "
        "docker run -d --name autonet-relay --restart unless-stopped --network host "
        "-v /home/ubuntu/autonet-wan/relay:/data --entrypoint python autonet-daemon:wan "
        "-m nodes.relay --port 4101 --host 127.0.0.1 --transport ws --no-upnp "
        f"--announce /dns4/{RELAY_HOST}/tcp/443/wss --data-dir /data "
        f"{flags} >/dev/null")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-peer", required=True, help="the relay's libp2p peer id")
    ap.add_argument("--owners", type=int, default=s0.HOUSEHOLDS,
                    help="owner wallets to fund (hardhat accounts 1..N)")
    ap.add_argument("--rep-owners", type=int, default=None,
                    help="how many of them get genesis REP (the rest join with none)")
    args = ap.parse_args()
    run = s0.S0_DIR / time.strftime("deploy-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    tunnel = subprocess.Popen(["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-N",
                               "-L", f"{s0.RPC_PORT}:127.0.0.1:8545", HOST])
    try:
        w3, accts = s0.w3_and_accounts()
        s0.wait_for(lambda: w3.is_connected() and w3.eth.chain_id == s0.CHAIN_ID, 60,
                    "the hub chain through the tunnel")
        deployer, owners = accts[0], accts[1:1 + args.owners]
        founders = owners[:args.rep_owners if args.rep_owners is not None else len(owners)]
        usdc_addr = s0.deploy_test_usdc(w3, deployer)
        usdc = w3.eth.contract(address=usdc_addr,
                               abi=s0.artifact("test/TestToken6.sol/TestToken6.json")["abi"])
        for o in owners:
            s0.send(w3, deployer, usdc.functions.transfer(o.address, s0.USDC_PER_OWNER))
        seed = f"/dns4/{RELAY_HOST}/tcp/443/wss/p2p/{args.relay_peer}"
        j = s0.deploy_jurisdiction(run, usdc_addr, [o.address for o in founders], seed)
        jkey = f"{s0.CHAIN_ID}:{j['dao']['governor'].lower()}"
        start_relay(jkey)
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        env = {
            "ATN_JURISDICTION_REGISTRY": j["dao"]["registry"],
            "ATN_JURISDICTION_RPC": f"https://{RPC_HOST}/{token}",
            "AUTONET_P2P_TRANSPORT": "ws",
        }
        (run / "vm_env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps(
            {"jurisdiction_key": jkey, "seed": seed, "usdc": usdc_addr,
             "owners": [o.address for o in owners],
             "founders": [o.address for o in founders]}, indent=1), encoding="utf-8")
        s0.log(f"deployed: governor {j['dao']['governor']} registry {j['dao']['registry']}")
        s0.log(f"relay restarted on {jkey}; VM env in {run / 'vm_env.json'}")
        return 0
    finally:
        tunnel.terminate()


if __name__ == "__main__":
    sys.exit(main())
