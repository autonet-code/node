#!/usr/bin/env python3
"""Bring up this VM's households for the WAN run (issue #23).

Run inside a cloud session, from a checkout of the run branch in a venv:

    python scripts/s2/vm_up.py --households 1,2 --obs-token <token>

For each household (an owner wallet on the hub's test chain, by index):
start one `atn --headless` daemon in its own home, onboard it the way its
owner would (operator registered and delegated REP, ATN bought, an external
agent created, funded and registered), then start the telemetry sidecar.
The run's files come from the hub (obs.autonet.computer/share). The process
then stays up: the daemons exit when it does.

Afterwards it prints, per household, the MCP command that makes the session
that household's agent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Cloudflare refuses urllib's default user agent.
UA = "autonet-wan/1"
REPO = Path(__file__).resolve().parents[2]
OBS = "https://obs.autonet.computer"


def fetch(name: str, token: str) -> dict:
    req = urllib.request.Request(f"{OBS}/share/{name}",
                                 headers={"User-Agent": UA, "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


async def amain(a) -> int:
    token = a.obs_token or Path(a.obs_token_file).read_text(encoding="utf-8").strip()
    j = fetch("jurisdiction.json", token)
    vm_env = fetch("vm_env.json", token)
    summary = fetch("summary.json", token)
    # The dao ABIs the onboarding needs, laid out like a trustless-contracts tree.
    dao = Path(tempfile.mkdtemp(prefix="dao-abi-"))
    for rel, name in (("test/TestToken6.sol/TestToken6.json", "abi_TestToken6.json"),
                      ("AutonetVault.sol/AutonetVault.json", "abi_AutonetVault.json")):
        p = dao / "artifacts" / "contracts" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fetch(name, token)), encoding="utf-8")

    os.environ.update({
        "TRUSTLESS_CONTRACTS_DIR": str(dao),
        "S0_RPC_URL": vm_env["ATN_JURISDICTION_RPC"],
        "S0_KEEP_PROXY": "1",
        "S0_DIR": str(Path(a.dir).expanduser()),
    })
    # Local listeners must never go through the egress proxy.
    for k in ("NO_PROXY", "no_proxy"):
        os.environ[k] = ",".join(filter(None, [os.environ.get(k, ""), "localhost", "127.0.0.1"]))
    sys.path.insert(0, str(REPO / "scripts"))
    import s0_local_run as s0

    run = s0.S0_DIR / time.strftime("vm-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    w3, accts = s0.w3_and_accounts()
    s0.wait_for(lambda: w3.is_connected(), 60, "the hub chain")
    usdc = w3.eth.contract(address=w3.to_checksum_address(summary["usdc"]),
                           abi=s0.artifact("test/TestToken6.sol/TestToken6.json")["abi"])
    procs = s0.Procs()
    households = []
    for idx in a.households:
        h = s0.Household(idx - 1, run, accts[idx])
        h.start(procs, vm_env["ATN_JURISDICTION_REGISTRY"])
        households.append(h)
    for h in households:
        h.client = s0.DaemonClient(h.client_url(), h.name)
        await h.client.connect()
        await s0.onboard(h, j, w3, usdc)
        s0.log(f"{h.name} onboarded: agent {h.agent_id} {h.agent_address}")

    obs_token_file = run / "obs.token"
    obs_token_file.write_text(token, encoding="utf-8")
    side = [sys.executable, str(REPO / "scripts" / "s2" / "telemetry.py"), "--obs", OBS,
            "--token-file", str(obs_token_file)]
    for h in households:
        side += ["--daemon", f"{a.name_prefix}{h.name}={h.client_url()},log={h.home / 'daemon.log'}"]
    procs.start("telemetry", side, cwd=REPO, env=s0.base_env(), log_path=run / "telemetry.log")

    for h in households:
        s0.log(f"MCP for {h.name}: claude mcp add atn-{h.name} -e ATN_MCP_AGENT={h.agent_id} "
               f"-e ATN_MCP_URL={h.client_url()} -- {sys.executable} -m atn.mcp_server")
    s0.log("all households up; keep this process running")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        procs.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", required=True,
                    type=lambda s: [int(x) for x in s.split(",")],
                    help="owner indices on the hub chain, e.g. 1,2")
    ap.add_argument("--obs-token", default="")
    ap.add_argument("--obs-token-file", default="")
    ap.add_argument("--dir", default="~/autonet-wan")
    ap.add_argument("--name-prefix", default="",
                    help="prefix for telemetry names, e.g. vm1-")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
