#!/usr/bin/env python3
"""Bring a hand-made local jurisdiction to life.

For the owner's manual test: hardhat is already running and the owner
creates the jurisdiction in werule (local mode) with the autonet module.
This script waits for that DAO (JurisdictionFactory ``CoreDeployed`` whose
Registry gets ``jurisdiction.autonet.substrate``), then starts:

  - the relay, with the peer id the owner put in the wizard's bootstrap
    field (``--relay-dir`` keeps its key);
  - two scripted households (hardhat #1 and #2, as in S0) that onboard and
    keep working: publish tools, use and review each other's, pay each
    other, aiming epochs at MANUAL_EPOCH_SECS;
  - a fresh household for the owner on 7700, the app's default port.

The scripted owners only get REP if the owner listed them as members in the
wizard; without it they still work, but their use never passes the
breakthrough gate. Their test stablecoin comes from hardhat #0.

Run:
    python scripts/manual_join_local.py --factory 0x... --relay-dir D:/.../relay

Everything stays up until Ctrl+C.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import s0_local_run as s0  # noqa: E402
from manual_local_net import OwnerHousehold, SCRIPTED  # noqa: E402

EPOCH_SECS = int(os.environ.get("MANUAL_EPOCH_SECS", "240"))
CORE_DEPLOYED_ABI = [{"anonymous": False, "name": "CoreDeployed", "type": "event", "inputs": [
    {"indexed": True, "name": "originator", "type": "address"},
    {"indexed": False, "name": "registry", "type": "address"},
    {"indexed": False, "name": "timelock", "type": "address"},
    {"indexed": False, "name": "token", "type": "address"},
    {"indexed": True, "name": "governor", "type": "address"}]}]


def wait_for_jurisdiction(w3, factory: str, from_block: int) -> dict:
    """The first DAO the factory creates from ``from_block`` on whose Registry
    carries an autonet suite, as an S0-shaped jurisdiction dict."""
    from atn.config import _fetch_onchain_registry
    f = w3.eth.contract(address=w3.to_checksum_address(factory), abi=CORE_DEPLOYED_ABI)
    s0.log(f"waiting for a DAO from JurisdictionFactory {factory}")
    while True:
        for ev in f.events.CoreDeployed().get_logs(from_block=from_block):
            doc = _fetch_onchain_registry(ev.args.registry, s0.RPC_URL, "autonet")
            if doc is None:
                continue
            c = doc["jurisdictions"]["autonet"]["contracts"]
            return {"dao": {"governor": ev.args.governor, "registry": ev.args.registry,
                            "timelock": ev.args.timelock, "repToken": ev.args.token},
                    "autonet": {"substrate": c["substrate"], "vault": c.get("vault", "")},
                    "contracts": c}
        time.sleep(3)


async def amain(args) -> int:
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    procs = s0.Procs()
    try:
        w3, accts = s0.w3_and_accounts()
        j = wait_for_jurisdiction(w3, args.factory, args.from_block)
        (run / "jurisdiction.json").write_text(json.dumps(j, indent=2), encoding="utf-8")
        s0.log(f"jurisdiction: governor {j['dao']['governor']} registry {j['dao']['registry']}")
        jkey = f"{s0.CHAIN_ID}:{j['dao']['governor'].lower()}"
        procs.start("relay", [sys.executable, "-m", "nodes.relay", "--port", str(s0.RELAY_PORT),
                              "--host", "127.0.0.1", "--transport", "ws", "--no-upnp",
                              "--data-dir", args.relay_dir, "--jurisdiction", jkey],
                    cwd=REPO, env=s0.base_env(), log_path=run / "relay.log")

        deployer, scripted = accts[0], accts[1:1 + SCRIPTED]
        usdc = w3.eth.contract(address=w3.to_checksum_address(args.stablecoin),
                               abi=s0.artifact("test/TestToken6.sol/TestToken6.json")["abi"])
        for o in scripted:
            s0.send(w3, deployer, usdc.functions.transfer(o.address, s0.USDC_PER_OWNER))

        hs = [s0.Household(i, run, o) for i, o in enumerate(scripted)]
        mine = OwnerHousehold(run, None)
        registry = j["dao"]["registry"]
        for h in hs + [mine]:
            h.start(procs, registry)
        for h in hs + [mine]:
            h.client = s0.DaemonClient(h.client_url(), h.name)
            try:
                await h.client.connect()
            except Exception:
                s0.log(f"{h.name} boot failed:\n{h.log_tail()}")
                raise
            s0.log(f"{h.name} up on :{h.ws_port}")
        for h in hs:
            await s0.onboard(h, j, w3, usdc)
        s0.log(f"READY: your daemon on :{mine.ws_port}, log {mine.home / 'daemon.log'}")

        stop = asyncio.Event()
        asyncio.create_task(s0.collector(run, hs, j, w3, stop))
        act = s0.Activity(run, hs, j, w3)
        await act.author()
        await act.adopt()
        s0.PHASES = [(1.0, EPOCH_SECS)]
        while True:
            await s0.activity_loop(act, 24 * 60)
    except Exception:
        s0.log(traceback.format_exc())
        return 1
    finally:
        procs.stop()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--factory", required=True, help="JurisdictionFactory address")
    p.add_argument("--stablecoin", required=True, help="the vault's test stablecoin")
    p.add_argument("--relay-dir", required=True, help="relay data dir (keeps its peer id)")
    p.add_argument("--run-dir", default=time.strftime("D:/autonet_manual/join-%Y%m%d-%H%M%S"))
    p.add_argument("--from-block", type=int, default=0)
    try:
        return asyncio.run(amain(p.parse_args()))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
