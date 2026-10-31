#!/usr/bin/env python3
"""A local network for the owner to test by hand.

The S0 setup (``s0_local_run.py``) with one difference: besides two scripted
households that keep working, there is a fresh household for the owner, on
the default local port (7700) so the app finds it. Nothing in it is set up:
the owner connects a wallet in the app and does the rest (bind, register,
delegate, publish, adopt, claim, buy a service).

The owner's wallet is a hardhat account (``MANUAL_OWNER_INDEX``, default #5;
its key is printed, local chain only). It holds genesis REP, test
stablecoin and test ETH. Add the chain to the wallet: RPC
http://127.0.0.1:8545, chain id 31337.

Run:
    python scripts/manual_local_net.py

Everything stays up until Ctrl+C. Env: MANUAL_DIR (default
D:/autonet_manual), MANUAL_OWNER_INDEX, MANUAL_EPOCH_SECS (what the scripted
payments aim epochs at, default 240), TRUSTLESS_CONTRACTS_DIR.
"""
from __future__ import annotations

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

MANUAL_DIR = Path(os.environ.get("MANUAL_DIR", r"D:/autonet_manual"))
OWNER_INDEX = int(os.environ.get("MANUAL_OWNER_INDEX", "5"))
EPOCH_SECS = int(os.environ.get("MANUAL_EPOCH_SECS", "240"))
SCRIPTED = 2
OWNER_WS_PORT = 7700
OWNER_P2P_PORT = 4199


class OwnerHousehold(s0.Household):
    """The owner's household: default ports, no owner wallet preset."""

    def __init__(self, run: Path, owner) -> None:
        super().__init__(SCRIPTED, run, owner)
        self.name = "you"
        self.home = run / "you"
        self.ws_port = OWNER_WS_PORT
        self.p2p_port = OWNER_P2P_PORT

    def write_config(self) -> None:
        import yaml
        atn_home = self.home / ".atn"
        atn_home.mkdir(parents=True, exist_ok=True)
        cfg = {
            "autonet": {"enabled": True, "local_ws_port": self.ws_port,
                        "remote_ws_port": self.ws_port + 1,
                        "rpc_url": s0.RPC_URL, "chain_id": s0.CHAIN_ID},
            "voice": {"enabled": False},
            "auto_update": {"enabled": False},
        }
        (atn_home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                              encoding="utf-8")


async def amain() -> int:
    run = MANUAL_DIR / time.strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    s0.log(f"run dir {run}")
    procs = s0.Procs()
    try:
        procs.start("hardhat", ["npx.cmd" if os.name == "nt" else "npx", "hardhat", "node",
                                "--port", str(s0.RPC_PORT)], cwd=s0.DAO_DIR, env=s0.base_env(),
                    log_path=run / "hardhat.log")
        w3, accts = s0.w3_and_accounts()
        s0.wait_for(lambda: w3.is_connected() and w3.eth.chain_id == s0.CHAIN_ID, 120, "hardhat")
        w3.provider.make_request("evm_setAutomine", [False])
        w3.provider.make_request("evm_setIntervalMining", [1000])
        deployer, scripted = accts[0], accts[1:1 + SCRIPTED]
        me = accts[OWNER_INDEX]
        usdc_addr = s0.deploy_test_usdc(w3, deployer)
        usdc = w3.eth.contract(address=usdc_addr,
                               abi=s0.artifact("test/TestToken6.sol/TestToken6.json")["abi"])
        for o in scripted + [me]:
            s0.send(w3, deployer, usdc.functions.transfer(o.address, s0.USDC_PER_OWNER))

        relay_dir = run / "relay"
        relay_dir.mkdir()
        relay_args = [sys.executable, "-m", "nodes.relay", "--port", str(s0.RELAY_PORT),
                      "--host", "127.0.0.1", "--transport", "ws", "--no-upnp",
                      "--data-dir", str(relay_dir)]
        procs.start("relay-probe", relay_args + ["--jurisdiction", "31337:0x0"],
                    cwd=REPO, env=s0.base_env(), log_path=run / "relay_probe.log")
        addr = s0.wait_for(lambda: next((ln.split("relay address: ", 1)[1].strip()
                                         for ln in (run / "relay_probe.log").read_text(
                                             encoding="utf-8", errors="replace").splitlines()
                                         if "relay address: " in ln), None), 90, "relay address")
        procs.stop("relay-probe")
        bootstrap = s0.seed_addr(addr.rsplit("/p2p/", 1)[1])

        j = s0.deploy_jurisdiction(run, usdc_addr, [o.address for o in scripted + [me]],
                                   bootstrap)
        jkey = f"{s0.CHAIN_ID}:{j['dao']['governor'].lower()}"
        procs.start("relay", relay_args + ["--jurisdiction", jkey], cwd=REPO,
                    env=s0.base_env(), log_path=run / "relay.log")
        registry = j["dao"]["registry"]

        hs = [s0.Household(i, run, o) for i, o in enumerate(scripted)]
        mine = OwnerHousehold(run, me)
        for h in hs + [mine]:
            h.start(procs, registry)
        for h in hs + [mine]:
            h.client = s0.DaemonClient(h.client_url(), h.name)
            await h.client.connect()
            s0.log(f"{h.name} up on :{h.ws_port}")
        for h in hs:
            await s0.onboard(h, j, w3, usdc)

        info = {
            "app_daemon": f"ws://127.0.0.1:{OWNER_WS_PORT}",
            "wallet_network": {"rpc": s0.RPC_URL, "chain_id": s0.CHAIN_ID},
            "owner_wallet": me.address,
            "owner_key_local_only": me.key.hex(),
            "test_stablecoin": usdc_addr,
            "jurisdiction": j,
            "your_daemon_log": str(mine.home / "daemon.log"),
        }
        (run / "manual.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        s0.log(f"READY: your daemon on :{OWNER_WS_PORT}, wallet {me.address}; "
               f"details in {run / 'manual.json'}")

        stop = asyncio.Event()
        col = asyncio.create_task(s0.collector(run, hs, j, w3, stop))
        act = s0.Activity(run, hs, j, w3)
        await act.author()
        await act.adopt()
        s0.PHASES = [(1.0, EPOCH_SECS)]
        while True:
            await s0.activity_loop(act, 24 * 60)
        await col  # unreachable; Ctrl+C ends the run
    except Exception:
        s0.log(traceback.format_exc())
        return 1
    finally:
        procs.stop()


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
