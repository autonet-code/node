#!/usr/bin/env python3
"""End-to-end proof that one daemon switches jurisdictions and back.

docs/multi_jurisdiction.md, decisions 1-8 (backlog item E1). Stands up a
LOCAL hardhat node and deploys TWO independent Autonet jurisdictions, A and
B, through trustless-contracts' own deploy path (scripts/lib/modules.js
``deployJurisdictionSuite``, then per DAO JurisdictionFactory.deployCore +
AutonetSetup.prepareInstallation + JurisdictionFactory.finalize, the steps
scripts/deploy-autonet-jurisdiction.js takes). Each DAO's Registry carries
the jurisdiction.autonet.* keys (substrate, rep_token, tool_bond, dao,
timelock, registry, the compact v1 charter, epoch_rules "1"), and the owner
wallet holds genesis REP in both.

Then it drives the DAEMON code paths against them, one data dir throughout,
a fresh ``Runtime`` standing in for each restart:

  1. chain up, two DAOs (+ two refusable ones for stage 7);
  2. join A from a saved selection: runtime.jurisdiction, the bridge's
     jurisdiction_errors, the world dir and gossip topic keyed by it;
  3. register an owner-bound root agent (owner wallet signs the EIP-712
     binding) and a parent-signed child (``register_parent_signed``);
  4. switch to B over the WS ``autonet_switch`` handler, restart: same 0x
     addresses, not registered in B, A's registration kept, the authored
     tool still runs, the tool adopted in A is parked;
  5. register the root agent in B too;
  6. switch back to A: registered again with A's original tx, the adopted
     tool visible again, A's world dir is the one stage 2 created;
  7. refused jurisdictions (epoch rules "2"; a Registry without tool_bond):
     the WS handler refuses each, naming the problem, selection unchanged;
  8. teardown.

The trustless-contracts checkout is never written: its contracts, module
metadata, autonet code-blob fixtures and scripts/lib/modules.js are COPIED
into a scratch hardhat project under build/ (gitignored) and compiled and
run there. Every runtime uses a temp HOME, KEYSTORE_DIR and data dir.

Run:
    python scripts/local_e2e_jurisdiction_switch.py

Env: E2E_JURISDICTION_SWITCH_PORT (default 8561), TRUSTLESS_CONTRACTS_DIR
(default C:/code/dao/trustless-contracts).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DAO_DIR = Path(os.environ.get("TRUSTLESS_CONTRACTS_DIR",
                              r"C:/code/dao/trustless-contracts"))
SCRATCH = REPO / "build" / "e2e_jurisdiction_switch"
PORT = int(os.environ.get("E2E_JURISDICTION_SWITCH_PORT", "8561"))
RPC_URL = f"http://127.0.0.1:{PORT}"
CHAIN_ID = 31337

# Hardhat's deterministic mnemonic accounts (well-known, local only).
HH_DEPLOYER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # #0
HH_OWNER_PK = "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"     # #3
E18 = 10 ** 18
GENESIS_REP = 1000

# The isolation must be in place before ANY atn / nodes import: the keystore
# resolves KEYSTORE_DIR at import time, and several modules read Path.home().
_ORIG_ENV = dict(os.environ)
_TMP_HOME = Path(tempfile.mkdtemp(prefix="e2e_jsw_home_"))
os.environ["KEYSTORE_DIR"] = str(_TMP_HOME / "keystore")
os.environ["USERPROFILE"] = str(_TMP_HOME)
os.environ["HOME"] = str(_TMP_HOME)
for _k in ("ATN_REGISTRY_URL", "ATN_JURISDICTION_REGISTRY", "ATN_JURISDICTION_RPC",
           "ATN_TOOL_BOND_ADDRESS", "ATN_TOOL_BOND_DEV_OVERRIDE"):
    os.environ.pop(_k, None)
os.environ["ATN_DISABLE_TOOL_FLAG_KEEPER"] = "1"
os.environ["ATN_DISABLE_PROVIDER_AUTODETECT"] = "1"

HARDHAT_CONFIG = """require("@nomicfoundation/hardhat-ethers");
module.exports = {
  solidity: { version: "0.8.26", settings: { evmVersion: "cancun",
    optimizer: { enabled: true, runs: 200 }, viaIR: true } },
  networks: {
    hardhat: { chainId: %d, allowUnlimitedContractSize: true },
    localhost: { url: process.env.E2E_RPC_URL || "http://127.0.0.1:8545" },
  },
};
""" % CHAIN_ID

# Runs inside the scratch project with `hardhat run --network localhost`.
# Uses trustless-contracts' own suite deployer (scripts/lib/modules.js) and
# the deploy-autonet-jurisdiction.js steps, once per DAO.
DEPLOY_JS = r"""
const { ethers } = require("hardhat");
const fs = require("fs");
const { deployJurisdictionSuite, encodeInstallData, metadata } = require("./lib/modules");

async function prepare(setup, from, core, name, values) {
  await (await setup.prepareInstallation(core, encodeInstallData(ethers, name, values))).wait();
  const id = await setup.lastPreparedBy(from.address);
  const [, p] = await setup.installationOf(id);
  return { id, module: p.module };
}

async function main() {
  const [deployer] = await ethers.getSigners();
  const owner = process.env.E2E_OWNER;
  const s = await deployJurisdictionSuite(ethers, deployer, { minCoolingOff: 60 });
  const factory = s.jurisdictionFactory;
  const autonetSetup = s.setups.autonet;
  const charterUi = (name) => metadata("autonet", "build").pluginSetup.prepareInstallation.inputs
    .find((i) => i.name === "substrate").components.find((c) => c.name === name)["x-ui"];
  // ToolBond Params (trustless-contracts 448023a: staked epoch reviews).
  const bondParams = {
    minStake: 100_000_000n, revealWindow: 86400n, degradeBps: 1000n,
    authorSlashBps: 10000n, unbondDelay: 30n * 86400n, reviewerShareBps: 4000n,
    appealPeriod: 10n * 86400n, harmWeightBps: 20000n,
  };
  async function launch(name, epochRules) {
    const params = {
      name, symbol: "AREP", description: "",
      initialMembers: [owner], initialAmounts: [ethers.parseEther(process.env.E2E_GENESIS_REP)],
      votingDelay: 0, votingPeriod: 2, proposalThreshold: ethers.parseEther("10"),
      quorumFraction: 4, executionDelay: 0, supermajorityBps: 0,
    };
    const c = await factory.deployCore.staticCall(params);
    await (await factory.deployCore(params)).wait();
    const core = { registry: c.registry, timelock: c.timelock, token: c.token, governor: c.governor };
    const autonet = await prepare(autonetSetup, deployer, core, "autonet", {
      substrate: { challengeWindow: 3600n, charter: charterUi("charter").default,
                   epochRules: epochRules ?? charterUi("epochRules").fixed },
      toolBond: bondParams });
    const d = await autonetSetup.deploymentOf(autonet.id);
    await (await factory.finalize(core, [{ setup: await autonetSetup.getAddress(), id: autonet.id }],
      ["jurisdiction.name"], [name])).wait();
    return { name, governor: core.governor, rep_token: core.token, registry: core.registry,
             timelock: core.timelock, substrate: d.substrate, tool_bond: d.toolBond,
             vault: d.vault, charter: d.charter, epoch_rules: d.epochRules };
  }
  const out = { A: await launch("Autonet A"), B: await launch("Autonet B"),
                C: await launch("Autonet C", "2") };

  // D: a Registry that names every address except tool_bond (written by
  // its owner, the deployer; no factory path produces such a DAO).
  const reg = await (await ethers.getContractFactory("Registry")).deploy(deployer.address, ethers.ZeroAddress);
  await reg.waitForDeployment();
  const src = await ethers.getContractAt("Registry", out.A.registry);
  const keys = await src.getAllKeys(), values = await src.getAllValues();
  const k2 = [], v2 = [];
  keys.forEach((k, i) => {
    if (!k.startsWith("jurisdiction.autonet.") || k.endsWith(".tool_bond")) return;
    k2.push(k);
    v2.push(k.endsWith(".registry") ? "" : values[i]);
  });
  await (await reg.batchEditRegistry(k2, v2)).wait();
  out.D = { name: "no tool_bond", registry: await reg.getAddress() };
  fs.writeFileSync(process.env.E2E_OUT, JSON.stringify(out, null, 2));
}
main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
"""


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

class Stage:
    def __init__(self, num: int, name: str):
        self.num, self.name = num, name
        self.status = "PENDING"
        self.notes: List[str] = []

    def note(self, k: str, v: Any) -> None:
        self.notes.append(f"{k}={v}")


class Board:
    def __init__(self) -> None:
        self.stages: Dict[int, Stage] = {}

    def stage(self, num: int, name: str) -> Stage:
        s = Stage(num, name)
        self.stages[num] = s
        return s

    def render(self) -> str:
        lines = ["", "=" * 74, "  JURISDICTION-SWITCH LOCAL E2E: SCOREBOARD", "=" * 74]
        for num in sorted(self.stages):
            s = self.stages[num]
            mark = {"PENDING": "----"}.get(s.status, s.status)
            lines.append(f"  [{mark}] Stage {s.num}: {s.name}")
            lines.extend(f"           {n}" for n in s.notes)
        lines.append("=" * 74)
        counts = {k: sum(1 for s in self.stages.values() if s.status == k)
                  for k in ("PASS", "FAIL", "SKIP")}
        lines.append(f"  {counts['PASS']} PASS / {counts['FAIL']} FAIL / "
                     f"{counts['SKIP']} SKIP")
        lines.append("=" * 74)
        return "\n".join(lines)


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


def check(cond: bool, what: str) -> None:
    print(f"    {'PASS' if cond else 'FAIL'}  {what}", flush=True)
    if not cond:
        raise AssertionError(what)


# ---------------------------------------------------------------------------
# Scratch hardhat project (copies; trustless-contracts is never written)
# ---------------------------------------------------------------------------

def _find_hardhat_cli() -> Path:
    for base in [REPO, *REPO.parents]:
        cli = base / "node_modules" / "hardhat" / "internal" / "cli" / "cli.js"
        if cli.exists() and (base / "node_modules" / "@nomicfoundation" / "hardhat-ethers").exists():
            return cli
    raise RuntimeError("no node_modules with hardhat + hardhat-ethers at or above "
                       f"{REPO}; run npm install")


def _copy_tree(src: Path, dst: Path, skip: tuple = ()) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part in skip for part in rel.parts) or not p.is_file():
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def build_scratch_project() -> None:
    if not (DAO_DIR / "scripts" / "lib" / "modules.js").exists():
        raise RuntimeError(f"trustless-contracts not found at {DAO_DIR} "
                           "(set TRUSTLESS_CONTRACTS_DIR)")
    for sub in ("contracts", "modules", "test", "scripts"):
        if (SCRATCH / sub).exists():
            shutil.rmtree(SCRATCH / sub)
    _copy_tree(DAO_DIR / "contracts", SCRATCH / "contracts", skip=(".deps", "artifacts"))
    _copy_tree(DAO_DIR / "modules", SCRATCH / "modules")
    _copy_tree(DAO_DIR / "test" / "fixtures" / "autonet",
               SCRATCH / "test" / "fixtures" / "autonet")
    (SCRATCH / "scripts" / "lib").mkdir(parents=True)
    shutil.copy2(DAO_DIR / "scripts" / "lib" / "modules.js",
                 SCRATCH / "scripts" / "lib" / "modules.js")
    (SCRATCH / "scripts" / "e2e_deploy.js").write_text(DEPLOY_JS, encoding="utf-8")
    (SCRATCH / "hardhat.config.js").write_text(HARDHAT_CONFIG, encoding="utf-8")
    pkg = SCRATCH / "package.json"
    if not pkg.exists():
        pkg.write_text('{"name": "e2e-jurisdiction-switch", "private": true}\n',
                       encoding="utf-8")


def _node(cli: Path, *args: str, timeout: int = 900, env: Optional[dict] = None):
    return subprocess.run(["node", str(cli), *args], cwd=str(SCRATCH),
                          capture_output=True, text=True, timeout=timeout,
                          env=env or _ORIG_ENV)


def compile_scratch(cli: Path) -> None:
    log("compiling the trustless-contracts copy (viaIR; first run takes minutes)...")
    proc = _node(cli, "compile", timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")


def deploy_daos(cli: Path, owner: str) -> Dict[str, Any]:
    out = SCRATCH / "e2e_daos.json"
    out.unlink(missing_ok=True)
    env = {**_ORIG_ENV, "E2E_RPC_URL": RPC_URL, "E2E_OWNER": owner,
           "E2E_GENESIS_REP": str(GENESIS_REP), "E2E_OUT": str(out)}
    proc = _node(cli, "run", "--network", "localhost", "scripts/e2e_deploy.js",
                 timeout=900, env=env)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"deploy failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


_NODES: List[subprocess.Popen] = []


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_node(cli: Path) -> subprocess.Popen:
    if not _port_free(PORT):
        raise RuntimeError(f"port {PORT} is busy (set E2E_JURISDICTION_SWITCH_PORT)")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        ["node", str(cli), "node", "--port", str(PORT)], cwd=str(SCRATCH),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
        env=_ORIG_ENV)
    _NODES.append(proc)
    return proc


def wait_rpc(timeout: float = 60.0) -> bool:
    from web3 import Web3
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if Web3(Web3.HTTPProvider(RPC_URL)).eth.block_number >= 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_node(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

REGISTRY_ABI = [
    {"name": "getAllKeys", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string[]"}]},
    {"name": "getAllValues", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string[]"}]},
]
ERC20_ABI = [{"name": "balanceOf", "type": "function", "stateMutability": "view",
              "inputs": [{"name": "a", "type": "address"}],
              "outputs": [{"name": "", "type": "uint256"}]}]


def registry_keys(w3, addr: str) -> Dict[str, str]:
    reg = w3.eth.contract(address=w3.to_checksum_address(addr), abi=REGISTRY_ABI)
    return dict(zip(reg.functions.getAllKeys().call(), reg.functions.getAllValues().call()))


def substrate_at(w3, addr: str):
    data = json.loads((REPO / "nodes" / "common" / "abi" / "Substrate.json")
                      .read_text(encoding="utf-8"))
    abi = data["abi"] if isinstance(data, dict) else data
    return w3.eth.contract(address=w3.to_checksum_address(addr), abi=abi)


def fund_eth(w3, to: str, eth: int = 5) -> None:
    from eth_account import Account
    acct = Account.from_key(HH_DEPLOYER_PK)
    tx = {"from": acct.address, "to": to, "value": eth * E18,
          "nonce": w3.eth.get_transaction_count(acct.address), "gas": 21000,
          "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID}
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(raw))


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------

AGENTS_YAML_DIR = "agents"


async def boot_runtime(root: Path, owner: str):
    """One daemon boot on ``root``: a fresh Runtime plus the CLI's agent
    load from the agents dir (atn/cli.py _load_agents). Each call stands in
    for a daemon restart."""
    from atn import cli as atn_cli
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime

    data_dir, agents_dir = root / "data", root / AGENTS_YAML_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)
    rt.tool_store._embedder = lambda text: (0.11, 0.22, 0.33) if text else ()
    await atn_cli._load_agents(rt, config)
    return rt, config


async def add_agent(rt, config, agent_id: str, parent_id: Optional[str] = None):
    from atn.loader import save_agent
    from atn.models import AgentDefinition, AgentMode
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE, parent_id=parent_id,
        system_prompt=f"You are {agent_id}.", provider="anthropic",
        cognitive_model="claude-sonnet-5")
    await rt.register_agent(defn)
    save_agent(rt.get_agent(agent_id), config.agents_dir)
    return rt.get_agent(agent_id)


def ws_bridge(rt):
    from atn.ws_auth import ClientSession
    from atn.ws_server import WebSocketBridge
    return WebSocketBridge(rt), ClientSession(local=True, authed=True, owner=True,
                                              scope_ids=None)


async def ws_call(rt, msg: Dict[str, Any]) -> Dict[str, Any]:
    from atn import cli as atn_cli
    bridge, session = ws_bridge(rt)
    resp = await bridge._handle_message({"msg_id": "e2e", **msg}, session)
    # autonet_switch asks the CLI loop to restart; the harness restarts by
    # building a fresh Runtime instead.
    atn_cli._restart_requested = False
    return resp


def library_for(rt, tmp_world_root: Path):
    """The LibraryService the autonet bridge's start() would build for this
    jurisdiction (nodes/service.py _init_library_service), its data root
    pointed at the temp HOME. Same key derivation: start() copies the
    bridge's resolved jurisdiction_key into the node config."""
    from nodes.common import library_service
    from nodes.common.config import AutonetConfig
    from nodes.service import AutonetService
    library_service._DEFAULT_DATA_ROOT = tmp_world_root
    cfg = AutonetConfig()
    cfg.jurisdiction_key = rt.autonet.state.jurisdiction_key
    svc = AutonetService(cfg, data_dir=str(rt._config.data_dir), configure_logging=False)
    svc._init_library_service()
    return svc._library


async def register_root(rt, owner_key: str, peer: bytes) -> Dict[str, Any]:
    """The WS register path with the owner wallet's signature: the owner
    signs OwnerBinding{agent, parent=0, nonce} for this jurisdiction's
    Substrate, the agent key sends registerAgentBound, then the handler's
    success bookkeeping."""
    from eth_account import Account
    from atn.on_chain import OnChainService
    rt.autonet.ensure_network_config()
    svc = OnChainService(rt.autonet.config)
    defn = rt.get_agent("root")
    nonce = await svc.binding_nonce(defn.identity.address)
    sig = svc.sign_binding(owner_key, defn.identity.address, "", nonce)
    result = await svc.register_agent_bound(
        identity=defn.identity, private_key=rt.registry.get_agent_key("root"),
        peer_id=peer, owner=Account.from_key(owner_key).address, owner_sig=sig)
    if result.get("success"):
        defn.identity.registered_on_chain = True
        defn.identity.registration_tx = result.get("tx_hash")
        rt.registry.persist_identity("root")
    return result


async def register_child(rt, peer: bytes) -> Dict[str, Any]:
    from atn.agent_tools import register_parent_signed
    from atn.on_chain import OnChainService
    rt.autonet.ensure_network_config()
    svc = OnChainService(rt.autonet.config)
    defn = rt.get_agent("child")
    result = await register_parent_signed(
        rt, svc, defn, rt.registry.get_agent_key("child"), peer)
    if result.get("success"):
        defn.identity.registered_on_chain = True
        defn.identity.registration_tx = result.get("tx_hash", "")
        rt.registry.persist_identity("child")
    return result


def ident(rt, agent_id: str):
    return rt.get_agent(agent_id).identity


TOOL_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'upper': str(args.get('x', '')).upper()}))\n"
)
REMOTE_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'rev': str(args.get('x', ''))[::-1]}))\n"
)
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> int:
    from eth_account import Account
    from web3 import Web3

    board = Board()
    node: Optional[subprocess.Popen] = None
    root = Path(tempfile.mkdtemp(prefix="e2e_jsw_daemon_"))
    remote_root = Path(tempfile.mkdtemp(prefix="e2e_jsw_remote_"))
    world_root = _TMP_HOME / "world"
    owner = Account.from_key(HH_OWNER_PK).address
    failed = False

    def fail(stage: Stage, exc: BaseException) -> None:
        nonlocal failed
        stage.status = "FAIL"
        stage.note("error", repr(exc)[:400])
        log(traceback.format_exc())
        failed = True

    def finish() -> int:
        s8 = board.stage(8, "teardown (hardhat killed, temps removed)")
        kill_node(node)
        for d in (root, remote_root, _TMP_HOME):
            shutil.rmtree(d, ignore_errors=True)
        s8.status = "PASS"
        print(board.render())
        return 1 if failed else 0

    # ------------------------------------------------------------------
    s1 = board.stage(1, "chain up: one hardhat node, DAOs A and B via the DAO deploy path")
    log("Stage 1: chain up")
    daos: Dict[str, Any] = {}
    try:
        cli = _find_hardhat_cli()
        build_scratch_project()
        compile_scratch(cli)
        node = start_node(cli)
        if not wait_rpc():
            raise RuntimeError("hardhat RPC never came up")
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        check(w3.eth.chain_id == CHAIN_ID, f"chain id {CHAIN_ID} on port {PORT}")
        log("deploying the jurisdiction suite + DAOs A, B, C and registry D...")
        daos = deploy_daos(cli, owner)
        from nodes.common.library.charter import DEFAULT_CHARTER
        CHARTER = DEFAULT_CHARTER.entries()
        v1 = json.dumps({"axes": [{"id": c["id"], "thesis": c["thesis"]} for c in CHARTER]},
                        separators=(",", ":"))
        for name in ("A", "B"):
            d = daos[name]
            keys = registry_keys(w3, d["registry"])
            p = "jurisdiction.autonet."
            for k, want in (("substrate", d["substrate"]), ("rep_token", d["rep_token"]),
                            ("tool_bond", d["tool_bond"]), ("dao", d["governor"]),
                            ("timelock", d["timelock"]), ("registry", d["registry"])):
                check(keys.get(p + k, "").lower() == want.lower(), f"{name}: {p}{k}")
            check(keys.get(p + "charter") == v1, f"{name}: charter = compact v1 CHARTER JSON")
            check(keys.get(p + "epoch_rules") == "1", f"{name}: epoch_rules = 1")
            rep = w3.eth.contract(address=d["rep_token"], abi=ERC20_ABI)
            check(rep.functions.balanceOf(owner).call() == GENESIS_REP * E18,
                  f"{name}: owner wallet holds {GENESIS_REP} genesis REP")
            s1.note(name, f"dao={d['governor']} registry={d['registry']} substrate={d['substrate']}")
        check(daos["A"]["substrate"] != daos["B"]["substrate"], "A and B are independent")
        s1.status = "PASS"
    except Exception as e:
        fail(s1, e)
        return finish()

    A, B = daos["A"], daos["B"]
    key_a = f"{CHAIN_ID}:{A['governor'].lower()}"
    key_b = f"{CHAIN_ID}:{B['governor'].lower()}"
    rt = config = None
    world_a_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    s2 = board.stage(2, "join A from a saved selection")
    log("Stage 2: join A")
    try:
        from atn.config import check_jurisdiction, save_selected_jurisdiction
        from nodes.common.event_gossip import topic_for_rpb
        chk = check_jurisdiction(A["registry"], RPC_URL)
        check(chk["ok"] and chk["key"] == key_a, f"check_jurisdiction(A) ok, key {chk.get('key')}")
        save_selected_jurisdiction({**chk, "name": "Autonet A"}, root / "data")
        rt, config = await boot_runtime(root, owner)
        check(rt.jurisdiction == key_a, f"runtime.jurisdiction == {key_a}")
        check(rt.registry.jurisdiction == key_a, "agent registry runs A")
        check(rt.autonet.ensure_network_config(), "bridge joined the network")
        st = rt.autonet.state
        check(st.jurisdiction_errors == [], f"jurisdiction_errors == [] ({st.jurisdiction_errors})")
        check(st.jurisdiction_key == key_a, "bridge jurisdiction_key == A")
        check(st.substrate_address.lower() == A["substrate"].lower()
              and st.tool_bond_address.lower() == A["tool_bond"].lower()
              and st.rpc_url == RPC_URL, "bridge resolved A's substrate, tool_bond and rpc")
        ws = library_for(rt, world_root)
        check(ws.rpb_address == key_a and ws.rpb_address != "default",
              f"library rpb id is the jurisdiction key ({ws.rpb_address})")
        world_a_dir = Path(ws.state_dir)
        check(world_a_dir.parent == world_root and world_a_dir.name != "default",
              f"world dir {world_a_dir.name}")
        check(topic_for_rpb(ws.rpb_address) == f"/autonet/world-events/{key_a}",
              "gossip topic keyed by the jurisdiction")
        (world_a_dir / "e2e_marker").write_text("stage 2", encoding="utf-8")
        ws.shutdown()
        s2.note("key", key_a)
        s2.note("world_dir", world_a_dir.name)
        s2.status = "PASS"
    except Exception as e:
        fail(s2, e)
        return finish()

    # ------------------------------------------------------------------
    s3 = board.stage(3, "register owner-bound root + parent-signed child in A; author + adopt tools")
    log("Stage 3: register in A")
    addrs: Dict[str, str] = {}
    tx_a: Dict[str, str] = {}
    authored = adopted = ""
    try:
        from atn.agent_tools import execute_tool
        sub_a = substrate_at(w3, A["substrate"])
        for aid, parent in (("root", None), ("child", "root")):
            defn = await add_agent(rt, config, aid, parent)
            addrs[aid] = Web3.to_checksum_address(defn.identity.address)
            fund_eth(w3, addrs[aid])
        r = await register_root(rt, HH_OWNER_PK, b"peer-root")
        check(r.get("success") is True, f"root registered (owner-signed): {r.get('error', '')}")
        c = await register_child(rt, b"peer-child")
        check(c.get("success") is True, f"child registered (parent-signed): {c.get('error', '')}")
        for aid in ("root", "child"):
            check(sub_a.functions.isLive(addrs[aid]).call(), f"A: isLive({aid})")
            i = ident(rt, aid)
            check(i.registered_on_chain and key_a in i.registrations,
                  f"{aid}: registrations has A")
            tx_a[aid] = i.registrations[key_a]
        check(sub_a.functions.getAgentOwner(addrs["child"]).call() == owner,
              "child landed in the owner's household")
        check(sub_a.functions.getAgentParent(addrs["child"]).call() == addrs["root"],
              "child's parent link = root")

        # External agents (atn/external_agents.md): a harness session creates
        # its agent over WS and registers it self-owned; the daemon's own
        # owner key signs, and every self-owned agent shares that household.
        from atn.ws_server import WebSocketBridge
        WebSocketBridge._read_libp2p_peer_id = staticmethod(lambda _a: "12D3KooWExtAgentPeer")
        ext_owner = ""
        for eid in ("ext1", "ext2"):
            cr = await ws_call(rt, {"type": "create_agent", "id": eid,
                                    "provider": "external", "harness": "e2e"})
            check(cr.get("ok") and cr["result"].get("external"),
                  f"{eid}: external agent created: {cr.get('error', '')}")
            addrs[eid] = Web3.to_checksum_address(cr["result"]["address"])
            fund_eth(w3, addrs[eid])
            rg = await ws_call(rt, {"type": "register_agent_on_chain",
                                    "agent_id": eid, "self_owned": True})
            check(rg.get("ok") is True,
                  f"{eid}: self-owned registration: {rg.get('error') or rg.get('result', {}).get('error', '')}")
            check(sub_a.functions.isLive(addrs[eid]).call(), f"A: isLive({eid})")
            o = sub_a.functions.getAgentOwner(addrs[eid]).call()
            ext_owner = ext_owner or o
            check(o == ext_owner and o != owner,
                  f"{eid}: bound to the daemon's own owner key ({o})")
        check(await rt.trigger_run("ext1") is None, "the daemon never runs an external agent")

        # An authored tool (stays usable everywhere).
        reg = await execute_tool("register_tool", {
            "name": "upper_case", "description": "Upper-cases a string.",
            "input_schema": SCHEMA, "code": TOOL_CODE,
            "novelty": {"statement": "Upper-cases a string; no comparable tool exists.", "prior_art": []}},
            rt, caller_id="root")
        check("digest" in reg, f"authored tool registered: {reg.get('error', '')}")
        authored = reg["digest"]
        used = await execute_tool("use_tool", {"name": "upper_case", "arguments": {"x": "ab"}},
                                  rt, caller_id="root")
        check("ab".upper() in json.dumps(used), f"authored tool runs in A: {str(used)[:120]}")

        # An adopted tool: authored by ANOTHER daemon, fetched over a blob
        # fetcher, proposed by an agent and approved by the owner in A.
        rr, rconfig = await boot_runtime(remote_root, owner)
        await add_agent(rr, rconfig, "remote-author")
        rreg = await execute_tool("register_tool", {
            "name": "reverse_text", "description": "Reverses a string.",
            "input_schema": SCHEMA, "code": REMOTE_CODE,
            "novelty": {"statement": "Reverses a string; no comparable tool exists.", "prior_art": []}},
            rr, caller_id="remote-author")
        check("digest" in rreg, f"remote tool registered: {rreg.get('error', '')}")
        adopted = rreg["digest"]
        remote_blobs = rr.tool_store._blob_store()

        async def fetch_remote(digest: str):
            return remote_blobs.get_bytes(digest)
        rt.tool_store.blob_fetcher = fetch_remote
        prop = await rt.tool_store.propose_adoption("root", adopted, "e2e")
        check(prop.get("status") == "pending", f"adoption proposed: {prop}")
        appr = await rt.tool_store.approve_adoption(adopted)
        check("error" not in appr, f"adoption approved: {appr}")
        rec = rt.tool_store.get(adopted)
        check(rec is not None and rec.origin == "adopted" and rec.jurisdiction == key_a,
              "adopted record belongs to A")
        s3.note("root", addrs["root"])
        s3.note("child", addrs["child"])
        s3.note("tx_root_A", tx_a["root"][:18] + "...")
        s3.status = "PASS"
    except Exception as e:
        fail(s3, e)
        return finish()

    # ------------------------------------------------------------------
    s4 = board.stage(4, "switch to B (WS autonet_switch, restart)")
    log("Stage 4: switch to B")
    try:
        sw = await ws_call(rt, {"type": "autonet_switch", "dao_registry": B["registry"],
                                "rpc_url": RPC_URL, "name": "Autonet B"})
        check(sw.get("ok") is True and sw["result"]["switched"] is True
              and sw["result"]["key"] == key_b, f"autonet_switch -> B: {sw.get('error', '')}")
        rt, config = await boot_runtime(root, owner)
        check(rt.jurisdiction == key_b, "runtime.jurisdiction == B after restart")
        rt.autonet.ensure_network_config()
        st = rt.autonet.state
        check(st.jurisdiction_errors == [] and st.jurisdiction_key == key_b
              and st.substrate_address.lower() == B["substrate"].lower(),
              "bridge resolved B, no errors")
        for aid in ("root", "child"):
            i = ident(rt, aid)
            check(Web3.to_checksum_address(i.address) == addrs[aid], f"{aid}: same 0x")
            check(i.registered_on_chain is False, f"{aid}: registered_on_chain False in B")
            check(i.registrations.get(key_a) == tx_a[aid] and key_b not in i.registrations,
                  f"{aid}: registrations still has A only")
        sub_b = substrate_at(w3, B["substrate"])
        check(not sub_b.functions.isRegistered(addrs["root"]).call(), "B: root not registered")
        used = await execute_tool("use_tool", {"name": "upper_case", "arguments": {"x": "cd"}},
                                  rt, caller_id="root")
        check("CD" in json.dumps(used), f"authored tool still runs in B: {str(used)[:120]}")
        check(rt.tool_store.get(adopted) is None, "adopted tool hidden in B")
        check(all(r.digest != adopted for r in rt.tool_store.visible_to(None)),
              "adopted tool not visible to the owner in B")
        check(any(r.digest == adopted for r in rt.tool_store._parked), "adopted tool parked")
        on_disk = (root / "data" / "tools" / "registry.jsonl").read_text(encoding="utf-8")
        check(adopted in on_disk, "parked record kept on disk")
        view = await ws_call(rt, {"type": "autonet_jurisdictions"})
        check(view["result"]["current"]["key"] == key_b
              and [r["key"] for r in view["result"]["registered_in"]] == [key_a],
              "autonet_jurisdictions: current B, registered in A")
        ws = library_for(rt, world_root)
        world_b_dir = Path(ws.state_dir)
        check(ws.rpb_address == key_b and world_b_dir != world_a_dir, "B gets its own world dir")
        check((world_a_dir / "e2e_marker").exists(), "A's world dir parked, not deleted")
        ws.shutdown()
        s4.status = "PASS"
    except Exception as e:
        fail(s4, e)

    # ------------------------------------------------------------------
    s5 = board.stage(5, "register the root agent in B too")
    log("Stage 5: register in B")
    tx_b_root = ""
    try:
        r = await register_root(rt, HH_OWNER_PK, b"peer-root")
        check(r.get("success") is True, f"root registered in B: {r.get('error', '')}")
        sub_b = substrate_at(w3, B["substrate"])
        check(sub_b.functions.isLive(addrs["root"]).call(), "B: isLive(root)")
        i = ident(rt, "root")
        check(set(i.registrations) == {key_a, key_b}, "root registrations has A and B")
        tx_b_root = i.registrations[key_b]
        check(tx_b_root != tx_a["root"], "B registration is its own tx")
        persisted = json.loads((config.agents_dir / "root" / "identity.json")
                               .read_text(encoding="utf-8"))
        check(set(persisted["registrations"]) == {key_a, key_b}, "identity.json keeps both")
        s5.status = "PASS"
    except Exception as e:
        fail(s5, e)

    # ------------------------------------------------------------------
    s6 = board.stage(6, "switch back to A (restart)")
    log("Stage 6: switch back to A")
    try:
        sel_before = (root / "data" / "jurisdiction.json").read_text(encoding="utf-8")
        sw = await ws_call(rt, {"type": "autonet_switch", "dao_registry": A["registry"],
                                "rpc_url": RPC_URL, "name": "Autonet A"})
        check(sw.get("ok") is True and sw["result"]["key"] == key_a, "autonet_switch -> A")
        check((root / "data" / "jurisdiction.json").read_text(encoding="utf-8") != sel_before,
              "selection rewritten")
        rt, config = await boot_runtime(root, owner)
        check(rt.jurisdiction == key_a, "runtime.jurisdiction == A again")
        rt.autonet.ensure_network_config()
        check(rt.autonet.state.jurisdiction_errors == [], "no jurisdiction errors")
        for aid in ("root", "child"):
            i = ident(rt, aid)
            check(i.registered_on_chain is True and i.registration_tx == tx_a[aid],
                  f"{aid}: registered in A with its original tx")
        check(ident(rt, "root").registrations.get(key_b) == tx_b_root,
              "root keeps its B registration on disk")
        rec = rt.tool_store.get(adopted)
        check(rec is not None and rec.origin == "adopted", "adopted tool visible again in A")
        check(rt.tool_store.get(authored) is not None, "authored tool still there")
        ws = library_for(rt, world_root)
        check(Path(ws.state_dir) == world_a_dir, "same world dir as stage 2")
        check((world_a_dir / "e2e_marker").read_text(encoding="utf-8") == "stage 2"
              and (world_a_dir / "blobs").is_dir(),
              "A's parked library data intact (stage 2 blobs + marker)")
        ws.shutdown()
        s6.status = "PASS"
    except Exception as e:
        fail(s6, e)

    # ------------------------------------------------------------------
    s7 = board.stage(7, "refused jurisdictions leave the selection unchanged")
    log("Stage 7: refused jurisdictions")
    try:
        sel_path = root / "data" / "jurisdiction.json"
        for name, needle in (("C", "epoch rules version 2"), ("D", "tool_bond")):
            before = sel_path.read_text(encoding="utf-8")
            resp = await ws_call(rt, {"type": "autonet_switch",
                                      "dao_registry": daos[name]["registry"],
                                      "rpc_url": RPC_URL})
            err = resp.get("error", "")
            check(resp.get("ok") is False and needle in err,
                  f"{name} refused, error names it: {err[:140]}")
            check(sel_path.read_text(encoding="utf-8") == before, f"{name}: selection unchanged")
            s7.note(name, err[:100])
        from atn.config import load_selected_jurisdiction
        check(load_selected_jurisdiction(root / "data")["dao_registry"] == A["registry"],
              "still selected: A")
        s7.status = "PASS"
    except Exception as e:
        fail(s7, e)

    s9 = board.stage(9, "A's anchored mints still claimable after returning")
    s9.status = "SKIP"
    s9.note("reason", "anchoring is governor-only (a DAO proposal through the "
            "timelock) and each epoch's mint is capped by fees burned in the "
            "previous anchor window, which needs ATN bought through the vault; "
            "not practical in this harness")

    return finish()


if __name__ == "__main__":
    try:
        rc = asyncio.run(amain())
    finally:
        for _proc in _NODES:
            kill_node(_proc)
    sys.exit(rc)
