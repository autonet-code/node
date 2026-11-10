#!/usr/bin/env python3
"""The economy proof-of-life: one turn of the household economy loop, end to
end on a local chain (Decisions 2026-11-07, 2026-11-08, 2026-11-09 and
2026-11-10; docs/economy_design.md).

Stands up a LOCAL hardhat node with this repo's Substrate and ServiceMarket
(ServiceRegistry + PaymentChannel) and the jurisdiction contracts the loop
needs from trustless-contracts (a Registry holding the charter and the
treasury, RepToken with ToolBond as slasher, ToolBond with short windows,
AutonetRepClaim as a RepToken minter). Three households run on it, each an
owner wallet plus a daemon whose operator key is registered on Substrate
bound to that wallet, and an external agent that acts on chain as the
operator. The DAEMON code paths drive every step:

  1. chain up;
  2. households: genesis REP to the owner wallets, ATN bought from the vault
     for the operators, the genesis anchor;
  3. household A's agent publishes a tool with a ToolBond stake
     (``register_tool`` + ``publish_tool``);
  4. households B and C adopt it, use it and review it (``attest_tools``
     with the four fields); each daemon's ``ReviewDriver`` commits, reveals
     after the anchor, finalizes and claims on ToolBond; the reviewing
     households clear the conclusion gate;
  5. B buys a service C sells (``pay_for_service`` + ``request_service``):
     the payment's fee is burned, which is what funds the next pool;
  6. ``read_voice_state`` pinned to the anchor after the burn feeds
     ``federated_epoch_close``: the tool concludes and is paid once from its
     reviews (40% to the reviewers by accuracy, the author side by
     V x C x S kept by N). Each daemon's ``ChainSubmissionDriver`` takes the
     close: A's anchors it (operator A is Substrate's governor on this dev
     network), then every operator claims its owner wallet's ATN
     (``claimFor``) and shows ToolBond the conclusion;
  7. the same driver step claims each household's REP 1:1 on its tool-pool
     earnings (``AutonetRepClaim.claim``, ``agentMintTotal``);
  8. B buys C's service again through a payment channel: two vouchers, C's
     settlement sweep closes the channel with the best one, B's refund
     sweep withdraws the rest after the challenge window;
  9. teardown.

The jurisdiction contracts are NOT vendored: they are copied at run time
from a trustless-contracts checkout (env TRUSTLESS_CONTRACTS_DIR, default
C:/code/dao/trustless-contracts) into a scratch hardhat project under
build/ (gitignored), compiled there with Substrate.sol and ServiceMarket.sol
from this repo. The checkout itself is never written. HOME, USERPROFILE and
KEYSTORE_DIR point at a temp dir for the whole run. The chain, setup and
helpers follow scripts/local_e2e_tool_security.py.

Run:
    python scripts/local_e2e_tool_economy.py

See docs/local_e2e.md.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
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

# The isolation must be in place before ANY atn / nodes import: the keystore
# resolves KEYSTORE_DIR at import time, and several modules read Path.home().
_ORIG_ENV = dict(os.environ)
_TMP_HOME = Path(tempfile.mkdtemp(prefix="e2e_econ_home_"))
os.environ["KEYSTORE_DIR"] = str(_TMP_HOME / "keystore")
os.environ["USERPROFILE"] = str(_TMP_HOME)
os.environ["HOME"] = str(_TMP_HOME)
for _k in ("ATN_REGISTRY_URL", "ATN_JURISDICTION_REGISTRY", "ATN_JURISDICTION_RPC",
           "ATN_TOOL_BOND_ADDRESS", "ATN_TOOL_BOND_DEV_OVERRIDE", "ATN_ARTIFACTS_DIR"):
    os.environ.pop(_k, None)
os.environ["ATN_DISABLE_PROVIDER_AUTODETECT"] = "1"
os.environ["ATN_DISABLE_REVIEW_DRIVER"] = "1"   # the harness ticks drivers itself
os.environ["ATN_SEARCH_MODEL"] = "off"

DAO_DIR = Path(os.environ.get("TRUSTLESS_CONTRACTS_DIR",
                              r"C:/code/dao/trustless-contracts"))
SCRATCH = REPO / "build" / "e2e_tool_economy"
PORT = int(os.environ.get("E2E_TOOL_ECONOMY_PORT", "8558"))
RPC_URL = f"http://127.0.0.1:{PORT}"
CHAIN_ID = 1337

# Hardhat's deterministic mnemonic accounts (well-known, local only).
HH_DEPLOYER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # #0
HH_TIMELOCK_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"  # #1
HH_TREASURY_PK = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"  # #2

E18 = 10 ** 18
ATN = 10 ** 6            # ATN has 6 decimals
MIN_STAKE = 100 * ATN
PARAMS = {  # ToolBond.Params (8dcc1be), short windows for a local run
    "minStake": MIN_STAKE, "revealWindow": 300, "degradeBps": 1000,
    "authorSlashBps": 10_000, "unbondDelay": 120, "reviewerShareBps": 4000,
}
SERVICE_FEE_BPS = 250  # Substrate default
FEE_TREASURY_BPS = 5000
CHANNEL_WINDOW = 60    # PaymentChannel challenge window, seconds
CHARTER_KEY = "jurisdiction.autonet.charter"
SERVICE_ASK = 400 * ATN  # one request of C's service: 10 ATN fee, 5 ATN burned

HOUSES = ("A", "B", "C")
AGENT = {h: f"agent-{h.lower()}" for h in HOUSES}
# REP at the genesis anchor, held by the owner wallets (the households).
REP_GENESIS = {"A": 100, "B": 300, "C": 200}
# ATN each operator buys from the vault (bond; service purchases; gas is ETH).
ATN_BOUGHT = {"A": 1000, "B": 2000, "C": 100}

# The reviews of epoch 1 (B holds 300 REP, C 200). Only B scores a charter
# axis; the others count 0.5 for each review.
REVIEWS = {
    "B": {"correctness": 0.9, "simplicity": 0.8, "novelty": 0.7, "malware": False,
          "axes": {"promotion_of_intelligence": 0.6}},
    "C": {"correctness": 0.5, "simplicity": 0.6, "novelty": 0.5, "malware": False},
}

DAO_SOURCES = ["ToolBond.sol", "RepToken.sol", "Registry.sol", "IAdminToken.sol",
               "IJurisdictionData.sol", "AutonetRepClaim.sol"]
# This repo's contracts go under contracts/core/, the Hardhat layout the
# daemon's ABI loader reads (ATN_ARTIFACTS_DIR, nodes/common/contract_artifacts).
REPO_SOURCES = ["Substrate.sol", "ServiceMarket.sol"]

HARDHAT_CONFIG = """module.exports = {
  solidity: { version: "0.8.26", settings: { evmVersion: "cancun",
    optimizer: { enabled: true, runs: 50 }, viaIR: true } },
  networks: { hardhat: { chainId: %d, allowUnlimitedContractSize: true } },
};
""" % CHAIN_ID

UPPER_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'upper': str(args.get('x', '')).upper()}))\n"
)
WORDS_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'words': len(str(args.get('text', '')).split())}))\n"
)
UPPER_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}
WORDS_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}

STAGES = [
    (1, "chain up (scratch compile, hardhat, Substrate + DAO suite + ServiceMarket)"),
    (2, "households (three daemons; operators bound to owner wallets; genesis REP + anchor)"),
    (3, "household A's agent publishes a bonded tool (register_tool + publish_tool)"),
    (4, "B and C adopt, use and review it (four fields, commit-reveal on ToolBond)"),
    (5, "service fees burned (B buys C's service through payForService)"),
    (6, "close: the tool concludes and is paid; anchor; operators claimFor their owners"),
    (7, "REP claimed 1:1 on tool-pool earnings (AutonetRepClaim)"),
    (8, "payment channel: vouchers, provider closes with its best, buyer withdraws the rest"),
    (9, "teardown"),
]


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
        lines = ["", "=" * 74, "  TOOL ECONOMY (HOUSEHOLD LOOP) LOCAL E2E: SCOREBOARD",
                 "=" * 74]
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
    """Assert with a printed PASS/FAIL line per check."""
    print(f"    {'PASS' if cond else 'FAIL'}  {what}", flush=True)
    if not cond:
        raise AssertionError(what)


def close_to(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def hundredths_mean(pairs: List[tuple]) -> float:
    """REP-weighted mean of (rep, score) pairs, scores in [-1, 1]."""
    return sum(r * round(s * 100) for r, s in pairs) / sum(r for r, _ in pairs) / 100


# ---------------------------------------------------------------------------
# Scratch hardhat project (copies; trustless-contracts is never written)
# ---------------------------------------------------------------------------

def _find_hardhat_cli() -> Path:
    for base in [REPO, *REPO.parents]:
        cli = base / "node_modules" / "hardhat" / "internal" / "cli" / "cli.js"
        if cli.exists() and (base / "node_modules" / "@openzeppelin" / "contracts").exists():
            return cli
    raise RuntimeError("no node_modules with hardhat + @openzeppelin/contracts "
                       f"at or above {REPO}; run npm install")


def build_scratch_project() -> Path:
    if not (DAO_DIR / "contracts" / "ToolBond.sol").exists():
        raise RuntimeError(f"trustless-contracts not found at {DAO_DIR} "
                           "(set TRUSTLESS_CONTRACTS_DIR)")
    src = SCRATCH / "contracts"
    if src.exists():
        shutil.rmtree(src)
    (src / "core").mkdir(parents=True)
    for name in DAO_SOURCES:
        shutil.copy2(DAO_DIR / "contracts" / name, src / name)
    for name in REPO_SOURCES:
        shutil.copy2(REPO / "contracts" / "core" / name, src / "core" / name)
    (SCRATCH / "hardhat.config.js").write_text(HARDHAT_CONFIG, encoding="utf-8")
    pkg = SCRATCH / "package.json"
    if not pkg.exists():
        pkg.write_text('{"name": "e2e-tool-economy", "private": true}\n',
                       encoding="utf-8")
    return SCRATCH


def _node(cli: Path, *args: str, timeout: int = 900):
    return subprocess.run(["node", str(cli), *args], cwd=str(SCRATCH),
                          capture_output=True, text=True, timeout=timeout,
                          env=_ORIG_ENV)


def compile_scratch(cli: Path) -> None:
    log("compiling scratch project (Substrate, ServiceMarket, ToolBond, RepToken, "
        "Registry, AutonetRepClaim)...")
    proc = _node(cli, "compile", timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")


def artifact(name: str, sol: Optional[str] = None, core: bool = False) -> Dict[str, Any]:
    base = SCRATCH / "artifacts" / "contracts" / ("core" if core else "")
    p = base / f"{sol or name}.sol" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8"))


_NODES: List[subprocess.Popen] = []  # killed on any exit (see __main__)


def _rpc_answers(port: int) -> bool:
    """Does something on 127.0.0.1:``port`` answer an eth_chainId request?"""
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}", method="POST",
        data=b'{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return b"result" in resp.read()
    except Exception:  # noqa: BLE001
        return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_node(cli: Path) -> subprocess.Popen:
    if not _port_free(PORT) or _rpc_answers(PORT):
        raise RuntimeError(f"port {PORT} is busy (set E2E_TOOL_ECONOMY_PORT)")
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
    """Stop the hardhat node this run started, by its PID."""
    if proc is None or proc.poll() is not None:
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

def new_key() -> str:
    from eth_account import Account
    return "0x" + Account.create().key.hex().removeprefix("0x")


def address_of(key: str) -> str:
    from eth_account import Account
    return Account.from_key(key).address


def send(w3, call, pk: str, gas: int = 3_000_000) -> Any:
    """Sign and send ``call`` from ``pk``. Nonces come from the daemon's own
    per-address manager, since the operators also send from daemon code."""
    from eth_account import Account
    from nodes.common.tx_nonce import next_nonce
    acct = Account.from_key(pk)
    tx = call.build_transaction({
        "from": acct.address, "nonce": next_nonce(w3, acct.address),
        "gas": gas, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID})
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    rcpt = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(raw), timeout=60)
    if rcpt.status != 1:
        raise RuntimeError(f"tx reverted: {call.fn_name}")
    return rcpt


def reverts(call, sender: str) -> str:
    """The revert reason of ``call`` from ``sender`` ('' when it would not
    revert), as the daemon's error strings carry it."""
    try:
        call.estimate_gas({"from": sender})
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc) or type(exc).__name__


def deploy(w3, name: str, args: list, pk: str = HH_DEPLOYER_PK,
           sol: Optional[str] = None, core: bool = False):
    art = artifact(name, sol, core)
    factory = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
    rcpt = send(w3, factory.constructor(*args), pk, gas=16_000_000)
    return w3.eth.contract(address=rcpt.contractAddress, abi=art["abi"])


def fund_eth(w3, to: str, eth: int = 5) -> None:
    from eth_account import Account
    from nodes.common.tx_nonce import next_nonce
    acct = Account.from_key(HH_DEPLOYER_PK)
    tx = {"from": acct.address, "to": to, "value": eth * E18,
          "nonce": next_nonce(w3, acct.address), "gas": 21000,
          "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID}
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(raw))


def advance(w3, seconds: int) -> None:
    w3.provider.make_request("evm_increaseTime", [int(seconds)])
    w3.provider.make_request("evm_mine", [])


def chain_time(w3) -> int:
    return int(w3.eth.get_block("latest")["timestamp"])


def anchor_epoch(w3, substrate, epoch_id: str, key: str) -> int:
    """A plain anchor by the governor (operator A on this dev network; on a
    live network the same call arrives through the AnchorElection). Returns
    the anchor's index."""
    from eth_utils import keccak
    prev_root = substrate.functions.latestEpochRoot().call()
    prev_hash = substrate.functions.latestAnchorHash().call()
    send(w3, substrate.functions.anchor(
        epoch_id, keccak(text=f"root:{epoch_id}"), prev_root, prev_hash,
        f"cid-{epoch_id}", keccak(text=f"payload:{epoch_id}"), b"\x00" * 32,
        chain_time(w3)), key)
    return int(substrate.functions.anchorCount().call()) - 1


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def make_runtime(tmp_root: Path, addrs: Dict[str, str], owner_wallet: str,
                 operator_key: str):
    """A daemon of one household. Its operator signs with the configured
    daemon key (``autonet.private_key``): the three daemons share one
    process and so one vault, which holds one operator key only."""
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime

    data_dir, agents_dir = tmp_root / "data", tmp_root / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    a = config.autonet
    a.owner_wallet = owner_wallet
    a.private_key = operator_key
    a.rpc_url = RPC_URL
    a.chain_id = CHAIN_ID
    a.substrate_address = addrs["substrate"]
    a.rep_token_address = addrs["rep"]
    a.tool_bond_address = addrs["tool_bond"]
    a.rep_claim_address = addrs["rep_claim"]
    a.service_registry_address = addrs["service_registry"]
    a.payment_channel_address = addrs["payment_channel"]
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)
    rt.tool_store._embedder = lambda text: (0.11, 0.22, 0.33) if text else ()
    return rt


def owner_session():
    """The owner's local session (a localhost app connection)."""
    from atn.ws_auth import ClientSession
    return ClientSession(local=True, is_loopback=True, authed=True, owner=True)


async def register_operator(rt, owner_key: str, peer: bytes) -> str:
    """Register the daemon's operator on Substrate bound to the owner
    wallet: the owner signs the OwnerBinding, the operator sends it (what
    the app's Household > operator registration does; WS register_operator
    also needs a running libp2p host for the peer id). Returns the tx."""
    from atn.models import AgentIdentity
    from atn.on_chain import OnChainService
    svc = OnChainService(rt._config.autonet)
    op_key = rt.registry.daemon_operator_key()
    op = address_of(op_key)
    nonce = await svc.binding_nonce(op)
    sig = svc.sign_binding(owner_key, op, "", nonce)
    res = await svc.register_agent_bound(
        identity=AgentIdentity(address=op), private_key=op_key, peer_id=peer,
        owner=address_of(owner_key), owner_sig=sig, parent="")
    if not res.get("success"):
        raise RuntimeError(f"operator registration failed: {res}")
    rt.registry.set_operator_registered(True, res.get("tx_hash"))
    return str(res.get("tx_hash"))


async def tick_all(drivers, ticks: int = 1) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for _ in range(ticks):
        for d in drivers:
            for k, v in (await d.tick()).items():
                total[k] = total.get(k, 0) + v
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> int:  # noqa: C901 (a linear E2E script, staged for reading)
    from web3 import Web3

    logging.basicConfig(level=logging.WARNING)
    for noisy in ("nodes.common.blob_store", "atn.tool_store", "atn.service_store",
                  "atn.runtime", "atn.runtime.agent_registry", "atn.harness_distro",
                  "websockets", "web3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    board = Board()
    for num, name in STAGES:
        board.stage(num, name)
    S = board.stages
    node: Optional[subprocess.Popen] = None
    rts: Dict[str, Any] = {}
    ws_c = None
    failed = False

    def fail(stage: Stage, exc: BaseException) -> None:
        nonlocal failed
        stage.status = "FAIL"
        stage.note("error", repr(exc)[:300])
        log(traceback.format_exc())
        failed = True

    async def finish() -> int:
        teardown = S[9]
        try:
            if ws_c is not None:
                await ws_c.stop()
            for rt in rts.values():
                try:
                    await rt.stop()
                except Exception:  # noqa: BLE001 (never started)
                    pass
            kill_node(node)
            shutil.rmtree(_TMP_HOME, ignore_errors=True)
            teardown.status = "PASS"
        except Exception as e:  # noqa: BLE001
            fail(teardown, e)
        print(board.render())
        return 1 if failed else 0

    # ------------------------------------------------------------------
    s1 = S[1]
    log("Stage 1: chain up")
    owner_key = {h: new_key() for h in HOUSES}
    op_key = {h: new_key() for h in HOUSES}
    owner = {h: address_of(k) for h, k in owner_key.items()}
    op = {h: address_of(k) for h, k in op_key.items()}
    try:
        from nodes.common.library.charter import DEFAULT_CHARTER
        cli = _find_hardhat_cli()
        build_scratch_project()
        compile_scratch(cli)
        # The daemon's chain code reads the ABIs compiled with this run's
        # bytecode, not the packaged copies.
        os.environ["ATN_ARTIFACTS_DIR"] = str(SCRATCH / "artifacts")
        node = start_node(cli)
        if not wait_rpc():
            raise RuntimeError("hardhat RPC never came up")
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        deployer = address_of(HH_DEPLOYER_PK)
        timelock = address_of(HH_TIMELOCK_PK)
        treasury = address_of(HH_TREASURY_PK)
        fund_eth(w3, timelock)
        # The timelock owns the Registry (the treasury), as on a live network.
        registry = deploy(w3, "Registry", [timelock, "0x" + "00" * 20])
        send(w3, registry.functions.editRegistry(CHARTER_KEY, DEFAULT_CHARTER.text),
             HH_TIMELOCK_PK)
        rep = deploy(w3, "RepToken", ["Autonet REP", "REP", registry.address, timelock,
                                      [], []])
        # (treasury, vaultMinter, governor, anchorer, genesisRoot, ...): the
        # deployer is the vault minter (sells ATN), operator A the governor
        # (anchors directly, an EOA governor as on a dev network).
        substrate = deploy(w3, "Substrate", [treasury, deployer, op["A"], "0x" + "00" * 20,
                                             bytes(32), "Test ATN", "REPATN",
                                             SERVICE_FEE_BPS, FEE_TREASURY_BPS], core=True)
        bond = deploy(w3, "ToolBond", [substrate.address, rep.address, registry.address,
                                       timelock, tuple(PARAMS.values())])
        send(w3, rep.functions.setSlasher(bond.address, True), HH_DEPLOYER_PK)
        rep_claim = deploy(w3, "AutonetRepClaim", [substrate.address, rep.address, [], []])
        send(w3, rep.functions.setMinter(rep_claim.address, True), HH_DEPLOYER_PK)
        svc_registry = deploy(w3, "ServiceRegistry", [substrate.address],
                              sol="ServiceMarket", core=True)
        channel = deploy(w3, "PaymentChannel", [substrate.address, CHANNEL_WINDOW],
                         sol="ServiceMarket", core=True)
        addrs = {"substrate": substrate.address, "rep": rep.address,
                 "tool_bond": bond.address, "registry": registry.address,
                 "rep_claim": rep_claim.address,
                 "service_registry": svc_registry.address,
                 "payment_channel": channel.address}
        check(substrate.functions.governor().call() == op["A"],
              "Substrate's governor is operator A (dev network: it anchors directly)")
        check(rep.functions.slashers(bond.address).call(), "ToolBond is a RepToken slasher")
        check(rep.functions.minters(rep_claim.address).call(),
              "AutonetRepClaim is a RepToken minter")
        check(rep_claim.functions.substrate().call() == substrate.address,
              "AutonetRepClaim reads its earnings from Substrate")
        check(bond.functions.reviewerShareBps().call() == 4000, "reviewerShareBps = 4000")
        check(registry.functions.getRegistryValue(CHARTER_KEY).call() == DEFAULT_CHARTER.text,
              "the Registry holds the charter text")
        for k, v in addrs.items():
            s1.note(k, v)
        s1.status = "PASS"
    except Exception as e:
        fail(s1, e)
        return await finish()

    def atn_of(addr: str) -> int:
        return substrate.functions.balanceOf(addr).call()

    def rep_of(addr: str) -> int:
        return rep.functions.balanceOf(addr).call()

    # ------------------------------------------------------------------
    s2 = S[2]
    log("Stage 2: households")
    try:
        from atn.agent_tools import execute_tool
        from atn.runtime.review_batch import ReviewBondClient
        for h in HOUSES:
            tmp = Path(tempfile.mkdtemp(prefix=f"e2e_econ_{h.lower()}_", dir=_TMP_HOME))
            rts[h] = make_runtime(tmp, addrs, owner[h], op_key[h])
        for h, rt in rts.items():
            # The external agent (POST /agents): no keypair, it is the operator.
            made = await execute_tool("create_agent", {
                "id": AGENT[h], "name": AGENT[h], "provider": "external",
                "harness": "e2e harness"}, rt, caller_id=None)
            check(made.get("external") and made.get("address", "").lower() == op[h].lower(),
                  f"{h}: external agent {AGENT[h]} acts as operator {op[h][:10]}")
            check(rt.registry.get_agent_key(AGENT[h]) == op_key[h],
                  f"{h}: the daemon signs for it with the operator key")
            fund_eth(w3, op[h])
            tx = await register_operator(rt, owner_key[h], f"peer-{h}".encode())
            check(substrate.functions.getAgentOwner(op[h]).call() == owner[h],
                  f"{h}: operator registered bound to owner {owner[h][:10]} ({tx[:12]})")
            check(rt.get_agent(AGENT[h]).identity.registered_on_chain,
                  f"{h}: so the external agent counts as registered")
            send(w3, rep.functions.mint(owner[h], REP_GENESIS[h] * E18), HH_DEPLOYER_PK)
            send(w3, substrate.functions.mintFromVault(op[h], ATN_BOUGHT[h] * ATN),
                 HH_DEPLOYER_PK)
            check(w3.eth.get_balance(owner[h]) == 0,
                  f"{h}: the owner wallet holds no gas (the operator pays it)")
        stores = {h: rt.tool_store._blob_store() for h, rt in rts.items()}

        def fetcher_for(me: str):
            async def fetch(digest: str):
                for h, bs in stores.items():
                    if h != me:
                        got = bs.get_bytes(digest)
                        if got is not None:
                            return got
                return None
            return fetch
        for h, rt in rts.items():
            # Manifests and code travel between the daemons' blob stores.
            rt.tool_store.blob_fetcher = fetcher_for(h)
        check(rep.functions.totalSupply().call() == 600 * E18, "REP supply = 600 (genesis)")
        # The genesis anchor: REP held now weighs epoch 1's reviews.
        advance(w3, 5)
        anchor_epoch(w3, substrate, "e0", op_key["A"])
        advance(w3, 5)
        check(bond.functions.currentEpoch().call() == 1, "genesis anchor e0: ToolBond epoch 1")
        for h in ("B", "C"):
            got = await ReviewBondClient(rts[h]).household_rep(op[h])
            check(got == REP_GENESIS[h] * E18,
                  f"{h}: the daemon weighs its operator by the owner's {REP_GENESIS[h]} REP")
        check(Path(os.path.expanduser("~")) == _TMP_HOME, "HOME is the temp dir")
        for h in HOUSES:
            s2.note(h, f"owner {owner[h][:10]} operator {op[h][:10]} REP {REP_GENESIS[h]}")
        s2.status = "PASS"
    except Exception as e:
        fail(s2, e)
        return await finish()

    rt_a, rt_b, rt_c = rts["A"], rts["B"], rts["C"]

    # ------------------------------------------------------------------
    s3 = S[3]
    log("Stage 3: publish")
    digest = ""
    announced: List[tuple] = []
    try:
        from atn.runtime.tool_bond import ToolBondGate
        # What the autonet bridge wires (autonet_service): a published
        # manifest goes to the library, which emits its tool_registered event.
        rt_a.tool_store.manifest_sink = lambda m, author: announced.append((m, author))
        atn0 = atn_of(op["A"])
        reg = await execute_tool("register_tool", {
            "name": "upper_case", "description": "Upper-cases a string.",
            "input_schema": UPPER_SCHEMA, "code": UPPER_CODE,
            "novelty": {"statement": "Upper-cases a string; no comparable tool exists yet.",
                        "prior_art": []}}, rt_a, caller_id=AGENT["A"])
        if "error" in reg:
            raise RuntimeError(f"register_tool: {reg}")
        pub = await execute_tool("publish_tool", {"digest": reg["digest"]}, rt_a,
                                 caller_id=AGENT["A"])
        if "error" in pub:
            raise RuntimeError(f"publish_tool: {pub}")
        digest = pub["digest"]
        b32 = bytes.fromhex(digest)
        check(pub.get("published") is True and pub["stake"]["amount"] == MIN_STAKE,
              f"published with a {MIN_STAKE // ATN} ATN bond ({len(pub['stake']['txs'])} txs)")
        manifest = rt_a.tool_store.get(digest).manifest
        check(str(manifest.get("author", "")).lower() == op["A"].lower(),
              "the manifest's author is operator A")
        check(substrate.functions.toolAuthor(b32).call() == op["A"],
              "Substrate.toolAuthor = operator A")
        check(bond.functions.isBonded(b32).call(), "ToolBond.isBonded")
        check(atn0 - atn_of(op["A"]) == MIN_STAKE, "operator A's ATN down by the bond")
        check((await ToolBondGate(rt_b).is_bonded(digest))[0],
              "daemon B's gate sees the bond")
        check(len(announced) == 1 and announced[0][0] == manifest
              and str(announced[0][1]).lower() == op["A"].lower(),
              "the daemon announces the published manifest once, author operator A")
        s3.note("digest", digest[:16] + "...")
        s3.status = "PASS"
    except Exception as e:
        fail(s3, e)
        return await finish()

    # ------------------------------------------------------------------
    s4 = S[4]
    log("Stage 4: reviews")
    drivers: List[Any] = []
    try:
        from atn.agent_tools import review_due_names
        from atn.review_driver import ReviewDriver
        from atn.runtime.review_batch import store_for
        b32 = bytes.fromhex(digest)
        # B adopts it, uses it and reviews it at the end of the work.
        ad = await execute_tool("adopt_tool", {"digest": digest, "reason": "headings"},
                                rt_b, caller_id=AGENT["B"])
        check("error" not in ad and rt_b.tool_store.get(digest) is not None,
              f"B adopts it over the blob rail ({ad.get('origin', ad.get('error'))})")
        check("upper_case" in review_due_names(rt_b, AGENT["B"]),
              "B's agent is reminded to review what it adopted")
        out = await execute_tool("use_tool", {"name": "upper_case",
                                              "arguments": {"x": "household loop"}},
                                 rt_b, caller_id=AGENT["B"])
        check("HOUSEHOLD LOOP" in json.dumps(out), f"B uses it: {json.dumps(out)[:80]}")
        att = await execute_tool("attest_tools", {
            "judgments": [{"tool": digest, "ok": True, **REVIEWS["B"]}],
            "context": "upper-casing headings"}, rt_b, caller_id=AGENT["B"])
        rv = att.get("reviews") or {}
        check(rv.get("queued") == 1 and rv.get("epoch") == 1
              and str(rv.get("reviewer", "")).lower() == op["B"].lower(),
              f"B's review queued for epoch 1 under operator B ({rv})")
        check("upper_case" not in review_due_names(rt_b, AGENT["B"]),
              "and the reminder clears")
        # C checks the code out, reads it, uses it and reviews it.
        checkout = _TMP_HOME / "checkout_c"
        ad = await execute_tool("adopt_tool", {"digest": digest, "path": str(checkout)},
                                rt_c, caller_id=AGENT["C"])
        code_seen = any(".upper()" in p.read_text(encoding="utf-8", errors="ignore")
                        for p in checkout.rglob("*") if p.is_file())
        check("error" not in ad and code_seen, "C checks the code out and reads it")
        out = await execute_tool("use_tool", {"name": "upper_case",
                                              "arguments": {"x": "second household"}},
                                 rt_c, caller_id=AGENT["C"])
        check("SECOND HOUSEHOLD" in json.dumps(out), "C uses it")
        att = await execute_tool("attest_tools", {
            "judgments": [{"tool": digest, "ok": True, **REVIEWS["C"]}],
            "context": "upper-casing a title"}, rt_c, caller_id=AGENT["C"])
        rv = att.get("reviews") or {}
        check(rv.get("queued") == 1 and rv.get("epoch") == 1,
              f"C's review queued for epoch 1 ({rv})")
        missing = await execute_tool("attest_tools", {
            "judgments": [{"tool": digest, "ok": True, "correctness": 0.5,
                           "simplicity": 0.5, "malware": False}],
            "context": "a review without novelty"}, rt_c, caller_id=AGENT["C"])
        skipped = list(missing.get("skipped") or []) + list(
            (missing.get("reviews") or {}).get("skipped") or [])
        check(not (missing.get("reviews") or {}).get("queued")
              and any("novelty" in str(s.get("error")) for s in skipped),
              "a review missing one of the four fields is refused")
        for h in ("B", "C"):
            batch = store_for(rts[h]).load(op[h], 1, [])
            check(len(batch.reviews) == 1, f"{h}: one review in the operator's epoch-1 batch")

        # The review driver: commit at the close, reveal after the anchor.
        drivers = [ReviewDriver(rt_b), ReviewDriver(rt_c)]
        idle = await tick_all(drivers)
        check(idle.get("committed", 0) == 0, "no commit before the epoch's close fires")
        atn0 = {h: atn_of(op[h]) for h in ("B", "C")}
        for d in drivers:
            d.on_local_close()
        got = await tick_all(drivers)
        check(got.get("committed") == 2, f"both batches committed ({got})")
        for h in ("B", "C"):
            c = bond.functions.commits(1, op[h]).call()
            check(c[0] != b"\x00" * 32 and c[1] == REP_GENESIS[h] * E18 and not c[2],
                  f"{h}: commitment from operator {h}, weight {REP_GENESIS[h]} REP "
                  "(its owner's, at the snapshot), hidden")
            check(atn_of(op[h]) == atn0[h], f"{h}: no ATN moved (no stake)")
        anchor_epoch(w3, substrate, "e1", op_key["A"])
        check(bond.functions.revealEnds(1).call() > 0, "anchor e1 opens the reveal window")
        got = await tick_all(drivers)
        check(got.get("revealed") == 2, f"both batches revealed ({got})")
        r = bond.functions.getRound(b32, 1).call()
        check(r[0] == 1 and r[2] == 2, "round Open with 2 reviews")
        logs = bond.events.Reviewed().get_logs(from_block=0, to_block="latest")
        rev_b = next(lg["args"] for lg in logs if lg["args"]["reviewer"] == op["B"])
        check(rev_b["correctness"] == 90 and rev_b["simplicity"] == 80
              and rev_b["novelty"] == 70 and rev_b["malware"] == 0,
              "Reviewed carries the four fields (B: 90, 80, 70, no malware)")
        supply0 = rep.functions.totalSupply().call()
        credit0 = rep.functions.slashedCredit(bond.address).call()
        advance(w3, PARAMS["revealWindow"] + 1)
        got = await tick_all(drivers, ticks=2)
        check(bond.functions.getRound(b32, 1).call()[0] == 3,
              f"after the reveal window: round finalized and claimed ({got})")
        credit = rep.functions.slashedCredit(bond.address).call()
        check(rep.functions.totalSupply().call() + (credit - credit0) == supply0,
              "REP conserved by the review settlement")
        corr_sum, corr_mass = bond.functions.toolRating(b32, (0).to_bytes(32, "big")).call()
        want_c = hundredths_mean([(300, 0.9), (200, 0.5)])
        check(corr_mass == 500 * E18 and close_to(corr_sum / corr_mass / 100, want_c),
              f"rating: correctness {corr_sum / corr_mass / 100:.3f} on 500 REP")
        statuses = {h: store_for(rts[h]).load(op[h], 1, []).status for h in ("B", "C")}
        check(statuses == {"B": "settled", "C": "settled"}, "both batches settled")
        # The conclusion gate (Registry defaults): the distinct reviewing
        # households hold 1% of REP and are 0.5% of REP-holding households.
        supply = rep.functions.totalSupply().call()
        holders = [h for h in HOUSES if rep_of(owner[h]) > 0]
        reviewing = sum(rep_of(owner[h]) for h in ("B", "C"))
        check(reviewing * 10_000 >= 100 * supply and 2 * 10_000 >= 50 * len(holders),
              f"B and C clear the gate: {reviewing * 100 / supply:.1f}% of REP, "
              f"2 of {len(holders)} REP-holding households")
        s4.note("rep_after_reviews", {h: round(rep_of(owner[h]) / E18, 6) for h in HOUSES})
        s4.status = "PASS"
    except Exception as e:
        fail(s4, e)
        return await finish()

    # ------------------------------------------------------------------
    s5 = S[5]
    log("Stage 5: service fees")
    service_id = 0
    try:
        from atn.on_chain import OnChainService
        from atn.ws_server import WebSocketBridge
        # C's owner sells a local tool as a service, registered and paid
        # under operator C, reachable at the endpoint it publishes on chain.
        reg = await execute_tool("register_tool", {
            "name": "word_count", "description": "Counts the words in a text.",
            "input_schema": WORDS_SCHEMA, "code": WORDS_CODE}, rt_c, caller_id=None)
        if "error" in reg:
            raise RuntimeError(f"register_tool (C): {reg}")
        port_c = _free_port()
        ws_c = WebSocketBridge(rt_c, host="127.0.0.1", port=port_c)
        await ws_c.start()
        listed = await ws_c._handle_message({
            "type": "register_service", "msg_id": "svc-1", "name": "word-count",
            "description": "Word counts on C's machine.", "input_schema": WORDS_SCHEMA,
            "backing_tool": reg["digest"], "ask": {"amount": str(SERVICE_ASK)},
            "unit": "one word count of a text"},
            owner_session())
        if not listed.get("ok") or not listed["result"].get("on_chain"):
            raise RuntimeError(f"register_service: {listed}")
        service_id = int(listed["result"]["service_id"])
        entry = svc_registry.functions.getService(service_id).call()
        check(entry[0] == op["C"] and entry[2] == SERVICE_ASK,
              f"service #{service_id} on ServiceRegistry: provider operator C, "
              f"ask {SERVICE_ASK // ATN} ATN")
        ep = await OnChainService(rt_c._config.autonet).update_endpoint(
            op_key["C"], f"ws://127.0.0.1:{port_c}")
        check(ep.get("success") and substrate.functions.getAgentEndpoint(op["C"]).call()
              == f"ws://127.0.0.1:{port_c}", "C's endpoint published on chain")

        # B's agent pays directly and sends the request with the proof.
        before = {"b": atn_of(op["B"]), "c": atn_of(op["C"]), "t": atn_of(treasury)}
        burned0 = substrate.functions.burnedSinceAnchor().call()
        paid = await execute_tool("pay_for_service", {
            "recipient": op["C"], "amount": SERVICE_ASK}, rt_b, caller_id=AGENT["B"])
        if not paid.get("tx_hash") or not paid.get("request_id"):
            raise RuntimeError(f"pay_for_service: {paid}")
        reply = await execute_tool("request_service", {
            "service_id": service_id, "payload": {"text": "one turn of the loop"},
            "tx_hash": paid["tx_hash"], "request_id": paid["request_id"]},
            rt_b, caller_id=AGENT["B"])
        check(reply.get("ok") and '"words": 5' in json.dumps(reply),
              f"C serves B's paid request over {reply.get('transport', '?')} "
              f"({json.dumps(reply.get('result'))[:80]})")
        pay = substrate.events.ServicePayment().get_logs(from_block=0, to_block="latest")[-1]
        check(pay["args"]["payer"] == op["B"] and pay["args"]["recipient"] == op["C"]
              and pay["args"]["amount"] == SERVICE_ASK,
              "ServicePayment: operator B pays operator C the ask")
        fee = SERVICE_ASK * SERVICE_FEE_BPS // 10_000
        burned = fee - fee * FEE_TREASURY_BPS // 10_000
        check(before["b"] - atn_of(op["B"]) == SERVICE_ASK
              and atn_of(op["C"]) - before["c"] == SERVICE_ASK - fee,
              f"B paid {SERVICE_ASK // ATN} ATN, C got it net of the {fee / ATN:g} ATN fee")
        check(atn_of(treasury) - before["t"] == fee - burned
              and substrate.functions.burnedSinceAnchor().call() - burned0 == burned,
              f"half the fee to the treasury, {burned / ATN:g} ATN burned: the next pool")
        s5.note("service_id", service_id)
        s5.note("burned", f"{burned / ATN:g} ATN")
        s5.status = "PASS"
    except Exception as e:
        fail(s5, e)
        return await finish()

    # ------------------------------------------------------------------
    s6 = S[6]
    log("Stage 6: close, anchor, ATN claims")
    close: Dict[str, Any] = {}
    rep_before: Dict[str, int] = {}
    outs: Dict[str, Dict[str, Any]] = {}
    delegate_a0 = ""
    try:
        from nodes.common.blob_resolver import InMemoryBlobResolver
        from nodes.common.canonical_ordering import canonical_order
        from nodes.common.chain_submission_driver import (
            AgentChainIdentity, ChainSubmissionConfig, ChainSubmissionDriver)
        from nodes.common.event_gossip import EventBatch, Keypair
        from nodes.common.federated_close_driver import FederatedCloseResult
        from nodes.common.federated_reconcile import UNMINTED, federated_epoch_close
        from nodes.common.library.tool_manifest import registration_event
        from nodes.common.mint_merkle import mint_merkle_root, scale_mint
        from nodes.common.voice_state import read_voice_state

        anchor_epoch(w3, substrate, "e2", op_key["A"])
        advance(w3, 5)   # RepToken checkpoints only answer for past timepoints
        state = read_voice_state(substrate.address, RPC_URL, rep_token_address=rep.address,
                                 tool_bond_address=bond.address,
                                 dao_registry_address=registry.address, web3=w3)
        snap = state["review_snapshot"]
        check(digest in snap["tools"] and digest in snap["rounds"],
              "the snapshot pinned to e2 holds the tool's rating and round")
        check(all(state["owner_map"].get(op[h].lower()) == owner[h].lower() for h in HOUSES),
              "owner map from chain: each operator to its owner wallet")
        pool = float(state["emission_pool"])
        check(close_to(pool, burned / ATN, 1e-9),
              f"pool = the fees burned between e1 and e2 = {pool} ATN")

        manifest, author = announced[0]
        kp = Keypair.generate()
        batch = EventBatch(rpb_address="rpb_e2e_economy", sender_pubkey=kp.public_key,
                           batch_seq=1, events=[registration_event(
                               manifest, digest, agent_id=author)],
                           prev_batch_hash=b"", timestamp=1_700_000_000.0)
        close = federated_epoch_close(
            canonical_order([batch]), emission_pool=pool,
            agent_owner_map=state["owner_map"], rep_shares=state["rep_shares"],
            review_snapshot=snap, mint_gate=tuple(state["mint_gate"]))
        check(tuple(state["mint_gate"]) == (100, 50), "gate: 1% of REP, 0.5% of households")
        check(close["concluded"] == {digest: 0},
              f"the tool concludes, no malware share ({close['concluded']})")
        tm = close["tool_mint"][digest]
        want_v = (56 + 50 + 50 + 50) / 4 / 100      # one axis scored 0.6 by B
        want_s = 0.5 + 0.5 * hundredths_mean([(300, 0.8), (200, 0.6)])
        want_n = hundredths_mean([(300, 0.7), (200, 0.5)])
        want_score = want_v * want_c * want_s
        check(not tm["malware"] and tm["reviewers"] == 2
              and close_to(tm["score"], want_score, 1e-9),
              f"score V x C x S = {tm['score']:.6f} (V {want_v:.3f}, C {want_c:.3f}, "
              f"S {want_s:.3f}), both reviewers paid")
        # One tool concludes, so its share is the whole pool.
        check(close_to(tm["author_mint"], pool * 0.6 * want_score * want_n, 1e-6),
              f"author side 60% x V x C x S x N {want_n:.2f} = {tm['author_mint']:.6f} ATN")
        am = close["agent_mint"]
        hk = {h: owner[h].lower() for h in HOUSES}
        check(set(am) == set(hk.values()) and UNMINTED not in am,
              "the mint is keyed by the three owner wallets (households), not operators")
        check(close_to(am[hk["A"]], tm["author_mint"], 2e-6),
              f"household A: {am[hk['A']]:.6f} ATN (the author side)")
        check(close_to(am[hk["B"]] + am[hk["C"]], 0.4 * pool, 2e-6),
              f"households B + C: {am[hk['B']] + am[hk['C']]:.6f} ATN "
              "(40% of the tool's share, by accuracy)")
        carry = close["tool_rounds"]["pool_carry"]
        minted_raw = sum(scale_mint(v) for v in am.values())
        check(carry == round(pool * ATN) - minted_raw,
              f"{carry / ATN:.6f} ATN unminted carries over to the next close")

        # The daemons take the close: A's driver anchors it (its operator is
        # Substrate's governor), every driver claims for its household.
        close["epoch_id"] = "e3"
        fed = FederatedCloseResult(
            epoch_id="e3", close_result=close, senders=[kp.public_key],
            winner=kp.public_key, is_winner=True, n_batches=1,
            snapshot_anchor_count=int(substrate.functions.anchorCount().call()))
        blobs = InMemoryBlobResolver()   # stands in for the blob rail between them
        csd = {h: ChainSubmissionDriver(
            config=ChainSubmissionConfig(
                substrate_address=substrate.address, rpc_url=RPC_URL, chain_id=CHAIN_ID,
                daemon_private_key=op_key[h], rep_claim_address=rep_claim.address,
                owner_wallet=owner[h], tool_bond_address=bond.address),
            agent_chain_resolver=(lambda h=h: [AgentChainIdentity(op[h], op_key[h])]),
            blob_resolver=blobs) for h in HOUSES}
        atn_before = {h: (atn_of(owner[h]), atn_of(op[h])) for h in HOUSES}
        rep_before.update({h: rep_of(owner[h]) for h in HOUSES})
        delegate_a0 = str(rep.functions.delegates(owner["A"]).call())
        outs["A"] = await asyncio.to_thread(csd["A"].handle_federated_close, fed)
        check(outs["A"].get("anchored"),
              f"daemon A's driver anchors e3 ({outs['A'].get('anchor_error', 'direct')})")
        a3 = substrate.functions.getAnchorByEpochId("e3").call()
        check(bytes(a3[9]) == mint_merkle_root(am, concluded=close["concluded"]),
              "e3's agentMintRoot is the close's mint and concluded leaves")
        for h in ("B", "C"):
            outs[h] = await asyncio.to_thread(
                csd[h].handle_federated_close, dataclasses.replace(fed, is_winner=False))
            check(outs[h].get("anchor_skipped") == "already anchored",
                  f"daemon {h}'s driver finds e3 anchored and only claims")
        for h in HOUSES:
            want = scale_mint(am[hk[h]])
            row = next((r for r in outs[h]["agent_submissions"]
                        if r["claimant"].lower() == hk[h]), {})
            got_owner = atn_of(owner[h]) - atn_before[h][0]
            check(row.get("success") and got_owner == want > 0
                  and substrate.functions.agentMintTotal(owner[h]).call() == want,
                  f"{h}: operator claimFor its owner: +{want / ATN:.6f} ATN, "
                  "agentMintTotal the same")
            check(atn_of(op[h]) == atn_before[h][1],
                  f"{h}: the operator received nothing (it paid the gas)")
        recorded = {e["args"]["agent"]: e["args"]["amount"] for e in
                    substrate.events.TrainingRecorded().get_logs(from_block=0,
                                                                 to_block="latest")}
        check(set(recorded) == {owner[h] for h in HOUSES},
              "TrainingRecorded names the owner wallets only")
        check(substrate.functions.networkMintTotal().call()
              <= substrate.functions.mintPoolTotal().call(),
              "claims stay within the cumulative pool")
        v = bond.functions.verdicts(bytes.fromhex(digest)).call()
        check(v[0] > 0 and v[1] == 0 and not v[2],
              "ToolBond: the tool concluded from the anchored leaf, no malware")
        check(bond.functions.isBonded(bytes.fromhex(digest)).call(), "and it keeps its bond")
        s6.note("pool", f"{pool} ATN")
        s6.note("mint", {h: round(am[hk[h]], 6) for h in HOUSES})
        s6.note("carry", f"{carry / ATN:.6f} ATN")
        s6.status = "PASS"
    except Exception as e:
        fail(s6, e)
        return await finish()

    # ------------------------------------------------------------------
    s7 = S[7]
    log("Stage 7: REP claims")
    try:
        claimed = {e["args"]["earner"]: dict(e["args"]) for e in
                   rep_claim.events.RepClaimed().get_logs(from_block=0, to_block="latest")}
        for h in HOUSES:
            earned = substrate.functions.agentMintTotal(owner[h]).call()
            ev = claimed.get(owner[h], {})
            check(ev.get("to") == owner[h] and ev.get("earningsDelta") == earned
                  and ev.get("repMinted") == earned * 10 ** 12,
                  f"{h}: RepClaimed {earned / ATN:.6f} REP to the owner wallet "
                  "(1:1 on agentMintTotal)")
            check(rep_of(owner[h]) - rep_before[h] == earned * 10 ** 12,
                  f"{h}: owner REP up by exactly that")
            check(rep_claim.functions.claimedEarnings(owner[h]).call() == earned
                  and rep_claim.functions.claimable(owner[h]).call() == 0,
                  f"{h}: watermark at its earnings, nothing left to claim")
        from atn.review_driver import reverted_with
        why = reverts(rep_claim.functions.claim(owner["A"]), op["A"])
        check(reverted_with({"error": why}, "NothingToClaim"),
              "a second claim reverts (NothingToClaim)")
        check(rep.functions.delegates(owner["A"]).call() == owner["A"]
              and rep.functions.getVotes(owner["A"]).call() == rep_of(owner["A"]),
              f"A's REP votes for its household ({delegate_a0[:10]} before the claim)")
        s7.note("rep_claimed", {h: round((rep_of(owner[h]) - rep_before[h]) / E18, 6)
                                for h in HOUSES})
        s7.status = "PASS"
    except Exception as e:
        fail(s7, e)
        return await finish()

    # ------------------------------------------------------------------
    s8 = S[8]
    log("Stage 8: payment channel")
    try:
        from atn.on_chain import ServiceMarketClient
        from atn.service_channels import sweep_refunds
        before = {"b": atn_of(op["B"]), "c": atn_of(op["C"])}
        replies = []
        for text in ("a channel request", "and a second one here"):
            replies.append(await execute_tool("request_service", {
                "service_id": service_id, "deposit": 3 * SERVICE_ASK,
                "payload": {"text": text}}, rt_b, caller_id=AGENT["B"]))
        check(all(r.get("ok") and r.get("paid") == SERVICE_ASK for r in replies)
              and replies[0]["channel_id"] == replies[1]["channel_id"],
              "two requests served on one channel, one voucher each")
        cid = int(replies[0]["channel_id"])
        ch = channel.functions.getChannel(cid).call()
        check(ch[0] == op["B"] and ch[1] == op["C"] and ch[2] == 3 * SERVICE_ASK
              and ch[5] == 1, f"channel {cid} open: operator B to operator C, "
                              f"{3 * SERVICE_ASK // ATN} ATN escrowed")
        kept = rt_c.service_store.get_channel_voucher(cid) or {}
        check(int(kept.get("cumulative", 0)) == 2 * SERVICE_ASK,
              "C keeps the best voucher: cumulative 2 x the ask")
        check(before["b"] - atn_of(op["B"]) == 3 * SERVICE_ASK,
              "B's deposit left its operator, nothing else")
        # C's settlement sweep closes the channel once it is idle.
        rt_c._config.autonet.channel_idle_close_s = 1
        await asyncio.sleep(1.5)
        out = await rt_c.settle_channels_once()
        settled = out.get("settled") or []
        check(settled and settled[0].get("ok") and settled[0].get("reason") == "idle",
              f"C's sweep closes the channel with its best voucher ({settled})")
        ch = channel.functions.getChannel(cid).call()
        fee = 2 * SERVICE_ASK * SERVICE_FEE_BPS // 10_000
        check(ch[5] == 2 and ch[3] == 2 * SERVICE_ASK
              and atn_of(op["C"]) - before["c"] == 2 * SERVICE_ASK - fee,
              f"closing: 2 x the ask claimed, C paid net of the {fee / ATN:g} ATN fee")
        # B's sweep sees the provider's close, then withdraws after the window.
        out = await rt_b.settle_channels_once()
        actions = [r["action"] for r in out.get("refunds") or []]
        check("closed_by_provider" in actions, f"B's sweep sees the close ({actions})")
        if "withdrew" not in actions:
            advance(w3, CHANNEL_WINDOW + 1)
            # The run moved the chain clock ahead of the wall clock
            # (evm_increaseTime), so the sweep is given the chain's time.
            refunds = await sweep_refunds(rt_b.channel_book,
                                          ServiceMarketClient(rt_b._config.autonet),
                                          rt_b._channel_key_for, now=chain_time(w3))
            actions = [r["action"] for r in refunds]
        check("withdrew" in actions and channel.functions.getChannel(cid).call()[5] == 3,
              f"B's refund sweep withdraws the remainder ({actions})")
        check(atn_of(op["B"]) - (before["b"] - 3 * SERVICE_ASK) == SERVICE_ASK
              and rt_b.channel_book.closing() == {},
              f"B got {SERVICE_ASK // ATN} ATN back, fee-free; its book is clear")
        s8.note("channel", cid)
        s8.note("claimed", f"{2 * SERVICE_ASK // ATN} ATN")
        s8.note("refunded", f"{SERVICE_ASK // ATN} ATN")
        s8.status = "PASS"
    except Exception as e:
        fail(s8, e)

    return await finish()


if __name__ == "__main__":
    try:
        rc = asyncio.run(amain())
    finally:
        for _proc in _NODES:
            kill_node(_proc)
    sys.exit(rc)
