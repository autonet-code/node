#!/usr/bin/env python3
"""The economy proof-of-life: one turn of the household economy loop, end to
end on a local chain (Decisions 2026-11-07, 2026-11-08, 2026-11-09 and
2026-11-10; docs/economy_design.md).

Stands up a LOCAL hardhat node with this repo's Substrate and ServiceMarket
(ServiceRegistry + PaymentChannel) and the jurisdiction contracts the loop
needs from trustless-contracts (a Registry holding the charter, the value
index and the treasury, RepToken with ToolBond as slasher, ToolBond with
short windows, AutonetRepClaim as a RepToken minter; the DAO's contracts
are EIP-1167 clones initialized by the deployer, as AutonetSetup makes
them). The value index holds two tokens: USD6 (6 decimals, parity 1e30, so
one token is one value unit; its issuer can pause it) and T18 (18 decimals,
parity 1.25e18, one token worth 1.25 units). Three households run on it,
each an owner wallet plus a daemon whose operator key is registered on
Substrate bound to that wallet, and an external agent that acts on chain as
the operator. The DAEMON code paths drive every step:

  1. chain up;
  2. households: genesis REP to the owner wallets, index tokens to the
     operators, the genesis anchor;
  3. household A's agent publishes a tool with a ToolBond stake in the one
     index token its operator holds (``register_tool`` + ``publish_tool``:
     registerTool, approve, bond); B puts a bounty on it in the bond's token;
  4. households B and C adopt it, use it and review it (``attest_tools``
     with the four fields); each daemon's ``ReviewDriver`` commits, reveals
     after the anchor, finalizes and claims on ToolBond, which pays the
     bounty parts in the same transaction (``payOut`` only for a deferred
     part); the reviewing households clear the conclusion gate;
  5. B buys two services C sells, one priced in each token
     (``request_service`` with no credits: one direct payment each): each
     payment's fee is collected, half to the treasury and half to that
     token's pool;
  6. ``read_voice_state`` pinned to the anchor after the payments feeds
     ``federated_epoch_close``: the tool concludes and is paid once from its
     reviews (40% to the reviewers by accuracy, the author side by
     V x C x S kept by N), each earner taking the same fraction of both
     pools. Each daemon's ``ChainSubmissionDriver`` takes the close: A's
     anchors it (operator A is Substrate's governor on this dev network),
     then every operator claims its owner wallet's basket, which the claim
     pays in the same transaction, and shows ToolBond the conclusion. USD6's
     issuer pauses it before C's daemon takes the close: C's claim pays its
     T18 and defers its USD6 (PayDeferred), which stays owed and is
     withdrawn once unpaused;
  7. the same driver step claims each household's REP 1:1 on the value of
     its pool earnings (``AutonetRepClaim.claim``, ``earnedValue``);
  8. the REP floor: the DAO sets ``jurisdiction.autonet.rep_floor``; A
     publishes a second tool, B and C review it, and it concludes in an
     epoch that collected no fees (its pool is only what the first close
     carried). The close seeds the difference between the floor and what
     the claims are worth to the same earners by weight, the anchor stores
     the floor, each claim adds its seed to ``earnedValue``, and
     AutonetRepClaim mints the floor in REP across the three households;
  9. B opens credits with C in T18 and buys C's T18 service twice on them
     (vouchers pay ahead); C's sweep settles what it served without
     closing; B closes, C settles inside the window, and B's refund sweep
     withdraws the rest;
  10. teardown.

The jurisdiction contracts are NOT vendored: they are copied at run time
from a trustless-contracts checkout (env TRUSTLESS_CONTRACTS_DIR, default
C:/code/dao/trustless-contracts) into a scratch hardhat project under
build/ (gitignored), compiled there with Substrate.sol, ServiceMarket.sol
and the test tokens (contracts/test/IndexTokenMocks.sol) from this repo.
The checkout itself is never written. HOME, USERPROFILE and KEYSTORE_DIR
point at a temp dir for the whole run. The chain, setup and helpers follow
scripts/local_e2e_tool_security.py.

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
U6 = 10 ** 6             # one USD6 (6 decimals); also one value unit
PARITY_SCALE = 10 ** 30  # value = amount x parity / 1e30 (6-decimal value units)
PARITY = {"usd6": 10 ** 30, "t18": 125 * 10 ** 16}   # 1 USD6 = 1 unit, 1 T18 = 1.25
MIN_STAKE = 100 * U6     # a value: 100 units of the index
PARAMS = {  # ToolBond.Params (97ad738), short windows for a local run
    "minStake": MIN_STAKE, "revealWindow": 300, "degradeBps": 1000,
    "authorSlashBps": 10_000, "unbondDelay": 120, "reviewerShareBps": 4000,
}
SERVICE_FEE_BPS = 250  # Substrate default
FEE_TREASURY_BPS = 5000
CHANNEL_WINDOW = 60    # PaymentChannel challenge window, seconds
CHARTER_KEY = "jurisdiction.autonet.charter"
REP_FLOOR_KEY = "jurisdiction.autonet.rep_floor"
REP_FLOOR = 10 * E18     # the REP floor of stage 8: 10 REP (10 value units)
# One request of each of C's services: 400 USD6 (10 fee, 5 to the pool),
# 80 T18 = 100 units (2 T18 fee, 1 T18 to the pool).
ASK = {"usd6": 400 * U6, "t18": 80 * E18}
BOUNTY = 10 * U6         # B's bounty on A's tool, in the bond's token (USD6)

HOUSES = ("A", "B", "C")
AGENT = {h: f"agent-{h.lower()}" for h in HOUSES}
# REP at the genesis anchor, held by the owner wallets (the households).
REP_GENESIS = {"A": 100, "B": 300, "C": 200}
# Index tokens each operator holds (the bond, the bounty, services; gas is
# ETH). A holds only USD6, so its bond defaults to it.
FUNDS = {"A": {"usd6": 1000 * U6}, "B": {"usd6": 2000 * U6, "t18": 1000 * E18},
         "C": {"t18": 10 * E18}}

# The reviews of epoch 1 (B holds 300 REP, C 200). Only B scores a charter
# axis; the others count 0.5 for each review.
REVIEWS = {
    "B": {"correctness": 0.9, "simplicity": 0.8, "novelty": 0.7, "malware": False,
          "axes": {"promotion_of_intelligence": 0.6}},
    "C": {"correctness": 0.5, "simplicity": 0.6, "novelty": 0.5, "malware": False},
}

DAO_SOURCES = ["ToolBond.sol", "RepToken.sol", "Registry.sol", "IAdminToken.sol",
               "IJurisdictionData.sol", "AutonetRepClaim.sol", "CloneMaker.sol"]
# This repo's contracts go under contracts/core/, the Hardhat layout the
# daemon's ABI loader reads (ATN_ARTIFACTS_DIR, nodes/common/contract_artifacts).
REPO_SOURCES = ["Substrate.sol", "ServiceMarket.sol"]
REPO_TEST_SOURCES = ["IndexTokenMocks.sol"]   # the index's test tokens

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
LOWER_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'lower': str(args.get('x', '')).lower()}))\n"
)
UPPER_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}
WORDS_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}

STAGES = [
    (1, "chain up (scratch compile, hardhat, Substrate + DAO suite + ServiceMarket)"),
    (2, "households (three daemons; operators bound to owner wallets; genesis REP + anchor)"),
    (3, "household A's agent publishes a tool bonded in an index token; B adds a bounty"),
    (4, "B and C adopt, use and review it (four fields, commit-reveal, bounty paid out)"),
    (5, "service fees collected in both tokens (B buys C's two services)"),
    (6, "close: baskets of both pools; anchor; claims that pay the basket; a paused token deferred"),
    (7, "REP claimed 1:1 on the value of the pool earnings (AutonetRepClaim)"),
    (8, "the REP floor: a tool concludes in an epoch with no fees; the floor is minted in REP"),
    (9, "credits in T18: vouchers pay ahead, the provider settles, the buyer closes and withdraws"),
    (10, "teardown"),
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
    (src / "test").mkdir(parents=True)
    for name in DAO_SOURCES:
        shutil.copy2(DAO_DIR / "contracts" / name, src / name)
    for name in REPO_SOURCES:
        shutil.copy2(REPO / "contracts" / "core" / name, src / "core" / name)
    for name in REPO_TEST_SOURCES:
        shutil.copy2(REPO / "contracts" / "test" / name, src / "test" / name)
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


def artifact(name: str, sol: Optional[str] = None, core: bool = False,
             sub: str = "") -> Dict[str, Any]:
    base = SCRATCH / "artifacts" / "contracts" / ("core" if core else sub)
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
           sol: Optional[str] = None, core: bool = False, sub: str = ""):
    art = artifact(name, sol, core, sub)
    factory = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
    rcpt = send(w3, factory.constructor(*args), pk, gas=16_000_000)
    return w3.eth.contract(address=rcpt.contractAddress, abi=art["abi"])


def deploy_clone(w3, maker, name: str, init_args: list, pk: str = HH_DEPLOYER_PK):
    """A DAO contract as AutonetSetup makes it: an EIP-1167 clone of a
    deployed implementation (whose constructor disables its initializers),
    initialized by ``pk`` (the caller becomes RepToken's admin)."""
    impl = deploy(w3, name, [], pk)
    rcpt = send(w3, maker.functions.clone(impl.address), pk)
    addr = maker.events.Cloned().process_receipt(rcpt)[0]["args"]["instance"]
    c = w3.eth.contract(address=addr, abi=impl.abi)
    send(w3, c.functions.initialize(*init_args), pk, gas=6_000_000)
    return c


def set_parity(w3, registry, token: str, parity: int) -> None:
    """Put ``token`` in the value index (the Registry's owner, the timelock)."""
    send(w3, registry.functions.editRegistry(
        f"jurisdiction.parity.{token.lower()}", str(int(parity))), HH_TIMELOCK_PK)


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
        teardown = S[10]
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
    tok: Dict[str, Any] = {}
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
        timelock = address_of(HH_TIMELOCK_PK)
        treasury = address_of(HH_TREASURY_PK)
        fund_eth(w3, timelock)
        maker = deploy(w3, "CloneMaker", [])
        # The timelock owns the Registry (the DAO treasury and the value
        # index), as on a live network; the deployer is RepToken's admin.
        registry = deploy_clone(w3, maker, "Registry", [timelock, "0x" + "00" * 20])
        send(w3, registry.functions.editRegistry(CHARTER_KEY, DEFAULT_CHARTER.text),
             HH_TIMELOCK_PK)
        rep = deploy_clone(w3, maker, "RepToken", ["Autonet REP", "REP", registry.address,
                                                   timelock, [], []])
        # The value index: a pausable 6-decimal token and an 18-decimal one.
        tok["usd6"] = deploy(w3, "PausableToken", [], sol="IndexTokenMocks", sub="test")
        tok["t18"] = deploy(w3, "MockIndexToken", ["Test 18", "T18", 18],
                            sol="IndexTokenMocks", sub="test")
        for k, t in tok.items():
            set_parity(w3, registry, t.address, PARITY[k])
        # (treasury, governor, anchorer, registry, fee, treasury share):
        # operator A is the governor (anchors directly, an EOA governor as on
        # a dev network).
        substrate = deploy(w3, "Substrate", [treasury, op["A"], "0x" + "00" * 20,
                                             registry.address, SERVICE_FEE_BPS,
                                             FEE_TREASURY_BPS], core=True)
        bond = deploy_clone(w3, maker, "ToolBond", [substrate.address, rep.address,
                                                    registry.address, timelock,
                                                    tuple(PARAMS.values())])
        send(w3, rep.functions.setSlasher(bond.address, True), HH_DEPLOYER_PK)
        rep_claim = deploy_clone(w3, maker, "AutonetRepClaim",
                                 [substrate.address, rep.address, [], []])
        send(w3, rep.functions.setMinter(rep_claim.address, True), HH_DEPLOYER_PK)
        svc_registry = deploy(w3, "ServiceRegistry", [substrate.address],
                              sol="ServiceMarket", core=True)
        channel = deploy(w3, "PaymentChannel", [substrate.address, CHANNEL_WINDOW],
                         sol="ServiceMarket", core=True)
        addrs = {"substrate": substrate.address, "rep": rep.address,
                 "tool_bond": bond.address, "registry": registry.address,
                 "rep_claim": rep_claim.address,
                 "service_registry": svc_registry.address,
                 "payment_channel": channel.address,
                 "usd6": tok["usd6"].address, "t18": tok["t18"].address}
        check(substrate.functions.governor().call() == op["A"],
              "Substrate's governor is operator A (dev network: it anchors directly)")
        for k, t in tok.items():
            check(substrate.functions.acceptsToken(t.address).call()
                  and substrate.functions.parityOf(t.address).call() == PARITY[k],
                  f"{k} is in the value index at parity {PARITY[k]}")
        check(substrate.functions.valueOf(tok["t18"].address, 4 * E18).call() == 5 * U6,
              "4 T18 are worth 5 value units")
        check(rep.functions.slashers(bond.address).call(), "ToolBond is a RepToken slasher")
        check(rep.functions.minters(rep_claim.address).call(),
              "AutonetRepClaim is a RepToken minter")
        check(rep_claim.functions.substrate().call() == substrate.address,
              "AutonetRepClaim reads its earnings from Substrate")
        check(bond.functions.minStake().call() == MIN_STAKE
              and bond.functions.reviewerShareBps().call() == 4000,
              "ToolBond: minStake 100 value units, reviewerShareBps 4000")
        check(registry.functions.getRegistryValue(CHARTER_KEY).call() == DEFAULT_CHARTER.text,
              "the Registry holds the charter text")
        for k, v in addrs.items():
            s1.note(k, v)
        s1.status = "PASS"
    except Exception as e:
        fail(s1, e)
        return await finish()

    def bal(k: str, addr: str) -> int:
        return tok[k].functions.balanceOf(addr).call()

    def rep_of(addr: str) -> int:
        return rep.functions.balanceOf(addr).call()

    T = {k: t.address.lower() for k, t in tok.items()}   # token keys as the close has them

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
            for k, amount in FUNDS[h].items():
                send(w3, tok[k].functions.mint(op[h], amount), HH_DEPLOYER_PK)
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
        listed = await execute_tool("list_index_tokens", {}, rts["B"],
                                    caller_id=AGENT["B"])
        check(sorted(t["token"] for t in listed.get("tokens", [])) == sorted(T.values()),
              f"the daemon lists the value index: {[t['symbol'] for t in listed.get('tokens', [])]}")
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
        from atn.runtime.tool_bond import ToolBondClient, ToolBondGate
        # What the autonet bridge wires (autonet_service): a published
        # manifest goes to the library, which emits its tool_registered event.
        rt_a.tool_store.manifest_sink = lambda m, author: announced.append((m, author))
        usd0 = bal("usd6", op["A"])
        reg = await execute_tool("register_tool", {
            "name": "upper_case", "description": "Upper-cases a string.",
            "input_schema": UPPER_SCHEMA, "code": UPPER_CODE,
            "novelty": {"statement": "Upper-cases a string; no comparable tool exists yet.",
                        "prior_art": []}}, rt_a, caller_id=AGENT["A"])
        if "error" in reg:
            raise RuntimeError(f"register_tool: {reg}")
        # No token named: A's operator holds one index token only.
        pub = await execute_tool("publish_tool", {"digest": reg["digest"]}, rt_a,
                                 caller_id=AGENT["A"])
        if "error" in pub:
            raise RuntimeError(f"publish_tool: {pub}")
        digest = pub["digest"]
        b32 = bytes.fromhex(digest)
        st = pub["stake"]
        check(pub.get("published") is True and st["token"] == T["usd6"]
              and st["amount"] == MIN_STAKE and st["value"] == MIN_STAKE,
              f"published with a {MIN_STAKE // U6} USD6 bond worth {st['value'] // U6} "
              f"units ({len(st['txs'])} txs: registerTool, approve, bond)")
        manifest = rt_a.tool_store.get(digest).manifest
        check(str(manifest.get("author", "")).lower() == op["A"].lower(),
              "the manifest's author is operator A")
        check(substrate.functions.toolAuthor(b32).call() == op["A"],
              "Substrate.toolAuthor = operator A")
        row = bond.functions.bonds(b32).call()
        check(bond.functions.isBonded(b32).call() and row[5] == tok["usd6"].address
              and row[1] == MIN_STAKE and row[6] == MIN_STAKE,
              "ToolBond: bonded in USD6, stake and value 100 units")
        check(usd0 - bal("usd6", op["A"]) == MIN_STAKE, "operator A's USD6 down by the bond")
        check((await ToolBondGate(rt_b).is_bonded(digest))[0],
              "daemon B's gate sees the bond")
        again = await ToolBondClient(rt_a).register_and_bond(op_key["A"], digest)
        check(again.get("success") and again.get("note") == "already bonded"
              and not again.get("txs"), "bonding a bonded tool again sends nothing")
        wrong = await ToolBondClient(rt_a).register_and_bond(
            op_key["A"], digest, amount=E18, token=T["t18"])
        check(not wrong.get("success") and "keeps its token" in str(wrong.get("error")),
              "a top-up in another token is refused before any transaction")
        check(len(announced) == 1 and announced[0][0] == manifest
              and str(announced[0][1]).lower() == op["A"].lower(),
              "the daemon announces the published manifest once, author operator A")
        # B puts a bounty on it for its reviewers, in the bond's token.
        bty = await ToolBondClient(rt_b).add_bounty(op_key["B"], digest, BOUNTY)
        check(bty.get("success") and bty["token"] == T["usd6"]
              and bond.functions.bonds(b32).call()[4] == BOUNTY,
              f"B adds a {BOUNTY // U6} USD6 bounty (approve, addBounty)")
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
        held0 = {h: {k: bal(k, op[h]) for k in tok} for h in ("B", "C")}
        for d in drivers:
            d.on_local_close()
        got = await tick_all(drivers)
        check(got.get("committed") == 2, f"both batches committed ({got})")
        for h in ("B", "C"):
            c = bond.functions.commits(1, op[h]).call()
            check(c[0] != b"\x00" * 32 and c[1] == REP_GENESIS[h] * E18 and not c[2],
                  f"{h}: commitment from operator {h}, weight {REP_GENESIS[h]} REP "
                  "(its owner's, at the snapshot), hidden")
            check({k: bal(k, op[h]) for k in tok} == held0[h],
                  f"{h}: no token moved (a review stakes nothing)")
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
        owner_usd0 = {h: bal("usd6", owner[h]) for h in HOUSES}
        advance(w3, PARAMS["revealWindow"] + 1)
        got = await tick_all(drivers, ticks=2)
        check(bond.functions.getRound(b32, 1).call()[0] == 3,
              f"after the reveal window: round finalized and claimed ({got})")
        credit = rep.functions.slashedCredit(bond.address).call()
        check(rep.functions.totalSupply().call() + (credit - credit0) == supply0,
              "REP conserved by the review settlement")
        # The bounty: each review's part paid to its household by the
        # review's claim, in the bond's token (payOut only if deferred).
        paid = {str(e["args"]["household"]): int(e["args"]["amount"]) for e in
                bond.events.BountyPaid().get_logs(from_block=0, to_block="latest")}
        check(set(paid) == {owner["B"], owner["C"]}
              and BOUNTY - 2 <= sum(paid.values()) <= BOUNTY,
              f"the bounty is owed to households B and C "
              f"({ {k[:8]: v / U6 for k, v in paid.items()} } USD6)")
        for h in ("B", "C"):
            check(bond.functions.owed(owner[h], tok["usd6"].address).call() == 0
                  and bal("usd6", owner[h]) - owner_usd0[h] == paid[owner[h]],
                  f"{h}: its claim paid it to the owner wallet")
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
        s4.note("bounty_paid", {h: paid[owner[h]] / U6 for h in ("B", "C")})
        s4.status = "PASS"
    except Exception as e:
        fail(s4, e)
        return await finish()

    # ------------------------------------------------------------------
    s5 = S[5]
    log("Stage 5: service fees")
    service_id: Dict[str, int] = {}
    fee: Dict[str, int] = {}
    pool_share: Dict[str, int] = {}
    try:
        from atn.on_chain import OnChainService
        from atn.ws_server import WebSocketBridge
        from atn import service_catalog as sc
        # C's owner sells a local tool as two services, one priced in each
        # token, registered and paid under operator C, reachable at the
        # endpoint it publishes on chain.
        reg = await execute_tool("register_tool", {
            "name": "word_count", "description": "Counts the words in a text.",
            "input_schema": WORDS_SCHEMA, "code": WORDS_CODE}, rt_c, caller_id=None)
        if "error" in reg:
            raise RuntimeError(f"register_tool (C): {reg}")
        port_c = _free_port()
        ws_c = WebSocketBridge(rt_c, host="127.0.0.1", port=port_c)
        await ws_c.start()
        catalog_b = sc.catalog_for(rt_b)
        for k in ("usd6", "t18"):
            listed = await ws_c._handle_message({
                "type": "register_service", "msg_id": f"svc-{k}", "mode": "request",
                "name": f"word-count-{k}", "description": f"Word counts, paid in {k}.",
                "backing_tool": reg["digest"], "token": T[k],
                "price": ASK[k], "voucher": ASK[k]}, owner_session())
            if not listed.get("ok") or not listed["result"].get("on_chain"):
                raise RuntimeError(f"register_service ({k}): {listed}")
            service_id[k] = int(listed["result"]["service_id"])
            entry = svc_registry.functions.getService(service_id[k]).call()
            check(entry[0] == op["C"] and entry[2] == 0,
                  f"service #{service_id[k]} ({k}): provider operator C, no registry "
                  "price (the signed price list names the token and the price)")
            record_c = rt_c.service_store.get(listed["result"]["digest"])
            check(str(record_c.price_list.get("token", "")).lower() == T[k],
                  f"its price list names {k}")
            await asyncio.to_thread(catalog_b.sync_chain)
            check(catalog_b.accept_announcement(sc.announcement(
                record_c.spec, service_id[k], record_c.price_list)),
                  f"B's catalog holds the {k} service's spec and price list")
        ep = await OnChainService(rt_c._config.autonet).update_endpoint(
            op_key["C"], f"ws://127.0.0.1:{port_c}")
        check(ep.get("success") and substrate.functions.getAgentEndpoint(op["C"]).call()
              == f"ws://127.0.0.1:{port_c}", "C's endpoint published on chain")

        for k in ("usd6", "t18"):
            addr = tok[k].address
            before = {"b": bal(k, op["B"]), "c": bal(k, op["C"]),
                      "t": substrate.functions.treasuryOwed(addr).call(),
                      "pool": substrate.functions.toolShareTotal(addr).call()}
            # No credits with C: request_service pays this one request directly.
            reply = await execute_tool("request_service", {
                "service_id": service_id[k], "payload": {"text": "one turn of the loop"}},
                rt_b, caller_id=AGENT["B"])
            check(bool(reply.get("paid_directly")), f"{k}: paid with one direct payment")
            check(reply.get("ok") and '"words": 5' in json.dumps(reply),
                  f"{k}: C serves B's paid request over {reply.get('transport', '?')}")
            pay = substrate.events.ServicePayment().get_logs(from_block=0, to_block="latest")[-1]
            check(pay["args"]["payer"] == op["B"] and pay["args"]["recipient"] == op["C"]
                  and pay["args"]["token"] == addr and pay["args"]["amount"] == ASK[k],
                  f"{k}: ServicePayment from operator B to operator C in {k}")
            fee[k] = -(-ASK[k] * SERVICE_FEE_BPS // 10_000)
            to_treasury = fee[k] * FEE_TREASURY_BPS // 10_000
            pool_share[k] = fee[k] - to_treasury
            check(before["b"] - bal(k, op["B"]) == ASK[k]
                  and bal(k, op["C"]) - before["c"] == ASK[k] - fee[k],
                  f"{k}: B paid {ASK[k]}, C got it net of the {fee[k]} fee")
            check(substrate.functions.treasuryOwed(addr).call() - before["t"] == to_treasury
                  and substrate.functions.toolShareTotal(addr).call() - before["pool"]
                  == pool_share[k],
                  f"{k}: half the fee owed to the treasury, {pool_share[k]} to the pool")
        s5.note("service_ids", service_id)
        s5.note("pool_share", pool_share)
        s5.status = "PASS"
    except Exception as e:
        fail(s5, e)
        return await finish()

    # ------------------------------------------------------------------
    s6 = S[6]
    log("Stage 6: close, anchor, claims and withdrawals")
    close: Dict[str, Any] = {}
    rep_before: Dict[str, int] = {}
    outs: Dict[str, Dict[str, Any]] = {}
    delegate_a0 = ""
    claims: Dict[str, Any] = {}
    try:
        from nodes.common.blob_resolver import InMemoryBlobResolver
        from nodes.common.canonical_ordering import canonical_order
        from nodes.common.chain_submission_driver import (
            AgentChainIdentity, ChainSubmissionConfig, ChainSubmissionDriver)
        from nodes.common.event_gossip import EventBatch, Keypair
        from nodes.common.federated_close_driver import FederatedCloseResult
        from nodes.common.federated_reconcile import UNMINTED, federated_epoch_close
        from nodes.common.library.tool_manifest import registration_event
        from nodes.common.mint_merkle import claim_merkle_root
        from nodes.common.voice_state import ANCHOR_CLAIM_ROOT, read_voice_state

        anchor_epoch(w3, substrate, "e2", op_key["A"])
        advance(w3, 5)   # RepToken checkpoints only answer for past timepoints
        state = read_voice_state(substrate.address, RPC_URL, rep_token_address=rep.address,
                                 tool_bond_address=bond.address,
                                 dao_registry_address=registry.address, web3=w3)
        snap = state["review_snapshot"]
        check(digest in snap["tools"] and digest in snap["rounds"],
              "the snapshot pinned to e2 holds the tool's rating and round")
        check(snap["bonds"][digest]["value"] == MIN_STAKE
              and snap["min_stake"] == MIN_STAKE,
              "the snapshot's bond: value 100 units against minStake 100")
        check(all(state["owner_map"].get(op[h].lower()) == owner[h].lower() for h in HOUSES),
              "owner map from chain: each operator to its owner wallet")
        window = {str(t).lower(): int(v) for t, v in state["tool_share_window"].items()}
        check(window == {T[k]: pool_share[k] for k in tok},
              f"the pools = the tool share collected between e1 and e2, per token ({window})")
        parity = {str(t).lower(): int(v) for t, v in state["parity"].items()}
        check(all(parity.get(T[k]) == PARITY[k] for k in tok),
              "the index pinned to e2: both parities")

        manifest, author = announced[0]
        kp = Keypair.generate()
        batch = EventBatch(rpb_address="rpb_e2e_economy", sender_pubkey=kp.public_key,
                           batch_seq=1, events=[registration_event(
                               manifest, digest, agent_id=author)],
                           prev_batch_hash=b"", timestamp=1_700_000_000.0)
        close = federated_epoch_close(
            canonical_order([batch]), tool_share_window=state["tool_share_window"],
            parity=state["parity"], agent_owner_map=state["owner_map"],
            rep_shares=state["rep_shares"], review_snapshot=snap,
            mint_gate=tuple(state["mint_gate"]))
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
        claims = close["claims"]
        hk = {h: owner[h].lower() for h in HOUSES}
        check(set(claims) == set(hk.values()) and UNMINTED not in claims,
              "the claims are keyed by the three owner wallets (households), not operators")
        pools = {str(t).lower(): int(v) for t, v in close["pools"].items()}
        check(pools == window, "one concluding tool: its share is the whole of each pool")

        def frac(h: str, k: str) -> float:
            return claims[hk[h]]["tokens"].get(T[k], 0) / pools[T[k]]
        # The basket: every earner takes the same fraction of every pool
        # (each amount floored, so within one base unit of each pool).
        for h in HOUSES:
            check(abs(frac(h, "usd6") - frac(h, "t18")) <= 2 / pools[T["usd6"]],
                  f"{h}: the same fraction of both pools ({frac(h, 'usd6'):.6f}, "
                  f"{frac(h, 't18'):.6f})")
        check(close_to(frac("A", "usd6"), 0.6 * want_score * want_n, 2e-6),
              f"household A: the author side, 60% x V x C x S x N {want_n:.2f} = "
              f"{frac('A', 'usd6'):.6f} of each pool")
        check(close_to(frac("B", "usd6") + frac("C", "usd6"), 0.4, 2e-6),
              "households B + C: 40% of each pool (the reviewers, by accuracy)")
        carry = {str(t).lower(): int(v) for t, v in close["tool_rounds"]["pool_carry"].items()}
        for k in tok:
            paid = sum(c["tokens"].get(T[k], 0) for c in claims.values())
            check(paid + carry.get(T[k], 0) == pools[T[k]],
                  f"{k}: paid {paid} + carried {carry.get(T[k], 0)} = the pool {pools[T[k]]}")
        for h in HOUSES:
            want_value = sum(a * PARITY[k] // PARITY_SCALE for k in tok
                             for a in [claims[hk[h]]["tokens"].get(T[k], 0)])
            check(abs(claims[hk[h]]["value"] - want_value) <= 1,
                  f"{h}: its basket is worth {claims[hk[h]]['value'] / U6:.6f} units")

        # The daemons take the close: A's driver anchors it (its operator is
        # Substrate's governor), every driver claims for its household; the
        # claim pays the basket in the same transaction.
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
        held0 = {h: {k: (bal(k, owner[h]), bal(k, op[h])) for k in tok} for h in HOUSES}
        rep_before.update({h: rep_of(owner[h]) for h in HOUSES})
        delegate_a0 = str(rep.functions.delegates(owner["A"]).call())
        outs["A"] = await asyncio.to_thread(csd["A"].handle_federated_close, fed)
        check(outs["A"].get("anchored"),
              f"daemon A's driver anchors e3 ({outs['A'].get('anchor_error', 'direct')})")
        a3 = substrate.functions.getAnchorByEpochId("e3").call()
        check(bytes(a3[ANCHOR_CLAIM_ROOT]) == claim_merkle_root(
            claims, concluded=close["concluded"]),
              "e3's claimRoot is the close's claim and concluded leaves")
        idx3 = int(substrate.functions.anchorCount().call()) - 1
        check(all(substrate.functions.anchorParity(idx3, t.address).call() == PARITY[k]
                  for k, t in tok.items()), "the anchor stored both parities")
        outs["B"] = await asyncio.to_thread(
            csd["B"].handle_federated_close, dataclasses.replace(fed, is_winner=False))
        check(outs["B"].get("anchor_skipped") == "already anchored",
              "daemon B's driver finds e3 anchored and only claims")
        # USD6's issuer pauses it before C's daemon takes the close.
        send(w3, tok["usd6"].functions.setPaused(True), HH_DEPLOYER_PK)
        outs["C"] = await asyncio.to_thread(
            csd["C"].handle_federated_close, dataclasses.replace(fed, is_winner=False))
        check(outs["C"].get("anchor_skipped") == "already anchored",
              "daemon C's driver finds e3 anchored and only claims")
        for h in HOUSES:
            want = claims[hk[h]]["tokens"]
            row = next((r for r in outs[h]["agent_submissions"]
                        if r["claimant"].lower() == hk[h]), {})
            check(row.get("success") and {str(t).lower(): int(v) for t, v in
                                          row.get("tokens", {}).items()} == want,
                  f"{h}: operator claims its owner's basket ({ {k: want.get(T[k], 0) for k in tok} })")
            for k in tok:
                got = bal(k, owner[h]) - held0[h][k][0]
                if h == "C" and k == "usd6":
                    deferred = [e for e in substrate.events.PayDeferred().get_logs(
                        from_block=0, to_block="latest")
                        if str(e["args"]["earner"]).lower() == owner[h].lower()]
                    check(got == 0 and substrate.functions.owed(owner[h], tok[k].address).call()
                          == want[T[k]] and len(deferred) == 1
                          and int(deferred[0]["args"]["amount"]) == want[T[k]]
                          and {str(t).lower(): int(v) for t, v in row.get("owed", {}).items()}
                          == {T[k]: want[T[k]]},
                          "C: USD6 is paused, so the claim deferred it (PayDeferred); it stays owed")
                else:
                    check(got == want.get(T[k], 0)
                          and substrate.functions.owed(owner[h], tok[k].address).call() == 0
                          and not row.get("withdrawn"),
                          f"{h}: {k} paid to the owner wallet by the claim itself (+{got})")
                check(bal(k, op[h]) == held0[h][k][1],
                      f"{h}: the operator received no {k} (it paid the gas)")
        send(w3, tok["usd6"].functions.setPaused(False), HH_DEPLOYER_PK)
        send(w3, substrate.functions.withdraw(owner["C"], tok["usd6"].address), HH_DEPLOYER_PK)
        check(bal("usd6", owner["C"]) - held0["C"]["usd6"][0] == claims[hk["C"]]["tokens"][T["usd6"]]
              and substrate.functions.owed(owner["C"], tok["usd6"].address).call() == 0,
              "unpaused, C's USD6 is withdrawn (anyone may send it; it goes to the earner)")
        for h in HOUSES:
            want_value = sum(claims[hk[h]]["tokens"].get(T[k], 0) * PARITY[k] // PARITY_SCALE
                             for k in tok)
            check(substrate.functions.earnedValue(owner[h]).call() == want_value,
                  f"{h}: earnedValue {want_value / U6:.6f} units (the amounts at the anchor's "
                  "parities)")
        for k, t in tok.items():
            check(substrate.functions.claimedTotal(t.address).call()
                  <= substrate.functions.poolCap(t.address).call(),
                  f"{k}: claims stay within the pool cap")
            owed_t = substrate.functions.treasuryOwed(t.address).call()
            t0 = bal(k, treasury)
            send(w3, substrate.functions.withdrawTreasury(t.address), HH_DEPLOYER_PK)
            check(bal(k, treasury) - t0 == owed_t == fee[k] - pool_share[k],
                  f"{k}: the treasury withdraws its share of the fee ({owed_t})")
        v = bond.functions.verdicts(bytes.fromhex(digest)).call()
        check(v[0] > 0 and v[1] == 0 and not v[2],
              "ToolBond: the tool concluded from the anchored leaf, no malware")
        check(bond.functions.isBonded(bytes.fromhex(digest)).call(), "and it keeps its bond")
        s6.note("pools", pools)
        s6.note("claims", {h: claims[hk[h]]["tokens"] for h in HOUSES})
        s6.note("carry", carry)
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
            earned = substrate.functions.earnedValue(owner[h]).call()
            ev = claimed.get(owner[h], {})
            check(ev.get("to") == owner[h] and ev.get("earningsDelta") == earned
                  and ev.get("repMinted") == earned * 10 ** 12,
                  f"{h}: RepClaimed {earned / U6:.6f} REP to the owner wallet "
                  "(1:1 on earnedValue)")
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
    log("Stage 8: the REP floor")
    try:
        # The DAO sets the REP floor (the timelock owns the Registry).
        send(w3, registry.functions.editRegistry(REP_FLOOR_KEY, str(REP_FLOOR)),
             HH_TIMELOCK_PK)
        floor_v = REP_FLOOR // 10 ** 12
        check(substrate.functions.repFloorValue().call() == floor_v,
              f"Substrate reads the floor: {REP_FLOOR // E18} REP = {floor_v / U6:g} value units")
        # A publishes a second tool; B and C adopt it, use it and review it.
        reg2 = await execute_tool("register_tool", {
            "name": "lower_case", "description": "Lower-cases a string.",
            "input_schema": UPPER_SCHEMA, "code": LOWER_CODE,
            "novelty": {"statement": "Lower-cases a string; the library only upper-cases.",
                        "prior_art": [digest]}}, rt_a, caller_id=AGENT["A"])
        if "error" in reg2:
            raise RuntimeError(f"register_tool (lower_case): {reg2}")
        n_announced = len(announced)
        pub2 = await execute_tool("publish_tool", {"digest": reg2["digest"]}, rt_a,
                                  caller_id=AGENT["A"])
        if "error" in pub2:
            raise RuntimeError(f"publish_tool (lower_case): {pub2}")
        digest2 = pub2["digest"]
        b32_2 = bytes.fromhex(digest2)
        check(bond.functions.isBonded(b32_2).call() and len(announced) == n_announced + 1,
              "A publishes lower_case, bonded and announced")
        epoch_r = int(bond.functions.currentEpoch().call())
        for h, word in (("B", "Floor Epoch"), ("C", "No Fees")):
            ad = await execute_tool("adopt_tool", {"digest": digest2, "reason": "titles"},
                                    rts[h], caller_id=AGENT[h])
            check("error" not in ad, f"{h} adopts lower_case")
            out = await execute_tool("use_tool", {"name": "lower_case",
                                                  "arguments": {"x": word}},
                                     rts[h], caller_id=AGENT[h])
            check(word.lower() in json.dumps(out), f"{h} uses it")
            att = await execute_tool("attest_tools", {
                "judgments": [{"tool": digest2, "ok": True, **REVIEWS[h]}],
                "context": "lower-casing a title"}, rts[h], caller_id=AGENT[h])
            rv = att.get("reviews") or {}
            check(rv.get("queued") == 1 and rv.get("epoch") == epoch_r,
                  f"{h}: review queued for epoch {epoch_r} ({rv})")
        for d in drivers:
            d.on_local_close()
        got = await tick_all(drivers)
        check(got.get("committed") == 2, f"both batches committed ({got})")
        anchor_epoch(w3, substrate, "e4", op_key["A"])
        got = await tick_all(drivers)
        check(got.get("revealed") == 2, f"anchor e4: both batches revealed ({got})")
        advance(w3, PARAMS["revealWindow"] + 1)
        got = await tick_all(drivers, ticks=2)
        check(bond.functions.getRound(b32_2, epoch_r).call()[0] == 3,
              f"after the reveal window: lower_case's round finalized ({got})")

        # An epoch that collected no fees: the snapshot at e5 has an empty
        # window, so the pool is only what the first close carried.
        anchor_epoch(w3, substrate, "e5", op_key["A"])
        advance(w3, 5)
        state2 = read_voice_state(substrate.address, RPC_URL, rep_token_address=rep.address,
                                  tool_bond_address=bond.address,
                                  dao_registry_address=registry.address, web3=w3)
        check(state2["tool_share_window"] == {},
              "no fees collected between e4 and e5: the window is empty")
        check(state2["rep_floor"] == floor_v, "the close reads the floor pinned to e5")
        manifest2, author2 = announced[-1]
        kp2 = Keypair.generate()
        batch2 = EventBatch(rpb_address="rpb_e2e_economy", sender_pubkey=kp2.public_key,
                            batch_seq=1, events=[registration_event(
                                manifest2, digest2, agent_id=author2)],
                            prev_batch_hash=b"", timestamp=1_700_000_100.0)
        close2 = federated_epoch_close(
            canonical_order([batch2]), tool_share_window=state2["tool_share_window"],
            parity=state2["parity"], agent_owner_map=state2["owner_map"],
            rep_shares=state2["rep_shares"], review_snapshot=state2["review_snapshot"],
            mint_gate=tuple(state2["mint_gate"]),
            tool_registrations=close["tool_registrations"],
            tool_rounds=close["tool_rounds"], rep_floor=state2["rep_floor"])
        check(close2["concluded"] == {digest2: 0}, "lower_case concludes")
        claims2 = close2["claims"]
        v2, s2_seed = close2["total_value"], close2["total_seed"]
        check(set(claims2) == set(hk.values()),
              "the three households earn (A the author side, B and C the reviewers')")
        check(close2["rep_floor"] == floor_v and s2_seed > 0
              and floor_v - len(claims2) <= v2 + s2_seed <= floor_v,
              f"the carry pays {v2 / U6:.6f} units, the floor seeds {s2_seed / U6:.6f}: "
              f"together the floor, {floor_v / U6:g}")
        weight = close2["household_weights"]
        w_sum = sum(weight.values())
        for h in HOUSES:
            want = (floor_v - v2) * weight[hk[h]] // w_sum
            check(claims2[hk[h]]["seed"] == want,
                  f"{h}: seed {want / U6:.6f} units, its weight's share of the difference")
        carry2 = close2["tool_rounds"]["pool_carry"]
        for k in tok:
            paid = sum(c["tokens"].get(T[k], 0) for c in claims2.values())
            check(paid + int(carry2.get(T[k], 0)) == int(close2["pools"].get(T[k], 0)),
                  f"{k}: the carry is paid or carried again, never more")

        fed2 = FederatedCloseResult(
            epoch_id="e6", close_result={**close2, "epoch_id": "e6"},
            senders=[kp2.public_key], winner=kp2.public_key, is_winner=True, n_batches=1,
            snapshot_anchor_count=int(substrate.functions.anchorCount().call()))
        earned0 = {h: substrate.functions.earnedValue(owner[h]).call() for h in HOUSES}
        rep0 = {h: rep_of(owner[h]) for h in HOUSES}
        outs2: Dict[str, Dict[str, Any]] = {}
        outs2["A"] = await asyncio.to_thread(csd["A"].handle_federated_close, fed2)
        check(outs2["A"].get("anchored"),
              f"daemon A's driver anchors e6 ({outs2['A'].get('anchor_error', 'direct')})")
        idx6 = int(substrate.functions.anchorCount().call()) - 1
        check(substrate.functions.anchorFloor(idx6).call() == floor_v,
              "the anchor stored the floor in value units")
        for h in ("B", "C"):
            outs2[h] = await asyncio.to_thread(
                csd[h].handle_federated_close, dataclasses.replace(fed2, is_winner=False))
        minted = 0
        for h in HOUSES:
            seed = claims2[hk[h]]["seed"]
            row = next((r for r in outs2[h]["agent_submissions"]
                        if r["claimant"].lower() == hk[h]), {})
            check(row.get("success") and row.get("seed") == seed,
                  f"{h}: operator claims its owner's leaf with its seed ({seed})")
            basket = sum(claims2[hk[h]]["tokens"].get(T[k], 0) * PARITY[k] // PARITY_SCALE
                         for k in tok)
            earned = substrate.functions.earnedValue(owner[h]).call() - earned0[h]
            check(earned == basket + seed
                  and substrate.functions.seededValue(owner[h]).call() == seed,
                  f"{h}: earnedValue +{earned / U6:.6f} (basket {basket / U6:.6f} + seed "
                  f"{seed / U6:.6f}); seededValue {seed / U6:.6f}")
            got_rep = rep_of(owner[h]) - rep0[h]
            check(got_rep == earned * 10 ** 12,
                  f"{h}: AutonetRepClaim mints {got_rep / E18:.6f} REP (1:1, the seed included)")
            minted += got_rep
        check(substrate.functions.seedClaimed(idx6).call() == s2_seed
              and substrate.functions.totalSeededValue().call() == s2_seed,
              "the seeds claimed are the close's, within the stored floor")
        check(REP_FLOOR - len(claims2) * 10 ** 12 <= minted <= REP_FLOOR,
              f"the epoch minted the floor: {minted / E18:.6f} REP")
        s8.note("floor_rep", REP_FLOOR // E18)
        s8.note("paid_value", v2)
        s8.note("seeded_value", s2_seed)
        s8.note("rep_minted", round(minted / E18, 6))
        s8.status = "PASS"
    except Exception as e:
        fail(s8, e)
        return await finish()

    # ------------------------------------------------------------------
    s9 = S[9]
    log("Stage 9: payment channel")
    try:
        from atn.on_chain import ServiceMarketClient
        from atn.service_channels import sweep_refunds
        k, price = "t18", ASK["t18"]
        before = {"b": bal(k, op["B"]), "c": bal(k, op["C"])}
        opened = await execute_tool("open_credits", {
            "provider_address": op["C"], "token": T[k], "deposit": 3 * price},
            rt_b, caller_id=AGENT["B"])
        if not opened.get("ok"):
            raise RuntimeError(f"open_credits: {opened}")
        cid = int(opened["credits"]["channel_id"])
        replies = []
        for text in ("a credits request", "and a second one here"):
            replies.append(await execute_tool("request_service", {
                "service_id": service_id[k], "payload": {"text": text}},
                rt_b, caller_id=AGENT["B"]))
        check(all(r.get("ok") for r in replies)
              and replies[1]["credits"]["used"] == 2 * price
              and not any(r.get("paid_directly") for r in replies),
              "two requests served on B's T18 credits with C, one price each")
        ch = channel.functions.getChannel(cid).call()
        check(ch[0] == op["B"] and ch[1] == op["C"] and ch[2] == tok[k].address
              and ch[3] == 3 * price and ch[6] == 1,
              f"channel {cid} open: operator B to operator C, 3 x the price in T18 escrowed")
        kept = rt_c.service_store.get_channel_voucher(cid) or {}
        check(int(kept.get("cumulative", 0)) == 2 * price
              and int(kept.get("consumed", 0)) == 2 * price,
              "C holds B's voucher for 2 x the price, all of it served")
        check(before["b"] - bal(k, op["B"]) == 3 * price,
              "B's deposit left its operator, nothing else")
        # C's sweep settles what it served once it is worth the minimum
        # (a value), and the channel stays open.
        rt_c._config.autonet.channel_settle_min = 1
        out = await rt_c.settle_channels_once()
        settled = out.get("settled") or []
        check(settled and settled[0].get("ok") and settled[0].get("action") == "settled",
              f"C's sweep settles without closing ({settled})")
        ch = channel.functions.getChannel(cid).call()
        cfee = -(-2 * price * SERVICE_FEE_BPS // 10_000)
        check(ch[6] == 1 and ch[4] == 2 * price
              and bal(k, op["C"]) - before["c"] == 2 * price - cfee,
              f"still open; 2 x the price paid, C net of the {cfee / E18:g} T18 fee")
        # B closes; C settles inside the window (nothing new), B withdraws.
        closed = await execute_tool("close_credits", {"provider_address": op["C"]},
                                    rt_b, caller_id=AGENT["B"])
        check(closed.get("ok"), f"B closes its credits ({closed})")
        await rt_c.settle_channels_once()
        advance(w3, CHANNEL_WINDOW + 1)
        # The run moved the chain clock ahead of the wall clock
        # (evm_increaseTime), so the sweep is given the chain's time.
        refunds = await sweep_refunds(rt_b.channel_book,
                                      ServiceMarketClient(rt_b._config.autonet),
                                      rt_b._channel_key_for, now=chain_time(w3))
        actions = [r["action"] for r in refunds]
        check("withdrew" in actions and channel.functions.getChannel(cid).call()[6] == 3,
              f"B's refund sweep withdraws the remainder ({actions})")
        check(bal(k, op["B"]) - (before["b"] - 3 * price) == price
              and rt_b.channel_book.closing() == {},
              f"B got {price / E18:g} T18 back, fee-free; its book is clear")
        s9.note("channel", cid)
        s9.note("settled", f"{2 * price / E18:g} T18")
        s9.note("refunded", f"{price / E18:g} T18")
        s9.status = "PASS"
    except Exception as e:
        fail(s9, e)

    return await finish()

if __name__ == "__main__":
    try:
        rc = asyncio.run(amain())
    finally:
        for _proc in _NODES:
            kill_node(_proc)
    sys.exit(rc)
