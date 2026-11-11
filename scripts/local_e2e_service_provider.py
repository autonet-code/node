#!/usr/bin/env python3
"""End-to-end proof of request services, credits and reviews (docs/
services_market.md, Decision 2026-11-12), on a real local chain.

Two daemons in one process (a seller and a buyer), each a real Runtime; the
seller's WebSocket server is real and the buyer reaches it over its on-chain
wss endpoint. Nothing is mocked but the close's gossip input:

    the seller sells one of its tools (it counts words) as a request service:
    a signed spec and price list, registered on chain with no registry price
      -> the buyer's catalog takes the spec and the signed price list
      -> a one-off request paid directly (payForService), its replay refused
      -> the buyer opens credits; vouchers pay ahead by the voucher size;
         a request the tool fails costs nothing
      -> a new price list, with no transaction and no new spec
      -> the seller cashes in without closing (settle, ChannelPaid; the fee
         burns), and the credits keep serving
      -> a top-up, then requests until the credits are spent
      -> the buyer closes; the seller settles inside the window without moving
         the buyer's deadline; the remainder comes back
      -> the buyer reviews the service: a close pinned before any payment holds
         it, one pinned after counts it, and search shows the rating

Sessions (a connection kept open, metered per second, gigabyte and token)
need libp2p between daemons and are proved by tests/test_service_session.py,
the mesh NAT harness and the S0 run.

Run:
    python scripts/local_e2e_service_provider.py
"""
from __future__ import annotations

import asyncio
import json
import os
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

# Isolation before ANY atn / nodes import: the keystore resolves
# KEYSTORE_DIR at import time and several modules read Path.home(), so the
# run's agent and operator keys must never land in the real ~/.atn vault.
# Child processes (npx / hardhat) get the original environment.
_ORIG_ENV = dict(os.environ)
_TMP_HOME = Path(tempfile.mkdtemp(prefix="e2e_svcprov_home_"))
os.environ["KEYSTORE_DIR"] = str(_TMP_HOME / "keystore")
os.environ["USERPROFILE"] = str(_TMP_HOME)
os.environ["HOME"] = str(_TMP_HOME)
os.environ.setdefault("ATN_SEARCH_MODEL", "off")

RPC_URL = "http://127.0.0.1:8545"
# `npx hardhat node` serves the in-process "hardhat" network (chainId 1337).
CHAIN_ID = 1337

# Hardhat's deterministic mnemonic accounts (well-known, safe: local only).
HH_ACCT0_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
HH_ACCT1_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

PRICE = 1000            # ATN base units per request
NEW_PRICE = 2000        # after the price change
VOUCHER = 3000          # what each voucher pays ahead
DEPOSIT = 9000          # the buyer's credits
TOP_UP = 6000
BUYER_FUNDING = 50_000
SELLER_WS_PORT = 7791
# PaymentChannel challenge window (seconds): short, so the refund stage
# waits it out in real time.
CHANNEL_WINDOW = 6


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

class Stage:
    def __init__(self, num: int, name: str):
        self.num = num
        self.name = name
        self.status = "PENDING"  # PASS | FAIL | SKIP
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
        lines = ["", "=" * 74,
                 "  SERVICES LOCAL E2E — SCOREBOARD", "=" * 74]
        for num in sorted(self.stages):
            s = self.stages[num]
            mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP",
                    "PENDING": "----"}[s.status]
            lines.append(f"  [{mark}] Stage {s.num} — {s.name}")
            for n in s.notes:
                lines.append(f"           {n}")
        lines.append("=" * 74)
        n_pass = sum(1 for s in self.stages.values() if s.status == "PASS")
        n_fail = sum(1 for s in self.stages.values() if s.status == "FAIL")
        n_skip = sum(1 for s in self.stages.values() if s.status == "SKIP")
        lines.append(f"  {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
        lines.append("=" * 74)
        return "\n".join(lines)


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


def step(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


# ---------------------------------------------------------------------------
# Chain harness (same shape as local_e2e_tool_economy.py)
# ---------------------------------------------------------------------------

def _npx() -> str:
    return "npx.cmd" if os.name == "nt" else "npx"


def start_hardhat(cwd: Path) -> subprocess.Popen:
    """Launch `npx hardhat node` detached; caller polls the RPC.

    stdout/stderr go to DEVNULL: the node logs every RPC call and a filled
    PIPE buffer blocks the process (surfaces as an undici HeadersTimeoutError
    on the deploy side).
    """
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        [_npx(), "hardhat", "node"],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        shell=(os.name == "nt"),
        creationflags=creationflags,
        env=dict(_ORIG_ENV),
    )


def wait_rpc(timeout: float = 45.0) -> bool:
    from web3 import Web3
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            w3 = Web3(Web3.HTTPProvider(RPC_URL))
            if w3.is_connected() and w3.eth.block_number >= 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_hardhat(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, shell=True)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def load_abi(name: str) -> list:
    """Load a compiled contract ABI from Hardhat artifacts.

    ServiceMarket.sol holds several contracts, so the .sol stem differs from
    the contract name — scan the tree rather than guess.
    """
    art = REPO / "artifacts" / "contracts"
    for sub in ("core", "test"):
        for sol in (name, "ServiceMarket"):
            p = art / sub / f"{sol}.sol" / f"{name}.json"
            if p.exists():
                with p.open("r", encoding="utf-8") as fh:
                    return json.load(fh)["abi"]
    for p in art.rglob(f"{name}.json"):
        if p.parent.name.endswith(".sol"):
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)["abi"]
    raise FileNotFoundError(f"no artifact ABI for {name} under {art}")


DEPLOY_JS = r"""
const { ethers } = require("hardhat");
const fs = require("fs");
async function main() {
  const [deployer] = await ethers.getSigners();
  // Substrate(treasury, vaultMinter, governor), Decision 2026-10-16:
  // deployer = treasury + test vaultMinter (purchased-ATN faucet) + governor
  // (the only anchoring account). Nobody retunes the fee here, so it stays
  // at its 2.5% default, which is what this script asserts.
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy(
    deployer.address, deployer.address, deployer.address, ethers.ZeroAddress,
    ethers.ZeroHash, "Test ATN", "REPATN", 250, 5000);
  await substrate.waitForDeployment();
  const substrateAddr = await substrate.getAddress();

  // ServiceRegistry's constructor takes the substrate address. Every buy
  // pays through the PaymentChannel: the bound child's provider and the
  // request_service agent tool alike.
  const Registry = await ethers.getContractFactory("ServiceRegistry");
  const registry = await Registry.deploy(substrateAddr);
  await registry.waitForDeployment();
  const registryAddr = await registry.getAddress();

  const Channel = await ethers.getContractFactory("PaymentChannel");
  const channel = await Channel.deploy(substrateAddr,
                                       Number(process.env.CHANNEL_WINDOW));
  await channel.waitForDeployment();
  const channelAddr = await channel.getAddress();

  const out = { substrate: substrateAddr, serviceRegistry: registryAddr,
                paymentChannel: channelAddr };
  fs.writeFileSync(process.env.DEPLOY_OUT, JSON.stringify(out, null, 2));
  console.log("DEPLOYED " + JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""


def deploy_contracts(cwd: Path) -> Dict[str, Optional[str]]:
    scripts_dir = cwd / "scripts"
    tmp_js = scripts_dir / "_e2e_svcprov_deploy_tmp.js"
    out_json = cwd / "deployments" / "_e2e_service_provider.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_js.write_text(DEPLOY_JS, encoding="utf-8")
    try:
        env = dict(_ORIG_ENV)
        env["DEPLOY_OUT"] = str(out_json)
        env["CHANNEL_WINDOW"] = str(CHANNEL_WINDOW)
        log("deploying Substrate + ServiceRegistry + PaymentChannel via "
            "hardhat run…")
        proc = subprocess.run(
            [_npx(), "hardhat", "run", str(tmp_js), "--network", "localhost"],
            cwd=str(cwd), capture_output=True, text=True, timeout=300,
            shell=(os.name == "nt"), env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"deploy failed (rc={proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return json.loads(out_json.read_text(encoding="utf-8"))
    finally:
        try:
            tmp_js.unlink()
        except OSError:
            pass


def resolve_binding_shape(substrate_abi: list) -> Dict[str, Any]:
    """Discover which owner-binding surface the deployed contract exposes
    (registerAgentBound vs the older registerAgentSponsored)."""
    fns = {a["name"]: a for a in substrate_abi if a.get("type") == "function"}
    shape: Dict[str, Any] = {}
    shape["register"] = ("registerAgentBound" if "registerAgentBound" in fns
                         else "registerAgentSponsored"
                         if "registerAgentSponsored" in fns else None)
    shape["hash_view"] = ("bindingHash" if "bindingHash" in fns
                          else "sponsorshipHash" if "sponsorshipHash" in fns
                          else None)
    shape["nonce_view"] = ("bindingNonce" if "bindingNonce" in fns
                           else "sponsorshipNonce" if "sponsorshipNonce" in fns
                           else None)
    return shape


def sign_binding_digest(contract, shape: Dict[str, Any], owner_pk: str,
                        agent_addr: str, parent_addr: str, nonce: int) -> bytes:
    """Ask the CONTRACT for the exact EIP-712 digest, then sign that raw
    digest. Sidesteps any struct-field / typehash drift entirely."""
    from eth_account import Account

    view_fn = getattr(contract.functions, shape["hash_view"])
    view_abi = next(a for a in contract.abi
                    if a.get("type") == "function"
                    and a["name"] == shape["hash_view"])
    if len(view_abi["inputs"]) >= 3:
        digest = view_fn(agent_addr, parent_addr, nonce).call()
    else:
        digest = view_fn(agent_addr, parent_addr).call()
    digest = bytes(digest)
    acct = Account.from_key(owner_pk)
    signed = (Account._sign_hash(digest, acct._private_key)
              if hasattr(Account, "_sign_hash")
              else acct.unsafe_sign_hash(digest))
    return bytes(signed.signature)


def send_tx(w3, call, pk: str, addr: str, gas: int = 1_500_000) -> str:
    tx = call.build_transaction({
        "from": addr,
        "nonce": w3.eth.get_transaction_count(addr),
        "gas": gas, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
    })
    signed = w3.eth.account.sign_transaction(tx, pk)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt.status != 1:
        raise RuntimeError(f"tx reverted: {receipt}")
    return tx_hash.hex()


def fund_gas(w3, from_acct, to_addr: str, ether: int = 1) -> None:
    tx = {
        "from": from_acct.address, "to": to_addr,
        "value": w3.to_wei(ether, "ether"),
        "nonce": w3.eth.get_transaction_count(from_acct.address),
        "gas": 21000, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
    }
    signed = w3.eth.account.sign_transaction(tx, from_acct._private_key.hex())
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    w3.eth.wait_for_transaction_receipt(
        w3.eth.send_raw_transaction(raw), timeout=60)


def register_agent_on_chain(w3, substrate, shape, *, agent_pk: str,
                            agent_addr: str, owner_acct, peer_id: bytes) -> str:
    """Owner-bound registration signed by the agent's own key."""
    nonce_val = 0
    if shape["nonce_view"]:
        nonce_val = getattr(substrate.functions,
                            shape["nonce_view"])(agent_addr).call()
    ZERO = "0x" + "00" * 20
    sig = sign_binding_digest(substrate, shape,
                              owner_acct._private_key.hex(),
                              agent_addr, ZERO, nonce_val)
    reg_fn = getattr(substrate.functions, shape["register"])
    return send_tx(
        w3,
        reg_fn(peer_id, owner_acct.address, ZERO, sig),
        agent_pk, agent_addr)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def make_runtime(tmp_root: Path, *, owner_wallet: str, owner_pk: str,
                 addresses: Dict[str, Optional[str]]):
    """A real ATN Runtime wired to the local chain.

    ``owner_pk`` empty is a normal install: no configured daemon key, so
    the operator key is the one the vault holds. The two daemons share one
    process and so one vault, so only one of them (the seller) can take
    that path; the buyer keeps a configured key."""
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime

    data_dir = tmp_root / "data"
    agents_dir = tmp_root / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    # autonet.enabled False keeps the p2p/epoch machinery out of the way; the
    # CHAIN surface (config.rpb) is fully live — that is what the payment gate,
    # the registry reads and the endpoint resolution all use.
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    config.autonet.private_key = owner_pk
    config.autonet.rpc_url = RPC_URL
    config.autonet.chain_id = CHAIN_ID
    config.autonet.substrate_address = addresses["substrate"] or ""
    config.autonet.service_registry_address = addresses["serviceRegistry"] or ""
    config.autonet.payment_channel_address = addresses.get("paymentChannel") or ""
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def owner_session():
    """A privileged LOCAL session — what a localhost app connection gets.

    The WS handlers take a ClientSession, not a bool; the local listener
    constructs one pre-authed as the owner (ws_auth.ClientSession docstring).
    """
    from atn.ws_auth import ClientSession
    return ClientSession(local=True, is_loopback=True, authed=True, owner=True)


async def register_local_agent(rt, agent_id: str,
                               parent_id: Optional[str] = None):
    from atn.models import AgentDefinition, AgentMode
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5", parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


# The tool the seller sells: counts words, says which language its
# settings name (the daemon passes them as AUTONET_SERVICE_SETTINGS).
TOOL_CODE = r'''
import json, os, sys
args = json.load(sys.stdin)
text = str(args.get("text") or "")
if text == "fail":
    sys.exit(3)
settings = json.loads(os.environ.get("AUTONET_SERVICE_SETTINGS") or "{}")
print(json.dumps({"words": len(text.split()), "language": settings.get("language", "")}))
'''


async def amain() -> int:  # noqa: C901 — a linear E2E script, staged for reading
    import logging

    from web3 import Web3

    logging.basicConfig(level=logging.WARNING)
    for noisy in ("nodes.common.blob_store", "atn.tool_store",
                  "atn.service_store", "atn.runtime", "atn.runtime.agent_registry",
                  "atn.harness_distro", "websockets", "web3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    board = Board()
    hh_proc: Optional[subprocess.Popen] = None
    tmp_seller = Path(tempfile.mkdtemp(prefix="e2e_svc_seller_"))
    tmp_buyer = Path(tempfile.mkdtemp(prefix="e2e_svc_buyer_"))
    ws_server = None
    failed = False

    # ------------------------------------------------------------------
    # Stage 1: chain up
    # ------------------------------------------------------------------
    s1 = board.stage(1, "chain up (hardhat node + Substrate + ServiceRegistry "
                        "+ PaymentChannel)")
    addresses: Dict[str, Optional[str]] = {}
    try:
        hh_proc = start_hardhat(REPO)
        if not wait_rpc(45.0):
            raise RuntimeError("hardhat RPC never came up")
        addresses = deploy_contracts(REPO)
        s1.note("substrate", addresses["substrate"])
        step(f"chain up; Substrate at {addresses['substrate']}")
        s1.status = "PASS"
    except Exception as e:
        s1.status = "FAIL"
        s1.note("error", repr(e))
        log(traceback.format_exc())
        print(board.render())
        kill_hardhat(hh_proc)
        return 1

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    substrate_abi = load_abi("Substrate")
    substrate = w3.eth.contract(
        address=Web3.to_checksum_address(addresses["substrate"]), abi=substrate_abi)
    registry_c = w3.eth.contract(
        address=Web3.to_checksum_address(addresses["serviceRegistry"]),
        abi=load_abi("ServiceRegistry"))
    channel_c = w3.eth.contract(
        address=Web3.to_checksum_address(addresses["paymentChannel"]),
        abi=load_abi("PaymentChannel"))

    from eth_account import Account
    owner_seller = Account.from_key(HH_ACCT0_PK)
    owner_buyer = Account.from_key(HH_ACCT1_PK)
    shape = resolve_binding_shape(substrate_abi)

    def atn(addr: str) -> int:
        return int(substrate.functions.balanceOf(Web3.to_checksum_address(addr)).call())

    # ------------------------------------------------------------------
    # Stage 2: the seller daemon's operator and a buyer agent, on chain
    # ------------------------------------------------------------------
    s2 = board.stage(2, "seller's operator and a buyer agent on chain, buyer funded")
    rt_seller = rt_buyer = None
    seller_key = buyer_key = seller_addr = buyer_addr = ""
    try:
        rt_seller = make_runtime(tmp_seller, owner_wallet=owner_seller.address,
                                 owner_pk="", addresses=addresses)
        rt_buyer = make_runtime(tmp_buyer, owner_wallet=owner_buyer.address,
                                owner_pk=owner_buyer._private_key.hex(),
                                addresses=addresses)
        seller_key = rt_seller.registry.daemon_operator_key()
        seller_addr = Account.from_key(seller_key).address
        buyer_defn = await register_local_agent(rt_buyer, "buyer-1")
        buyer_key = rt_buyer.registry.get_agent_key("buyer-1")
        buyer_addr = Web3.to_checksum_address(buyer_defn.identity.address)
        for owner, addr in ((owner_seller, seller_addr), (owner_buyer, buyer_addr)):
            fund_gas(w3, owner, addr)
        op_tx = register_agent_on_chain(
            w3, substrate, shape, agent_pk=seller_key, agent_addr=seller_addr,
            owner_acct=owner_seller, peer_id=b"peer-seller")
        rt_seller.registry.set_operator_registered(True, op_tx)
        register_agent_on_chain(w3, substrate, shape, agent_pk=buyer_key,
                                agent_addr=buyer_addr, owner_acct=owner_buyer,
                                peer_id=b"peer-buyer")
        buyer_defn.identity.registered_on_chain = True
        send_tx(w3, substrate.functions.mintFromVault(buyer_addr, BUYER_FUNDING),
                HH_ACCT0_PK, owner_seller.address)
        assert atn(buyer_addr) == BUYER_FUNDING
        s2.note("seller_operator", seller_addr)
        s2.note("buyer", buyer_addr)
        step(f"seller operator {seller_addr[:10]} and buyer {buyer_addr[:10]} "
             f"registered; buyer holds {BUYER_FUNDING} ATN base units")
        s2.status = "PASS"
    except Exception as e:
        s2.status = "FAIL"
        s2.note("error", repr(e))
        log(traceback.format_exc())
        print(board.render())
        kill_hardhat(hh_proc)
        return 1

    # ------------------------------------------------------------------
    # Stage 3: the seller sells one of its tools as a request service
    # ------------------------------------------------------------------
    s3 = board.stage(3, "seller sells a tool as a request service; the buyer's "
                        "catalog has its spec and signed price list")
    spec_digest, service_id = "", 0
    try:
        from atn import service_catalog as sc
        from atn.on_chain import OnChainService
        from atn.ws_server import WebSocketBridge

        tool = rt_seller.tool_store.register(
            name="word_count", description="Counts the words in a text.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"]},
            author="user", code=TOOL_CODE)
        ws_server = WebSocketBridge(rt_seller, host="127.0.0.1", port=SELLER_WS_PORT)
        await ws_server.start()
        reg = await ws_server._handle_message({
            "type": "register_service", "mode": "request",
            "backing_tool": tool["digest"], "name": "count_words",
            "description": "Counts the words in a text.",
            "settings": {"language": "English"},
            "price": PRICE, "voucher": VOUCHER}, owner_session())
        if not reg.get("ok") or not reg["result"].get("on_chain"):
            raise RuntimeError(f"register_service failed: {reg}")
        spec_digest = reg["result"]["digest"]
        service_id = int(reg["result"]["service_id"])
        record = rt_seller.service_store.get(spec_digest)
        assert record.mode == "request" and record.prices == {"requests": PRICE}
        assert record.spec["input_schema"] == tool["manifest"]["input_schema"]
        assert "prices" not in record.spec and "unit" not in record.spec
        entry = registry_c.functions.getService(service_id).call()
        assert entry[0].lower() == seller_addr.lower() and entry[2] == 0, entry
        ep = await OnChainService(rt_seller._config.rpb).update_endpoint(
            seller_key, f"ws://127.0.0.1:{SELLER_WS_PORT}")
        if not ep.get("success"):
            raise RuntimeError(f"updateEndpoint failed: {ep}")
        # The buyer's catalog: the chain entry, then the announcement a
        # provider gossips (spec and price list).
        catalog = sc.catalog_for(rt_buyer)
        synced = await asyncio.to_thread(catalog.sync_chain)
        assert not synced.get("error"), synced
        assert catalog.accept_announcement(sc.announcement(
            record.spec, service_id, record.price_list))
        row = [r for r in catalog.search("count words") if r["digest"] == spec_digest][0]
        assert row["prices"] == {"requests": PRICE} and row["voucher"] == VOUCHER, row
        assert row["settings"] == {"language": "English"} and row["mode"] == "request"
        assert row["registry_price"] is None, row
        s3.note("service_id", service_id)
        step(f"service #{service_id} on chain with no registry price; the price "
             f"list ({PRICE} per request, vouchers of {VOUCHER}) reached the buyer")
        s3.status = "PASS"
    except Exception as e:
        s3.status = "FAIL"
        s3.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    block_before_payment = w3.eth.block_number
    from atn.agent_tools import (_close_credits, _open_credits, _request_service,
                                 _review_service, _top_up_credits)

    async def buy(text: str, **extra: Any) -> Dict[str, Any]:
        return await _request_service(rt_buyer, {
            "_caller_id": "buyer-1", "service_id": service_id,
            "payload": {"text": text}, **extra})

    # ------------------------------------------------------------------
    # Stage 4: a one-off request, paid directly; its replay is refused
    # ------------------------------------------------------------------
    s4 = board.stage(4, "one-off request paid directly (no credits); replay refused")
    try:
        if s3.status != "PASS":
            raise RuntimeError("stage 3 did not complete")
        from atn.service_client import sign_request
        seller_before = atn(seller_addr)
        reply = await buy("one two three")
        assert reply.get("ok"), reply
        assert reply["result"]["result"] == {"words": 3, "language": "English"}, reply
        assert reply["paid_directly"]["amount"] == PRICE
        assert atn(buyer_addr) == BUYER_FUNDING - PRICE
        fee = PRICE * 250 // 10_000
        assert abs(atn(seller_addr) - seller_before - (PRICE - fee)) <= 1, (
            atn(seller_addr), seller_before)
        rid = reply["request_id"]
        args = {"text": "one two three"}
        replay = await ws_server._handle_message({
            "type": "service_request", "spec_digest": spec_digest,
            "request_id": rid, "args": args,
            "tx_hash": reply["paid_directly"]["tx_hash"],
            "payer_signature": sign_request(buyer_key, rid, spec_digest, args, seller_addr),
        }, owner_session())
        assert not replay.get("ok") and "replay" in str(replay.get("error")), replay
        s4.note("seller_net", PRICE - fee)
        step(f"served with one payForService ({PRICE}, the seller got {PRICE - fee}); "
             "the same request id again is refused")
        s4.status = "PASS"
    except Exception as e:
        s4.status = "FAIL"
        s4.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 5: credits; vouchers pay ahead; a failed request costs nothing
    # ------------------------------------------------------------------
    s5 = board.stage(5, "credits: vouchers pay ahead by the voucher size; a failed "
                        "request costs nothing")
    channel_id = 0
    try:
        if s4.status != "PASS":
            raise RuntimeError("stage 4 did not complete")
        opened = await _open_credits(rt_buyer, {"_caller_id": "buyer-1",
                                                "provider_address": seller_addr,
                                                "deposit": DEPOSIT})
        assert opened.get("ok"), opened
        channel_id = int(opened["credits"]["channel_id"])
        again = await _open_credits(rt_buyer, {"_caller_id": "buyer-1",
                                               "provider_address": seller_addr,
                                               "deposit": DEPOSIT})
        assert "already open" in str(again.get("error")), again
        ledger = rt_seller.service_store

        r1 = await buy("a b")
        assert r1.get("ok"), r1
        assert ledger.balance(channel_id) == (PRICE, VOUCHER), ledger.balance(channel_id)
        r2 = await buy("a b c")
        assert r2.get("ok") and ledger.balance(channel_id) == (2 * PRICE, VOUCHER)
        bad = await buy("fail")
        assert not bad.get("ok"), bad
        assert ledger.balance(channel_id) == (2 * PRICE, VOUCHER), ledger.balance(channel_id)
        r3 = await buy("x")
        assert r3.get("ok") and ledger.balance(channel_id) == (3 * PRICE, VOUCHER)
        r4 = await buy("y")   # nothing paid ahead: a new voucher goes with it
        assert r4.get("ok") and ledger.balance(channel_id) == (4 * PRICE, 2 * VOUCHER)
        assert r4["credits"]["used"] == 4 * PRICE, r4["credits"]
        assert atn(buyer_addr) == BUYER_FUNDING - PRICE - DEPOSIT
        s5.note("channel", channel_id)
        s5.note("used_of_signed", f"{4 * PRICE}/{2 * VOUCHER}")
        step(f"credits {channel_id}: 4 requests used {4 * PRICE}, two vouchers "
             f"signed {2 * VOUCHER} ahead; the failed one was charged nothing")
        s5.status = "PASS"
    except Exception as e:
        s5.status = "FAIL"
        s5.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 6: a new price list: no transaction, no new spec
    # ------------------------------------------------------------------
    s6 = board.stage(6, "the seller changes the price: a new signed list, no "
                        "transaction, no new spec")
    try:
        if s5.status != "PASS":
            raise RuntimeError("stage 5 did not complete")
        nonce = w3.eth.get_transaction_count(Web3.to_checksum_address(seller_addr))
        changed = await ws_server._handle_message({
            "type": "set_service_prices", "digest": spec_digest, "price": NEW_PRICE},
            owner_session())
        assert changed.get("ok"), changed
        record = rt_seller.service_store.get(spec_digest)
        assert record.price_list["seq"] == 2 and record.prices == {"requests": NEW_PRICE}
        assert w3.eth.get_transaction_count(Web3.to_checksum_address(seller_addr)) == nonce
        assert sc.catalog_for(rt_buyer).accept_announcement(
            sc.announcement(record.spec, service_id, record.price_list))
        assert sc.catalog_for(rt_buyer).price_list(spec_digest)["seq"] == 2
        before = rt_seller.service_store.balance(channel_id)[0]
        r = await buy("priced anew")
        assert r.get("ok"), r
        assert rt_seller.service_store.balance(channel_id)[0] - before == NEW_PRICE
        step(f"price {PRICE} -> {NEW_PRICE} with no transaction; the next request "
             f"used {NEW_PRICE}")
        s6.status = "PASS"
    except Exception as e:
        s6.status = "FAIL"
        s6.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 7: the seller settles without closing; the fee burns
    # ------------------------------------------------------------------
    s7 = board.stage(7, "the seller cashes in without closing (settle, ChannelPaid)")
    try:
        if s6.status != "PASS":
            raise RuntimeError("stage 6 did not complete")
        from atn.channel_settlement import settle_channels
        from atn.on_chain import ServiceMarketClient
        smc_s = ServiceMarketClient(rt_seller._config.rpb)
        cumulative = rt_seller.service_store.balance(channel_id)[1]
        seller_before = atn(seller_addr)
        burned_before = int(substrate.functions.totalBurned().call())
        out = await settle_channels(rt_seller.service_store, smc_s,
                                    lambda p: seller_key, settle_min=1)
        assert any(o.get("action") == "settled" and o.get("ok") for o in out), out
        state = channel_c.functions.getChannel(channel_id).call()
        assert int(state[5]) == 1, state            # still Open
        assert int(state[3]) == cumulative, (state, cumulative)
        fee = cumulative * 250 // 10_000
        assert abs(atn(seller_addr) - seller_before - (cumulative - fee)) <= 1
        assert int(substrate.functions.totalBurned().call()) > burned_before, "no fee burned"
        pays = await smc_s.channel_payments(0)
        assert any(int(p["channel_id"]) == channel_id
                   and p["client"].lower() == buyer_addr.lower() for p in pays), pays
        r = await buy("still open")
        assert r.get("ok"), r
        step(f"settled {cumulative} (seller net about {cumulative - fee}, the fee partly burned); "
             "the credits stayed open and kept serving")
        s7.status = "PASS"
    except Exception as e:
        s7.status = "FAIL"
        s7.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 8: top-up, then credits running out
    # ------------------------------------------------------------------
    s8 = board.stage(8, "top-up; credits run out with an error the Agent sees")
    try:
        if s7.status != "PASS":
            raise RuntimeError("stage 7 did not complete")
        up = await _top_up_credits(rt_buyer, {"_caller_id": "buyer-1",
                                              "provider_address": seller_addr,
                                              "amount": TOP_UP})
        assert up.get("ok") and int(up["deposit"]) == DEPOSIT + TOP_UP, up
        state = channel_c.functions.getChannel(channel_id).call()
        assert int(state[2]) == DEPOSIT + TOP_UP, state
        spent = None
        for _ in range(40):
            r = await buy("more")
            if not r.get("ok"):
                spent = r
                break
        assert spent is not None and "top them up" in str(spent.get("error")), spent
        used = rt_seller.service_store.balance(channel_id)[0]
        assert used <= DEPOSIT + TOP_UP, used
        step(f"topped up to {DEPOSIT + TOP_UP}; requests ran until the credits were "
             f"spent ({used} used), then: {spent['error'][:60]}")
        s8.status = "PASS"
    except Exception as e:
        s8.status = "FAIL"
        s8.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 9: the buyer closes; the seller settles inside the window; the
    # remainder comes back
    # ------------------------------------------------------------------
    s9 = board.stage(9, "the buyer closes; the seller settles in the window; the "
                        "remainder comes back")
    try:
        if s8.status != "PASS":
            raise RuntimeError("stage 8 did not complete")
        from atn.channel_settlement import settle_channels
        from atn.on_chain import ServiceMarketClient
        from atn.service_channels import sweep_refunds
        closed = await _close_credits(rt_buyer, {"_caller_id": "buyer-1",
                                                 "provider_address": seller_addr})
        assert closed.get("ok"), closed
        closes_at = int(channel_c.functions.getChannel(channel_id).call()[4])
        served = rt_seller.service_store.balance(channel_id)[0]
        out = await settle_channels(rt_seller.service_store,
                                    ServiceMarketClient(rt_seller._config.rpb),
                                    lambda p: seller_key, settle_min=1)
        state = channel_c.functions.getChannel(channel_id).call()
        assert int(state[4]) == closes_at, "the settle moved the buyer's deadline"
        claimed = int(state[3])
        deposit = int(state[2])
        assert claimed == served, (claimed, served)   # never what was only paid ahead
        while int(w3.eth.get_block("latest")["timestamp"]) <= closes_at:
            time.sleep(1)
            send_tx(w3, substrate.functions.mintFromVault(owner_seller.address, 1),
                    HH_ACCT0_PK, owner_seller.address)
        before = atn(buyer_addr)
        # Hardhat's block clock runs ahead of the wall clock: the sweep
        # judges the window by the chain's time.
        swept = await sweep_refunds(rt_buyer.channel_book,
                                    ServiceMarketClient(rt_buyer._config.rpb),
                                    lambda p: buyer_key,
                                    now=float(w3.eth.get_block("latest")["timestamp"]))
        assert any(o.get("action") == "withdrew" for o in swept), swept
        assert atn(buyer_addr) - before == deposit - claimed
        step(f"the seller collected what it served ({claimed}) inside the window "
             f"without moving the deadline; the buyer got {deposit - claimed} back")
        s9.status = "PASS"
    except Exception as e:
        s9.status = "FAIL"
        s9.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 10: a review counts once the reviewer has paid the provider
    # ------------------------------------------------------------------
    s10 = board.stage(10, "the buyer reviews the service; the close counts it once "
                          "the buyer has paid the provider; search shows the rating")
    try:
        if s3.status != "PASS":
            raise RuntimeError("stage 3 did not complete")
        from unittest.mock import MagicMock

        from nodes.common.authoritative_encoding import decode_carry_bundle
        from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
        from nodes.common.federated_close_driver import FederatedCloseDriver
        from nodes.common.service_reviews import service_key
        from nodes.common.voice_state import read_service_snapshot

        jurisdiction = "0xe2e-service-reviews"
        sent: List[Dict[str, Any]] = []
        rt_buyer.service_reviews.attach(jurisdiction, sent.append)
        review = await _review_service(rt_buyer, {
            "_caller_id": "buyer-1", "service_id": service_id,
            "correctness": 0.9, "simplicity": 0.7, "novelty": 0.2, "malware": False,
            "note": "counted right"})
        assert review.get("ok"), review
        assert len(sent) == 1 and sent[0]["reviewer"] == buyer_addr.lower()

        def pinned(block: int) -> Dict[str, Any]:
            return read_service_snapshot(
                w3, block=block, substrate_address=addresses["substrate"],
                service_registry_address=addresses["serviceRegistry"],
                payment_channel_address=addresses["paymentChannel"])

        kp = Keypair.generate()
        batch = EventBatch(rpb_address=jurisdiction, sender_pubkey=kp.public_key,
                           batch_seq=1, events=list(sent), prev_batch_hash=b"",
                           timestamp=time.time())
        gossip = MagicMock(spec=EventGossip)
        gossip.known_senders.return_value = [kp.public_key]
        gossip.sender_pubkey = kp.public_key
        gossip.drain_epoch_batches.return_value = [batch]
        close_dir = Path(tempfile.mkdtemp(prefix="e2e_svc_close_"))
        drv = FederatedCloseDriver(
            gossip=gossip,
            tool_registrations_path=close_dir / "tool_registrations.json",
            tool_rounds_path=close_dir / "tool_rounds.json",
            tool_ratings_path=close_dir / "tool_ratings.json",
            service_reviews_path=close_dir / "service_reviews.json",
            service_ratings_path=close_dir / "service_ratings.json")
        key = service_key(seller_addr, spec_digest)
        drv.service_snapshot = pinned(block_before_payment)
        waiting = drv.run({"epoch_id": "e2e-service-reviews-0"})
        assert waiting is not None
        assert key not in waiting.close_result["service_ratings"]
        assert waiting.close_result["service_reviews"][key][buyer_addr.lower()]["paid"] is False

        drv.service_snapshot = pinned(w3.eth.block_number)
        gossip.drain_epoch_batches.return_value = [batch]
        fed = drv.run({"epoch_id": "e2e-service-reviews-1"})
        carry = decode_carry_bundle(fed.carry_bundle_blob)["carry"]
        rating = carry["service_ratings"].get(key)
        assert rating and rating["payers"] == 1 and rating["correctness"] == 0.9, rating
        assert rating["paid"] >= PRICE, rating
        catalog = sc.catalog_for(rt_buyer)
        catalog.rating_provider = drv.service_rating
        rows = [r for r in catalog.search("") if r["digest"] == spec_digest]
        assert rows and rows[0]["rating"]["payers"] == 1, rows
        s10.note("rating", {k: rating[k] for k in ("payers", "correctness", "paid")})
        step(f"the review waited before any payment and counts after it "
             f"(paid {rating['paid']} to the provider); search shows it")
        s10.status = "PASS"
    except Exception as e:
        s10.status = "FAIL"
        s10.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 11: teardown
    # ------------------------------------------------------------------
    s11 = board.stage(11, "teardown")
    try:
        if ws_server is not None:
            await ws_server.stop()
        for rt in (rt_seller, rt_buyer):
            if rt is not None:
                try:
                    await rt.stop()
                except Exception:
                    pass
        kill_hardhat(hh_proc)
        hh_proc = None
        s11.status = "PASS"
    except Exception as e:
        s11.status = "FAIL"
        s11.note("error", repr(e))
        log(traceback.format_exc())
        failed = True
    finally:
        kill_hardhat(hh_proc)

    print(board.render())
    any_fail = any(s.status == "FAIL" for s in board.stages.values())
    return 1 if (failed or any_fail) else 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
