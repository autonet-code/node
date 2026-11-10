#!/usr/bin/env python3
"""End-to-end proof of the per-agent marketplace inference binding.

Ratified 2026-07-26 (docs/services_market.md, "Decision (2026-07-26): LLM
inference as a marketplace service", plus the employer-chooses-the-tool
ruling): an agent's MODEL/PROVIDER is set only by its PARENT — an agent must
never switch its own substrate, and there is no self-switching surface at all.
But a parent MAY acquire a marketplace inference service and provision a CHILD
bound to it as its provider, then scrutinize the child's output from OUTSIDE
the purchased substrate. The child pays each call from its OWN wallet: spend
authority is literal token custody, so the parent funds that wallet on-chain
and controls the tap by controlling the refills.

This script proves the whole rail on a real local deployment:

    owner registers an inference-backed service (WS surface, real spec + real
    on-chain ServiceRegistry entry, real published ws endpoint), under the
    selling daemon's operator: a normal install, no configured daemon key,
    the vault-held operator key signs and is paid
      -> parent agent, on-chain, with a funded child
      -> parent binds the CHILD to that service (update_agent, parent-only)
      -> the child does an ordinary chat completion through its provider
      -> a payment channel from the CHILD'S wallet, one voucher per call
         (one call is one unit of the service, Decision 2026-11-10)
      -> provider-side gate verifies the voucher on chain
      -> the canned completion round-trips back into a ProviderResponse

The channel carries every later stage, through the daemons' own code: the
running total grows by one price per call, a price changed on chain applies
to the next voucher, a call whose dispatch fails is not charged on either
side (through the provider and through the request_service agent tool, which
pays on the same channel), the seller's settlement sweep closes the channel
with its best voucher once the deposit no longer covers the price, and the
buyer's sweep takes the remainder back after the challenge window.

Then the review (stage 12, Decision 2026-11-10): the buyer's daemon kept
the seller's signed receipt of the served request, the child reviews it
with the review_service agent tool, and the event rides a batch into a
real federated close pinned to the chain (read_service_snapshot). Pinned
before the seller's channel close, the review waits for its payment;
pinned after it, the settlement covers the review, the close's carry
bundle (the anchored state) holds the service's rating, and the buyer's
catalog search shows it.

The LLM seam is the ONE thing mocked: the selling daemon's provider manager
builds a canned-response fake for the upstream provider its service names,
so this needs no real model.
Everything below it — chain, payment, gate, replay guard, websocket transport,
provider resolution, key selection — is the production path.

Run:
    python scripts/local_e2e_service_provider.py

Modeled on scripts/local_e2e_tool_economy.py (same hardhat harness, same
stage-by-stage scoreboard). See docs/local_e2e.md.
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
# `npx hardhat node` serves the in-process "hardhat" network (chainId 1337,
# see hardhat.config.js) — NOT 31337. Signing for the wrong id gives
# "incompatible EIP-155 transaction, signed for another chain".
CHAIN_ID = 1337

# Hardhat's deterministic mnemonic accounts (well-known, safe: local only).
# account[0] = the OWNER (deployer, seller's human wallet)
# account[1] = the PARENT's human wallet (the employer funding the child)
HH_ACCT0_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
HH_ACCT1_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"

# The service's ask, in ATN base units. Small enough that the seller's
# genesis-free balance is irrelevant: the child is funded explicitly.
SERVICE_ASK = 1000
CHILD_FUNDING = 20_000        # ATN the parent's owner grants the child's wallet
SELLER_WS_PORT = 7791         # the selling daemon's local WS listener
CANNED_REPLY = "The bound child thinks on purchased cognition."
# A message the canned provider fails on: the seller's dispatch errors.
FAIL_PROMPT = "please fail this one"
# PaymentChannel challenge window (seconds): short, so the refund stage
# waits it out in real time (hardhat stamps each block with the wall clock).
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
                 "  SERVICE-BACKED PROVIDER LOCAL E2E — SCOREBOARD", "=" * 74]
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


class CannedProvider:
    """The mocked LLM seam — the ONLY fake in this script.

    Stands in for the upstream provider the SELLING daemon builds for its
    service (`WebSocketBridge._inference_upstream`, through the provider
    manager). Everything on the buying side is real, so the child cannot
    tell this from a GPU.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "canned"

    async def send(self, *, messages, system="", model="", max_tokens=4096,
                   tools=None, temperature=0.0):
        from atn.providers.base import ProviderResponse, Usage
        self.calls.append({
            "messages": messages, "system": system, "model": model,
            "max_tokens": max_tokens,
        })
        if any(m.get("content") == FAIL_PROMPT for m in messages):
            # The seller's upstream fails: its dispatch returns an error.
            raise RuntimeError("canned upstream failure")
        return ProviderResponse(
            text=CANNED_REPLY,
            model=model or "canned-model-1",
            stop_reason="end_turn",
            usage=Usage(input_tokens=17, output_tokens=9),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> int:  # noqa: C901 — a linear E2E script, staged for reading
    import logging

    from web3 import Web3

    # Two real Runtimes boot ~40 pinned tools each and log every blob; that
    # buries the scoreboard. Keep WARNING and above (the payment-gate and
    # payment-skipped warnings are exactly what a reader wants to see).
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
    # Stage 1 — chain up
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
        s1.note("serviceRegistry", addresses["serviceRegistry"])
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
        address=Web3.to_checksum_address(addresses["substrate"]),
        abi=substrate_abi)
    registry_c = w3.eth.contract(
        address=Web3.to_checksum_address(addresses["serviceRegistry"]),
        abi=load_abi("ServiceRegistry"))

    from eth_account import Account
    owner_seller = Account.from_key(HH_ACCT0_PK)   # the seller's human wallet
    owner_buyer = Account.from_key(HH_ACCT1_PK)    # the employer's human wallet
    shape = resolve_binding_shape(substrate_abi)
    log(f"owner-binding shape: {shape}")

    # ------------------------------------------------------------------
    # Stage 2 — two daemons, three agents, on-chain identities
    # ------------------------------------------------------------------
    s2 = board.stage(2, "fleet (seller daemon's operator + buyer agents on chain)")
    rt_seller = rt_buyer = None
    parent_defn = child_defn = None
    seller_key = parent_key = child_key = ""
    seller_addr = parent_addr = child_addr = ""
    try:
        if shape["register"] is None or shape["hash_view"] is None:
            raise RuntimeError(
                f"contract exposes no owner-binding surface (shape={shape})")

        # The seller daemon is a normal install: no configured daemon key.
        # It sells as its operator, the vault-held key the owner surface and
        # every external agent act as (Decision 2026-11-08).
        rt_seller = make_runtime(
            tmp_seller, owner_wallet=owner_seller.address,
            owner_pk="", addresses=addresses)
        rt_buyer = make_runtime(
            tmp_buyer, owner_wallet=owner_buyer.address,
            owner_pk=owner_buyer._private_key.hex(), addresses=addresses)
        assert not rt_seller._config.autonet.private_key
        seller_key = rt_seller.registry.daemon_operator_key()
        seller_addr = Account.from_key(seller_key).address

        # Buyer daemon: the PARENT (employer) and its CHILD. parent_id is what
        # makes the binding a parent act rather than a self-act.
        parent_defn = await register_local_agent(rt_buyer, "parent-1")
        child_defn = await register_local_agent(rt_buyer, "child-1",
                                                parent_id="parent-1")

        parent_key = rt_buyer.registry.get_agent_key("parent-1")
        child_key = rt_buyer.registry.get_agent_key("child-1")
        parent_addr = Web3.to_checksum_address(parent_defn.identity.address)
        child_addr = Web3.to_checksum_address(child_defn.identity.address)
        assert child_key and child_key != parent_key, (
            "the child must hold its OWN distinct wallet key")
        assert seller_key not in (parent_key, child_key)

        # Gas for every key that signs (registration, service ops, payment).
        for owner, addr in ((owner_seller, seller_addr),
                            (owner_buyer, parent_addr),
                            (owner_buyer, child_addr)):
            fund_gas(w3, owner, addr)

        op_tx = register_agent_on_chain(
            w3, substrate, shape, agent_pk=seller_key, agent_addr=seller_addr,
            owner_acct=owner_seller, peer_id=b"peer-seller")
        rt_seller.registry.set_operator_registered(True, op_tx)
        for defn, pk, addr, owner, peer in (
            (parent_defn, parent_key, parent_addr, owner_buyer, b"peer-parent"),
            (child_defn, child_key, child_addr, owner_buyer, b"peer-child"),
        ):
            register_agent_on_chain(
                w3, substrate, shape, agent_pk=pk, agent_addr=addr,
                owner_acct=owner, peer_id=peer)
            defn.identity.registered_on_chain = True

        assert substrate.functions.getAgentOwner(seller_addr).call() == \
            owner_seller.address
        assert substrate.functions.getAgentOwner(child_addr).call() == \
            owner_buyer.address
        s2.note("seller_operator", seller_addr)
        s2.note("parent", parent_addr)
        s2.note("child", child_addr)
        s2.note("child_key_distinct", True)
        step("seller's operator (vault key) and two buyer agents registered "
             "on chain; child holds its own key")
        s2.status = "PASS"
    except Exception as e:
        s2.status = "FAIL"
        s2.note("error", repr(e))
        log(traceback.format_exc())
        print(board.render())
        kill_hardhat(hh_proc)
        return 1

    # ------------------------------------------------------------------
    # Stage 3 — the seller lists inference for sale (WS surface + chain)
    # ------------------------------------------------------------------
    s3 = board.stage(3, "seller lists an inference-backed service")
    spec_digest = ""
    try:
        from atn.ws_server import WebSocketBridge

        # Mock ONLY the LLM seam on the selling daemon. Everything the buyer
        # touches stays production code.
        canned = CannedProvider()
        real_resolve = rt_seller.providers.resolve_provider_with_fallback
        rt_seller.providers.resolve_provider_with_fallback = (
            lambda defn: canned if str(defn.id).startswith("service-")
            else real_resolve(defn))

        ws_server = WebSocketBridge(rt_seller, host="127.0.0.1",
                                    port=SELLER_WS_PORT)
        await ws_server.start()

        # Register the service through the real WS handler (owner surface):
        # an `inference` block instead of a backing tool. The owner's
        # services are registered, signed and paid under the operator.
        reg = await ws_server._handle_message(
            {
                "type": "register_service",
                "name": "canned-cognition",
                "description": "Chat completions off the seller's own stack.",
                "input_schema": {"type": "object",
                                 "properties": {"messages": {"type": "array"}}},
                "ask": {"amount": str(SERVICE_ASK), "unit": "per_item"},
                # What one work item is: the meaning of the on-chain price.
                "unit": "one completion up to 512 output tokens",
                "inference": {"provider": "ollama",
                              "model": "canned-model-1",
                              "max_tokens_cap": 512},
            },
            owner_session())
        if not reg.get("ok") or not reg["result"].get("on_chain"):
            raise RuntimeError(f"register_service failed: {reg}")
        spec_digest = reg["result"]["digest"]
        record = rt_seller.service_store.get(spec_digest)
        assert record.inference, "spec carries no inference backing"
        # The spec names its signer, the address it is registered under:
        # the operator's.
        assert record.spec["author_pubkey"].lower() == seller_addr.lower(), (
            record.spec["author_pubkey"], seller_addr)

        # The chain leg ran through the daemon, signed by the operator key:
        # the entry's provider is the payee the gate checks.
        service_id = reg["result"]["service_id"]
        assert record.service_id == service_id
        entry = registry_c.functions.getService(service_id).call()
        assert entry[0].lower() == seller_addr.lower() and entry[2] == SERVICE_ASK

        # Publish the seller's reachable ws endpoint on chain. The buyer's
        # provider resolves the counterparty this way — no hardcoded address
        # anywhere on the buying side.
        from atn.on_chain import OnChainService
        oc_seller = OnChainService(rt_seller._config.rpb)
        ep = await oc_seller.update_endpoint(
            seller_key, f"ws://127.0.0.1:{SELLER_WS_PORT}")
        if not ep.get("success"):
            raise RuntimeError(f"updateEndpoint failed: {ep}")
        assert await oc_seller.get_agent_endpoint(seller_addr) == \
            f"ws://127.0.0.1:{SELLER_WS_PORT}"

        s3.note("spec_digest", spec_digest[:16])
        s3.note("service_id", service_id)
        s3.note("ask", SERVICE_ASK)
        s3.note("endpoint_on_chain", f"ws://127.0.0.1:{SELLER_WS_PORT}")
        step(f"inference service listed on chain (id={service_id}, "
             f"ask={SERVICE_ASK})")
        s3.status = "PASS"
    except Exception as e:
        s3.status = "FAIL"
        s3.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 4 — the parent funds the child's wallet, then binds it
    # ------------------------------------------------------------------
    s4 = board.stage(4, "parent funds the child's wallet + binds it (parent-only)")
    binding = {"provider_address": seller_addr, "spec_digest": spec_digest}
    try:
        from atn.agent_tools import execute_tool

        s4.note("child_atn_before",
                substrate.functions.balanceOf(child_addr).call())

        # PARENT-ONLY ENFORCEMENT — the ruling under test.
        # (a) the child may NOT bind itself.
        self_attempt = await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(binding)},
            rt_buyer, caller_id="child-1")
        assert "error" in self_attempt, (
            "a child was allowed to set its OWN binding — the whole ruling is "
            f"that it cannot: {self_attempt}")
        assert rt_buyer.get_agent("child-1").service_provider is None
        s4.note("child_self_bind_refused", True)
        step("child CANNOT bind itself (no self-switching surface)")

        # (b) a non-parent may not bind it either.
        stranger = await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(binding)},
            rt_buyer, caller_id="seller-1")
        assert "error" in stranger, stranger
        s4.note("stranger_bind_refused", True)

        # (c) the PARENT may.
        bound = await execute_tool(
            "update_agent",
            {"agent_id": "child-1", "service_provider": dict(binding)},
            rt_buyer, caller_id="parent-1")
        if "error" in bound:
            raise RuntimeError(f"parent could not bind its child: {bound}")
        assert "service_provider" in bound["changed"], bound
        stored = rt_buyer.get_agent("child-1").service_provider
        assert stored == {"provider_address": seller_addr,
                          "spec_digest": spec_digest}, stored
        step("parent bound the child to the purchased service")

        # It survives a restart: the binding decides whose wallet pays, so
        # losing it on save would silently re-route the spend to the owner.
        import yaml
        raw = yaml.safe_load(
            (tmp_buyer / "agents" / "child-1" / "agent.yaml").read_text(
                encoding="utf-8"))
        assert raw["service_provider"]["spec_digest"] == spec_digest, raw
        s4.note("persisted_to_yaml", True)

        # And it is visible to the parent scrutinizing from outside.
        info = await execute_tool("get_agent", {"agent_id": "child-1"},
                                  rt_buyer, caller_id="parent-1")
        assert info["service_provider"]["provider_address"] == seller_addr
        s4.note("visible_in_get_agent", True)
        step("binding persisted to YAML and visible via get_agent")
        s4.status = "PASS"
    except Exception as e:
        s4.status = "FAIL"
        s4.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 5 — provider resolution: the child's own key, from the binding
    # ------------------------------------------------------------------
    s5 = board.stage(5, "provider resolution (binding -> child's own key)")
    provider = None
    try:
        from atn.providers.service import ServiceProvider

        if failed:
            raise RuntimeError(
                "an earlier stage failed; refusing to resolve a provider "
                "(an unbound agent would fall through to a REAL local "
                "provider and burn subscription tokens)")

        provider = rt_buyer.providers.resolve_provider_with_fallback(
            rt_buyer.get_agent("child-1"))
        assert isinstance(provider, ServiceProvider), type(provider)
        assert provider.provider_address == seller_addr
        assert provider.spec_digest == spec_digest
        # THE doctrine assertion: the child pays with ITS OWN key, never the
        # daemon owner's.
        assert provider._owner_key == child_key, "child key not threaded through"
        assert provider._owner_key != rt_buyer._config.autonet.private_key, (
            "the child is paying with the daemon OWNER's key")
        assert provider.payer_kind == "agent"
        assert provider.payer_id == "child-1"
        assert provider._client_address.lower() == child_addr.lower()

        # The unbound parent is untouched by any of this.
        parent_provider = None
        try:
            parent_provider = rt_buyer.providers.resolve_provider_with_fallback(
                rt_buyer.get_agent("parent-1"))
        except Exception:
            pass  # no local model configured — fine, it just isn't a ServiceProvider
        assert not isinstance(parent_provider, ServiceProvider), (
            "an UNBOUND agent resolved to a marketplace provider")

        s5.note("payer_kind", provider.payer_kind)
        s5.note("payer_id", provider.payer_id)
        s5.note("signs_with_child_key", True)
        s5.note("unbound_parent_unaffected", True)
        step("bound child resolves to ServiceProvider signing with its OWN key")
        s5.status = "PASS"
    except Exception as e:
        s5.status = "FAIL"
        s5.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stages 6-11 — the payment channel (Decision 2026-11-10, metered on
    # channels), through the daemons' own code: the child's `service`
    # provider and the request_service agent tool pay on ONE channel from
    # the child's wallet, a voucher per call (one call is one unit); the
    # seller's gate keeps a voucher only for a served call; the seller's
    # sweep closes the channel to collect, and the buyer's sweep takes the
    # remainder back after the challenge window.
    # ------------------------------------------------------------------
    channel_c = w3.eth.contract(
        address=Web3.to_checksum_address(addresses["paymentChannel"]),
        abi=load_abi("PaymentChannel"))
    seller_store = rt_seller.service_store if rt_seller is not None else None
    book = rt_buyer.channel_book if rt_buyer is not None else None
    channel_id = None

    def held() -> tuple:
        """(buyer's book, seller's highest accepted, seller's kept voucher)."""
        ch = book.get(child_addr, seller_addr) or {}
        kept = seller_store.get_channel_voucher(channel_id) or {}
        return (int(ch.get("cumulative", -1)),
                seller_store.last_channel_cumulative(channel_id),
                int(kept.get("cumulative", 0)))

    s6 = board.stage(6, "paid completion on the payment channel (child's wallet)")
    try:
        if provider is None:
            raise RuntimeError("no ServiceProvider from stage 5")
        # Fund the child's wallet. A training mint needs a governor anchor
        # (2026-10-16) and a fee-backed pool, so local float comes
        # from the purchased-money rail: the deploy wires vaultMinter to the
        # deployer (hardhat account 0).
        send_tx(w3, substrate.functions.mintFromVault(
            child_addr, CHILD_FUNDING), HH_ACCT0_PK, owner_seller.address)
        child_atn = substrate.functions.balanceOf(child_addr).call()
        assert child_atn >= SERVICE_ASK * 10, (
            f"child wallet only holds {child_atn}, needs >= {SERVICE_ASK * 10}")
        s6.note("child_atn_funded", child_atn)
        step(f"child's wallet funded with {child_atn} ATN")

        seller_before = substrate.functions.balanceOf(seller_addr).call()
        child_before = substrate.functions.balanceOf(child_addr).call()
        owner_before = substrate.functions.balanceOf(
            rt_buyer._config.autonet.owner_wallet).call()

        # THE CALL. An ordinary chat completion from the child's seat.
        resp = await provider.send(
            messages=[{"role": "user", "content": "who pays for this thought?"}],
            system="be brief", max_tokens=128)

        assert resp.text == CANNED_REPLY, resp.text
        assert resp.usage.input_tokens == 17
        assert resp.usage.output_tokens == 9
        step(f"completion round-tripped: {resp.text!r}")

        # The seller's daemon actually dispatched to its provider stack, with
        # the buyer's messages and the CLAMPED token cap.
        assert canned.calls, "the seller never dispatched to its provider"
        assert canned.calls[0]["messages"][0]["content"] == \
            "who pays for this thought?"
        assert canned.calls[0]["max_tokens"] == 128
        assert canned.calls[0]["system"] == "be brief"
        s6.note("seller_dispatched", True)

        # --- One channel, from the child's wallet, a voucher for one unit -
        paid = provider.last_payment
        assert paid.get("paid") == SERVICE_ASK, paid
        assert paid["cumulative_amount"] == SERVICE_ASK, paid
        channel_id = int(paid["channel_id"])
        client, payee, deposit, claimed, _closes, status = \
            channel_c.functions.getChannel(channel_id).call()[:6]
        assert client.lower() == child_addr.lower(), (client, child_addr)
        assert payee.lower() == seller_addr.lower(), (payee, seller_addr)
        assert deposit == SERVICE_ASK * 10 and status == 1 and claimed == 0
        assert held() == (SERVICE_ASK, SERVICE_ASK, SERVICE_ASK), held()
        s6.note("channel_id", channel_id)
        s6.note("deposit", deposit)
        step(f"the child's provider opened channel {channel_id} (deposit "
             f"{deposit}) and signed {SERVICE_ASK} for the call; the seller "
             "kept that voucher")

        # --- Money: escrowed from the RIGHT wallet, paid at settlement ----
        seller_after = substrate.functions.balanceOf(seller_addr).call()
        child_after = substrate.functions.balanceOf(child_addr).call()
        owner_after = substrate.functions.balanceOf(
            rt_buyer._config.autonet.owner_wallet).call()
        assert child_before - child_after == deposit, (
            f"child moved {child_before - child_after}, expected the deposit "
            f"{deposit}")
        assert seller_after == seller_before, (
            "the seller was paid before closing the channel")
        assert owner_after == owner_before, (
            "the daemon OWNER's wallet moved — the child must pay for itself")
        pay_logs = substrate.events.ServicePayment().get_logs(from_block=0)
        assert not pay_logs, (
            f"a direct payForService was sent instead of a voucher: {pay_logs}")
        s6.note("child_escrowed", child_before - child_after)
        s6.note("owner_wallet_untouched", True)
        s6.note("direct_payments", 0)
        step(f"child escrowed {deposit} once; no per-call transaction; "
             "owner wallet untouched")

        # --- The gate really VERIFIED, it did not degrade open ----------
        # A degraded gate keeps no voucher and consumes no request id.
        assert len(seller_store._seen_requests) == 1, (
            "the gate consumed no request id: it degraded open instead of "
            "verifying the voucher on chain")
        s6.note("gate_verified_not_degraded", True)
        step("provider gate verified the voucher on chain (not degraded)")
        s6.status = "PASS"
    except Exception as e:
        s6.status = "FAIL"
        s6.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 7 — the running total grows by one price per call; a replayed
    # request or a reused voucher is refused
    # ------------------------------------------------------------------
    s7 = board.stage(7, "replay guard (fresh request_id, one price per voucher)")
    try:
        if provider is None or s6.status != "PASS":
            raise RuntimeError("stage 6 did not complete a paid call")
        from atn.on_chain import ServiceMarketClient
        from atn.service_client import sign_request

        seen_before = set(seller_store._seen_requests)
        resp2 = await provider.send(
            messages=[{"role": "user", "content": "and again?"}],
            max_tokens=64)
        assert resp2.text == CANNED_REPLY
        paid = provider.last_payment
        assert paid["channel_id"] == channel_id, paid
        assert paid["cumulative_amount"] == 2 * SERVICE_ASK, paid
        assert held() == (2 * SERVICE_ASK,) * 3, held()
        seen_after = set(seller_store._seen_requests)
        assert len(seen_after) == len(seen_before) + 1, (
            f"the gate consumed {len(seen_after) - len(seen_before)} request "
            f"ids on the second call, expected exactly 1")
        s7.note("running_total", 2 * SERVICE_ASK)
        s7.note("request_ids_distinct", True)
        step(f"second call: same channel, running total {2 * SERVICE_ASK}, "
             "a fresh request id")

        # A REPLAYED request id is refused, even with a fresh voucher on top
        # of it; the last voucher on a NEW request id pays nothing more.
        smc_b = ServiceMarketClient(rt_buyer._config.rpb)
        first_rid = "0x" + sorted(seen_before)[0]
        args = {"messages": [{"role": "user", "content": "replay"}]}

        async def resend(rid: str, cumulative: int) -> Dict[str, Any]:
            return await ws_server._handle_message({
                "type": "service_request", "spec_digest": spec_digest,
                "request_id": rid, "args": args, "client": child_addr,
                "channel_id": channel_id, "cumulative_amount": cumulative,
                "signature": smc_b.sign_voucher(child_key, channel_id, cumulative),
                "payer_signature": sign_request(child_key, rid, spec_digest,
                                                args, seller_addr),
            }, owner_session())

        replay = await resend(first_rid, 3 * SERVICE_ASK)
        assert not replay.get("ok"), (
            f"the gate ACCEPTED a replayed request_id: {replay}")
        assert "replay" in str(replay.get("error", "")).lower(), replay
        from atn.service_client import new_request_id
        reused = await resend(new_request_id(), 2 * SERVICE_ASK)
        assert not reused.get("ok"), (
            f"the gate accepted a voucher that adds nothing: {reused}")
        assert "increment" in str(reused.get("error", "")).lower(), reused
        assert held() == (2 * SERVICE_ASK,) * 3, held()
        s7.note("replay_refused", True)
        s7.note("reused_voucher_refused", True)
        step("gate refuses a replayed request id and a voucher that adds "
             "no price")
        s7.status = "PASS"
    except Exception as e:
        s7.status = "FAIL"
        s7.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 8 — the chain is the only reference for the price (Decision
    # 2026-11-10): updateServiceAsk applies to the next voucher
    # ------------------------------------------------------------------
    s8 = board.stage(8, "price changed on chain applies to the next call")
    new_ask = SERVICE_ASK * 3
    try:
        if provider is None or s7.status != "PASS":
            raise RuntimeError("stage 7 did not complete")
        record = rt_seller.service_store.get(spec_digest)
        sid = record.service_id
        assert sid is not None, "the seller's daemon kept no service number"
        send_tx(w3, registry_c.functions.updateServiceAsk(sid, new_ask),
                seller_key, seller_addr)
        assert registry_c.functions.getService(sid).call()[2] == new_ask
        assert "ask" not in record.spec, "the spec still names a price"

        seen_before = set(seller_store._seen_requests)
        resp3 = await provider.send(
            messages=[{"role": "user", "content": "at the new price?"}],
            max_tokens=64)
        assert resp3.text == CANNED_REPLY
        total = 2 * SERVICE_ASK + new_ask
        assert provider.last_payment["cumulative_amount"] == total
        assert held() == (total,) * 3, held()
        assert len(seller_store._seen_requests) == len(seen_before) + 1, \
            "the gate did not verify the new voucher"
        s8.note("service_id", sid)
        s8.note("new_ask", new_ask)
        s8.note("running_total", total)
        step(f"price changed on chain to {new_ask}: the next voucher added "
             "it and the gate verified it, with no new spec")
        s8.status = "PASS"
    except Exception as e:
        s8.status = "FAIL"
        s8.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    s9 = board.stage(9, "a call whose dispatch fails is not charged "
                        "(provider and agent tool, one channel)")
    try:
        if s8.status != "PASS":
            raise RuntimeError("stage 8 did not complete")
        from atn.agent_tools import _request_service
        from atn.providers.base import ProviderError

        price = int(registry_c.functions.getService(service_id).call()[2])
        assert price == new_ask
        base = held()

        # Through the child's provider: the seller's upstream fails.
        calls_before = len(canned.calls)
        try:
            await provider.send(messages=[{"role": "user", "content": FAIL_PROMPT}],
                                max_tokens=32)
            raise AssertionError("a failed dispatch came back as a completion")
        except ProviderError as exc:
            assert "canned upstream failure" in str(exc), exc
        assert len(canned.calls) == calls_before + 1, "the seller never dispatched"
        assert provider.last_payment["paid"] == 0, provider.last_payment
        assert held() == base, held()
        step("a failed call through the provider: paid 0, the seller kept no "
             "voucher, the buyer's book did not move")

        async def buy(text: str) -> Dict[str, Any]:
            # By the service's number, as find_services returns it.
            return await _request_service(rt_buyer, {
                "_caller_id": "child-1", "service_id": service_id,
                "payload": {"messages": [{"role": "user", "content": text}],
                            "max_tokens": 32}})

        # The deposit left covers one more price: a failure first (it costs
        # nothing, so the same total is signed again), then a served call.
        failed2 = await buy(FAIL_PROMPT)
        assert failed2.get("delivered") and not failed2.get("ok"), failed2
        assert failed2["paid"] == 0 and failed2["channel_id"] == channel_id, (
            "the agent tool paid on another channel than the provider's",
            failed2)
        assert held() == base, held()
        served = await buy("served through the agent tool")
        assert served.get("ok") and served["paid"] == price, served
        assert served["channel_id"] == channel_id, served
        assert held() == (base[0] + price,) * 3, held()
        on_chain = channel_c.functions.getChannel(channel_id).call()
        assert on_chain[5] == 1 and on_chain[2] == SERVICE_ASK * 10, on_chain
        s9.note("channel_id", channel_id)
        s9.note("price", price)
        s9.note("requests", "provider fail, tool fail, tool ok")
        s9.note("buyer_book_and_seller_voucher", base[0] + price)
        step(f"the agent tool paid on the same channel: both sides hold "
             f"{base[0] + price} (the failures cost nothing)")
        s9.status = "PASS"
    except Exception as e:
        s9.status = "FAIL"
        s9.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    s10 = board.stage(10, "the provider's own close collects its best voucher")
    closed_view = None
    # The last block before the channel's settlement (stage 12 pins a close
    # to it: the review there waits for its payment).
    block_before_close = w3.eth.block_number
    try:
        if s9.status != "PASS":
            raise RuntimeError("stage 9 did not complete")
        kept = held()[2]
        seller_before = substrate.functions.balanceOf(seller_addr).call()
        # The deposit left (10 x the first price - kept) no longer covers
        # the current price: the seller's sweep collects at once.
        assert SERVICE_ASK * 10 - kept < price
        out = await rt_seller.settle_channels_once()
        settled = out.get("settled") or []
        assert settled and settled[0]["ok"], out
        assert settled[0]["reason"] == "deposit_spent", out
        closed_view = channel_c.functions.getChannel(channel_id).call()
        client, payee, deposit, claimed, closes_at, status = closed_view[:6]
        assert status == 2 and claimed == kept, closed_view
        assert payee.lower() == seller_addr.lower()
        fee = (kept * 250) // 10000
        seller_after = substrate.functions.balanceOf(seller_addr).call()
        assert seller_after - seller_before == kept - fee, (
            seller_after - seller_before, kept - fee)
        assert str(channel_id) not in seller_store.channel_vouchers()
        pay_logs = substrate.events.ServicePayment().get_logs(from_block=0)
        assert len(pay_logs) == 1 and int(pay_logs[0]["args"]["amount"]) == kept, \
            pay_logs
        s10.note("claimed", claimed)
        s10.note("seller_credited_net", seller_after - seller_before)
        s10.note("closed_with", "operator key (vault)")
        s10.note("service_payments_on_chain", len(pay_logs))
        step(f"the seller's sweep closed channel {channel_id} with its best "
             f"voucher: {kept} claimed in one settlement, {kept - fee} net "
             "to the operator")
        s10.status = "PASS"
    except Exception as e:
        s10.status = "FAIL"
        s10.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    s11 = board.stage(11, "the buyer's refund sweep returns the remainder")
    try:
        if s10.status != "PASS" or closed_view is None:
            raise RuntimeError("stage 10 did not close the channel")
        deposit, claimed, closes_at = (int(closed_view[2]), int(closed_view[3]),
                                       int(closed_view[4]))
        remainder = deposit - claimed
        child_before = substrate.functions.balanceOf(child_addr).call()
        out = await rt_buyer.settle_channels_once()
        actions = [r["action"] for r in out.get("refunds") or []]
        assert "closed_by_provider" in actions, out
        assert book.get(child_addr, seller_addr) is None
        withdrew = "withdrew" in actions
        if not withdrew:
            assert str(channel_id) in book.closing()
            assert substrate.functions.balanceOf(child_addr).call() == child_before
        deadline = time.time() + CHANNEL_WINDOW + 30
        while not withdrew and time.time() < deadline:
            await asyncio.sleep(1.0)
            if time.time() > closes_at:
                # Hardhat mines a block only for a transaction, so its clock
                # stands still between them and the withdraw's estimate
                # would still see the window open. A live chain's blocks
                # keep coming; here one empty block moves the clock.
                w3.provider.make_request("evm_mine", [])
            out = await rt_buyer.settle_channels_once()
            withdrew = any(r["action"] == "withdrew"
                           for r in out.get("refunds") or [])
        assert withdrew, (f"no refund by {deadline} (closes_at {closes_at}): "
                          f"{out}")
        child_after = substrate.functions.balanceOf(child_addr).call()
        assert child_after - child_before == remainder, (
            child_after - child_before, remainder)
        assert channel_c.functions.getChannel(channel_id).call()[5] == 3
        assert book.closing() == {}
        s11.note("remainder_refunded", remainder)
        step(f"the buyer's sweep noticed the provider's close and took back "
             f"{remainder} after the {CHANNEL_WINDOW}s window")
        s11.status = "PASS"
    except Exception as e:
        s11.status = "FAIL"
        s11.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 12: the buyer reviews a paid request; the next close rates it
    # (service reviews in the epoch-close, Decision 2026-11-10)
    # ------------------------------------------------------------------
    s12 = board.stage(12, "the buyer reviews a paid request; the next "
                          "close's anchored state carries the rating")
    try:
        if s9.status != "PASS":
            raise RuntimeError("stage 9 did not complete")
        from unittest.mock import MagicMock

        from atn import service_catalog as sc_mod
        from atn.agent_tools import _review_service
        from nodes.common.authoritative_encoding import cid_for_blob, decode_carry_bundle
        from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
        from nodes.common.federated_close_driver import FederatedCloseDriver
        from nodes.common.service_reviews import service_key

        receipts = rt_buyer.service_receipts
        jurisdiction = "0xe2e-service-reviews"
        sent: List[Dict[str, Any]] = []
        receipts.attach(jurisdiction, sent.append)

        # The seller's signed reply left a receipt on the buyer's daemon:
        # the payer the seller's gate verified, what it charged, no result.
        rid = str(served["request_id"])
        kept = receipts.get(rid)
        assert kept and kept["charged"] is True, kept
        assert kept["payer"] == child_addr.lower(), (kept["payer"], child_addr)
        assert kept["provider"] == seller_addr.lower()
        assert int(kept["amount"]) == price, kept
        assert CANNED_REPLY not in json.dumps(kept)
        # The failed request's receipt says it charged nothing: no review.
        refused = await _review_service(rt_buyer, {
            "_caller_id": "child-1", "request_id": str(failed2["request_id"]),
            "correctness": -1, "simplicity": 0, "novelty": 0, "malware": False})
        assert "not charged" in str(refused.get("error")), refused

        review = await _review_service(rt_buyer, {
            "_caller_id": "child-1", "request_id": rid, "correctness": 0.9,
            "simplicity": 0.7, "novelty": 0.2, "malware": False,
            "note": "answered as described"})
        assert review.get("ok"), review
        assert len(sent) == 1 and sent[0]["reviewer"] == child_addr.lower()

        # The event rides the buyer's batch into the next close, which reads
        # the chain pinned to its snapshot block: the registered services,
        # the payments and the closed channels (read_service_snapshot).
        from nodes.common.voice_state import read_service_snapshot

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
        cid = str(kept["channel_id"])
        assert int(cid) == channel_id, (cid, channel_id)

        # Pinned before the seller's close: registered, not yet settled.
        before = pinned(block_before_close)
        assert key in before["registered"] and cid not in before["channels"]
        drv.service_snapshot = before
        waiting = drv.run({"epoch_id": "e2e-service-reviews-0"})
        assert waiting is not None
        assert key not in waiting.close_result["service_ratings"]
        held = waiting.close_result["service_reviews"][key][child_addr.lower()]
        assert held["paid"] is False, held

        # Pinned after it: the settlement covers the review, which counts.
        drv.service_snapshot = pinned(w3.eth.block_number)
        # The channel's settlement covers the reviewed payment (it holds
        # every served call on the channel, so at least this one).
        settled = int(drv.service_snapshot["channels"][cid]["paid"])
        assert settled >= int(kept["amount"]), (settled, kept)
        gossip.drain_epoch_batches.return_value = [batch]   # replayed: no change
        fed = drv.run({"epoch_id": "e2e-service-reviews-1"})
        assert fed is not None, "the close returned nothing"
        assert fed.close_result["authoritative_payload"]["carry_cid"] == \
            cid_for_blob(fed.carry_bundle_blob)
        carry = decode_carry_bundle(fed.carry_bundle_blob)["carry"]
        rating = carry["service_ratings"].get(key)
        assert rating, carry["service_ratings"]
        assert rating["payers"] == 1 and rating["correctness"] == 0.9, rating
        assert rating["simplicity"] == 0.7 and rating["malware"] == 0.0, rating
        assert rating["charged"] == price, rating
        assert carry["service_reviews"][key][child_addr.lower()]["note"] == \
            "answered as described"
        assert fed.close_result["agent_mint"] == {}

        # Search shows it: the buyer's catalog reads the close's rating.
        catalog = sc_mod.catalog_for(rt_buyer)
        synced = await asyncio.to_thread(catalog.sync_chain)
        assert not synced.get("error"), synced
        catalog.note_spec(spec_digest, rt_seller.service_store.get(spec_digest).spec)
        catalog.rating_provider = drv.service_rating
        rows = [r for r in catalog.search("") if r["digest"] == spec_digest]
        assert rows and rows[0]["rating"]["payers"] == 1, rows
        s12.note("rating", {k: rating[k] for k in ("payers", "correctness", "charged")})
        s12.note("carry_cid", fed.carry_cid[:16])
        step(f"the child reviewed request {rid[:10]}; the close's carry bundle "
             f"rates the service ({rating['payers']} payer, correctness "
             f"{rating['correctness']}) and search shows it")
        s12.status = "PASS"
    except Exception as e:
        s12.status = "FAIL"
        s12.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 13: teardown
    # ------------------------------------------------------------------
    s13 = board.stage(13, "teardown")
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
        s13.status = "PASS"
    except Exception as e:
        s13.status = "FAIL"
        s13.note("error", repr(e))
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
