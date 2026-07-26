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
    on-chain ServiceRegistry entry, real published ws endpoint)
      -> parent agent, on-chain, with a funded child
      -> parent binds the CHILD to that service (update_agent, parent-only)
      -> the child does an ordinary chat completion through its provider
      -> real Substrate.payForService signed with the CHILD'S key
      -> provider-side gate verifies the payment on chain
      -> the canned completion round-trips back into a ProviderResponse

The LLM seam is the ONE thing mocked: ``_resolve_sponsor_provider`` on the
selling daemon returns a canned-response fake, so this needs no real model.
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
CHILD_FUNDING = 10_000        # ATN the parent's owner grants the child's wallet
SELLER_WS_PORT = 7791         # the selling daemon's local WS listener
CANNED_REPLY = "The bound child thinks on purchased cognition."


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
  const zero = "0x0000000000000000000000000000000000000000";
  // Substrate(treasury, vaultMinter, governor). governor=zero => the service
  // fee is frozen at its 2.5% default, which is what this script asserts.
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy(deployer.address, zero, zero);
  await substrate.waitForDeployment();
  const substrateAddr = await substrate.getAddress();

  // ServiceRegistry's constructor takes the substrate address. PaymentChannel
  // is not deployed here: this rail settles by DIRECT payForService per call
  // (channels are the v2 upgrade).
  const Registry = await ethers.getContractFactory("ServiceRegistry");
  const registry = await Registry.deploy(substrateAddr);
  await registry.waitForDeployment();
  const registryAddr = await registry.getAddress();

  const out = { substrate: substrateAddr, serviceRegistry: registryAddr };
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
        env = dict(os.environ)
        env["DEPLOY_OUT"] = str(out_json)
        log("deploying Substrate + ServiceRegistry via hardhat run…")
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
                            agent_addr: str, owner_acct, lineage_hash: bytes,
                            peer_id: bytes) -> str:
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
        reg_fn(lineage_hash, peer_id, owner_acct.address, ZERO, sig),
        agent_pk, agent_addr)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def make_runtime(tmp_root: Path, *, owner_wallet: str, owner_pk: str,
                 addresses: Dict[str, Optional[str]]):
    """A real ATN Runtime wired to the local chain."""
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

    Stands in for whatever provider the SELLING daemon would resolve out of
    its own stack (`AutonetBridge._resolve_sponsor_provider`). Everything on
    the buying side is real, so the child cannot tell this from a GPU.
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
    s1 = board.stage(1, "chain up (hardhat node + Substrate + ServiceRegistry)")
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
    s2 = board.stage(2, "fleet (seller daemon + buyer daemon, agents on chain)")
    rt_seller = rt_buyer = None
    seller_defn = parent_defn = child_defn = None
    seller_key = parent_key = child_key = ""
    seller_addr = parent_addr = child_addr = ""
    try:
        if shape["register"] is None or shape["hash_view"] is None:
            raise RuntimeError(
                f"contract exposes no owner-binding surface (shape={shape})")

        rt_seller = make_runtime(
            tmp_seller, owner_wallet=owner_seller.address,
            owner_pk=owner_seller._private_key.hex(), addresses=addresses)
        rt_buyer = make_runtime(
            tmp_buyer, owner_wallet=owner_buyer.address,
            owner_pk=owner_buyer._private_key.hex(), addresses=addresses)

        # Seller daemon: the agent that SELLS cognition.
        seller_defn = await register_local_agent(rt_seller, "seller-1")
        # Buyer daemon: the PARENT (employer) and its CHILD. parent_id is what
        # makes the binding a parent act rather than a self-act.
        parent_defn = await register_local_agent(rt_buyer, "parent-1")
        child_defn = await register_local_agent(rt_buyer, "child-1",
                                                parent_id="parent-1")

        seller_key = rt_seller.registry.get_agent_key("seller-1")
        parent_key = rt_buyer.registry.get_agent_key("parent-1")
        child_key = rt_buyer.registry.get_agent_key("child-1")
        seller_addr = Web3.to_checksum_address(seller_defn.identity.address)
        parent_addr = Web3.to_checksum_address(parent_defn.identity.address)
        child_addr = Web3.to_checksum_address(child_defn.identity.address)
        assert child_key and child_key != parent_key, (
            "the child must hold its OWN distinct wallet key")

        # Gas for every agent that signs (registration, service ops, payment).
        for owner, addr in ((owner_seller, seller_addr),
                            (owner_buyer, parent_addr),
                            (owner_buyer, child_addr)):
            fund_gas(w3, owner, addr)

        for defn, pk, addr, owner, peer in (
            (seller_defn, seller_key, seller_addr, owner_seller, b"peer-seller"),
            (parent_defn, parent_key, parent_addr, owner_buyer, b"peer-parent"),
            (child_defn, child_key, child_addr, owner_buyer, b"peer-child"),
        ):
            lin = bytes.fromhex(
                defn.identity.lineage_hash.replace("0x", "").ljust(64, "0")[:64])
            register_agent_on_chain(
                w3, substrate, shape, agent_pk=pk, agent_addr=addr,
                owner_acct=owner, lineage_hash=lin, peer_id=peer)
            defn.identity.registered_on_chain = True

        assert substrate.functions.getAgentOwner(seller_addr).call() == \
            owner_seller.address
        assert substrate.functions.getAgentOwner(child_addr).call() == \
            owner_buyer.address
        s2.note("seller", seller_addr)
        s2.note("parent", parent_addr)
        s2.note("child", child_addr)
        s2.note("child_key_distinct", True)
        step("three agents registered on chain; child holds its own key")
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
        rt_seller.autonet._resolve_sponsor_provider = (
            lambda cfg, model: canned)

        ws_server = WebSocketBridge(rt_seller, host="127.0.0.1",
                                    port=SELLER_WS_PORT)
        await ws_server.start()

        # Register the service through the real WS handler (owner surface):
        # an `inference` block instead of a backing tool.
        reg = await ws_server._handle_message(
            {
                "type": "register_service",
                "name": "canned-cognition",
                "description": "Chat completions off the seller's own stack.",
                "input_schema": {"type": "object",
                                 "properties": {"messages": {"type": "array"}}},
                "agent_id": "seller-1",
                "ask": {"token": addresses["substrate"],
                        "amount": str(SERVICE_ASK), "unit": "per_item"},
                "inference": {"model": "canned-model-1",
                              "max_tokens_cap": 512},
            },
            owner_session())
        if not reg.get("ok"):
            raise RuntimeError(f"register_service failed: {reg}")
        spec_digest = reg["result"]["digest"]
        record = rt_seller.service_store.get(spec_digest)
        assert record.inference, "spec carries no inference backing"
        # The gate verifies payment against the spec's author_pubkey, so it
        # must be the seller agent's own 0x.
        assert record.spec["author_pubkey"].lower() == seller_addr.lower(), (
            record.spec["author_pubkey"], seller_addr)

        # On-chain listing, signed by the seller AGENT's key (the provider of
        # record — and the payment recipient the gate will check).
        send_tx(w3,
                registry_c.functions.registerService(
                    bytes.fromhex(spec_digest), SERVICE_ASK),
                seller_key, seller_addr)
        service_id = registry_c.functions.serviceCount().call()

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
        from atn.orchestrator.tools import execute_tool

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
    # Stage 6 — fund the child, then a real paid completion end to end
    # ------------------------------------------------------------------
    s6 = board.stage(6, "paid completion (payForService from the child's wallet)")
    try:
        if provider is None:
            raise RuntimeError("no ServiceProvider from stage 5")
        # Fund the child's wallet. ATN only enters existence through the epoch
        # mint path (Substrate has no admin mint), so this uses the production
        # rail: anchor an epoch whose agentMintRoot commits the child's share,
        # then have the CHILD record its own mint from its own key. That is
        # also the honest story of where a child's operating float comes from —
        # the employer allocates it in the close, the child claims it itself.
        from eth_utils import keccak
        from nodes.common.mint_merkle import mint_merkle_proof, mint_merkle_root

        epoch_id = "e2e-svcprov-epoch-1"
        epoch_hash = keccak(text=epoch_id)
        # mint_scale=1: CHILD_FUNDING is already in base units here.
        mint_map = {child_addr: float(CHILD_FUNDING)}
        mint_root = mint_merkle_root(mint_map, 1)
        proof = mint_merkle_proof(mint_map, child_addr, 1) or []
        ZERO32 = b"\x00" * 32
        send_tx(w3, substrate.functions.submitAnchor(
            epoch_id,          # epochId (string)
            ZERO32,            # epochRoot (unused by this rail)
            ZERO32,            # prevEpochRoot (clean genesis)
            ZERO32,            # prevAnchorHash (clean genesis)
            "",                # agentMintCid (blob served off-chain)
            ZERO32,            # payloadHash
            mint_root,         # agentMintRoot — what the proof verifies against
        ), owner_seller._private_key.hex(), owner_seller.address)
        send_tx(w3, substrate.functions.recordTrainingForEpoch(
            CHILD_FUNDING, epoch_hash, proof), child_key, child_addr)
        child_atn = substrate.functions.balanceOf(child_addr).call()
        assert child_atn >= SERVICE_ASK, (
            f"child wallet only holds {child_atn}, needs >= {SERVICE_ASK}")
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

        # --- Money moved, and from the RIGHT wallet ---------------------
        seller_after = substrate.functions.balanceOf(seller_addr).call()
        child_after = substrate.functions.balanceOf(child_addr).call()
        owner_after = substrate.functions.balanceOf(
            rt_buyer._config.autonet.owner_wallet).call()

        # The 2.5% service fee is burned at payForService (fee-recycled
        # emission), so the recipient receives NET.
        fee = (SERVICE_ASK * 250) // 10000
        net = SERVICE_ASK - fee
        assert child_before - child_after == SERVICE_ASK, (
            f"child paid {child_before - child_after}, expected {SERVICE_ASK}")
        assert seller_after - seller_before == net, (
            f"seller received {seller_after - seller_before}, expected {net}")
        assert owner_after == owner_before, (
            "the daemon OWNER's wallet moved — the child must pay for itself")
        s6.note("child_debited", child_before - child_after)
        s6.note("seller_credited_net", seller_after - seller_before)
        s6.note("fee_burned", fee)
        s6.note("owner_wallet_untouched", True)
        step(f"child debited {SERVICE_ASK}, seller credited {net} net of "
             f"{fee} fee; owner wallet untouched")

        # --- The payment event names the right parties ------------------
        pay_logs = substrate.events.ServicePayment().get_logs(from_block=0)
        assert pay_logs, "no ServicePayment event on chain"
        ev = pay_logs[-1]["args"]
        assert ev["recipient"].lower() == seller_addr.lower(), ev
        # The payer is the CHILD's 0x — the on-chain record of who bought it.
        payer = ev.get("payer") or ev.get("client") or ev.get("from")
        if payer is not None:
            assert str(payer).lower() == child_addr.lower(), (payer, child_addr)
        assert int(ev["amount"]) == SERVICE_ASK, ev
        s6.note("event_recipient", ev["recipient"])
        s6.note("event_amount", int(ev["amount"]))
        step("ServicePayment event: correct recipient, amount, and payer")

        # --- The gate really VERIFIED, it did not degrade open ----------
        # A degraded gate would have served this without any chain check, so
        # the request id would never have been consumed. Its presence in the
        # served set is the proof that the enforced path ran.
        # The gate stores whatever string the wire carried; the event gives us
        # bytes32. Compare in a normalized space rather than guessing the
        # prefix convention.
        seen = {str(r).lower().removeprefix("0x")
                for r in rt_seller.service_store._seen_requests}
        rid = provider_last_request_id(pay_logs).lower().removeprefix("0x")
        assert seen, (
            "the gate consumed NO request ids — it DEGRADED OPEN instead of "
            "verifying the payment on chain")
        assert rid in seen, (
            f"request id {rid[:18]} was never marked served (gate saw "
            f"{[s[:18] for s in seen]}) — the payment was not the one verified")
        s6.note("gate_verified_not_degraded", True)
        step("provider gate verified the payment on chain (not degraded)")
        s6.status = "PASS"
    except Exception as e:
        s6.status = "FAIL"
        s6.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 7 — replay guard: a second call must use a FRESH request id
    # ------------------------------------------------------------------
    s7 = board.stage(7, "replay guard (fresh request_id per call)")
    try:
        if provider is None or s6.status != "PASS":
            raise RuntimeError("stage 6 did not complete a paid call")
        seen_before = set(rt_seller.service_store._seen_requests)

        resp2 = await provider.send(
            messages=[{"role": "user", "content": "and again?"}],
            max_tokens=64)
        assert resp2.text == CANNED_REPLY

        pay_logs = substrate.events.ServicePayment().get_logs(from_block=0)
        rids = [bytes(l["args"]["requestId"]).hex()
                if isinstance(l["args"].get("requestId"), (bytes, bytearray))
                else str(l["args"].get("requestId"))
                for l in pay_logs]
        assert len(rids) >= 2, rids
        assert rids[-1] != rids[-2], (
            "the second call reused a request_id — the provider gate would "
            "treat that as a replay and refuse it")
        seen_after = set(rt_seller.service_store._seen_requests)
        assert len(seen_after) == len(seen_before) + 1, (
            f"the gate consumed {len(seen_after) - len(seen_before)} request "
            f"ids on the second call, expected exactly 1")
        s7.note("payments", len(rids))
        s7.note("request_ids_distinct", True)
        s7.note("ids_consumed_by_gate", len(seen_after))
        step("second call used a FRESH request_id (replay guard intact)")

        # And a REPLAYED id is refused by the gate — assert the guard directly
        # by re-sending an already-served id in the EXACT form the gate stored
        # it (the seen-set is string-keyed, so the prefix convention matters).
        first_rid = next(
            r for r in seen_before
            if str(r).lower().removeprefix("0x") == rids[-2].lower()
            .removeprefix("0x"))
        replay = await ws_server._handle_message(
            {
                "type": "service_request",
                "spec_digest": spec_digest,
                "request_id": first_rid,
                "args": {"messages": [{"role": "user", "content": "replay"}]},
                "client": child_addr,
                "tx_hash": pay_logs[-2]["transactionHash"].hex(),
            },
            owner_session())
        assert not replay.get("ok"), (
            f"the gate ACCEPTED a replayed request_id: {replay}")
        assert "replay" in str(replay.get("error", "")).lower() or \
            "already served" in str(replay.get("error", "")).lower(), replay
        s7.note("replay_refused", True)
        step("gate refuses a replayed request_id")
        s7.status = "PASS"
    except Exception as e:
        s7.status = "FAIL"
        s7.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 8 — teardown
    # ------------------------------------------------------------------
    s8 = board.stage(8, "teardown")
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
        s8.status = "PASS"
    except Exception as e:
        s8.status = "FAIL"
        s8.note("error", repr(e))
        log(traceback.format_exc())
        failed = True
    finally:
        kill_hardhat(hh_proc)

    print(board.render())
    any_fail = any(s.status == "FAIL" for s in board.stages.values())
    return 1 if (failed or any_fail) else 0


def provider_last_request_id(pay_logs) -> str:
    """The request id of the most recent ServicePayment, as stored by the gate."""
    if not pay_logs:
        return ""
    rid = pay_logs[-1]["args"].get("requestId")
    if isinstance(rid, (bytes, bytearray)):
        return "0x" + bytes(rid).hex()
    return str(rid or "")


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
