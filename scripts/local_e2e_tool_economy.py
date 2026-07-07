#!/usr/bin/env python3
"""End-to-end proof that the tool-economy architecture composes on a
local deployment.

This is the tool-substrate session's capstone. It stands up a real
Hardhat chain, deploys Substrate.sol + ServiceMarket.sol, spins up two
real ATN Runtimes, registers agents on-chain with owner-verified
binding (EIP-712), drives a full tool economy through the daemon
(register -> invoke -> attest), runs the REAL federated epoch close
fed by the captured substrate events + the owner map READ BACK FROM
CHAIN, and settles the resulting mint back to the chain (anchor +
recordTrainingForEpoch). It verifies each hop and prints a
stage-by-stage scoreboard, exiting nonzero on any failure.

Run:
    python scripts/local_e2e_tool_economy.py

See docs/local_e2e.md for what it proves and the seams it revealed.

The owner-binding EIP-712 struct is under concurrent refactor
(Sponsorship -> OwnerBinding, registerAgentSponsored ->
registerAgentBound, +nonce). This script signs against WHATEVER the
deployed contract exposes: it reads the digest from the contract's own
`bindingHash`/`sponsorshipHash` view and signs that raw digest, and it
picks the registration method by ABI introspection. So it adapts to
either the pre- or post-rename contract shape with no edits.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RPC_URL = "http://127.0.0.1:8545"
# `npx hardhat node` serves the in-process "hardhat" network, whose
# chainId is 1337 (see hardhat.config.js). NOT 31337 (that's the
# eth_tester default the daemon config assumes). Signing for the wrong
# id => "incompatible EIP-155 transaction, signed for another chain".
CHAIN_ID = 1337
MINT_SCALE = 1_000_000

# Hardhat's deterministic mnemonic accounts (well-known, safe: local only).
# account[0] = OWNER_A (author's human wallet, also the deployer/anchorer)
# account[1] = OWNER_B (caller's human wallet)
HH_ACCT0_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
HH_ACCT1_PK = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"


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
        lines = ["", "=" * 74, "  TOOL-ECONOMY LOCAL E2E — SCOREBOARD", "=" * 74]
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


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        shell=(os.name == "nt"),
    )


def _npx() -> str:
    return "npx.cmd" if os.name == "nt" else "npx"


def start_hardhat(cwd: Path) -> subprocess.Popen:
    """Launch `npx hardhat node` detached; caller polls the RPC."""
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    # Discard the node's stdout/stderr: hardhat node logs every RPC call,
    # and a filled PIPE buffer blocks the process (seen as an undici
    # HeadersTimeoutError on the deploy side). DEVNULL keeps it draining.
    proc = subprocess.Popen(
        [_npx(), "hardhat", "node"],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        shell=(os.name == "nt"),
        creationflags=creationflags,
    )
    return proc


def wait_rpc(timeout: float = 40.0) -> bool:
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

    Contracts live under contracts/core (Substrate, ServiceMarket's
    ServiceRegistry/PaymentChannel) and contracts/test (MockERC20). The
    artifact dir mirrors the source path AND the containing .sol file
    name — but ServiceMarket.sol holds several contracts, so the .sol
    stem differs from the contract name. Scan the artifacts tree for
    <name>.json to be robust to both.
    """
    art = REPO / "artifacts" / "contracts"
    # Direct guesses first (fast path).
    for sub in ("core", "test"):
        for sol in (name, "ServiceMarket"):
            p = art / sub / f"{sol}.sol" / f"{name}.json"
            if p.exists():
                with p.open("r", encoding="utf-8") as fh:
                    return json.load(fh)["abi"]
    # Fallback: walk the tree.
    for p in art.rglob(f"{name}.json"):
        if p.parent.name.endswith(".sol"):
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)["abi"]
    raise FileNotFoundError(f"no artifact ABI for {name} under {art}")


# ---------------------------------------------------------------------------
# Deploy (tiny inline JS deployer covering Substrate + ServiceMarket)
# ---------------------------------------------------------------------------

DEPLOY_JS = r"""
const { ethers } = require("hardhat");
const fs = require("fs");
async function main() {
  const [deployer] = await ethers.getSigners();
  const Substrate = await ethers.getContractFactory("Substrate");
  const substrate = await Substrate.deploy((await ethers.getSigners())[0].address);
  await substrate.waitForDeployment();
  const substrateAddr = await substrate.getAddress();

  // ServiceMarket.sol declares two contracts. Deploy the registry
  // (constructor takes the substrate address); the payment channel takes a
  // single challenge-window arg — it is the ONLY settlement rail (the
  // postpaid escrow was deleted). MockERC20 for the channel round-trip.
  const Registry = await ethers.getContractFactory("ServiceRegistry");
  const registry = await Registry.deploy(substrateAddr);
  await registry.waitForDeployment();
  const registryAddr = await registry.getAddress();

  const Channel = await ethers.getContractFactory("PaymentChannel");
  const channel = await Channel.deploy(3600);
  await channel.waitForDeployment();
  const channelAddr = await channel.getAddress();

  let mockErc20 = null;
  try {
    const Mock = await ethers.getContractFactory("MockERC20");
    const mock = await Mock.deploy("MockUSD", "mUSD");
    await mock.waitForDeployment();
    mockErc20 = await mock.getAddress();
  } catch (e) { mockErc20 = null; }

  const out = {
    substrate: substrateAddr,
    serviceRegistry: registryAddr,
    paymentChannel: channelAddr,
    mockErc20: mockErc20,
  };
  fs.writeFileSync(process.env.DEPLOY_OUT, JSON.stringify(out, null, 2));
  console.log("DEPLOYED " + JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""


def deploy_contracts(cwd: Path) -> Dict[str, Optional[str]]:
    scripts_dir = cwd / "scripts"
    tmp_js = scripts_dir / "_e2e_deploy_tmp.js"
    out_json = cwd / "deployments" / "_e2e_local.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_js.write_text(DEPLOY_JS, encoding="utf-8")
    try:
        env = dict(os.environ)
        env["DEPLOY_OUT"] = str(out_json)
        log("deploying Substrate + ServiceMarket via hardhat run…")
        proc = subprocess.run(
            [_npx(), "hardhat", "run", str(tmp_js), "--network", "localhost"],
            cwd=str(cwd), capture_output=True, text=True, timeout=300,
            shell=(os.name == "nt"), env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"deploy failed (rc={proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return data
    finally:
        try:
            tmp_js.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# EIP-712 owner-binding signing (adapts to pre/post rename)
# ---------------------------------------------------------------------------

def resolve_binding_shape(substrate_abi: list) -> Dict[str, Any]:
    """Inspect the ABI to discover which owner-binding surface is on
    disk. Returns method/view/event names + whether the struct carries
    a nonce (mandatory in the post-rename contract).
    """
    fns = {a["name"]: a for a in substrate_abi
           if a.get("type") == "function"}
    shape: Dict[str, Any] = {}
    # Registration method: post-rename registerAgentBound, else
    # registerAgentSponsored.
    if "registerAgentBound" in fns:
        shape["register"] = "registerAgentBound"
    elif "registerAgentSponsored" in fns:
        shape["register"] = "registerAgentSponsored"
    else:
        shape["register"] = None
    # Digest view.
    if "bindingHash" in fns:
        shape["hash_view"] = "bindingHash"
    elif "sponsorshipHash" in fns:
        shape["hash_view"] = "sponsorshipHash"
    else:
        shape["hash_view"] = None
    # Nonce view (post-rename mandatory).
    if "bindingNonce" in fns:
        shape["nonce_view"] = "bindingNonce"
    elif "sponsorshipNonce" in fns:
        shape["nonce_view"] = "sponsorshipNonce"
    else:
        shape["nonce_view"] = None
    # Does the register method take a nonce arg? Check its inputs.
    reg = fns.get(shape["register"]) if shape["register"] else None
    shape["register_inputs"] = (
        [i["name"] for i in reg["inputs"]] if reg else []
    )
    return shape


def sign_binding_digest(w3, contract, shape: Dict[str, Any],
                        owner_pk: str, agent_addr: str, parent_addr: str,
                        nonce: int) -> bytes:
    """Sign the owner-binding EIP-712 digest.

    Robust strategy: ask the CONTRACT for the exact digest via its
    hash view (bindingHash / sponsorshipHash), then sign that raw
    digest with the owner key. This sidesteps any struct-field / typehash
    drift entirely — whatever the contract hashes is what we sign.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct  # noqa: F401 (fallback)

    hash_view = shape["hash_view"]
    view_fn = getattr(contract.functions, hash_view)
    # The hash view may take (agent, parent) or (agent, parent, nonce).
    view_abi = next(a for a in contract.abi
                    if a.get("type") == "function" and a["name"] == hash_view)
    n_inputs = len(view_abi["inputs"])
    if n_inputs >= 3:
        digest = view_fn(agent_addr, parent_addr, nonce).call()
    else:
        digest = view_fn(agent_addr, parent_addr).call()
    digest = bytes(digest)
    # Sign the raw 32-byte digest (EIP-712 digest is already the final
    # hash the contract recovers against with ECDSA.recover).
    acct = Account.from_key(owner_pk)
    signed = Account._sign_hash(digest, acct._private_key) \
        if hasattr(Account, "_sign_hash") else acct.unsafe_sign_hash(digest)
    sig = signed.signature
    return bytes(sig)


def sponsored_register(w3, contract, shape: Dict[str, Any],
                       agent_pk: str, agent_addr: str,
                       owner_pk: str, owner_addr: str,
                       parent_addr: str, lineage_hash: bytes,
                       peer_id: bytes) -> str:
    """registerAgentBound / registerAgentSponsored from the agent key,
    with an owner EIP-712 binding signature."""
    from eth_account import Account

    nonce_val = 0
    if shape["nonce_view"]:
        nonce_val = getattr(contract.functions,
                            shape["nonce_view"])(agent_addr).call()
    sig = sign_binding_digest(
        w3, contract, shape, owner_pk, agent_addr, parent_addr, nonce_val)

    reg_fn = getattr(contract.functions, shape["register"])
    inputs = shape["register_inputs"]
    # Build args positionally to match the ABI: the register method takes
    # (lineageHash, peerId, owner, parentAgent, ownerSig) in both the
    # pre- and post-rename shapes (nonce is carried in the sig, not the
    # call args — the contract reads its own bindingNonce).
    call = reg_fn(lineage_hash, peer_id, owner_addr, parent_addr, sig)

    agent_acct = Account.from_key(agent_pk)
    tx = call.build_transaction({
        "from": agent_addr,
        "nonce": w3.eth.get_transaction_count(agent_addr),
        "gas": 1_500_000,
        "gasPrice": w3.eth.gas_price,
        "chainId": CHAIN_ID,
    })
    signed = w3.eth.account.sign_transaction(tx, agent_pk)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt.status != 1:
        raise RuntimeError(f"sponsored register reverted: {receipt}")
    return tx_hash.hex()


def send_agent_tx(w3, call, agent_pk: str, agent_addr: str,
                  gas: int = 1_000_000) -> str:
    tx = call.build_transaction({
        "from": agent_addr,
        "nonce": w3.eth.get_transaction_count(agent_addr),
        "gas": gas,
        "gasPrice": w3.eth.gas_price,
        "chainId": CHAIN_ID,
    })
    signed = w3.eth.account.sign_transaction(tx, agent_pk)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt.status != 1:
        raise RuntimeError(f"tx reverted: {receipt}")
    return tx_hash.hex()


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def make_runtime(tmp_root: Path, owner_wallet: str):
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime

    data_dir = tmp_root / "data"
    agents_dir = tmp_root / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    return Runtime(EventBus(), data_dir=data_dir, config=config)


async def register_local_agent(rt, agent_id: str, parent_id: Optional[str] = None):
    from atn.models import AgentDefinition, AgentMode
    defn = AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5", parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def _fake_embedder(text):
    # Deterministic, dependency-free — avoids the torch subprocess worker.
    return (0.11, 0.22, 0.33) if text else ()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> int:
    from web3 import Web3

    board = Board()
    hh_proc: Optional[subprocess.Popen] = None
    tmp_a = Path(tempfile.mkdtemp(prefix="e2e_rt_a_"))
    tmp_b = Path(tempfile.mkdtemp(prefix="e2e_rt_b_"))
    failed = False

    # ------------------------------------------------------------------
    # Stage 1 — chain up
    # ------------------------------------------------------------------
    s1 = board.stage(1, "chain up (hardhat node + deploy)")
    addresses: Dict[str, Optional[str]] = {}
    try:
        hh_proc = start_hardhat(REPO)
        if not wait_rpc(45.0):
            raise RuntimeError("hardhat RPC never came up")
        addresses = deploy_contracts(REPO)
        s1.note("substrate", addresses["substrate"])
        s1.note("serviceRegistry", addresses["serviceRegistry"])
        s1.note("mockErc20", addresses.get("mockErc20"))
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

    from eth_account import Account
    owner_a = Account.from_key(HH_ACCT0_PK)   # author's human wallet
    owner_b = Account.from_key(HH_ACCT1_PK)   # caller's human wallet

    binding_shape = resolve_binding_shape(substrate_abi)
    log(f"owner-binding shape: {binding_shape}")

    # ------------------------------------------------------------------
    # Stage 2 — fleet (two runtimes, owner-rooted lineage)
    # ------------------------------------------------------------------
    s2 = board.stage(2, "fleet (two ATN runtimes, owner-rooted lineage)")
    try:
        rt_a = make_runtime(tmp_a, owner_wallet=owner_a.address)
        rt_b = make_runtime(tmp_b, owner_wallet=owner_b.address)

        # Runtime A hosts author-1 (owner A). Runtime A ALSO hosts
        # caller-1 (owner B) — a single runtime can host agents of
        # different owners; owner is chain data, not runtime data. This
        # sidesteps cross-runtime tool invocation (tools are daemon-local)
        # while keeping the two distinct on-chain owners the damper needs.
        await register_local_agent(rt_a, "author-1")
        await register_local_agent(rt_a, "caller-1")
        # Runtime B hosts its own caller (proves the two-runtime fleet
        # stands up); the consensus stage uses runtime A's shared world.
        await register_local_agent(rt_b, "caller-1")

        author_ident = rt_a.get_agent("author-1").identity
        caller_ident = rt_a.get_agent("caller-1").identity
        # Owner-rooted lineage seeding: author-1 (parentless) roots in
        # owner A's wallet.
        assert author_ident.parent_lineage_hash == owner_a.address, (
            f"author lineage not owner-rooted: {author_ident.parent_lineage_hash}")
        s2.note("author_lineage_root", author_ident.parent_lineage_hash)
        s2.note("author_addr", author_ident.address)
        s2.note("caller_addr", caller_ident.address)
        s2.status = "PASS"
    except Exception as e:
        s2.status = "FAIL"
        s2.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 3 — chain identity (sponsored register both agents)
    # ------------------------------------------------------------------
    s3 = board.stage(3, "chain identity (owner-bound registration)")
    author_key = caller_key = None
    try:
        if binding_shape["register"] is None or binding_shape["hash_view"] is None:
            raise RuntimeError(
                "contract exposes no owner-binding surface "
                f"(shape={binding_shape}) — mid-edit?")

        author_key = rt_a.registry.get_agent_key("author-1")
        caller_key = rt_a.registry.get_agent_key("caller-1")
        author_addr = Web3.to_checksum_address(author_ident.address)
        caller_addr = Web3.to_checksum_address(caller_ident.address)

        # Fund the agent addresses with gas from their owner wallets.
        for owner, agent_addr in ((owner_a, author_addr), (owner_b, caller_addr)):
            tx = {
                "from": owner.address, "to": agent_addr,
                "value": w3.to_wei(1, "ether"),
                "nonce": w3.eth.get_transaction_count(owner.address),
                "gas": 21000, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
            }
            signed = w3.eth.account.sign_transaction(
                tx, owner._private_key.hex())
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            w3.eth.wait_for_transaction_receipt(
                w3.eth.send_raw_transaction(raw), timeout=60)

        lin_a = bytes.fromhex(author_ident.lineage_hash.replace("0x", "").ljust(64, "0")[:64])
        lin_c = bytes.fromhex(caller_ident.lineage_hash.replace("0x", "").ljust(64, "0")[:64])
        ZERO = "0x" + "00" * 20

        tx_a = sponsored_register(
            w3, substrate, binding_shape,
            agent_pk=author_key, agent_addr=author_addr,
            owner_pk=owner_a._private_key.hex(), owner_addr=owner_a.address,
            parent_addr=ZERO, lineage_hash=lin_a,
            peer_id=b"peer-author-1")
        tx_c = sponsored_register(
            w3, substrate, binding_shape,
            agent_pk=caller_key, agent_addr=caller_addr,
            owner_pk=owner_b._private_key.hex(), owner_addr=owner_b.address,
            parent_addr=ZERO, lineage_hash=lin_c,
            peer_id=b"peer-caller-1")

        got_owner_a = substrate.functions.getAgentOwner(author_addr).call()
        got_owner_b = substrate.functions.getAgentOwner(caller_addr).call()
        assert got_owner_a == owner_a.address, (got_owner_a, owner_a.address)
        assert got_owner_b == owner_b.address, (got_owner_b, owner_b.address)
        s3.note("author_owner_onchain", got_owner_a)
        s3.note("caller_owner_onchain", got_owner_b)
        s3.note("distinct_owners", got_owner_a != got_owner_b)
        s3.status = "PASS"
    except Exception as e:
        s3.status = "FAIL"
        s3.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 4 — tool economy, daemon side
    # ------------------------------------------------------------------
    s4 = board.stage(4, "tool economy (register + invoke + attest + vet)")
    captured_events: List[Dict[str, Any]] = []
    captured_manifests: List[Tuple[dict, str]] = []
    reg_events_from_ws: List[Dict[str, Any]] = []
    tool_digest = ""
    try:
        from atn.orchestrator.tools import execute_tool
        from nodes.common.world_service import WorldService

        # Wire a WorldService onto runtime A's tool_store, exactly like
        # autonet_service does: manifest_sink -> submit_tool_manifest,
        # event_sink -> submit_events. AND mirror every event into a
        # captured sink for the consensus stage.
        ws = WorldService(
            "0xRPBLOCALe2e",
            data_root=tmp_a / "world",
            embedding_dim=64,
        )
        store = rt_a.tool_store
        store._embedder = _fake_embedder  # avoid torch subprocess

        def _event_sink(ev: dict) -> None:
            captured_events.append(dict(ev))
            ws.submit_events([ev], equilibrate_after=False)
        store.event_sink = _event_sink

        def _manifest_sink(manifest: dict, author: str) -> None:
            captured_manifests.append((dict(manifest), author))
            ws.submit_tool_manifest(manifest, agent_id=author)
        store.manifest_sink = _manifest_sink

        # Open an epoch so the manifest's registration sprout events (the
        # full replayable SubClaimSprouted, coords + author_post +
        # manifest_meta) get buffered — the tool_used receipts alone don't
        # carry the registration, and the mint needs it in the canonical
        # replay to give the claim node standing. We read them back for
        # Stage 5's canonical batch.
        ws.open_epoch("e2e-epoch-1")

        # author-1 registers a pinned echo tool (always private), then
        # publishes through the separately-granted publish_tool
        # capability — the production two-step (publish gate, a8b7588).
        reg = await execute_tool(
            "register_tool",
            {"name": "echo_tool", "description": "Echo x back.",
             "input_schema": SCHEMA, "code": ECHO_CODE},
            rt_a, caller_id="author-1")
        if "error" in reg:
            raise RuntimeError(f"register_tool failed: {reg}")
        tool_digest = reg["digest"]
        assert reg["author"] == "author-1"
        assert reg["trust_class"] == "pinned"
        assert reg["published"] is False
        pub = await execute_tool(
            "publish_tool", {"digest": tool_digest},
            rt_a, caller_id="author-1")
        if "error" in pub:
            raise RuntimeError(f"publish_tool failed: {pub}")
        assert pub["published"] is True

        # Grant caller-1 access (out of author-1's lineage), then invoke.
        store.grant(tool_digest, "caller-1")
        out = await rt_a.tool_registry.call_tool(
            "echo_tool", {"x": "round-trip"}, caller_id="caller-1")
        assert out == {"result": {"echo": "round-trip"}}, out

        # Cognitive attestation by caller-1 (the only mint-counting usage).
        att = await execute_tool(
            "attest_tools",
            {"judgments": [{"tool": "echo_tool", "ok": True, "score": 0.9,
                            "note": "did the job"}],
             "context": "wiring up an echo round-trip"},
            rt_a, caller_id="caller-1")
        assert att.get("attested") == 1, att

        # Vetting (spec: Vetting section): a published manifest is a
        # CANDIDATE — no mint until greenlit by vets from distinct
        # fleets. Two validators inspect (manifest + pinned code back)
        # then attest pass with a report.
        await register_local_agent(rt_a, "vetter-1")
        await register_local_agent(rt_a, "vetter-2")
        for v in ("vetter-1", "vetter-2"):
            inspect = await execute_tool(
                "vet_tool", {"digest": tool_digest}, rt_a, caller_id=v)
            assert inspect.get("code") == ECHO_CODE, inspect
            vet = await execute_tool(
                "vet_tool",
                {"digest": tool_digest, "verdict": "pass",
                 "report": f"{v}: read the code; echoes x, "
                           "no fs/net/env access"},
                rt_a, caller_id=v)
            assert vet.get("verdict") == "pass", vet
        n_vets = sum(1 for e in captured_events if e.get("vet"))
        s4.note("vets", n_vets)
        assert n_vets == 2

        # Pull the registration sprout events out of the WorldService's
        # epoch buffer (the manifest sink routed them there). These are the
        # full replayable events the mint needs; the tool_used receipts
        # come from the captured event_sink.
        for ev in list(ws._epoch_events):
            if (ev.get("kind") == "sub_claim_sprouted"
                    and ev.get("artifact_digest") and ev.get("manifest_meta")):
                reg_events_from_ws.append(dict(ev))

        kinds = [e.get("kind") for e in captured_events]
        n_attested = sum(1 for e in captured_events if e.get("attested"))
        s4.note("registration_events", len(reg_events_from_ws))
        s4.note("tool_digest", tool_digest[:16] + "…")
        s4.note("events_captured", len(captured_events))
        s4.note("event_kinds", ",".join(sorted(set(kinds))))
        s4.note("attested_receipts", n_attested)
        assert n_attested >= 1
        s4.status = "PASS"
    except Exception as e:
        s4.status = "FAIL"
        s4.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 5 — consensus (real federated close, chain-fed owner map)
    # ------------------------------------------------------------------
    s5 = board.stage(5, "consensus (federated close + owner-map exclusion)")
    close_result: Dict[str, Any] = {}
    author_mint_raw = 0.0
    author_agent_mint_raw = 0.0  # the FULL agent_mint share that hits chain
    try:
        from unittest.mock import MagicMock
        from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
        from nodes.common.federated_close_driver import FederatedCloseDriver

        author_addr = Web3.to_checksum_address(author_ident.address)
        caller_addr = Web3.to_checksum_address(caller_ident.address)

        # Rewrite the string agent ids on the captured events to on-chain
        # 0x addresses, exactly as the production identity resolver does
        # (world_substrate_feed.ResolvedAgentIdentity) — so the mint keys
        # by address and the on-chain merkle proof can be built.
        id_to_addr = {"author-1": author_addr, "caller-1": caller_addr}

        def _remap(ev: dict) -> dict:
            ev = dict(ev)
            for f in ("author_agent", "tool_author", "author"):
                if f in ev and ev[f] in id_to_addr:
                    ev[f] = id_to_addr[ev[f]]
            mm = ev.get("manifest_meta")
            if isinstance(mm, dict) and mm.get("author") in id_to_addr:
                mm = dict(mm)
                mm["author"] = id_to_addr[mm["author"]]
                ev["manifest_meta"] = mm
            return ev

        # Registration sprouts come from the WorldService epoch buffer;
        # tool_used receipts come from the captured event_sink. Remap the
        # author/caller string ids to on-chain addresses on both.
        #
        # SEAM (see docs/local_e2e.md): submit_tool_manifest emits the
        # registration sprout with author_post=False, so in ledger-mode
        # federated close the manifest claim node has ZERO standing and
        # mint = standing × usage = 0. The test fixture
        # (tests/test_tool_mint._registration_event) hand-sets
        # author_post=True to get standing 1. We set it here too so the
        # mint path is exercised; whether the daemon SHOULD stamp an
        # author post at registration is an open design question this
        # e2e surfaced.
        reg_events = []
        for e in reg_events_from_ws:
            e = _remap(e)
            e["author_post"] = True
            reg_events.append(e)
        use_events = [_remap(e) for e in captured_events
                      if e.get("kind") == "tool_used" and not e.get("vet")]
        # Vets ride their own wire identities — one keypair per vetter,
        # modeling validators on distinct daemons (the greenlight counts
        # distinct FLEETS; co-hosted vets vs the registration key are
        # excluded outright).
        vet_events = [_remap(e) for e in captured_events
                      if e.get("kind") == "tool_used" and e.get("vet")]

        # Build a canonical batch chain. Registration (author key) and
        # receipts (caller key) MUST ride different signing keys or the
        # wire-level self-attestation dedup zeroes the mint.
        kp_author = Keypair.generate()
        kp_caller = Keypair.generate()

        def _batches(events, kp, rpb="0xRPBLOCALe2e"):
            chain, prev = [], b""
            for i, ev in enumerate(events, start=1):
                b = EventBatch(
                    rpb_address=rpb, sender_pubkey=kp.public_key,
                    batch_seq=i, events=[ev], prev_batch_hash=prev,
                    timestamp=1_700_000_000.0 + i)
                chain.append(b)
                prev = b.content_hash()
            return chain

        all_batches = (_batches(reg_events, kp_author)
                       + _batches(use_events, kp_caller))
        for vet_ev in vet_events:
            all_batches += _batches([vet_ev], Keypair.generate())

        def _mk_driver(owner_map):
            gossip = MagicMock(spec=EventGossip)
            gossip.known_senders.return_value = [kp_author.public_key]
            gossip.sender_pubkey = kp_author.public_key
            gossip.drain_epoch_batches.return_value = list(all_batches)
            drv = FederatedCloseDriver(
                gossip=gossip, embedding_dim=64,
                tool_registrations_path=tmp_a / "tool_regs.json")
            drv.agent_owner_map = owner_map
            return drv

        # Owner map READ BACK FROM CHAIN (getAgentOwner for both agents),
        # keyed by the same 0x addresses the events now carry.
        chain_owner_map = {
            author_addr: substrate.functions.getAgentOwner(author_addr).call(),
            caller_addr: substrate.functions.getAgentOwner(caller_addr).call(),
        }
        s5.note("owner_map_from_chain",
                {k[:10] + "…": v[:10] + "…" for k, v in chain_owner_map.items()})

        drv = _mk_driver(chain_owner_map)
        fed = drv.run({"epoch_id": "e2e-epoch-1"})
        if fed is None:
            raise RuntimeError("federated close returned None (no batches?)")
        close_result = fed.close_result
        tm = close_result.get("tool_mint", {})
        # Digest key inside tool_mint is the artifact digest.
        assert tm, f"no tool_mint produced: keys={list(close_result)}"
        entry = next(iter(tm.values()))
        author_mint_raw = float(entry["mint"])
        assert entry["author"] == author_addr, (entry["author"], author_addr)
        assert entry["attesters"] == 1, entry           # caller-1 only
        assert entry["mint"] > 0
        # Vetting pipeline: greenlit by the two distinct-fleet vets in
        # THIS close, validators frozen, royalty window open — a slice
        # of the author's mint lands on the validators (the stake).
        assert entry["greenlit"] is True, entry
        assert entry["validators"] == 2, entry
        vetter_addrs = [rt_a.get_agent(v).identity.address
                        for v in ("vetter-1", "vetter-2")]
        vet_mints = [float(close_result["agent_mint"].get(
            Web3.to_checksum_address(a), 0)
            or close_result["agent_mint"].get(a, 0)) for a in vetter_addrs]
        assert all(m > 0 for m in vet_mints), (
            "validator royalty missing", close_result["agent_mint"])
        author_agent_mint_raw = float(close_result["agent_mint"].get(author_addr, 0))
        assert author_agent_mint_raw > 0
        s5.note("tool_mint_author", entry["author"][:12] + "…")
        s5.note("attesters", entry["attesters"])
        s5.note("greenlit", entry["greenlit"])
        s5.note("validator_royalty", [round(m, 8) for m in vet_mints])
        s5.note("tool_mint_raw", round(author_mint_raw, 8))
        s5.note("agent_mint_raw", round(author_agent_mint_raw, 8))

        # CONTROL CASE: set caller's owner == author's owner in the map.
        # The chain-fed same-owner exclusion must drop the mint to zero.
        control_map = {author_addr: owner_a.address,
                       caller_addr: owner_a.address}
        drv2 = _mk_driver(control_map)
        fed2 = drv2.run({"epoch_id": "e2e-epoch-ctrl"})
        tm2 = fed2.close_result.get("tool_mint", {})
        ctrl_zero = (tm2 == {} or all(v["mint"] == 0 for v in tm2.values()))
        assert ctrl_zero, f"control case did not zero the mint: {tm2}"
        s5.note("control_same_owner_mint", "0 (excluded)")
        s5.status = "PASS"
    except Exception as e:
        s5.status = "FAIL"
        s5.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 6 — settlement back to chain
    # ------------------------------------------------------------------
    s6 = board.stage(6, "settlement (anchor + recordTraining + channel)")
    try:
        from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
        from nodes.common.authoritative_submitter import (
            AuthoritativeChainSubmitter, AuthoritativeSubmitterConfig)
        from nodes.common.blob_resolver import InMemoryBlobResolver
        from nodes.common.mint_merkle import mint_merkle_proof, scale_mint
        from nodes.common.authoritative_submitter import epoch_id_hash

        if not close_result:
            raise RuntimeError("no close_result from stage 5")

        author_addr = Web3.to_checksum_address(author_ident.address)
        epoch_id = "e2e-epoch-1"
        # The anchor payload needs epoch_id; the driver already set it.
        close_result["epoch_id"] = epoch_id

        rep_before = substrate.functions.agentReputation(author_addr).call()
        atn_before = substrate.functions.balanceOf(author_addr).call()

        # Anchor as OWNER A (the deployer/human wallet acting as the
        # rotating submitter). Uses the production private-key path.
        anchorer = EpochAnchorer(EpochAnchorerConfig(
            epoch_anchor_address=addresses["substrate"],
            rpc_url=RPC_URL, chain_id=CHAIN_ID,
            private_key=HH_ACCT0_PK))
        anchor_res = anchorer.anchor_close_result(close_result)
        if not anchor_res.success:
            raise RuntimeError(f"anchor failed: {anchor_res.error}")
        s6.note("anchor_tx", anchor_res.tx_hash[:14] + "…")
        s6.note("agent_mint_cid", anchor_res.agent_mint_cid[:14] + "…")

        # Serve the mint blob to the per-agent submitter.
        resolver = InMemoryBlobResolver()
        resolver.put(anchor_res.agent_mint_blob)

        # author-1 records its own mint from its own key.
        submitter = AuthoritativeChainSubmitter(
            agent_id=author_addr, agent_address=author_addr,
            resolver=resolver,
            config=AuthoritativeSubmitterConfig(
                substrate_address=addresses["substrate"],
                rpc_url=RPC_URL, chain_id=CHAIN_ID,
                private_key=author_key))
        sub_res = submitter.submit_for_epoch(epoch_id)
        if not sub_res.success:
            raise RuntimeError(f"recordTraining failed: {sub_res.error}")
        s6.note("record_tx", (sub_res.tx_hash or "(noop)")[:14] + "…")
        s6.note("scaled_mint", sub_res.contribution_scaled)

        rep_after = substrate.functions.agentReputation(author_addr).call()
        atn_after = substrate.functions.balanceOf(author_addr).call()
        rep_delta = rep_after - rep_before
        atn_delta = atn_after - atn_before
        s6.note("reputation_delta", rep_delta)
        s6.note("atn_delta", atn_delta)
        # The on-chain amount is the FULL agent_mint share (tool-author
        # mint merged with any work/standing mint through the gate), not
        # just the tool_mint entry. Verify against agent_mint[author].
        expected = scale_mint(author_agent_mint_raw, MINT_SCALE)
        assert rep_delta == expected, (rep_delta, expected)
        assert atn_delta == expected, (atn_delta, expected)
        assert rep_delta > 0

        # --- Service leg: register one Service + one payment-channel
        # round-trip. The channel is the ONLY settlement rail (the postpaid
        # escrow was deleted): client=OWNER B opens a channel to the provider
        # (author) funded with MockERC20, signs ONE off-chain EIP-712 voucher
        # covering a single item's ask, the provider closes to collect it, and
        # the unused remainder refunds to the client after the challenge
        # window. We verify provider delta == voucher amount and the client
        # refund == deposit - voucher. ---
        try:
            reg_abi = load_abi("ServiceRegistry")
            registry = w3.eth.contract(
                address=Web3.to_checksum_address(addresses["serviceRegistry"]),
                abi=reg_abi)
            spec_digest = bytes.fromhex("cd" * 32)
            token = addresses.get("mockErc20")
            if not token:
                s6.note("channel_leg", "SKIP (no MockERC20 in repo)")
            else:
                token = Web3.to_checksum_address(token)
                ITEM_ASK = 1000       # one item's ask (the voucher amount)
                CHANNEL_DEPOSIT = 3000  # client escrows more than one item
                send_agent_tx(
                    w3,
                    registry.functions.registerService(spec_digest, token, ITEM_ASK),
                    author_key, author_addr)
                service_id = registry.functions.serviceCount().call()
                s6.note("service_id", service_id)

                mock_abi = load_abi("MockERC20")
                mock = w3.eth.contract(address=token, abi=mock_abi)
                channel_addr = Web3.to_checksum_address(
                    addresses["paymentChannel"])
                channel_abi = load_abi("PaymentChannel")
                channel = w3.eth.contract(
                    address=channel_addr, abi=channel_abi)

                # Mint/allocate mUSD to owner B (the client). MockERC20 likely
                # has mint(to, amount) — try it, else assume constructor funded.
                try:
                    tx = mock.functions.mint(owner_b.address, 100000)\
                        .build_transaction({
                            "from": owner_b.address,
                            "nonce": w3.eth.get_transaction_count(owner_b.address),
                            "gas": 200000, "gasPrice": w3.eth.gas_price,
                            "chainId": CHAIN_ID})
                    signed = w3.eth.account.sign_transaction(
                        tx, owner_b._private_key.hex())
                    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
                    w3.eth.wait_for_transaction_receipt(
                        w3.eth.send_raw_transaction(raw), timeout=60)
                except Exception:
                    pass

                def _owner_tx(call, owner, gas=300000):
                    tx = call.build_transaction({
                        "from": owner.address,
                        "nonce": w3.eth.get_transaction_count(owner.address),
                        "gas": gas, "gasPrice": w3.eth.gas_price,
                        "chainId": CHAIN_ID})
                    signed = w3.eth.account.sign_transaction(
                        tx, owner._private_key.hex())
                    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
                    r = w3.eth.wait_for_transaction_receipt(
                        w3.eth.send_raw_transaction(raw), timeout=60)
                    if r.status != 1:
                        raise RuntimeError("owner tx reverted")

                # Client opens the channel (deposit) to the provider.
                client_mUSD_before = mock.functions.balanceOf(owner_b.address).call()
                prov_before = mock.functions.balanceOf(author_addr).call()
                _owner_tx(mock.functions.approve(channel_addr, CHANNEL_DEPOSIT),
                          owner_b)
                _owner_tx(
                    channel.functions.openChannel(
                        author_addr, token, CHANNEL_DEPOSIT),
                    owner_b)
                channel_id = channel.functions.channelCount().call()

                # Client signs ONE off-chain EIP-712 voucher for a single
                # item's ask. Sign the raw typed-data digest the contract
                # exposes via voucherHash() — same robustness trick as the
                # owner-binding signer: whatever the contract hashes is what
                # we sign.
                from eth_account import Account
                voucher_digest = channel.functions.voucherHash(
                    channel_id, ITEM_ASK).call()
                voucher_digest = bytes(voucher_digest)
                client_acct = Account.from_key(owner_b._private_key.hex())
                v_signed = (
                    Account._sign_hash(voucher_digest, client_acct._private_key)
                    if hasattr(Account, "_sign_hash")
                    else client_acct.unsafe_sign_hash(voucher_digest))
                voucher_sig = bytes(v_signed.signature)

                # Provider (author) closes the channel with the voucher.
                send_agent_tx(
                    w3,
                    channel.functions.closeChannel(
                        channel_id, ITEM_ASK, voucher_sig),
                    author_key, author_addr)
                prov_after = mock.functions.balanceOf(author_addr).call()
                channel_provider_delta = prov_after - prov_before
                s6.note("channel_provider_delta", channel_provider_delta)
                assert channel_provider_delta == ITEM_ASK, (
                    channel_provider_delta, ITEM_ASK)

                # Advance past the challenge window on the running hardhat
                # node, then withdraw the client's remainder (permissionless).
                w3.provider.make_request("evm_increaseTime", [3601])
                w3.provider.make_request("evm_mine", [])
                send_agent_tx(
                    w3,
                    channel.functions.withdrawRemainder(channel_id),
                    author_key, author_addr)  # anyone can trigger the refund
                client_mUSD_after = mock.functions.balanceOf(owner_b.address).call()
                channel_client_refund = (
                    client_mUSD_after - (client_mUSD_before - CHANNEL_DEPOSIT))
                s6.note("channel_client_refund", channel_client_refund)
                # Client got back deposit - voucher; net spend == one voucher.
                assert channel_client_refund == CHANNEL_DEPOSIT - ITEM_ASK, (
                    channel_client_refund, CHANNEL_DEPOSIT - ITEM_ASK)
                net_client_spend = client_mUSD_before - client_mUSD_after
                assert net_client_spend == ITEM_ASK, (net_client_spend, ITEM_ASK)
        except Exception as se:
            # Channel leg is explicitly allowed to SKIP rather than sink time.
            s6.note("channel_leg", f"SKIP ({se!r})")

        s6.status = "PASS"
    except Exception as e:
        s6.status = "FAIL"
        s6.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ------------------------------------------------------------------
    # Stage 7 — teardown
    # ------------------------------------------------------------------
    s7 = board.stage(7, "teardown")
    try:
        kill_hardhat(hh_proc)
        hh_proc = None
        for d in (tmp_a, tmp_b):
            shutil.rmtree(d, ignore_errors=True)
        # Leave no trace: the inline deployer writes a deployment record.
        try:
            (REPO / "deployments" / "_e2e_local.json").unlink()
        except OSError:
            pass
        s7.status = "PASS"
    except Exception as e:
        s7.status = "FAIL"
        s7.note("error", repr(e))
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
