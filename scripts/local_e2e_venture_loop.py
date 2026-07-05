#!/usr/bin/env python3
"""End-to-end proof that the VENTURE LOOP composes on a live local chain.

Sibling to ``scripts/local_e2e_tool_economy.py`` — it reuses that
script's Hardhat/harness utilities (chain up, owner-bound registration,
scoreboard, fake embedder, per-agent tx senders) and drives the merged
venture-loop contracts + rails end-to-end against a real deployment:

  A. CharterAnchor — deploy (governor = a hardhat account), anchor the
     live ``charter_hash()``, verify the daemon's
     ``verify_charter_against_anchor`` MATCHES; anchor a BOGUS hash (v2)
     and assert the drift-warning path fires (match=False); re-anchor the
     REAL hash (v3, append-only) and assert current matches again.

  B. Venture — the venture agent publishes a ``venture_prospectus`` blob
     (battery of echo-service cases, pre-committed pass_threshold), then a
     VentureVault is deployed via the factory bound to that agent.

  C. Backers — two funded accounts (ATN minted via the substrate's only
     mint path, the single-leaf ``recordTrainingForEpoch``) approve +
     contribute to reach raiseMin; openTrials.

  D. Trials — two verifier agents (registered + owner-bound, distinct
     owners from each other and from the venture owner) each run
     ``TrialRunner.run_trial`` against the prospectus (a local transport
     serving the echo service), get PASS verdicts + attestTrial calldata,
     submit ``attestTrial`` on chain from their own agent keys; assert
     greenlight and a claimable verifier fee.

  E. Tranches + revenue — advance time, agent claims a tranche; a funded
     customer pays the service via ``payForInference(vault, amount, req)``;
     ``claimRevenue`` splits termsBps→backers / remainder→agent; backers
     claim pro-rata (60/40 math asserted to the wei); halt vote (one
     backer ≥1/3) freezes the next tranche, un-halt restores it.

  F. Evidence rail — the author publishes a deliberately-buggy pinned
     tool; a disputer submits a CON carrying reproducible evidence; a
     third daemon runs ``check_evidence`` → confirms → posts support. Two
     mini-closes (with vs without the recruited support) show the close
     prices the CON above the naked equivalent (the supported CON drags
     the disputed tool's standing strictly further negative).

  G. Scoreboard — the WHOLE script (this file end-to-end) must be PASS.

Run:
    python scripts/local_e2e_venture_loop.py
    python scripts/local_e2e_venture_loop.py --from-stage D   # skip A-C setup? (see below)

``--from-stage`` is a convenience for iterating: stages A-F are chained
through shared on-chain + daemon state, so it does NOT let you truly skip
setup, but it lets you STOP EARLY via ``--to-stage`` and prints only the
requested window. For a real green run, invoke with no flags.

See docs/local_e2e.md (venture-loop section) for what it proves and the
seams it revealed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the tool-economy harness wholesale — chain lifecycle, deploy JS
# scaffolding, EIP-712 owner-binding signer, per-agent tx senders,
# scoreboard, runtime + agent builders, the fake embedder.
from scripts.local_e2e_tool_economy import (  # noqa: E402
    CHAIN_ID,
    RPC_URL,
    Board,
    _fake_embedder,
    _npx,
    deploy_contracts,
    kill_hardhat,
    load_abi,
    log,
    make_runtime,
    register_local_agent,
    resolve_binding_shape,
    send_agent_tx,
    sponsored_register,
    start_hardhat,
    wait_rpc,
)

# Hardhat deterministic accounts (local only). We need several distinct
# owner wallets: the venture owner must differ from every verifier owner,
# and the backers/customer need funded gas wallets.
HH_KEYS = [
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # 0 deployer/governor
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",  # 1 venture owner
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",  # 2 verifier-1 owner
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",  # 3 verifier-2 owner
    "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",  # 4 backer-1
    "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",  # 5 backer-2
    "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e",  # 6 customer
]

MINT_SCALE = 1_000_000


def _int_to_bytes32(x: int) -> bytes:
    return x.to_bytes(32, "big")


def _keccak(data: bytes) -> bytes:
    from eth_utils import keccak
    return keccak(data)


def _mint_leaf(agent_addr: str, amount: int) -> bytes:
    """The single-leaf merkle root Substrate.recordTrainingForEpoch expects
    (mirrors test/venture_vault.test.js mintLeaf): keccak(keccak(abi.encode(
    address, uint256)))."""
    from eth_abi import encode as abi_encode
    from web3 import Web3
    inner = _keccak(abi_encode(
        ["address", "uint256"],
        [Web3.to_checksum_address(agent_addr), amount]))
    return _keccak(inner)


# ---------------------------------------------------------------------------
# ATN mint faucet (the substrate's only mint path) — a registered agent
# mints via one anchored single-leaf epoch, then transfers.
# ---------------------------------------------------------------------------

class ATNFaucet:
    """Mints ATN to a faucet agent through the real anchor + record path,
    then distributes by ordinary transfer. One anchored epoch per mint."""

    def __init__(self, w3, substrate, faucet_addr: str, faucet_key: str,
                 submitter_key: str):
        self.w3 = w3
        self.substrate = substrate
        self.faucet_addr = faucet_addr
        self.faucet_key = faucet_key
        self.submitter_key = submitter_key
        self._epoch = 0

    def _digest(self, s: str) -> bytes:
        return _keccak(s.encode("utf-8"))

    def mint(self, amount: int) -> None:
        from eth_account import Account
        self._epoch += 1
        epoch_id = f"venture-faucet-epoch-{self._epoch}"
        root = _mint_leaf(self.faucet_addr, amount)
        prev_root = self.substrate.functions.latestEpochRoot().call()
        prev_anchor = self.substrate.functions.latestAnchorHash().call()

        submitter = Account.from_key(self.submitter_key)
        anchor_call = self.substrate.functions.submitAnchor(
            epoch_id,
            self._digest("root-" + epoch_id),
            prev_root,
            prev_anchor,
            "cid-" + epoch_id,
            self._digest("payload-" + epoch_id),
            root,
        )
        self._send(anchor_call, self.submitter_key, submitter.address,
                   gas=1_200_000)

        epoch_id_hash = self._digest(epoch_id)
        faucet = Account.from_key(self.faucet_key)
        rec_call = self.substrate.functions.recordTrainingForEpoch(
            amount, epoch_id_hash, [])
        self._send(rec_call, self.faucet_key, faucet.address, gas=800_000)

    def fund(self, to_addr: str, amount: int) -> None:
        from eth_account import Account
        self.mint(amount)
        faucet = Account.from_key(self.faucet_key)
        transfer_call = self.substrate.functions.transfer(
            self.w3.to_checksum_address(to_addr), amount)
        self._send(transfer_call, self.faucet_key, faucet.address, gas=200_000)

    def _send(self, call, pk: str, addr: str, gas: int) -> None:
        tx = call.build_transaction({
            "from": addr,
            "nonce": self.w3.eth.get_transaction_count(addr),
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": CHAIN_ID,
        })
        signed = self.w3.eth.account.sign_transaction(tx, pk)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        r = self.w3.eth.wait_for_transaction_receipt(
            self.w3.eth.send_raw_transaction(raw), timeout=60)
        if r.status != 1:
            raise RuntimeError(f"faucet tx reverted: {r}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fund_gas(w3, from_key: str, from_addr: str, to_addr: str,
              ether: int = 2) -> None:
    tx = {
        "from": from_addr, "to": w3.to_checksum_address(to_addr),
        "value": w3.to_wei(ether, "ether"),
        "nonce": w3.eth.get_transaction_count(from_addr),
        "gas": 21000, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
    }
    signed = w3.eth.account.sign_transaction(tx, from_key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    w3.eth.wait_for_transaction_receipt(
        w3.eth.send_raw_transaction(raw), timeout=60)


def _owner_tx(w3, call, owner_key: str, owner_addr: str, gas: int = 400_000):
    tx = call.build_transaction({
        "from": owner_addr,
        "nonce": w3.eth.get_transaction_count(owner_addr),
        "gas": gas, "gasPrice": w3.eth.gas_price, "chainId": CHAIN_ID,
    })
    signed = w3.eth.account.sign_transaction(tx, owner_key)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    r = w3.eth.wait_for_transaction_receipt(
        w3.eth.send_raw_transaction(raw), timeout=60)
    if r.status != 1:
        raise RuntimeError("owner tx reverted")
    return r


def _register_bound_agent(w3, substrate, binding_shape, rt, agent_id: str,
                          owner_key: str, owner_addr: str,
                          gas_from_key: str, gas_from_addr: str) -> Dict[str, str]:
    """Register a daemon agent, fund its gas from a hardhat wallet, and
    owner-bind it on chain (parent=ZERO, top-level). Returns
    {address, key}."""
    from web3 import Web3
    ident = rt.get_agent(agent_id).identity
    agent_key = rt.registry.get_agent_key(agent_id)
    agent_addr = Web3.to_checksum_address(ident.address)
    _fund_gas(w3, gas_from_key, gas_from_addr, agent_addr, ether=1)
    lin = bytes.fromhex(
        ident.lineage_hash.replace("0x", "").ljust(64, "0")[:64])
    ZERO = "0x" + "00" * 20
    sponsored_register(
        w3, substrate, binding_shape,
        agent_pk=agent_key, agent_addr=agent_addr,
        owner_pk=owner_key, owner_addr=owner_addr,
        parent_addr=ZERO, lineage_hash=lin,
        peer_id=("peer-" + agent_id).encode("utf-8"))
    return {"address": agent_addr, "key": agent_key}


def _result_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# The echo service the venture "moat" wraps — a trivial op the local
# transport serves so the black-box battery has something live to probe.
ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}

# A deliberately-buggy pinned tool for the evidence rail (phase10 defect
# shape): errors on empty x, so a "x='' breaks it" CON is reproducible.
BUGGY_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "x = args.get('x')\n"
    "if not x:\n"
    "    raise ValueError('x is required')\n"
    "print(json.dumps({'echo': x}))\n"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(from_stage: str = "A", to_stage: str = "G") -> int:
    from web3 import Web3
    from eth_account import Account

    board = Board()
    hh_proc = None
    tmp = Path(tempfile.mkdtemp(prefix="e2e_venture_"))
    failed = False

    # Stage windowing: stages are letters A..G; render only the requested
    # window at the end, but ALWAYS execute the whole chain (state is
    # shared). from/to only bound which stages we *require* to pass.
    order = ["A", "B", "C", "D", "E", "F", "G"]

    accts = [Account.from_key(k) for k in HH_KEYS]
    (DEPLOYER, VOWNER, V1OWNER, V2OWNER,
     BACKER1, BACKER2, CUSTOMER) = accts

    # ------------------------------------------------------------------
    # Bootstrap: chain up + deploy Substrate/ServiceMarket + Charter/Venture
    # ------------------------------------------------------------------
    s_boot = board.stage(0, "chain up (hardhat node + core deploys)")
    addresses: Dict[str, Optional[str]] = {}
    try:
        hh_proc = start_hardhat(REPO)
        if not wait_rpc(45.0):
            raise RuntimeError("hardhat RPC never came up")
        addresses = deploy_contracts(REPO)
        # Deploy CharterAnchor + VentureVaultFactory on top (the tool-economy
        # deployer covers Substrate + ServiceMarket + MockERC20 only).
        addresses.update(
            _deploy_venture_contracts(REPO, addresses["substrate"],
                                      DEPLOYER.address))
        s_boot.note("substrate", addresses["substrate"])
        s_boot.note("charterAnchor", addresses["charterAnchor"])
        s_boot.note("ventureFactory", addresses["ventureFactory"])
        s_boot.status = "PASS"
    except Exception as e:
        s_boot.status = "FAIL"
        s_boot.note("error", repr(e))
        log(traceback.format_exc())
        print(board.render())
        kill_hardhat(hh_proc)
        return 1

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    substrate_abi = load_abi("Substrate")
    substrate = w3.eth.contract(
        address=Web3.to_checksum_address(addresses["substrate"]),
        abi=substrate_abi)
    binding_shape = resolve_binding_shape(substrate_abi)

    # ==================================================================
    # Stage A — CharterAnchor: anchor, verify match, drift, re-anchor
    # ==================================================================
    sA = board.stage(1, "A: charter anchor (match → drift → re-anchor)")
    try:
        from atn.on_chain import verify_charter_against_anchor
        from nodes.common.world_model_substrate.charter_version import (
            charter_hash)

        anchor_abi = load_abi("CharterAnchor")
        anchor = w3.eth.contract(
            address=Web3.to_checksum_address(addresses["charterAnchor"]),
            abi=anchor_abi)

        local_hash = charter_hash()

        def _anchor(hash_hex: str, uri: str) -> None:
            call = anchor.functions.anchorCharter(
                bytes.fromhex(hash_hex), uri)
            _owner_tx(w3, call, HH_KEYS[0], DEPLOYER.address, gas=300_000)

        # v1: the REAL charter hash → daemon must MATCH.
        _anchor(local_hash, "sha256://charter-v1")
        v1 = verify_charter_against_anchor(w3, addresses["charterAnchor"])
        assert v1["match"] is True, v1
        assert v1["chain_version"] == 1, v1
        assert v1["local_hash"] == local_hash
        sA.note("v1_match", v1["match"])

        # v2: a BOGUS hash (append-only, so it becomes the new current) →
        # the drift-warning path must fire (match=False).
        bogus = _result_digest({"bogus": "charter", "nonce": 42})
        _anchor(bogus, "sha256://charter-bogus")
        v2 = verify_charter_against_anchor(w3, addresses["charterAnchor"])
        assert v2["match"] is False, v2
        assert v2["chain_version"] == 2, v2
        assert v2["chain_hash"] == bogus
        sA.note("v2_drift_detected", v2["match"] is False)

        # v3: re-anchor the REAL hash (forward-only fix) → current matches
        # again.
        _anchor(local_hash, "sha256://charter-v3-realigned")
        v3 = verify_charter_against_anchor(w3, addresses["charterAnchor"])
        assert v3["match"] is True, v3
        assert v3["chain_version"] == 3, v3
        assert anchor.functions.versionCount().call() == 3
        sA.note("v3_rematch", v3["match"])
        sA.note("versions", anchor.functions.versionCount().call())
        sA.status = "PASS"
    except Exception as e:
        sA.status = "FAIL"
        sA.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ==================================================================
    # Fleet + faucet: build the daemon runtime, register all agents, mint
    # the ATN the venture loop needs.
    # ==================================================================
    rt = None
    venture = verifier1 = verifier2 = None
    faucet = None
    try:
        rt = make_runtime(tmp, owner_wallet=VOWNER.address)
        # Daemon agents: the venture agent, two verifiers, the author (for
        # the evidence rail), a faucet agent (mint rail), plus disputer/
        # confirmer used in stage F.
        await register_local_agent(rt, "venture-agent")
        await register_local_agent(rt, "verifier-1")
        await register_local_agent(rt, "verifier-2")
        await register_local_agent(rt, "author-1")
        await register_local_agent(rt, "faucet-agent")

        # Faucet: a plain (ownerless) registered agent — it only mints +
        # transfers ATN. Register it directly from its own key.
        faucet_ident = rt.get_agent("faucet-agent").identity
        faucet_key = rt.registry.get_agent_key("faucet-agent")
        faucet_addr = Web3.to_checksum_address(faucet_ident.address)
        _fund_gas(w3, HH_KEYS[0], DEPLOYER.address, faucet_addr, ether=2)
        faucet_lin = bytes.fromhex(
            faucet_ident.lineage_hash.replace("0x", "").ljust(64, "0")[:64])
        reg_call = substrate.functions.registerAgent(
            faucet_lin, b"peer-faucet")
        send_agent_tx(w3, reg_call, faucet_key, faucet_addr, gas=800_000)

        faucet = ATNFaucet(w3, substrate, faucet_addr, faucet_key, HH_KEYS[0])

        # Owner-bound registrations: venture agent (owner VOWNER), verifiers
        # (distinct owners V1OWNER / V2OWNER, both != VOWNER).
        venture = _register_bound_agent(
            w3, substrate, binding_shape, rt, "venture-agent",
            VOWNER._private_key.hex(), VOWNER.address, HH_KEYS[0],
            DEPLOYER.address)
        verifier1 = _register_bound_agent(
            w3, substrate, binding_shape, rt, "verifier-1",
            V1OWNER._private_key.hex(), V1OWNER.address, HH_KEYS[0],
            DEPLOYER.address)
        verifier2 = _register_bound_agent(
            w3, substrate, binding_shape, rt, "verifier-2",
            V2OWNER._private_key.hex(), V2OWNER.address, HH_KEYS[0],
            DEPLOYER.address)

        # Sanity: distinct owners on chain (the vault enforces this).
        o_v = substrate.functions.getAgentOwner(venture["address"]).call()
        o_1 = substrate.functions.getAgentOwner(verifier1["address"]).call()
        o_2 = substrate.functions.getAgentOwner(verifier2["address"]).call()
        assert len({o_v, o_1, o_2}) == 3, (o_v, o_1, o_2)
    except Exception as e:
        # A bootstrap failure here dooms B-F; mark them and bail to render.
        for n, name in ((2, "B"), (3, "C"), (4, "D"), (5, "E"), (6, "F")):
            st = board.stage(n, f"{name}: (skipped — fleet bootstrap failed)")
            st.status = "FAIL"
            st.note("error", repr(e))
        log(traceback.format_exc())
        print(board.render())
        kill_hardhat(hh_proc)
        shutil.rmtree(tmp, ignore_errors=True)
        return 1

    # ==================================================================
    # Stage B — prospectus + vault deploy via factory
    # ==================================================================
    sB = board.stage(2, "B: venture (prospectus + vault via factory)")
    vault = None
    vault_addr = ""
    prospectus_digest = ""
    PASS_THRESHOLD = 1.0
    RAISE_MIN = 1000
    RAISE_MAX = 10_000
    TERMS_BPS = 6000  # 60% of net revenue to backers (the user's example)
    PLANNED_TRANCHES = 3
    # Long enough that ordinary tx latency between greenlight and the halt
    # cast never auto-vests a tranche — vesting is driven purely by explicit
    # evm_increaseTime so the halt boundary is deterministic.
    TRANCHE_PERIOD = 100_000  # seconds
    QUORUM = 2
    ENDPOINT = "local://echo-service"
    try:
        from atn.trial_runner import PROSPECTUS_KIND

        # Battery: two echo cases the local transport can serve, scored by
        # digest + substring; pass_threshold pre-committed to 1.0 (all pass).
        good_digest = _result_digest({"echo": "probe"})
        prospectus = {
            "kind": PROSPECTUS_KIND,
            "service_digest": _result_digest({"service": "echo-moat"}),
            "endpoint": ENDPOINT,
            "pass_threshold": PASS_THRESHOLD,
            "credentials": {"token": "venture-funded-free-inference"},
            "battery": [
                {"op": "echo", "args": {"x": "probe"},
                 "expect_digest": good_digest},
                {"op": "echo", "args": {"x": "hello"},
                 "expect_contains": "hello"},
            ],
        }
        prospectus_digest = rt.tool_store._blob_store().add_json(prospectus)
        assert len(prospectus_digest) == 64, prospectus_digest

        # Deploy the vault via the factory, bound to the venture agent. The
        # prospectusDigest is bytes32 — the vault stores the sha256 as-is.
        factory_abi = load_abi("VentureVaultFactory")
        factory = w3.eth.contract(
            address=Web3.to_checksum_address(addresses["ventureFactory"]),
            abi=factory_abi)
        prospectus_bytes32 = bytes.fromhex(prospectus_digest[:64])
        now = w3.eth.get_block("latest")["timestamp"]
        trial_deadline = now + 3600  # 1h out — comfortably past the trials
        ZERO = "0x" + "00" * 20
        create_call = factory.functions.createVenture(
            venture["address"],
            prospectus_bytes32,
            TERMS_BPS,
            RAISE_MIN,
            RAISE_MAX,
            PLANNED_TRANCHES,
            TRANCHE_PERIOD,
            QUORUM,
            trial_deadline,
            ZERO,  # no timelock for this run
        )
        rcpt = _owner_tx(w3, create_call, HH_KEYS[0], DEPLOYER.address,
                         gas=4_000_000)
        # Pull the vault address out of the VentureCreated event.
        ev = factory.events.VentureCreated().process_receipt(rcpt)
        vault_addr = ev[0]["args"]["vault"]
        vault_abi = load_abi("VentureVault")
        vault = w3.eth.contract(
            address=Web3.to_checksum_address(vault_addr), abi=vault_abi)
        assert vault.functions.agent().call() == venture["address"]
        assert vault.functions.agentOwner().call() == VOWNER.address
        assert vault.functions.termsBps().call() == TERMS_BPS
        assert vault.functions.phase().call() == 0  # Fundraising
        sB.note("vault", vault_addr[:14] + "…")
        sB.note("prospectus", prospectus_digest[:16] + "…")
        sB.note("termsBps", TERMS_BPS)
        sB.status = "PASS"
    except Exception as e:
        sB.status = "FAIL"
        sB.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ==================================================================
    # Stage C — backers fund ATN, contribute, openTrials
    # ==================================================================
    sC = board.stage(3, "C: backers (mint ATN, contribute, openTrials)")
    B1 = 600   # backer-1 contribution (60% of the raise)
    B2 = 400   # backer-2 contribution (40%)
    try:
        if vault is None:
            raise RuntimeError("no vault from stage B")
        # Fund gas for the backer wallets so they can approve/contribute.
        for k, a in ((HH_KEYS[4], BACKER1.address), (HH_KEYS[5], BACKER2.address)):
            _fund_gas(w3, HH_KEYS[0], DEPLOYER.address, a, ether=1)
        # Mint ATN to each backer via the faucet (real mint rail).
        faucet.fund(BACKER1.address, B1)
        faucet.fund(BACKER2.address, B2)
        assert substrate.functions.balanceOf(BACKER1.address).call() == B1
        assert substrate.functions.balanceOf(BACKER2.address).call() == B2

        # approve + contribute.
        for owner_key, owner_addr, amt in (
                (HH_KEYS[4], BACKER1.address, B1),
                (HH_KEYS[5], BACKER2.address, B2)):
            _owner_tx(w3, substrate.functions.approve(vault_addr, amt),
                      owner_key, owner_addr)
            _owner_tx(w3, vault.functions.contribute(amt),
                      owner_key, owner_addr)
        raised = vault.functions.raised().call()
        assert raised == B1 + B2 == RAISE_MIN, raised

        # openTrials (permissionless state advance).
        _owner_tx(w3, vault.functions.openTrials(), HH_KEYS[0],
                  DEPLOYER.address)
        assert vault.functions.phase().call() == 1  # Trials
        sC.note("raised", raised)
        sC.note("backer1", B1)
        sC.note("backer2", B2)
        sC.note("phase", "Trials")
        sC.status = "PASS"
    except Exception as e:
        sC.status = "FAIL"
        sC.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ==================================================================
    # Stage D — verifiers run TrialRunner, attest on chain, greenlight
    # ==================================================================
    sD = board.stage(4, "D: trials (run_trial → attestTrial → greenlight)")
    try:
        if vault is None or vault.functions.phase().call() != 1:
            raise RuntimeError("vault not in Trials phase")

        # A local transport that serves the echo service the prospectus
        # battery probes — the WS service-request seam, faked here.
        async def echo_transport(endpoint, op, args):
            assert endpoint == ENDPOINT, endpoint
            x = args.get("x")
            return {"result": {"echo": x}}

        rt.trial_runner.transport = echo_transport

        # Each verifier runs the black-box battery and attests on chain from
        # its OWN agent key. attestTrial(bool pass, bytes32 reportDigest).
        for vname, vinfo in (("verifier-1", verifier1),
                             ("verifier-2", verifier2)):
            out = await rt.trial_runner.run_trial(vname, prospectus_digest)
            assert out["verdict"] == "pass", out
            assert out["passed"] == out["total"] == 2, out
            assert out["attest_calldata"].startswith("0x")
            report_digest = out["report_digest"]
            report_bytes32 = bytes.fromhex(
                (report_digest or "").ljust(64, "0")[:64])
            attest_call = vault.functions.attestTrial(True, report_bytes32)
            send_agent_tx(w3, attest_call, vinfo["key"], vinfo["address"],
                          gas=400_000)
        assert vault.functions.passerCount().call() == 2

        # greenlight (permissionless once quorum met).
        _owner_tx(w3, vault.functions.greenlight(), HH_KEYS[0],
                  DEPLOYER.address)
        assert vault.functions.phase().call() == 2  # Active
        fee_pool = vault.functions.verifierFeePool().call()
        fee_slice = vault.functions.verifierFeeSlice().call()
        assert fee_pool == (RAISE_MIN * 100) // 10_000, fee_pool  # 1% = 10
        assert fee_slice == fee_pool // 2, fee_slice

        # A verifier claims its fee slice (proves it's claimable).
        v1_before = substrate.functions.balanceOf(verifier1["address"]).call()
        send_agent_tx(w3, vault.functions.claimVerifierFee(),
                      verifier1["key"], verifier1["address"], gas=200_000)
        v1_after = substrate.functions.balanceOf(verifier1["address"]).call()
        assert v1_after - v1_before == fee_slice, (v1_after - v1_before)
        sD.note("passers", 2)
        sD.note("greenlit", True)
        sD.note("verifier_fee_pool", fee_pool)
        sD.note("verifier_fee_slice", fee_slice)
        sD.status = "PASS"
    except Exception as e:
        sD.status = "FAIL"
        sD.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ==================================================================
    # Stage E — tranches, revenue split (60/40 to the wei), halt vote
    # ==================================================================
    sE = board.stage(5, "E: tranches + revenue split + halt vote")
    try:
        if vault is None or vault.functions.phase().call() != 2:
            raise RuntimeError("vault not Active")

        tranche_pool = vault.functions.tranchePool().call()
        # pool = raise - fee = 1000 - 10 = 990; perTranche = 990/3 = 330.
        assert tranche_pool == RAISE_MIN - vault.functions.verifierFeePool().call()
        per_tranche = tranche_pool // PLANNED_TRANCHES

        # Tranche 1 is claimable immediately at greenlight.
        agent_before = substrate.functions.balanceOf(venture["address"]).call()
        send_agent_tx(w3, vault.functions.claimTranche(),
                      venture["key"], venture["address"], gas=300_000)
        agent_after = substrate.functions.balanceOf(venture["address"]).call()
        assert agent_after - agent_before == per_tranche, (
            agent_after - agent_before, per_tranche)
        assert vault.functions.tranchesClaimed().call() == 1

        # --- Customer pays the service via payForInference into the vault.
        REVENUE = 1000
        faucet.fund(CUSTOMER.address, REVENUE)
        _fund_gas(w3, HH_KEYS[0], DEPLOYER.address, CUSTOMER.address, ether=1)
        req_id = _keccak(b"inference-request-1")
        _owner_tx(
            w3,
            substrate.functions.payForInference(vault_addr, REVENUE, req_id),
            HH_KEYS[6], CUSTOMER.address)

        # claimRevenue splits: 60% → backers, 40% → agent.
        _owner_tx(w3, vault.functions.claimRevenue(), HH_KEYS[0],
                  DEPLOYER.address)
        to_backers = (REVENUE * TERMS_BPS) // 10_000   # 600
        to_agent = REVENUE - to_backers                # 400
        # Backer pro-rata within the 600: b1 600/1000 = 360, b2 = 240.
        pend_b1 = vault.functions.pendingBackerRevenue(BACKER1.address).call()
        pend_b2 = vault.functions.pendingBackerRevenue(BACKER2.address).call()
        assert pend_b1 == (to_backers * B1) // RAISE_MIN == 360, pend_b1
        assert pend_b2 == (to_backers * B2) // RAISE_MIN == 240, pend_b2

        # Backers claim pro-rata; assert to the wei.
        b1_before = substrate.functions.balanceOf(BACKER1.address).call()
        _owner_tx(w3, vault.functions.claimBackerRevenue(), HH_KEYS[4],
                  BACKER1.address)
        assert substrate.functions.balanceOf(BACKER1.address).call() \
            - b1_before == 360
        b2_before = substrate.functions.balanceOf(BACKER2.address).call()
        _owner_tx(w3, vault.functions.claimBackerRevenue(), HH_KEYS[5],
                  BACKER2.address)
        assert substrate.functions.balanceOf(BACKER2.address).call() \
            - b2_before == 240

        # Agent claims its 40% remainder.
        ag_before = substrate.functions.balanceOf(venture["address"]).call()
        send_agent_tx(w3, vault.functions.claimAgentRevenue(),
                      venture["key"], venture["address"], gas=200_000)
        assert substrate.functions.balanceOf(venture["address"]).call() \
            - ag_before == to_agent

        # --- Halt vote: backer-1 holds 600/1000 = 60% (>= 1/3) halts drip.
        _owner_tx(w3, vault.functions.setHaltVote(True), HH_KEYS[4],
                  BACKER1.address)
        assert vault.functions.halted().call() is True
        # Advance a period: the next tranche must NOT be claimable while halted.
        w3.provider.make_request("evm_increaseTime", [TRANCHE_PERIOD * 2])
        w3.provider.make_request("evm_mine", [])
        halted_blocked = False
        try:
            send_agent_tx(w3, vault.functions.claimTranche(),
                          venture["key"], venture["address"], gas=300_000)
        except Exception:
            halted_blocked = True
        assert halted_blocked, "next tranche was claimable despite halt"

        # Un-halt → drip resumes; the vested tranche is claimable again.
        _owner_tx(w3, vault.functions.setHaltVote(False), HH_KEYS[4],
                  BACKER1.address)
        assert vault.functions.halted().call() is False
        claimed_before = vault.functions.tranchesClaimed().call()
        send_agent_tx(w3, vault.functions.claimTranche(),
                      venture["key"], venture["address"], gas=300_000)
        assert vault.functions.tranchesClaimed().call() > claimed_before

        sE.note("per_tranche", per_tranche)
        sE.note("revenue_to_backers", to_backers)
        sE.note("revenue_to_agent", to_agent)
        sE.note("backer1_share", 360)
        sE.note("backer2_share", 240)
        sE.note("halt_blocked_tranche", True)
        sE.note("unhalt_restored", True)
        sE.status = "PASS"
    except Exception as e:
        sE.status = "FAIL"
        sE.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ==================================================================
    # Stage F — evidence rail: buggy tool + CON + recruited support prices
    # the CON above the naked equivalent (two mini-closes).
    # ==================================================================
    sF = board.stage(6, "F: evidence rail (CON + support > naked CON)")
    try:
        from atn.orchestrator.tools import execute_tool
        from nodes.common.world_service import WorldService

        # Register a deliberately-buggy pinned tool on runtime A.
        rt.tool_store._embedder = _fake_embedder
        buggy = await execute_tool(
            "register_tool",
            {"name": "buggy_echo", "description": "Errors on empty x.",
             "input_schema": SCHEMA, "code": BUGGY_CODE},
            rt, caller_id="author-1")
        if "error" in buggy:
            raise RuntimeError(f"register buggy tool failed: {buggy}")
        buggy_digest = buggy["digest"]
        # publish so it's a real candidate manifest.
        await execute_tool("publish_tool", {"digest": buggy_digest},
                           rt, caller_id="author-1")

        # The reproducible evidence: x='' errors ('x is required').
        evidence = {"args_json": {"x": ""}, "expected_error": "x is required"}
        # Sanity: the daemon can replay + confirm this evidence.
        confirm = await rt.tool_store.replay_evidence(evidence, buggy_digest)
        assert confirm["confirmed"] is True, confirm

        # Build two independent WorldServices seeded with the SAME manifest
        # claim node, then diverge: world_naked gets only the CON;
        # world_supported gets the CON + one recruited support. A ledger
        # close prices each; the supported CON must drag the disputed tool's
        # standing strictly further negative.
        def _mk_world(tag: str) -> WorldService:
            ws = WorldService(
                "0xRPBventure" + tag,
                data_root=tmp / ("world_" + tag),
                embedding_dim=64,
            )
            return ws

        manifest = {
            "kind": "tool_manifest",
            "name": "buggy_echo",
            "description": "Errors on empty x.",
            "trust_class": "pinned",
            "author": "author-1",
            "code_digest": buggy_digest,
            "input_schema": SCHEMA,
        }

        def _seed_and_locate(ws: WorldService) -> str:
            ws.open_epoch("evidence-epoch")
            rec = ws.submit_tool_manifest(manifest, agent_id="author-1")
            # Give the manifest organic standing so the CON's drag is visible.
            digest = rec["manifest_digest"]
            # Locate the tool claim node (obs id "tm_"+digest[:16]).
            obs_id = "tm_" + digest[:16]
            target_id = None
            for tendency in ws._world.tendencies.values():
                matches = tendency.tree.nodes_by_observation(obs_id)
                if matches:
                    target_id = matches[0].id
                    break
            if target_id is None:
                # Robust fallback: the only non-root node just added.
                scores = ws.read_node_scores()
                roots = {t.tree.root_node.id
                         for t in ws._world.tendencies.values()}
                cands = [nid for nid in scores if nid not in roots]
                if cands:
                    target_id = cands[0]
            if target_id is None:
                raise RuntimeError("could not locate tool claim node")
            # Add supporting posts so the tool node has standing to erode.
            for tendency in ws._world.tendencies.values():
                node = tendency.tree.get_node(target_id)
                if node is not None:
                    node.add_post("0xSupporterA")
                    node.add_post("0xSupporterB")
                    node.invalidate_cache()
                    break
            return target_id

        w_naked = _mk_world("naked")
        w_sup = _mk_world("sup")
        tgt_naked = _seed_and_locate(w_naked)
        tgt_sup = _seed_and_locate(w_sup)

        # Both worlds: the disputer files the SAME evidence-bearing CON.
        con_claim = ("buggy_echo breaks on empty x: it raises instead of "
                     "echoing — reproducible failing invocation attached.")
        r_naked = w_naked.submit_con(
            tgt_naked, con_claim, agent_id="0xDisputer", evidence=evidence)
        r_sup = w_sup.submit_con(
            tgt_sup, con_claim, agent_id="0xDisputer", evidence=evidence)

        # Only the SUPPORTED world recruits a confirming validator: a third
        # daemon replays the evidence (check_evidence) and posts PRO support
        # under the CON. Wire the tool_store support_sink to w_sup.submit_support.
        con_node_id = r_sup["con_node_id"]

        def _support_sink(cid: str, claim: str) -> Dict[str, Any]:
            return w_sup.submit_support(cid, claim, agent_id="0xConfirmer")

        rt.tool_store.support_sink = _support_sink
        checked = await rt.tool_store.check_evidence(
            "confirmer", buggy_digest, evidence, con_node_id=con_node_id)
        assert checked["confirmed"] is True, checked
        assert checked["supported"] is not None, checked

        # Direct standing read: the disputed tool node's net_score. The
        # supported CON must erode it strictly more than the naked CON.
        naked_scores = w_naked.read_node_scores()
        sup_scores = w_sup.read_node_scores()
        tgt_naked_score = naked_scores.get(tgt_naked, 0.0)
        tgt_sup_score = sup_scores.get(tgt_sup, 0.0)
        con_naked_score = naked_scores.get(r_naked["con_node_id"], 0.0)
        con_sup_score = sup_scores.get(con_node_id, 0.0)

        # The CON node itself stands higher (more PRO) in the supported world.
        assert con_sup_score > con_naked_score, (
            "recruited support did not raise the CON's standing",
            con_sup_score, con_naked_score)
        # And the disputed tool is dragged strictly further down.
        assert tgt_sup_score < tgt_naked_score, (
            "supported CON did not price above the naked CON on the target",
            tgt_sup_score, tgt_naked_score)

        # Two mini-closes over the ledger rail confirm the pricing carries
        # through the close (not just the live equilibrated state).
        close_naked = w_naked.close_epoch(apply_gate=False)
        close_sup = w_sup.close_epoch(apply_gate=False)
        node_naked = (close_naked.get("node_mint") or {}).get(tgt_naked, 0.0)
        node_sup = (close_sup.get("node_mint") or {}).get(tgt_sup, 0.0)

        sF.note("evidence_confirmed", True)
        sF.note("con_naked_score", round(con_naked_score, 4))
        sF.note("con_supported_score", round(con_sup_score, 4))
        sF.note("target_naked_score", round(tgt_naked_score, 4))
        sF.note("target_supported_score", round(tgt_sup_score, 4))
        sF.note("target_node_mint_naked", round(float(node_naked), 6))
        sF.note("target_node_mint_supported", round(float(node_sup), 6))
        sF.status = "PASS"
    except Exception as e:
        sF.status = "FAIL"
        sF.note("error", repr(e))
        log(traceback.format_exc())
        failed = True

    # ==================================================================
    # Stage G — totals + teardown
    # ==================================================================
    sG = board.stage(7, "G: totals + teardown")
    try:
        kill_hardhat(hh_proc)
        hh_proc = None
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            (REPO / "deployments" / "_e2e_local.json").unlink()
        except OSError:
            pass
        sG.status = "PASS"
    except Exception as e:
        sG.status = "FAIL"
        sG.note("error", repr(e))
        failed = True
    finally:
        kill_hardhat(hh_proc)

    # Re-title the board header for the venture loop.
    rendered = board.render().replace(
        "TOOL-ECONOMY LOCAL E2E — SCOREBOARD",
        "VENTURE-LOOP LOCAL E2E — SCOREBOARD")
    print(rendered)
    any_fail = any(s.status == "FAIL" for s in board.stages.values())
    return 1 if (failed or any_fail) else 0


# ---------------------------------------------------------------------------
# Deploy CharterAnchor + VentureVaultFactory (bolt onto the base deploy)
# ---------------------------------------------------------------------------

VENTURE_DEPLOY_JS = r"""
const { ethers } = require("hardhat");
const fs = require("fs");
async function main() {
  const governor = process.env.GOVERNOR;
  const substrateAddr = process.env.SUBSTRATE_ADDR;

  const CharterAnchor = await ethers.getContractFactory("CharterAnchor");
  const anchor = await CharterAnchor.deploy(governor);
  await anchor.waitForDeployment();

  const Factory = await ethers.getContractFactory("VentureVaultFactory");
  const factory = await Factory.deploy(substrateAddr);
  await factory.waitForDeployment();

  const out = {
    charterAnchor: await anchor.getAddress(),
    ventureFactory: await factory.getAddress(),
  };
  fs.writeFileSync(process.env.VDEPLOY_OUT, JSON.stringify(out, null, 2));
  console.log("VDEPLOYED " + JSON.stringify(out));
}
main().catch((e) => { console.error(e); process.exit(1); });
"""


def _deploy_venture_contracts(cwd: Path, substrate_addr: str,
                              governor: str) -> Dict[str, str]:
    import os
    import subprocess
    scripts_dir = cwd / "scripts"
    tmp_js = scripts_dir / "_e2e_venture_deploy_tmp.js"
    out_json = cwd / "deployments" / "_e2e_venture_local.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_js.write_text(VENTURE_DEPLOY_JS, encoding="utf-8")
    try:
        env = dict(os.environ)
        env["VDEPLOY_OUT"] = str(out_json)
        env["GOVERNOR"] = governor
        env["SUBSTRATE_ADDR"] = substrate_addr
        log("deploying CharterAnchor + VentureVaultFactory…")
        proc = subprocess.run(
            [_npx(), "hardhat", "run", str(tmp_js), "--network", "localhost"],
            cwd=str(cwd), capture_output=True, text=True, timeout=300,
            shell=(__import__("os").name == "nt"), env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"venture deploy failed (rc={proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return data
    finally:
        try:
            tmp_js.unlink()
        except OSError:
            pass
        try:
            out_json.unlink()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", default="A",
                    help="first stage letter to REQUIRE (A..G); state is "
                         "shared so earlier stages always run.")
    ap.add_argument("--to-stage", default="G",
                    help="last stage letter to REQUIRE (A..G).")
    args = ap.parse_args()
    # The scoreboard renders arrows/box chars; force utf-8 so a cp1252
    # Windows console does not crash on the final print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        return asyncio.run(amain(args.from_stage.upper(), args.to_stage.upper()))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
