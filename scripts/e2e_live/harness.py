#!/usr/bin/env python3
"""Live E2E harness: a REAL isolated daemon on a local chain, for app tests.

One command up, one down, one status:

    python scripts/e2e_live/harness.py up       # chain + stub + daemon + fixtures
    python scripts/e2e_live/harness.py status   # exit 0 iff everything answers
    python scripts/e2e_live/harness.py down     # kill everything we started

``up`` leaves three detached processes running and records them in
``<state dir>/state.json``:

  1. a hardhat node on 127.0.0.1:E2E_LIVE_RPC_PORT (default 18545, chain 1337)
     with Substrate, ServiceRegistry, PaymentChannel, CharterAnchor and
     VentureVaultFactory deployed from the compiled artifacts (web3.py, signed
     by hardhat account 0 — no ``hardhat run``, no deployments/*.json written);
  2. the deterministic stub (``stub_server.py``) on E2E_LIVE_STUB_PORT (default
     18080): the daemon's model provider AND the ``ATN_REGISTRY_URL`` target;
  3. a real ``python -m atn`` daemon whose home is ``<state dir>/home``
     (USERPROFILE/HOME redirect + KEYSTORE_DIR), privileged local listener on
     E2E_LIVE_WS_PORT (default 7799), remote and integration listeners off.

Then it seeds fixtures over the daemon's own WS surface (the same frames the
app sends): three agents, an owner-authored pinned tool, the first agent
registered on the local Substrate (owner-bound to hardhat account 1, the
daemon's owner wallet), and a tool-backed service listed on the local
ServiceRegistry, signed by that agent's own key. A deterministic agent turn
proves the stub provider answers through the real execution engine.

Safety rails (see README.md): refuses ports 7700/7701, refuses a state dir
inside the real ``~/.atn``, refuses any port already in use, never kills by
port (only the PIDs it spawned), pins every chain field explicitly in the
daemon's config.yaml, and asserts the RPC is chain 1337 on loopback before
deploying anything.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the cross-daemon E2E's proven pieces (isolated config writer, home
# redirect env, the WS client, owner-bound registration) rather than fork them.
import local_e2e_cross_daemon_inference as xd  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

WS_PORT = int(os.environ.get("E2E_LIVE_WS_PORT", "7799"))
RPC_PORT = int(os.environ.get("E2E_LIVE_RPC_PORT", "18545"))
STUB_PORT = int(os.environ.get("E2E_LIVE_STUB_PORT", "18080"))
STATE_DIR = Path(os.environ.get("E2E_LIVE_DIR") or
                 (Path(tempfile.gettempdir()) / "atn_e2e_live")).resolve()

RPC_URL = f"http://127.0.0.1:{RPC_PORT}"
WS_URL = f"ws://127.0.0.1:{WS_PORT}"
STUB_URL = f"http://127.0.0.1:{STUB_PORT}"
CHAIN_ID = 1337           # hardhat.config.js network "hardhat"
FORBIDDEN_WS_PORTS = {7700, 7701, 7710}   # the user's daemon listeners

# Hardhat's well-known mnemonic accounts. LOCAL CHAIN ONLY.
HH_ACCOUNTS = [
    ("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
     "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"),
    ("0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
     "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"),
    ("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
     "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"),
]
DEPLOYER = HH_ACCOUNTS[0]   # deploys + funds gas
OWNER = HH_ACCOUNTS[1]      # the daemon's owner wallet
SPARE = HH_ACCOUNTS[2]      # untouched; free for tests (e.g. a buyer)

STUB_PROVIDER = "e2e_stub"
STUB_MODEL = "echo-1"

AGENTS = [
    {"id": "e2e-alpha", "name": "E2E Alpha",
     "description": "Live-harness fixture: on-chain agent, sells a service."},
    {"id": "e2e-beta", "name": "E2E Beta",
     "description": "Live-harness fixture: plain local agent."},
    {"id": "e2e-alpha-child", "name": "E2E Alpha Child", "parent": "e2e-alpha",
     "description": "Live-harness fixture: child of E2E Alpha."},
]
TOOL_NAME = "e2e_echo_upper"
SERVICE_NAME = "E2E Upper Service"
SERVICE_ASK = 1000
BOOT_TIMEOUT = 180.0


def log(msg: str) -> None:
    print(xd._safe(f"[e2e-live] {msg}"), flush=True)


# ---------------------------------------------------------------------------
# State + process helpers
# ---------------------------------------------------------------------------

def state_path() -> Path:
    return STATE_DIR / "state.json"


def load_state() -> Dict[str, Any]:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = state_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(state_path())


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != "zombie"
    except Exception:
        pass
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(pid: int) -> None:
    """Kill one recorded PID and its children. Never by port: a port can be
    held by the user's own daemon; a PID we recorded cannot."""
    if not pid_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
    else:
        try:
            import psutil
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.kill()
            proc.kill()
        except Exception:
            pass


def spawn_detached(argv: List[str], *, cwd: Path, env: Dict[str, str],
                   log_path: Path, append: bool = False) -> int:
    """Start a process that outlives this script; returns its PID.

    stdin is DEVNULL: the daemon detects a non-tty stdin and runs headless
    (atn/cli.py) instead of exiting on EOF.
    """
    fh = open(log_path, "a" if append else "w", encoding="utf-8",
              errors="replace")
    kwargs: Dict[str, Any] = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                  stdout=fh, stderr=subprocess.STDOUT)
    if os.name == "nt":
        base = (subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW)
        try:
            # Break away from the caller's job object so a terminal/CI job
            # teardown does not take the harness down mid-test.
            proc = subprocess.Popen(argv, creationflags=base | 0x01000000,
                                    **kwargs)
        except OSError:
            proc = subprocess.Popen(argv, creationflags=base, **kwargs)
    else:
        proc = subprocess.Popen(argv, start_new_session=True, **kwargs)
    fh.close()
    return proc.pid


def wait_until(pred, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def http_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(f"    | {ln}" for ln in lines[-n:])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight() -> None:
    if WS_PORT in FORBIDDEN_WS_PORTS:
        raise SystemExit(f"refusing E2E_LIVE_WS_PORT={WS_PORT}: that is a port "
                         "the user's real daemon listens on")
    real_atn = (Path(os.path.expanduser("~")) / ".atn").resolve()
    if STATE_DIR == real_atn or real_atn in STATE_DIR.parents:
        raise SystemExit(f"refusing state dir {STATE_DIR}: inside the real "
                         f"{real_atn}")
    st = load_state()
    live = [k for k in ("hardhat_pid", "stub_pid", "daemon_pid")
            if pid_alive(int(st.get(k) or 0))]
    if live:
        raise SystemExit(f"harness already up ({', '.join(live)} alive in "
                         f"{state_path()}); run `down` first")
    busy = [p for p in (WS_PORT, WS_PORT + 1, RPC_PORT, STUB_PORT)
            if port_in_use(p)]
    if busy:
        raise SystemExit(f"ports already in use: {busy}. Pick others with "
                         "E2E_LIVE_WS_PORT / E2E_LIVE_RPC_PORT / "
                         "E2E_LIVE_STUB_PORT")


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def start_chain(state: Dict[str, Any]) -> None:
    node = shutil.which("node")
    cli = REPO / "node_modules" / "hardhat" / "internal" / "cli" / "cli.js"
    if not node or not cli.exists():
        raise RuntimeError("need node + node_modules/hardhat (run npm install)")
    # Compile is a no-op when artifacts are current; deploy reads them.
    log("hardhat compile (cached when current)...")
    rc = subprocess.run([node, str(cli), "compile", "--quiet"], cwd=str(REPO),
                        capture_output=True, text=True, timeout=600)
    if rc.returncode != 0:
        raise RuntimeError(f"hardhat compile failed:\n{rc.stdout}\n{rc.stderr}")
    pid = spawn_detached(
        [node, str(cli), "node", "--hostname", "127.0.0.1",
         "--port", str(RPC_PORT)],
        cwd=REPO, env=dict(os.environ), log_path=STATE_DIR / "hardhat.log")
    state["hardhat_pid"] = pid
    save_state(state)
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not wait_until(lambda: w3.is_connected() and w3.eth.block_number >= 0,
                      90.0):
        raise RuntimeError(f"hardhat RPC never came up at {RPC_URL}\n"
                           f"{tail(STATE_DIR / 'hardhat.log')}")
    chain_id = w3.eth.chain_id
    if chain_id != CHAIN_ID:
        raise RuntimeError(f"RPC at {RPC_URL} is chain {chain_id}, expected "
                           f"{CHAIN_ID}: refusing to deploy")
    log(f"hardhat node up at {RPC_URL} (pid {pid}, chain {chain_id})")


def _artifact(name: str, sol: str) -> Dict[str, Any]:
    p = REPO / "artifacts" / "contracts" / "core" / f"{sol}.sol" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def deploy(w3) -> Dict[str, str]:
    """Deploy the contract set the local E2E scripts use, signed by hardhat
    account 0. Same constructor args as local_e2e_tool_economy.py /
    local_e2e_venture_loop.py."""
    from eth_account import Account
    acct = Account.from_key(DEPLOYER[1])
    zero = "0x" + "00" * 20

    def _deploy(name: str, sol: str, *args: Any) -> str:
        art = _artifact(name, sol)
        c = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        tx = c.constructor(*args).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 15_000_000, "gasPrice": w3.eth.gas_price,
            "chainId": CHAIN_ID,
        })
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        rcpt = w3.eth.wait_for_transaction_receipt(
            w3.eth.send_raw_transaction(raw), timeout=120)
        if rcpt.status != 1 or not rcpt.contractAddress:
            raise RuntimeError(f"deploy {name} failed: {rcpt}")
        return rcpt.contractAddress

    sub = _deploy("Substrate", "Substrate", acct.address, zero, zero)
    out = {
        "substrate": sub,
        "service_registry": _deploy("ServiceRegistry", "ServiceMarket", sub),
        "payment_channel": _deploy("PaymentChannel", "ServiceMarket", sub, 3600),
        "charter_anchor": _deploy("CharterAnchor", "CharterAnchor", acct.address),
        "venture_vault_factory": _deploy("VentureVaultFactory", "VentureVault",
                                         sub),
    }
    log("deployed " + ", ".join(f"{k}={v}" for k, v in out.items()))
    return out


# ---------------------------------------------------------------------------
# Stub + daemon
# ---------------------------------------------------------------------------

def write_registry(contracts: Dict[str, str]) -> Path:
    path = STATE_DIR / "registry.json"
    path.write_text(json.dumps({
        "version": 1,
        "jurisdictions": {"autonet": {
            "network": {"rpc_url": RPC_URL, "chain_id": CHAIN_ID,
                        "gas_symbol": "ETH", "gas_decimals": 18},
            "contracts": {
                "substrate": contracts["substrate"],
                "service_registry": contracts["service_registry"],
                "payment_channel": contracts["payment_channel"],
                "charter_anchor": contracts["charter_anchor"],
            },
        }},
    }, indent=2), encoding="utf-8")
    return path


def start_stub(state: Dict[str, Any], registry: Path) -> None:
    pid = spawn_detached(
        [sys.executable, str(HERE / "stub_server.py"), "--host", "127.0.0.1",
         "--port", str(STUB_PORT), "--registry", str(registry)],
        cwd=STATE_DIR, env=dict(os.environ), log_path=STATE_DIR / "stub.log")
    state["stub_pid"] = pid
    save_state(state)
    if not wait_until(lambda: http_json(f"{STUB_URL}/health")["ok"], 30.0):
        raise RuntimeError(f"stub never answered\n{tail(STATE_DIR / 'stub.log')}")
    reg = http_json(f"{STUB_URL}/registry.json")
    assert reg["jurisdictions"]["autonet"]["network"]["rpc_url"] == RPC_URL
    log(f"stub up at {STUB_URL} (pid {pid})")


def daemon_config(contracts: Dict[str, str]) -> Dict[str, Any]:
    """Every chain field explicit: explicit fields are never replaced by the
    registry (atn/config.py resolve_network_registry), so even a failed
    ATN_REGISTRY_URL fetch cannot pull Shadownet addresses in."""
    return {
        "providers": {
            STUB_PROVIDER: {"base_url": f"{STUB_URL}/v1", "api_key": "e2e",
                            "default_model": STUB_MODEL,
                            "models": [STUB_MODEL]},
        },
        "autonet": {
            "enabled": False,
            "local_ws_port": WS_PORT,
            "remote_ws_host": "",
            "remote_ws_port": WS_PORT + 1,
            "integration_ws_enabled": False,
            "owner_wallet": OWNER[0],
            "private_key": OWNER[1],
            "rpc_url": RPC_URL,
            "chain_id": CHAIN_ID,
            "gas_symbol": "ETH",
            "dao_address": "0x000000000000000000000000000000000000dEaD",
            "substrate_address": contracts["substrate"],
            "service_registry_address": contracts["service_registry"],
            "payment_channel_address": contracts["payment_channel"],
            "charter_anchor_address": contracts["charter_anchor"],
        },
        "voice": {"enabled": False},
        "auto_update": {"enabled": False},
    }


def daemon_env(home: Path) -> Dict[str, str]:
    env = xd.daemon_env(home)
    env["KEYSTORE_DIR"] = str(home / ".atn" / "keystore")
    env["ATN_REGISTRY_URL"] = f"{STUB_URL}/registry.json"
    env["ATN_INTEGRATION_WS"] = "0"
    env["ATN_AUTO_UPDATE"] = "0"
    env["AUTONET_CONFIG"] = str(home / "autonet.yaml")
    # Hash-based usefulness coords: deterministic, and no sentence-embedding
    # model download into the empty redirected home.
    env["ATN_USEFULNESS_EMBEDDER"] = "hashing"
    # Only the providers config.yaml names: no Claude Max / Codex bridge
    # probe adopting this machine's subscription logins.
    env["ATN_DISABLE_PROVIDER_AUTODETECT"] = "1"
    # Never inherit listener / owner overrides from the caller's shell.
    for k in ("ATN_REMOTE_WS_HOST", "ATN_REMOTE_WS_PORT", "ATN_OWNER_WALLET",
              "ATN_INTEGRATION_WS_HOST", "ATN_INTEGRATION_WS_PORT"):
        env.pop(k, None)
    return env


def start_daemon(state: Dict[str, Any], contracts: Dict[str, str], *,
                 restart: bool = False) -> Path:
    """Launch the daemon. ``restart`` relaunches it on the SAME home without
    rewriting config.yaml (tests may have changed it through the daemon) and
    appends to daemon.log."""
    home = STATE_DIR / "home"
    if not restart:
        home.mkdir(parents=True, exist_ok=True)
        xd.write_daemon_config(home, cfg=daemon_config(contracts))
        (home / "autonet.yaml").write_text(
            "p2p:\n  listen_host: 127.0.0.1\n  bootstrap_peers: []\n",
            encoding="utf-8")
    # cwd = the isolated home, NOT the repo: a ./agents dir in the cwd wins
    # over the data dir (atn/config.py _default_agents_dir).
    pid = spawn_detached([sys.executable, "-m", "atn", "--headless"], cwd=home,
                         env=daemon_env(home),
                         log_path=STATE_DIR / "daemon.log", append=restart)
    state["daemon_pid"] = pid
    state["home"] = str(home)
    save_state(state)
    log(f"daemon starting (pid {pid}, home {home})")
    return home


# ---------------------------------------------------------------------------
# Fixtures (over the daemon's own WS surface)
# ---------------------------------------------------------------------------

TOOL_CODE = (
    "import json, sys\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'text': str(args.get('text', '')).upper()}))\n"
)


ADOPTABLE_NAME = "e2e_foreign_lower"
ADOPTABLE_CODE = (
    "import json, sys\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'text': str(args.get('text', '')).lower()}))\n"
)


def seed_adoptable_tool(home: Path) -> str:
    """Write a pinned manifest authored by the SPARE account (a "foreign"
    author) plus its code into the daemon's tool blob store, without
    registering it. Returns the manifest digest."""
    from nodes.common.blob_store import BlobStore
    from nodes.common.world_model_substrate.tool_manifest import (
        build_tool_manifest,
    )
    blobs = BlobStore(data_dir=str(home / ".atn" / "tools" / "blobs"))
    code_digest = blobs.add_bytes(ADOPTABLE_CODE.encode("utf-8"))
    manifest = build_tool_manifest(
        name=ADOPTABLE_NAME,
        description="Live-harness fixture: a foreign pinned tool that "
                    "lower-cases `text`, adoptable but not installed.",
        input_schema={"type": "object",
                      "properties": {"text": {"type": "string"}},
                      "required": ["text"]},
        author=SPARE[0], trust_class="pinned", code_digest=code_digest,
        runtime="python3", created_ts=1)
    return blobs.add_json(manifest)


async def seed(state: Dict[str, Any], contracts: Dict[str, str]) -> None:
    from eth_account import Account
    from web3 import Web3

    client = xd.DaemonClient(WS_URL, "e2e-live")
    await client.connect(timeout=BOOT_TIMEOUT)
    try:
        snap = await client.ok({"type": "snapshot"})
        existing = set((snap.get("agents") or {}).keys())
        log(f"daemon up at {WS_URL}; boot agents: {sorted(existing)}")
        # Isolation check: the daemon must be on the stub alone.
        leaked = [pid for pid, p in (snap.get("providers") or {}).items()
                  if pid in ("claude_max", "codex_max") and p.get("active")]
        if leaked:
            raise RuntimeError(f"subscription providers active in the "
                               f"isolated daemon: {leaked}")

        # 1. Agents. system_prompt (not prompt) registers them IDLE.
        for a in AGENTS:
            if a["id"] in existing:
                continue
            frame: Dict[str, Any] = {
                "type": "create_agent", "id": a["id"], "name": a["name"],
                "description": a["description"], "mode": "cognitive",
                "system_prompt": "You are a live-harness fixture. Reply briefly.",
                "provider": STUB_PROVIDER, "model": STUB_MODEL,
            }
            if a.get("parent"):
                frame["caller_id"] = a["parent"]
            await client.ok(frame)
        snap = await client.ok({"type": "snapshot"})
        agents = snap.get("agents") or {}
        missing = [a["id"] for a in AGENTS if a["id"] not in agents]
        if missing:
            raise RuntimeError(f"agents missing after create: {missing}")
        child = agents["e2e-alpha-child"]
        if child.get("parent_id") != "e2e-alpha":
            raise RuntimeError(f"child parent_id={child.get('parent_id')!r}")
        log(f"seeded agents: {[a['id'] for a in AGENTS]}")

        # 2. A pinned, owner-authored tool.
        tool = await client.ok({
            "type": "register_tool", "name": TOOL_NAME,
            "description": "Live-harness fixture: upper-cases `text`.",
            "input_schema": {"type": "object",
                             "properties": {"text": {"type": "string"}},
                             "required": ["text"]},
            "code": TOOL_CODE,
        })
        tool_digest = tool["digest"]
        log(f"registered tool {TOOL_NAME} ({tool_digest[:12]})")

        # 2b. An ADOPTABLE foreign tool: a pinned manifest + code that sit
        #     in the daemon's blob store (as if fetched from a peer) but are
        #     not registered locally. adopt_tool resolves blobs locally
        #     first, so the adoption approve path is testable offline.
        adoptable = seed_adoptable_tool(Path(state["home"]))
        log(f"seeded adoptable foreign tool {ADOPTABLE_NAME} "
            f"({adoptable[:12]})")

        # 3. e2e-alpha on chain: gas from the deployer, owner-bound to OWNER.
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        exported = await client.ok({"type": "export_agent_key",
                                    "agent_id": "e2e-alpha"})
        alpha_addr = Web3.to_checksum_address(exported["address"])
        alpha_key = exported["private_key"]
        deployer = Account.from_key(DEPLOYER[1])
        owner = Account.from_key(OWNER[1])
        xd.fund_gas(w3, deployer, alpha_addr, ether=10)
        substrate = w3.eth.contract(
            address=Web3.to_checksum_address(contracts["substrate"]),
            abi=xd.load_abi("Substrate"))
        shape = xd.resolve_binding_shape(substrate.abi)
        lineage = bytes(Web3.keccak(text=f"e2e-live:{alpha_addr}"))
        xd.register_agent_on_chain(
            w3, substrate, shape, agent_pk=alpha_key, agent_addr=alpha_addr,
            owner_acct=owner, lineage_hash=lineage, peer_id=b"e2e-live-alpha")
        reg = await client.ok({"type": "check_agent_registration",
                               "agent_id": "e2e-alpha"})
        if not reg.get("registered"):
            raise RuntimeError(f"daemon does not see e2e-alpha on chain: {reg}")
        log(f"e2e-alpha registered on chain as {alpha_addr}")

        # 4. A tool-backed service on the local ServiceRegistry, signed by
        #    e2e-alpha's own key through the daemon's register_service leg.
        svc = await client.ok({
            "type": "register_service", "name": SERVICE_NAME,
            "description": "Live-harness fixture: upper-cases text.",
            "input_schema": {"type": "object",
                             "properties": {"text": {"type": "string"}}},
            "agent_id": "e2e-alpha",
            "tool_digest": tool_digest,
            "ask": {"amount": str(SERVICE_ASK), "unit": "per_item"},
        })
        if not svc.get("on_chain"):
            raise RuntimeError(f"service not listed on chain: {svc}")
        log(f"listed service {svc['digest'][:12]} on chain "
            f"(id {svc.get('service_id')})")

        # 5. A deterministic agent turn through the real execution engine.
        await client.ok({"type": "activate_agent", "agent_id": "e2e-beta"})
        sent = await client.ok({"type": "send_agent_message",
                                "agent_id": "e2e-beta",
                                "content": "harness ping"})
        exec_id = sent.get("execution_id") or ""
        if not exec_id:
            raise RuntimeError(f"send_agent_message did not run: {sent}")
        rec: Dict[str, Any] = {}
        deadline = time.time() + 90
        while time.time() < deadline:
            rec = await client.ok({"type": "get_execution",
                                   "execution_id": exec_id})
            if rec.get("status") in ("completed", "failed", "error"):
                break
            await asyncio.sleep(1.0)
        out = rec.get("output")
        answer = str(out.get("result") if isinstance(out, dict) else out or "")
        if "E2E-STUB-REPLY" not in answer:
            raise RuntimeError(f"stub turn did not answer: status="
                               f"{rec.get('status')} output={answer[:200]!r} "
                               f"error={rec.get('error')}")
        await client.ok({"type": "deactivate_agent", "agent_id": "e2e-beta"})
        log(f"stub turn answered: {answer[:60]!r}")

        state["fixtures"] = {
            "agents": [{"id": a["id"], "name": a["name"],
                        "parent": a.get("parent", "")} for a in AGENTS],
            "boot_agents": sorted(existing),
            "alpha_address": alpha_addr,
            "tool": {"name": TOOL_NAME, "digest": tool_digest},
            "adoptable_tool": {"name": ADOPTABLE_NAME, "digest": adoptable,
                               "author": SPARE[0]},
            "service": {"name": SERVICE_NAME, "digest": svc["digest"],
                        "service_id": svc.get("service_id"),
                        "ask": SERVICE_ASK},
            "stub_turn": {"execution_id": exec_id, "answer": answer},
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_up(args: argparse.Namespace) -> int:
    preflight()
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR, ignore_errors=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"state_dir": str(STATE_DIR), "ws_url": WS_URL,
                             "rpc_url": RPC_URL, "stub_url": STUB_URL,
                             "chain_id": CHAIN_ID, "started": time.time()}
    save_state(state)
    try:
        start_chain(state)
        from web3 import Web3
        contracts = deploy(Web3(Web3.HTTPProvider(RPC_URL)))
        state["contracts"] = contracts
        save_state(state)
        start_stub(state, write_registry(contracts))
        start_daemon(state, contracts)
        asyncio.run(seed(state, contracts))
    except BaseException as exc:
        log(f"up FAILED: {exc!r}")
        if state.get("daemon_pid"):
            log(f"daemon log tail:\n{tail(STATE_DIR / 'daemon.log')}")
        if not args.keep_on_fail:
            teardown(state, remove=False)
        return 1
    state["accounts"] = {
        "deployer": {"address": DEPLOYER[0], "private_key": DEPLOYER[1]},
        "owner": {"address": OWNER[0], "private_key": OWNER[1]},
        "spare": {"address": SPARE[0], "private_key": SPARE[1]},
    }
    state["ready"] = True
    save_state(state)
    print_summary(state)
    return 0


def print_summary(state: Dict[str, Any]) -> None:
    fx = state.get("fixtures", {})
    lines = [
        "", "=" * 70, "  LIVE E2E HARNESS UP", "=" * 70,
        f"  DAEMON_WS   {state['ws_url']}",
        f"  RPC         {state['rpc_url']}  (chain {state['chain_id']})",
        f"  stub        {state['stub_url']}",
        f"  state dir   {state['state_dir']}",
        f"  pids        hardhat={state.get('hardhat_pid')} "
        f"stub={state.get('stub_pid')} daemon={state.get('daemon_pid')}",
        "  contracts:",
    ]
    for k, v in (state.get("contracts") or {}).items():
        lines.append(f"    {k:22s} {v}")
    lines.append("  funded test accounts (LOCAL hardhat keys, 10000 ETH each):")
    for role, a in (state.get("accounts") or {}).items():
        lines.append(f"    {role:8s} {a['address']}  {a['private_key']}")
    lines.append("  fixtures:")
    for a in fx.get("agents", []):
        lines.append(f"    agent    {a['id']:18s} {a['name']}"
                     + (f" (parent {a['parent']})" if a.get("parent") else ""))
    if fx.get("tool"):
        lines.append(f"    tool     {fx['tool']['name']} {fx['tool']['digest'][:16]}")
    if fx.get("adoptable_tool"):
        a = fx["adoptable_tool"]
        lines.append(f"    adopt    {a['name']} {a['digest']} (blob only)")
    if fx.get("service"):
        s = fx["service"]
        lines.append(f"    service  {s['name']} {s['digest'][:16]} "
                     f"(on-chain id {s['service_id']}, ask {s['ask']})")
    lines += ["", "  flutter test integration_test/live -d windows "
              f"--dart-define=DAEMON_WS={state['ws_url']}", "=" * 70]
    print("\n".join(lines), flush=True)


async def _ws_probe(timeout: float = 15.0) -> Dict[str, Any]:
    client = xd.DaemonClient(WS_URL, "e2e-status")
    await client.connect(timeout=timeout)
    try:
        return await client.ok({"type": "snapshot"}, timeout=15.0)
    finally:
        await client.close()


def cmd_status(args: argparse.Namespace) -> int:
    st = load_state()
    if not st:
        print(f"DOWN: no state at {state_path()}")
        return 1
    ok = True
    rows = []
    for k in ("hardhat_pid", "stub_pid", "daemon_pid"):
        alive = pid_alive(int(st.get(k) or 0))
        ok &= alive
        rows.append(f"  {k:12s} {st.get(k)}  {'alive' if alive else 'DEAD'}")
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(st["rpc_url"]))
        rows.append(f"  rpc          chain {w3.eth.chain_id} block "
                    f"{w3.eth.block_number}")
    except Exception as exc:
        ok = False
        rows.append(f"  rpc          FAIL {exc!r}")
    try:
        h = http_json(f"{st['stub_url']}/health")
        rows.append(f"  stub         ok ({h.get('requests')} completions served)")
    except Exception as exc:
        ok = False
        rows.append(f"  stub         FAIL {exc!r}")
    try:
        snap = asyncio.run(_ws_probe())
        rows.append(f"  daemon       {st['ws_url']} agents="
                    f"{sorted((snap.get('agents') or {}).keys())}")
    except Exception as exc:
        ok = False
        rows.append(f"  daemon       FAIL {exc!r}")
    print(("UP" if ok and st.get("ready") else "DEGRADED") + f" ({state_path()})")
    print("\n".join(rows))
    return 0 if ok and st.get("ready") else 1


def _state_dir_marker() -> str:
    """A path fragment that names STATE_DIR in both Windows and MSYS forms
    (Git's gpg/keyboxd see %TEMP% as /tmp): the components below the temp
    dir, or the last two components when STATE_DIR lives elsewhere."""
    try:
        rel = STATE_DIR.relative_to(Path(tempfile.gettempdir()).resolve())
        parts = rel.parts
    except ValueError:
        parts = STATE_DIR.parts[-2:]
    return "/" + "/".join(parts).lower() + "/"


def kill_strays() -> None:
    """Kill processes the harness did not record but that run out of its
    state dir: the daemon's host scan runs git/gpg with the redirected HOME,
    which leaves a keyboxd/gpg-agent holding <home>/.gnupg open after the
    daemon dies (seen live: the state dir could not be removed). Matches
    only command lines naming STATE_DIR, never by name or port."""
    try:
        import psutil
    except ImportError:
        return
    marker = _state_dir_marker()
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == me:
                continue
            cmd = " ".join(proc.info.get("cmdline") or [])
            if marker in cmd.replace("\\", "/").lower() + "/":
                proc.kill()
                log(f"killed stray pid {proc.info['pid']}: {cmd[:120]}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def teardown(state: Dict[str, Any], *, remove: bool) -> None:
    for k in ("daemon_pid", "stub_pid", "hardhat_pid"):
        pid = int(state.get(k) or 0)
        if pid:
            kill_tree(pid)
            log(f"killed {k}={pid}")
    kill_strays()
    for port in (WS_PORT, STUB_PORT, RPC_PORT):
        if not wait_until(lambda p=port: not port_in_use(p), 15.0):
            log(f"WARNING: port {port} still in use after teardown")
    state["ready"] = False
    if remove:
        # The daemon may hold file handles a moment after the kill.
        for _ in range(10):
            shutil.rmtree(STATE_DIR, ignore_errors=True)
            if not STATE_DIR.exists():
                break
            time.sleep(0.5)
        log(f"removed {STATE_DIR}" if not STATE_DIR.exists()
            else f"WARNING: could not fully remove {STATE_DIR}")
    else:
        save_state(state)


def cmd_daemon_stop(args: argparse.Namespace) -> int:
    """Kill ONLY the daemon (chain, stub, home and fixtures stay), so a test
    can watch the app lose and regain it. Pairs with ``daemon-start``."""
    st = load_state()
    if not st.get("ready"):
        log(f"harness is not up ({state_path()})")
        return 1
    pid = int(st.get("daemon_pid") or 0)
    if pid:
        kill_tree(pid)
        log(f"killed daemon_pid={pid}")
    if not wait_until(lambda: not port_in_use(WS_PORT), 15.0):
        log(f"port {WS_PORT} still in use after killing the daemon")
        return 1
    st["daemon_pid"] = 0
    save_state(st)
    return 0


def cmd_daemon_start(args: argparse.Namespace) -> int:
    """Relaunch the daemon on the same home after ``daemon-stop`` and wait
    until it answers a snapshot."""
    st = load_state()
    if not st.get("ready"):
        log(f"harness is not up ({state_path()})")
        return 1
    if pid_alive(int(st.get("daemon_pid") or 0)):
        log(f"daemon already running (pid {st['daemon_pid']})")
        return 0
    if port_in_use(WS_PORT):
        log(f"port {WS_PORT} is busy; refusing to start")
        return 1
    start_daemon(st, st["contracts"], restart=True)
    try:
        snap = asyncio.run(_ws_probe(timeout=BOOT_TIMEOUT))
    except Exception as exc:
        log(f"daemon did not come back: {exc!r}\n"
            f"{tail(STATE_DIR / 'daemon.log')}")
        return 1
    log(f"daemon back at {WS_URL}; agents "
        f"{sorted((snap.get('agents') or {}).keys())}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    st = load_state()
    if not st:
        log(f"nothing to tear down at {STATE_DIR}")
        return 0
    teardown(st, remove=not args.keep_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live E2E harness: isolated daemon + local chain.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="start chain, stub, daemon; seed fixtures")
    up.add_argument("--keep-on-fail", action="store_true",
                    help="leave processes running when up fails (debugging)")
    sub.add_parser("status", help="check every piece answers")
    sub.add_parser("daemon-stop", help="kill only the daemon (keeps chain, "
                   "stub, home)")
    sub.add_parser("daemon-start", help="relaunch the daemon on the same home")
    down = sub.add_parser("down", help="kill everything up started")
    down.add_argument("--keep-dir", action="store_true",
                      help="keep the state dir (logs, daemon home)")
    args = ap.parse_args()
    return {"up": cmd_up, "status": cmd_status, "down": cmd_down,
            "daemon-stop": cmd_daemon_stop,
            "daemon-start": cmd_daemon_start}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
