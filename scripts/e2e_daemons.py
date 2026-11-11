"""Helpers for scripts that run real daemons on a local chain: chain calls
(ABIs, owner-bound registration, gas), a daemon's config and environment,
and a small WebSocket client. Used by scripts/e2e_live/harness.py."""

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
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent


CHAIN_ID = 1337


OLLAMA_URL = os.environ.get("E2E_OLLAMA_URL", "http://127.0.0.1:11434")


DAEMON_BOOT_TIMEOUT = 180.0


WS_CALL_TIMEOUT = 240.0


def _safe(msg: str) -> str:
    """Strip anything the console codec can't encode.

    Windows consoles default to cp1252; a daemon log tail carrying a rich-console
    box-drawing char or an em dash crashes ``print`` with a UnicodeEncodeError —
    which would then mask the real failure we were trying to report.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return msg.encode(enc, errors="replace").decode(enc, errors="replace")


def log(msg: str) -> None:
    print(_safe(f"[e2e] {msg}"), flush=True)


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
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
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
        w3.eth.send_raw_transaction(raw), timeout=90)


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


def write_daemon_config(home: Path, *, cfg: Dict[str, Any]) -> Path:
    """Write an isolated ~/.atn/config.yaml inside a temp home.

    The daemon has no --config flag: ``load_config()`` reads
    ``Path.home()/".atn"/"config.yaml"``. We redirect ``Path.home()`` for the
    child via USERPROFILE (ntpath.expanduser honours it), so this file IS the
    child's config.
    """
    import yaml
    atn_home = home / ".atn"
    atn_home.mkdir(parents=True, exist_ok=True)
    path = atn_home / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def daemon_env(home: Path) -> Dict[str, str]:
    """Environment for a daemon subprocess with a fully redirected home.

    ``Path.home()`` on Windows resolves USERPROFILE, then HOMEDRIVE+HOMEPATH,
    then HOME. All four are typically set, so redirecting USERPROFILE alone
    leaves HOMEDRIVE/HOMEPATH able to leak the real home to anything that reads
    them directly — clear them too. On POSIX, HOME is the one that matters.
    """
    env = dict(os.environ)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    env.pop("HOMEDRIVE", None)
    env.pop("HOMEPATH", None)
    # The repo root is on sys.path via cwd; make it explicit since we launch
    # the child from the temp home (NOT the repo) on purpose — see below.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (f"{REPO}{os.pathsep}{existing}" if existing
                         else str(REPO))
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the child's ollama pointed where we probed.
    env["OLLAMA_HOST"] = OLLAMA_URL
    # The agents here are native (the daemon runs their loop on ollama), a
    # mode that is off on a fresh daemon (atn/native_mode.py).
    env["ATN_NATIVE_AGENTS"] = "1"
    return env


class DaemonClient:
    """A WS client for one real daemon, speaking the production frames.

    Connects to the PRIVILEGED LOCAL listener (loopback), which is what a
    local app gets: pre-authed as owner, no handshake. Every call is a real
    frame over a real socket into a real process.
    """

    def __init__(self, url: str, label: str) -> None:
        self.url = url
        self.label = label
        self._ws: Any = None
        self._seq = 0

    async def connect(self, timeout: float = DAEMON_BOOT_TIMEOUT) -> None:
        """Dial with retries: the daemon's port LISTENS before the runtime has
        finished booting (heavy imports), so an early connect can be accepted
        and then hang. Retry the whole handshake, and prove liveness with a
        real ``snapshot`` round trip before declaring the daemon up.
        """
        import websockets

        deadline = time.time() + timeout
        last: Exception | None = None
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                # A generous ping timeout: a service_request drives real
                # inference, and if the daemon ever stalls the loop (the bug
                # this script found in the payment gate) a tight keepalive
                # drops the connection mid-request and reports a transport
                # error for what is really a server-side stall. Keep the
                # keepalive on — a silent dead socket is worse — just patient.
                self._ws = await websockets.connect(
                    self.url, open_timeout=15,
                    ping_interval=20, ping_timeout=120,
                    max_size=16 * 1024 * 1024)
                # Liveness probe, not just a TCP accept.
                await self.call({"type": "snapshot"}, timeout=45.0)
                log(f"{self.label} up at {self.url} (attempt {attempt})")
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                await asyncio.sleep(2.0)
        raise RuntimeError(
            f"{self.label} never answered at {self.url} after {attempt} "
            f"attempts: {last!r}")

    async def call(self, frame: Dict[str, Any], *,
                   timeout: float = WS_CALL_TIMEOUT) -> Dict[str, Any]:
        """One request/response. The daemon broadcasts events on the same
        socket, so read until the reply carrying OUR msg_id arrives."""
        if self._ws is None:
            raise RuntimeError(f"{self.label} not connected")
        self._seq += 1
        msg_id = f"{self.label}-{self._seq}"
        frame = dict(frame)
        frame["msg_id"] = msg_id
        await self._ws.send(json.dumps(frame))
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(1.0, deadline - time.time())
            raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(msg, dict) and msg.get("msg_id") == msg_id:
                return msg
        raise TimeoutError(
            f"{self.label}: no reply to {frame.get('type')} in {timeout}s")

    async def ok(self, frame: Dict[str, Any], *,
                 timeout: float = WS_CALL_TIMEOUT) -> Dict[str, Any]:
        """Call and require ok:true, returning the result payload."""
        reply = await self.call(frame, timeout=timeout)
        if not reply.get("ok"):
            raise RuntimeError(
                f"{self.label}: {frame.get('type')} failed — "
                f"{reply.get('error')}")
        return reply.get("result") or {}

    async def export_key(self, agent_id: str, owner_acct: Any) -> Dict[str, Any]:
        """``export_agent_key`` with the owner wallet's signature over a
        one-use challenge, which the daemon requires for any key export
        (OWNER_SIGNED_MESSAGES in atn/ws_server.py)."""
        from eth_account import Account
        from eth_account.messages import encode_defunct

        ch = await self.ok({"type": "owner_challenge",
                            "action": "export_agent_key"})
        sig = Account.sign_message(encode_defunct(text=ch["challenge"]),
                                   owner_acct.key).signature
        return await self.ok({
            "type": "export_agent_key", "agent_id": agent_id,
            "owner_nonce": ch["nonce"], "owner_sig": "0x" + bytes(sig).hex()})

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
