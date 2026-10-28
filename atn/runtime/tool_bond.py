"""ToolBond gate for core-replacing tools (docs/tool_substrate.md,
Decision 2026-10-12).

A pinned tool may replace a core built-in (shell / file tools) only while
its manifest digest is BONDED on-chain: ``ToolBond.isBonded(bytes32)`` is
true while the author's stake >= minStake and no unbond is pending
(contract: trustless-contracts/contracts/ToolBond.sol, 448023a). The bond is
what staked reviews can take: a settled harm H slashes H of it
(docs/multi_jurisdiction.md decision 10; reviews live in review_batch.py).

FAIL CLOSED. The address comes from the jurisdiction registry
(``contracts.tool_bond`` in atn/registry.json, or ATN_REGISTRY_URL), or the
ATN_TOOL_BOND_ADDRESS env var. With no address, or any RPC fault, the answer
is "not bonded" and the swap is refused. The single exception is the dev
override ``ATN_TOOL_BOND_DEV_OVERRIDE=1`` for local testing, which treats
every digest as bonded and logs a warning each time it is consulted.

Results are cached briefly (positive 60s, negative 15s) so a dispatch hot
path does not hit the RPC per call. The web3 call runs in a thread.

There is no "vetted" flag: ratings are a gradient (``ToolBondClient.rating``
returns the REP-weighted per-axis means, harm and the REP mass behind them),
and the owner or agent decides how much backing is enough.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Optional
from nodes.common.tx_nonce import next_nonce

log = logging.getLogger(__name__)

DEV_OVERRIDE_ENV = "ATN_TOOL_BOND_DEV_OVERRIDE"
ADDRESS_ENV = "ATN_TOOL_BOND_ADDRESS"
_POSITIVE_TTL_S = 60.0
_NEGATIVE_TTL_S = 15.0
_RPC_TIMEOUT_S = 5.0

TOOL_BOND_ABI = [{
    "name": "isBonded", "type": "function", "stateMutability": "view",
    "inputs": [{"name": "digest", "type": "bytes32"}],
    "outputs": [{"name": "", "type": "bool"}],
}]


def dev_override() -> bool:
    return os.environ.get(DEV_OVERRIDE_ENV, "").strip() == "1"


def _registry_seed(runtime: Any) -> dict[str, Any]:
    try:
        from ..config import _load_registry_seed
        cfg = getattr(getattr(runtime, "_config", None), "rpb", None)
        jid = str(getattr(cfg, "jurisdiction_id", "") or "autonet")
        return _load_registry_seed(jid)
    except Exception:  # noqa: BLE001
        return {}


class ToolBondGate:
    """``await gate.is_bonded(digest) -> (bonded, reason)``. Never raises.

    ``checker`` (tests / alternate transports): a sync callable
    ``digest -> bool`` used instead of web3 when given.
    """

    def __init__(self, runtime: Any = None, *,
                 checker: Optional[Callable[[str], bool]] = None) -> None:
        self._runtime = runtime
        self._checker = checker
        self._cache: dict[str, tuple[bool, float, str]] = {}
        self._address: Optional[str] = None
        self._rpc_url: Optional[str] = None

    # -- configuration -------------------------------------------------
    def _resolve_target(self) -> tuple[str, str]:
        if self._address is None:
            addr = os.environ.get(ADDRESS_ENV, "").strip()
            rpc = ""
            cfg = getattr(getattr(self._runtime, "_config", None), "rpb", None)
            if not addr:
                addr = str(getattr(cfg, "tool_bond_address", "") or "")
            seed: dict[str, Any] = {}
            if not addr:
                seed = _registry_seed(self._runtime)
                addr = str(seed.get("tool_bond_address") or "")
            rpc = str(getattr(cfg, "rpc_url", "") or seed.get("rpc_url") or "")
            if addr and not rpc:
                rpc = str(_registry_seed(self._runtime).get("rpc_url") or "")
            self._address, self._rpc_url = addr, rpc
        return self._address or "", self._rpc_url or ""

    def configured(self) -> bool:
        return bool(self._checker) or bool(self._resolve_target()[0])

    # -- query ---------------------------------------------------------
    def _query(self, digest: str) -> bool:
        if self._checker is not None:
            return bool(self._checker(digest))
        addr, rpc = self._resolve_target()
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": _RPC_TIMEOUT_S}))
        c = w3.eth.contract(address=Web3.to_checksum_address(addr),
                            abi=TOOL_BOND_ABI)
        return bool(c.functions.isBonded(bytes.fromhex(digest)).call())

    async def is_bonded(self, digest: str) -> tuple[bool, str]:
        digest = str(digest or "").lower().removeprefix("0x")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return False, "malformed manifest digest"
        if self._checker is None and dev_override():
            log.warning("ToolBond DEV OVERRIDE: treating %s as bonded "
                        "(%s=1; never set this outside local testing)",
                        digest[:16], DEV_OVERRIDE_ENV)
            return True, "dev override"
        now = time.monotonic()
        hit = self._cache.get(digest)
        if hit is not None and hit[1] > now:
            return hit[0], f"{hit[2]} (cached)"
        if not self.configured():
            return False, ("no ToolBond address configured (registry key "
                           "'tool_bond'); fails closed")
        try:
            ok = await asyncio.to_thread(self._query, digest)
            reason = "bonded" if ok else "not bonded on ToolBond"
        except Exception as exc:  # noqa: BLE001 — fail closed
            log.warning("ToolBond isBonded(%s) failed: %s", digest[:16], exc)
            ok, reason = False, f"ToolBond query failed: {exc}"
        ttl = _POSITIVE_TTL_S if ok else _NEGATIVE_TTL_S
        self._cache[digest] = (ok, now + ttl, reason)
        return ok, reason


def gate_for(runtime: Any) -> ToolBondGate:
    """The runtime's gate (created lazily, so tests can pre-set one)."""
    gate = getattr(runtime, "tool_bond_gate", None)
    if isinstance(gate, ToolBondGate):
        return gate
    gate = ToolBondGate(runtime)
    try:
        setattr(runtime, "tool_bond_gate", gate)
    except Exception:  # noqa: BLE001
        pass
    return gate


# ---------------------------------------------------------------------------
# ToolBond client (trustless-contracts ToolBond 448023a): the author's bond
# and the settled ratings. Reviews (commit / reveal / finalize / claim) are in
# review_batch.py, which reuses this client's plumbing. Reads never raise
# (None on fault); writes sign with the AGENT key and return {success,
# tx_hash} or {success: False, error}.
# ---------------------------------------------------------------------------

def _fn(name: str, inputs: list, outputs: list = (), mutability: str = "nonpayable") -> dict[str, Any]:
    return {"name": name, "type": "function", "stateMutability": mutability,
            "inputs": [{"name": n, "type": t} for n, t in inputs],
            "outputs": [{"name": n, "type": t} for n, t in outputs]}


TOOL_BOND_CLIENT_ABI = [
    _fn("isBonded", [("digest", "bytes32")], [("", "bool")], "view"),
    _fn("minStake", [], [("", "uint256")], "view"),
    _fn("reviewerShareBps", [], [("", "uint256")], "view"),
    _fn("bonds", [("", "bytes32")],
        [("author", "address"), ("stake", "uint256"), ("unbondRequestedAt", "uint256"),
         ("unbondEpoch", "uint256"), ("pending", "uint32")], "view"),
    _fn("toolRating", [("digest", "bytes32"), ("axisId", "bytes32")],
        [("sum", "int256"), ("mass", "uint256")], "view"),
    _fn("toolHarm", [("digest", "bytes32")], [("sum", "uint256"), ("mass", "uint256")], "view"),
    _fn("bond", [("digest", "bytes32"), ("amount", "uint256")]),
    _fn("requestUnbond", [("digest", "bytes32")]),
    _fn("withdraw", [("digest", "bytes32")]),
]

SUBSTRATE_TOOL_ABI = [
    _fn("registerTool", [("manifestDigest", "bytes32")]),
    _fn("toolAuthor", [("", "bytes32")], [("", "address")], "view"),
    _fn("approve", [("spender", "address"), ("amount", "uint256")], [("", "bool")]),
    _fn("allowance", [("owner", "address"), ("spender", "address")], [("", "uint256")], "view"),
    _fn("balanceOf", [("account", "address")], [("", "uint256")], "view"),
]


def _b32(hex_digest: str) -> bytes:
    h = str(hex_digest or "").lower().removeprefix("0x")
    if len(h) != 64:
        raise ValueError(f"expected a 32-byte hex digest, got {hex_digest!r}")
    return bytes.fromhex(h)


def axis_key(axis_id: str) -> bytes:
    """ToolBond's axis key: keccak256 of the charter axis id."""
    from eth_utils import keccak
    return keccak(text=str(axis_id))


class ToolBondClient:
    """Chain surface of the tool bond for one runtime.

    Addresses resolve like ``ToolBondGate`` (env, config, registry seed);
    Substrate comes from ``rpb.substrate_address``. Cheap to construct per use.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._gate = ToolBondGate(runtime)

    # -- plumbing ------------------------------------------------------
    def _cfg(self) -> Any:
        return getattr(getattr(self._runtime, "_config", None), "rpb", None)

    def _addresses(self) -> tuple[str, str, str]:
        bond_addr, rpc = self._gate._resolve_target()
        sub = str(getattr(self._cfg(), "substrate_address", "") or "")
        if not sub:
            sub = str(_registry_seed(self._runtime).get("substrate_address") or "")
        return bond_addr, sub, rpc

    def available(self) -> bool:
        return all(self._addresses())

    def _w3(self):
        from web3 import Web3
        return Web3(Web3.HTTPProvider(self._addresses()[2],
                                      request_kwargs={"timeout": _RPC_TIMEOUT_S * 4}))

    def _contracts(self, w3):
        from web3 import Web3
        bond_addr, sub, _ = self._addresses()
        if not bond_addr or not sub:
            raise ValueError("ToolBond or Substrate address not configured")
        return (w3.eth.contract(address=Web3.to_checksum_address(bond_addr), abi=TOOL_BOND_CLIENT_ABI),
                w3.eth.contract(address=Web3.to_checksum_address(sub), abi=SUBSTRATE_TOOL_ABI))

    def _send(self, w3, account, func, gas_floor: int) -> dict[str, Any]:
        try:
            gas = max(int(func.estimate_gas({"from": account.address}) * 16 // 10), gas_floor)
        except Exception as exc:  # noqa: BLE001 — a revert shows up here first
            return {"success": False, "error": f"would revert: {exc}"}
        chain_id = int(getattr(self._cfg(), "chain_id", 0) or w3.eth.chain_id)
        tx = func.build_transaction({
            "from": account.address, "gas": gas, "gasPrice": w3.eth.gas_price, "chainId": chain_id,
            "nonce": next_nonce(w3, account.address)})
        signed = account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status == 1:
            return {"success": True, "tx_hash": tx_hash.hex()}
        return {"success": False, "error": "transaction reverted", "tx_hash": tx_hash.hex()}

    async def _write(self, private_key: str, build: Callable[[Any, Any, Any], Any],
                     gas_floor: int = 300_000) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            from eth_account import Account
            w3 = self._w3()
            bond, substrate = self._contracts(w3)
            account = Account.from_key(private_key)
            return self._send(w3, account, build(bond, substrate, account), gas_floor)
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            log.warning("ToolBond write failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def _read(self, fn: Callable[[Any, Any], Any]) -> Any:
        def run() -> Any:
            w3 = self._w3()
            return fn(*self._contracts(w3))
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            log.debug("ToolBond read failed: %s", exc)
            return None

    # -- reads ---------------------------------------------------------
    async def min_stake(self) -> Optional[int]:
        return await self._read(lambda b, s: int(b.functions.minStake().call()))

    async def tool_author(self, digest: str) -> Optional[str]:
        return await self._read(lambda b, s: s.functions.toolAuthor(_b32(digest)).call())

    async def status(self, digest: str) -> Optional[dict[str, Any]]:
        """The author's bond: what the UI and the bond gate show."""
        def fn(b, s):
            d = _b32(digest)
            author, stake, unbond_at, unbond_epoch, pending = b.functions.bonds(d).call()
            return {
                "author": author, "stake": int(stake), "unbond_requested_at": int(unbond_at),
                "unbond_epoch": int(unbond_epoch), "pending_rounds": int(pending),
                "bonded": bool(b.functions.isBonded(d).call()),
            }
        return await self._read(fn)

    async def rating(self, digest: str, axis_ids: list[str]) -> Optional[dict[str, Any]]:
        """The settled rating over every finalized round, per charter axis:
        the REP-weighted sum and REP mass (``mean`` = sum / mass / 100, in
        -1..1), and harm the same way (``mean`` in 0..1). Numbers, not a
        verdict: what is trustworthy enough is the reader's call."""
        def fn(b, s):
            d = _b32(digest)
            axes: dict[str, Any] = {}
            for axis_id in axis_ids:
                total, mass = b.functions.toolRating(d, axis_key(axis_id)).call()
                axes[axis_id] = {"sum": int(total), "mass": int(mass),
                                 "mean": (int(total) / int(mass) / 100.0) if mass else None}
            h_sum, h_mass = b.functions.toolHarm(d).call()
            return {"axes": axes,
                    "harm": {"sum": int(h_sum), "mass": int(h_mass),
                             "mean": (int(h_sum) / int(h_mass) / 100.0) if h_mass else None}}
        return await self._read(fn)

    # -- writes: author ------------------------------------------------
    async def register_and_bond(self, private_key: str, digest: str,
                                amount: Optional[int] = None) -> dict[str, Any]:
        """Substrate.registerTool (if unclaimed), approve, ToolBond.bond.

        ``amount`` defaults to the current minStake. Safe to retry after a
        partial attempt: skips a registration the caller already holds and an
        approval that already covers the amount.
        """
        from eth_account import Account
        me = Account.from_key(private_key).address
        author = await self.tool_author(digest)
        if author is None:
            return {"success": False, "error": "chain unreachable"}
        txs: list[str] = []
        if int(author, 16) == 0:
            res = await self._write(private_key, lambda b, s, a: s.functions.registerTool(_b32(digest)))
            if not res.get("success"):
                return {**res, "step": "registerTool"}
            txs.append(res["tx_hash"])
        elif author.lower() != me.lower():
            return {"success": False, "error": f"tool is registered to another author ({author})"}
        if amount is None:
            amount = await self.min_stake()
            if amount is None:
                return {"success": False, "error": "could not read minStake"}
        allowance = await self._read(lambda b, s: int(s.functions.allowance(me, b.address).call()))
        if allowance is None or allowance < amount:
            res = await self._write(private_key, lambda b, s, a: s.functions.approve(b.address, int(amount)))
            if not res.get("success"):
                return {**res, "step": "approve", "txs": txs}
            txs.append(res["tx_hash"])
        res = await self._write(private_key, lambda b, s, a: b.functions.bond(_b32(digest), int(amount)))
        if not res.get("success"):
            return {**res, "step": "bond", "txs": txs}
        txs.append(res["tx_hash"])
        return {"success": True, "tx_hash": res["tx_hash"], "txs": txs, "amount": int(amount)}

    async def request_unbond(self, private_key: str, digest: str) -> dict[str, Any]:
        return await self._write(private_key, lambda b, s, a: b.functions.requestUnbond(_b32(digest)))

    async def withdraw(self, private_key: str, digest: str) -> dict[str, Any]:
        return await self._write(private_key, lambda b, s, a: b.functions.withdraw(_b32(digest)))


def evidence_hash(*, tool_digest: str, transcript_hash: str, dep_digests: list,
                  code_ref: Optional[dict[str, Any]] = None) -> str:
    """A review's evidenceHash (hex, no 0x): keccak256 of canonical JSON
    {tool, transcript, deps (sorted), code_ref}. A harm claim's code_ref names
    the exact malicious code ({path, start, end, sha256 of the cited lines}).
    Only the hash goes on chain; the evidence stays local."""
    import json
    from eth_utils import keccak

    def norm(h: Any) -> str:
        return str(h).lower().removeprefix("0x")
    body = {"tool": norm(tool_digest), "transcript": norm(transcript_hash),
            "deps": sorted(norm(d) for d in (dep_digests or [])), "code_ref": code_ref or None}
    return keccak(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hex()
