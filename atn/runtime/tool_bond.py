"""ToolBond gate for core-replacing tools (docs/tool_substrate.md,
Decision 2026-10-12).

A pinned tool may replace a core built-in (shell / file tools) only while
its manifest digest is BONDED on-chain: ``ToolBond.isBonded(bytes32)`` is
true while the stake's value when posted >= minStake (a value, 6-decimal
units) and no unbond is pending (contract: trustless-contracts/contracts/
ToolBond.sol, 97ad738: a bond is in one token of the value index). A malware
verdict takes its share of it (reviews live in review_batch.py).

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
# ToolBond client (trustless-contracts ToolBond 97ad738): the author's bond in
# an index token, the bounty, and what ToolBond owes. Reviews (commit /
# reveal / finalize / claim) are in
# review_batch.py, which reuses this client's plumbing. Reads never raise
# (None on fault); writes sign with the AGENT key and return {success,
# tx_hash} or {success: False, error}.
# ---------------------------------------------------------------------------

def _fn(name: str, inputs: list, outputs: list = (), mutability: str = "nonpayable") -> dict[str, Any]:
    return {"name": name, "type": "function", "stateMutability": mutability,
            "inputs": [{"name": n, "type": t} for n, t in inputs],
            "outputs": [{"name": n, "type": t} for n, t in outputs]}


# ToolBond in the value index's tokens (trustless-contracts 97ad738): a bond
# is in one index token, fixed while it holds anything; each deposit is
# valued through Substrate.valueOf when it arrives, and minStake is a value
# (6-decimal units). Bounty parts are owed per household and token and paid
# by payOut.
BOND_FIELDS = ("author", "stake", "unbondRequestedAt", "pending", "bounty", "token", "value")

TOOL_BOND_CLIENT_ABI = [
    _fn("isBonded", [("digest", "bytes32")], [("", "bool")], "view"),
    _fn("minStake", [], [("", "uint256")], "view"),
    _fn("reviewerShareBps", [], [("", "uint256")], "view"),
    _fn("bonds", [("", "bytes32")],
        [("author", "address"), ("stake", "uint256"), ("unbondRequestedAt", "uint256"),
         ("pending", "uint32"), ("bounty", "uint256"), ("token", "address"),
         ("value", "uint256")], "view"),
    _fn("owed", [("", "address"), ("", "address")], [("", "uint256")], "view"),
    _fn("treasuryOwed", [("", "address")], [("", "uint256")], "view"),
    _fn("bond", [("digest", "bytes32"), ("token", "address"), ("amount", "uint256")]),
    _fn("addBounty", [("digest", "bytes32"), ("amount", "uint256")]),
    _fn("payOut", [("account", "address"), ("token", "address")]),
    _fn("payTreasury", [("token", "address")]),
    _fn("requestUnbond", [("digest", "bytes32")]),
    _fn("withdraw", [("digest", "bytes32")]),
]

SUBSTRATE_TOOL_ABI = [
    _fn("registerTool", [("manifestDigest", "bytes32")]),
    _fn("toolAuthor", [("", "bytes32")], [("", "address")], "view"),
    _fn("valueOf", [("token", "address"), ("amount", "uint256")], [("", "uint256")], "view"),
    _fn("parityOf", [("token", "address")], [("", "uint256")], "view"),
    _fn("acceptsToken", [("token", "address")], [("", "bool")], "view"),
    _fn("registry", [], [("", "address")], "view"),
    _fn("poolTokens", [], [("", "address[]")], "view"),
]

_ZERO = "0x" + "00" * 20
# value = amount x parity / 1e30 (Substrate.valueOf)
_PARITY_SCALE = 10 ** 30


# ToolBond.bond's reverts, as an author reads them (trustless-contracts f297209:
# NotInIndex only for a token outside the index, DustAmount for a deposit
# worth less than one value base unit).
_BOND_ERRORS = {
    "DustAmount": "the deposit is worth less than one value base unit at the index "
                  "(DustAmount): bond more",
    "NotInIndex": "the token is not in the value index (NotInIndex): bond in a token "
                  "list_index_tokens lists",
    "WrongToken": "this tool's bond holds another token (WrongToken): a bond keeps its "
                  "token while it holds anything",
    "NotAuthor": "only the tool's Substrate author can bond it (NotAuthor)",
    "NotLiveAgent": "the author's agent key is revoked (NotLiveAgent)",
    "ShortReceipt": "the token delivered less than the amount (ShortReceipt): a token "
                    "that takes a cut on transfer can't bond",
}


def bond_error(error: Any) -> str:
    """A readable reason for a failed bond, keeping the raw error."""
    from eth_utils import keccak
    text = str(error or "")
    for name, why in _BOND_ERRORS.items():
        # A custom error arrives by name or as its 4-byte selector
        sig = f"{name}(uint256,uint256)" if name == "ShortReceipt" else f"{name}()"
        if name in text or "0x" + keccak(text=sig)[:4].hex() in text:
            return why if text == name else f"{why}; {text}"
    return text or "unknown error"


def _amount_for_value(value: int, parity: int) -> int:
    """The fewest base units of a token at ``parity`` worth ``value``."""
    return -(-int(value) * _PARITY_SCALE // int(parity)) if parity > 0 else 0


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
        """The author's bond: what the UI and the bond gate show. ``token``
        is the bond's index token ('' before any bond), ``stake`` and
        ``bounty`` are in its base units, ``value`` is the stake's value
        when posted (6-decimal units, what ``minStake`` is compared with)."""
        def fn(b, s):
            d = _b32(digest)
            row = dict(zip(BOND_FIELDS, b.functions.bonds(d).call()))
            token = str(row["token"])
            return {
                "author": row["author"], "stake": int(row["stake"]),
                "unbond_requested_at": int(row["unbondRequestedAt"]),
                "pending_rounds": int(row["pending"]), "bounty": int(row["bounty"]),
                "token": "" if int(token, 16) == 0 else token.lower(),
                "value": int(row["value"]),
                "bonded": bool(b.functions.isBonded(d).call()),
            }
        return await self._read(fn)

    # -- token choice --------------------------------------------------
    def _index_tokens(self, w3, substrate) -> list[str]:
        """The tokens Substrate accepts now (the value index), lowercase."""
        from ..on_chain import PARITY_KEY_PREFIX, REGISTRY_ENUM_ABI
        found: set[str] = set()
        try:
            reg = w3.eth.contract(address=substrate.functions.registry().call(),
                                  abi=REGISTRY_ENUM_ABI)
            for key in reg.functions.getAllKeys().call():
                key = str(key)
                if key.startswith(PARITY_KEY_PREFIX):
                    t = key[len(PARITY_KEY_PREFIX):].strip().lower()
                    if len(t) == 42 and t.startswith("0x"):
                        found.add(t)
        except Exception as exc:  # noqa: BLE001 (no enumeration: pool tokens only)
            log.debug("Registry key enumeration failed: %s", exc)
        try:
            found.update(str(t).lower() for t in substrate.functions.poolTokens().call())
        except Exception as exc:  # noqa: BLE001
            log.debug("poolTokens read failed: %s", exc)
        from web3 import Web3
        return sorted(t for t in found
                      if substrate.functions.acceptsToken(Web3.to_checksum_address(t)).call())

    def _plan_bond(self, me: str, digest: str, token: Optional[str],
                   amount: Optional[int]) -> dict[str, Any]:
        """What to bond: the token (the bond's own while it holds anything,
        else the caller's, else the one index token the signer holds) and
        the amount (the caller's, else the fewest base units that lift the
        bond's value to minStake). ``{"error": ...}`` when it can't."""
        from web3 import Web3
        from ..on_chain import ERC20_ABI
        w3 = self._w3()
        bond, substrate = self._contracts(w3)
        row = dict(zip(BOND_FIELDS, bond.functions.bonds(_b32(digest)).call()))
        held_token = str(row["token"]).lower()
        holds = int(row["stake"]) or int(row["bounty"]) or int(row["pending"])
        want = str(token or "").strip().lower()
        if want and not (len(want) == 42 and want.startswith("0x")):
            return {"error": f"token must be a 0x address, got {token!r}"}
        if holds and int(held_token, 16) != 0:
            if want and want != held_token:
                return {"error": f"this tool's bond is in {held_token}; a bond keeps its "
                                 "token while it holds anything, so top it up in that token"}
            want = held_token
        index = self._index_tokens(w3, substrate)
        if not index:
            return {"error": "the value index lists no token Substrate accepts: there "
                             "is nothing to bond in"}
        cs_me = Web3.to_checksum_address(me)

        def balance(t: str) -> int:
            return int(w3.eth.contract(address=Web3.to_checksum_address(t), abi=ERC20_ABI)
                       .functions.balanceOf(cs_me).call())
        if not want:
            held = [t for t in index if balance(t) > 0]
            if len(held) != 1:
                return {"error": ("name the index token to bond in (`token`): "
                                  + ("you hold none of them" if not held
                                     else "you hold several") + ". The index: "
                                  + ", ".join(index)),
                        "index_tokens": index}
            want = held[0]
        if want not in index:
            return {"error": f"{want} is not in the value index (Substrate does not "
                             f"accept it). The index: {', '.join(index)}",
                    "index_tokens": index}
        cs_token = Web3.to_checksum_address(want)
        min_stake = int(bond.functions.minStake().call())
        cur_value = int(row["value"]) if want == held_token else 0
        bonded = bool(bond.functions.isBonded(_b32(digest)).call())
        if amount is None:
            if bonded:
                return {"already_bonded": True, "token": want, "amount": 0,
                        "value": cur_value}
            parity = int(substrate.functions.parityOf(cs_token).call())
            # At least one value unit: a top-up that only cancels an unbond
            need = max(min_stake - cur_value, 1)
            amount = _amount_for_value(need, parity)
            for _ in range(3):   # the contract's rounding is the word
                if int(substrate.functions.valueOf(cs_token, amount).call()) >= need:
                    break
                amount += 1
        amount = int(amount)
        if amount <= 0:
            return {"error": "the bond amount must be positive"}
        value = int(substrate.functions.valueOf(cs_token, amount).call())
        if value == 0:
            # ToolBond would revert DustAmount
            return {"error": bond_error("DustAmount") + f" ({amount} base units of {want})"}
        have = balance(want)
        if have < amount:
            return {"error": f"the bond needs {amount} base units of {want} "
                             f"(minStake {min_stake} value units); {me} holds {have}. "
                             "Ask the owner to fund it in that token",
                    "token": want, "amount": amount, "balance": have}
        allowance = int(w3.eth.contract(address=cs_token, abi=ERC20_ABI).functions
                        .allowance(cs_me, bond.address).call())
        return {"token": want, "amount": amount, "value": value,
                "approve": allowance < amount}

    # -- writes: author ------------------------------------------------
    async def register_and_bond(self, private_key: str, digest: str,
                                amount: Optional[int] = None,
                                token: Optional[str] = None) -> dict[str, Any]:
        """Substrate.registerTool (if unclaimed), then the bond in an index
        token: ``token.approve(ToolBond, amount)`` when the allowance falls
        short, then ``ToolBond.bond(digest, token, amount)``.

        ``token``: the bond's own while it holds anything; else the caller's;
        else the one index token the signer holds (several or none: an
        error naming the index). ``amount`` (base units): default the fewest
        that lift the bond's value to minStake; a tool already bonded is
        left as it is. Safe to retry: skips a registration the caller
        already holds and an approve already made.
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
        try:
            plan = await asyncio.to_thread(self._plan_bond, me, digest, token, amount)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"could not read the bond: {exc}",
                    "step": "bond", "txs": txs}
        if plan.get("error"):
            return {"success": False, "step": "bond", "txs": txs, **plan}
        if plan.get("already_bonded"):
            return {"success": True, "tx_hash": txs[-1] if txs else "", "txs": txs,
                    "amount": 0, "token": plan["token"], "value": plan["value"],
                    "note": "already bonded"}
        tok, amt = plan["token"], int(plan["amount"])
        if plan.get("approve"):
            res = await self._approve(private_key, tok, amt)
            if not res.get("success"):
                return {**res, "step": "approve", "txs": txs}
            txs.append(res["tx_hash"])
        from web3 import Web3
        res = await self._write(private_key, lambda b, s, a: b.functions.bond(
            _b32(digest), Web3.to_checksum_address(tok), amt))
        if not res.get("success"):
            return {**res, "error": bond_error(res.get("error")), "step": "bond", "txs": txs}
        txs.append(res["tx_hash"])
        return {"success": True, "tx_hash": res["tx_hash"], "txs": txs, "amount": amt,
                "token": tok, "value": int(plan["value"])}

    async def _approve(self, private_key: str, token: str, amount: int) -> dict[str, Any]:
        """``token.approve(ToolBond, amount)`` from the signer."""
        from web3 import Web3
        from ..on_chain import ERC20_ABI

        def build(b, s, a):
            erc20 = b.w3.eth.contract(address=Web3.to_checksum_address(token), abi=ERC20_ABI)
            return erc20.functions.approve(b.address, int(amount))
        return await self._write(private_key, build, gas_floor=80_000)

    async def add_bounty(self, private_key: str, digest: str, amount: int) -> dict[str, Any]:
        """Add ``amount`` of the bond's token to a bonded tool's bounty
        (approve, then ``ToolBond.addBounty``). Anyone can."""
        st = await self.status(digest)
        if st is None:
            return {"success": False, "error": "chain unreachable"}
        if not st.get("bonded") or not st.get("token"):
            return {"success": False, "error": "a bounty goes only on a bonded tool"}
        txs: list[str] = []
        res = await self._approve(private_key, st["token"], int(amount))
        if not res.get("success"):
            return {**res, "step": "approve"}
        txs.append(res["tx_hash"])
        res = await self._write(private_key, lambda b, s, a: b.functions.addBounty(
            _b32(digest), int(amount)))
        if not res.get("success"):
            return {**res, "step": "addBounty", "txs": txs}
        txs.append(res["tx_hash"])
        return {"success": True, "tx_hash": res["tx_hash"], "txs": txs,
                "token": st["token"], "amount": int(amount)}

    async def owed(self, account: str, token: str) -> Optional[int]:
        """What ToolBond owes ``account`` in ``token`` (bounty parts)."""
        from web3 import Web3
        return await self._read(lambda b, s: int(b.functions.owed(
            Web3.to_checksum_address(account), Web3.to_checksum_address(token)).call()))

    async def pay_out(self, private_key: str, account: str, token: str) -> dict[str, Any]:
        """``ToolBond.payOut(account, token)``: the tokens go to ``account``
        only; anyone can send it. One token per call, so a token that
        freezes ``account`` blocks nothing else."""
        from web3 import Web3
        return await self._write(private_key, lambda b, s, a: b.functions.payOut(
            Web3.to_checksum_address(account), Web3.to_checksum_address(token)))

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
