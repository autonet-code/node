"""Tool rail, daemon side: publishing with a stake, and discovery context.

The chain surface is ``runtime/tool_bond.py``. Two pieces live here:

  - PUBLISH (``publish_with_stake``): every published tool carries an ATN
    stake. Publishing requires a novelty claim in the manifest, a successor
    only by the same author, and ``ToolBondClient.register_and_bond`` signed
    with the author's key (the operator's, for an external agent). No
    ToolBond configured = refused (fail closed), except under the dev
    override ``ATN_TOOL_BOND_DEV_OVERRIDE=1``.
  - DISCOVERY (``annotate_probe_rows``): search rows get the bond state, the
    tool's rating (plain means as of the latest close: correctness,
    simplicity, novelty, the malware share, the charter axes, the REP
    behind them) and author context (REP, other tools and their malware
    share). Context only: the order is relevance. A chain fault marks the
    bond unknown, never breaks search.

The framework rates tools; it does not try to make running them safe
(Decision 2026-11-09). There is no run gate, no containment tier and no
trust label: what a household runs is its own policy, read from the ratings.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from .runtime.tool_bond import (
    DEV_OVERRIDE_ENV, ToolBondClient, _registry_seed, dev_override,
)

log = logging.getLogger(__name__)

_STATUS_TTL_S = 60.0
_STATUS_FAULT_TTL_S = 15.0
_READ_TIMEOUT_S = 8.0
_PROBE_CONCURRENCY = 4
_AUTHOR_TOOLS_MAX = 10

_REP_ABI = [{
    "name": "balanceOf", "type": "function", "stateMutability": "view",
    "inputs": [{"name": "account", "type": "address"}],
    "outputs": [{"name": "", "type": "uint256"}],
}]


def client_for(runtime: Any) -> Any:
    """The runtime's ToolBond client (tests pre-set ``tool_bond_client``)."""
    client = getattr(runtime, "tool_bond_client", None)
    return client if client is not None else ToolBondClient(runtime)


# ---------------------------------------------------------------------------
# The owner's per-tool notes
# ---------------------------------------------------------------------------

class RunOptIns:
    """Per-digest list the owner sets (WS ``set_tool_run_optin``): the
    owner's own notes on which tools they chose to run. It gates nothing. Persisted as JSON; an
    unreadable file reads as empty."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._rows: dict[str, dict[str, Any]] = self._load()

    @staticmethod
    def _norm(digest: str) -> str:
        return str(digest or "").strip().lower().removeprefix("0x")

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            rows = data.get("digests") if isinstance(data, dict) else None
            return {self._norm(k): dict(v) for k, v in (rows or {}).items()
                    if isinstance(v, dict) and len(self._norm(k)) == 64}
        except FileNotFoundError:
            return {}
        except Exception:  # noqa: BLE001 — fail closed: no opt-ins
            log.warning("run opt-in file %s unreadable; treating as empty",
                        self._path, exc_info=True)
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"digests": self._rows}, indent=2,
                                  sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)

    def allows(self, digest: str) -> bool:
        return self._norm(digest) in self._rows

    def set(self, digest: str, allowed: bool, *, name: str = "") -> bool:
        """Add or remove ``digest``. Returns the resulting state."""
        d = self._norm(digest)
        if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
            raise ValueError("digest must be a 64-hex manifest digest")
        if allowed:
            self._rows[d] = {"ts": int(time.time()), "name": str(name or "")}
        else:
            self._rows.pop(d, None)
        self._save()
        return bool(allowed)

    def list(self) -> list[dict[str, Any]]:
        return [{"digest": d, **row} for d, row in sorted(self._rows.items())]


def run_optins(tool_store: Any) -> RunOptIns:
    """The store's opt-in list (created lazily next to registry.jsonl)."""
    optins = getattr(tool_store, "run_optins", None)
    if isinstance(optins, RunOptIns):
        return optins
    base = getattr(tool_store, "_dir", None)
    if base is None:
        raise ValueError("tool store has no data directory for run opt-ins")
    optins = RunOptIns(Path(base) / "run_optins.json")
    tool_store.run_optins = optins
    return optins


def rating_for(runtime: Any, digest: str) -> Optional[dict[str, Any]]:
    """The tool's rating as of the latest close (``tool_store.
    rating_provider``, the anchored means), or None when unknown. Never
    raises."""
    store = getattr(runtime, "tool_store", None)
    provider = getattr(store, "rating_provider", None)
    if provider is None:
        return None
    try:
        rating = provider(str(digest).lower().removeprefix("0x"))
    except Exception as exc:  # noqa: BLE001 (the rating is context)
        log.debug("rating read failed: %s", exc)
        return None
    return dict(rating) if isinstance(rating, dict) else None


# ---------------------------------------------------------------------------
# Publish: novelty + successor rule + stake
# ---------------------------------------------------------------------------

async def publish_with_stake(runtime: Any, record: Any, *,
                             novelty: Any = None) -> dict[str, Any]:
    """Publish ``record`` (already resolved, author-checked by the caller):
    novelty claim present, successor by the same author, stake bonded on
    ToolBond, THEN the local publish flip. Returns a result dict or
    ``{"error": ...}``; the chain step happens before anything is marked
    published, so a failed stake leaves the tool private.

    ``novelty`` given for a private tool registered without it (or with a
    different claim) stamps it in first (``ToolStore.restamp_novelty``):
    the claim is inside the signed manifest, so the published tool is a NEW
    digest, returned as ``digest`` (with ``replaces_private`` naming the
    retired original), on failure too so a retry targets the new one."""
    from nodes.common.library.tool_manifest import (
        normalize_novelty,
    )

    if record.origin == "adopted":
        return {"error": "adopted tools cannot be re-published: the original "
                         "author's publication stands"}
    if record.published:
        return {"digest": record.digest, "name": record.name,
                "published": True, "note": "already published"}
    err = unpublished_dependencies_error(runtime.tool_store, record)
    if err:
        return {"error": err}
    stamped: dict[str, Any] = {}
    if novelty is not None:
        if not isinstance(novelty, dict):
            return {"error": "novelty must be an object {statement, prior_art}"}
        if normalize_novelty(novelty) != record.manifest.get("novelty"):
            try:
                res = runtime.tool_store.restamp_novelty(record.digest, novelty)
            except ValueError as exc:
                return {"error": str(exc)}
            record = runtime.tool_store.get(res["digest"])
            stamped = {"digest": record.digest}
            if res.get("replaced"):
                stamped["replaces_private"] = res["replaced"]
    return {**await _publish_staked(runtime, record), **stamped}


def unpublished_dependencies_error(store: Any, record: Any) -> str:
    """Why a composite cannot be published yet, or "" (#38). Every declared
    dependency must be on the network, or other households can neither find
    nor run the composite: own dependencies must be published first, and an
    adopted dependency is a network tool, published by definition. No
    auto-publishing: each publish is its own staked act."""
    missing: list[str] = []
    for dep in record.manifest.get("dependencies") or []:
        dep_rec = store.get(str(dep))
        if dep_rec is not None and (dep_rec.origin == "adopted"
                                    or dep_rec.published):
            continue
        missing.append(f"{dep_rec.name} ({str(dep)[:16]})" if dep_rec is not None
                       else f"{str(dep)[:16]} (not found locally)")
    if not missing:
        return ""
    return ("this composite depends on tools that are not published: "
            + ", ".join(missing) + ". Publish them first (publish_tool), "
            "then publish this one (if publishing a dependency gave it a new "
            "digest, register this composite again with that digest)")


async def _publish_staked(runtime: Any, record: Any) -> dict[str, Any]:
    from nodes.common.library.tool_manifest import (
        publish_novelty_errors,
    )
    from .tool_store import OWNER_AUTHOR

    store = runtime.tool_store
    manifest = record.manifest
    errors = publish_novelty_errors(manifest)
    if errors:
        return {"error": "; ".join(errors)}
    if manifest.get("version_of"):
        err = store.successor_error(manifest["version_of"],
                                    str(manifest.get("author") or ""))
        if err:
            return {"error": err}

    stake: dict[str, Any]
    if dev_override():
        log.warning("ToolBond DEV OVERRIDE: publishing %s (%s) WITHOUT a stake "
                    "(%s=1; never set this outside local testing)",
                    record.name, record.digest[:16], DEV_OVERRIDE_ENV)
        stake = {"skipped": "dev override", "txs": []}
    else:
        local_author = str(record.local_author or "")
        if not local_author or local_author == OWNER_AUTHOR:
            return {"error": "owner-authored tools cannot be staked: the stake "
                             "is signed by the authoring agent's key and the "
                             "daemon holds no owner key. Have an agent register "
                             "the tool (register_tool) and publish it "
                             "(publish_tool) instead"}
        client = client_for(runtime)
        if not client.available():
            return {"error": "publishing requires a ToolBond stake, and no "
                             "ToolBond/Substrate address or RPC is configured "
                             "(registry key 'tool_bond'); refusing to publish "
                             "unstaked"}
        defn = runtime.get_agent(local_author)
        identity = getattr(defn, "identity", None) if defn else None
        address = str(getattr(identity, "address", "") or "")
        check = getattr(getattr(runtime, "registry", None), "is_operator_identity", None)
        as_operator = bool(check(identity)) if callable(check) and identity is not None else False
        if as_operator and not getattr(identity, "registered_on_chain", False):
            # An external agent publishes as the household's operator
            # (Decision 2026-11-08): the operator signs and funds the bond.
            return {"error": f"you publish as this household's operator "
                             f"({address}), which is not registered on chain "
                             "yet. Ask the owner to register the operator in "
                             "the Autonet app (Household) and to fund it with "
                             "ATN for the tool's stake and with gas. Then "
                             "publish again; nothing else to redo."}
        if not address or not getattr(identity, "registered_on_chain", False):
            return {"error": f"agent '{local_author}' is not registered on chain "
                             "yet. Publishing needs it: ask the owner to "
                             "connect their wallet in the Autonet app and "
                             "approve your registration (Household), and to "
                             "fund you with some ATN for the tool's stake. "
                             "Then publish again; nothing else to redo."}
        if str(manifest.get("author") or "").lower() != address.lower():
            # Registered before the agent was: re-author it now, the same
            # tool under the agent's on-chain identity (a new digest).
            try:
                restamped = store.restamp_author(record.digest)
            except ValueError as exc:
                return {"error": str(exc)}
            record = store.get(restamped["digest"])
            manifest = record.manifest
            if str(manifest.get("author") or "").lower() != address.lower():
                return {"error": "could not re-author this tool as "
                                 f"{address}; register it again"}
        key = runtime.registry.get_agent_key(local_author)
        if not key:
            return {"error": f"No private key stored for agent '{local_author}'."}
        res = await client.register_and_bond(key, record.digest)
        if not res.get("success"):
            return {"error": "ToolBond stake failed"
                             + (f" at {res['step']}" if res.get("step") else "")
                             + f": {res.get('error') or 'unknown error'}",
                    "txs": list(res.get("txs") or [])}
        stake = {"tx_hash": res.get("tx_hash", ""),
                 "txs": list(res.get("txs") or []),
                 "amount": res.get("amount")}
    try:
        store.set_published(record.digest, True)
    except ValueError as exc:  # orphan-author gate
        return {"error": str(exc), "stake": stake}
    _forget_status(runtime, record.digest)
    return {"digest": record.digest, "name": record.name, "published": True,
            "stake": stake}


# ---------------------------------------------------------------------------
# Discovery context
# ---------------------------------------------------------------------------

def _status_cache(runtime: Any) -> dict[str, tuple[Any, float]]:
    cache = getattr(runtime, "_tool_bond_status_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(runtime, "_tool_bond_status_cache", cache)
        except Exception:  # noqa: BLE001
            pass
    return cache


def _forget_status(runtime: Any, digest: str) -> None:
    _status_cache(runtime).pop(str(digest).lower(), None)


async def _guarded(sem: asyncio.Semaphore, coro_fn, *args) -> Any:
    async with sem:
        try:
            return await asyncio.wait_for(coro_fn(*args), _READ_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — unknown, never fatal
            log.debug("tool bond context read failed: %s", exc)
            return None


async def cached_status(runtime: Any, client: Any, digest: str,
                        sem: asyncio.Semaphore) -> Optional[dict[str, Any]]:
    """The bond (``client.status``) with a short cache; None = unknown."""
    key = str(digest).lower()
    cache = _status_cache(runtime)
    now = time.monotonic()
    hit = cache.get(key)
    if hit is not None and hit[1] > now:
        return hit[0]
    st = await _guarded(sem, client.status, digest)
    cache[key] = (st, now + (_STATUS_TTL_S if st is not None else _STATUS_FAULT_TTL_S))
    return st


def _rep_reader(runtime: Any):
    """Async ``address -> int | None`` over RepToken.balanceOf, or None when
    no RepToken address / RPC is configured."""
    cfg = getattr(getattr(runtime, "_config", None), "rpb", None)
    addr = str(getattr(cfg, "rep_token_address", "") or "")
    rpc = str(getattr(cfg, "rpc_url", "") or "")
    if not addr or not rpc:
        seed = _registry_seed(runtime)
        addr = addr or str(seed.get("rep_token_address") or "")
        rpc = rpc or str(seed.get("rpc_url") or "")
    if not addr or not rpc:
        return None

    async def read(author: str) -> Optional[int]:
        def run() -> int:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5}))
            c = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=_REP_ABI)
            return int(c.functions.balanceOf(Web3.to_checksum_address(author)).call())
        return await asyncio.to_thread(run)
    return read


def _is_address(value: str) -> bool:
    return isinstance(value, str) and len(value) == 42 and value.startswith("0x")


async def annotate_probe_rows(runtime: Any, rows: list[dict[str, Any]],
                              ) -> list[dict[str, Any]]:
    """Add the bond state, the tool's rating and author context to search
    rows. Every row stays: an unbonded tool shows ``bonded: false`` and a
    tool rated malware shows its share (Decision 2026-11-09: disqualified
    means unpaid, nothing else). A chain fault marks the bond unknown
    (``bonded: None``). Never raises; order kept."""
    if not rows:
        return rows
    try:
        return await _annotate(runtime, rows)
    except Exception as exc:  # noqa: BLE001 — context is additive
        log.debug("probe bond annotation failed: %s", exc)
        for row in rows:
            row.setdefault("bonded", None)
            row.setdefault("rating", rating_for(runtime, row.get("digest", "")))
        return rows


def _malware_of(rating: Optional[dict[str, Any]]) -> Optional[float]:
    return (rating or {}).get("malware")


async def _annotate(runtime: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
    client = client_for(runtime)
    override = dev_override()
    chain = (not override) and bool(client.available())

    # Author -> tools this node can see (search rows + the local store).
    by_author: dict[str, dict[str, str]] = {}
    for row in rows:
        a = str(row.get("author") or "").lower()
        if a:
            by_author.setdefault(a, {})[row["digest"]] = row.get("name", "")
    store = getattr(runtime, "tool_store", None)
    for rec in (getattr(store, "_records", {}) or {}).values():
        a = str(rec.manifest.get("author") or "").lower()
        if a in by_author:
            by_author[a].setdefault(rec.digest, rec.name)

    statuses: dict[str, Optional[dict[str, Any]]] = {}
    if chain:
        wanted = list(dict.fromkeys(r["digest"] for r in rows))
        got = await asyncio.gather(*(cached_status(runtime, client, d, sem)
                                     for d in wanted))
        statuses = dict(zip(wanted, got))

    rep: dict[str, Optional[int]] = {}
    reader = _rep_reader(runtime) if chain else None
    if reader is not None:
        authors = [a for a in by_author if _is_address(a)]
        got = await asyncio.gather(*(_guarded(sem, reader, a) for a in authors))
        rep = dict(zip(authors, got))

    for row in rows:
        digest = row["digest"]
        st = statuses.get(digest)
        if st is not None:
            row.update({"bonded": bool(st.get("bonded")),
                        "stake": str(st.get("stake", ""))})
        else:
            row["bonded"] = True if override else None
        row["rating"] = rating_for(runtime, digest)
        author = str(row.get("author") or "").lower()
        others = {d: n for d, n in by_author.get(author, {}).items() if d != digest}
        other_rows = [{"digest": d, "name": n,
                       "malware": _malware_of(rating_for(runtime, d))}
                      for d, n in list(others.items())[:_AUTHOR_TOOLS_MAX]]
        shares = [o["malware"] for o in other_rows if o["malware"] is not None]
        r = rep.get(author)
        row["author_context"] = {
            "rep": None if r is None else str(r),
            "other_tools": other_rows,
            # The highest malware share among the author's other tools this
            # node has ratings for; None = none rated yet.
            "max_other_malware": max(shares) if shares else None,
        }
    return rows
