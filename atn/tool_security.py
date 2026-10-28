"""Tool security rail, daemon side (docs/tool_substrate.md, Decision
2026-10-13: staked tools; docs/multi_jurisdiction.md decision 10: staked
reviews rate them, with no verdict flag).

Three pieces live here; the chain surface is ``runtime/tool_bond.py``:

  - PUBLISH (``publish_with_stake``): every published tool carries an ATN
    stake. Publishing requires a novelty claim in the manifest, a successor
    only by the same author, and ``ToolBondClient.register_and_bond`` signed
    with the AUTHOR AGENT's key. No ToolBond configured = refused (fail
    closed), except under the dev override ``ATN_TOOL_BOND_DEV_OVERRIDE=1``.
  - RUN (``run_clearance`` + ``containment_tier``): a network tool
    (authored elsewhere, i.e. an adopted record the owner approved for
    that exact digest) runs only while ``isBonded``, and its sandbox
    follows the network's settled harm for that digest (Decision
    2026-10-27): denied, strict, or its declared capabilities. Locally authored
    tools are not gated. Enforced in ``ToolStore.call`` (the one dispatch
    choke point) and ahead of a core swap in
    ``runtime/shell_provider.dispatch_shell`` so a refused swap falls back.
    ``RunOptIns`` is kept as the owner's per-digest note list for the UI; it
    no longer changes what runs.
  - DISCOVERY (``annotate_probe_rows``): probe_tools drops tools that are
    known not to be bonded and adds the bond, the ToolBond rating (per-axis
    REP-weighted means, harm, and the REP mass behind each) plus author
    context (REP, other tools and their harm). Context only: ranking stays
    the close's. A chain fault marks a row unknown, never breaks probe.
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
    DEV_OVERRIDE_ENV, ToolBondClient, _registry_seed, dev_override, gate_for,
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


def is_network_tool(record: Any) -> bool:
    """Code authored elsewhere: the run gate applies. Adopted records are
    the only foreign code a ToolStore holds (peer manifests are fetched for
    inspection, never executed until adopted)."""
    return str(getattr(record, "origin", "") or "") == "adopted"


def client_for(runtime: Any) -> Any:
    """The runtime's ToolBond client (tests pre-set ``tool_bond_client``)."""
    client = getattr(runtime, "tool_bond_client", None)
    return client if client is not None else ToolBondClient(runtime)


# ---------------------------------------------------------------------------
# Owner opt-in: run an unvetted (but bonded) network tool, per digest
# ---------------------------------------------------------------------------

class RunOptIns:
    """Per-digest list the owner sets (WS ``set_tool_run_optin``). Since
    staked reviews replaced the vetted flag it no longer gates a run; the
    UI keeps it as the owner's explicit trust notes. Persisted as JSON; an
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


async def run_clearance(runtime: Any, record: Any) -> tuple[bool, str]:
    """May this node run ``record``? ``(ok, reason)``, never raises.

    Own tools: always (not gated). Network tools: bonded on ToolBond (the
    author's stake stands behind the code; a settled harm slashes it in
    proportion, so a tool rated harmful loses its bond). Refusals start with
    "not bonded"."""
    if not is_network_tool(record):
        return True, "locally authored"
    digest = str(getattr(record, "digest", "") or "")
    try:
        bonded, why = await gate_for(runtime).is_bonded(digest)
        if not bonded:
            return False, f"not bonded ({why}): unstaked, unbonding or slashed"
        return True, "bonded"
    except Exception as exc:  # noqa: BLE001 — fail closed
        log.warning("run clearance for %s failed: %s", digest[:16], exc)
        return False, f"not bonded (clearance check failed: {exc})"


# ---------------------------------------------------------------------------
# Containment by consensus standing (Decision 2026-10-27, docs/tool_substrate.md)
# ---------------------------------------------------------------------------
# A network tool's sandbox follows the network's settled judgement of its
# exact code (per digest, from the anchor-pinned ToolBond review snapshot):
#
#   denied    settled harm >= deny_harm, with at least trust_mass_bps of REP
#             supply behind the harm reviews: refused at adoption and at run.
#   declared  harm <  trust_harm with that same REP behind it: the manifest's
#             declared net/fs/spawn.
#   strict    anything else (new, thinly reviewed, or contested): no net, no
#             fs outside its sandbox, no spawn, whatever the manifest says.
#
# Harm is ToolBond's 0..100 scale. One reviewer can never deny or free a tool
# on its own: both ends need REP mass. Defaults are PROVISIONAL; the DAO sets
# them (jurisdiction.<id>.containment_*), and the owner's own tools never pass
# through here.
from nodes.common.containment import (  # noqa: E402  (one rule, shared with the close)
    CONTAINMENT_DECLARED, CONTAINMENT_DENIED, CONTAINMENT_STRICT,
    CONTAINMENT_DEFAULTS as _CONTAINMENT_DEFAULTS, tier_for_standing,
)


def containment_params(runtime: Any) -> dict[str, float]:
    cfg = getattr(getattr(runtime, "_config", None), "autonet", None)
    out = dict(_CONTAINMENT_DEFAULTS)
    for key in out:
        val = getattr(cfg, f"containment_{key}", None) if cfg is not None else None
        if val not in (None, ""):
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                pass
    return out




def containment_tier(runtime: Any, record: Any) -> tuple[str, str]:
    """``(tier, reason)`` for running ``record`` here; never raises. Own
    tools return ``("own", ...)`` and keep the authored path."""
    if not is_network_tool(record):
        return "own", "locally authored"
    store = getattr(runtime, "tool_store", None)
    provider = getattr(store, "security_standing_provider", None)
    standing = None
    if provider is not None:
        try:
            standing = provider(str(getattr(record, "digest", "") or ""))
        except Exception as exc:  # noqa: BLE001 — unknown standing = strict
            log.debug("security standing read failed: %s", exc)
    return tier_for_standing(standing, containment_params(runtime))


def finally_denied(runtime: Any, digest: str) -> bool:
    """True once the network's harm verdict on ``digest`` is past appeal
    (settled and not overruled, or its appeal window expired). Containment
    acts on harm still under appeal; discovery waits for the final verdict
    before hiding a tool. Unknown standing is never hidden."""
    store = getattr(runtime, "tool_store", None)
    provider = getattr(store, "security_standing_provider", None)
    if provider is None:
        return False
    try:
        standing = provider(str(digest), include_cooling=False)
    except Exception as exc:  # noqa: BLE001 — unknown = shown
        log.debug("final standing read failed: %s", exc)
        return False
    return tier_for_standing(standing, containment_params(runtime))[0] == "denied"


def rating_view(rating: Optional[dict[str, Any]],
                rep_supply: Optional[int] = None) -> dict[str, Any]:
    """A ToolBond rating (``ToolBondClient.rating``) as display numbers:
    per-axis mean in -1..1 and REP mass, harm mean in 0..1 and its mass,
    and the mass as a share of REP supply when known. None values mean no
    REP has rated that axis yet."""
    if not rating:
        return {"chain_rating": None, "harm": None, "harm_mass": None}
    axes = rating.get("axes") or {}
    harm = rating.get("harm") or {}

    def share(mass: Any) -> Optional[float]:
        if not rep_supply:
            return None
        return round(int(mass or 0) / int(rep_supply), 9)
    return {
        "chain_rating": {a: v.get("mean") for a, v in sorted(axes.items())},
        "rating_mass": {a: str(v.get("mass", 0)) for a, v in sorted(axes.items())},
        "rating_mass_share": ({a: share(v.get("mass")) for a, v in sorted(axes.items())}
                              if rep_supply else None),
        "harm": harm.get("mean"),
        "harm_mass": str(harm.get("mass", 0)),
    }


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
    from nodes.common.world_model_substrate.tool_manifest import (
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
    from nodes.common.world_model_substrate.tool_manifest import (
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
        if not address or not getattr(identity, "registered_on_chain", False):
            return {"error": f"agent '{local_author}' is not registered on-chain; "
                             "register it (register_on_chain), then register "
                             "the tool again so the agent is its author"}
        if str(manifest.get("author") or "").lower() != address.lower():
            return {"error": "this manifest's author is "
                             f"{manifest.get('author')!r}, not the agent's own "
                             f"address {address}: register the tool again now "
                             "that the agent is registered, then publish that"}
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
    except ValueError as exc:  # orphan-author / guest-stamp gates
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


async def _bond_view(client: Any, digest: str) -> Optional[dict[str, Any]]:
    """The bond plus the settled ToolBond rating on the active charter's
    axes, or None when the bond read fails."""
    st = await client.status(digest)
    if st is None:
        return None
    st = dict(st)
    reader = getattr(client, "rating", None)
    if reader is not None:
        from nodes.common.world_model_substrate.charter import active_charter
        try:
            st["rating"] = await reader(digest, list(active_charter().ids))
        except Exception as exc:  # noqa: BLE001 (the rating is context)
            log.debug("ToolBond rating read failed: %s", exc)
            st["rating"] = None
    return st


async def cached_status(runtime: Any, client: Any, digest: str,
                        sem: asyncio.Semaphore) -> Optional[dict[str, Any]]:
    """Bond + rating (``_bond_view``) with a short cache; None = unknown."""
    key = str(digest).lower()
    cache = _status_cache(runtime)
    now = time.monotonic()
    hit = cache.get(key)
    if hit is not None and hit[1] > now:
        return hit[0]
    st = await _guarded(sem, _bond_view, client, digest)
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
    """Add bond state, the ToolBond rating and author context to probe rows;
    drop rows whose digest is known NOT to be bonded. Unknown (no ToolBond
    configured, RPC fault) keeps the row with ``bonded: None``. Never
    raises; order kept."""
    if not rows:
        return rows
    try:
        return await _annotate(runtime, rows)
    except Exception as exc:  # noqa: BLE001 — context is additive
        log.debug("probe bond annotation failed: %s", exc)
        for row in rows:
            row.setdefault("bonded", None)
            row.setdefault("harm", None)
        return rows


def _harm_of(st: Optional[dict[str, Any]]) -> Optional[float]:
    harm = ((st or {}).get("rating") or {}).get("harm") or {}
    return harm.get("mean")


async def _annotate(runtime: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
    client = client_for(runtime)
    override = dev_override()
    chain = (not override) and bool(client.available())

    # Author -> tools this node can see (probe rows + the local store).
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
        wanted: list[str] = [r["digest"] for r in rows]
        for tools in by_author.values():
            wanted.extend(list(tools)[:_AUTHOR_TOOLS_MAX])
        wanted = list(dict.fromkeys(wanted))
        got = await asyncio.gather(*(cached_status(runtime, client, d, sem)
                                     for d in wanted))
        statuses = dict(zip(wanted, got))

    rep: dict[str, Optional[int]] = {}
    reader = _rep_reader(runtime) if chain else None
    if reader is not None:
        authors = [a for a in by_author if _is_address(a)]
        got = await asyncio.gather(*(_guarded(sem, reader, a) for a in authors))
        rep = dict(zip(authors, got))

    out: list[dict[str, Any]] = []
    for row in rows:
        digest = row["digest"]
        st = statuses.get(digest)
        if st is not None and not st.get("bonded"):
            continue  # unstaked / unbonding / slashed: not in the economy
        if finally_denied(runtime, digest):
            continue  # harmful by the network's final verdict
        if st is not None:
            row.update({"bonded": True, "stake": str(st.get("stake", "")),
                        **rating_view(st.get("rating"))})
        else:
            row.update({"bonded": True if override else None,
                        **rating_view(None)})
        author = str(row.get("author") or "").lower()
        others = {d: n for d, n in by_author.get(author, {}).items() if d != digest}
        other_rows = []
        for d, n in list(others.items())[:_AUTHOR_TOOLS_MAX]:
            ost = statuses.get(d)
            other_rows.append({"digest": d, "name": n,
                               "bonded": None if ost is None else bool(ost.get("bonded")),
                               "harm": _harm_of(ost)})
        harms = [o["harm"] for o in other_rows if o["harm"] is not None]
        r = rep.get(author)
        row["author_context"] = {
            "rep": None if r is None else str(r),
            "other_tools": other_rows,
            # The highest settled harm among the author's other tools this
            # node can see; None = unknown or none rated yet.
            "max_other_harm": max(harms) if harms else None,
        }
        out.append(row)
    return out
