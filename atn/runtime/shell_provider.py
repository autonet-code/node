"""Core built-in swap resolution: LIVE behind three gates (Decision 2026-10-12).

docs/tool_substrate.md: "Resident tools, loadouts, distros", the 2026-07-29
decision (the shell bundle is runnable; the swap is gated off) and the
2026-10-12 decision that supersedes its deciding factor.

WHAT THIS DOES. Lets a pinned tool REPLACE the daemon's built-in shell/file
tools for one agent: agent -> AgentDefinition.loadout (an owner-bound distro
digest) -> distro manifest -> dependency digests -> the record whose
``capabilities.provides`` claims e.g. "bash". The harness then runs that
substrate tool instead of its own component, which is what the distro DAG,
adoption credit and mint fan-out were built for.

WHY IT WAS OFF (2026-07-29), AND WHAT CHANGED. Nothing contains this tool
class: a substituted ``bash`` or ``read_file`` sees every command string and
every file body the agent touches (credentials included), while needing no
evasion to exfiltrate. That analysis stands (and since 2026-11-09 the daemon
contains no tool at all). What changed is the deciding factor: the owner decided
(2026-10-12) that every part of the harness should be swappable through the
tool library, with the market selecting the best, and that the residual risk
is carried by STAKE and EVIDENCE instead of being avoided by a release-only
upgrade path. So the old three preconditions are replaced by these gates,
ALL of which must hold for a swap to happen:

  (a) Core-replacement approval. The record was adopted with an explicit
      owner acknowledgment (``ToolRecord.core_replacement_ack``), a
      distinct act from ordinary adoption approval: ``approve_adoption``
      rejects a tool whose manifest claims core names unless the approval
      carries ``core_replacement_ack``; the UI shows an extra warning
      ("This tool replaces <names> and will see every command and file your
      agent touches"). Authored (local) records cannot replace core tools.
  (b) Bond. ``ToolBond.isBonded(manifestDigest)`` is true on-chain
      (``runtime/tool_bond.py``; registry key ``tool_bond``). No address
      configured => refused (fail closed) unless ATN_TOOL_BOND_DEV_OVERRIDE=1.
  (c) Per-agent binding. Only an agent whose owner-bound loadout
      (AgentDefinition.loadout) lists the provider as a dependency gets the
      swap. The daemon-wide reference distro never swaps anything.

A provider must also be SEALED (no declared dependencies), so every call goes
through the one path where evidence capture works.

EVIDENCE. Every swapped call writes a record (tool digest, input, output,
timestamps), signs it with the agent key, stores it in the blob store,
records its digest on the tool record (``last_evidence_digest``) and in
``evidence_index.jsonl``, and logs it. A reviewer can cite it as evidence.
What the provider does while it runs is not observed.

Any refused gate falls back to the BUILT-IN, never to an error: the built-in
is the trusted default at every step. ``SHELL_SWAP_ENABLED`` stays as the
kernel kill switch (also ATN_CORE_SWAP_DISABLED=1).

PERF: the default path (no loadout bound) is one dict lookup and returns None.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Kernel kill switch. True since 2026-10-12: swaps happen only through the
# three gates in the module docstring.
SHELL_SWAP_ENABLED = True
KILL_SWITCH_ENV = "ATN_CORE_SWAP_DISABLED"

_EVIDENCE_OUTPUT_CAP = 64_000


def swap_enabled() -> bool:
    return SHELL_SWAP_ENABLED and os.environ.get(KILL_SWITCH_ENV, "") != "1"


def _shell_names() -> frozenset[str]:
    """The overridable set. A provider may not claim a name outside it."""
    try:
        from ..shell_tools import SHELL_TOOL_EXECUTORS
        return frozenset(SHELL_TOOL_EXECUTORS)
    except Exception:  # noqa: BLE001 — fail closed
        return frozenset()


def core_claims(manifest: dict[str, Any] | None) -> list[str]:
    """Core built-in names a manifest claims to provide (sorted, clamped to
    the overridable set). Empty = an ordinary tool."""
    caps = (manifest or {}).get("capabilities") or {}
    raw = caps.get("provides") if isinstance(caps, dict) else None
    if not isinstance(raw, (list, tuple)):
        return []
    names = _shell_names()
    return sorted({s.strip() for s in raw
                   if isinstance(s, str) and s.strip() in names})


def resolve_shell_provider(runtime: Any, agent_id: str) -> dict[str, Any]:
    """Map ``tool_name -> ToolRecord`` for providers in ``agent_id``'s OWN
    bound loadout that pass gates (a) and (c). The bond gate (b) is async and
    checked at dispatch. {} means "use the built-in". Never raises."""
    if not swap_enabled():
        return {}
    try:
        from ..harness_policy import agent_loadout
        loadout = agent_loadout(runtime, agent_id)
        store = getattr(runtime, "tool_store", None)
        if not store or not loadout:
            return {}
        distro = store.resolve(loadout)
        if distro is None or not distro.enabled:
            return {}
        out: dict[str, Any] = {}
        claims: dict[str, str] = {}
        for dep in (distro.manifest.get("dependencies") or []):
            record = store.resolve(str(dep))
            if record is None or not record.enabled:
                continue
            # Gate (a): adopted with the explicit core-replacement ack. The
            # daemon's own atn_shell module (authored) is never an override.
            if record.origin != "adopted" or not record.core_replacement_ack:
                continue
            if record.manifest.get("dependencies"):
                continue  # sealed providers only (evidence capture)
            if not store.allowed(agent_id, record):
                continue
            for claimed in core_claims(record.manifest):
                prior = claims.get(claimed)
                # Deterministic on conflict: lexicographically smaller digest
                # wins (never dict / registration order).
                if prior is not None and prior <= record.digest:
                    log.warning("core provider conflict for %r: %s and %s",
                                claimed, prior[:16], record.digest[:16])
                    continue
                claims[claimed] = record.digest
                out[claimed] = record
        return out
    except Exception:  # noqa: BLE001 — never break dispatch
        log.debug("core provider resolution failed", exc_info=True)
        return {}


async def dispatch_shell(runtime: Any, agent_id: str, name: str,
                         tool_input: dict) -> Optional[dict]:
    """Run ``name`` through a swapped provider, or return None.

    None means "no override — the caller runs the built-in". The caller MUST
    treat None as fall-through rather than as an error.
    """
    if not swap_enabled() or name not in _shell_names():
        return None
    providers = resolve_shell_provider(runtime, agent_id)
    record = providers.get(name)
    if record is None:
        return None
    # Gate (b): bonded on ToolBond (fail closed).
    from .tool_bond import gate_for
    bonded, reason = await gate_for(runtime).is_bonded(record.digest)
    await note_swap_state(runtime, agent_id, _bond_row(record, bonded, reason))
    if not bonded:
        log.debug("core provider %s for %r refused for agent %s (%s); "
                  "running the built-in", record.digest[:16], name,
                  agent_id, reason)
        return None
    store = runtime.tool_store
    started = time.time()
    try:
        out = await store.call(
            record, {"tool": name, "args": tool_input}, caller_id=agent_id)
    except Exception as exc:  # noqa: BLE001
        out = {"error": f"core provider {record.digest[:16]} failed: {exc}"}
    ended = time.time()
    try:
        capture_evidence(runtime, record, agent_id, name, tool_input, out,
                         started, ended)
    except Exception:  # noqa: BLE001 — evidence must not break the call
        log.warning("core provider evidence capture failed", exc_info=True)
    if "error" in out:
        return {"error": out["error"]}
    envelope = out.get("result")
    if not isinstance(envelope, dict):
        return {"error": "shell provider returned a malformed envelope"}
    if envelope.get("ok"):
        return envelope.get("result")
    return {"error": str(envelope.get("error") or "shell provider error")}


# ---------------------------------------------------------------------------
# Swap status (owner-visible fallback)
# ---------------------------------------------------------------------------
#
# A failed gate keeps the built-in running (fail closed), but the owner must
# see it: an agent whose loadout names a core-replacing tool and is quietly
# running the built-in is not running the harness the owner picked. Status is
# per (agent, tool). HARNESS_SWAP_STATUS fires on a state TRANSITION only
# (fallback reason changed, recovered to active, or the tool left the
# loadout), never per call, so a refused bash call in a loop costs nothing.

FALLBACK_REASONS = ("not_approved", "not_bonded", "bond_check_error",
                    "not_in_loadout", "swaps_disabled")


def _row(record: Any, reason: str = "", detail: str = "",
         state: str = "fallback") -> dict[str, Any]:
    return {"tool_digest": record.digest,
            "tool_name": str(record.manifest.get("name") or ""),
            "provides": core_claims(record.manifest),
            "state": state, "reason": reason if state == "fallback" else "",
            "detail": detail}


def _bond_row(record: Any, bonded: bool, reason: str) -> dict[str, Any]:
    if bonded:
        return _row(record, state="active", detail=reason)
    code = "not_bonded" if reason.startswith("not bonded") else "bond_check_error"
    return _row(record, code, reason)


def _gate_rows(runtime: Any, agent_id: str) -> list[dict[str, Any]]:
    """Every core-claiming dep of ``agent_id``'s bound loadout with its
    sync-gate verdict: a fallback row, or state "eligible" (bond unchecked).
    Mirrors resolve_shell_provider (which stays the dispatch authority)."""
    from ..harness_policy import agent_loadout
    store = getattr(runtime, "tool_store", None)
    loadout = agent_loadout(runtime, agent_id)
    if not store or not loadout:
        return []
    distro = store.resolve(loadout)
    if distro is None:
        return []
    winners = {r.digest for r in resolve_shell_provider(runtime, agent_id).values()}
    # The reference distro's resident modules (atn_shell claims every core
    # name) ARE the built-ins, never an override: not reported.
    ref = store.resolve(getattr(store, "active_loadout", "") or "")
    resident = set(ref.manifest.get("dependencies") or []) if ref else set()
    rows: list[dict[str, Any]] = []
    for dep in (distro.manifest.get("dependencies") or []):
        if str(dep) in resident:
            continue
        record = store.resolve(str(dep))
        if record is None or not core_claims(record.manifest):
            continue
        if not swap_enabled():
            row = _row(record, "swaps_disabled",
                       "core swaps are switched off on this daemon")
        elif not distro.enabled:
            row = _row(record, "not_in_loadout", "the bound harness distro is disabled")
        elif not record.enabled:
            row = _row(record, "not_approved", "the tool is disabled")
        elif record.origin != "adopted" or not record.core_replacement_ack:
            row = _row(record, "not_approved",
                       "adopted without the core-replacement acknowledgment"
                       if record.origin == "adopted"
                       else "only an adopted tool can replace a core tool")
        elif record.manifest.get("dependencies"):
            row = _row(record, "not_in_loadout",
                       "a provider with dependencies cannot replace core tools")
        elif not store.allowed(agent_id, record):
            row = _row(record, "not_in_loadout", "the tool is not granted to this agent")
        elif record.digest not in winners:
            row = _row(record, "not_in_loadout",
                       "another provider in the loadout claims the same core tool")
        else:
            row = _row(record, state="eligible")
        rows.append(row)
    return rows


def _tracker(runtime: Any) -> dict[tuple[str, str], str]:
    t = getattr(runtime, "_core_swap_state", None)
    if not isinstance(t, dict):
        t = {}
        try:
            setattr(runtime, "_core_swap_state", t)
        except Exception:  # noqa: BLE001
            pass
    return t


async def _emit(runtime: Any, agent_id: str, row: dict[str, Any]) -> None:
    events = getattr(runtime, "events", None)
    if events is None:
        return
    from ..events import Event, EventType
    try:
        await events.emit(Event(type=EventType.HARNESS_SWAP_STATUS,
                                source=agent_id,
                                data={"agent_id": agent_id, **row}))
    except Exception:  # noqa: BLE001 — a notice must never break dispatch
        log.debug("swap status emit failed", exc_info=True)


async def note_swap_state(runtime: Any, agent_id: str, row: dict[str, Any]) -> None:
    """Record one (agent, tool) state; emit only on a transition. A tool
    that is active from the first look is silent."""
    state = row["reason"] if row["state"] == "fallback" else "active"
    tracker = _tracker(runtime)
    key = (agent_id, row["tool_digest"])
    prior = tracker.get(key)
    tracker[key] = state
    if prior == state or (prior is None and state == "active"):
        return
    if state != "active":
        log.warning("agent %s: core provider %s (%s) falls back to the built-in: "
                    "%s (%s)", agent_id, row["tool_digest"][:16],
                    ",".join(row["provides"]), state, row["detail"])
    await _emit(runtime, agent_id, row)


async def swap_status(runtime: Any, agent_id: str) -> list[dict[str, Any]]:
    """Current per-tool swap status for ``agent_id`` (bond gate included)
    and sync the transition tracker. [] when no core-replacing tool is in
    the agent's loadout. Never raises."""
    try:
        rows = _gate_rows(runtime, agent_id)
        if rows and any(r["state"] == "eligible" for r in rows):
            from .tool_bond import gate_for
            gate = gate_for(runtime)
            for i, r in enumerate(rows):
                if r["state"] != "eligible":
                    continue
                record = runtime.tool_store.resolve(r["tool_digest"])
                bonded, reason = await gate.is_bonded(r["tool_digest"])
                rows[i] = _bond_row(record, bonded, reason)
        for r in rows:
            await note_swap_state(runtime, agent_id, r)
        tracker = _tracker(runtime)
        present = {r["tool_digest"] for r in rows}
        for key in [k for k in tracker if k[0] == agent_id and k[1] not in present]:
            prior = tracker.pop(key)
            if prior != "active":
                await _emit(runtime, agent_id, {
                    "tool_digest": key[1], "tool_name": "", "provides": [],
                    "state": "gone", "reason": "",
                    "detail": "no longer in the agent's loadout"})
        return rows
    except Exception:  # noqa: BLE001
        log.debug("swap status failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Evidence capture
# ---------------------------------------------------------------------------

EVIDENCE_KIND = "core_replacement_trace"
CAPTURE_LIMITS = (
    "The call's input and output only. The daemon observes nothing the "
    "provider does while it runs (no audit hook, Decision 2026-11-09).")


def canonical_evidence_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sign(runtime: Any, agent_id: str, body: dict[str, Any]) -> tuple[str, str]:
    """(signer_address, signature_hex) with the agent key; ("", "") if the
    agent has no key (unregistered) or signing is unavailable."""
    registry = getattr(runtime, "registry", None)
    key = None
    if registry is not None and hasattr(registry, "get_agent_key"):
        try:
            key = registry.get_agent_key(agent_id)
        except Exception:  # noqa: BLE001
            key = None
    if not key:
        return "", ""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        signed = Account.sign_message(
            encode_defunct(canonical_evidence_bytes(body)), private_key=key)
        return Account.from_key(key).address, signed.signature.hex()
    except Exception:  # noqa: BLE001
        log.debug("evidence signing unavailable", exc_info=True)
        return "", ""


def capture_evidence(runtime: Any, record: Any, agent_id: str, name: str,
                     tool_input: dict, out: dict,
                     started: float, ended: float) -> str:
    """Build, sign and blob-store the evidence record. Returns its digest."""
    try:
        output_text = json.dumps(out, default=str)
    except (TypeError, ValueError):
        output_text = str(out)
    body = {
        "kind": EVIDENCE_KIND,
        "version": 2,
        "tool_digest": record.digest,
        "tool_name": record.name,
        "tool_author": record.author,
        "replaces": name,
        "agent_id": agent_id,
        "input": tool_input,
        "output": output_text[:_EVIDENCE_OUTPUT_CAP],
        "output_truncated": len(output_text) > _EVIDENCE_OUTPUT_CAP,
        "started_ts": round(started, 3),
        "ended_ts": round(ended, 3),
        "capture_limits": CAPTURE_LIMITS,
    }
    signer, sig = _sign(runtime, agent_id, body)
    evidence = {"body": body, "signer": signer, "sig": sig,
                "sig_scheme": "eip191:canonical_json" if sig else "unsigned"}
    store = runtime.tool_store
    digest = store._blob_store().add_json(evidence) or ""
    if not digest:
        return ""
    record.last_evidence_digest = digest
    try:
        store._persist()
    except Exception:  # noqa: BLE001
        log.debug("persisting last_evidence_digest failed", exc_info=True)
    try:
        idx = Path(store._dir) / "evidence_index.jsonl"
        with idx.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": int(ended), "tool_digest": record.digest,
                "tool_name": record.name, "replaces": name,
                "agent_id": agent_id, "evidence_digest": digest,
                "signed": bool(sig),
            }) + "\n")
    except OSError:
        log.debug("evidence index append failed", exc_info=True)
    log.info("core-replacement evidence %s (tool %s, %s, agent %s, %s)",
             digest, record.digest[:16], name, agent_id,
             "signed" if sig else "UNSIGNED")
    return digest


def list_evidence(store: Any, tool_digest: str = "",
                  limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first rows of the local evidence index."""
    idx = Path(store._dir) / "evidence_index.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        with idx.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if tool_digest and row.get("tool_digest") != tool_digest:
                    continue
                rows.append(row)
    except OSError:
        return []
    return list(reversed(rows))[:max(1, min(int(limit or 50), 500))]
