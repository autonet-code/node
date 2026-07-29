"""Shell-bundle swap resolution — DESIGNED, WIRED, AND DISABLED.

docs/tool_substrate.md — "Resident tools, loadouts, distros".

WHAT THIS WOULD DO. Let an adopted pinned tool REPLACE the daemon's built-in
shell bundle for an agent whose active loadout names a substitute: agent ->
active_loadout digest -> distro manifest -> dependency digests -> the record
whose ``capabilities.provides`` claims "bash". The harness would then swap out
its own components for tools from the substrate, which is the premise the
distro DAG, adoption credit, and mint fan-out were all built for.

WHY IT IS OFF. Containment for this specific tool class does not exist.

``tool_guard.py`` has exactly three checks: socket.* (net), open-outside-prefix
(fs), and a spawn-event tuple. A shell bundle needs net, fs, AND spawn by
definition — it runs commands and reads files. Every branch of the guard falls
through, so the audit hook is a literal no-op for this tool. The destination
allowlist added in 664ad0a does not save it either: it is only populated when
the tool declares secrets, and ``spawn: True`` defeats it regardless (curl in a
child process is not audited by the parent's hook). What survives on the
adopted branch is the env scrub and a sandbox cwd — and cwd is meaningless for
a tool whose read_file/write_file take absolute paths.

The marginal exposure is uniquely high HERE, not merely equal to other tools:
a substituted ``bash`` sees every command string the agent runs (git remotes,
ssh invocations, curl-with-token-in-URL), and a substituted ``read_file`` sees
the body of every file the agent reads — which is the plane where credentials
actually live, and precisely the plane the tool-secret binding does NOT cover
(that binding clamps which VAULT services a tool may request; it says nothing
about a tool that simply reads ~/.aws/credentials as a file).

And exfiltration would need no evasion at all: net and spawn are HONESTLY
declared, so there is no manifest/behavior mismatch for the CON evidence rail
to bite on. The forge-resistance the substrate relies on assumes a liar; this
tool class does not have to lie.

THE DECIDING FACTOR. The owner has stated that core logic upgrading only via
daemon release is acceptable. The swap buys a capability that is not needed at
a cost that cannot currently be bounded. So: the extraction shipped (the shell
bundle is a genuinely runnable, self-describing reference tool — see
``atn/shell_tools.py``), and the resolution is written down here, wired at ONE
convergence point, and hard-disabled.

WHAT WOULD HAVE TO CHANGE to flip ``SHELL_SWAP_ENABLED``:
  1. OS-level isolation for adopted tools (the vault-track runner), not an
     in-process audit hook. The guard's own docstring already says the hook is
     "the tripwire in front of" that wall; for net+fs+spawn tools it is ONLY
     the tripwire, with no wall behind it.
  2. A privilege-class notion at adoption: approving a tool that claims
     ``provides`` over core shell names must be a visibly different act from
     approving a CSV parser. Today ``approve_adoption`` cannot tell them apart.
  3. A per-agent loadout binding. ``active_loadout`` is currently a single
     daemon-wide string used only to stamp attestations; it is not consulted at
     dispatch and cannot express "this agent runs a different shell".

Until all three exist, this module returns None and the built-in always runs.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

# THE GATE. Flipping this to True without the three preconditions in the module
# docstring hands every adopted shell provider a view of every command and file
# body the agent touches, with no containment. Do not flip it casually.
SHELL_SWAP_ENABLED = False


def _shell_names() -> frozenset[str]:
    """The overridable set. A provider may not claim a name outside it."""
    try:
        from ..shell_tools import SHELL_TOOL_EXECUTORS
        return frozenset(SHELL_TOOL_EXECUTORS)
    except Exception:  # noqa: BLE001 — fail closed
        return frozenset()


def resolve_shell_provider(runtime: Any, agent_id: str) -> dict[str, Any]:
    """Map ``tool_name -> ToolRecord`` for providers in the active loadout.

    Returns {} when swapping is disabled, when no loadout is stamped, or on any
    fault — {} means "use the built-in", which is the safe default at every
    step. Never raises: a broken loadout must not break tool dispatch.

    Kept live (not stubbed) even while the gate is off so the resolution logic
    is reviewable and testable on its own terms; ``dispatch_shell`` is what
    actually refuses to act on it.
    """
    if not SHELL_SWAP_ENABLED:
        return {}
    try:
        store = getattr(runtime, "tool_store", None)
        loadout = getattr(store, "active_loadout", "") if store else ""
        if not store or not loadout:
            return {}
        distro = store.resolve(loadout)
        if distro is None:
            return {}
        names = _shell_names()
        out: dict[str, Any] = {}
        claims: dict[str, str] = {}
        for dep in (distro.manifest.get("dependencies") or []):
            record = store.resolve(str(dep))
            if record is None or not record.enabled:
                continue
            # The daemon's OWN shell module is not an override of itself.
            if record.name == "atn_shell" and record.origin != "adopted":
                continue
            if not store.allowed(agent_id, record):
                continue
            caps = record.manifest.get("capabilities") or {}
            for claimed in (caps.get("provides") or []):
                if not isinstance(claimed, str) or claimed not in names:
                    continue
                prior = claims.get(claimed)
                # Deterministic on conflict: lexicographically smaller digest
                # wins. Last-writer-wins would make dispatch depend on dict
                # ordering, i.e. on registration history.
                if prior is not None:
                    if prior <= record.digest:
                        log.warning(
                            "shell provider conflict for %r: %s and %s",
                            claimed, prior[:16], record.digest[:16])
                        continue
                    log.warning("shell provider conflict for %r: %s and %s",
                                claimed, prior[:16], record.digest[:16])
                claims[claimed] = record.digest
                out[claimed] = record
        return out
    except Exception:  # noqa: BLE001 — never break dispatch
        log.debug("shell provider resolution failed", exc_info=True)
        return {}


async def dispatch_shell(runtime: Any, agent_id: str, name: str,
                         tool_input: dict) -> Optional[dict]:
    """Run ``name`` through a swapped provider, or return None.

    None means "no override — the caller runs the built-in", which is the
    zero-cost path and the one taken on every call today. The caller MUST
    treat None as fall-through rather than as an error.
    """
    if not SHELL_SWAP_ENABLED:
        return None
    if name not in _shell_names():
        return None
    providers = resolve_shell_provider(runtime, agent_id)
    record = providers.get(name)
    if record is None:
        return None
    try:
        store = runtime.tool_store
        out = await store.call(
            record, {"tool": name, "args": tool_input}, caller_id=agent_id)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"shell provider {record.digest[:16]} failed: {exc}"}
    if "error" in out:
        return {"error": out["error"]}
    envelope = out.get("result")
    if not isinstance(envelope, dict):
        return {"error": "shell provider returned a malformed envelope"}
    if envelope.get("ok"):
        return envelope.get("result")
    return {"error": str(envelope.get("error") or "shell provider error")}
