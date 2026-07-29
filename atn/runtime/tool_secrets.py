"""Tool-scoped secret binding (docs/tool_secret_binding.md — change 1, shape B).

THE PROBLEM THIS CLOSES. The broker session is keyed by kernel PID and its
policy set is fixed at register-time (``vault_broker.py`` register/mint_nonce).
At the moment the broker authorizes a ``request``, NO TOOL IS EXECUTING — the
caller is the worker, i.e. the agent's own LLM loop. So "intersect the agent's
allowance with the tool's declared secrets at call time" cannot be expressed on
the worker's session: the agent asks for a secret, receives a staged path, and
only then decides what to do with it. The tool is not a party to the exchange.

THE FIX (shape B). Invert who holds the secret. A pinned tool declares the
services it needs (``capabilities.secrets``); the daemon mints a SEPARATE,
short-lived broker session bound to the TOOL SUBPROCESS's PID, scoped to

    L_tool = L_agent  ∩  manifest.capabilities.secrets

and the tool reads its own secret from its own session. The agent never
receives a handle. It cannot exfiltrate what it never holds — which is the
difference between a structural control and a detective one.

WHY THIS IS ENFORCEABLE WITH EXISTING MACHINERY. ``register`` already accepts a
target PID distinct from the caller (``vault_broker.py``: ``target =
req.get("pid", pid)``), and both ``mint_nonce`` and ``register`` are owner-gated
on ``BROKER_OWNER_SECRET``, which only the daemon holds. The tool subprocess is
a child of the daemon, so the daemon knows its PID. No broker change is needed.

FAIL-CLOSED, EVERY GATE. The default outcome is a tool with NO secrets:

  1. no ``capabilities.secrets`` on the manifest      -> deny-all (absent != all)
  2. dotted names (``app.*``, ``agent-key.*``)        -> stripped, always
  3. caller's own allowance unknown / unresolvable    -> deny-all
  4. intersection empty                               -> no session minted
  5. no broker client / mint / register not-ok        -> no session, tool still runs

(5) is deliberate: a tool that cannot get its secret should fail on the missing
secret, in its own code, not be blocked from running. The secret is the thing
gated, not the execution.

DECLARATION IS NOT A GRANT. The intersection means a manifest declaring
``{STRIPE_KEY}`` gets nothing if the calling agent's allowance lacks it. An
author cannot widen their own reach by writing a manifest — the ceiling is
always the caller's, and the caller's ceiling is itself already clamped by the
fractal parent/child intersection in ``worker_host.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

# Ceiling on how many services one tool may bind. A manifest asking for more is
# not an error — the excess is dropped after sorting, so the outcome is stable
# and always a NARROWING. Guards a manifest that declares the whole vault.
_MAX_TOOL_SECRETS = 16


def declared_tool_secrets(manifest: dict[str, Any]) -> frozenset[str]:
    """The service names a tool manifest DECLARES it needs.

    Read from ``capabilities.secrets``. Fail-closed and total: any shape that
    is not a list of non-empty strings yields the empty set.

    Dotted names are stripped unconditionally — the daemon plane
    (CredentialStore ``app.*`` payloads, ``agent-key.*`` wallet keys) must never
    enter a tool binding by ANY route, mirroring the worker-side choke point in
    ``worker_host._resolve_spec``. A tool cannot name them even if its author
    wrote them into the manifest by hand.
    """
    if not isinstance(manifest, dict):
        return frozenset()
    caps = manifest.get("capabilities")
    if not isinstance(caps, dict):
        return frozenset()
    raw = caps.get("secrets")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    names = {
        s.strip() for s in (x for x in raw if isinstance(x, str))
        if s.strip() and "." not in s.strip()
    }
    if len(names) > _MAX_TOOL_SECRETS:
        # Stable narrowing, not an error: sort and truncate.
        names = set(sorted(names)[:_MAX_TOOL_SECRETS])
        log.warning("tool manifest declares > %d secrets; truncated",
                    _MAX_TOOL_SECRETS)
    return frozenset(names)


def _caller_allowance(runtime: Any, caller_id: str) -> Optional[frozenset[str]]:
    """L_agent for ``caller_id``, resolved DAEMON-SIDE from registry state.

    Returns ``None`` when the allowance is unbounded ("all"), a concrete
    frozenset otherwise. Reuses ``worker_host``'s resolver verbatim so the two
    planes can never drift: the same spec syntax, the same dotted-name strip,
    the same fail-closed posture.

    ``None`` (unbounded) is NOT "allow everything" downstream — it means the
    caller imposes no ceiling, so the tool's own declaration becomes the whole
    binding. That is still a narrowing relative to today, where the tool would
    inherit the agent's full reach through the environment.
    """
    try:
        from .worker_host import (
            _ALLOWANCE_ALL,
            _resolve_parent_allowance,
        )
    except Exception:  # noqa: BLE001 — import fault => deny-all
        log.debug("tool_secrets: allowance resolver unavailable", exc_info=True)
        return frozenset()

    try:
        grant = _resolve_parent_allowance(caller_id, runtime)
    except Exception:  # noqa: BLE001 — never fail open
        log.exception("tool_secrets: allowance resolution failed for %s",
                      caller_id)
        return frozenset()

    if isinstance(grant, _ALLOWANCE_ALL.__class__):
        return None
    try:
        return frozenset(grant)
    except TypeError:
        return frozenset()


def resolve_tool_secrets(
    manifest: dict[str, Any],
    runtime: Any,
    caller_id: str,
) -> frozenset[str]:
    """L_tool = L_agent ∩ declared(manifest) — the monotone clamp.

    The same algebra as the fractal parent/child intersection, applied on a
    second axis. A tool can only ever hold a SUBSET of what its caller holds,
    and only the part it declared in advance.

    Empty result = bind nothing (the common, safe case).
    """
    declared = declared_tool_secrets(manifest)
    if not declared:
        return frozenset()
    if not caller_id:
        # No authenticated caller => no ceiling to clamp against => deny.
        return frozenset()

    allowed = _caller_allowance(runtime, caller_id)
    if allowed is None:
        # Caller unbounded: the declaration IS the binding.
        return declared
    return declared & allowed


class ToolSecretSession:
    """A short-lived broker session bound to ONE tool subprocess PID.

    Lifecycle is strictly scoped to a single tool call:

        session = ToolSecretSession(runtime, services)
        nonce   = session.mint()          # before spawn (owner-gated)
        session.bind(proc.pid)            # immediately after spawn
        ...tool runs, reads its own secret via the broker...
        session.release()                 # ALWAYS, in a finally

    ``release`` is owner-gated teardown: it drops the PID's policies and push
    sink AND unlinks every file staged for it, so no raw value outlives the
    subprocess. It is idempotent and never raises — a teardown fault must not
    mask the tool's own result.
    """

    def __init__(self, runtime: Any, services: Iterable[str],
                 agent_id: str = "", tool_digest: str = "") -> None:
        self._runtime = runtime
        self._services = sorted({str(s) for s in services if s})
        self._agent_id = agent_id or ""
        self._tool = tool_digest or ""
        self._nonce: Optional[str] = None
        self._pid: Optional[int] = None

    @property
    def services(self) -> list[str]:
        return list(self._services)

    @property
    def active(self) -> bool:
        return self._pid is not None

    def _client(self) -> Any:
        return getattr(self._runtime, "_broker_client", None)

    def mint(self) -> bool:
        """Mint the one-time nonce for this tool's service set (owner-gated).

        Returns False on any fault. A False here means the tool runs WITHOUT
        secrets — it does not block the call.

        Gated on the tripwire being up, exactly as the worker grant path is
        (``Runtime._on_pid_bound``): monitor down => no value-push => a staged
        secret nobody is scanning. Deny instead.
        """
        if not self._services:
            return False
        client = self._client()
        if client is None:
            return False
        if not getattr(client, "value_push_armed", False):
            log.warning("tool_secrets: value-push not armed; no tool session")
            return False
        monitor = getattr(self._runtime, "security_monitor", None)
        try:
            if monitor is None or not monitor.is_healthy():
                log.warning("tool_secrets: monitor down; no tool session")
                return False
        except Exception:  # noqa: BLE001 — unknown health => deny
            return False

        try:
            r = client.mint_nonce(self._services, agent_id=self._agent_id)
        except Exception:  # noqa: BLE001
            log.warning("tool_secrets: mint_nonce raised", exc_info=True)
            return False
        if not (isinstance(r, dict) and r.get("ok") and r.get("nonce")):
            log.warning("tool_secrets: mint_nonce not-ok: %s",
                        (r or {}).get("error"))
            return False
        self._nonce = str(r["nonce"])
        return True

    def bind(self, pid: int) -> bool:
        """Bind the minted nonce to the tool subprocess PID (consumes it).

        Must be called immediately after spawn. The broker records
        ``pid:starttime`` as the session identity, so a PID recycled to another
        process cannot inherit this session.
        """
        if self._nonce is None:
            return False
        client = self._client()
        if client is None:
            return False
        try:
            r = client.register(int(pid), self._nonce)
        except Exception:  # noqa: BLE001
            log.warning("tool_secrets: register raised for pid=%s", pid,
                        exc_info=True)
            self._nonce = None
            return False
        finally:
            # The nonce is one-time whatever the outcome; never retry with it.
            nonce_used, self._nonce = self._nonce, None
            del nonce_used
        if not (isinstance(r, dict) and r.get("ok")):
            log.warning("tool_secrets: register not-ok for pid=%s: %s", pid,
                        (r or {}).get("error"))
            return False
        self._pid = int(pid)
        self._audit("granted")
        return True

    def release(self) -> None:
        """Owner-gated teardown. Idempotent; never raises."""
        pid, self._pid = self._pid, None
        self._nonce = None
        if pid is None:
            return
        client = self._client()
        if client is None:
            return
        try:
            client.release_session(int(pid))
        except Exception:  # noqa: BLE001 — teardown must never mask a result
            log.debug("tool_secrets: release_session failed for pid=%s", pid,
                      exc_info=True)
        self._audit("revoked", pid=pid)

    def _audit(self, action: str, pid: Optional[int] = None) -> None:
        audit = getattr(self._runtime, "secret_audit", None)
        if audit is None:
            return
        try:
            audit.record(action, agent_id=self._agent_id,
                         services=self._services,
                         pid=pid if pid is not None else self._pid,
                         tool=self._tool)
        except Exception:  # noqa: BLE001
            log.debug("tool_secrets: audit record failed", exc_info=True)
