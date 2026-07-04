"""Daemon-side RPC/event/status wiring for ONE isolated agent worker (Phase 4).

P1 built ``WorkerChannel`` with *injected* handler slots but left them
``NotImplemented``. This module is the daemon-side counterpart to
``atn.runtime.worker_loop`` (the worker-side loop): it constructs the concrete
handlers/sinks that make a worker's RPCs execute IN THE DAEMON's shared context
(the real Runtime, registries, InboxManager, EventBus) and returns them ready to
inject into a ``WorkerChannel``.

Nothing here is reachable with ``ATN_WORKER_ISOLATION`` OFF — the daemon only
builds a ``WorkerHost`` when ``trigger_run`` routes a run into a worker (that
cutover is the coordinator's wiring; this module is the piece it drives).

THE SEAM (worker RPC name -> daemon action), all bound to the AUTHORITATIVE
pipe-bound ``agent_id`` (the worker's body never carries identity):

    framework_tool / surface_tool / mcp_tool
        -> engine.route_tool_call(name, input, agent_id)     [REUSED, not forked]
    budget_check   -> registry.check_budget(...)             -> {ok, blocker}
    budget_record  -> registry.record_token_usage(...)       -> {exceeded}
                      AND accumulate engine._booked_by_execution[execution_id]
                      so the daemon ledger is the single source of truth for
                      "already booked" (P0 risk #1: survives a hard worker kill).
    inbox_drain    -> InboxManager.drain(agent_id) -> [InboxMessage.dict...]
    inbox_post     -> InboxManager.post(InboxMessage(...))    -> {message_id}
    spawn_child    -> the REAL create_agent, run DAEMON-SIDE (P6): the daemon is
                      the trusted launcher; it clamps the requested spec against
                      daemon-held parent state (the D5 binding seam, no-op here),
                      derives parent_id from the AUTHORITATIVE pipe-bound
                      agent_id, registers the child + triggers its run (which
                      recurses through trigger_run -> the child becomes its OWN
                      isolated worker under the flag). Returns child agent_id/
                      handle to the parent worker.

EVENTS (worker -> daemon, droppable): re-emitted on the real EventBus with
``source`` OVERWRITTEN to the pipe-bound agent_id, so surfaces/WS/trace receive
them exactly as an in-process agent's events. A worker cannot forge another
agent's events.

STATUS (worker -> daemon):
    ready          -> handled by WorkerManager's handshake wrapper (not here)
    heartbeat      -> liveness only (the supervisor monitors the PID)
    execution_done -> populate record.output + record.token_usage from the
                      returned dict BEFORE finalize runs (P0: COMPLETED
                      output_store write + previews need it), then signal the
                      supervisor/caller that the clean path may finalize.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, FrozenSet, Optional, Union

from ..events import Event, EventType
from ..models import (
    ExecutionRecord,
    ExecutionStatus,
    InboxMessage,
    MessagePriority,
    MessageType,
    TokenUsage,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ALLOWANCE ALGEBRA (secret-allowance/tripwire track, Phase 1)
#
# ``resolve_spec`` (kevin/keystore.py) is the single, shared expander from a
# comma-separated launch spec (none/all/<service>/<bundle>) to the concrete set
# of vault service NAMES that actually exist. We import it verbatim — no fork,
# no drift. If that import fails (kevin not on path, module missing), we bind a
# FAIL-CLOSED shim that grants nothing, so a broken keystore can never widen an
# allowance.
try:  # pragma: no cover - import guard exercised only when the keystore is absent
    # The vendored in-wheel copy is authoritative (ships with the product so
    # `pip install autonet-computer` brings the vault); fall back to a dev
    # checkout of kevin on the path, then to a fail-closed deny-all shim.
    try:
        from atn._vendor.kevin.keystore import resolve_spec as _resolve_spec  # type: ignore
    except Exception:
        from kevin.keystore import resolve_spec as _resolve_spec  # type: ignore
except Exception:  # ANY import failure => deny-all
    def _resolve_spec(spec: Any) -> list:  # type: ignore
        """FAIL-CLOSED shim: keystore unavailable => grant nothing."""
        return []


# ---------------------------------------------------------------------------
# BINDING SEAM (design D5) — allowance intersection. LIVE (secret track P1).
#
# ``_resolve_spec_allowance`` turns a parent's requested child spec into the
# allowance it is ASKING for (via the shared ``resolve_spec``);
# ``_resolve_parent_allowance`` returns the parent's OWN allowance (L_parent),
# read from daemon-held registry state keyed by the authoritative pipe-bound
# parent id; the child grant is their INTERSECTION (monotone clamp). As of P1
# the algebra is REAL and the computed grant is materialized to concrete service
# NAMES and stashed in ``runtime._pending_grants[child_id]`` — but NOTHING
# CONSUMES it yet (no nonce mint, no secret tools, no monitor). Seam B
# (_on_pid_bound, P4) pops it; the secret_* tools (P5) expose it. This phase
# only computes + stashes. Flag OFF forces the grant to deny-all.
#
# FRACTAL GUARANTEE (why this MUST run daemon-side): the parent worker's
# requested spec is only an UPPER-BOUND WISH. The daemon clamps it (intersection)
# using daemon-held parent state. The parent never holds L_parent as a token,
# never mints the child's nonce, never sees the child's grant — so a compromised
# or buggy parent worker can only ever REQUEST a narrower-or-equal grant, never
# widen one. That clamp is the entire reason process isolation makes the fractal
# allowance safe.
class _AllowanceAll:
    """Sentinel type: an UNBOUNDED allowance ("all vault services, incl. any
    added later"). A distinct class (not a bare ``object()``) so ``isinstance``
    checks in the intersection are unambiguous and the value never leaks onto
    the RPC/worker path (it is materialized to concrete names before it is ever
    stashed in ``runtime._pending_grants``)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "_ALLOWANCE_ALL"


_ALLOWANCE_ALL = _AllowanceAll()  # sentinel: "unrestricted" (root / default-all)

# AllowanceGrant = the sentinel (unbounded, picks up new services) OR a concrete
# frozenset of resolved vault service NAMES. It is NEVER a str spec and NEVER a
# secret value.
AllowanceGrant = Union[_AllowanceAll, FrozenSet[str]]


def _resolve_parent_allowance(agent_id: str, runtime: Any) -> AllowanceGrant:
    """Return the parent's OWN allowance (L_parent), held DAEMON-SIDE.

    The parent's authored wish lives on its ``AgentDefinition.secrets_allowance``
    (a spec string, or ``None``). This is read from daemon-held registry state,
    keyed by the AUTHORITATIVE pipe-bound parent id — never from the worker body.

    Fail-closed mapping:
      - unknown agent / no runtime / no keystore    -> frozenset()  (deny-all)
      - wish is ``None`` or ``"none"`` (or blank)   -> frozenset()  (deny-all)
      - wish is ``"all"``                           -> _ALLOWANCE_ALL (unbounded)
      - otherwise                                   -> frozenset(resolve_spec(wish))
    """
    if runtime is None:
        return frozenset()
    try:
        defn = runtime.get_agent(agent_id)
    except Exception:
        return frozenset()
    if defn is None:
        return frozenset()
    wish = getattr(defn, "secrets_allowance", None)
    if wish is None:
        return frozenset()
    tokens = [t.strip().lower() for t in str(wish).split(",") if t.strip()]
    if not tokens:
        return frozenset()
    # A wish that is exactly/among "none" resolves to deny-all (resolve_spec's
    # own rule); we short-circuit to be explicit and independent of vault state.
    if any(t == "none" for t in tokens):
        return frozenset()
    if any(t == "all" for t in tokens):
        return _ALLOWANCE_ALL
    return frozenset(_resolve_spec(wish))


def _resolve_spec_allowance(payload: dict, runtime: Any) -> AllowanceGrant:
    """Turn the parent worker's REQUESTED child spec into the allowance it asks
    for. This is the ONLY place a worker-supplied spec is honored — and it is
    only ever a REQUEST: ``_intersect_allowance`` clamps it against L_parent, so
    a forged/over-broad spec can never widen the child beyond the parent.

    The requested spec is ``payload["secrets_allowance"]`` (the authored wish on
    the child definition). Mapping mirrors ``_resolve_parent_allowance``:
      - missing / None / "none" / blank -> frozenset()  (asks for nothing)
      - "all"                           -> _ALLOWANCE_ALL (asks for everything)
      - otherwise                       -> frozenset(resolve_spec(spec))
    """
    if not isinstance(payload, dict):
        return frozenset()
    spec = payload.get("secrets_allowance")
    if spec is None:
        return frozenset()
    tokens = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        return frozenset()
    if any(t == "none" for t in tokens):
        return frozenset()
    if any(t == "all" for t in tokens):
        return _ALLOWANCE_ALL
    return frozenset(_resolve_spec(spec))


def _intersect_allowance(requested: AllowanceGrant,
                         parent: AllowanceGrant) -> AllowanceGrant:
    """L_child = requested INTERSECT L_parent — a MONOTONE CLAMP.

    The worker only influences ``requested``; ``∩ parent`` guarantees the
    invariant  child ⊆ parent  (a child can never hold more than its parent).

      - parent is ALL  -> child = requested  (parent imposes no ceiling)
      - requested ALL  -> child = parent     (asked for everything; clamp to L_parent)
      - both concrete  -> child = requested & parent
    """
    if isinstance(parent, _AllowanceAll):
        return requested
    if isinstance(requested, _AllowanceAll):
        return parent
    # both are frozensets
    return requested & parent


def _allowance_to_spec(grant: AllowanceGrant) -> str:
    """Serialize a computed grant back to a canonical launch-spec string so it can
    be PERSISTED as the child's authored ``secrets_allowance``.

    This is what makes the fractal clamp COMPOSE across depth. Without it, the
    child's stored wish is the parent's (possibly wider) REQUEST, and
    ``_resolve_parent_allowance`` reads that wish as the ceiling for the NEXT
    spawn level — so a compromised child could grant its grandchild a service it
    was itself denied. Persisting the clamped grant instead means the stored wish
    is always the ENFORCED grant, so ``child ⊆ parent`` holds at every hop.

      - ``_ALLOWANCE_ALL`` -> ``"all"``  (child holds all iff its parent did)
      - empty frozenset    -> ``"none"`` (deny-all)
      - concrete names     -> comma-joined (round-trips through ``resolve_spec``,
                              since these names came from it in the first place)
    """
    if isinstance(grant, _AllowanceAll):
        return "all"
    names = sorted(grant)
    return ",".join(names) if names else "none"


# Callback the host invokes exactly once when the worker reports execution_done.
# The caller (trigger_run cutover / supervisor) uses it to run the clean
# finalize path. Signature: (status, error, accumulated_tool_calls) -> awaitable.
ExecutionDoneCb = Callable[[ExecutionStatus, Optional[str], list], Awaitable[None]]


class WorkerHost:
    """Daemon-side host for one worker: owns the injectable handlers/sinks and
    the per-execution budget tally, bound to one (agent_id, execution record).

    Construct one per spawned worker, pass ``rpc_handlers`` / ``event_sink`` /
    ``status_sink`` to ``WorkerManager.ensure_worker`` (or directly to a
    ``WorkerChannel``). The channel supplies the AUTHORITATIVE agent_id to every
    handler; this host trusts that argument and ignores any identity in the body.
    """

    def __init__(
        self,
        *,
        engine: Any,
        agent_id: str,
        record: ExecutionRecord,
        on_execution_done: ExecutionDoneCb,
    ) -> None:
        self._engine = engine
        self._agent_id = agent_id
        self._record = record
        self._on_execution_done = on_execution_done
        self._done_fired = False

    # ------------------------------------------------------------------
    # Injectable deps for WorkerChannel / ensure_worker
    # ------------------------------------------------------------------

    def rpc_handlers(self) -> dict[str, Callable[[str, dict], Awaitable[dict]]]:
        return {
            "framework_tool": self._rpc_tool,
            "surface_tool": self._rpc_tool,
            "mcp_tool": self._rpc_tool,
            "budget_check": self._rpc_budget_check,
            "budget_record": self._rpc_budget_record,
            "inbox_drain": self._rpc_inbox_drain,
            "inbox_post": self._rpc_inbox_post,
            "spawn_child": self._rpc_spawn_child,
        }

    def event_sink(self) -> Callable[[str, str, dict], Awaitable[None]]:
        return self._event_sink

    def status_sink(self) -> Callable[[str, str, dict], Awaitable[None]]:
        return self._status_sink

    # ------------------------------------------------------------------
    # RPC handlers — every one executes in the daemon's shared context.
    # ``agent_id`` is the pipe-bound authoritative id (never from the body).
    # ------------------------------------------------------------------

    async def _rpc_tool(self, agent_id: str, payload: dict) -> dict:
        """framework_tool / surface_tool / mcp_tool.

        All three collapse onto ``engine.route_tool_call``, which is the SAME
        dispatch the in-process cognitive loop uses (surface_* -> surfaces,
        mcp_* -> ConnectorManager, else -> framework tools via execute_tool).
        We do NOT re-derive the target here — the worker already picked the RPC
        name for observability, but route_tool_call re-decides from the tool
        name so a mismatched name can't reach the wrong authority.
        """
        name = str(payload.get("name") or "")
        tool_input = payload.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        if not name:
            return {"error": "tool rpc missing 'name'"}
        try:
            result = await self._engine.route_tool_call(name, tool_input, agent_id)
        except Exception as exc:
            log.debug("route_tool_call(%s) for %s failed", name, agent_id, exc_info=True)
            return {"error": f"tool {name} failed: {exc}"}
        return result if isinstance(result, dict) else {"result": result}

    async def _rpc_budget_check(self, agent_id: str, payload: dict) -> dict:
        provider = str(payload.get("provider") or "")
        model_id = str(payload.get("model_id") or "")
        ok, blocker = self._engine.registry.check_budget(
            agent_id, provider, model_id=model_id,
        )
        return {"ok": bool(ok), "blocker": blocker}

    async def _rpc_budget_record(self, agent_id: str, payload: dict) -> dict:
        """Book per-turn usage against the DAEMON ledger (the single source of
        truth) and mirror the increment into ``engine._booked_by_execution`` so
        the post-loop reconciliation in ``_finalize_execution`` knows exactly
        what has already been booked for THIS execution — even if the worker is
        hard-killed and its in-memory tally dies with it (P0 risk #1).

        ``tokens`` here is the per-turn ``budget_tokens()`` (cache_read already
        excluded worker-side), identical to the in-process inner-loop recorder.
        """
        provider = str(payload.get("provider") or "")
        tokens = payload.get("tokens") or 0
        try:
            tokens = float(tokens)
        except (TypeError, ValueError):
            tokens = 0.0
        if tokens <= 0:
            return {"exceeded": None}
        model_class = str(payload.get("model_class") or "")
        model_id = str(payload.get("model_id") or "")
        tpp = payload.get("tokens_per_pct")
        if not isinstance(tpp, dict):
            tpp = None

        exceeded = self._engine.registry.record_token_usage(
            agent_id, provider, tokens,
            model_class=model_class,
            model_id=model_id,
            tokens_per_pct=tpp,
        )

        # Mirror the booking into the daemon-authoritative per-execution tally.
        # Key by the SAME provider key record.token_usage uses (the provider
        # name), so the finalize reconciliation subtracts a like-for-like total.
        booked = self._engine._booked_by_execution.setdefault(
            self._record.execution_id, {})
        booked[provider] = booked.get(provider, 0.0) + tokens

        if exceeded:
            # Mirror the in-process auto-pause + event so behavior matches.
            from ..models import AgentStatus
            self._engine.registry._status[exceeded] = AgentStatus.BUDGET_PAUSED
            try:
                await self._engine.events.emit(Event(
                    type=EventType.BUDGET_EXCEEDED,
                    source=exceeded,
                    data={"agent_id": exceeded, "triggered_by": agent_id,
                          "provider": provider},
                ))
            except Exception:
                log.debug("budget_exceeded emit failed", exc_info=True)
        return {"exceeded": exceeded}

    async def _rpc_inbox_drain(self, agent_id: str, payload: dict) -> dict:
        """Drain the agent's daemon inbox, serialized to plain dicts.

        ``InboxManager.drain`` returns ``InboxMessage`` objects — NOT JSON. The
        worker only reads ``type`` / ``data`` / ``source`` off each, so we ship
        those plus id/priority/timestamp as a flat dict.
        """
        messages = self._engine.inbox.drain(agent_id)
        return {"messages": [_inbox_msg_to_dict(m) for m in messages]}

    async def _rpc_inbox_post(self, agent_id: str, payload: dict) -> dict:
        """Post a message to another agent's daemon inbox on the worker's behalf.

        ``source`` is forced to the pipe-bound agent_id — a worker cannot forge
        another sender. ``target`` comes from the payload (posting to peers is
        legitimate; the worker is just a message producer here).
        """
        target = str(payload.get("target") or "")
        if not target:
            return {"error": "inbox_post missing 'target'"}
        # Hierarchy scoping (M4): mirror the _post_message tool — an agent may
        # only post to its parent, a direct child, a sibling, or itself. The RPC
        # path previously skipped this, so a malicious worker could plant a
        # message (any type, incl. TRIGGER) into ANY agent's inbox daemon-wide.
        # source is already forced to the pipe-bound agent_id below, so only the
        # target needs bounding here.
        runtime = getattr(self._engine, "_runtime_ref", None)
        if runtime is not None:
            from ..orchestrator import ORCHESTRATOR_ID
            tgt_defn = runtime.get_agent(target)
            if tgt_defn is None:
                return {"error": f"inbox_post target '{target}' not found"}
            if agent_id != ORCHESTRATOR_ID:
                src_defn = runtime.get_agent(agent_id)
                if src_defn is not None:
                    is_parent = (src_defn.parent_id == target)
                    is_child = (tgt_defn.parent_id == agent_id)
                    is_sibling = bool(src_defn.parent_id and tgt_defn.parent_id
                                      and src_defn.parent_id == tgt_defn.parent_id)
                    is_self = (agent_id == target)
                    if not (is_parent or is_child or is_sibling or is_self):
                        log.warning("inbox_post: agent %s blocked posting to %s "
                                    "(outside hierarchy)", agent_id, target)
                        return {"error": f"agent '{agent_id}' cannot post to "
                                         f"'{target}': outside its hierarchy"}
        try:
            mtype = MessageType(str(payload.get("type") or "info"))
        except ValueError:
            mtype = MessageType.INFO
        try:
            prio = MessagePriority(str(payload.get("priority") or "normal"))
        except ValueError:
            prio = MessagePriority.NORMAL
        data = payload.get("data")
        if not isinstance(data, dict):
            data = {}
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=agent_id,          # authoritative — never from the body
            target=target,
            type=mtype,
            priority=prio,
            data=data,
        )
        self._engine.inbox.post(msg)
        try:
            await self._engine.events.emit(Event(
                type=EventType.MESSAGE_POSTED,
                source=agent_id,
                data={"message_id": msg.id, "target": target,
                      "type": mtype.value, "priority": prio.value},
            ))
        except Exception:
            log.debug("message_posted emit failed", exc_info=True)
        return {"message_id": msg.id}

    async def _rpc_spawn_child(self, agent_id: str, payload: dict) -> dict:
        """Spawn a delegate for an isolated worker — the REAL create_agent, run
        in the daemon's shared context (P6).

        ``agent_id`` is the AUTHORITATIVE pipe-bound PARENT id. It is the caller
        the daemon trusts; the child's parent_id is derived from it, never from
        the payload. The payload is the parent's requested child spec (name/
        prompt/model/mode/tools/… — JSON-only, per the P6 audit) and is treated
        as an UPPER-BOUND WISH only.

        Flow:
          1. BINDING SEAM (D5, no-op in P6): clamp the requested spec against the
             parent's daemon-held allowance. The intersection is what task 16
             will register against the child's pid at ``_on_pid_bound`` (fired
             inside the CHILD's own _run_cognitive_in_worker, after child-ready,
             before child-"go"). Here it computes but does not enforce.
          2. Run the daemon-side create_agent via ``execute_tool`` with
             ``caller_id`` FORCED to the pipe-bound parent id, so _create_agent
             derives ``parent_id = caller_id`` (tools.py). This reuses the exact
             dispatch route_tool_call already uses — no forked spawn logic — and
             therefore preserves create_agent's existing depth/spawn-count guards
             (register_agent -> _enforce_spawn_limits raises on over-depth/
             over-count; _create_agent turns that into a clean {"error": ...}
             tool-result). create_agent then triggers the child's run, which
             recurses through trigger_run: under the flag an API/bridge cognitive
             child spawns as ITS OWN worker (own PID), and the supervisor's
             _children map (populated with this parent id at that child's
             register) gives the leaf-first subtree kill.

        The child agent_id/status/execution_id dict is returned verbatim to the
        parent worker as the tool-result. The parent never receives L_parent, a
        nonce, or the child's grant — see the module-level FRACTAL GUARANTEE.
        """
        if not isinstance(payload, dict):
            return {"error": "spawn_child payload must be an object"}

        from ..orchestrator.tools import execute_tool

        # Hoisted above the binding seam: the seam reads daemon-held parent state
        # (registry) through the runtime, and the stash target
        # (runtime._pending_grants) also lives here. No runtime => nothing to
        # spawn against AND nothing to clamp against => fail closed.
        runtime = getattr(self._engine, "_runtime_ref", None)
        if runtime is None:
            return {"error": "spawn_child unavailable: no runtime bound to engine"}

        # --- BINDING SEAM (D5) — LIVE (Phase 1): compute + clamp + stash. ---
        # Runs DAEMON-SIDE with daemon-held parent state; a parent worker can
        # never widen a child's grant because it never participates in this
        # clamp (see the module-level FRACTAL GUARANTEE). The parent's requested
        # spec (payload["secrets_allowance"]) is an UPPER-BOUND WISH only.
        try:
            l_parent = _resolve_parent_allowance(agent_id, runtime)
            l_requested = _resolve_spec_allowance(payload, runtime)
            l_child = _intersect_allowance(l_requested, l_parent)
        except Exception:
            # A runtime fault in resolve_spec (e.g. transient keystore error,
            # after a successful import) must NEVER fail open. Deny-all; worst
            # case is a child with no secret grant, never an over-broad one.
            log.exception("allowance resolution failed; denying all secrets")
            l_child = frozenset()

        # Isolation OFF => no tripwire => FORCE deny-all. Secrets only ever
        # activate under ATN_WORKER_ISOLATION; with the flag off the grant is
        # empty and nothing downstream (P4/P5) will mint or expose a secret.
        _wi = getattr(getattr(runtime, "_config", None), "worker_isolation", None)
        if not getattr(_wi, "enabled", False):
            l_child = frozenset()

        try:
            # caller_id = the AUTHORITATIVE pipe-bound parent id. execute_tool
            # injects it as input["_caller_id"]; _create_agent pops it and sets
            # parent_id = caller_id. Any parent/_caller_id in the body is ignored.
            #
            # FRACTAL CLAMP (H1): persist the CLAMPED grant, not the parent's raw
            # requested wish, as the child's authored secrets_allowance. The
            # parent worker only ever influences ``requested`` (already clamped
            # into l_child above); storing l_child's canonical spec guarantees
            # that when THIS child later spawns, _resolve_parent_allowance reads a
            # ceiling that already obeys child ⊆ parent. Overwrites any wish the
            # worker put in the body — that value is an upper-bound request only.
            child_payload = {**payload,
                             "secrets_allowance": _allowance_to_spec(l_child)}
            result = await execute_tool(
                "create_agent", child_payload, runtime, caller_id=agent_id,
            )
        except Exception as exc:
            log.debug("spawn_child create_agent for parent %s failed",
                      agent_id, exc_info=True)
            return {"error": f"spawn_child failed: {exc}"}

        # Stash the daemon-computed grant for the child, keyed by its new id, so
        # Seam B (_on_pid_bound) can pop it once when the child's pid binds. The
        # value stashed is ALWAYS concrete JSON-safe service NAMES — the ALL
        # sentinel is materialized here via resolve_spec("all"), never carried on
        # the RPC/worker path. (Written only on a clean create; a failed create
        # leaves _pending_grants untouched.)
        child_id = (result or {}).get("agent_id") if isinstance(result, dict) else None
        if child_id:
            if isinstance(l_child, _AllowanceAll):
                materialized = sorted(_resolve_spec("all"))
            else:
                materialized = sorted(l_child)
            try:
                runtime._pending_grants[child_id] = materialized
            except Exception:
                log.debug("spawn_child: could not stash grant for child %s",
                          child_id, exc_info=True)
            log.info("spawn_child: parent=%s -> child=%s grant=%d service(s)%s",
                     agent_id, child_id, len(materialized),
                     " [ALL]" if isinstance(l_child, _AllowanceAll) else "")

        log.info("spawn_child: parent=%s -> child=%s status=%s", agent_id,
                 (result or {}).get("agent_id"), (result or {}).get("status"))
        return result if isinstance(result, dict) else {"result": result}

    def _bump_wedge_clock(self, agent_id: str) -> None:
        """Bump the supervisor's per-worker heartbeat clock on any inbound frame
        so a busy worker (RPCs, streamed output, heartbeats) is never killed as
        wedged. Best-effort: no supervisor / unknown agent => no-op."""
        sup = getattr(self._engine, "supervisor", None)
        if sup is not None:
            try:
                sup.heartbeat(agent_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Event sink — re-emit on the real EventBus, source overwritten.
    # ------------------------------------------------------------------

    async def _event_sink(self, agent_id: str, name: str, payload: dict) -> None:
        # Any inbound traffic is liveness — bump the supervisor's wedge clock so
        # a busy worker streaming output isn't mistaken for wedged.
        self._bump_wedge_clock(agent_id)
        etype = _EVENT_TYPE_BY_VALUE.get(name)
        if etype is None:
            # Unknown/opaque worker event — forward under CUSTOM so nothing is
            # silently lost, but tag the original name.
            etype = EventType.CUSTOM
            payload = {**payload, "_event_name": name}
        data = dict(payload or {})
        # OVERWRITE the source identity: the worker's advisory agent_id in the
        # body is ignored; the pipe-bound id is authoritative.
        data["agent_id"] = agent_id
        try:
            await self._engine.events.emit(Event(type=etype, source=agent_id, data=data))
        except Exception:
            log.debug("re-emit of worker event %r for %s failed", name, agent_id,
                      exc_info=True)

    # ------------------------------------------------------------------
    # Status sink — heartbeat + execution_done.
    # ------------------------------------------------------------------

    async def _status_sink(self, agent_id: str, name: str, payload: dict) -> None:
        # Any inbound status (heartbeat, execution_done, ...) is liveness.
        self._bump_wedge_clock(agent_id)
        if name == "execution_done":
            await self._handle_execution_done(agent_id, payload or {})
        elif name == "heartbeat":
            # Liveness only; the supervisor's monitor loop owns idle/runtime
            # limits. Nothing to do here beyond noting it.
            log.debug("worker %s heartbeat", agent_id)
        elif name == "ready":
            # Handled by the WorkerManager handshake wrapper; ignore here.
            pass
        else:
            log.debug("worker %s status %r (ignored)", agent_id, name)

    async def _handle_execution_done(self, agent_id: str, payload: dict) -> None:
        """Populate record.output + record.token_usage from the worker's final
        dict BEFORE finalize, then invoke the done callback exactly once.

        The dict shape is produced by ``worker_loop.run_cognitive_loop``:
            {status, error, output{result,tokens_used,usage},
             token_usage{provider:{...}}, tool_calls, streamed_text,
             session_stats}
        """
        if self._done_fired:
            log.warning("duplicate execution_done for %s ignored", agent_id)
            return
        self._done_fired = True

        rec = self._record

        # record.output — verbatim from the worker (drives COMPLETED
        # output_store write + previews on the clean path).
        rec.output = payload.get("output")

        # record.token_usage — rebuild TokenUsage objects keyed by provider so
        # the finalize reconciliation can call ``budget_tokens()`` and subtract
        # the daemon tally.
        tu = payload.get("token_usage")
        if isinstance(tu, dict):
            for provider_key, u in tu.items():
                if not isinstance(u, dict):
                    continue
                rec.token_usage[provider_key] = TokenUsage(
                    provider=provider_key,
                    input_tokens=int(u.get("input_tokens", 0) or 0),
                    output_tokens=int(u.get("output_tokens", 0) or 0),
                    cache_read_tokens=int(u.get("cache_read_tokens", 0) or 0),
                    cache_creation_tokens=int(u.get("cache_creation_tokens", 0) or 0),
                )

        # P5 subscription-budget snapshot: an isolated bridge agent's provider
        # lives in the WORKER, so it is NOT in the daemon's _active_providers and
        # get_my_budget_status (orchestrator/tools.py) would answer empty. Cache
        # the worker's last session_stats (which carries ``rate_limits`` +
        # ``tokens_per_pct_by_class`` for bridge) under this agent_id, reusing the
        # SAME dict the "provider is gone, answer from cache" path already reads.
        # Tradeoff: this is as-of the last completed orchestration, not live
        # mid-run — correct + non-empty, but a long single run shows slightly
        # stale subscription util. A live provider_status RPC per turn would
        # close that gap (heavier); the piggyback is the light correct option.
        sstats = payload.get("session_stats")
        if isinstance(sstats, dict):
            try:
                pmgr = getattr(self._engine, "provider_manager", None)
                cache = getattr(pmgr, "_cached_session_stats", None)
                if isinstance(cache, dict):
                    cache[agent_id] = sstats
            except Exception:
                log.debug("caching worker session_stats for %s failed",
                          agent_id, exc_info=True)

        # §5 undelivered steering re-post: the worker accepted these mid-run but
        # its loop ended before consuming them. The in-process finalize path
        # (execution_engine.py ~:1205) re-posts each to the agent's inbox as a
        # HIGH WORK message so nothing is silently dropped; the provider lives in
        # the worker, so it ships the list here and the DAEMON (which owns the
        # InboxManager) does the identical re-post. Same shape as the in-process
        # path: source="steering-fallback", WORK/HIGH, data={"instruction": ...}.
        undelivered = payload.get("undelivered_steering") or []
        if isinstance(undelivered, list) and undelivered:
            inbox = getattr(self._engine, "inbox", None)
            if inbox is not None:
                for instruction in undelivered:
                    try:
                        inbox.post(InboxMessage(
                            id=InboxMessage.generate_id(),
                            source="steering-fallback",
                            target=agent_id,
                            type=MessageType.WORK,
                            priority=MessagePriority.HIGH,
                            data={"instruction": str(instruction)},
                        ))
                    except Exception:
                        log.debug("undelivered-steering re-post failed for %s",
                                  agent_id, exc_info=True)

        status = _STATUS_BY_NAME.get(
            str(payload.get("status") or "").lower(), ExecutionStatus.FAILED)
        error = payload.get("error")
        tool_calls = payload.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []

        try:
            await self._on_execution_done(status, error, tool_calls)
        except Exception:
            log.exception("execution_done callback failed for %s", agent_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_BY_NAME = {
    "completed": ExecutionStatus.COMPLETED,
    "failed": ExecutionStatus.FAILED,
    "killed": ExecutionStatus.KILLED,
}

# Reverse lookup: EventType by its wire value ("agent.tool_use_start", etc.).
_EVENT_TYPE_BY_VALUE = {et.value: et for et in EventType}


def _inbox_msg_to_dict(m: InboxMessage) -> dict:
    return {
        "id": m.id,
        "source": m.source,
        "target": m.target,
        "type": m.type.value if hasattr(m.type, "value") else str(m.type),
        "priority": m.priority.value if hasattr(m.priority, "value") else str(m.priority),
        "data": m.data,
        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
    }


__all__ = ["WorkerHost"]
