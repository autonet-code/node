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
    spawn_child    -> structured "not supported under isolation yet" (P6)

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
from typing import Any, Awaitable, Callable, Optional

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


# Refusal returned for any spawn attempt from an isolated worker (delegate spawn
# is P6). Kept in lockstep with the worker-side ``_SPAWN_REFUSAL``.
_SPAWN_REFUSAL = {
    "error": "agent spawning is not yet supported under process isolation (P6). "
             "Run this agent with worker isolation OFF to spawn delegates."
}


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
        # P6 implements real in-isolation spawn. For P4 refuse cleanly so a
        # worker that tries to delegate gets a tool-result, not a crash.
        log.info("spawn_child refused for isolated worker %s (P6 not yet wired)", agent_id)
        return dict(_SPAWN_REFUSAL)

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
