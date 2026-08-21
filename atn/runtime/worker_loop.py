"""The in-worker cognitive loop (Phase 4).

This is the worker-side mirror of ``ExecutionEngine._execute_cognitive_agent``'s
inner body — the part that, under isolation, runs in the WORKER process. It
drives ``provider.send_orchestrate`` with:

  - a ``_tool_executor`` that splits LOCAL shell tools (run here) from AUTHORITY
    tools (RPC to the daemon over ``DaemonClient.rpc``);
  - an ``_on_chunk`` streaming callback that forwards text deltas to the daemon
    via ``DaemonClient.emit`` (the daemon re-emits on the real EventBus with the
    source overwritten to the pipe-bound agent_id);
  - a ``_per_turn_recorder`` that books usage over the ``budget_record`` RPC so
    the DAEMON ledger stays the single source of truth (P0 risk #1);
  - a pre-flight ``budget_check`` RPC.

What DOESN'T run here (stays daemon-side, shipped in the manifest or done after
``execution_done``): building the system prompt, draining the inbox, reading
conversation history from disk, resolving the tool surface, writing the output
store, updating the delegate registry, and the budget reconciliation. The worker
receives the already-built ``message`` / ``system`` / ``tools`` / ``history`` and
returns the ProviderResponse-derived dict.

The manifest keys this reads (all JSON-safe, all built daemon-side):
    provider_config : dict  — for worker_provider.build_provider_from_manifest
    agent_id        : str   — logging only; identity is pipe-bound on the daemon
    provider_name   : str   — budget key (matches record.token_usage key)
    message         : str   — the fully-assembled first user message
    system          : str   — the system prompt (constitutional preamble folded in)
    tools           : list[dict]        — the resolved tool surface (name/desc/schema)
    native_tools    : bool  — SDK builtins flag (inert for API providers)
    max_turns       : int
    session_id      : str   — "" for API providers (they rebuild from history)
    history         : list[dict] | None — prior turns as canonical messages
    per_turn_input_max : int | None
    repeat_call_limit  : int | None
    model           : str   — model override for send_orchestrate
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..shell_tools import SHELL_TOOL_EXECUTORS as _SHELL_TOOL_EXECUTORS

log = logging.getLogger("atn.agent_worker.loop")


# ---------------------------------------------------------------------------
# Secret tool surface (P5 — worker-side broker access from the worker's PID)
# ---------------------------------------------------------------------------
#
# A GRANTED isolated worker (double-gated daemon-side: it has a non-empty
# ``runtime._pending_grants`` entry AND ``worker_isolation.enabled``) is handed
# these three tools in its manifest ``tools`` list. They let the worker MODEL:
#   - see WHICH services it may reach (names only);
#   - REQUEST a secret — which stages the raw value to a nameless 128-bit file
#     owned by the broker and returns ONLY {var_name, path}. The VALUE is never
#     returned to the worker; it reaches the daemon monitor via the broker's
#     owner-authenticated value-push (P2), where the tripwire scans for it.
#   - RELEASE a service it no longer needs.
#
# The worker reaches the broker over its OWN kernel PID (WorkerBrokerClient holds
# no secret; the broker authenticates the PID directly). An UNGRANTED or
# flag-off worker never sees these schemas at all (the daemon only appends them
# when the double gate passes), so the model cannot even name the tools.
SECRET_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "secret_list_services",
        "description": (
            "List the secret services this agent is allowed to request. "
            "Returns names only (never values). Use this to discover which "
            "credentials are available before requesting one."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "secret_request_secret",
        "description": (
            "Request a granted secret by service name. The secret VALUE is "
            "staged to a private file on disk and is NOT returned to you. You "
            "receive only {var_name, path}: read the file at 'path' at the "
            "moment of use (e.g. pass it to a subprocess). Do NOT echo the file "
            "contents or the path into your reply — doing so trips a security "
            "alarm. Requesting a service you were not granted is denied."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "The service name to request."},
            },
            "required": ["service"],
            "additionalProperties": False,
        },
    },
    {
        "name": "secret_release",
        "description": (
            "Release a secret you previously requested (unstages its file). "
            "Call when you no longer need the credential."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "The service name to release."},
            },
            "required": ["service"],
            "additionalProperties": False,
        },
    },
]

# The tool NAMES intercepted worker-side BEFORE the LOCAL/AUTHORITY split. These
# are neither LOCAL shell executors nor daemon-RPC AUTHORITY tools: they talk to
# the vault-broker directly from the worker PID.
_SECRET_TOOLS = frozenset({
    "secret_list_services", "secret_request_secret", "secret_release",
})


# Spawn-family tools (delegate spawn). Under isolation these DO NOT run in the
# worker — the worker is not a trusted launcher and must never mint a child's
# identity/grant. Instead the worker issues a ``spawn_child`` RPC and the DAEMON
# runs the real create_agent logic in its own context, deriving the child's
# parent_id from the AUTHORITATIVE pipe-bound agent_id (never a body field). The
# child then recurses through trigger_run and, under the flag, becomes its own
# isolated worker. See ``worker_host.WorkerHost._rpc_spawn_child``.
_SPAWN_TOOLS = frozenset({"create_agent"})


def _is_authority_tool(name: str) -> bool:
    """Worker-side copy of ExecutionEngine.is_authority_tool.

    LOCAL (False): the sandboxed shell executors — run in this process.
    AUTHORITY (True): everything else (surface_*, mcp_*, framework tools) —
    RPC back to the daemon.
    """
    if name in _SHELL_TOOL_EXECUTORS:
        return False
    return True


def _rpc_kind_for(name: str) -> str:
    """Which RPC name carries this authority tool to the daemon.

    Mirrors ExecutionEngine.route_tool_call's authority-target selection:
      surface_* -> surface_tool ; mcp_* -> mcp_tool ; else -> framework_tool.
    The daemon-side handler resolves the concrete target using the pipe-bound
    agent_id (the worker never sends agent_id in the body).
    """
    if name.startswith("surface_"):
        return "surface_tool"
    if name.startswith("mcp_"):
        return "mcp_tool"
    return "framework_tool"


async def run_cognitive_loop(
    *,
    client: Any,
    provider: Any,
    manifest: dict[str, Any],
    agent_label: str = "?",
) -> dict[str, Any]:
    """Drive one cognitive orchestration in the worker; return the output dict.

    Returns a JSON-safe dict shaped for the daemon's ``execution_done`` handler::

        {
          "status": "completed" | "failed" | "killed",
          "error": <str|None>,
          "output": {                     # -> record.output verbatim
             "result": <str>,
             "tokens_used": <int>,
             "usage": {input_tokens, output_tokens,
                       cache_read_tokens, cache_creation_tokens},
          },
          "token_usage": {                # -> record.token_usage[provider_name]
             <provider_name>: {input_tokens, output_tokens,
                               cache_read_tokens, cache_creation_tokens},
          },
          "tool_calls": [ {tool, args, result, success}, ... ],  # run summary
          "streamed_text": <bool>,        # did any text stream as chunks
        }
    """
    provider_name = str(manifest.get("provider_name") or provider.name or "")
    model = str(manifest.get("model", "") or "")
    message = str(manifest.get("message", "") or "")
    system = str(manifest.get("system", "") or "")
    tools = list(manifest.get("tools") or [])
    native_tools = bool(manifest.get("native_tools", False))
    max_turns = int(manifest.get("max_turns", 20) or 20)
    session_id = str(manifest.get("session_id", "") or "")
    history = manifest.get("history") or None
    per_turn_input_max = manifest.get("per_turn_input_max")
    repeat_call_limit = manifest.get("repeat_call_limit")

    # --- Wire the provider's event bus to the daemon over the pipe. The base
    # provider emits AGENT_TOOL_USE_* + usage events on ``event_bus``; we bridge
    # those to ``client.emit`` so the daemon re-emits them (source overwritten to
    # the pipe-bound agent_id). Text deltas come through ``on_chunk`` below.
    provider.event_bus = _PipeEventBus(client)
    provider.source_agent_id = manifest.get("agent_id", "") or agent_label

    # --- Pre-flight budget check (daemon ledger authoritative). ---
    try:
        chk = await client.rpc("budget_check", {
            "provider": provider_name, "model_id": model,
        })
        if isinstance(chk, dict) and chk.get("ok") is False:
            blocker = chk.get("blocker")
            return {
                "status": "failed",
                "error": f"Budget exceeded (blocked by {blocker})",
                "output": None,
                "token_usage": {},
                "tool_calls": [],
                "streamed_text": False,
            }
    except Exception:
        # A failed pre-flight RPC must not silently run an unbudgeted agent.
        log.exception("[%s] budget_check RPC failed", agent_label)
        return {
            "status": "failed",
            "error": "budget_check RPC failed",
            "output": None,
            "token_usage": {},
            "tool_calls": [],
            "streamed_text": False,
        }

    # --- Tool executor (LOCAL vs AUTHORITY split). ---
    accumulated_tool_calls: list[dict[str, Any]] = []

    async def _tool_executor(name: str, tool_input: dict) -> dict:
        result = await _route_tool(client, name, tool_input, agent_label)
        is_error = isinstance(result, dict) and "error" in result
        accumulated_tool_calls.append({
            "tool": name,
            "args": tool_input,
            "result": str(result)[:4000],
            "success": not is_error,
        })
        return result

    # --- Streaming callback: forward text deltas to the daemon. ---
    streamed = {"text": False}

    async def _on_chunk(text: str) -> None:
        if text:
            streamed["text"] = True
        await client.emit("step.output", {
            "channel": "text",
            "content": text,
            "cognitive": True,
        })

    # --- Per-turn budget recorder (books over RPC; daemon ledger is truth). ---
    # Async: the base loop calls this via ``_call_recorder`` then awaits the
    # result through ``_maybe_await`` (see providers/base.py), so returning a
    # coroutine is the supported path — no nested event loop.
    async def _per_turn_recorder(turn_tokens: float, model_str: str = "") -> tuple[bool, str | None]:
        if turn_tokens <= 0:
            return True, None
        from ..model_specs import resolve as _resolve_model
        spec = _resolve_model(model_str)
        # API providers don't implement tokens_per_pct_for_model (that's a
        # bridge method), so for them the subscription-rate path is inert:
        # tokens_per_pct stays None and only the raw token unit is booked.
        # For the P5 bridge this IS live — tokens_per_pct_for_model resolves the
        # per-class 5h-window rate; the pct math runs here (worker-side, provider
        # is local) and crosses to the DAEMON ledger via the budget_record RPC
        # under the same reconciliation rule as P4.
        rate_for_model = None
        fn = getattr(provider, "tokens_per_pct_for_model", None)
        if fn is not None:
            try:
                rate_for_model = fn(spec.id)
            except Exception:
                rate_for_model = None
        tpp = {spec.klass: rate_for_model} if rate_for_model else None
        try:
            res = await client.rpc("budget_record", {
                "provider": provider_name,
                "tokens": turn_tokens,
                "model_class": spec.klass,
                "model_id": spec.id if spec.id != "default" else "",
                "tokens_per_pct": tpp,
            })
        except Exception:
            # A failed record RPC shouldn't kill the turn; the daemon's
            # post-loop reconciliation still books the real spend from the
            # returned token_usage. Continue.
            log.debug("[%s] budget_record RPC failed", agent_label, exc_info=True)
            return True, None
        exceeded = res.get("exceeded") if isinstance(res, dict) else None
        if exceeded:
            return False, exceeded
        return True, None

    # --- Run the orchestration. ---
    send_kwargs: dict[str, Any] = {
        "message": message,
        "system": system,
        "tools": tools,
        "native_tools": native_tools,
        "max_turns": max_turns,
        "tool_executor": _tool_executor,
        "on_chunk": _on_chunk,
        "usage_recorder": _per_turn_recorder,
        "model": model,
    }
    if session_id:
        send_kwargs["session_id"] = session_id
    if history:
        send_kwargs["history"] = history
    if per_turn_input_max is not None:
        send_kwargs["per_turn_input_max"] = per_turn_input_max
    if repeat_call_limit is not None:
        send_kwargs["repeat_call_limit"] = repeat_call_limit

    response = await provider.send_orchestrate(**send_kwargs)

    # v3 review step for providers whose loop can't inject it (the SDK
    # bridge): one follow-up turn on the same session prompting
    # attest_tools. Mirrors the in-process engine block exactly.
    from ..delegate_prompts import (
        REVIEW_STEP_PROMPT,
        VERIFY_STEP_PROMPT,
        needs_review_reinvoke,
        needs_verify_reinvoke,
    )
    _review_session = getattr(provider, "_session_id", "") or ""
    # §16 verify step (bridge path) — capture before any follow-up
    # orchestrate resets the provider's tracking.
    _modified_code = set(getattr(
        provider, "last_modified_code_files", None) or ())
    if _review_session and needs_verify_reinvoke(
            provider, _modified_code, response.stop_reason):
        log.info("[%s] verify step: follow-up verification turn (%d code "
                 "files)", agent_label, len(_modified_code))
        try:
            verify_kwargs = dict(send_kwargs)
            verify_kwargs.pop("history", None)  # session carries context
            verify_kwargs.update(
                message=VERIFY_STEP_PROMPT.format(
                    files=", ".join(sorted(_modified_code)[:20])),
                max_turns=8,
                session_id=_review_session,
            )
            await provider.send_orchestrate(**verify_kwargs)
        except Exception:
            log.warning("[%s] verify follow-up turn failed; continuing",
                        agent_label, exc_info=True)
    if _review_session and needs_review_reinvoke(
            provider, accumulated_tool_calls, response.stop_reason):
        log.info("[%s] review step: follow-up attest turn", agent_label)
        try:
            review_kwargs = dict(send_kwargs)
            review_kwargs.pop("history", None)  # session carries context
            review_kwargs.update(
                message=REVIEW_STEP_PROMPT,
                max_turns=4,
                session_id=_review_session,
            )
            await provider.send_orchestrate(**review_kwargs)
        except Exception:
            log.warning("[%s] review follow-up turn failed; continuing",
                        agent_label, exc_info=True)

    # --- Derive the output + status (mirrors the in-process result block). ---
    result_text = response.text or ""
    u = response.usage
    total_tokens = u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_creation_tokens

    if response.stop_reason == "interrupted":
        status = "killed"
        error: str | None = "Interrupted"
    elif response.stop_reason == "budget_exceeded":
        if result_text.strip():
            status, error = "completed", None
        else:
            status, error = "failed", "Budget exceeded before any output was produced"
    else:
        status, error = "completed", None

    output = {
        "result": result_text,
        "tokens_used": total_tokens,
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
        },
    }

    # If the answer never streamed, emit it once as a final text event so live
    # consumers get the full text (matches the in-process fallback).
    if result_text and not streamed["text"]:
        await client.emit("step.output", {
            "channel": "text",
            "content": result_text,
            "cognitive": True,
            "final": True,
        })

    token_usage = {
        provider_name: {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
        }
    }

    # Post-loop reconciliation (P5, bridge subscription providers). The
    # in-process path runs this at ExecutionEngine finalize (execution_engine.py
    # ~:1136); worker_loop omitted it in P4 because API providers don't
    # implement it. For bridge it MUST run WORKER-SIDE: refresh_usage() reads
    # ~/.claude/.credentials.json off disk and makes the API call from the
    # provider, so it can only happen where the provider lives. It also freshens
    # ``_rate_limits`` for the subscription snapshot below. Best-effort — a
    # reconcile failure never affects the user-visible result.
    if hasattr(provider, "reconcile_after_orchestration"):
        try:
            await provider.reconcile_after_orchestration()
        except Exception:
            log.debug("[%s] reconcile_after_orchestration failed; continuing",
                      agent_label, exc_info=True)

    # Session-stat sync (the daemon owns the conversation store; it records the
    # assistant turn and persists stats after execution_done). We surface
    # session_stats so the daemon can persist them if it wants. For bridge,
    # ``session_stats["rate_limits"]`` is the subscription snapshot the daemon
    # caches so get_my_budget_status can answer for this now-isolated agent
    # (the provider is in the worker, not the daemon's _active_providers).
    session_stats = None
    _stats = getattr(provider, "session_stats", None)
    if isinstance(_stats, dict):
        session_stats = dict(_stats)
        # Attach the per-class tokens-per-pct table (bridge only) so the daemon
        # can answer the full subscription block, not just raw rate_limits.
        tpbc = getattr(provider, "tokens_per_pct_by_class", None)
        if isinstance(tpbc, dict):
            session_stats["tokens_per_pct_by_class"] = dict(tpbc)
        tpp = getattr(provider, "tokens_per_percent", None)
        if tpp is not None:
            session_stats["tokens_per_percent"] = tpp

    # §5 undelivered steering: any mid-run steering the loop accepted but never
    # consumed before it ended lives on the ProviderResponse. The in-process
    # path (execution_engine.py ~:1205) re-posts each to the agent's inbox as a
    # HIGH WORK message so nothing is silently dropped. The provider is in THIS
    # worker, so we ship the list back and the DAEMON does the inbox re-post
    # (it owns the InboxManager) — see WorkerHost._handle_execution_done.
    undelivered_steering = list(getattr(response, "undelivered_steering", None) or [])

    return {
        "status": status,
        "error": error,
        "output": output,
        "token_usage": token_usage,
        "tool_calls": accumulated_tool_calls,
        "streamed_text": streamed["text"],
        "session_stats": session_stats,
        "undelivered_steering": undelivered_steering,
    }


# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------

async def _route_tool(client: Any, name: str, tool_input: dict, agent_label: str) -> dict:
    """Route one tool call: LOCAL executors run here; AUTHORITY tools RPC out."""
    # Spawn family (P6): issue a dedicated ``spawn_child`` RPC BEFORE the
    # LOCAL/AUTHORITY split. The daemon is the trusted launcher — it runs the
    # real create_agent logic and derives parent_id from the pipe-bound agent_id.
    # We deliberately do NOT send parent_agent_id (or any identity) in the body:
    # the worker's requested spec is only an UPPER-BOUND WISH; the daemon clamps
    # it against daemon-held parent state and the parent worker never holds its
    # own or the child's grant. The daemon's result (child agent_id/status/…) is
    # returned verbatim as this tool's result.
    if name in _SPAWN_TOOLS:
        try:
            result = await client.rpc("spawn_child", dict(tool_input))
        except Exception as exc:
            return {"error": f"{name} (rpc spawn_child) failed: {exc}"}
        return result if isinstance(result, dict) else {"result": result}

    # Secret family (P5): intercept BEFORE the LOCAL/AUTHORITY split. These talk
    # to the vault-broker directly from THIS worker's kernel PID (no daemon RPC,
    # no owner secret). The broker authenticates the PID and only ever exposes
    # the services this PID was granted at register-time. secret_request_secret
    # returns {var_name, path} — NEVER the value; the value reaches the daemon
    # monitor via the broker's owner-authenticated push (P2), never through here.
    if name in _SECRET_TOOLS:
        return await _handle_secret_tool(name, tool_input)

    if not _is_authority_tool(name):
        try:
            return await _SHELL_TOOL_EXECUTORS[name](tool_input)
        except Exception as exc:
            return {"error": f"{name} failed: {exc}"}

    rpc_kind = _rpc_kind_for(name)
    try:
        result = await client.rpc(rpc_kind, {"name": name, "input": tool_input})
    except Exception as exc:
        return {"error": f"{name} (rpc {rpc_kind}) failed: {exc}"}
    return result if isinstance(result, dict) else {"result": result}


# Worker-context marker: set by ``agent_worker._amain`` in the worker process
# ONLY. A secret tool that fires outside a worker (e.g. accidentally routed in
# the daemon) is fail-closed refused — the broker authenticates by PID and the
# daemon's PID is NOT a granted worker, but we refuse before we ever touch the
# broker so a value can never be staged for the wrong process.
_WORKER_CONTEXT_ENV = "ATN_IS_WORKER"


def _in_worker_context() -> bool:
    return os.environ.get(_WORKER_CONTEXT_ENV) == "1"


async def _handle_secret_tool(name: str, tool_input: dict) -> dict:
    """Service one secret_* tool from the worker PID via the vault-broker.

    FAIL-CLOSED throughout:
      - not in a worker context           -> {"ok": False, "error": ...} (no broker call)
      - broker down / bad reply / POSIX   -> WorkerBrokerClient returns an error dict
      - a value would leak                 -> impossible: request returns {var_name, path}

    The WorkerBrokerClient is built lazily, does exactly one round-trip, and is
    thrown away — it holds no secret and its only authority is this process's
    kernel PID, which the broker reads directly.
    """
    if not _in_worker_context():
        # Never touch the broker off a worker PID: a secret must only ever be
        # staged for a granted worker process, never the daemon.
        return {"ok": False, "error": "secret tools are worker-only"}

    # Lazy import: keeps the daemon-side manifest builder free of broker plumbing
    # and matches the "one round-trip, then close" discipline.
    from .broker_client import WorkerBrokerClient

    broker = WorkerBrokerClient()

    if name == "secret_list_services":
        res = broker.list_services()
        if not isinstance(res, dict) or res.get("ok") is False:
            return {"ok": False, "services": [],
                    "error": (res or {}).get("error", "broker unavailable")}
        services = res.get("services")
        return {"ok": True, "services": list(services) if isinstance(services, list) else []}

    if name == "secret_request_secret":
        service = str((tool_input or {}).get("service", "") or "")
        if not service:
            return {"ok": False, "error": "service is required"}
        res = broker.request(service)
        if not isinstance(res, dict) or res.get("ok") is False:
            # Denied / unmapped / broker down: NO staging happened.
            return {"ok": False,
                    "error": (res or {}).get("error", "request denied")}
        # SAFETY-CRITICAL: return ONLY the indirection, never the value. Even if
        # the broker ever returned a value, we strip everything except the two
        # safe fields so a value can never ride this tool result into the model.
        return {
            "ok": True,
            "var_name": str(res.get("var_name", "") or ""),
            "path": str(res.get("path", "") or ""),
        }

    if name == "secret_release":
        service = str((tool_input or {}).get("service", "") or "")
        if not service:
            return {"ok": False, "error": "service is required"}
        res = broker.release(service)
        if not isinstance(res, dict) or res.get("ok") is False:
            return {"ok": False,
                    "error": (res or {}).get("error", "release failed")}
        return {"ok": True}

    return {"ok": False, "error": f"unknown secret tool: {name}"}


# ---------------------------------------------------------------------------
# Pipe-backed EventBus shim
# ---------------------------------------------------------------------------

class _PipeEventBus:
    """A drop-in for the provider's ``event_bus`` that forwards emitted Events to
    the daemon over ``DaemonClient.emit`` instead of a real in-process EventBus.

    The provider constructs ``Event`` objects (see providers/base.py); we only
    need ``.type`` (an EventType) and ``.data`` (a dict). We forward the event's
    type value as the emit name and the data as the payload. The daemon's
    WorkerChannel re-emits on the real bus with the source overwritten to the
    pipe-bound agent_id, so ``data["agent_id"]`` from the worker is advisory
    only.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    async def emit(self, event: Any) -> None:
        try:
            etype = getattr(event, "type", None)
            name = getattr(etype, "value", None) or str(etype)
            data = getattr(event, "data", None) or {}
            await self._client.emit(name, dict(data))
        except Exception:
            log.debug("event forward failed", exc_info=True)
