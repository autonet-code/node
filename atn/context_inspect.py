"""Context-window breakdown — the daemon's /context surface.

Computes a per-category composition of an agent's LLM context window:
system prompt, tool definitions, message history (itemized per message),
output reservation, the pre-send reduction buffer, and free space.

All per-category figures are chars/4 estimates (the codebase-wide
convention — there is no exact tokenizer anywhere in atn/). The payload
also carries the provider's real measured ``last_input_tokens`` from the
previous turn so consumers can show estimate vs. measured; for bridge
(SDK) agents the difference is the SDK's own overhead (its system
prompt + native tools), reported as ``unaccounted_tokens``.

Three sources, in fidelity order:
  live_worker     — read inside the worker process off the running
                    provider's live snapshot (isolation ON)
  live_inprocess  — read off the daemon-side provider (isolation OFF,
                    or cached provider after a run)
  reconstructed   — rebuilt from the agent definition + persisted
                    conversation store (idle / bridge)
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_PREVIEW_CHARS = 220
# Cap itemized message entries; older messages beyond the cap are rolled
# into one aggregate entry so the payload stays bounded on long histories.
_MAX_MESSAGE_ITEMS = 300


def _est_tokens(chars: int) -> int:
    return max(0, chars) // _CHARS_PER_TOKEN


def _preview(text: str, cap: int = _PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _message_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, default=str))
    except Exception:
        return len(str(content))


def _describe_message(msg: dict[str, Any]) -> dict[str, Any]:
    """One itemized entry for a canonical message.

    Canonical content is either a plain string or a list of blocks
    (text / tool_use / tool_result). The entry surfaces enough to build
    a meaningful hover: role, kind, per-block tool names, sizes, preview.
    """
    role = str(msg.get("role", ""))
    content = msg.get("content")
    chars = _message_chars(content)
    kinds: list[str] = []
    tool_names: list[str] = []
    preview = ""
    if isinstance(content, str):
        kinds.append("text")
        preview = _preview(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                kinds.append("text")
                if not preview:
                    preview = _preview(str(block.get("text", "")))
            elif btype == "tool_use":
                kinds.append("tool_use")
                name = str(block.get("name", ""))
                if name:
                    tool_names.append(name)
                if not preview:
                    try:
                        preview = _preview(json.dumps(block.get("input", {}), default=str))
                    except Exception:
                        pass
            elif btype == "tool_result":
                kinds.append("tool_result")
                body = block.get("content")
                if not preview and isinstance(body, str):
                    preview = _preview(body)
            elif btype:
                kinds.append(str(btype))
    entry: dict[str, Any] = {
        "role": role,
        "kinds": kinds or ["text"],
        "chars": chars,
        "est_tokens": _est_tokens(chars),
        "preview": preview,
    }
    if tool_names:
        entry["tool_names"] = tool_names
    ts = msg.get("timestamp") or msg.get("ts")
    if ts:
        entry["ts"] = str(ts)
    return entry


def _message_items(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Itemize messages (newest kept, oldest aggregated past the cap)."""
    entries = [_describe_message(m) for m in messages]
    total_tokens = sum(e["est_tokens"] for e in entries)
    if len(entries) > _MAX_MESSAGE_ITEMS:
        overflow = entries[: len(entries) - _MAX_MESSAGE_ITEMS]
        kept = entries[len(entries) - _MAX_MESSAGE_ITEMS:]
        agg_chars = sum(e["chars"] for e in overflow)
        entries = [{
            "role": "aggregate",
            "kinds": ["aggregate"],
            "chars": agg_chars,
            "est_tokens": _est_tokens(agg_chars),
            "preview": f"{len(overflow)} older messages (itemization capped)",
            "count": len(overflow),
        }] + kept
    for i, e in enumerate(entries):
        e["index"] = i
    return entries, total_tokens


def _tool_items(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    total = 0
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        try:
            chars = len(json.dumps(t, default=str))
        except Exception:
            chars = len(str(t))
        tokens = _est_tokens(chars)
        total += tokens
        desc = str(t.get("description", ""))
        items.append({
            "name": str(t.get("name", "")),
            "est_tokens": tokens,
            "chars": chars,
            "preview": _preview(desc.split("\n", 1)[0], 140),
        })
    items.sort(key=lambda x: -x["est_tokens"])
    return items, total


def breakdown_from_parts(
    *,
    system: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    stats: dict[str, Any] | None,
    source: str,
    output_reserve_tokens: int = 0,
) -> dict[str, Any]:
    """Assemble the breakdown payload from raw context parts + session stats."""
    from .providers.base import get_context_window, _reduction_buffer

    stats = dict(stats or {})
    model = str(stats.get("active_model") or "")
    window = int(stats.get("context_window") or 0)
    if window <= 0 and model:
        window = get_context_window(model)

    sys_chars = len(system or "")
    tool_list, tools_tokens = _tool_items(tools or [])
    msg_list, msg_tokens = _message_items(messages or [])
    sys_tokens = _est_tokens(sys_chars)
    est_used = sys_tokens + tools_tokens + msg_tokens

    measured = int(stats.get("last_input_tokens") or 0)
    buffer_tokens = _reduction_buffer(window) if window > 0 else 0
    # Best single number for "how full": the real measurement when we have
    # one and it exceeds our estimate (it includes provider-side overhead
    # the estimate can't see), else the estimate.
    used_tokens = max(est_used, measured)
    free_tokens = max(0, window - used_tokens - output_reserve_tokens - buffer_tokens) if window > 0 else 0

    return {
        "source": source,
        "model": model,
        "context_window": window,
        "output_reserve_tokens": int(output_reserve_tokens or 0),
        "reduction_buffer_tokens": buffer_tokens,
        "est_used_tokens": est_used,
        "used_tokens": used_tokens,
        "free_tokens": free_tokens,
        "measured_last_input_tokens": measured,
        "unaccounted_tokens": max(0, measured - est_used),
        "context_used_pct": stats.get("context_used_pct"),
        "num_turns": stats.get("num_turns", 0),
        "total_cost_usd": stats.get("total_cost_usd", 0),
        "compaction_count": stats.get("compaction_count", 0),
        "session_id": stats.get("session_id", ""),
        "cache": {
            "read": stats.get("cumulative_cache_read", 0),
            "creation": stats.get("cumulative_cache_creation", 0),
        },
        "cumulative": {
            "input": stats.get("cumulative_input_tokens", 0),
            "output": stats.get("cumulative_output_tokens", 0),
        },
        "system_prompt": {
            "est_tokens": sys_tokens,
            "chars": sys_chars,
            "preview": _preview(system or "", 400),
        },
        "tools": {
            "est_tokens": tools_tokens,
            "count": len(tool_list),
            "items": tool_list,
        },
        "messages": {
            "est_tokens": msg_tokens,
            "count": len(messages or []),
            "items": msg_list,
        },
    }


def breakdown_from_provider(provider: Any, source: str) -> dict[str, Any] | None:
    """Breakdown from a provider's live snapshot (set by send_orchestrate).

    Returns None when the provider has no live snapshot — e.g. bridge/SDK
    providers, which own their loop elsewhere, or a provider that has
    never run. Callers fall back to the reconstructed path.
    """
    messages = getattr(provider, "_live_messages", None)
    if not isinstance(messages, list):
        return None
    system = getattr(provider, "_live_system", "") or ""
    tools = getattr(provider, "_live_tools", None) or []
    reserve = int(getattr(provider, "_live_max_tokens", 0) or 0)
    stats = None
    try:
        raw = getattr(provider, "session_stats", None)
        if isinstance(raw, dict):
            stats = raw
    except Exception:
        log.debug("session_stats read failed for live breakdown", exc_info=True)
    return breakdown_from_parts(
        system=str(system),
        tools=list(tools),
        messages=list(messages),
        stats=stats,
        source=source,
        output_reserve_tokens=reserve,
    )


def breakdown_reconstructed(runtime: Any, agent_id: str) -> dict[str, Any]:
    """Breakdown for an idle (or bridge) agent, rebuilt daemon-side.

    Mirrors the execution engine's context assembly without running it:
    system prompt from the definition (custom or delegate template), tool
    surface via resolve_tool_surface, history from the persisted
    conversation store, stats from the live/cached/persisted 3-tier lookup.
    Connector- and surface-contributed tools are not included (they only
    exist while a run is being assembled), so tool counts here are a floor.
    """
    defn = None
    try:
        defn = runtime.get_agent(agent_id)
    except Exception:
        pass

    system = ""
    tools: list[dict[str, Any]] = []
    if defn is not None:
        try:
            system = getattr(defn, "system_prompt", "") or ""
            if not system:
                from .delegate_prompts import build_delegate_prompt
                system = build_delegate_prompt(
                    getattr(defn, "agent_type", "") or "generic",
                    defn.id,
                    getattr(defn, "parent_id", None),
                    tool_categories=getattr(defn, "tools", None) or [],
                )
        except Exception:
            log.debug("system prompt reconstruction failed for %s", agent_id, exc_info=True)
        try:
            from .orchestrator.tools import resolve_tool_surface
            tools = resolve_tool_surface(getattr(defn, "tools", None) or [])
        except Exception:
            log.debug("tool surface reconstruction failed for %s", agent_id, exc_info=True)

    messages: list[dict[str, Any]] = []
    try:
        store = runtime.get_agent_conversation_store(agent_id)
        for turn in store.get_turns():
            messages.append({
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
            })
    except Exception:
        log.debug("conversation store read failed for %s", agent_id, exc_info=True)

    stats = None
    try:
        raw = runtime.get_session_stats(agent_id)
        if isinstance(raw, dict) and "error" not in raw:
            stats = raw
    except Exception:
        pass

    bd = breakdown_from_parts(
        system=system,
        tools=tools,
        messages=messages,
        stats=stats,
        source="reconstructed",
    )
    bd["agent_id"] = agent_id
    return bd
