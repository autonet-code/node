"""The system prompt an agent runs with, in one place.

The execution engine, the /context breakdown and the app (to show the user
the real prompt before they change it) all read it from here, so what the
user sees is what the agent gets.

Parts, in order:
  constitution  the jurisdiction's, prepended for on-chain registered agents;
                never editable by the owner or the agent.
  body          the owner's custom prompt when one is set, else ATN's
                (common base + the tool notes for its granted categories +
                the generalist section).

An imported Claude Code session runs on Claude Code's own system prompt
(the bridge loads it with the user's settings), so ATN adds none.
"""
from __future__ import annotations

from typing import Any


def system_prompt_parts(runtime: Any, defn: Any = None, *,
                        tools: list[str] | None = None) -> dict[str, Any]:
    """``defn`` is the agent; None describes a new agent with ``tools``."""
    from .delegate_prompts import build_system_prompt
    from .harness_policy import policy_for_agent, reference_policy

    if defn is not None and getattr(defn, "claude_session", None):
        return {"imported": True, "custom": False, "constitution": "",
                "constitution_applies": False, "body": "", "default_body": "",
                "prompt": ""}
    policy = policy_for_agent(runtime, defn) if defn is not None else reference_policy()
    granted = (getattr(defn, "tools", None) or []) if defn is not None else (tools or [])
    default_body = build_system_prompt(
        "general", getattr(defn, "id", "") or "", getattr(defn, "parent_id", None),
        tool_categories=policy.select_tools(granted), policy=policy)
    custom = str(getattr(defn, "system_prompt", "") or "") if defn is not None else ""
    body = custom or default_body

    identity = getattr(defn, "identity", None) if defn is not None else None
    registered = bool(identity and getattr(identity, "registered_on_chain", False))
    bridge = getattr(runtime, "autonet", None) if runtime is not None else None
    constitution_text = str(getattr(bridge, "constitution_text", "") or "")
    constitution = ""
    if constitution_text:
        from .delegate_prompts import build_constitutional_preamble
        constitution = build_constitutional_preamble(constitution_text)
    applies = registered and bool(constitution)
    return {"imported": False, "custom": bool(custom),
            "constitution": constitution, "constitution_applies": applies,
            "body": body, "default_body": default_body,
            "prompt": (constitution if applies else "") + body}
