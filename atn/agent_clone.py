"""Agent cloning — conversation branching via full-memory duplicates.

A clone is a new agent provisioned with a copy of the original's
definition AND its conversation history — unlike a spawned child, which
starts with a fresh context. The intended use is a human-driven
"sidequest": fork the agent, explore something in the branch without
polluting the original's memory, then merge back a self-authored brief.

Deliberate properties:
  * HUMAN-ONLY. clone/merge are exposed as WS handlers, never registered
    in any agent tool surface — an agent cannot name the capability, so
    self-replication is structurally impossible rather than permission-
    checked.
  * NO INHERITED IDENTITY. The clone gets none of the original's
    identity/wallet fields; registration generates it a fresh local
    keypair like any new agent, and it only goes on-chain if the user
    registers it later.
  * SHARED BUDGET. The clone is registered with parent_id = original,
    so its token spend rolls up into the original's budget counters
    (AgentRegistry.record_token_usage walks ancestors) — the human's
    spend ceiling does not multiply by cloning.
  * NO SCHEDULE. Heartbeat/schedule are stripped; a clone runs only
    when spoken to.

Merge-back: the clone itself authors the brief (it knows why the
sidequest happened), the daemon delivers that reply verbatim to the
original's inbox and deactivates the clone.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from .models import AgentDefinition

log = logging.getLogger(__name__)

# How long the merge watcher waits for the brief run to start / finish.
_MERGE_START_TIMEOUT_S = 90.0
_MERGE_FINISH_TIMEOUT_S = 600.0
_MERGE_POLL_S = 2.0

_MERGE_INSTRUCTION = (
    "[MERGE-BACK REQUEST] You are a clone of agent '{original}' that was "
    "forked for a sidequest, and the sidequest is now concluding. Write a "
    "briefing for '{original}' covering: what the sidequest explored, what "
    "was learned or decided, anything rejected and why, and concrete "
    "next-step recommendations. Write it AS the briefing itself (no "
    "preamble to the user) — your entire reply will be delivered verbatim "
    "to '{original}', and you will then be deactivated."
)


def _unique_clone_id(runtime: Any, base_id: str) -> str:
    candidate = f"{base_id}-clone"
    n = 2
    while runtime.get_agent(candidate) is not None:
        candidate = f"{base_id}-clone{n}"
        n += 1
    return candidate


async def clone_agent(runtime: Any, agent_id: str) -> dict[str, Any]:
    """Create a full-memory clone of a cognitive agent. Human-only surface."""
    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Agent '{agent_id}' not found."}
    if not defn.is_cognitive:
        return {"error": "Only cognitive agents can be cloned (pipeline agents have no conversation to fork)."}

    clone_id = _unique_clone_id(runtime, agent_id)
    clone = AgentDefinition(
        id=clone_id,
        name=f"{defn.name} (clone)",
        mode=defn.mode,
        provider=defn.provider,
        cognitive_model=defn.cognitive_model,
        system_prompt=defn.system_prompt,
        task_prompt=defn.task_prompt,
        agent_type=defn.agent_type,
        max_turns=defn.max_turns,
        tools=list(defn.tools),
        concurrency=defn.concurrency,
        # A clone never inherits autonomy: no schedule, no heartbeat.
        schedule=None,
        heartbeat=None,
        output_schema=defn.output_schema,
        description=defn.description,
        # NO own budget declaration: a declared child budget is a carve-out
        # (the cascade enforcer requires it to fit the ancestor's REMAINING
        # allowance), but a clone SHARES the original's ceiling instead —
        # its spend rolls up into the original's counters via parent_id and
        # the original's own caps bind the combined total.
        budgets={},
        secrets_allowance=defn.secrets_allowance,
        max_children=defn.max_children,
        max_depth_below=defn.max_depth_below,
        per_turn_input_max=defn.per_turn_input_max,
        repeat_call_limit=defn.repeat_call_limit,
        connector_ids=list(defn.connector_ids),
        # parent_id = original → token spend rolls up into the original's
        # budget counters (shared ceiling). The UI presents the clone
        # relationship via cloned_from, not the parent pointer.
        parent_id=agent_id,
        created_by="user",
        # The deliberate merge brief is the reporting channel; skip the
        # framework's lean parent completion notifications.
        notify_parent=False,
        cloned_from=agent_id,
        # identity / wallet / alignment intentionally NOT copied — the
        # registry mints a fresh local keypair at registration.
        sponsor_agent_id=defn.sponsor_agent_id,
        sponsor_address=defn.sponsor_address,
        training_charter=defn.training_charter,
        # Marketplace inference binding carries over: a clone must think on the
        # same substrate as the original or it isn't a clone. It pays from its
        # OWN fresh wallet though (identity is not copied, see above), so the
        # human doing the cloning must fund it before it can buy a completion —
        # the provider fails loud on an unfunded call rather than quietly
        # billing the original.
        service_provider=(dict(defn.service_provider)
                          if defn.service_provider else None),
    )

    # legacy=True: a clone preserves the original's budget posture even if
    # the original predates mandatory budgets; enforcement still applies
    # through the ancestor chain.
    try:
        await runtime.registry.register_agent(clone, legacy=True)
    except ValueError as exc:
        # Spawn-limit / budget-cascade rejection — report, don't crash the
        # WS surface.
        return {"error": f"Clone registration rejected: {exc}"}

    # Fork the conversation: copy the persisted store files BEFORE anything
    # hydrates a store for the clone. Works for the orchestrator too (its
    # store lives centrally; the clone gets a normal per-agent store).
    try:
        src_store = runtime.get_agent_conversation_store(agent_id)
        dst_dir = runtime._config.data_dir / "agents" / clone_id / "conversations"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if src_store._active_path.exists():
            shutil.copy2(src_store._active_path, dst_dir / "active.jsonl")
        if src_store._stats_path.exists():
            shutil.copy2(src_store._stats_path, dst_dir / "session_stats.json")
    except Exception:
        log.warning("Conversation fork failed for clone %s of %s",
                    clone_id, agent_id, exc_info=True)

    # Persist the definition and wake the clone up (idle until messaged).
    try:
        from .loader import save_agent
        save_agent(clone, runtime._config.agents_dir)
    except Exception:
        log.warning("save_agent failed for clone %s", clone_id, exc_info=True)
    await runtime.registry.activate_agent(clone_id)

    turns = 0
    try:
        turns = len(runtime.get_agent_conversation_store(clone_id).get_turns())
    except Exception:
        pass
    log.info("Cloned agent %s -> %s (%d turns forked)", agent_id, clone_id, turns)
    return {
        "status": "cloned",
        "agent_id": clone_id,
        "cloned_from": agent_id,
        "name": clone.name,
        "turns_forked": turns,
    }


async def merge_clone(runtime: Any, agent_id: str) -> dict[str, Any]:
    """Ask a clone to author its merge-back brief, then deliver + retire it.

    Returns immediately with status "merging"; a background task waits for
    the brief run to finish, posts the reply to the original's inbox and
    deactivates the clone.
    """
    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Agent '{agent_id}' not found."}
    original_id = getattr(defn, "cloned_from", None)
    if not original_id:
        return {"error": f"Agent '{agent_id}' is not a clone."}
    if runtime.get_agent(original_id) is None:
        return {"error": f"Original agent '{original_id}' no longer exists."}
    if runtime.registry._running_count.get(agent_id, 0) > 0:
        return {"error": "Clone is mid-run — wait for it to finish before merging."}

    instruction = _MERGE_INSTRUCTION.format(original=original_id)
    # Trusted internal caller (surface=None) — never mic-gated.
    result = await runtime.send_agent_message(agent_id, instruction)
    if isinstance(result, dict) and result.get("error"):
        return {"error": f"Could not deliver merge request: {result['error']}"}

    asyncio.create_task(
        _watch_merge(runtime, agent_id, original_id),
        name=f"merge-clone-{agent_id}",
    )
    return {"status": "merging", "agent_id": agent_id, "original": original_id}


async def _watch_merge(runtime: Any, clone_id: str, original_id: str) -> None:
    """Wait for the brief run, deliver the reply to the original, retire the clone."""
    try:
        # Phase 1: wait for the run to start (the message triggers it async).
        waited = 0.0
        while runtime.registry._running_count.get(clone_id, 0) == 0 \
                and waited < _MERGE_START_TIMEOUT_S:
            await asyncio.sleep(_MERGE_POLL_S)
            waited += _MERGE_POLL_S
        # Phase 2: wait for it to finish.
        waited = 0.0
        while runtime.registry._running_count.get(clone_id, 0) > 0 \
                and waited < _MERGE_FINISH_TIMEOUT_S:
            await asyncio.sleep(_MERGE_POLL_S)
            waited += _MERGE_POLL_S

        brief = ""
        try:
            turns = runtime.get_agent_conversation_store(clone_id).get_turns()
            for turn in reversed(turns):
                if turn.role == "assistant" and turn.content.strip():
                    brief = turn.content.strip()
                    break
        except Exception:
            log.warning("Could not read brief from clone %s", clone_id, exc_info=True)

        if not brief:
            log.warning("Merge of clone %s produced no brief; delivering a stub", clone_id)
            brief = ("(The clone produced no brief before retiring — check its "
                     "conversation history directly.)")

        delivery = (
            f"[SIDEQUEST BRIEF from clone '{clone_id}']\n\n{brief}"
        )
        res = await runtime.send_agent_message(original_id, delivery)
        if isinstance(res, dict) and res.get("error"):
            log.warning("Brief delivery to %s failed: %s", original_id, res["error"])

        await runtime.registry.deactivate_agent(clone_id)
        log.info("Clone %s merged back into %s and deactivated", clone_id, original_id)
    except Exception:
        log.warning("Merge watcher for %s failed", clone_id, exc_info=True)
