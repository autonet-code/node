"""Orchestrator — the meta-agent that manages the fleet.

The orchestrator is a cognitive-mode AgentDefinition — identical to any other
cognitive agent, just with a full tool surface and its own system prompt.
It's registered in the Runtime like any other agent and wakes on inbox
messages (user input, alert escalations).
"""
from __future__ import annotations

from pathlib import Path

from ..config import OrchestratorConfig
from ..models import AgentDefinition, AgentMode

ORCHESTRATOR_ID = "orchestrator"


def build_system_prompt_with_context(
    *,
    data_dir: Path | None = None,
    onboarding: bool = False,
) -> str:
    """Build the orchestrator's system prompt.

    Uses the common base from delegate_prompts (shared with all cognitive
    agents) plus the orchestrator specialization layer.

    Args:
        data_dir: ATN data directory (for resolving USER.md path).
        onboarding: If True, return the onboarding prompt instead.

    Returns:
        The complete system prompt string.
    """
    if onboarding:
        from ..onboarding_prompt import ONBOARDING_SYSTEM_PROMPT
        return ONBOARDING_SYSTEM_PROMPT

    from ..delegate_prompts import build_common_base, build_orchestrator_layer

    user_md = str((data_dir or Path.home() / ".atn") / "USER.md")
    base = build_common_base(
        agent_id=ORCHESTRATOR_ID,
        agent_type="orchestrator",
        parent_id="(root — user-facing)",
    )
    orch_layer = build_orchestrator_layer(user_md_path=user_md)
    return base + orch_layer


def create_orchestrator_agent(
    config: OrchestratorConfig | None = None,
    *,
    system_prompt: str = "",
    max_turns: int = 50,
) -> AgentDefinition:
    """Build the orchestrator AgentDefinition.

    The orchestrator is a **cognitive-mode** agent — identical in structure
    to any child cognitive agent.  The differences are configuration-level:
    progressive tool surface ("atn_progressive"), provider fallback chain,
    and the fleet management system prompt.

    Args:
        config: Orchestrator config from ATNConfig (provider, model).
        system_prompt: Override the default system prompt.
        max_turns: Max LLM turns per invocation.

    Returns:
        An AgentDefinition ready to register with the Runtime.
    """
    config = config or OrchestratorConfig()

    # Primary provider/model come from config; defaults target Claude Max + Opus 4.7.
    primary_provider = config.provider or "claude_max"
    model = config.model or "claude-opus-4-7"
    # Provider fallback chain: primary first, then alternatives.
    # Providers that aren't configured are silently skipped at runtime.
    # Ollama is excluded — local models can't reliably handle 20+ tools,
    # multi-turn planning, and the complex reasoning the orchestrator needs.
    provider_chain = [primary_provider]
    _FALLBACK_ORDER = ["claude_max", "codex_max", "anthropic", "gemini"]
    for p in _FALLBACK_ORDER:
        if p not in provider_chain:
            provider_chain.append(p)

    built_prompt = system_prompt or build_system_prompt_with_context()

    return AgentDefinition(
        id=ORCHESTRATOR_ID,
        name="Orchestrator",
        mode=AgentMode.COGNITIVE,
        description="Meta-agent that manages the fleet via multi-turn tool use.",
        system_prompt=built_prompt,
        provider=provider_chain,
        cognitive_model=model,
        max_turns=max_turns,
        tools=["atn_progressive"],
        concurrency=1,
    )
