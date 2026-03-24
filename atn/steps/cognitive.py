"""Cognitive step executor — LLM reasoning via provider abstraction.

Supports single-turn (maxTurns=1, the default) and multi-turn tool-use loops.
In multi-turn mode, when the LLM returns tool_use blocks, the step executes
the tool functions, feeds results back, and loops until end_turn or max_turns.

Config keys:
  provider    (str)    Provider name.  Must match a configured provider.
  model       (str)    Model ID.  Falls back to the provider's default.
  system      (str)    System prompt.  Inline string or file reference (see below).
  max_tokens  (int)    Maximum response tokens per turn.  Default: 1024.
  temperature (float)  Sampling temperature.  Default: 0.0.
  tools       (list)   Tool definitions (optional).  Each: {name, description, input_schema}.
  prompt      (str)    User message template.  Inline string or file reference (see below).
                       Can reference previous outputs with {prev} (last step output) or
                       {prev_N} (output of step N).
                       If omitted, previous step output is sent as the user message.

  max_turns   (int)    Maximum LLM turns.  Default: 1 (single turn, no tool loop).
  tool_executors (str) Name of tool executor set.  "orchestrator" uses the built-in
                       orchestrator tools.  Default: None (tools returned but not executed).

File references:
  If system or prompt ends with .md, .txt, or .prompt, it is treated as a
  file path relative to the agent's directory (agents/<id>/).  The file
  contents replace the config value at execution time.  Examples:
    system: system.md           -> reads agents/<id>/system.md
    prompt: prompts/analyze.txt -> reads agents/<id>/prompts/analyze.txt
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from ..events import Event, EventType
from ..models import ExecutionStatus, StepDefinition, StepResult, StepType
from ..providers.base import Provider, ProviderError, ProviderResponse, ToolDefinition, Usage
from .base import StepContext, StepExecutor

# File extensions that signal "this is a file path, not inline content"
_PROMPT_FILE_EXTENSIONS = (".md", ".txt", ".prompt")

log = logging.getLogger(__name__)


class CognitiveStepExecutor(StepExecutor):

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register_provider(self, provider: Provider) -> None:
        """Register an LLM provider by name."""
        self._providers[provider.name] = provider

    def has_provider(self, name: str) -> bool:
        return name in self._providers

    def _resolve_provider_chain(
        self, config: dict[str, Any],
    ) -> tuple[Provider | None, str, list[str], bool]:
        """Resolve the primary provider and fallback chain from step config.

        Config supports:
          provider: "claude_max"                    (single provider)
          provider: ["claude_max", "anthropic"]     (fallback chain)

        Returns (provider, provider_name, fallback_names, is_fallback).
        ``is_fallback`` is True when the resolved provider is NOT the first
        choice in the chain (so the configured model may not apply).
        If resolution fails, returns (None, error_message, [], False).
        """
        raw = config.get("provider", "")
        if not raw:
            return (None, "Cognitive step missing 'provider' in config", [], False)

        if isinstance(raw, list):
            chain = raw
        else:
            chain = [raw]

        # Find the first available provider
        for idx, name in enumerate(chain):
            provider = self._providers.get(name)
            if provider is not None:
                fallbacks = [n for n in chain if n != name and n in self._providers]
                is_fallback = idx > 0  # not the first-choice provider
                return (provider, name, fallbacks, is_fallback)

        available = ", ".join(sorted(self._providers.keys())) or "(none)"
        requested = ", ".join(chain)
        return (None, f"No configured provider found for [{requested}].  Available: {available}", [], False)

    def _pick_fallback(
        self, fallback_names: list[str], *, exclude: set[str],
    ) -> Provider | None:
        """Pick the next available fallback provider, skipping already-tried ones."""
        for name in fallback_names:
            if name not in exclude:
                provider = self._providers.get(name)
                if provider is not None:
                    return provider
        return None

    async def execute(
        self,
        step: StepDefinition,
        step_index: int,
        context: StepContext,
    ) -> StepResult:
        result = StepResult(
            step_index=step_index,
            step_name=step.name or f"cognitive_{step_index}",
            step_type=StepType.COGNITIVE,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        if context.cancel_event.is_set():
            result.status = ExecutionStatus.KILLED
            result.completed_at = datetime.now(timezone.utc)
            return result

        # --- Resolve provider (with fallback chain) ---
        provider, provider_name, fallback_providers, is_fallback = self._resolve_provider_chain(step.config)
        if provider is None:
            result.status = ExecutionStatus.FAILED
            result.error = provider_name  # error message stored here when None
            result.completed_at = datetime.now(timezone.utc)
            return result

        # --- Build user message ---
        user_content = _build_user_message(step.config, context)

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

        # --- Build tools ---
        tools = _build_tools(step.config, context)

        # --- Resolve tool executors ---
        tool_executor_set = step.config.get("tool_executors")
        tool_exec_fn = None
        if tool_executor_set == "orchestrator":
            from ..orchestrator.tools import execute_tool
            tool_exec_fn = execute_tool

        max_turns = step.config.get("max_turns", 1)
        step_name = step.name or f"cognitive_{step_index}"

        # --- Build streaming callback ---
        on_chunk, on_thinking = _make_event_emitters(context, step_index, step_name)

        # --- Orchestrate path ---
        # When tool_executors == "orchestrator", delegate the entire
        # multi-turn tool loop to the provider's send_orchestrate().
        # Every provider supports this: BridgeProvider uses the Claude Agent
        # SDK natively; other providers use the generic base-class loop
        # that drives send_stream() with tool execution.
        use_orchestrate = (
            tool_executor_set == "orchestrator"
            and provider.supports_orchestrate
            and context.runtime is not None
        )

        if use_orchestrate:
            orch_result = await _orchestrate(
                provider, step, context, result, on_chunk,
            )
            if orch_result.status != ExecutionStatus.FAILED:
                return orch_result
            # Primary provider failed — try fallback chain
            log.warning(
                "Orchestrate failed for %s via '%s': %s — trying fallbacks",
                context.agent_id, provider_name, orch_result.error,
            )
            already_tried = {provider_name}
            fallback_provider = self._pick_fallback(fallback_providers, exclude=already_tried)
            while fallback_provider:
                log.info("Falling back to provider '%s' for %s", fallback_provider.name, context.agent_id)
                already_tried.add(fallback_provider.name)
                # Reset result for retry
                result.status = ExecutionStatus.RUNNING
                result.error = None
                orch_result = await _orchestrate(
                    fallback_provider, step, context, result, on_chunk,
                    use_default_model=True,
                )
                if orch_result.status != ExecutionStatus.FAILED:
                    orch_result.output = orch_result.output or {}
                    orch_result.output["fallback_provider"] = fallback_provider.name
                    return orch_result
                log.warning(
                    "Fallback '%s' also failed: %s",
                    fallback_provider.name, orch_result.error,
                )
                fallback_provider = self._pick_fallback(fallback_providers, exclude=already_tried)
            return orch_result  # all providers exhausted

        # --- Simple LLM call (single-turn or multi-turn without orchestrator) ---
        try:
            model = "" if is_fallback else step.config.get("model", "")
            response = await provider.send_stream(
                messages=messages,
                system=_resolve_text(step.config.get("system", ""), context.work_dir),
                model=model,
                max_tokens=step.config.get("max_tokens", 1024),
                tools=tools,
                temperature=step.config.get("temperature", 0.0),
                on_chunk=on_chunk,
                on_thinking=on_thinking,
            )

            result.status = ExecutionStatus.COMPLETED
            result.output = _format_output(response, response.usage, [])
            result.completed_at = datetime.now(timezone.utc)

        except (ProviderError, Exception) as exc:
            error_msg = str(exc) if isinstance(exc, ProviderError) else f"Unexpected error: {exc}"
            if not isinstance(exc, ProviderError):
                log.exception("Cognitive step failed for agent %s", context.agent_id)

            # Try fallback providers
            already_tried = {provider_name}
            fallback_provider = self._pick_fallback(fallback_providers, exclude=already_tried)
            while fallback_provider:
                log.warning(
                    "Provider '%s' failed for %s: %s — falling back to '%s'",
                    provider.name, context.agent_id, error_msg, fallback_provider.name,
                )
                already_tried.add(fallback_provider.name)
                try:
                    response = await fallback_provider.send_stream(
                        messages=[{"role": "user", "content": user_content}],
                        system=_resolve_text(step.config.get("system", ""), context.work_dir),
                        model="",  # use fallback's default
                        max_tokens=step.config.get("max_tokens", 1024),
                        tools=tools,
                        temperature=step.config.get("temperature", 0.0),
                        on_chunk=on_chunk,
                        on_thinking=on_thinking,
                    )
                    result.status = ExecutionStatus.COMPLETED
                    result.output = _format_output(response, response.usage, [])
                    result.output["fallback_provider"] = fallback_provider.name
                    result.completed_at = datetime.now(timezone.utc)
                    break
                except (ProviderError, Exception) as fallback_exc:
                    error_msg = str(fallback_exc)
                    if not isinstance(fallback_exc, ProviderError):
                        log.exception("Fallback provider '%s' also failed", fallback_provider.name)
                    fallback_provider = self._pick_fallback(fallback_providers, exclude=already_tried)
            else:
                # All providers exhausted
                result.status = ExecutionStatus.FAILED
                result.error = error_msg
                result.completed_at = datetime.now(timezone.utc)

        return result


# ---------------------------------------------------------------------------
# Orchestrate — delegates multi-turn tool loop to the provider
# ---------------------------------------------------------------------------

async def _orchestrate(
    provider: Any,
    step: StepDefinition,
    context: StepContext,
    result: StepResult,
    on_chunk: Any,
    *,
    use_default_model: bool = False,
) -> StepResult:
    """Run the orchestrator through the provider's send_orchestrate().

    Works with ANY provider:
    - BridgeProvider: delegates to the Claude Agent SDK subprocess, which
      manages the multi-turn loop natively.
    - Other providers (Anthropic, OpenAI, Ollama): use the generic base-class
      implementation that drives send_stream() in a multi-turn loop.

    Bridge-specific features (session resumption, interrupt hooks) are
    activated when the provider has the relevant attributes.
    """
    from ..orchestrator.tools import execute_tool, get_tool_definitions_for_bridge

    runtime = context.runtime

    # Bridge providers have a _session_id for SDK session resumption.
    # When an active session exists, the SDK handles conversation continuity
    # via resume — skip history prepending in the user message.
    session_id = getattr(provider, '_session_id', "") or ""
    has_session = bool(session_id)
    user_content = _build_user_message(step.config, context, skip_history=has_session)
    system = _resolve_text(step.config.get("system", ""), context.work_dir)

    try:
        # Create a tool executor that routes to connectors or orchestrator tools
        async def _tool_executor(name: str, input: dict) -> dict:
            return await _route_tool_call(
                name, input, context,
                orchestrator_exec_fn=execute_tool,
            )

        # Build tool list: orchestrator tools + connector tools
        orch_tools = get_tool_definitions_for_bridge()
        if context.connectors and context.connector_ids:
            connector_tools = context.connectors.get_all_tools(context.connector_ids)
            orch_tools.extend(
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in connector_tools
            )

        # Register an interrupt hook if the provider supports it
        # (BridgeProvider has an .interrupt() method for kill_execution())
        has_interrupt = hasattr(provider, 'interrupt')
        if has_interrupt and runtime is not None:
            runtime.register_interrupt_hook(
                context.execution_id,
                provider.interrupt,
            )

        model = "" if use_default_model else step.config.get("model", "")
        response = await provider.send_orchestrate(
            message=user_content,
            system=system,
            model=model,
            tools=orch_tools,
            max_turns=step.config.get("max_turns", 20),
            tool_executor=_tool_executor,
            on_chunk=on_chunk,
            session_id=session_id,
        )

        if response.stop_reason == "interrupted":
            result.status = ExecutionStatus.KILLED
            result.error = "Interrupted by user"
        else:
            result.status = ExecutionStatus.COMPLETED
        output: dict[str, Any] = {
            "text": response.text,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": response.usage.cache_read_tokens,
                "cache_creation_tokens": response.usage.cache_creation_tokens,
            },
            "mode": "orchestrate",
        }
        if response.thinking:
            output["thinking"] = response.thinking
        result.output = output
        result.completed_at = datetime.now(timezone.utc)

        # Record assistant turn in conversation history
        if response.text and runtime is not None and hasattr(runtime, "conversation"):
            runtime.conversation.add_assistant_turn(
                response.text, execution_id=context.execution_id,
            )

    except ProviderError as exc:
        result.status = ExecutionStatus.FAILED
        result.error = str(exc)
        result.completed_at = datetime.now(timezone.utc)

    except Exception as exc:
        result.status = ExecutionStatus.FAILED
        result.error = f"Unexpected error: {exc}"
        result.completed_at = datetime.now(timezone.utc)
        log.exception("Orchestrate failed for agent %s via %s", context.agent_id, provider.name)

    finally:
        # Unregister the interrupt hook now that orchestration is done
        if has_interrupt and runtime is not None:
            runtime.unregister_interrupt_hook(context.execution_id)

    return result


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _make_event_emitters(
    context: StepContext, step_index: int, step_name: str,
) -> tuple[Any, Any]:
    """Build on_chunk and on_thinking callbacks that emit step.output events.

    Returns (on_chunk, on_thinking) — both None if no event_bus is available.
    """
    if not context.event_bus:
        return None, None

    async def on_chunk(text: str) -> None:
        if not text:
            return
        await context.event_bus.emit(Event(
            type=EventType.STEP_OUTPUT,
            source=context.agent_id,
            data={
                "agent_id": context.agent_id,
                "execution_id": context.execution_id,
                "step_index": step_index,
                "step_name": step_name,
                "channel": "text",
                "content": text,
            },
        ))

    async def on_thinking(text: str) -> None:
        if not text:
            return
        await context.event_bus.emit(Event(
            type=EventType.STEP_OUTPUT,
            source=context.agent_id,
            data={
                "agent_id": context.agent_id,
                "execution_id": context.execution_id,
                "step_index": step_index,
                "step_name": step_name,
                "channel": "thinking",
                "content": text,
            },
        ))

    return on_chunk, on_thinking


def _resolve_text(value: str, work_dir: Path) -> str:
    """Resolve a config string that may be a file reference.

    If the value ends with a known file extension (.md, .txt, .prompt),
    treat it as a path relative to the agent's work_dir and read the file.
    Otherwise return the string as-is (inline content).
    """
    stripped = value.strip()
    if not stripped:
        return value
    if any(stripped.endswith(ext) for ext in _PROMPT_FILE_EXTENSIONS):
        path = work_dir / stripped
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("Prompt file not found: %s (resolved to %s)", stripped, path)
            return value  # fall back to the raw string
        except Exception as exc:
            log.warning("Failed to read prompt file %s: %s", path, exc)
            return value
    return value


def _build_user_message(
    config: dict[str, Any],
    context: StepContext,
    *,
    skip_history: bool = False,
) -> str:
    """Build the user message from config template + previous outputs.

    Placeholders:
      {prev}    - last step's output
      {prev_N}  - output of step N
      {inbox}   - inbox messages (work messages drained at pipeline start)

    Args:
        skip_history: When True, don't prepend conversation history to the
            message.  Used by the bridge orchestrate path where the SDK
            manages session continuity via ``resume``.
    """
    prompt_template = _resolve_text(config.get("prompt", ""), context.work_dir)

    # Inject current UTC time so every cognitive agent knows what time it is
    now = datetime.now(timezone.utc)
    time_line = f"Current time: {now.strftime('%Y-%m-%dT%H:%M:%SZ')} ({now.strftime('%A, %B %d, %Y')})"

    if prompt_template:
        # Substitute {prev} and {prev_N} placeholders
        prev = context.previous_outputs[-1] if context.previous_outputs else ""
        text = prompt_template.replace("{prev}", _to_str(prev))

        for i, output in enumerate(context.previous_outputs):
            text = text.replace(f"{{prev_{i}}}", _to_str(output))

        # Substitute {inbox} — extract user-facing content from inbox messages
        if "{inbox}" in text:
            parts: list[str] = []
            for m in context.inbox_messages:
                # Prefer data.instruction (the user's actual text)
                instruction = m.data.get("instruction", "") if isinstance(m.data, dict) else ""
                if instruction:
                    # Tag voice-sourced messages so the LLM knows the input
                    # was speech-to-text (may contain transcription artifacts)
                    if m.source == "voice":
                        instruction = f"🎤 [Voice Input] {instruction}"
                    parts.append(instruction)
                elif m.data:
                    parts.append(_to_str(m.data))
            inbox_text = "\n\n".join(parts) if parts else "(no messages)"

            # Prepend conversation history for non-bridge providers that
            # don't have SDK session management.  The bridge path skips
            # this because it uses resume/systemPrompt for continuity.
            if not skip_history:
                history = ""
                if context.runtime and hasattr(context.runtime, "conversation"):
                    history = context.runtime.conversation.get_history_for_prompt()
                if history:
                    inbox_text = history + "\n\nUser: " + inbox_text
            text = text.replace("{inbox}", inbox_text)

        return f"[{time_line}]\n\n{text}"

    # No explicit prompt — use last step's output as the message
    if context.previous_outputs:
        return f"[{time_line}]\n\n{_to_str(context.previous_outputs[-1])}"

    # No previous output and no template — try inbox messages
    if context.inbox_messages:
        inbox_data = [
            {"source": m.source, "type": m.type.value, "data": m.data}
            for m in context.inbox_messages
        ]
        return f"[{time_line}]\n\n{_to_str(inbox_data)}"

    # Nothing available — fall back to a generic message
    return f"[{time_line}]\n\n{config.get('default_prompt', 'Proceed with the task.')}"


def _to_str(val: Any) -> str:
    """Convert any value to a string suitable for an LLM prompt."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, indent=2, default=str)
    except (TypeError, ValueError):
        return str(val)


def _build_tools(config: dict[str, Any], context: StepContext) -> list[ToolDefinition] | None:
    """Parse tool definitions from step config or load from a named set.

    Merges connector tools if the agent has connector_ids.
    """
    tools: list[ToolDefinition] = []

    # Named tool set (e.g. "orchestrator")
    tool_executor_set = config.get("tool_executors")
    if tool_executor_set == "orchestrator":
        from ..orchestrator.tools import get_tool_definitions
        tools.extend(get_tool_definitions())
    else:
        # Inline tool definitions from config
        raw_tools = config.get("tools")
        if raw_tools:
            for t in raw_tools:
                tools.append(ToolDefinition(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", {"type": "object", "properties": {}}),
                ))

    # Merge connector tools if agent has connectors
    if context.connectors and context.connector_ids:
        connector_tools = context.connectors.get_all_tools(context.connector_ids)
        tools.extend(connector_tools)

    return tools if tools else None


async def _route_tool_call(
    name: str,
    input: dict[str, Any],
    context: StepContext,
    orchestrator_exec_fn: Any = None,
) -> dict[str, Any]:
    """Route a tool call to the correct handler.

    Checks for connector tool prefix first, falls back to orchestrator tools.
    """
    # Check if it's a connector tool (mcp_{connector_id}_{tool_name})
    if context.connectors:
        parsed = context.connectors.parse_tool_name(name)
        if parsed:
            connector_id, tool_name = parsed
            return await context.connectors.call_tool(connector_id, tool_name, input)

    # Fall back to orchestrator tool executor
    if orchestrator_exec_fn and context.runtime:
        return await orchestrator_exec_fn(name, input, context.runtime)

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_output(
    response: ProviderResponse,
    cumulative_usage: Usage,
    turn_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format the provider response as structured step output."""
    usage_dict: dict[str, Any] = {
        "input_tokens": cumulative_usage.input_tokens,
        "output_tokens": cumulative_usage.output_tokens,
    }
    if cumulative_usage.cache_read_tokens:
        usage_dict["cache_read_tokens"] = cumulative_usage.cache_read_tokens
    if cumulative_usage.cache_creation_tokens:
        usage_dict["cache_creation_tokens"] = cumulative_usage.cache_creation_tokens

    output: dict[str, Any] = {
        "text": response.text,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "usage": usage_dict,
    }

    # Include thinking blocks (provider-agnostic reasoning trace)
    if response.thinking:
        output["thinking"] = response.thinking

    # Include tool calls from the final response (for single-turn compatibility)
    if response.tool_calls:
        output["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "input": tc.input}
            for tc in response.tool_calls
        ]

    # Include turn history for multi-turn interactions
    if len(turn_history) > 1:
        output["turns"] = turn_history
        output["total_turns"] = len(turn_history)

    return output
