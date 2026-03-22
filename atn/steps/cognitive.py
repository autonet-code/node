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

        # --- Check for bridge orchestrate shortcut ---
        # When the orchestrator runs through the BridgeProvider, delegate the
        # entire multi-turn tool loop to the bridge (SDK handles it natively).
        # On failure, fall back to the generic multi-turn loop (possibly with
        # a different provider from the fallback chain).
        from ..providers.bridge import BridgeProvider
        use_bridge_orchestrate = (
            tool_executor_set == "orchestrator"
            and isinstance(provider, BridgeProvider)
            and context.runtime is not None
        )

        if use_bridge_orchestrate:
            bridge_result = await _bridge_orchestrate(
                provider, step, context, result, on_chunk,
            )
            if bridge_result.status != ExecutionStatus.FAILED:
                return bridge_result
            # Bridge failed — try fallback providers via generic loop
            log.warning(
                "Bridge orchestrate failed for %s: %s — trying fallback providers",
                context.agent_id, bridge_result.error,
            )
            fallback_provider = self._pick_fallback(fallback_providers, exclude={provider_name})
            if fallback_provider:
                provider = fallback_provider
                log.info("Falling back to provider '%s' for %s", provider.name, context.agent_id)
                # Reset result for retry
                result.status = ExecutionStatus.RUNNING
                result.error = None
            else:
                return bridge_result  # no fallback available

        # --- LLM call (single or multi-turn) ---
        try:
            cumulative_usage = Usage()
            turn_history: list[dict[str, Any]] = []
            final_response: ProviderResponse | None = None

            for turn in range(max_turns):
                if context.cancel_event.is_set():
                    result.status = ExecutionStatus.KILLED
                    result.completed_at = datetime.now(timezone.utc)
                    return result

                # When using a fallback provider, use its default model instead
                # of the configured model (which belongs to the primary provider).
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

                # Accumulate usage across turns
                cumulative_usage.input_tokens += response.usage.input_tokens
                cumulative_usage.output_tokens += response.usage.output_tokens
                cumulative_usage.cache_read_tokens += response.usage.cache_read_tokens
                cumulative_usage.cache_creation_tokens += response.usage.cache_creation_tokens

                final_response = response

                # Record this turn
                turn_record: dict[str, Any] = {
                    "turn": turn + 1,
                    "text": response.text,
                    "stop_reason": response.stop_reason,
                }
                if response.tool_calls:
                    turn_record["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in response.tool_calls
                    ]

                # Single-turn or end_turn — done
                if not response.tool_calls or response.stop_reason != "tool_use":
                    turn_history.append(turn_record)
                    break

                # Multi-turn with tool calls — execute tools and continue
                has_connectors = bool(context.connectors and context.connector_ids)
                if tool_exec_fn is None and not has_connectors:
                    # No executor configured — return tool calls as-is (Phase 1 behavior)
                    turn_history.append(turn_record)
                    break

                # Execute each tool call
                tool_results: list[dict[str, Any]] = []
                for tc in response.tool_calls:
                    log.info(
                        "Executing tool %s for agent %s (turn %d)",
                        tc.name, context.agent_id, turn + 1,
                    )
                    tr = await _route_tool_call(
                        tc.name, tc.input, context,
                        orchestrator_exec_fn=tool_exec_fn,
                    )
                    tool_results.append({
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "result": tr,
                    })

                turn_record["tool_results"] = tool_results
                turn_history.append(turn_record)

                # Build assistant message with tool_use content blocks
                assistant_content: list[dict[str, Any]] = []
                if response.text:
                    assistant_content.append({"type": "text", "text": response.text})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Build tool_result messages
                tool_result_content: list[dict[str, Any]] = []
                for tr in tool_results:
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tr["tool_call_id"],
                        "content": json.dumps(tr["result"], default=str),
                    })
                messages.append({"role": "user", "content": tool_result_content})

            # --- Format output ---
            if final_response is None:
                result.status = ExecutionStatus.FAILED
                result.error = "No response from provider"
                result.completed_at = datetime.now(timezone.utc)
                return result

            result.status = ExecutionStatus.COMPLETED
            result.output = _format_output(final_response, cumulative_usage, turn_history)
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
                provider = fallback_provider
                messages = [{"role": "user", "content": user_content}]  # reset conversation
                try:
                    cumulative_usage = Usage()
                    turn_history = []
                    final_response = None
                    for turn in range(max_turns):
                        if context.cancel_event.is_set():
                            result.status = ExecutionStatus.KILLED
                            result.completed_at = datetime.now(timezone.utc)
                            return result
                        response = await provider.send_stream(
                            messages=messages,
                            system=_resolve_text(step.config.get("system", ""), context.work_dir),
                            model="",  # use fallback provider's default model
                            max_tokens=step.config.get("max_tokens", 1024),
                            tools=tools,
                            temperature=step.config.get("temperature", 0.0),
                            on_chunk=on_chunk,
                            on_thinking=on_thinking,
                        )
                        cumulative_usage.input_tokens += response.usage.input_tokens
                        cumulative_usage.output_tokens += response.usage.output_tokens
                        cumulative_usage.cache_read_tokens += response.usage.cache_read_tokens
                        cumulative_usage.cache_creation_tokens += response.usage.cache_creation_tokens
                        final_response = response
                        turn_record = {"turn": turn + 1, "text": response.text, "stop_reason": response.stop_reason}
                        if response.tool_calls:
                            turn_record["tool_calls"] = [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in response.tool_calls]
                        if not response.tool_calls or response.stop_reason != "tool_use":
                            turn_history.append(turn_record)
                            break
                        has_connectors = bool(context.connectors and context.connector_ids)
                        if tool_exec_fn is None and not has_connectors:
                            turn_history.append(turn_record)
                            break
                        tool_results = []
                        for tc in response.tool_calls:
                            log.info("Executing tool %s for agent %s (turn %d)", tc.name, context.agent_id, turn + 1)
                            tr = await _route_tool_call(tc.name, tc.input, context, orchestrator_exec_fn=tool_exec_fn)
                            tool_results.append({"tool_call_id": tc.id, "name": tc.name, "result": tr})
                        turn_record["tool_results"] = tool_results
                        turn_history.append(turn_record)
                        assistant_content = []
                        if response.text:
                            assistant_content.append({"type": "text", "text": response.text})
                        for tc in response.tool_calls:
                            assistant_content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
                        messages.append({"role": "assistant", "content": assistant_content})
                        tool_result_content = []
                        for tr in tool_results:
                            tool_result_content.append({"type": "tool_result", "tool_use_id": tr["tool_call_id"], "content": json.dumps(tr["result"], default=str)})
                        messages.append({"role": "user", "content": tool_result_content})

                    if final_response is not None:
                        result.status = ExecutionStatus.COMPLETED
                        result.output = _format_output(final_response, cumulative_usage, turn_history)
                        result.output["fallback_provider"] = provider.name
                        result.completed_at = datetime.now(timezone.utc)
                        break  # success
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

        # Record assistant turn in conversation history (orchestrator)
        if (result.status == ExecutionStatus.COMPLETED
                and tool_executor_set == "orchestrator"
                and context.runtime is not None
                and hasattr(context.runtime, "conversation")
                and result.output):
            text = result.output.get("text", "")
            if text:
                context.runtime.conversation.add_assistant_turn(
                    text, execution_id=context.execution_id,
                )

        return result


# ---------------------------------------------------------------------------
# Bridge orchestrate — delegates multi-turn to SDK
# ---------------------------------------------------------------------------

async def _bridge_orchestrate(
    provider: Any,
    step: StepDefinition,
    context: StepContext,
    result: StepResult,
    on_chunk: Any,
) -> StepResult:
    """Run the orchestrator through BridgeProvider.send_orchestrate().

    The bridge handles the entire multi-turn tool loop natively via the
    Claude Agent SDK.  Tool calls are relayed back to Python for execution
    against the Runtime.
    """
    from ..orchestrator.tools import execute_tool, get_tool_definitions_for_bridge

    try:
        # If we have an active SDK session (session_id), the SDK handles
        # conversation continuity via resume — skip history prepending.
        # If session_id is empty (fresh start or daemon restart), prepend
        # conversation history so the model gets context from active.jsonl.
        has_session = bool(provider._session_id)
        user_content = _build_user_message(step.config, context, skip_history=has_session)
        system = _resolve_text(step.config.get("system", ""), context.work_dir)
        # Create a tool executor that routes to connectors or orchestrator tools
        runtime = context.runtime

        async def _tool_executor(name: str, input: dict) -> dict:
            return await _route_tool_call(
                name, input, context,
                orchestrator_exec_fn=execute_tool,
            )

        # Build tool list: orchestrator tools + connector tools
        bridge_tools = get_tool_definitions_for_bridge()
        if context.connectors and context.connector_ids:
            connector_tools = context.connectors.get_all_tools(context.connector_ids)
            bridge_tools.extend(
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in connector_tools
            )

        # Register an interrupt hook so kill_execution() can interrupt the bridge
        if runtime is not None:
            runtime.register_interrupt_hook(
                context.execution_id,
                provider.interrupt,
            )

        # If we got here via fallback, use provider's default model
        bridge_model = step.config.get("model", "")
        response = await provider.send_orchestrate(
            message=user_content,
            system=system,
            model=bridge_model,
            tools=bridge_tools,
            max_turns=step.config.get("max_turns", 20),
            tool_executor=_tool_executor,
            on_chunk=on_chunk,
            session_id=provider._session_id,
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
            "mode": "bridge_orchestrate",
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
        log.exception("Bridge orchestrate failed for agent %s", context.agent_id)

    finally:
        # Unregister the interrupt hook now that orchestration is done
        if runtime is not None:
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

        return text

    # No explicit prompt — use last step's output as the message
    if context.previous_outputs:
        return _to_str(context.previous_outputs[-1])

    # No previous output and no template — try inbox messages
    if context.inbox_messages:
        inbox_data = [
            {"source": m.source, "type": m.type.value, "data": m.data}
            for m in context.inbox_messages
        ]
        return _to_str(inbox_data)

    # Nothing available — fall back to a generic message
    return config.get("default_prompt", "Proceed with the task.")


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
