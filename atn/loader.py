"""Agent definition loader — reads YAML files into AgentDefinition objects.

Each agent lives in its own directory:

    agents/
      myagent/
        agent.yaml              # agent definition
        execution.jsonl          # execution history
        *.running.json           # crash recovery files
        (future: script files, prompt files)

The agent.yaml schema maps directly to the AgentDefinition and StepDefinition
dataclasses:

    id: myagent
    name: My Agent
    description: Does useful things
    schedule: 5m
    concurrency: 1
    steps:
      - name: fetch
        type: script
        config:
          command: "curl https://example.com"
          timeout: 30
      - name: process
        type: script
        config:
          command: "python process.py"
          timeout: 60
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

from .models import AgentDefinition, AgentMode, HeartbeatConfig, StepDefinition, StepType

log = logging.getLogger(__name__)

_VALID_STEP_TYPES = {t.value for t in StepType}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class LoadError:
    """A single validation error from loading an agent file."""
    def __init__(self, file: Path, message: str) -> None:
        self.file = file
        self.message = message

    def __str__(self) -> str:
        return f"{self.file.name}: {self.message}"


def _validate_step(step_raw: dict, index: int, errors: list[LoadError], file: Path) -> StepDefinition | None:
    """Validate and convert a raw step dict.  Appends errors, returns None on failure."""
    step_type_str = step_raw.get("type")
    if not step_type_str:
        errors.append(LoadError(file, f"step[{index}]: missing 'type'"))
        return None
    if step_type_str not in _VALID_STEP_TYPES:
        errors.append(LoadError(file, f"step[{index}]: unknown type '{step_type_str}' (valid: {', '.join(sorted(_VALID_STEP_TYPES))})"))
        return None

    config = step_raw.get("config")
    if not isinstance(config, dict):
        errors.append(LoadError(file, f"step[{index}]: 'config' must be a mapping"))
        return None

    return StepDefinition(
        type=StepType(step_type_str),
        config=config,
        name=step_raw.get("name"),
    )


def _validate_agent(raw: dict, file: Path) -> tuple[AgentDefinition | None, list[LoadError]]:
    """Validate a raw YAML dict and return an AgentDefinition or errors."""
    errors: list[LoadError] = []

    # Required fields
    agent_id = raw.get("id")
    if not agent_id or not isinstance(agent_id, str):
        errors.append(LoadError(file, "missing or invalid 'id'"))

    name = raw.get("name")
    if not name or not isinstance(name, str):
        errors.append(LoadError(file, "missing or invalid 'name'"))

    # --- Agent mode (defaults to pipeline for backward compat) ---
    mode_str = raw.get("mode", "pipeline")
    try:
        mode = AgentMode(mode_str)
    except ValueError:
        errors.append(LoadError(file, f"invalid 'mode': '{mode_str}' (valid: pipeline, cognitive)"))
        return None, errors

    # --- Steps (required for pipeline, empty for cognitive) ---
    steps: list[StepDefinition] = []
    steps_raw = raw.get("steps")
    if mode == AgentMode.PIPELINE:
        if not isinstance(steps_raw, list) or len(steps_raw) == 0:
            errors.append(LoadError(file, "'steps' must be a non-empty list for pipeline agents"))
            return None, errors
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                errors.append(LoadError(file, f"step[{i}]: must be a mapping"))
                continue
            step = _validate_step(s, i, errors, file)
            if step:
                steps.append(step)
    elif steps_raw:
        # Cognitive agents may have steps too (ignored in cognitive execution but valid in YAML)
        for i, s in enumerate(steps_raw):
            if isinstance(s, dict):
                step = _validate_step(s, i, errors, file)
                if step:
                    steps.append(step)

    if errors:
        return None, errors

    # Optional fields
    concurrency = raw.get("concurrency", 1)
    if not isinstance(concurrency, int) or concurrency < 1:
        errors.append(LoadError(file, f"'concurrency' must be a positive integer, got {concurrency!r}"))
        return None, errors

    schedule = raw.get("schedule")
    if schedule is not None and not isinstance(schedule, str):
        errors.append(LoadError(file, f"'schedule' must be a string like '30s', '5m', '1h'"))
        return None, errors

    # Token budgets per provider
    budgets: dict[str, int] = {}
    budgets_raw = raw.get("budgets")
    if budgets_raw is not None:
        if not isinstance(budgets_raw, dict):
            errors.append(LoadError(file, "'budgets' must be a mapping of provider name to token limit"))
            return None, errors
        for k, v in budgets_raw.items():
            if not isinstance(v, int) or v < 0:
                errors.append(LoadError(file, f"budgets['{k}'] must be a non-negative integer"))
                return None, errors
            budgets[str(k)] = v

    # Connector IDs
    connector_ids: list[str] = []
    connector_ids_raw = raw.get("connector_ids")
    if connector_ids_raw is not None:
        if not isinstance(connector_ids_raw, list):
            errors.append(LoadError(file, "'connector_ids' must be a list of connector names"))
            return None, errors
        connector_ids = [str(c) for c in connector_ids_raw]

    # --- Hierarchy ---
    parent_id = raw.get("parent_id")
    if parent_id is not None and not isinstance(parent_id, str):
        errors.append(LoadError(file, "'parent_id' must be a string"))
        return None, errors

    created_by = str(raw.get("created_by", ""))

    # --- Parent notification ---
    notify_parent = bool(raw.get("notify_parent", True))
    wake_parent_on_child = bool(raw.get("wake_parent_on_child", False))

    # --- Heartbeat ---
    heartbeat: HeartbeatConfig | None = None
    heartbeat_raw = raw.get("heartbeat")
    if heartbeat_raw is not None:
        if isinstance(heartbeat_raw, dict):
            heartbeat = HeartbeatConfig(
                interval=str(heartbeat_raw.get("interval", "5m")),
                on_complete=str(heartbeat_raw.get("on_complete", "notify_parent")),
            )
        elif isinstance(heartbeat_raw, str):
            # Shorthand: heartbeat: 5m
            heartbeat = HeartbeatConfig(interval=heartbeat_raw)
        else:
            errors.append(LoadError(file, "'heartbeat' must be a mapping or interval string"))
            return None, errors

    # --- Cognitive mode fields ---
    provider = raw.get("provider", "")
    if isinstance(provider, list):
        provider = [str(p) for p in provider]
    elif provider:
        provider = str(provider)

    cognitive_model = str(raw.get("cognitive_model", raw.get("model", "")))
    system_prompt = str(raw.get("system_prompt", ""))
    task_prompt = str(raw.get("task_prompt", ""))
    agent_type = str(raw.get("agent_type", "general"))
    max_turns = raw.get("max_turns", 50)
    if not isinstance(max_turns, int) or max_turns < 1:
        max_turns = 50

    tools_raw = raw.get("tools")
    tools: list[str] = []
    if isinstance(tools_raw, list):
        tools = [str(t) for t in tools_raw]

    # --- Tool exposure ---
    expose_as_tool = bool(raw.get("expose_as_tool", False))
    tool_input_schema = raw.get("tool_input_schema")
    if tool_input_schema is not None and not isinstance(tool_input_schema, dict):
        errors.append(LoadError(file, "'tool_input_schema' must be a mapping (JSON Schema object)"))
        return None, errors

    # --- Secret allowance (authored wish; fail-closed None when absent) ---
    secrets_allowance = raw.get("secrets_allowance")
    if secrets_allowance is not None and not isinstance(secrets_allowance, str):
        errors.append(LoadError(file, "'secrets_allowance' must be a string spec (e.g. 'none', 'all', 'bundle:x,svc')"))
        return None, errors

    return AgentDefinition(
        id=agent_id,
        name=name,
        mode=mode,
        steps=steps,
        provider=provider,
        cognitive_model=cognitive_model,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        agent_type=agent_type,
        max_turns=max_turns,
        tools=tools,
        concurrency=concurrency,
        schedule=schedule,
        output_schema=raw.get("output_schema"),
        description=raw.get("description", ""),
        budgets=budgets,
        connector_ids=connector_ids,
        parent_id=parent_id,
        created_by=created_by,
        notify_parent=notify_parent,
        wake_parent_on_child=wake_parent_on_child,
        cloned_from=raw.get("cloned_from") or None,
        heartbeat=heartbeat,
        expose_as_tool=expose_as_tool,
        tool_input_schema=tool_input_schema,
        secrets_allowance=secrets_allowance,
    ), errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_agent_file(path: Path) -> tuple[AgentDefinition | None, list[LoadError]]:
    """Load a single agent definition from a YAML file or directory.

    Accepts either:
      - A directory containing ``agent.yaml``
      - A YAML file directly
    """
    if path.is_dir():
        path = path / "agent.yaml"
    if not path.exists():
        return None, [LoadError(path, "file not found")]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [LoadError(path, f"YAML parse error: {exc}")]

    if not isinstance(raw, dict):
        return None, [LoadError(path, "file must contain a YAML mapping")]

    return _validate_agent(raw, path)


def _externalize_step_files(step: StepDefinition, step_index: int, agent_dir: Path) -> dict[str, Any]:
    """Write inline step content to files and return a config dict with file references.

    - script steps: command -> <name>.sh
    - cognitive steps: system -> <name>.system.md, prompt -> <name>.prompt.md

    The agent directory must already exist.
    """
    config = dict(step.config)
    name = step.name or f"step_{step_index}"

    if step.type == StepType.SCRIPT:
        command = config.get("command", "")
        if command and not _is_file_reference(command):
            script_file = f"{name}.sh"
            (agent_dir / script_file).write_text(command + "\n", encoding="utf-8")
            config["command"] = f"bash {script_file}"
            log.info("Externalized script command to %s", script_file)

    elif step.type == StepType.COGNITIVE:
        for field, suffix in [("system", "system.md"), ("prompt", "prompt.md")]:
            value = config.get(field, "")
            if value and not _is_file_reference(value):
                filename = f"{name}.{suffix}"
                (agent_dir / filename).write_text(value, encoding="utf-8")
                config[field] = filename
                log.info("Externalized %s to %s", field, filename)

    return config


def _is_file_reference(value: str) -> bool:
    """Check if a config value is already a file reference."""
    stripped = value.strip()
    return any(stripped.endswith(ext) for ext in (".md", ".txt", ".prompt", ".sh", ".py", ".ps1", ".bat"))


def save_agent(defn: AgentDefinition, directory: Path) -> Path:
    """Serialize an AgentDefinition to ``directory/{id}/agent.yaml``.

    Creates the agent subdirectory if it doesn't exist.  Inline script
    commands and cognitive prompts are automatically written to files
    inside the agent directory.

    Returns the path of the written file.
    """
    agent_dir = directory / defn.id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "agent.yaml"

    data: dict[str, Any] = {
        "id": defn.id,
        "name": defn.name,
    }
    if defn.mode != AgentMode.PIPELINE:
        data["mode"] = defn.mode.value
    if defn.description:
        data["description"] = defn.description
    if defn.schedule:
        data["schedule"] = defn.schedule
    if defn.concurrency != 1:
        data["concurrency"] = defn.concurrency
    if defn.budgets:
        data["budgets"] = defn.budgets
    if defn.secrets_allowance is not None:
        data["secrets_allowance"] = defn.secrets_allowance
    if defn.output_schema:
        data["output_schema"] = defn.output_schema
    if defn.connector_ids:
        data["connector_ids"] = defn.connector_ids

    # Tool exposure
    if defn.expose_as_tool:
        data["expose_as_tool"] = True
    if defn.tool_input_schema:
        data["tool_input_schema"] = defn.tool_input_schema

    # Hierarchy
    if defn.parent_id:
        data["parent_id"] = defn.parent_id
    if defn.created_by:
        data["created_by"] = defn.created_by
    if not defn.notify_parent:
        data["notify_parent"] = False
    if defn.wake_parent_on_child:
        data["wake_parent_on_child"] = True
    if defn.cloned_from:
        data["cloned_from"] = defn.cloned_from

    # Heartbeat
    if defn.heartbeat:
        hb: dict[str, str] = {"interval": defn.heartbeat.interval}
        if defn.heartbeat.on_complete != "notify_parent":
            hb["on_complete"] = defn.heartbeat.on_complete
        data["heartbeat"] = hb

    # Cognitive mode fields
    if defn.is_cognitive:
        if defn.provider:
            data["provider"] = defn.provider
        if defn.cognitive_model:
            data["cognitive_model"] = defn.cognitive_model
        if defn.system_prompt:
            data["system_prompt"] = defn.system_prompt
        if defn.task_prompt:
            data["task_prompt"] = defn.task_prompt
        if defn.agent_type != "general":
            data["agent_type"] = defn.agent_type
        if defn.max_turns != 50:
            data["max_turns"] = defn.max_turns
        if defn.tools:
            data["tools"] = defn.tools

    # Steps (pipeline mode, or hybrid)
    if defn.steps:
        data["steps"] = []
        for i, s in enumerate(defn.steps):
            config = _externalize_step_files(s, i, agent_dir)
            data["steps"].append({
                "name": s.name,
                "type": s.type.value,
                "config": config,
            })

    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    log.info("Saved agent '%s' to %s", defn.id, path)
    return path


def delete_agent_dir(agent_id: str, directory: Path) -> bool:
    """Delete an agent's entire directory (definition, execution history, content).

    Returns True if the directory existed.
    """
    agent_dir = directory / agent_id
    if agent_dir.is_dir():
        shutil.rmtree(agent_dir)
        log.info("Deleted agent directory %s", agent_dir)
        return True
    # Backward compat: remove flat YAML file if it exists
    for suffix in (".yaml", ".yml"):
        path = directory / f"{agent_id}{suffix}"
        if path.exists():
            path.unlink()
            log.info("Deleted agent file %s", path)
            return True
    return False


def load_agents_dir(directory: Path) -> tuple[list[AgentDefinition], list[LoadError]]:
    """Load all agent definitions from a directory.

    Scans for subdirectories containing ``agent.yaml``.  Also loads any
    flat ``*.yaml`` files for backward compatibility (with a warning).

    Returns (agents, errors).  Agents that fail validation are skipped
    but their errors are collected so the caller can report them.
    """
    agents: list[AgentDefinition] = []
    all_errors: list[LoadError] = []

    if not directory.exists():
        log.info("Agents directory %s does not exist — no agents loaded", directory)
        return agents, all_errors

    seen_ids: dict[str, Path] = {}

    # Primary: subdirectories with agent.yaml
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        agent_yaml = child / "agent.yaml"
        if not agent_yaml.exists():
            continue
        defn, errors = load_agent_file(agent_yaml)
        all_errors.extend(errors)
        if defn is None:
            continue
        if defn.id in seen_ids:
            all_errors.append(LoadError(agent_yaml, f"duplicate agent id '{defn.id}' (first seen in {seen_ids[defn.id].name})"))
            continue
        seen_ids[defn.id] = agent_yaml
        agents.append(defn)

    # Backward compat: flat YAML files in the directory root
    flat_files = sorted(
        p for p in directory.iterdir()
        if p.suffix in (".yaml", ".yml") and p.is_file()
    )
    for f in flat_files:
        defn, errors = load_agent_file(f)
        all_errors.extend(errors)
        if defn is None:
            continue
        if defn.id in seen_ids:
            # Already loaded from a subdirectory — skip the flat file silently
            continue
        log.warning(
            "Agent '%s' loaded from flat file %s — consider migrating to %s/%s/agent.yaml",
            defn.id, f.name, directory, defn.id,
        )
        seen_ids[defn.id] = f
        agents.append(defn)

    if not agents and not all_errors:
        log.info("No agents found in %s", directory)

    return agents, all_errors
