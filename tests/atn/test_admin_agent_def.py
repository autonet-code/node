"""The shipped Admin Agent definition (agents/admin/agent.yaml) must load,
validate, and wire its two cadences mechanically.

OPS cadence  = the runtime `schedule` (a periodic timer).
SCOUT cadence = self-tracked (no second timer; the prompt drives it), so the
definition must NOT also declare a heartbeat (schedule + heartbeat are mutually
exclusive in the runtime, and only one can be a timer).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atn.loader import load_agent_file
from atn.models import AgentMode
from atn.runtime.agent_registry import parse_interval
from atn.agent_tools import resolve_tool_surface, _EXECUTORS


ADMIN_YAML = Path(__file__).resolve().parents[2] / "agents" / "admin" / "agent.yaml"


def test_admin_agent_loads_without_errors():
    assert ADMIN_YAML.exists(), f"missing shipped admin agent at {ADMIN_YAML}"
    defn, errors = load_agent_file(ADMIN_YAML)
    assert not errors, f"admin agent failed to validate: {[str(e) for e in errors]}"
    assert defn is not None
    assert defn.id == "admin"
    assert defn.mode == AgentMode.COGNITIVE


def test_admin_ops_cadence_is_schedule_scout_is_self_tracked():
    defn, _ = load_agent_file(ADMIN_YAML)
    # OPS cadence: a parseable schedule interval (the periodic timer).
    assert defn.schedule is not None
    assert parse_interval(defn.schedule) > 0
    # SCOUT cadence is self-tracked, NOT a second runtime timer: no heartbeat,
    # so the runtime registers exactly one cadence (the schedule).
    assert defn.heartbeat is None


def test_admin_budget_parses_as_placeholder_daily_cap():
    defn, _ = load_agent_file(ADMIN_YAML)
    assert "claude_max" in defn.budgets
    cap = defn.budgets["claude_max"]
    assert cap["limit"] == 100_000
    assert cap["period"] == "daily"


def test_admin_tool_surface_grants_metering_report():
    defn, _ = load_agent_file(ADMIN_YAML)
    surface = resolve_tool_surface(defn.tools)
    names = {t["name"] for t in surface}
    # The ops pass needs the daemon-wide metering view and a way to post findings.
    assert "metering_report" in names
    assert "post_message" in names
    # metering_report must be an actually-dispatchable executor.
    assert "metering_report" in _EXECUTORS


def test_admin_defines_two_cadences_in_prose():
    """The one-agent-two-cadences doctrine must be embodied in the prompt."""
    defn, _ = load_agent_file(ADMIN_YAML)
    p = defn.system_prompt.lower()
    assert "ops" in p and "scout" in p
    assert "propose" in p  # propose-never-apply doctrine
