"""Tests for the per-turn budget stamp injected into the newest incoming
message (atn/effective_limits.format_budget_line + humanize_magnitude).

Covers the three spec cases: budget set (with / without a prior metered turn)
and no budget (→ unlimited). The stamp is what an agent sees at every wake so it
can read its economic state at each decision point.
"""
from __future__ import annotations

from atn.effective_limits import (
    EffectiveLimits,
    format_budget_line,
    humanize_magnitude,
)


def _limits(*, limit, remaining, unit="tokens"):
    return EffectiveLimits(
        agent_id="a",
        source="agent_budget",
        provider="claude_max",
        entries=[{
            "key": "claude_max", "provider": "claude_max", "scope": "provider",
            "unit": unit, "limit": limit, "used": None, "remaining": remaining,
            "period": "monthly", "origin": "agent_budget",
        }],
    )


# --- humanize --------------------------------------------------------------

def test_humanize_magnitudes():
    assert humanize_magnitude(512) == "512"
    assert humanize_magnitude(999) == "999"
    assert humanize_magnitude(1000) == "1.0k"
    assert humanize_magnitude(38_200) == "38.2k"
    assert humanize_magnitude(2_100) == "2.1k"
    assert humanize_magnitude(1_200_000) == "1.2M"
    assert humanize_magnitude(0) == "0"
    assert humanize_magnitude(None) == "0"


# --- budget set, with a prior metered turn ---------------------------------

def test_budget_with_last_turn():
    line = format_budget_line(_limits(limit=40_000, remaining=38_200),
                              last_turn_tokens=2_100)
    assert line == "[Budget: 38.2k credits remaining | last turn: ~2.1k]"


# --- budget set, no prior metered turn (last-turn absent) -------------------

def test_budget_without_last_turn():
    line = format_budget_line(_limits(limit=40_000, remaining=512),
                              last_turn_tokens=None)
    assert line == "[Budget: 512 credits remaining]"
    # zero prior spend is treated the same as absent
    assert format_budget_line(_limits(limit=40_000, remaining=512),
                              last_turn_tokens=0) == line


# --- no budget configured → unlimited --------------------------------------

def test_no_budget_is_unlimited():
    # empty entries
    assert format_budget_line(EffectiveLimits(agent_id="a")) == "[Budget: unlimited]"
    # unlimited sentinel (-1 remaining) from an uncapped agent budget
    assert format_budget_line(_limits(limit=0, remaining=-1)) == "[Budget: unlimited]"
    # daemon ceiling with no known figure (remaining None)
    assert format_budget_line(
        _limits(limit=None, remaining=None, unit="usd")) == "[Budget: unlimited]"


def test_unlimited_drops_last_turn_segment():
    # even with a prior metered turn, an unlimited agent keeps the terse stamp
    line = format_budget_line(EffectiveLimits(agent_id="a"),
                              last_turn_tokens=2_100)
    assert line == "[Budget: unlimited]"


# --- dollar-cap (metered provider) rendering -------------------------------

def test_usd_remaining_rendering():
    line = format_budget_line(_limits(limit=50.0, remaining=4.21, unit="usd"),
                              last_turn_tokens=2_100)
    assert line == "[Budget: $4.21 remaining | last turn: ~2.1k]"


# --- engine _last_turn_tokens: exclude in-flight record, pick newest prior --

def _rec(eid, *, budget_tokens=0):
    from atn.models import ExecutionRecord, ExecutionStatus, TokenUsage
    r = ExecutionRecord(execution_id=eid, agent_id="a",
                        status=ExecutionStatus.COMPLETED, trigger_source="user")
    if budget_tokens:
        # budget_tokens() = input + cache_creation + output; put it all in input
        r.token_usage["claude_max"] = TokenUsage(
            provider="claude_max", input_tokens=budget_tokens)
    return r


def test_last_turn_tokens_selection():
    from atn.store import ExecutionLog
    from atn.runtime.execution_engine import ExecutionEngine

    log = ExecutionLog()
    log.record(_rec("old", budget_tokens=999))
    log.record(_rec("prev", budget_tokens=2_100))
    log.record(_rec("cur"))  # in-flight, empty usage — must be skipped

    class _Stub:
        execution_log = log

    got = ExecutionEngine._last_turn_tokens(_Stub(), "a", "cur")
    assert got == 2_100  # newest PRIOR metered turn, not the in-flight one


def test_last_turn_tokens_none_when_no_prior():
    from atn.store import ExecutionLog
    from atn.runtime.execution_engine import ExecutionEngine

    log = ExecutionLog()
    log.record(_rec("cur"))  # only the in-flight record exists

    class _Stub:
        execution_log = log

    assert ExecutionEngine._last_turn_tokens(_Stub(), "a", "cur") is None
