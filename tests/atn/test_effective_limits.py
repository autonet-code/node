"""Tests for the uniform budget pathway (atn/effective_limits.py).

A budgeted agent and an UNBUDGETED agent must surface their limits through the
SAME shape with no verbiage difference — the unbudgeted agent's effective
"budget" is the daemon-wide provider ceiling (dollar cap for metered providers,
inferred subscription remaining for subscription providers).
"""
from __future__ import annotations

import pytest

from atn.config import ATNConfig, ProviderConfig
from atn.effective_limits import compute_effective_limits
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.orchestrator.tools import _get_my_budget_status
from atn.runtime import Runtime


def _make_runtime(tmp_path, providers=None) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    for name, pc in (providers or {}).items():
        config.providers[name] = pc
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id, parent_id=None, *, budgets=None, provider="", model=""):
    return AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        parent_id=parent_id, budgets=budgets or {},
        provider=provider, cognitive_model=model,
    )


# ---------------------------------------------------------------------------
# Budgeted agent → source = agent_budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budgeted_agent_reports_own_caps(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", budgets={"claude_max": 40_000}))
    el = compute_effective_limits(
        "root", registry=rt.registry, metering=rt.metering,
        config=rt._config)
    assert el.source == "agent_budget"
    assert len(el.entries) == 1
    entry = el.entries[0]
    assert entry["limit"] == 40_000
    assert entry["origin"] == "agent_budget"
    assert entry["unit"] == "tokens"


# ---------------------------------------------------------------------------
# Unbudgeted agent on a subscription provider → daemon subscription remaining
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unbudgeted_subscription_agent_uses_inferred_quota(tmp_path):
    rt = _make_runtime(tmp_path)
    # No budget; provider resolves to claude_max (a subscription provider).
    await rt.registry.register_agent(
        _agent("root", provider="claude_max"))
    # Feed the metering estimator two snapshots so quota is inferrable.
    rt.metering.record_snapshot("claude_max", 0.10, 100_000)
    rt.metering.record_snapshot("claude_max", 0.40, 400_000)  # quota = 1e6

    el = compute_effective_limits(
        "root", registry=rt.registry, metering=rt.metering,
        config=rt._config)
    assert el.source == "daemon_provider"
    assert el.provider == "claude_max"
    entry = el.entries[0]
    assert entry["origin"] == "daemon_subscription"
    assert entry["unit"] == "tokens"          # SAME unit as a budgeted agent
    assert entry["limit"] == pytest.approx(1_000_000, rel=0.01)
    # At 40% utilization, remaining = 60% of quota.
    assert entry["remaining"] == pytest.approx(600_000, rel=0.02)
    assert entry["confidence"] > 0.0


# ---------------------------------------------------------------------------
# Unbudgeted agent on a metered provider → daemon dollar cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unbudgeted_metered_agent_uses_dollar_cap(tmp_path):
    rt = _make_runtime(tmp_path, providers={
        "anthropic": ProviderConfig(name="anthropic", dollar_limit=50.0),
    })
    await rt.registry.register_agent(
        _agent("root", provider="anthropic"))
    rt.metering.record_usage(
        "anthropic", "claude-opus-4-8", output_tokens=1_000_000)  # $25

    el = compute_effective_limits(
        "root", registry=rt.registry, metering=rt.metering,
        config=rt._config)
    assert el.source == "daemon_provider"
    assert el.provider == "anthropic"
    entry = el.entries[0]
    assert entry["origin"] == "daemon_dollar_cap"
    assert entry["unit"] == "usd"
    assert entry["limit"] == pytest.approx(50.0)
    assert entry["used"] == pytest.approx(25.0)
    assert entry["remaining"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Both cases flow through the SAME tool with the SAME shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_tool_carries_effective_limits_for_both(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("budgeted", budgets={"claude_max": 10_000}))
    await rt.registry.register_agent(
        _agent("unbudgeted", provider="claude_max"))
    rt.metering.record_snapshot("claude_max", 0.10, 100_000)
    rt.metering.record_snapshot("claude_max", 0.50, 500_000)

    r1 = await _get_my_budget_status(rt, {"_caller_id": "budgeted"})
    r2 = await _get_my_budget_status(rt, {"_caller_id": "unbudgeted"})

    # Same top-level key, same entry field names — no verbiage branch.
    assert "effective_limits" in r1 and "effective_limits" in r2
    assert r1["effective_limits"]["source"] == "agent_budget"
    assert r2["effective_limits"]["source"] == "daemon_provider"
    keys1 = set(r1["effective_limits"]["limits"][0].keys())
    keys2 = set(r2["effective_limits"]["limits"][0].keys())
    # Both share the uniform core fields.
    core = {"key", "provider", "scope", "unit", "limit", "used",
            "remaining", "period", "origin"}
    assert core <= keys1
    assert core <= keys2
