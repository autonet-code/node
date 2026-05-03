"""Tests for percent-based budget API (issue #21 / task #16).

Covers:
  • {pct, of: parent} freezes to absolute cap at registration
  • {pct, of: subscription_5h} stays as pct unit, recorder uses estimator
  • Cascade clamp using pct of parent
  • Mixed-unit chains: child pct of subscription under parent in tokens (no clamp)
"""
from __future__ import annotations

import pytest

from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.runtime.agent_registry import _resolve_budget


def _make_runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id: str, parent_id: str | None, *, mode=AgentMode.COGNITIVE,
           budgets=None) -> AgentDefinition:
    return AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=mode,
        parent_id=parent_id,
        budgets=budgets or {},
    )


# ---------------------------------------------------------------------------
# pct of parent — frozen at registration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pct_of_parent_frozen_to_absolute_tokens(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    child = _agent(
        "root.child", "root",
        budgets={"claude_max": {"pct": 30, "of": "parent"}},
    )
    await rt.registry.register_agent(child)

    # After registration, the child's budget is absolute tokens.
    assert "limit" in child.budgets["claude_max"]
    assert child.budgets["claude_max"]["limit"] == 3000
    limit, _period, unit = _resolve_budget(child, "claude_max")
    assert (limit, unit) == (3000, "tokens")


@pytest.mark.asyncio
async def test_pct_of_parent_rejects_when_no_ancestor_caps(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent("root", None))  # no budget
    with pytest.raises(ValueError, match="no ancestor declares a cap"):
        await rt.registry.register_agent(_agent(
            "root.child", "root",
            budgets={"claude_max": {"pct": 50, "of": "parent"}},
        ))


@pytest.mark.asyncio
async def test_pct_of_parent_rejects_invalid_pct(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    with pytest.raises(ValueError, match="invalid pct"):
        await rt.registry.register_agent(_agent(
            "root.child", "root",
            budgets={"claude_max": {"pct": 150, "of": "parent"}},
        ))


@pytest.mark.asyncio
async def test_pct_of_parent_under_pct_subscription_stays_pct(tmp_path):
    """If the ancestor is in pct_subscription, child's pct-of-parent is also pct."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None,
               budgets={"claude_max": {"pct": 80, "of": "subscription_5h"}}),
    )
    child = _agent(
        "root.child", "root",
        budgets={"claude_max": {"pct": 50, "of": "parent"}},
    )
    await rt.registry.register_agent(child)
    limit, _p, unit = _resolve_budget(child, "claude_max")
    # 50% of 80% = 40% of subscription
    assert unit == "pct_subscription_5h"
    assert limit == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# pct_subscription_5h unit — stays dynamic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subscription_pct_resolves_to_pct_unit(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent(
        "root", None,
        budgets={"claude_max": {"pct": 60, "of": "subscription_5h",
                                "period": "weekly"}},
    ))
    defn = rt.registry.get_agent("root")
    limit, period, unit = _resolve_budget(defn, "claude_max")
    assert (limit, period, unit) == (60.0, "weekly", "pct_subscription_5h")


# ---------------------------------------------------------------------------
# Cascade in pct of parent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pct_of_parent_respects_remaining_headroom(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(
        _agent("root", None, budgets={"claude_max": 10_000}),
    )
    # Burn most of root's headroom.
    rt.registry.record_token_usage("root", "claude_max", 8_000)
    # 30% of root's cap = 3000, but root has only 2000 left.
    with pytest.raises(ValueError, match="cascade violation"):
        await rt.registry.register_agent(_agent(
            "root.child", "root",
            budgets={"claude_max": {"pct": 30, "of": "parent"}},
        ))
    # 15% of root's cap = 1500, fits in remaining 2000.
    await rt.registry.register_agent(_agent(
        "root.child", "root",
        budgets={"claude_max": {"pct": 15, "of": "parent"}},
    ))


# ---------------------------------------------------------------------------
# Mixed-unit chains skip cascade clamp
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_unit_chain_skips_clamp(tmp_path):
    """Child capped in tokens, parent in subscription_5h → no clamp (incomparable).

    The runtime check still binds the subtree at execution time via the parent's cap.
    """
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent(
        "root", None,
        budgets={"claude_max": {"pct": 50, "of": "subscription_5h"}},
    ))
    # Child in tokens — no cascade comparison possible.
    await rt.registry.register_agent(_agent(
        "root.child", "root",
        budgets={"claude_max": 1_000_000_000},  # absurd, but registers fine
    ))


# ---------------------------------------------------------------------------
# Recorder converts tokens → pct using model class
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recorder_increments_pct_unit_with_estimator(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent(
        "root", None,
        budgets={"claude_max": {"pct": 80, "of": "subscription_5h",
                                "period": "weekly"}},
    ))
    # tokens_per_pct: sonnet uses 5000 tokens per 1pp.
    rt.registry.record_token_usage(
        "root", "claude_max", 10_000,
        model_class="sonnet",
        tokens_per_pct={"sonnet": 5000.0},
    )
    info = rt.registry.get_budget_info("root")
    # 10000 tokens / 5000 per pct = 2.0 pct
    assert info["claude_max"]["unit"] == "pct_subscription_5h"
    assert info["claude_max"]["used"] == pytest.approx(2.0)
    assert info["claude_max"]["limit"] == 80.0


@pytest.mark.asyncio
async def test_recorder_skips_pct_when_estimator_missing(tmp_path):
    """Without a tokens_per_pct rate, pct-unit caps are observed but not enforced yet."""
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent(
        "root", None,
        budgets={"claude_max": {"pct": 80, "of": "subscription_5h"}},
    ))
    rt.registry.record_token_usage(
        "root", "claude_max", 10_000,
        model_class="sonnet",
        tokens_per_pct=None,
    )
    info = rt.registry.get_budget_info("root")
    assert info["claude_max"]["used"] == 0  # no enforcement increment


@pytest.mark.asyncio
async def test_recorder_increments_tokens_unit_normally(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.registry.register_agent(_agent(
        "root", None, budgets={"claude_max": 100_000},
    ))
    rt.registry.record_token_usage(
        "root", "claude_max", 5000,
        model_class="opus",
        tokens_per_pct={"opus": 1000.0},
    )
    info = rt.registry.get_budget_info("root")
    # tokens unit ignores model-class conversion.
    assert info["claude_max"]["used"] == 5000
