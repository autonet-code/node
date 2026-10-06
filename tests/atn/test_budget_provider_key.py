"""Budgets are keyed to the provider an agent actually runs on.

Older clients keyed budgets under ``claude_max`` whatever provider the agent
ran on, so the cap never bound. The budget key is the agent's own provider
(atn/provider_identity.py), surfaced to clients, and legacy ``claude_max``
keys on agents that do not run on claude_max are moved on load.
"""
from __future__ import annotations

import pytest

from atn.agent_tools import _get_agent, _update_agent
from atn.config import ATNConfig, ProviderConfig
from atn.events import EventBus
from atn.loader import load_agents_dir, save_agent
from atn.models import AgentDefinition, AgentMode
from atn.provider_identity import (
    effective_provider_id,
    migrate_legacy_budget_keys,
)
from atn.legacy_routing_migration import (
    pin_unpinned_agents_from_legacy_defaults,
)
from atn.providers.openai_compat import OpenAICompatibleProvider
from atn.runtime import Runtime
from atn.runtime.execution_engine import _provider_name


def _agent(agent_id="a", *, provider="", model="", budgets=None,
           service_provider=None):
    return AgentDefinition(
        id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
        provider=provider, cognitive_model=model, budgets=budgets or {},
        service_provider=service_provider,
    )


def _keys(*pids):
    return lambda pid: pid in pids


# ---------------------------------------------------------------------------
# Pure resolver mirrors the engine's routing
# ---------------------------------------------------------------------------

def test_agent_without_a_provider_has_no_effective_provider():
    # No routing by model prefix: an agent that names no provider is a
    # routing error, not a guess.
    assert effective_provider_id(_agent(model="claude-sonnet-4-6")) == ""
    assert effective_provider_id(_agent(model="gpt-5")) == ""
    assert effective_provider_id(_agent(provider="gpt-5", model="gpt-5")) == ""


def test_explicit_pin_and_custom_provider_are_used_verbatim():
    assert effective_provider_id(_agent(provider="ollama", model="sonnet")) == "ollama"
    assert effective_provider_id(
        _agent(provider="my-llm"), custom_providers={"my-llm"}) == "my-llm"


def test_fallback_chain_takes_first_buildable():
    defn = _agent(provider=["anthropic", "claude_max"], model="sonnet")
    assert effective_provider_id(defn) == "claude_max"
    assert effective_provider_id(
        defn, has_api_key=_keys("anthropic")) == "anthropic"


def test_empty_chain_names_no_provider():
    assert effective_provider_id(_agent(provider=[], model="gpt-5")) == ""


def test_service_binding_overrides_everything():
    defn = _agent(provider="anthropic", service_provider={
        "provider_address": "0x" + "1" * 40, "spec_digest": "a" * 64})
    assert effective_provider_id(defn) == "service"


# ---------------------------------------------------------------------------
# Legacy key migration
# ---------------------------------------------------------------------------

def test_migration_moves_provider_and_model_keys():
    defn = _agent(budgets={
        "claude_max": {"limit": 5000, "period": "weekly"},
        "claude_max:qwen3:4b": 100,
    })
    moved = migrate_legacy_budget_keys(defn, "ollama")
    assert moved == {"claude_max": "ollama",
                     "claude_max:qwen3:4b": "ollama:qwen3:4b"}
    assert defn.budgets == {
        "ollama": {"limit": 5000, "period": "weekly"},
        "ollama:qwen3:4b": 100,
    }


def test_migration_noop_when_agent_runs_on_claude_max():
    defn = _agent(budgets={"claude_max": 5000})
    assert migrate_legacy_budget_keys(defn, "claude_max") == {}
    assert defn.budgets == {"claude_max": 5000}


def test_migration_never_overwrites_an_existing_limit():
    defn = _agent(budgets={"claude_max": 5000, "ollama": 700})
    assert migrate_legacy_budget_keys(defn, "ollama") == {}
    assert defn.budgets == {"claude_max": 5000, "ollama": 700}


def test_migration_keeps_claude_max_named_in_the_chain():
    defn = _agent(provider=["anthropic", "claude_max"],
                  budgets={"claude_max": 5000})
    assert migrate_legacy_budget_keys(defn, "anthropic") == {}


# ---------------------------------------------------------------------------
# Runtime: the engine's budget key and the load-time migration
# ---------------------------------------------------------------------------

def test_per_agent_openai_instance_books_under_provider_id():
    prov = OpenAICompatibleProvider(
        name="openai-agent7", provider_id="openai",
        base_url="https://api.openai.com/v1", api_key="k")
    assert prov.name == "openai-agent7"
    assert _provider_name(prov) == "openai"


def test_nameless_provider_does_not_book_under_another_provider():
    class _Nameless:
        name = None
    assert _provider_name(_Nameless()) == "unknown"


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


@pytest.mark.asyncio
async def test_load_moves_legacy_key_to_routed_provider(tmp_path):
    rt = _make_runtime(tmp_path, providers={
        "openai": ProviderConfig(name="openai", api_key="sk-test")})
    legacy = _agent("worker", provider="gpt-5", model="gpt-5", budgets={
        "claude_max": {"limit": 9000, "period": "daily"}})
    save_agent(legacy, rt._config.agents_dir)

    defs, errors = load_agents_dir(rt._config.agents_dir)
    assert not errors
    # Boot order: the one-time pinning migration turns the model-shaped
    # provider hint into the provider it routed to, then registration moves
    # the legacy budget key onto it.
    assert pin_unpinned_agents_from_legacy_defaults(
        defs, rt._config, rt.providers) == ["worker"]
    rt.registry._budget_used["worker"] = {"claude_max": 1234}
    await rt.register_agent(defs[0], legacy=True)

    defn = rt.get_agent("worker")
    assert defn.budgets == {"openai": {"limit": 9000, "period": "daily"}}
    # Spend already booked follows the key; the limit binds the real provider.
    assert rt.registry._budget_used["worker"] == {"openai": 1234}
    ok, _ = rt.registry.check_budget("worker", "openai")
    assert ok
    rt.registry.record_token_usage("worker", "openai", 9000)
    ok, blocker = rt.registry.check_budget("worker", "openai")
    assert (ok, blocker) == (False, "worker")
    # Persisted, so the next boot reads the real key.
    reloaded, _ = load_agents_dir(rt._config.agents_dir)
    assert reloaded[0].budgets == {"openai": {"limit": 9000, "period": "daily"}}


@pytest.mark.asyncio
async def test_load_keeps_key_when_agent_runs_on_claude_max(tmp_path):
    rt = _make_runtime(tmp_path)  # no anthropic key: claude routes to the bridge
    await rt.register_agent(
        _agent("kevin", provider="claude_max", model="sonnet",
               budgets={"claude_max": 40_000}), legacy=True)
    assert rt.get_agent("kevin").budgets == {"claude_max": 40_000}


@pytest.mark.asyncio
async def test_get_agent_reports_effective_provider(tmp_path):
    rt = _make_runtime(tmp_path, providers={
        "anthropic": ProviderConfig(name="anthropic", api_key="sk-ant-test")})
    await rt.register_agent(
        _agent("pinned", provider="ollama", model="sonnet"), legacy=True)
    res = await _get_agent(rt, {"agent_id": "pinned"})
    assert res["effective_provider"] == "ollama"


@pytest.mark.asyncio
async def test_snapshot_reports_the_agents_own_provider(tmp_path):
    rt = _make_runtime(tmp_path, providers={
        "openai": ProviderConfig(name="openai", api_key="sk-test")})
    await rt.register_agent(
        _agent("pinned", provider="openai", model="gpt-5"), legacy=True)
    snap = rt.snapshot()
    assert snap["agents"]["pinned"]["effective_provider"] == "openai"
    assert snap["agents"]["pinned"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_update_agent_rekeys_legacy_budget_from_old_client(tmp_path):
    rt = _make_runtime(tmp_path)
    await rt.register_agent(
        _agent("local", provider="ollama", model="qwen3:4b"), legacy=True)
    res = await _update_agent(rt, {
        "agent_id": "local", "budgets": {"claude_max": 2500}})
    assert res.get("status") == "updated", res
    assert rt.get_agent("local").budgets == {"ollama": 2500}
