"""Model selection is deliberate (decision 2026-10-06).

No daemon-wide default provider/model, no inheritance from a parent agent and
no hardcoded fallback model: every cognitive agent names its provider and
model at create time, can change but never clear them, and an agent that
names neither fails loud instead of being routed. Kevin is seeded on the first
configured provider with an explicit model, and agents written before the
decision are pinned once to what they already ran on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from atn.agent_tools import _create_agent, _update_agent
from atn.config import ATNConfig, ProviderConfig, load_config
from atn.events import EventBus
from atn.fleet_seed import KEVIN_ID, seed_default_fleet, seed_route
from atn.legacy_routing_migration import (
    STAMP_NAME,
    _drop_legacy_defaults_section,
    legacy_resolved_route,
    pin_unpinned_agents_from_legacy_defaults,
)
from atn.loader import load_agents_dir, save_agent
from atn.models import AgentDefinition, AgentMode
from atn.provider_identity import routing_error
from atn.providers.base import ProviderError
from atn.runtime import Runtime

STUB = "e2e_stub"
BINDING = {"provider_address": "0x" + "1" * 40, "spec_digest": "a" * 64}


def _runtime(tmp_path, *, stub=True) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    if stub:
        config.providers[STUB] = ProviderConfig(
            name=STUB, base_url="http://127.0.0.1:9/v1", api_key="x",
            models=["echo-1", "echo-2"])
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _agent(agent_id, *, provider="", model="", **kw):
    return AgentDefinition(id=agent_id, name=agent_id,
                           mode=AgentMode.COGNITIVE, provider=provider,
                           cognitive_model=model, **kw)


# ---------------------------------------------------------------------------
# The validation rule
# ---------------------------------------------------------------------------

def test_routing_error_requires_both():
    assert "provider is required" in routing_error("", "gpt-5")
    assert "provider is required" in routing_error([], "gpt-5")
    assert "model is required" in routing_error("openai", "")
    assert "unknown provider" in routing_error("gpt-5", "gpt-5")
    assert routing_error("openai", "gpt-5") is None
    assert routing_error("my-llm", "m", custom_providers={"my-llm"}) is None
    assert routing_error(["anthropic", "claude_max"], "sonnet") is None


def test_purchased_service_names_its_own_model():
    assert routing_error("", "", service_bound=True) is None
    assert routing_error("service", "") is None


@pytest.mark.asyncio
async def test_no_daemon_level_choice_on_the_wire(tmp_path):
    from atn.agent_tools import _get_agent
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "a", "name": "a",
                             "provider": STUB, "model": "echo-1"})
    retired = "default_" + "provider"   # the retired wire key
    assert retired not in rt.snapshot()
    assert retired not in await _get_agent(rt, {"agent_id": "a"})


def test_unpinned_agent_fails_loud_at_resolve(tmp_path):
    rt = _runtime(tmp_path)
    with pytest.raises(ProviderError, match="provider is required"):
        rt.providers.resolve_provider_with_fallback(_agent("a", model="gpt-5"))
    with pytest.raises(ProviderError, match="model is required"):
        rt.providers.resolve_provider_with_fallback(_agent("a", provider=STUB))


# ---------------------------------------------------------------------------
# create_agent (tool and WS path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_requires_provider_and_model(tmp_path):
    rt = _runtime(tmp_path)
    r = await _create_agent(rt, {"id": "x", "name": "x", "model": "echo-1"})
    assert "provider is required" in r["error"]
    r = await _create_agent(rt, {"id": "x", "name": "x", "provider": STUB})
    assert "model is required" in r["error"]
    assert rt.get_agent("x") is None

    r = await _create_agent(rt, {"id": "x", "name": "x",
                                 "provider": STUB, "model": "echo-1"})
    assert r.get("status") == "registered", r
    defn = rt.get_agent("x")
    assert (defn.provider, defn.cognitive_model) == (STUB, "echo-1")


@pytest.mark.asyncio
async def test_child_never_inherits_its_parents_choice(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "boss", "name": "boss",
                             "provider": STUB, "model": "echo-1"})
    r = await _create_agent(rt, {"id": "kid", "name": "kid",
                                 "_caller_id": "boss"})
    assert "provider is required" in r["error"]
    assert "never inherits" in r["error"]
    r = await _create_agent(rt, {"id": "kid", "name": "kid",
                                 "_caller_id": "boss", "provider": STUB,
                                 "model": "echo-2"})
    assert r.get("status") == "registered", r
    assert rt.get_agent("kid").cognitive_model == "echo-2"


@pytest.mark.asyncio
async def test_create_bound_to_a_purchase_needs_no_provider(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "boss", "name": "boss",
                             "provider": STUB, "model": "echo-1"})
    r = await _create_agent(rt, {"id": "kid", "name": "kid",
                                 "_caller_id": "boss",
                                 "service_provider": dict(BINDING)})
    assert r.get("status") == "registered", r


def test_tool_schema_tells_agents_to_choose():
    from atn.agent_tools import _TOOLS
    tool = next(t for t in _TOOLS if t.name == "create_agent")
    props = tool.input_schema["properties"]
    assert "REQUIRED" in props["provider"]["description"]
    assert "REQUIRED" in props["model"]["description"]
    assert "never inherited" in tool.description


# ---------------------------------------------------------------------------
# update_agent / set_agent_model: change, never clear
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_cannot_clear_provider_or_model(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "a", "name": "a",
                             "provider": STUB, "model": "echo-1"})
    for change in ({"provider": ""}, {"model": ""},
                   {"provider": "", "model": ""}):
        r = await _update_agent(rt, {"agent_id": "a", **change})
        assert "not cleared" in r["error"], (change, r)
    defn = rt.get_agent("a")
    assert (defn.provider, defn.cognitive_model) == (STUB, "echo-1")


@pytest.mark.asyncio
async def test_update_model_keeps_provider(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "a", "name": "a",
                             "provider": STUB, "model": "echo-1"})
    r = await _update_agent(rt, {"agent_id": "a", "model": "echo-2"})
    assert r.get("status") == "updated", r
    defn = rt.get_agent("a")
    # Setting a model no longer rewrites the provider into a model hint.
    assert (defn.provider, defn.cognitive_model) == (STUB, "echo-2")


@pytest.mark.asyncio
async def test_unbinding_a_purchase_needs_a_provider(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "a", "name": "a",
                             "service_provider": dict(BINDING)})
    r = await _update_agent(rt, {"agent_id": "a", "service_provider": None})
    assert "provider is required" in r["error"]
    r = await _update_agent(rt, {"agent_id": "a", "service_provider": None,
                                 "provider": STUB, "model": "echo-1"})
    assert r.get("status") == "updated", r


@pytest.mark.asyncio
async def test_set_agent_model_rejects_empty_and_unpinned(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "a", "name": "a",
                             "provider": STUB, "model": "echo-1"})
    with pytest.raises(ValueError, match="model is required"):
        await rt.set_agent_model("a", "")
    await rt.set_agent_model("a", "echo-2")
    assert rt.get_agent("a").provider == STUB
    await rt.register_agent(_agent("loose", model="echo-1"), legacy=True)
    with pytest.raises(ValueError, match="provider is required"):
        await rt.set_agent_model("loose", "echo-2")


# ---------------------------------------------------------------------------
# Kevin seeding
# ---------------------------------------------------------------------------

def _fake_rt(order, models):
    providers = SimpleNamespace(
        registered_provider_ids=lambda: list(order),
        get_available_models=lambda pid: models.get(pid, []))
    return SimpleNamespace(providers=providers)


def test_seed_route_rule():
    models = {
        "service": [],
        "claude_max": [
            {"id": "claude-sonnet-5", "capability_tier": 3},
            {"id": "claude-opus-4-8", "capability_tier": 4},
            {"id": "claude-fable-5", "capability_tier": 4},
        ],
        "ollama": [{"id": "qwen3:4b", "capability_tier": 2}],
    }
    # First provider WITH models; highest tier; ties by list order.
    assert seed_route(_fake_rt(["service", "claude_max", "ollama"], models)) \
        == ("claude_max", "claude-opus-4-8")
    assert seed_route(_fake_rt(["ollama", "claude_max"], models)) \
        == ("ollama", "qwen3:4b")
    assert seed_route(_fake_rt([], models)) is None


@pytest.mark.asyncio
async def test_seeding_defers_until_a_provider_is_configured(tmp_path):
    rt = _runtime(tmp_path, stub=False)
    cfg = rt._config
    assert await seed_default_fleet(rt, cfg) is None
    assert not (cfg.data_dir / ".fleet_seeded").exists()
    assert rt.get_agent(KEVIN_ID) is None

    # The user configures a provider: the deferred seed runs on it. (The
    # real add_custom_provider probes the endpoint; register it directly.)
    async def _add(provider_id, name, base_url, api_key="", label="",
                   models=None):
        from atn.models import StepType
        from atn.providers.openai_compat import OpenAICompatibleProvider
        cfg.providers[provider_id] = ProviderConfig(
            name=provider_id, base_url=base_url, api_key=api_key,
            models=list(models or []))
        rt.providers._custom_providers.add(provider_id)
        rt.providers._executors[StepType.COGNITIVE].register_provider(
            OpenAICompatibleProvider(name=provider_id, base_url=base_url,
                                     api_key=api_key))
        return {"status": "ok"}

    rt.providers.add_custom_provider = _add
    await rt.add_custom_provider(STUB, "Stub", "http://127.0.0.1:9/v1",
                                 api_key="x", models=["echo-1"])
    kevin = rt.get_agent(KEVIN_ID)
    assert kevin is not None
    assert (kevin.provider, kevin.cognitive_model) == (STUB, "echo-1")
    assert (cfg.data_dir / ".fleet_seeded").read_text().startswith("seeded")


@pytest.mark.asyncio
async def test_seeding_at_boot_uses_the_configured_provider(tmp_path):
    rt = _runtime(tmp_path)
    assert await seed_default_fleet(rt, rt._config) == KEVIN_ID
    kevin = rt.get_agent(KEVIN_ID)
    assert (kevin.provider, kevin.cognitive_model) == (STUB, "echo-1")


# ---------------------------------------------------------------------------
# One-time migration of pre-decision agents
# ---------------------------------------------------------------------------

def test_legacy_route_matches_the_old_rules():
    no_keys = lambda _pid: False  # noqa: E731
    anth = lambda pid: pid == "anthropic"  # noqa: E731
    # Unpinned, no model, no defaults: the old hardcoded fallback.
    assert legacy_resolved_route(_agent("a"), has_api_key=no_keys) \
        == ("claude_max", "claude-sonnet-4-6")
    assert legacy_resolved_route(_agent("a"), has_api_key=anth) \
        == ("anthropic", "claude-sonnet-4-6")
    # The defaults: section's model fills a missing model.
    assert legacy_resolved_route(_agent("a"), legacy_model="gpt-5") \
        == ("openai", "gpt-5")
    # Model-shaped hint routes by cognitive_model, else the hint.
    assert legacy_resolved_route(_agent("a", provider="sonnet", model="gpt-5")) \
        == ("openai", "gpt-5")
    assert legacy_resolved_route(_agent("a", provider="qwen3:4b")) \
        == ("ollama", "qwen3:4b")
    # Pinned provider, no model: the provider keeps the default model.
    assert legacy_resolved_route(_agent("a", provider="ollama"),
                                 legacy_model="llama3") == ("ollama", "llama3")


def test_drop_defaults_section_keeps_the_rest(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# my config\n"
        "data_dir: ~/.atn\n"
        "defaults:\n"
        "  provider: openai   # old\n"
        "\n"
        "  model: gpt-5\n"
        "voice:\n"
        "  enabled: false\n", encoding="utf-8")
    assert _drop_legacy_defaults_section(path)
    text = path.read_text(encoding="utf-8")
    assert "defaults" not in text and "gpt-5" not in text
    assert text.startswith("# my config\ndata_dir: ~/.atn\nvoice:\n")
    assert not _drop_legacy_defaults_section(path)


def _write_config(tmp_path, body: str):
    home = tmp_path / "home"
    (home / "agents").mkdir(parents=True)
    path = home / "config.yaml"
    path.write_text(
        f"data_dir: {home.as_posix()}\n"
        f"agents_dir: {(home / 'agents').as_posix()}\n" + body,
        encoding="utf-8")
    return path


@pytest.mark.parametrize("section", ["defaults", "orchestrator"])
def test_migration_pins_agents_then_drops_the_section(tmp_path, section):
    # The orchestrator migration rewrites orchestrator: -> defaults:, which
    # this migration then consumes and deletes: the two compose.
    path = _write_config(tmp_path, f"{section}:\n  provider: claude_max\n"
                                   "  model: gpt-5\nvoice:\n  enabled: false\n")
    cfg = load_config(path)
    assert cfg.legacy_agent_routing == {"provider": "claude_max",
                                        "model": "gpt-5"}
    save_agent(_agent("loose"), cfg.agents_dir)
    save_agent(_agent("hinted", provider="qwen3:4b", model="qwen3:4b"),
               cfg.agents_dir)
    save_agent(_agent("pinned", provider="ollama", model="llama3"),
               cfg.agents_dir)
    from atn.models import StepDefinition, StepType
    save_agent(AgentDefinition(
        id="pipe", name="pipe", steps=[StepDefinition(
            type=StepType.SCRIPT, config={"command": "echo hi"})]),
        cfg.agents_dir)
    agents, errors = load_agents_dir(cfg.agents_dir)
    assert not errors

    pinned = pin_unpinned_agents_from_legacy_defaults(agents, cfg)
    assert sorted(pinned) == ["hinted", "loose"]

    on_disk = {a.id: a for a in load_agents_dir(cfg.agents_dir)[0]}
    assert (on_disk["loose"].provider, on_disk["loose"].cognitive_model) \
        == ("openai", "gpt-5")
    assert (on_disk["hinted"].provider, on_disk["hinted"].cognitive_model) \
        == ("ollama", "qwen3:4b")
    assert (on_disk["pinned"].provider, on_disk["pinned"].cognitive_model) \
        == ("ollama", "llama3")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "defaults" not in raw and "orchestrator" not in raw
    assert raw["voice"] == {"enabled": False}
    assert (cfg.data_dir / STAMP_NAME).exists()
    assert load_config(path).legacy_agent_routing == {}

    # One-time: a later unpinned agent is NOT silently pinned.
    late = [_agent("late")]
    assert pin_unpinned_agents_from_legacy_defaults(late, cfg) == []
    assert late[0].provider == ""


def test_migration_skips_bound_agents(tmp_path):
    path = _write_config(tmp_path, "")
    cfg = load_config(path)
    bound = _agent("bound", service_provider=dict(BINDING))
    assert pin_unpinned_agents_from_legacy_defaults([bound], cfg) == []
    assert bound.provider == ""
