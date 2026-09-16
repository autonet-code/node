"""Custom providers must survive a daemon restart.

``add_custom_provider`` requires a base_url and persists it to config.yaml, so
on the next boot every custom entry came back with a truthy ``base_url``. The
generic ``or pconfig.base_url`` arm in ``setup_providers`` swallowed the name
before the custom branch could run, which meant:

  - a keyless local endpoint (LM Studio / vLLM / a local gateway) hit the
    API-key gate and was dropped outright;
  - ``_custom_providers`` stayed empty, so the provider had no card, no delete
    button, and any agent pinned to it raised "Unknown provider".

The discriminator is now built-in membership, not whether a base_url happens to
be set. Built-in slots must keep registering from ``_PROVIDER_DEFAULTS`` and
must NOT be labelled custom; deepseek is the case that had no by-name arm at
all, so it registered nothing.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from atn.config import ATNConfig, ProviderConfig
from atn.models import StepType
from atn.runtime.provider_manager import ProviderManager
from atn.steps.cognitive import CognitiveStepExecutor


def _mgr(providers: dict[str, ProviderConfig]) -> ProviderManager:
    config = ATNConfig()
    config.providers.update(providers)
    creds = MagicMock()
    creds.load.return_value = {}
    return ProviderManager(config=config, credential_store=creds,
                           executors={}, events=MagicMock())


def _setup(mgr: ProviderManager) -> CognitiveStepExecutor:
    cognitive = CognitiveStepExecutor()
    mgr._executors[StepType.COGNITIVE] = cognitive
    mgr.setup_providers(cognitive)
    return cognitive


def test_keyless_custom_provider_survives_reload():
    mgr = _mgr({"my-llm": ProviderConfig(
        name="my-llm",
        base_url="http://localhost:1234/v1",
        default_model="qwen",
        extra={"type": "openai_compat"},
    )})
    cognitive = _setup(mgr)

    assert "my-llm" in mgr._custom_providers
    assert "my-llm" in cognitive._providers


def test_builtin_with_base_url_is_not_labelled_custom():
    mgr = _mgr({"deepseek": ProviderConfig(
        name="deepseek", default_model="deepseek-chat", api_key="k")})
    cognitive = _setup(mgr)

    assert "deepseek" not in mgr._custom_providers
    assert "deepseek" in cognitive._providers
