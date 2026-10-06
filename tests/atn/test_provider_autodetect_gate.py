"""ATN_DISABLE_PROVIDER_AUTODETECT=1 keeps a test daemon on its configured
providers only.

The live E2E harness (scripts/e2e_live) boots a real daemon in a redirected
home with a deterministic stub provider. Without the gate,
``auto_detect_providers`` still probed the machine's Claude Max bridge (bun on
PATH) and registered ``claude_max``, putting the developer's subscription
quota one fallback away from any test agent turn.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from atn.config import ATNConfig
from atn.models import StepType
from atn.runtime.provider_manager import ProviderManager
from atn.steps.cognitive import CognitiveStepExecutor


def _mgr() -> tuple[ProviderManager, CognitiveStepExecutor]:
    creds = MagicMock()
    creds.load.return_value = {}
    cognitive = CognitiveStepExecutor()
    mgr = ProviderManager(config=ATNConfig(), credential_store=creds,
                          executors={StepType.COGNITIVE: cognitive},
                          events=MagicMock())
    mgr.probe_bridge = AsyncMock(return_value=True)          # type: ignore[method-assign]
    mgr.probe_codex_bridge = AsyncMock(return_value=True)    # type: ignore[method-assign]
    mgr._hot_register_provider = MagicMock()                 # type: ignore[method-assign]
    return mgr, cognitive


def test_gate_skips_every_probe(monkeypatch):
    monkeypatch.setenv("ATN_DISABLE_PROVIDER_AUTODETECT", "1")
    mgr, _ = _mgr()
    asyncio.run(mgr.auto_detect_providers())
    mgr.probe_bridge.assert_not_called()
    mgr.probe_codex_bridge.assert_not_called()
    mgr._hot_register_provider.assert_not_called()


def test_default_still_probes_the_bridge(monkeypatch):
    monkeypatch.delenv("ATN_DISABLE_PROVIDER_AUTODETECT", raising=False)
    mgr, _ = _mgr()
    asyncio.run(mgr.auto_detect_providers())
    mgr.probe_bridge.assert_awaited_once()
    mgr._hot_register_provider.assert_any_call("claude_max", "")
