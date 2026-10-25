"""A chat message to a running agent is delivered, or queued, never lost
behind a false "injected" (2026-10-26 survey, runtime findings 2, 3, 11)."""
from __future__ import annotations

import pytest

from atn.agent_tools import _create_agent
from atn.config import ATNConfig, ProviderConfig
from atn.events import EventBus
from atn.runtime import Runtime

STUB = "e2e_stub"


def _runtime(tmp_path) -> Runtime:
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir(parents=True)
    agents_dir.mkdir(parents=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.worker_isolation.enabled = False
    config.providers[STUB] = ProviderConfig(
        name=STUB, base_url="http://127.0.0.1:9/v1", api_key="x", models=["echo-1"])
    return Runtime(EventBus(), data_dir=data_dir, config=config)


class Provider:
    """A provider whose loop either takes a steering message (True) or is
    idle (False), the two answers a generic-loop provider gives."""

    def __init__(self, takes: bool):
        self.takes, self.sent = takes, []

    async def send_user_message(self, text):
        self.sent.append(text)
        return self.takes


async def _running_agent(tmp_path, takes):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "a", "name": "a", "provider": STUB, "model": "echo-1"})
    prov = Provider(takes)
    rt.providers._active_providers["a"] = prov
    rt.registry._running_count["a"] = 1
    return rt, prov


@pytest.mark.asyncio
async def test_idle_provider_queues_instead_of_reporting_injected(tmp_path):
    rt, prov = await _running_agent(tmp_path, takes=False)
    out = await rt.send_agent_message("a", "follow-up")
    assert out["status"] == "queued", out
    assert prov.sent == ["follow-up"]
    assert rt.inbox.count("a") == 1
    # Recorded by the engine when it drains the inbox, not here.
    turns = rt.get_agent_conversation_store("a").get_turns()
    assert [t.content for t in turns if t.role == "user"] == []


@pytest.mark.asyncio
async def test_injected_message_is_recorded_once(tmp_path):
    rt, prov = await _running_agent(tmp_path, takes=True)
    out = await rt.send_agent_message("a", "follow-up")
    assert out["status"] == "injected", out
    assert rt.inbox.count("a") == 0
    store = rt.get_agent_conversation_store("a")
    assert [t.content for t in store.get_turns() if t.role == "user"] == ["follow-up"]
    # If the loop ends before reading it, it comes back as a steering
    # fallback: the engine sees it was recorded already, once.
    assert rt.sessions.take_recorded_injection("a", "follow-up") is True
    assert rt.sessions.take_recorded_injection("a", "follow-up") is False
