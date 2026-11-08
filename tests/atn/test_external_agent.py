"""External agents (atn/external_agents.md): an outside harness drives the
agent over the daemon's WS port; the daemon holds its key and never runs it."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from atn.agent_tools import _create_agent, _review_due
from atn.config import ATNConfig
from atn.events import EventBus
from atn.provider_identity import EXTERNAL_PROVIDER, is_external, routing_error
from atn.runtime import Runtime


def _runtime(tmp_path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def test_external_needs_no_model():
    assert routing_error(EXTERNAL_PROVIDER, "") is None


@pytest.mark.asyncio
async def test_create_external_agent(tmp_path):
    rt = _runtime(tmp_path)
    out = await _create_agent(rt, {"id": "cc-session", "provider": "external",
                                   "harness": "claude-code"})
    assert out.get("external") is True, out
    assert out["address"].startswith("0x")
    defn = rt.get_agent("cc-session")
    assert is_external(defn)
    assert "unified_tools" in defn.tools and "publishing" in defn.tools
    # The daemon never runs it.
    assert await rt.trigger_run("cc-session") is None


@pytest.mark.asyncio
async def test_external_agent_refuses_a_prompt(tmp_path):
    rt = _runtime(tmp_path)
    out = await _create_agent(rt, {"id": "x", "provider": "external",
                                   "prompt": "do things"})
    assert "error" in out


@pytest.mark.asyncio
async def test_review_due_is_adopted_minus_reviewed(tmp_path):
    """Due = the tools the agent adopted minus those it reviewed, derived
    from the tool store, held reviews and batches on disk (Decision
    2026-11-08), so it holds across a restart."""
    from atn.runtime import review_batch
    from atn.tool_store import ToolRecord

    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "cc", "provider": "external"})
    address = rt.get_agent("cc").identity.address
    store = rt.tool_store
    for d, name, origin, who in (("a" * 64, "word_count", "adopted", "cc"),
                                 ("b" * 64, "csv_parse", "adopted", "cc"),
                                 ("c" * 64, "mine", "authored", "cc"),
                                 ("e" * 64, "theirs", "adopted", "other")):
        store._records[d] = ToolRecord(digest=d, manifest={"name": name},
                                       origin=origin, local_author=who)
    assert _review_due(rt, "cc") == {"a" * 64: "word_count", "b" * 64: "csv_parse"}

    # Held before registration: reviewed.
    review_batch.hold_reviews(rt, "cc", [{"digest": "a" * 64, "axes": {}}])
    assert _review_due(rt, "cc") == {"b" * 64: "csv_parse"}

    # In a ToolBond batch: reviewed; in a forfeited one: due again.
    batches = review_batch.store_for(rt)
    entry = review_batch.ReviewEntry(digest="b" * 64, axes={"correctness": 0.5})
    batch = batches.add(address, 3, ["correctness"], entry)
    assert _review_due(rt, "cc") == {}
    batch.status = "forfeited"
    batches.save(batch)
    import os, time
    later = time.time() + 5
    os.utime(batches._dir, (later, later))
    assert _review_due(rt, "cc") == {"b" * 64: "csv_parse"}

    # A restart forgets nothing: the cache is gone, the answer is the same.
    rt.__dict__.pop("_review_due_cache", None)
    assert _review_due(rt, "cc") == {"b" * 64: "csv_parse"}


def test_local_owner_key_is_stable(tmp_path):
    rt = _runtime(tmp_path)
    k1 = rt.registry.local_owner_key()
    k2 = rt.registry.local_owner_key()
    assert k1 == k2 and k1.startswith("0x") and len(k1) == 66
