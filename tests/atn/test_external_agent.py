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


# ---------------------------------------------------------------------------
# External agents act as the operator (Decision 2026-11-08)
# ---------------------------------------------------------------------------

def _operator_address(rt) -> str:
    from eth_account import Account
    return Account.from_key(rt.registry.daemon_operator_key()).address


async def _native(rt, agent_id):
    from atn.models import AgentDefinition, AgentMode
    defn = AgentDefinition(id=agent_id, name=agent_id, mode=AgentMode.COGNITIVE,
                           system_prompt="x", cognitive_model="claude-sonnet-5")
    await rt.register_agent(defn)
    return rt.get_agent(agent_id)


@pytest.mark.asyncio
async def test_external_agents_are_the_operator_native_ones_are_not(tmp_path):
    rt = _runtime(tmp_path)
    op = _operator_address(rt)
    a = await _create_agent(rt, {"id": "cc-a", "provider": "external"})
    b = await _create_agent(rt, {"id": "cc-b", "provider": "external"})
    assert a["address"] == b["address"] == op and a["acts_as_operator"] is True
    # Both sessions sign with the operator key, and share one identity.
    key = rt.registry.daemon_operator_key()
    assert rt.registry.get_agent_key("cc-a") == rt.registry.get_agent_key("cc-b") == key
    assert rt.get_agent("cc-a").identity is rt.get_agent("cc-b").identity
    # No keypair of their own: no identity file.
    assert not (rt._config.agents_dir / "cc-a" / "identity.json").exists()
    # A native agent keeps its own keypair.
    nat = await _native(rt, "nat")
    assert nat.identity.address != op
    assert rt.registry.get_agent_key("nat") != key
    assert (rt._config.agents_dir / "nat" / "identity.json").exists()
    # The operator's address names no single agent.
    assert rt.registry.get_agent_id_by_address(op) is None
    assert rt.registry.get_agent_id_by_address(nat.identity.address) == "nat"
    # It is advertised once.
    ads = [ad["address"] for ad in rt.registry.build_agent_advertisements()]
    assert ads.count(op) == 1


@pytest.mark.asyncio
async def test_registration_is_the_operators_and_persists(tmp_path):
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "cc-a", "provider": "external"})
    await _create_agent(rt, {"id": "cc-b", "provider": "external"})
    assert not rt.get_agent("cc-a").identity.registered_on_chain
    rt.registry.set_operator_registered(True, "0xreg")
    assert rt.get_agent("cc-a").identity.registered_on_chain
    assert rt.get_agent("cc-b").identity.registered_on_chain
    # persist_identity of an external agent writes the operator's file,
    # never the operator key under the agent's vault name.
    rt.registry.persist_identity("cc-a")
    assert (rt._config.data_dir / "operator_identity.json").exists()
    assert not (rt._config.agents_dir / "cc-a" / "identity.json").exists()

    # A restart reads it back.
    from atn.loader import load_agents_dir
    rt2 = _runtime(tmp_path)
    defns, _errors = load_agents_dir(rt2._config.agents_dir)
    for defn in defns:
        await rt2.register_agent(defn, legacy=True)
    assert rt2.get_agent("cc-b").identity.registered_on_chain
    assert rt2.get_agent("cc-b").identity.address == _operator_address(rt2)


@pytest.mark.asyncio
async def test_an_older_external_key_is_ignored(tmp_path):
    """An external agent from an older build had its own keypair: it now
    resolves to the operator, and its old file does no harm."""
    import json as _json
    from eth_account import Account
    rt = _runtime(tmp_path)
    old = Account.create()
    folder = rt._config.agents_dir / "old-cc"
    folder.mkdir(parents=True)
    (folder / "identity.json").write_text(_json.dumps({
        "address": old.address, "public_key": old.address,
        "private_key": old.key.hex(), "registered_on_chain": True}))
    await _create_agent(rt, {"id": "old-cc", "provider": "external"})
    assert rt.get_agent("old-cc").identity.address == _operator_address(rt)
    assert rt.registry.get_agent_key("old-cc") == rt.registry.daemon_operator_key()


@pytest.mark.asyncio
async def test_register_on_chain_says_it_acts_as_the_operator(tmp_path):
    from atn.agent_tools import execute_tool
    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "cc", "provider": "external"})
    out = await execute_tool("register_on_chain", {}, rt, caller_id="cc")
    assert "error" not in out, out
    assert out["acts_as_operator"] is True and out["registered_on_chain"] is False
    assert "operator" in out["note"]
    rt.registry.set_operator_registered(True)
    out = await execute_tool("register_on_chain", {}, rt, caller_id="cc")
    assert out["registered_on_chain"] is True


@pytest.mark.asyncio
async def test_review_due_counts_the_households_reviews(tmp_path):
    """Each agent is reminded of the tools IT adopted, minus what anyone in
    the household reviewed: reviews held for any agent on this daemon and
    every batch that still counts (Decision 2026-11-08)."""
    import os, time
    from atn.runtime import review_batch
    from atn.tool_store import ToolRecord

    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "cc-a", "provider": "external"})
    await _create_agent(rt, {"id": "cc-b", "provider": "external"})
    nat = await _native(rt, "nat")
    store = rt.tool_store
    X, Y, Z = "a" * 64, "b" * 64, "c" * 64
    store._records[X] = ToolRecord(digest=X, manifest={"name": "word_count"},
                                   origin="adopted", local_author="cc-a")
    store._records[Y] = ToolRecord(digest=Y, manifest={"name": "csv_parse"},
                                   origin="adopted", local_author="cc-b")
    store._records[Z] = ToolRecord(digest=Z, manifest={"name": "pdf_text"},
                                   origin="adopted", local_author="nat")
    assert _review_due(rt, "cc-a") == {X: "word_count"}
    assert _review_due(rt, "cc-b") == {Y: "csv_parse"}
    assert _review_due(rt, "nat") == {Z: "pdf_text"}

    # cc-b's session reviews X into the operator's batch: X is no longer
    # due for cc-a, who adopted it.
    batches = review_batch.store_for(rt)
    op = rt.get_agent("cc-b").identity.address
    batch = batches.add(op, 3, ["correctness"],
                        review_batch.ReviewEntry(digest=X, axes={"correctness": 0.5}))
    assert _review_due(rt, "cc-a") == {}
    # A review held for the native agent counts for the household too.
    review_batch.hold_reviews(rt, "nat", [{"digest": Y, "axes": {}}])
    assert _review_due(rt, "cc-b") == {}
    assert _review_due(rt, "nat") == {Z: "pdf_text"}
    # Cached until something changes; a forfeited batch never counted.
    assert rt.__dict__["_review_due_cache"]["cc-a"][1] == {}
    batch.status = "forfeited"
    batches.save(batch)
    later = time.time() + 5
    os.utime(batches._dir, (later, later))
    assert _review_due(rt, "cc-a") == {X: "word_count"}
    assert nat.identity.address != op


@pytest.mark.asyncio
async def test_review_due_clears_when_the_batch_dir_mtime_stands_still(tmp_path):
    """On exFAT a directory's mtime never moves when a file inside changes
    (S0 2026-11-09: review_due never cleared on D:). Every write to the
    batches or held reviews invalidates the cache whatever the filesystem."""
    import os
    from atn.runtime import review_batch
    from atn.tool_store import ToolRecord

    rt = _runtime(tmp_path)
    await _create_agent(rt, {"id": "cc", "provider": "external"})
    op = rt.get_agent("cc").identity.address
    X, Y = "a" * 64, "b" * 64
    for d, name in ((X, "word_count"), (Y, "csv_parse")):
        rt.tool_store._records[d] = ToolRecord(digest=d, manifest={"name": name},
                                               origin="adopted", local_author="cc")
    batches = review_batch.store_for(rt)
    batches.add(op, 1, ["correctness"],
                review_batch.ReviewEntry(digest=X, axes={"correctness": 0.5}))
    frozen = os.stat(batches._dir).st_mtime
    assert _review_due(rt, "cc") == {Y: "csv_parse"}
    # The next epoch's batch, and the directory's mtime pinned as exFAT
    # leaves it.
    batches.add(op, 2, ["correctness"],
                review_batch.ReviewEntry(digest=Y, axes={"correctness": 0.5}))
    os.utime(batches._dir, (frozen, frozen))
    assert _review_due(rt, "cc") == {}


@pytest.mark.asyncio
async def test_external_agents_can_buy_services_by_default(tmp_path):
    """The operator pays (Decision 2026-11-08), so buying a service is on by
    default; selling one is a separate grant."""
    from atn.agent_tools import _TOOL_CATEGORIES, EXTERNAL_DEFAULT_TOOLS
    granted = set().union(*(_TOOL_CATEGORIES[b] for b in EXTERNAL_DEFAULT_TOOLS))
    assert {"find_services", "request_service", "close_service_channel"} <= granted
    assert "register_service" not in granted


# ---------------------------------------------------------------------------
# Adopted tools are shared across the household's external sessions
# (Decision 2026-11-09)
# ---------------------------------------------------------------------------

async def _adopting_daemon(tmp_path):
    """A daemon with two external sessions and a native agent, fetching from
    a separate author daemon that published one tool."""
    from atn.runtime.tool_bond import ToolBondGate
    from tests.atn.test_tool_store import TestAdoption
    _author, digest, blobs = await TestAdoption()._publish_on_author_daemon(
        tmp_path / "net")
    rt = _runtime(tmp_path / "home")
    rt.tool_bond_gate = ToolBondGate(rt, checker=lambda d: True)

    async def _fetch(d):
        return blobs.get_bytes(d)
    rt.tool_store.blob_fetcher = _fetch
    await _create_agent(rt, {"id": "cc-a", "provider": "external"})
    await _create_agent(rt, {"id": "cc-b", "provider": "external"})
    await _native(rt, "nat")
    return rt, digest


def _names(listing):
    return {t.get("tool_name") for t in listing.get("tools", [])}


@pytest.mark.asyncio
async def test_a_tool_one_session_adopted_serves_every_external_session(tmp_path):
    from atn.agent_tools import execute_tool
    from atn.tool_search import search_tools
    rt, digest = await _adopting_daemon(tmp_path)
    store = rt.tool_store
    first = await execute_tool("adopt_tool", {"digest": digest}, rt, caller_id="cc-a")
    assert first["origin"] == "adopted" and first["adopted_for"] == "cc-a", first
    record = store.get(digest)

    # Each session is reminded of what IT adopted.
    assert _review_due(rt, "cc-a") == {digest: "net_tool"}
    assert _review_due(rt, "cc-b") == {}

    # The other session sees it, uses it and may review it.
    assert store.allowed("cc-b", record)
    listed = await execute_tool("list_tools", {"category": "registered"}, rt,
                                caller_id="cc-b")
    assert "net_tool" in _names(listed), listed
    used = await execute_tool("use_tool", {"name": "net_tool", "arguments": {"x": "hi"}},
                              rt, caller_id="cc-b")
    assert used.get("result") == {"echo": "hi"}, used
    reviewed = await execute_tool(
        "attest_tools", {"context": "work", "judgments": [
            {"tool": "net_tool", "correctness": 0.5, "simplicity": 0.5,
             "novelty": 0.5, "malware": False}]},
        rt, caller_id="cc-b")
    assert reviewed.get("attested") == 1 and not reviewed.get("skipped"), reviewed
    # Reviewed by the household: due for nobody.
    assert _review_due(rt, "cc-a") == {} and "review_due" not in reviewed

    # Adopting it again is "already installed", not an error, and installs
    # nothing new: the record stays cc-a's.
    again = await execute_tool("adopt_tool", {"digest": digest}, rt, caller_id="cc-b")
    assert "error" not in again, again
    assert again["already_installed"] is True and again["adopted_for"] == "cc-a"
    assert "trust" not in again and "rating" in again
    assert store.get(digest).local_author == "cc-a"
    mine = await execute_tool("adopt_tool", {"digest": digest}, rt, caller_id="cc-a")
    assert mine["already_installed"] is True and "already" in mine["note"]

    # It checks out for the other session too.
    dest = tmp_path / "cc-b-checkout"
    out = await execute_tool("adopt_tool", {"digest": digest, "path": str(dest)},
                             rt, caller_id="cc-b")
    assert out["checkout"]["path"] == str(dest.resolve()), out
    assert (dest / "main.py").exists()

    # Search marks it adopted here for the other session.
    found = await search_tools(rt, "network published tool", 5, caller_id="cc-b")
    row = next(m for m in found["matches"] if m["digest"] == digest)
    assert row["household"]["adopted"] is True


@pytest.mark.asyncio
async def test_native_agents_keep_their_lineage_scoping(tmp_path):
    from atn.agent_tools import execute_tool
    rt, digest = await _adopting_daemon(tmp_path)
    store = rt.tool_store
    await execute_tool("adopt_tool", {"digest": digest}, rt, caller_id="cc-a")
    record = store.get(digest)
    # A native agent does not get an external session's adoption...
    assert not store.allowed("nat", record)
    listed = await execute_tool("list_tools", {"category": "registered"}, rt,
                                caller_id="nat")
    assert "net_tool" not in _names(listed)
    used = await execute_tool("use_tool", {"name": "net_tool", "arguments": {"x": "hi"}},
                              rt, caller_id="nat")
    assert "not available" in used.get("error", ""), used
    reviewed = await execute_tool(
        "attest_tools", {"context": "work", "judgments": [
            {"tool": "net_tool", "correctness": 0.5, "simplicity": 0.5,
             "novelty": 0.5, "malware": False}]}, rt, caller_id="nat")
    assert reviewed.get("attested") == 0 and reviewed["skipped"], reviewed
    # ...nor its code.
    dest = tmp_path / "nat-checkout"
    out = await execute_tool("adopt_tool", {"digest": digest, "path": str(dest)},
                             rt, caller_id="nat")
    assert "checkout" not in out and not dest.exists(), out

    # And a native adopter's tool stays its lineage's.
    record.local_author = "nat"
    assert store.allowed("nat", record)
    assert not store.allowed("cc-a", record) and not store.allowed("cc-b", record)
