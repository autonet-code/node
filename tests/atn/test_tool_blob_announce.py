"""A published tool's blobs are announced in the DHT before its
registration event is sent (docs/p2p_networking.md, "Blobs"): peers that
react to the registration can already find a holder of the manifest and
the code. Uses the ToolStore's blob_announcer hook with a recording fake
(the real one is AutonetHost.provide_blobs_blocking).
"""
from __future__ import annotations

import asyncio

import pytest

from atn.agent_tools import execute_tool

from tests.atn.test_tool_store import SCHEMA, ECHO_CODE, _family, _make_runtime


@pytest.mark.asyncio
async def test_publish_announces_manifest_and_code_before_registering(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_TOOL_BOND_DEV_OVERRIDE", "1")
    rt = _make_runtime(tmp_path)
    rt._config.autonet.owner_wallet = "0x" + "44" * 20
    await _family(rt)
    events: list[tuple] = []
    rt.tool_store.manifest_sink = lambda m, a: events.append(("register", m["name"]))
    rt.tool_store.blob_announcer = lambda digests: events.append(("announce", list(digests)))

    res = await execute_tool(
        "register_tool",
        {"name": "echo_tool", "description": "Echo x back.",
         "input_schema": SCHEMA, "code": ECHO_CODE},
        rt, caller_id="child")
    assert events == []                       # private: nothing announced
    pub = await execute_tool(
        "publish_tool", {"digest": res["digest"], "novelty": {
            "statement": "No comparable tool exists.", "prior_art": []}},
        rt, caller_id="child")
    assert pub["published"] is True
    for _ in range(100):
        if any(e[0] == "register" for e in events):
            break
        await asyncio.sleep(0.02)
    record = rt.tool_store.get(pub["digest"])
    code = record.manifest["code_digest"]
    assert events[0] == ("announce", [pub["digest"], code])
    assert events[1] == ("register", "echo_tool")
    # What the host re-provides at start: every tool blob we hold.
    held = rt.tool_store.held_blob_digests()
    assert pub["digest"] in held and code in held


@pytest.mark.asyncio
async def test_registration_goes_out_even_if_announcing_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_TOOL_BOND_DEV_OVERRIDE", "1")
    rt = _make_runtime(tmp_path)
    rt._config.autonet.owner_wallet = "0x" + "44" * 20
    await _family(rt)
    pushed: list[str] = []
    rt.tool_store.manifest_sink = lambda m, a: pushed.append(m["name"])

    def broken(_digests):
        raise RuntimeError("no DHT peers")
    rt.tool_store.blob_announcer = broken
    res = await execute_tool(
        "register_tool",
        {"name": "echo_tool", "description": "Echo x back.",
         "input_schema": SCHEMA, "code": ECHO_CODE},
        rt, caller_id="child")
    await execute_tool(
        "publish_tool", {"digest": res["digest"], "novelty": {
            "statement": "No comparable tool exists.", "prior_art": []}},
        rt, caller_id="child")
    for _ in range(100):
        if pushed:
            break
        await asyncio.sleep(0.02)
    assert pushed == ["echo_tool"]
