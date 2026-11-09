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


@pytest.mark.asyncio
async def test_a_malware_flag_announces_its_evidence_and_nothing_else(tmp_path):
    """Decision 2026-11-09: the reviewer's daemon provides the evidence of
    its own malware flags in the DHT, so any daemon can fetch it while this
    one is online. Other review blobs (a note, a vet report) are not
    announced."""
    from tests.atn.test_review_driver import FIELDS, _reviewer

    rt, digest, bond, addr = await _reviewer(tmp_path)
    announced: list[list[str]] = []
    rt.tool_store.blob_announcer = lambda digests: announced.append(list(digests))

    out = await execute_tool("attest_tools", {
        "judgments": [{"tool": "echo_tool", **FIELDS, "note": "fine, echoes x"}],
        "context": "x"}, rt, caller_id="parent")
    assert out["attested"] == 1
    await asyncio.sleep(0.1)
    assert announced == []                    # a plain review: nothing

    out = await execute_tool("attest_tools", {
        "judgments": [{"tool": "echo_tool", **FIELDS, "correctness": -1.0,
                       "malware": True,
                       "evidence": "tool.py line 3 posts ~/.ssh/id_rsa to evil.example"}],
        "context": "x"}, rt, caller_id="parent")
    assert out["attested"] == 1
    evidence_hash = rt.tool_store.recent_attestations(digest)[0]["evidence_hash"]
    assert len(evidence_hash) == 64
    for _ in range(100):
        if announced:
            break
        await asyncio.sleep(0.02)
    assert announced == [[evidence_hash]]
    # What the host provides at start: the flags' evidence still held.
    assert rt.tool_store.held_evidence_digests() == [evidence_hash]
    assert evidence_hash not in rt.tool_store.held_blob_digests()


@pytest.mark.asyncio
async def test_a_flag_is_kept_when_announcing_fails(tmp_path):
    from tests.atn.test_review_driver import FIELDS, _reviewer

    rt, digest, bond, addr = await _reviewer(tmp_path)

    def broken(_digests):
        raise RuntimeError("no DHT peers")
    rt.tool_store.blob_announcer = broken
    out = await execute_tool("attest_tools", {
        "judgments": [{"tool": "echo_tool", **FIELDS, "malware": True,
                       "evidence": "line 9 deletes the home dir"}],
        "context": "x"}, rt, caller_id="parent")
    assert out["attested"] == 1 and out["reviews"]["queued"] == 1
    await asyncio.sleep(0.1)
    (h,) = rt.tool_store.held_evidence_digests()
    got = await execute_tool("fetch_evidence", {"evidence_hash": h}, rt,
                             caller_id="parent")
    assert got["found"] and got["evidence"]["evidence"].startswith("line 9")
