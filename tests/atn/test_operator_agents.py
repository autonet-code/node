"""External agents act on chain as the household's operator (Decision
2026-11-08): they publish, bond and review with the operator key, and their
reviews merge into the operator's one batch per epoch.

No chain: ToolBond and the review client are stubs throughout."""
from __future__ import annotations

import asyncio

import pytest

from atn.agent_tools import _create_agent, execute_tool
from atn.runtime.review_batch import (
    HELD_NOTE_OPERATOR, flush_held_reviews, queue_reviews, store_for,
)
from tests.atn.test_external_agent import _operator_address, _runtime
from tests.atn.test_review_driver import FakeBond
from tests.atn.test_tool_security import NOVEL, SCHEMA, ECHO, FakeClient

D1, D2, D3 = "a1" * 32, "b2" * 32, "c3" * 32


@pytest.fixture(autouse=True)
def _no_chain_reads(monkeypatch):
    from atn.runtime.tool_bond import DEV_OVERRIDE_ENV
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("ATN_TOOL_BOND_ADDRESS", raising=False)
    monkeypatch.setattr("atn.tool_security._rep_reader", lambda r: None)


async def _externals(tmp_path, *ids, registered=True):
    rt = _runtime(tmp_path)
    for aid in ids:
        out = await _create_agent(rt, {"id": aid, "provider": "external"})
        assert out.get("acts_as_operator"), out
    if registered:
        rt.registry.set_operator_registered(True, "0xreg")
    return rt


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

class TestPublish:
    async def test_publishes_and_bonds_as_the_operator(self, tmp_path):
        rt = await _externals(tmp_path, "cc")
        rt.tool_bond_client = client = FakeClient()
        res = await execute_tool("register_tool", {
            "name": "echo_tool", "description": "Echo x.", "input_schema": SCHEMA,
            "code": ECHO, "novelty": NOVEL}, rt, caller_id="cc")
        assert "error" not in res, res
        assert rt.tool_store.get(res["digest"]).manifest["author"] == _operator_address(rt)
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="cc")
        assert out["published"] is True, out
        assert client.bonds == [(rt.registry.daemon_operator_key(), res["digest"])]

    async def test_unregistered_operator_is_named(self, tmp_path):
        rt = await _externals(tmp_path, "cc", registered=False)
        rt.tool_bond_client = client = FakeClient()
        res = await execute_tool("register_tool", {
            "name": "echo_tool", "description": "Echo x.", "input_schema": SCHEMA,
            "code": ECHO, "novelty": NOVEL}, rt, caller_id="cc")
        out = await execute_tool("publish_tool", {"digest": res["digest"]}, rt,
                                 caller_id="cc")
        assert "operator" in out["error"] and client.bonds == []


# ---------------------------------------------------------------------------
# Reviews: one batch per epoch for the household's sessions
# ---------------------------------------------------------------------------

class KeyedBond(FakeBond):
    def __init__(self):
        super().__init__()
        self.keys: list[tuple[str, str]] = []

    async def commit(self, key, batch):
        self.keys.append(("commit", key))
        return await super().commit(key, batch)


class TestReviews:
    async def test_sessions_merge_into_one_operator_batch(self, tmp_path):
        from atn.review_driver import ReviewDriver
        rt = await _externals(tmp_path, "cc-a", "cc-b")
        rt.review_bond_client = bond = KeyedBond()
        op = _operator_address(rt)
        a = await queue_reviews(rt, "cc-a", [{"digest": D1, "axes": {"correctness": 0.2}}])
        assert a["queued"] == 1 and a["reviewer"] == op
        b = await queue_reviews(rt, "cc-b", [
            {"digest": D1, "axes": {"correctness": 0.9}},
            {"digest": D2, "harm": 0.0},
            # A tool the household published: never its own review.
            {"digest": D3, "axes": {"correctness": 1.0}, "author": op}])
        assert b["queued"] == 2 and b["skipped"][0]["digest"] == D3
        (batch,) = store_for(rt).all()
        assert batch.agent == op and batch.epoch == 3
        # The second review of D1 replaced the first.
        assert [(r.digest, r.axes) for r in batch.reviews] == [
            (D1, {"correctness": 0.9}), (D2, {})]

        driver = ReviewDriver(rt)
        driver.on_local_close()
        done = await driver.tick()
        assert done["committed"] == 1
        assert [c for c in bond.calls if c[0] == "commit"] == [("commit", op, 3)]
        assert bond.keys == [("commit", rt.registry.daemon_operator_key())]

    async def test_held_until_the_operator_registers(self, tmp_path):
        rt = await _externals(tmp_path, "cc", registered=False)
        rt.review_bond_client = KeyedBond()
        out = await queue_reviews(rt, "cc", [{"digest": D1, "axes": {"correctness": 0.5}}])
        assert out["held"] == 1 and out["note"] == HELD_NOTE_OPERATOR
        assert await flush_held_reviews(rt) == 0          # still unregistered
        rt.registry.set_operator_registered(True)
        assert await flush_held_reviews(rt) == 1
        (batch,) = store_for(rt).all()
        assert batch.agent == _operator_address(rt)
        assert [r.digest for r in batch.reviews] == [D1]


# ---------------------------------------------------------------------------
# Services: the household's sessions share the operator's channel
# ---------------------------------------------------------------------------

async def test_sessions_sign_one_voucher_at_a_time(tmp_path, monkeypatch):
    """Two sessions paying the same provider at once use the operator's one
    channel: each request's voucher carries its own running total."""
    from atn import agent_tools, on_chain, service_client
    from atn.service_channels import ChannelBook
    from tests.atn.test_service_channels import FakeMarket

    rt = _runtime(tmp_path)
    for aid in ("cc-a", "cc-b"):
        await _create_agent(rt, {"id": aid, "provider": "external",
                                 "tools": ["services"]})
    rt.channel_book = book = ChannelBook(tmp_path / "channels.json")
    market = FakeMarket()

    class Market(FakeMarket):
        channel_available = True

        def __init__(self, cfg):
            self.__dict__ = market.__dict__

    async def ask(cfg, provider, service_id):
        return {"active": True, "ask_amount": 100}

    sent = []

    async def request(cfg, provider, service_id, payload, **kw):
        sent.append(kw["voucher"]["cumulative_amount"])
        await asyncio.sleep(0.01)          # the provider takes a moment
        return {"delivered": True, "result": {}}

    async def clear(runtime, address, what):
        return None

    monkeypatch.setattr(on_chain, "ServiceMarketClient", Market)
    monkeypatch.setattr(service_client, "lookup_service_ask", ask)
    monkeypatch.setattr(service_client, "request_service", request)
    monkeypatch.setattr(agent_tools, "_sanctioned", clear)
    provider = "0x" + "56" * 20
    args = {"provider_address": provider, "service_id": "s", "payload": {}}
    out = await asyncio.gather(
        execute_tool("request_service", args, rt, caller_id="cc-a"),
        execute_tool("request_service", args, rt, caller_id="cc-b"))
    assert [o["paid"] for o in out] == [100, 100], out
    assert sorted(sent) == [100, 200] and len(market.opened) == 1
    assert book.get(_operator_address(rt), provider)["cumulative"] == 200


def _bridge(rt):
    from atn.ws_server import WebSocketBridge
    return WebSocketBridge(rt)


# ---------------------------------------------------------------------------
# The app's WS surface
# ---------------------------------------------------------------------------

class TestWsSurface:
    async def test_registering_an_external_agent_registers_the_operator(self, tmp_path):
        from atn.ws_auth import ClientSession
        rt = await _externals(tmp_path, "cc", registered=False)
        cfg = rt.autonet.config
        cfg.substrate_address, cfg.rpc_url = "0x" + "34" * 20, "http://127.0.0.1:1"
        rt.autonet.ensure_network_config = lambda: True
        bridge = _bridge(rt)
        asked = []

        async def register_operator(owner="", owner_sig="", self_owned=False):
            asked.append(self_owned)
            rt.registry.set_operator_registered(True, "0xop")
            return {"success": True, "tx_hash": "0xop"}

        bridge._handle_register_operator = register_operator
        local = ClientSession(local=True, authed=True, owner=True)
        resp = await bridge._handle_message({"type": "register_agent_on_chain",
                                             "agent_id": "cc", "self_owned": True}, local)
        assert resp["ok"] and asked == [True]
        r = resp["result"]
        assert r["acts_as_operator"] is True and r["agent_address"] == _operator_address(rt)
        assert "private_key" not in r            # the operator key never leaks here
        assert rt.get_agent("cc").identity.registered_on_chain
        # A second time: already the operator's.
        resp = await bridge._handle_message({"type": "register_agent_on_chain",
                                             "agent_id": "cc"}, local)
        assert resp["result"]["already_registered"] is True and asked == [True]

    async def test_no_agent_key_export_for_an_external_agent(self, tmp_path):
        from atn.ws_auth import ClientSession
        rt = await _externals(tmp_path, "cc")
        bridge = _bridge(rt)

        async def signed(msg, action):
            return ""

        bridge._owner_proof_refusal = signed
        local = ClientSession(local=True, authed=True, owner=True)
        resp = await bridge._handle_message({"type": "export_agent_key",
                                             "agent_id": "cc"}, local)
        assert not resp["ok"] and "export_operator_key" in resp["error"]
