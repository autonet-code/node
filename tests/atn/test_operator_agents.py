"""External agents act on chain as the household's operator (Decision
2026-11-08): they publish, bond, review and send transactions with the
operator key, their reviews merge into the operator's one batch per epoch,
and every operator transaction draws its nonce from one manager.

No chain: ToolBond, the review client and web3 are stubs throughout."""
from __future__ import annotations

import asyncio
import threading

import pytest
from eth_account import Account
from hexbytes import HexBytes

from atn.agent_tools import _create_agent, execute_tool
from atn.runtime.review_batch import (
    HELD_NOTE_OPERATOR, flush_held_reviews, queue_reviews, store_for,
)
from tests.atn.test_external_agent import _native, _operator_address, _runtime
from tests.atn.test_review_driver import FakeBond
from tests.atn.test_tool_security import NOVEL, SCHEMA, ECHO, FakeClient

D1, D2, D3 = "a1" * 32, "b2" * 32, "c3" * 32
TARGET = "0x" + "12" * 20


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


_F = {"correctness": 0.5, "simplicity": 0.5, "novelty": 0.5, "malware": False}


class TestReviews:
    async def test_sessions_merge_into_one_operator_batch(self, tmp_path):
        from atn.review_driver import ReviewDriver
        rt = await _externals(tmp_path, "cc-a", "cc-b")
        rt.review_bond_client = bond = KeyedBond()
        op = _operator_address(rt)
        a = await queue_reviews(rt, "cc-a", [{"digest": D1, **_F, "correctness": 0.2}])
        assert a["queued"] == 1 and a["reviewer"] == op
        b = await queue_reviews(rt, "cc-b", [
            {"digest": D1, **_F, "correctness": 0.9},
            {"digest": D2, **_F},
            # A tool the household published: never its own review.
            {"digest": D3, **_F, "author": op}])
        assert b["queued"] == 2 and b["skipped"][0]["digest"] == D3
        (batch,) = store_for(rt).all()
        assert batch.agent == op and batch.epoch == 3
        # The second review of D1 replaced the first.
        assert [(r.digest, r.correctness) for r in batch.reviews] == [
            (D1, 0.9), (D2, 0.5)]

        driver = ReviewDriver(rt)
        driver.on_local_close()
        done = await driver.tick()
        assert done["committed"] == 1
        assert [c for c in bond.calls if c[0] == "commit"] == [("commit", op, 3)]
        assert bond.keys == [("commit", rt.registry.daemon_operator_key())]

    async def test_held_until_the_operator_registers(self, tmp_path):
        rt = await _externals(tmp_path, "cc", registered=False)
        rt.review_bond_client = KeyedBond()
        out = await queue_reviews(rt, "cc", [{"digest": D1, **_F}])
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


# ---------------------------------------------------------------------------
# send_transaction
# ---------------------------------------------------------------------------

class _Eth:
    def __init__(self, count=7):
        self.count = count
        self.sent: list[bytes] = []
        self.status = 1
        self.estimate_error = None
        self.gas_price = 10 ** 9
        self.chain_id = 31337
        self._lock = threading.Lock()

    def estimate_gas(self, tx):
        if self.estimate_error:
            raise self.estimate_error
        return 50_000

    def get_transaction_count(self, address, block="latest"):
        return self.count        # an RPC that never sees the sends

    def send_raw_transaction(self, raw):
        with self._lock:
            self.sent.append(bytes(raw))
        return HexBytes(b"\x01" * 32)

    def wait_for_transaction_receipt(self, tx_hash, timeout=120):
        from web3.datastructures import AttributeDict
        return AttributeDict({"status": self.status, "blockNumber": 42, "gasUsed": 41_000})

    def call(self, tx, block=None):
        raise ValueError("execution reverted: NotAuthor")


class _W3:
    def __init__(self, count=7):
        self.eth = _Eth(count)

    @staticmethod
    def to_checksum_address(a):
        from web3 import Web3
        return Web3.to_checksum_address(a)


def _decoded(raw: bytes) -> tuple[str, int]:
    """(sender, nonce) of a signed legacy transaction."""
    import rlp
    return Account.recover_transaction(raw), int.from_bytes(rlp.decode(raw)[0], "big")


def _bridge(rt):
    from atn.ws_server import WebSocketBridge
    return WebSocketBridge(rt)


class TestSendTransaction:
    async def test_signs_as_the_operator_and_lands_in_activity(self, tmp_path):
        from atn.ws_auth import ClientSession
        rt = await _externals(tmp_path, "cc", registered=False)
        rt.tx_web3 = w3 = _W3()
        bridge = _bridge(rt)
        owner = ClientSession(local=True, authed=True, owner=True)
        data = "0x12345678" + "00" * 32
        resp = await bridge._handle_message({
            "type": "send_transaction", "caller_id": "cc", "to": TARGET,
            "data": data, "value": 5, "_note": "vote For on proposal 7"}, owner)
        assert resp["ok"], resp
        r = resp["result"]
        assert r["status"] == 1 and r["block"] == 42 and r["gas_used"] == 41_000
        assert r["tx_hash"] and "revert_reason" not in r
        op = _operator_address(rt)
        assert r["from"] == op and r["selector"] == "0x12345678" and r["value"] == "5"
        assert _decoded(w3.eth.sent[0]) == (op, 7)
        (entry,) = bridge._activity_log().read("cc")
        assert entry["tool"] == "send_transaction" and entry["ok"]
        assert entry["note"] == "vote For on proposal 7"
        assert TARGET in entry["result"] and "0x12345678" in entry["result"]

        # The next one takes the next nonce though the RPC lags.
        out = await execute_tool("send_transaction", {"to": TARGET}, rt, caller_id="cc")
        assert out["status"] == 1 and _decoded(w3.eth.sent[1]) == (op, 8)

    async def test_reverts_and_refusals(self, tmp_path):
        rt = await _externals(tmp_path, "cc")
        rt.tx_web3 = w3 = _W3()
        w3.eth.status = 0
        out = await execute_tool("send_transaction", {
            "to": TARGET, "data": "0xdeadbeef", "gas": 90_000}, rt, caller_id="cc")
        assert out["status"] == 0 and "NotAuthor" in out["revert_reason"]
        # Without a gas limit, a call that would revert is not sent.
        w3.eth.estimate_error = ValueError("execution reverted: TooEarly")
        out = await execute_tool("send_transaction", {"to": TARGET, "data": "0xdeadbeef"},
                                 rt, caller_id="cc")
        assert "TooEarly" in out["error"] and "0xdeadbeef" in out["error"]
        assert len(w3.eth.sent) == 1
        # Bad input never reaches the chain.
        for bad in ({"to": "0x12"}, {"to": TARGET, "data": "0xabc"},
                    {"to": TARGET, "value": -1}):
            assert "error" in await execute_tool("send_transaction", bad, rt,
                                                 caller_id="cc")
        # Native agents and the owner surface are refused.
        await _native(rt, "nat")
        out = await execute_tool("send_transaction", {"to": TARGET}, rt, caller_id="nat")
        assert "external agents" in out["error"]
        out = await execute_tool("send_transaction", {"to": TARGET}, rt, caller_id="")
        assert "external agents" in out["error"]
        assert len(w3.eth.sent) == 1

    async def test_granted_to_every_external_agent(self, tmp_path):
        rt = _runtime(tmp_path)
        await _create_agent(rt, {"id": "narrow", "provider": "external",
                                 "tools": ["unified_tools"]})
        rt.tx_web3 = _W3()
        out = await execute_tool("send_transaction", {"to": TARGET}, rt,
                                 caller_id="narrow")
        assert out.get("status") == 1, out


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


# ---------------------------------------------------------------------------
# One nonce manager for every operator transaction
# ---------------------------------------------------------------------------

def test_operator_paths_never_share_a_nonce(tmp_path):
    """Agent transactions and the ToolBond path (stakes, bonds, reviews)
    run concurrently from the operator key against an RPC that lags: the
    shared per-address nonce manager hands each its own nonce."""
    from atn.agent_tools import _send_operator_tx
    from atn.runtime.tool_bond import ToolBondClient

    key = Account.create().key.hex()
    w3 = _W3(count=20)

    class _Func:
        def estimate_gas(self, tx):
            return 100_000

        def build_transaction(self, tx):
            return {**tx, "to": w3.to_checksum_address(TARGET), "data": "0x", "value": 0}

    bond_client = ToolBondClient(None)
    acct = Account.from_key(key)
    jobs = [lambda: _send_operator_tx(w3, key, TARGET, "0x", 0, None, 31337)
            for _ in range(4)]
    jobs += [lambda: bond_client._send(w3, acct, _Func(), 100_000) for _ in range(4)]
    threads = [threading.Thread(target=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    nonces = sorted(_decoded(raw)[1] for raw in w3.eth.sent)
    assert nonces == list(range(20, 28))
