"""The provider's payment gate (atn/ws_server.py, Decision 2026-11-12).

A request is paid from the buyer's credits with the provider
(``_credit_gate``: a payment channel, vouchers paying ahead) or directly
(``_direct_gate``: one ``payForService``); a free service needs neither, and
a daemon without chain serves unpaid only on its local listener. The chain
is faked: the registry entry, the channels the vouchers name, and the
payments by transaction.

Run: pytest tests/test_service_payment_gate.py
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from eth_account import Account

import atn.on_chain as oc
from atn.events import EventBus
from atn.service_client import new_request_id, sign_request
from atn.ws_server import WebSocketBridge

SUBSTRATE_ADDR = "0x4C4dAEd19B98ddaEc6cC421b6a781Bd3fBB7af25"
CHANNEL_ADDR = "0x97956A322dC5585157c91840d39292c396032508"
OTHER_PROVIDER = "0x" + "99" * 20
PAYER_KEY = "0x" + "11" * 32
OTHER_KEY = "0x" + "22" * 32
PRICE = 1000
VOUCHER = 5000
DEPOSIT = 20_000

ECHO = """
import json, os, sys
args = json.load(sys.stdin)
print(json.dumps({"echo": args,
                  "settings": json.loads(os.environ.get("AUTONET_SERVICE_SETTINGS") or "null")}))
"""


def _addr(key: str) -> str:
    return Account.from_key(key).address


def _payer() -> str:
    return _addr(PAYER_KEY)


def _make_runtime(tmp_path: Path, *, chain: bool):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    if chain:
        config.autonet.substrate_address = SUBSTRATE_ADDR
        config.autonet.rpc_url = "http://127.0.0.1:8545"
        config.autonet.payment_channel_address = CHANNEL_ADDR
        config.autonet.service_registry_address = SUBSTRATE_ADDR
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _register_service(rt, *, provider: str = "", active: bool = True, price: int = PRICE,
                      name: str = "echo_svc", code: str = ECHO,
                      settings: dict | None = None):
    """A request service backed by a real tool, and the registry entry the
    gate reads for it (served by ``_patch_listing``). The provider defaults
    to the operator, the address that registers and signs the owner's
    services."""
    tool = rt.tool_store.register(name=f"{name}_tool", description="echo",
                                  input_schema={"type": "object"}, author="user",
                                  code=code)
    res = rt.service_store.register(
        name=name, description="Echo.", mode="request", author="user",
        backing_tool=tool["digest"], prices={"requests": price},
        voucher=VOUCHER if price else 0, settings=settings)
    record = rt.service_store.get(res["digest"])
    operator = _addr(rt.registry.daemon_operator_key())
    record.test_listing = {"service_id": 7, "provider": provider or operator,
                           "ask_amount": 0, "active": active}
    return record


def _patch_listing(monkeypatch):
    """Serve each record's ``test_listing`` as its registry entry."""
    async def _chain(self, record):
        return dict(getattr(record, "test_listing", None)
                    or {"error": "no on-chain listing"})
    monkeypatch.setattr(WebSocketBridge, "_chain_service", _chain)


@pytest.fixture(autouse=True)
def _listing(monkeypatch):
    _patch_listing(monkeypatch)


class Chain:
    """Channels by id and direct payments by transaction, as the gate's
    verify calls read them. A voucher's signature here is
    ``<signer>:<channel>:<total>``."""

    def __init__(self, monkeypatch):
        self.channels: dict[int, dict] = {}
        self.payments: dict[str, dict] = {}
        self.verified: list[dict] = []
        chain = self

        async def verify_voucher(_self, cid, cum, signature):
            ch = chain.channels.get(int(cid))
            if ch is None:
                return {"valid": False, "reason": "no such channel"}
            if str(signature).split(":")[0].lower() != ch["client"].lower():
                return {"valid": False, "reason": "signer is not the channel client"}
            return {"valid": True, "reason": "ok", "client": ch["client"],
                    "provider": ch["provider"], "deposit": str(ch["deposit"]),
                    "exceeds_deposit": int(cum) > ch["deposit"]}

        async def verify_payment(_self, *, request_id, recipient, min_amount,
                                 tx_hash=None, payer=None):
            chain.verified.append({"request_id": request_id, "recipient": recipient,
                                   "min_amount": min_amount, "payer": payer})
            p = chain.payments.get(tx_hash)
            if p is None or p["request_id"] != request_id:
                return {"verified": False, "reason": "no ServicePayment matched"}
            if p["recipient"].lower() != recipient.lower():
                return {"verified": False, "reason": "recipient mismatch"}
            if p["amount"] < min_amount:
                return {"verified": False,
                        "reason": f"amount {p['amount']} < min_amount {min_amount}"}
            if p["payer"].lower() != str(payer).lower():
                return {"verified": False, "reason": "payer mismatch"}
            return {"verified": True, "reason": "ok", "paid_at": p["paid_at"]}

        self.verify_voucher = verify_voucher
        monkeypatch.setattr(oc.ServiceMarketClient, "verify_voucher", verify_voucher)
        monkeypatch.setattr(oc.OnChainService, "verify_service_payment", verify_payment)

    def open(self, cid: int, provider: str, *, client: str = "", deposit: int = DEPOSIT):
        self.channels[cid] = {"client": client or _payer(), "provider": provider,
                              "deposit": deposit}

    def pay(self, rid: str, recipient: str, amount: int, *, key: str = PAYER_KEY,
            paid_at: int = 0) -> str:
        tx = "0x" + f"{len(self.payments) + 1:064x}"
        self.payments[tx] = {"request_id": rid, "recipient": recipient, "amount": amount,
                             "payer": _addr(key), "paid_at": paid_at or int(time.time())}
        return tx


def _request(record, *, key: str = PAYER_KEY, rid: str = "", signed: bool = True,
             **payment) -> dict:
    rid = rid or new_request_id()
    args = {"x": 1}
    msg = {"type": "service_request", "spec_digest": record.digest, "request_id": rid,
           "args": args, **payment}
    if signed:
        msg["payer_signature"] = sign_request(key, rid, record.digest, args,
                                              record.test_listing["provider"])
    return msg


def _voucher(cid: int, total: int, key: str = PAYER_KEY) -> dict:
    return {"channel_id": cid, "cumulative_amount": total,
            "signature": f"{_addr(key)}:{cid}:{total}"}


class Served:
    """Counts the tool runs (the gate tests fake the tool)."""

    def __init__(self, rt, monkeypatch):
        self.calls: list[dict] = []

        async def call(tool, args, caller_id=None, extra_env=None, **kw):
            self.calls.append(args)
            return {"result": {"ok": True}}
        monkeypatch.setattr(rt.tool_store, "call", call)


def _setup(tmp_path, monkeypatch, **kw):
    rt = _make_runtime(tmp_path, chain=kw.pop("chain", True))
    record = _register_service(rt, **kw)
    return rt, record, WebSocketBridge(rt), Chain(monkeypatch), Served(rt, monkeypatch)


async def _send(server, msg, *, local=False):
    return await server._service_request_reply(msg, "m", local=local)


# ---------------------------------------------------------------------------
# Credits
# ---------------------------------------------------------------------------

class TestCredits:
    @pytest.mark.asyncio
    async def test_a_new_voucher_is_verified_and_held(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(5, record.test_listing["provider"])
        out = await _send(server, _request(record, **_voucher(5, VOUCHER)))
        assert out["ok"] is True, out
        assert out["credit"] == {"channel_id": 5, "consumed": PRICE, "cumulative": VOUCHER}
        ledger = rt.service_store.channel_ledger(5)
        assert ledger["client"] == _payer().lower()
        assert ledger["signature"] == f"{_payer()}:5:{VOUCHER}"
        assert len(served.calls) == 1

    @pytest.mark.asyncio
    async def test_existing_credits_serve_without_a_voucher(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(5, record.test_listing["provider"])
        await _send(server, _request(record, **_voucher(5, VOUCHER)))
        out = await _send(server, _request(record, channel_id=5))
        assert out["ok"] is True
        assert rt.service_store.balance(5) == (2 * PRICE, VOUCHER)
        assert len(served.calls) == 2

    @pytest.mark.asyncio
    async def test_not_paid_up_is_refused_with_the_counts(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(5, record.test_listing["provider"])
        # No credits held yet: a voucher is needed, nothing is charged.
        out = await _send(server, _request(record, channel_id=5))
        assert out["ok"] is False and "voucher needed" in out["error"]
        assert out["credit"] == {"channel_id": 5, "consumed": 0, "cumulative": 0}
        # A voucher covering one request, then a second without one.
        assert (await _send(server, _request(record, **_voucher(5, PRICE))))["ok"]
        out = await _send(server, _request(record, channel_id=5))
        assert out["ok"] is False and "voucher needed" in out["error"]
        assert out["credit"] == {"channel_id": 5, "consumed": PRICE, "cumulative": PRICE}
        assert rt.service_store.balance(5) == (PRICE, PRICE)
        assert len(served.calls) == 1

    @pytest.mark.asyncio
    async def test_a_request_id_is_used_once(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(5, record.test_listing["provider"])
        msg = _request(record, **_voucher(5, VOUCHER))
        assert (await _send(server, msg))["ok"] is True
        out = await _send(server, msg)
        assert out["ok"] is False and "replay" in out["error"]
        # The same request sent twice at once (its voucher being checked on
        # chain) is charged once.
        twice = _request(record, **_voucher(5, 2 * VOUCHER))
        outs = await asyncio.gather(_send(server, twice), _send(server, dict(twice)))
        assert sorted(o["ok"] for o in outs) == [False, True]
        assert rt.service_store.balance(5) == (2 * PRICE, 2 * VOUCHER)
        assert len(served.calls) == 2

    @pytest.mark.asyncio
    async def test_the_payer_must_be_the_channels_client(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(5, record.test_listing["provider"])
        # The payer's voucher on a request someone else signed.
        out = await _send(server, _request(record, key=OTHER_KEY, **_voucher(5, VOUCHER)))
        assert out["ok"] is False and "not from the channel's client" in out["error"]
        assert rt.service_store.channel_ledger(5) is None
        # The payer's credits, held, spent by someone else's request.
        assert (await _send(server, _request(record, **_voucher(5, VOUCHER))))["ok"]
        out = await _send(server, _request(record, key=OTHER_KEY, channel_id=5))
        assert out["ok"] is False and "not the payer's" in out["error"]
        # Unsigned.
        out = await _send(server, _request(record, signed=False, channel_id=5))
        assert out["ok"] is False and "payer_signature" in out["error"]
        assert rt.service_store.balance(5) == (PRICE, VOUCHER)
        assert len(served.calls) == 1

    @pytest.mark.asyncio
    async def test_another_providers_channel_is_refused(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(6, OTHER_PROVIDER)
        out = await _send(server, _request(record, **_voucher(6, VOUCHER)))
        assert out["ok"] is False and "not this service's provider" in out["error"]
        assert rt.service_store.channel_ledger(6) is None and served.calls == []

    @pytest.mark.asyncio
    async def test_a_voucher_past_the_deposit_is_refused(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        chain.open(5, record.test_listing["provider"], deposit=VOUCHER)
        out = await _send(server, _request(record, **_voucher(5, VOUCHER + 1)))
        assert out["ok"] is False and "exceeds the channel deposit" in out["error"]
        assert rt.service_store.channel_ledger(5) is None and served.calls == []

    @pytest.mark.asyncio
    async def test_a_service_retired_on_chain_charges_nothing(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch, active=False)
        chain.open(5, record.test_listing["provider"])
        out = await _send(server, _request(record, **_voucher(5, VOUCHER)))
        assert out["ok"] is False and "retired on chain" in out["error"]
        assert rt.service_store.channel_ledger(5) is None and served.calls == []


# ---------------------------------------------------------------------------
# A direct payment
# ---------------------------------------------------------------------------

class TestDirect:
    @pytest.mark.asyncio
    async def test_paid_at_the_price_lists_price(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        provider = record.test_listing["provider"]
        rid = new_request_id()
        out = await _send(server, _request(record, rid=rid,
                                           tx_hash=chain.pay(rid, provider, PRICE)))
        assert out["ok"] is True, out
        assert chain.verified[-1] == {"request_id": rid, "recipient": provider,
                                      "min_amount": PRICE, "payer": _payer()}
        assert rt.service_store.has_served_request(rid)
        # The price list changes (no transaction): the old price falls short.
        rt.service_store.set_prices(record.digest, {"requests": 2 * PRICE})
        rid = new_request_id()
        out = await _send(server, _request(record, rid=rid,
                                           tx_hash=chain.pay(rid, provider, PRICE)))
        assert out["ok"] is False and "min_amount 2000" in out["error"]
        assert not rt.service_store.has_served_request(rid)
        assert len(served.calls) == 1

    @pytest.mark.asyncio
    async def test_a_payment_serves_once(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        rid = new_request_id()
        msg = _request(record, rid=rid,
                       tx_hash=chain.pay(rid, record.test_listing["provider"], PRICE))
        assert (await _send(server, msg))["ok"] is True
        out = await _send(server, msg)
        assert out["ok"] is False and "replay" in out["error"]
        assert len(served.calls) == 1

    @pytest.mark.asyncio
    async def test_a_failed_request_keeps_its_payment_for_a_resend(self, tmp_path,
                                                                   monkeypatch):
        """Not served = not paid: a directly paid request the tool fails does
        not use its request id up, so the same paid request can be sent
        again; once served it is used up."""
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        outcomes = iter([{"error": "tool exited 3: boom"}, {"result": {"ok": True}}])

        async def call(tool, args, caller_id=None, extra_env=None, **kw):
            served.calls.append(args)
            return next(outcomes)
        monkeypatch.setattr(rt.tool_store, "call", call)
        rid = new_request_id()
        msg = _request(record, rid=rid,
                       tx_hash=chain.pay(rid, record.test_listing["provider"], PRICE))
        out = await _send(server, msg)
        assert out["ok"] is False and "send it again" in out["error"], out
        assert not rt.service_store.has_served_request(rid)
        assert (await _send(server, msg))["ok"] is True
        assert rt.service_store.has_served_request(rid)
        assert "replay" in (await _send(server, msg))["error"]
        assert len(served.calls) == 2

    @pytest.mark.asyncio
    async def test_a_payment_past_the_horizon_is_refused(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        rid = new_request_id()
        old = int(time.time()) - rt.service_store.REPLAY_HORIZON_S - 60
        tx = chain.pay(rid, record.test_listing["provider"], PRICE, paid_at=old)
        out = await _send(server, _request(record, rid=rid, tx_hash=tx))
        assert out["ok"] is False and "replay horizon" in out["error"]
        assert served.calls == []

    @pytest.mark.asyncio
    async def test_an_unsigned_or_someone_elses_request_is_refused(self, tmp_path,
                                                                  monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        rid = new_request_id()
        tx = chain.pay(rid, record.test_listing["provider"], PRICE)
        out = await _send(server, _request(record, rid=rid, tx_hash=tx, signed=False))
        assert out["ok"] is False and "unsigned request" in out["error"]
        out = await _send(server, _request(record, rid=rid, tx_hash=tx, key=OTHER_KEY))
        assert out["ok"] is False and "payer mismatch" in out["error"]
        # Neither used the payment up: its payer still can.
        assert (await _send(server, _request(record, rid=rid, tx_hash=tx)))["ok"] is True
        assert len(served.calls) == 1


# ---------------------------------------------------------------------------
# Free services, no payment, no chain
# ---------------------------------------------------------------------------

class TestUnpaid:
    @pytest.mark.asyncio
    async def test_a_free_service_needs_no_payment(self, tmp_path, monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch, price=0)
        out = await _send(server, _request(record, signed=False))
        assert out["ok"] is True and chain.verified == [] and len(served.calls) == 1

    @pytest.mark.asyncio
    async def test_a_priced_request_without_payment_is_refused(self, tmp_path,
                                                              monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch)
        out = await _send(server, _request(record))
        assert out["ok"] is False and "no payment" in out["error"]
        assert served.calls == []

    @pytest.mark.asyncio
    async def test_no_chain_serves_unpaid_on_the_local_listener_only(self, tmp_path,
                                                                    monkeypatch):
        rt, record, server, chain, served = _setup(tmp_path, monkeypatch, chain=False)
        out = await _send(server, _request(record, signed=False), local=False)
        assert out["ok"] is False and "no chain configuration" in out["error"]
        assert served.calls == []
        out = await _send(server, _request(record, signed=False), local=True)
        assert out["ok"] is True and len(served.calls) == 1
        # Logged as served, unpaid.
        assert rt.service_store.summary()[record.digest]["ok_count"] == 1
        rows = (rt.service_store._requests_path.read_text().strip().splitlines())
        assert '"amount": "0"' in rows[-1] and '"payer": ""' in rows[-1]
