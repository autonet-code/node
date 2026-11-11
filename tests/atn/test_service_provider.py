"""A request service end to end (Decision 2026-11-12), only the chain and
the wire faked: the buyer's credits (atn/service_channels.py) send a
request and a voucher, the provider's gate (atn/ws_server.py) checks them,
runs the service's tool once with the service's settings, signs the reply
and reports its credit count. A served request stays charged; a refused or
failed one costs nothing. Then a native agent's model calls bought the
same way (atn/providers/service.py)."""

from __future__ import annotations

import json

import pytest

import atn.on_chain as oc
from atn import service_client
from atn.providers.base import ProviderError
from atn.providers.service import ServiceProvider
from atn.service_channels import (
    CLIENT_CLOSING,
    OPEN,
    ChannelBook,
    open_credits,
    request_on_credits,
)
from atn.ws_server import WebSocketBridge
from tests.test_service_payment_gate import (
    DEPOSIT,
    PAYER_KEY,
    PRICE,
    VOUCHER,
    Chain,
    _make_runtime,
    _patch_listing,
    _payer,
    _register_service,
)

FAILS = "import sys\nsys.exit(3)\n"
ANSWERS = """
import json, os, sys
args = json.load(sys.stdin)
settings = json.loads(os.environ["AUTONET_SERVICE_SETTINGS"])
print(json.dumps({"content": "re: " + args["messages"][-1]["content"],
                  "model": settings["model"], "usage": {"input_tokens": 3}}))
"""


class Market:
    """The buyer's PaymentChannel client, opening channels on ``chain``
    (signatures in the format the provider's fake verify reads)."""

    channel_available = registry_available = True

    def __init__(self, chain: Chain, provider: str):
        self.chain, self.provider = chain, provider
        self.signed: list[int] = []

    async def open_channel(self, key, provider, deposit):
        cid = len(self.chain.channels) + 1
        self.chain.open(cid, provider, deposit=deposit)
        return {"success": True, "channel_id": cid}

    async def get_channel(self, cid):
        ch = self.chain.channels.get(cid)
        return None if ch is None else {"status": ch.get("status", OPEN), "closes_at": 0}

    async def start_close(self, key, cid):
        self.chain.channels[cid]["status"] = CLIENT_CLOSING
        return {"success": True, "tx_hash": "0xc"}

    def sign_voucher(self, key, cid, cumulative):
        self.signed.append(cumulative)
        return f"{_payer()}:{cid}:{cumulative}"

    async def verify_voucher(self, cid, cumulative, signature):
        # The provider's gate reads the same chain.
        return await self.chain.verify_voucher(self, cid, cumulative, signature)


class _Host:
    _trio_token = object()


@pytest.fixture
def market(tmp_path, monkeypatch):
    """A provider daemon with the libp2p wire to it faked: a request is
    handed to its gate exactly as the host would hand it over."""
    _patch_listing(monkeypatch)
    rt = _make_runtime(tmp_path, chain=True)
    server = WebSocketBridge(rt)
    chain = Chain(monkeypatch)

    async def p2p(host, peer, frame, timeout, accept=None):
        return await server._serve_p2p_service_request(json.loads(json.dumps(frame)))
    monkeypatch.setattr(service_client, "p2p_request", p2p)
    return rt, server, chain


async def _buy(rt, chain, record, args=None, book=None):
    provider = record.test_listing["provider"]
    m = Market(chain, provider)
    if book is None:
        book = ChannelBook(rt._config.data_dir / f"buyer_{record.digest[:8]}.json")
        assert (await open_credits(book, m, PAYER_KEY, _payer(), provider, DEPOSIT))["ok"]
    out = await request_on_credits(
        book, m, rt._config.rpb, PAYER_KEY, _payer(), provider, record.digest,
        args or {"messages": [{"role": "user", "content": "hi"}]},
        price=PRICE, voucher=VOUCHER, p2p_host=_Host(), peer_id="peer", timeout=30)
    return out, book, m


def _rows(rt) -> list[dict]:
    path = rt.service_store._requests_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_a_served_request_stays_charged(market):
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "m1"})
    out, book, m = await _buy(rt, chain, record)
    assert out["ok"] and out["authenticated"], out
    assert out["result"]["result"] == {"content": "re: hi", "model": "m1",
                                       "usage": {"input_tokens": 3}}
    assert out["credit"] == {"channel_id": 1, "consumed": PRICE, "cumulative": VOUCHER}
    assert out["credits"]["used"] == PRICE and out["credits"]["paid_ahead"] == VOUCHER - PRICE
    assert rt.service_store.balance(1) == (PRICE, VOUCHER)
    # The next request is paid from what is held: no new voucher.
    out, _, _ = await _buy(rt, chain, record, book=book)
    assert out["ok"] and m.signed == [VOUCHER]
    assert rt.service_store.balance(1) == (2 * PRICE, VOUCHER)
    assert book.get(_payer(), record.test_listing["provider"])["consumed"] == 2 * PRICE
    assert [(r["ok"], r["amount"], r["payer"]) for r in _rows(rt)] == \
        [(True, str(PRICE), _payer().lower())] * 2


@pytest.mark.asyncio
async def test_a_failed_tool_run_is_refunded(market):
    rt, server, chain = market
    record = _register_service(rt, code=FAILS)
    out, book, _ = await _buy(rt, chain, record)
    assert out["ok"] is False and out["authenticated"]
    assert "the service's tool failed" in out["error"] and "exited 3" in out["error"]
    assert out["credit"] == {"channel_id": 1, "consumed": 0, "cumulative": VOUCHER}
    assert rt.service_store.balance(1) == (0, VOUCHER)
    assert out["credits"]["used"] == 0 and out["credits"]["paid_ahead"] == VOUCHER
    assert [(r["ok"], r["amount"]) for r in _rows(rt)] == [(False, "0")]


@pytest.mark.asyncio
async def test_a_session_service_is_not_served_requests(market):
    rt, server, chain = market
    tool = rt.tool_store.register(
        name="chat_server", description="serves chat", input_schema={"type": "object"},
        author="user", code="print(1)",
        session={"protocol": {"id": "openai-chat/1", "spec": "cd" * 32},
                 "max_sessions": 1, "process": "shared"})
    res = rt.service_store.register(
        name="chat", description="Chat.", mode="session", author="user",
        backing_tool=tool["digest"], prices={"seconds": 10}, voucher=VOUCHER,
        measures=[{"measure": "seconds", "quantity": 60}])
    record = rt.service_store.get(res["digest"])
    record.test_listing = _register_service(rt, name="other").test_listing
    out, _, _ = await _buy(rt, chain, record)
    assert out["ok"] is False and out["authenticated"]
    assert "session service" in out["error"]
    assert rt.service_store.channel_ledger(1) is None and _rows(rt) == []


@pytest.mark.asyncio
async def test_a_retired_service_charges_nothing(market):
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "m1"})
    rt.service_store.retire(record.digest)
    out, _, _ = await _buy(rt, chain, record)
    assert out["ok"] is False and out["error"] == "Service retired"
    # Retired on chain (the daemon's record still live).
    live = _register_service(rt, name="live", code=ANSWERS, settings={"model": "m1"},
                             active=False)
    out, _, _ = await _buy(rt, chain, live)
    assert out["ok"] is False and "retired on chain" in out["error"]
    assert rt.service_store.channel_vouchers() == {} and _rows(rt) == []


@pytest.mark.asyncio
async def test_settings_reach_the_tool(market):
    rt, server, chain = market
    record = _register_service(rt, price=0, settings={"model": "large", "size": "7b"})
    out = await service_client.request_service(
        rt._config.rpb, record.test_listing["provider"], record.digest, {"q": 1},
        p2p_host=_Host(), peer_id="peer", timeout=30)
    assert out["ok"] and out["authenticated"]
    assert out["result"]["result"] == {"echo": {"q": 1},
                                       "settings": {"model": "large", "size": "7b"}}


# ---------------------------------------------------------------------------
# A native agent's model calls (atn/providers/service.py)
# ---------------------------------------------------------------------------

class _Catalog:
    def __init__(self, rt):
        self.store = rt.service_store

    def price_list(self, digest, provider=""):
        rec = self.store.get(digest)
        return dict(rec.price_list) if rec and rec.price_list else None

    def spec(self, digest):
        rec = self.store.get(digest)
        return rec.spec if rec else None


class _Clear:
    def check(self, address):
        return None


def _buyer_side(chain, record, monkeypatch):
    """What a buyer's daemon reads and sends, faked around the provider:
    the registry entry, the provider's peer id (the wire is ``market``'s),
    direct payments onto ``chain`` and the PaymentChannel client (which the
    provider's gate shares). Returns the channel client and the payments."""
    provider = record.test_listing["provider"]
    m = Market(chain, provider)

    async def entry(config, *a, **kw):
        return {"service_id": 7, "provider": provider, "spec_digest": record.digest,
                "ask_amount": 0, "active": record.test_listing["active"]}

    async def peer(config, address, host=None):
        return "peer"
    paid: list[dict] = []

    async def pay(config, key, recipient, amount, request_id=""):
        paid.append({"recipient": recipient, "amount": amount})
        return {"tx_hash": chain.pay(request_id, recipient, amount, key=key),
                "request_id": request_id}
    monkeypatch.setattr(service_client, "lookup_service_ask", entry)
    monkeypatch.setattr(service_client, "read_service", entry)
    monkeypatch.setattr(service_client, "resolve_provider_peer_id", peer)
    monkeypatch.setattr(service_client, "pay_for_service", pay)
    monkeypatch.setattr(service_client, "_p2p_host", _Host())
    monkeypatch.setattr(oc, "ServiceMarketClient", lambda cfg: m)
    return m, paid


def _model_provider(rt, chain, record, monkeypatch, book):
    m, paid = _buyer_side(chain, record, monkeypatch)
    monkeypatch.setattr(service_client, "_catalog", _Catalog(rt))
    prov = ServiceProvider(config=rt._config.rpb,
                           provider_address=record.test_listing["provider"],
                           spec_digest=record.digest, signing_key=PAYER_KEY,
                           channel_book=book)
    prov._chain_configured = lambda: True
    prov._sanctions = _Clear()
    return prov, m, paid


@pytest.mark.asyncio
async def test_a_model_call_is_paid_from_credits_else_directly(market, monkeypatch):
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "seller-1"})
    book = ChannelBook(rt._config.data_dir / "buyer_channels.json")
    prov, m, paid = _model_provider(rt, chain, record, monkeypatch, book)
    # No credits with the provider: one direct payment for the call.
    resp = await prov.send(messages=[{"role": "user", "content": "one"}])
    assert resp.text == "re: one" and resp.model == "seller-1"
    assert resp.usage.input_tokens == 3
    assert paid == [{"recipient": record.test_listing["provider"], "amount": PRICE}]
    assert prov.last_payment["paid"] == PRICE and m.signed == []
    # With credits: a voucher, no payment.
    assert (await open_credits(book, m, PAYER_KEY, _payer(),
                               record.test_listing["provider"], DEPOSIT))["ok"]
    resp = await prov.send(messages=[{"role": "user", "content": "two"}])
    assert resp.text == "re: two" and len(paid) == 1 and m.signed == [VOUCHER]
    assert prov.last_payment == {"credits": prov.last_payment["credits"], "paid": PRICE}
    assert prov.last_payment["credits"]["used"] == PRICE


@pytest.mark.asyncio
async def test_a_model_call_needs_a_priced_request_service(market, monkeypatch):
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "seller-1"})
    book = ChannelBook(rt._config.data_dir / "buyer_channels.json")
    prov, _, paid = _model_provider(rt, chain, record, monkeypatch, book)
    monkeypatch.setattr(_Catalog, "spec", lambda self, d: {"mode": "session"})
    with pytest.raises(ProviderError, match="session service"):
        await prov.send(messages=[{"role": "user", "content": "x"}])
    monkeypatch.setattr(_Catalog, "spec", lambda self, d: None)
    monkeypatch.setattr(_Catalog, "price_list", lambda self, d, p="": None)
    with pytest.raises(ProviderError, match="no price list"):
        await prov.send(messages=[{"role": "user", "content": "x"}])
    record.test_listing["active"] = False
    with pytest.raises(ProviderError, match="retired"):
        await prov.send(messages=[{"role": "user", "content": "x"}])
    assert paid == []


# ---------------------------------------------------------------------------
# The owner's purchases (invoke_service, the credits handlers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_owner_buys_from_its_own_service(market, monkeypatch):
    """A local digest: the operator pays directly, and the request goes
    through the same gate a stranger's would."""
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "m1"})
    _, paid = _buyer_side(chain, record, monkeypatch)
    out = await server._handle_invoke_service(
        {"digest": record.digest,
         "args": {"messages": [{"role": "user", "content": "own"}]}}, "iv")
    assert out["ok"], out
    assert out["result"]["output"]["result"]["content"] == "re: own"
    receipt = out["result"]["receipt"]
    assert receipt["paid"] is True and receipt["amount"] == str(PRICE)
    assert receipt["recipient"] == record.test_listing["provider"]
    assert paid == [{"recipient": record.test_listing["provider"], "amount": PRICE}]
    # The gate verified the operator's own payment.
    assert chain.verified[-1]["payer"] == record.test_listing["provider"]


async def _async(value):
    return value


def _buyer_daemon(tmp_path, record, monkeypatch):
    """Another daemon whose operator (PAYER_KEY) buys ``record``, which its
    catalog knows."""
    from atn.service_catalog import ServiceCatalog

    rt2 = _make_runtime(tmp_path / "buyer", chain=True)
    rt2._config.rpb.private_key = PAYER_KEY
    buyer = WebSocketBridge(rt2)
    assert buyer._buyer_address() == _payer()
    cat = ServiceCatalog(rt2, None)
    monkeypatch.setattr(ServiceCatalog, "synced", property(lambda self: True))
    cat.entry_for = lambda d, p="": {"service_id": 7}
    cat.price_list = lambda d, p="": dict(record.price_list)
    cat.spec = lambda d: record.spec
    rt2.__dict__["service_catalog"] = cat
    monkeypatch.setattr(oc.OnChainService, "get_agent_endpoint",
                        lambda self, addr: _async(""))
    return rt2, buyer, cat


@pytest.mark.asyncio
async def test_the_owner_buys_from_another_daemon_on_credits_else_directly(
        tmp_path, market, monkeypatch):
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "m1"})
    m, paid = _buyer_side(chain, record, monkeypatch)
    rt2, buyer, _ = _buyer_daemon(tmp_path, record, monkeypatch)
    provider = record.test_listing["provider"]

    async def invoke(text):
        return await buyer._handle_invoke_service(
            {"digest": record.digest,
             "args": {"messages": [{"role": "user", "content": text}]}}, "iv")
    # No credits with the provider: paid directly.
    out = await invoke("one")
    assert out["ok"], out
    assert out["result"]["output"]["result"]["content"] == "re: one"
    assert paid == [{"recipient": provider, "amount": PRICE}]
    assert out["result"]["receipt"]["tx_hash"]
    # The owner opens credits: the next purchase sends a voucher, no payment.
    opened = await buyer._owner_credits(
        "open_credits", {"provider_address": provider, "amount": DEPOSIT})
    assert opened["ok"] and opened["result"]["credits"]["deposit"] == DEPOSIT
    out = await invoke("two")
    assert out["ok"] and len(paid) == 1 and m.signed == [VOUCHER]
    assert out["result"]["receipt"]["credits"]["used"] == PRICE
    listed = await buyer._owner_credits("list_credits", {})
    assert [(r["provider"], r["used"]) for r in listed["result"]["credits"]] == [
        (provider, PRICE)]
    closed = await buyer._owner_credits("close_credits", {"provider_address": provider})
    assert closed["ok"]
    assert (await buyer._owner_credits("list_credits", {}))["result"]["credits"] == []


@pytest.mark.asyncio
async def test_the_owner_pays_nothing_for_a_service_without_a_price_list(
        tmp_path, market, monkeypatch):
    rt, server, chain = market
    record = _register_service(rt, code=ANSWERS, settings={"model": "m1"})
    _, paid = _buyer_side(chain, record, monkeypatch)
    _, buyer, cat = _buyer_daemon(tmp_path, record, monkeypatch)
    cat.price_list = lambda d, p="": None
    out = await buyer._handle_invoke_service({"digest": record.digest, "args": {}}, "iv")
    assert out["ok"] is False and "no price list" in out["error"]
    assert paid == [] and _rows(rt) == []
