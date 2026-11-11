"""The Agents' service commands (atn/agent_tools.py, Decision 2026-11-12):
find_services, request_service (credits, else one direct payment),
open_session (on credits only when priced), the credits commands,
review_service and set_service_prices."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from eth_account import Account

import atn.on_chain as oc
from atn import agent_tools as at
from atn import service_client
from atn.service_catalog import spec_digest
from atn.service_channels import ChannelBook
from atn.service_review_store import ReviewStore
from nodes.common import service_reviews as sr
from tests.atn.test_service_catalog import ADDR_A, KEY_A, _catalog, _listed, _prices, _spec

BUYER_KEY = "0x" + "b7" * 32
BUYER = Account.from_key(BUYER_KEY).address
OPERATOR_KEY = "0x" + "0e" * 32
OPERATOR = Account.from_key(OPERATOR_KEY).address
JUR = "0xjurisdiction"
FIELDS = {"correctness": 0.5, "simplicity": 1, "novelty": 0, "malware": False}


class _Market:
    """ServiceMarketClient for the credits: one channel per open."""

    channel_available = True
    registry_available = False

    def __init__(self):
        self.opened, self.closed, self.topped = [], [], []
        self.state: dict[int, dict] = {}

    async def open_channel(self, key, provider, deposit):
        cid = len(self.opened) + 1
        self.opened.append((provider, deposit))
        self.state[cid] = {"status": 1, "deposit": deposit}
        return {"success": True, "channel_id": cid, "tx_hash": "0xo"}

    async def top_up_channel(self, key, cid, amount):
        self.topped.append((cid, amount))
        self.state[cid]["deposit"] += amount
        return {"success": True, "deposit": self.state[cid]["deposit"]}

    async def start_close(self, key, cid):
        self.closed.append(cid)
        self.state[cid]["status"] = 4
        return {"success": True, "tx_hash": "0xc"}

    async def get_channel(self, cid):
        return self.state.get(cid)

    def sign_voucher(self, key, cid, cumulative):
        return f"sig:{cid}:{cumulative}"


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A runtime with a catalog holding one request and one session service
    of ADDR_A, a buyer agent and the operator, and fakes for the chain and
    the provider."""
    cat = _catalog()
    req = _spec("transcribe", "speech to text")
    box = _spec("rent_a_box", "a shell on a fast machine", mode="session")
    _listed(cat, 1, req, pl=_prices(req))                  # 100 per request
    _listed(cat, 2, box, pl=_prices(box))                  # priced per second
    market = _Market()
    monkeypatch.setattr(oc, "ServiceMarketClient", lambda cfg: market)
    sent, paid = [], []

    async def request(config, provider, digest, payload, **kw):
        sent.append({"provider": provider, "digest": digest, **kw})
        reply = {"ok": True, "result": {"text": "hi"}, "delivered": True}
        if kw.get("voucher") or kw.get("channel_id") is not None:
            cid = (kw.get("voucher") or {}).get("channel_id") or kw["channel_id"]
            reply["credit"] = {"channel_id": cid, "consumed": 100,
                               "cumulative": (kw.get("voucher") or {}).get(
                                   "cumulative_amount", 0)}
        return reply

    async def pay(config, key, recipient, amount, request_id=""):
        paid.append((Account.from_key(key).address, recipient, amount))
        return {"tx_hash": "0xpaid", "request_id": request_id}

    monkeypatch.setattr(service_client, "request_service", request)
    monkeypatch.setattr(service_client, "pay_for_service", pay)
    keys = {"buyer": BUYER_KEY, "operator": OPERATOR_KEY}
    rt = SimpleNamespace(
        _config=SimpleNamespace(rpb=SimpleNamespace(network_joined=True)),
        registry=SimpleNamespace(get_agent_key=lambda aid: keys.get(aid, ""),
                                 daemon_operator_key=lambda: keys["operator"]),
        channel_book=ChannelBook(tmp_path / "channels.json"),
        service_catalog=cat)
    return SimpleNamespace(rt=rt, cat=cat, req=req, box=box, market=market,
                           sent=sent, paid=paid)


def _ref(spec, **kw):
    return {"spec_digest": spec_digest(spec), "provider_address": ADDR_A, **kw}


async def _call(d, name, args, caller="buyer"):
    return await getattr(at, f"_{name}")(d.rt, {"_caller_id": caller, **args})


@pytest.mark.asyncio
async def test_find_services_rows(daemon):
    daemon.cat._last_block = 1
    out = await _call(daemon, "find_services", {})
    rows = {r["service_id"]: r for r in out["services"]}
    assert out["count"] == 2
    req, box = rows[1], rows[2]
    assert req["mode"] == "request" and req["prices"] == {"requests": 100}
    assert req["measures"] == [{"measure": "requests", "quantity": 1}]
    assert req["voucher"] == 500 and req["protocol"] is None
    assert req["spec_available"] is True and req["your_credits"] is None
    assert box["mode"] == "session" and box["protocol"]["id"] == "ssh/2"
    daemon.rt.channel_book.set_open(BUYER, ADDR_A, 7, 1000)
    out = await _call(daemon, "find_services", {"query": "speech"})
    assert out["services"][0]["your_credits"] == {
        "channel_id": 7, "deposit": 1000, "used": 0, "paid_ahead": 0, "left": 1000}
    daemon.rt._config.rpb.network_joined = False
    assert (await _call(daemon, "find_services", {}))["status"] == "not_joined"


class TestRequestService:
    @pytest.mark.asyncio
    async def test_without_credits_one_direct_payment(self, daemon):
        out = await _call(daemon, "request_service", _ref(daemon.req, payload={"a": 1}))
        assert out["ok"] and out["paid_directly"]["tx_hash"] == "0xpaid"
        assert out["paid_directly"]["amount"] == 100 and out["paid_directly"]["request_id"]
        assert daemon.paid == [(BUYER, ADDR_A, 100)]
        assert daemon.sent[-1]["tx_hash"] == "0xpaid"
        assert daemon.sent[-1]["client_address"] == BUYER
        # Paying from credits needs credits.
        refused = await _call(daemon, "request_service",
                              _ref(daemon.req, payload={}, pay="credits"))
        assert "open_credits first" in refused["error"]

    @pytest.mark.asyncio
    async def test_with_credits_a_voucher_and_no_transaction(self, daemon):
        opened = await _call(daemon, "open_credits",
                             {"provider_address": ADDR_A, "deposit": 1000})
        assert opened["ok"] and opened["credits"]["deposit"] == 1000
        out = await _call(daemon, "request_service", _ref(daemon.req, payload={}))
        assert out["ok"] and daemon.paid == []
        # The voucher pays ahead by the provider's voucher size.
        assert daemon.sent[-1]["voucher"] == {
            "channel_id": 1, "cumulative_amount": 500, "signature": "sig:1:500"}
        assert out["credits"] == {"channel_id": 1, "deposit": 1000, "used": 100,
                                  "paid_ahead": 400, "left": 900}
        # pay='direct' still pays directly.
        direct = await _call(daemon, "request_service",
                             _ref(daemon.req, payload={}, pay="direct"))
        assert direct["paid_directly"]["amount"] == 100 and len(daemon.paid) == 1

    @pytest.mark.asyncio
    async def test_a_free_service_needs_no_key(self, daemon):
        free = _spec("echo", "free echo")
        _listed(daemon.cat, 3, free, pl=_prices(free, prices={"requests": 0}, voucher=0))
        out = await _call(daemon, "request_service", _ref(free, payload={}),
                          caller="nobody")
        assert out["ok"] and daemon.paid == []
        assert daemon.sent[-1]["payer_key"] == "" and "tx_hash" not in daemon.sent[-1]
        paid = await _call(daemon, "request_service", _ref(daemon.req, payload={}),
                           caller="nobody")
        assert "no key" in paid["error"]

    @pytest.mark.asyncio
    async def test_refusals(self, daemon):
        session = await _call(daemon, "request_service", _ref(daemon.box, payload={}))
        assert "use open_session" in session["error"]
        unpriced = _spec("new", "no price list here yet")
        daemon.cat._entries[4] = {**daemon.cat._entries[1], "service_id": 4,
                                  "spec_digest": spec_digest(unpriced)}
        out = await _call(daemon, "request_service", _ref(unpriced, payload={}))
        assert out["error"] == at._NO_PRICE_LIST
        assert "Name the service" in (await _call(
            daemon, "request_service", {"payload": {}}))["error"]
        # A service the catalog knows is retired is not paid for.
        daemon.cat._entries[1]["active"] = False
        retired = await _call(daemon, "request_service", _ref(daemon.req, payload={}))
        assert retired["error"] == "that service is retired"
        assert daemon.sent == [] and daemon.paid == []


class TestOpenSession:
    @pytest.fixture
    def opened(self, daemon, monkeypatch):
        calls = []

        async def open_local_session(runtime, **kw):
            calls.append(kw)
            return {"ok": True, "port": 4242}
        import atn.service_sessions as ss
        monkeypatch.setattr(ss, "open_local_session", open_local_session)
        return calls

    @pytest.mark.asyncio
    async def test_a_priced_session_needs_credits(self, daemon, opened):
        out = await _call(daemon, "open_session", _ref(daemon.box))
        assert "open_credits first" in out["error"] and opened == []
        daemon.rt.channel_book.set_open(BUYER, ADDR_A, 7, 1000)
        out = await _call(daemon, "open_session", _ref(daemon.box, args={"user": "me"}))
        assert out == {"ok": True, "port": 4242}
        assert opened[0]["payer"] == BUYER and opened[0]["payer_key"] == BUYER_KEY
        assert opened[0]["args"] == {"user": "me"}

    @pytest.mark.asyncio
    async def test_a_free_session_opens_without_credits(self, daemon, opened):
        free = _spec("free_box", "a shell for free", mode="session")
        _listed(daemon.cat, 3, free,
                pl=_prices(free, prices={"seconds": 0, "gigabytes": 0}, voucher=0))
        assert (await _call(daemon, "open_session", _ref(free)))["ok"]

    @pytest.mark.asyncio
    async def test_a_request_service_is_refused(self, daemon, opened):
        out = await _call(daemon, "open_session", _ref(daemon.req))
        assert "use request_service" in out["error"] and opened == []


@pytest.mark.asyncio
async def test_credits_open_top_up_list_close(daemon):
    args = {"provider_address": ADDR_A}
    assert "deposit above 0" in (await _call(daemon, "open_credits",
                                             {**args, "deposit": 0}))["error"]
    await _call(daemon, "open_credits", {**args, "deposit": 1000})
    again = await _call(daemon, "open_credits", {**args, "deposit": 5})
    assert "already open" in again["error"]
    topped = await _call(daemon, "top_up_credits", {**args, "amount": 250})
    assert topped["ok"] and topped["deposit"] == 1250
    listed = await _call(daemon, "list_credits", {})
    assert listed["credits"] == [{"provider": ADDR_A, "channel_id": 1,
                                  "deposit": 1250, "used": 0, "paid_ahead": 0,
                                  "left": 1250}]
    # The operator's credits are its own.
    assert (await _call(daemon, "list_credits", {}, caller=None))["credits"] == []
    closed = await _call(daemon, "close_credits", args)
    assert closed["ok"] and daemon.market.closed == [1]
    assert (await _call(daemon, "list_credits", {}))["credits"] == []


class TestReviewService:
    @pytest.fixture
    def reviews(self, daemon, tmp_path, monkeypatch):
        sent = []
        store = ReviewStore(tmp_path / "reviews")
        store.attach(JUR, sent.append)
        daemon.rt.service_reviews = store

        async def read_service(config, number):
            if number != 1:
                return {"error": f"no on-chain service #{number}"}
            return {"service_id": 1, "provider": ADDR_A, "active": False,
                    "spec_digest": "0x" + spec_digest(daemon.req)}
        monkeypatch.setattr(service_client, "read_service", read_service)
        return sent

    @pytest.mark.asyncio
    async def test_review_service_by_service_id(self, daemon, reviews):
        out = await _call(daemon, "review_service", {"service_id": 1, **FIELDS})
        assert out.get("ok"), out
        (event,) = reviews
        assert event["service"] == {"provider": ADDR_A.lower(),
                                    "spec_digest": spec_digest(daemon.req)}
        assert event["reviewer"] == BUYER.lower()
        assert sr.validate_event(event, jurisdiction=JUR, batch_ts=out["ts"])[0]
        # The owner surface reviews as the operator.
        await _call(daemon, "review_service", {"service_id": 1, **FIELDS}, caller=None)
        assert reviews[-1]["reviewer"] == OPERATOR.lower()

    @pytest.mark.asyncio
    async def test_the_service_must_be_named_consistently(self, daemon, reviews):
        other = await _call(daemon, "review_service",
                            {"service_id": 1, "spec_digest": "ee" * 32, **FIELDS})
        assert "another spec" in other["error"]
        assert "provided by" in (await _call(
            daemon, "review_service",
            {"service_id": 1, "provider_address": BUYER, **FIELDS}))["error"]
        assert "no on-chain service #9" in (await _call(
            daemon, "review_service", {"service_id": 9, **FIELDS}))["error"]
        assert "Name the service" in (await _call(daemon, "review_service",
                                                  FIELDS))["error"]
        assert "provider_address" in (await _call(
            daemon, "review_service", {"spec_digest": "ab" * 32, **FIELDS}))["error"]
        assert "number from find_services" in (await _call(
            daemon, "review_service", {"service_id": "ab" * 32, **FIELDS}))["error"]
        by_digest = await _call(daemon, "review_service", {**_ref(daemon.req), **FIELDS})
        assert by_digest.get("ok") and len(reviews) == 1


@pytest.mark.asyncio
async def test_set_service_prices_only_by_its_author(daemon):
    d = spec_digest(daemon.req)
    changed = []

    def set_prices(digest, prices, voucher):
        changed.append((digest, prices, voucher))
        return {"prices": prices, "voucher": voucher}
    daemon.rt.service_store = SimpleNamespace(
        get=lambda digest: SimpleNamespace(author="seller") if digest == d else None,
        set_prices=set_prices)
    out = await _call(daemon, "set_service_prices",
                      {"spec_digest": "0x" + d, "price": 70}, caller="seller")
    assert out["ok"] and changed == [(d, {"requests": 70}, None)]
    await _call(daemon, "set_service_prices",
                {"spec_digest": d, "prices": {"seconds": "5"}, "voucher": 50},
                caller="seller")
    assert changed[-1] == (d, {"seconds": 5}, 50)
    assert "only the service's author" in (await _call(
        daemon, "set_service_prices", {"spec_digest": d, "price": 1}))["error"]
    assert "name the prices" in (await _call(
        daemon, "set_service_prices", {"spec_digest": d}, caller="seller"))["error"]
    assert "no service" in (await _call(
        daemon, "set_service_prices", {"spec_digest": "ab" * 32, "price": 1},
        caller="seller"))["error"]


@pytest.mark.asyncio
async def test_register_service_publishes_under_the_agents_key(daemon, monkeypatch):
    import atn.service_publish as sp
    published = []

    async def publish(runtime, *, author, signing_key, form):
        published.append((author, signing_key, form))
        return {"digest": "ab" * 32, "spec": {}, "price_list": {"seq": 1},
                "on_chain": True, "service_id": 3, "tx_hash": "0xr"}
    monkeypatch.setattr(sp, "publish_service", publish)
    identity = SimpleNamespace(address=BUYER)
    daemon.rt.get_agent = lambda aid: SimpleNamespace(identity=identity)
    out = await _call(daemon, "register_service", {
        "mode": "request", "backing_tool": "cd" * 32, "name": "echo",
        "description": "d", "price": "25", "voucher": 100})
    assert out == {"service_id": 3, "tx_hash": "0xr", "spec_digest": "ab" * 32,
                   "provider": BUYER, "price_list": {"seq": 1}, "on_chain": True}
    author, key, form = published[0]
    assert (author, key) == ("buyer", BUYER_KEY)
    assert form["prices"] == {"requests": 25} and form["voucher"] == 100
    assert "prices" in (await _call(daemon, "register_service",
                                    {"mode": "request"}))["error"]
    owner = await _call(daemon, "register_service", {"mode": "request"}, caller=None)
    assert "called by an agent" in owner["error"] and len(published) == 1


def test_the_command_set():
    names = {t.name for t in at.get_tool_definitions()}
    assert {"find_services", "register_service", "set_service_prices",
            "request_service", "open_session", "open_credits", "top_up_credits",
            "close_credits", "list_credits", "review_service"} <= names
    assert not {"pay_for_service", "close_service_channel",
                "open_service_stream"} & names
    assert at._TOOL_CATEGORIES["services_buy"] == \
        at._TOOL_CATEGORIES["services"] - {"register_service", "set_service_prices"}
