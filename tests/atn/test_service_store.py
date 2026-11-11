"""ServiceStore (atn/service_store.py, Decision 2026-11-12 in
docs/services_market.md): registering a service backed by a tool, its signed
price list, the request log and replay guard, and the credit ledger."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest
from eth_account import Account

from atn.events import EventBus
from atn.service_prices import signed_by_provider, validate_price_list
from atn.service_spec import signed_by_author
from atn.service_store import ServiceStore

KEY = "0x" + "55" * 32
ADDR = Account.from_key(KEY).address

ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}
SESSION = {"protocol": {"id": "openai-chat/1", "spec": "ab" * 32},
           "reports_usage": ["input_tokens"]}


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime

    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    # The owner's services are signed by the operator key.
    config.autonet.private_key = KEY
    return Runtime(EventBus(), data_dir=data_dir, config=config)


@pytest.fixture
def rt(tmp_path):
    return _make_runtime(tmp_path)


def _tool(rt, name="echo_tool", session=None):
    return rt.tool_store.register(
        name=name, description="Echo x back.", input_schema=SCHEMA,
        author="user", code=ECHO_CODE + (f"# {name}\n"), session=session)["digest"]


def _request_svc(rt, tool=None, **kw):
    args = dict(name="echo", description="Echo service.", mode="request",
                author="user", backing_tool=tool or _tool(rt),
                prices={"requests": 5}, voucher=50)
    args.update(kw)
    return rt.service_store.register(**args)


def _session_svc(rt, **kw):
    args = dict(name="chat", description="Chat.", mode="session", author="user",
                backing_tool=_tool(rt, "chat_server", session=SESSION),
                settings={"model": "qwen2.5:14b"},
                measures=[{"measure": "seconds", "quantity": 60},
                          {"measure": "input_tokens", "quantity": 1000}],
                prices={"seconds": 10, "input_tokens": 3}, voucher=100)
    args.update(kw)
    return rt.service_store.register(**args)


def _reload(rt):
    return ServiceStore(rt, rt._config.data_dir / "services")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_request_service_copies_the_tools_schema_and_is_signed(rt):
    tool = _tool(rt)
    res = _request_svc(rt, tool=tool)
    spec, pl = res["spec"], res["price_list"]
    assert len(res["digest"]) == 64 and spec["mode"] == "request"
    assert spec["input_schema"] == SCHEMA
    assert "measures" not in spec and "protocol" not in spec
    assert spec["author_pubkey"] == ADDR and signed_by_author(spec)
    assert pl["service"] == res["digest"] and pl["provider"] == ADDR
    assert pl["prices"] == {"requests": 5} and pl["voucher"] == 50 and pl["seq"] == 1
    assert signed_by_provider(pl) and validate_price_list(pl, spec) == []
    record = rt.service_store.get(res["digest"])
    assert record.backing_tool == tool and record.mode == "request"
    assert record.prices == {"requests": 5} and record.voucher == 50


def test_backing_tool_by_name_and_explicit_schema(rt):
    tool = _tool(rt)
    explicit = {"type": "object", "properties": {"q": {"type": "string"}}}
    res = _request_svc(rt, tool="echo_tool", input_schema=explicit,
                       output_schema={"type": "object"})
    assert rt.service_store.get(res["digest"]).backing_tool == tool
    assert res["spec"]["input_schema"] == explicit
    assert res["spec"]["output_schema"] == {"type": "object"}


@pytest.mark.parametrize("backing, err", [
    ("", "name the tool"),
    ("f" * 64, "no tool"),
])
def test_a_service_needs_a_registered_tool(rt, backing, err):
    with pytest.raises(ValueError, match=err):
        _request_svc(rt, tool=None, backing_tool=backing)
    assert len(rt.service_store) == 0


def test_mode_must_match_the_tool(rt):
    session_tool = _tool(rt, "chat_server", session=SESSION)
    with pytest.raises(ValueError, match="serves sessions"):
        _request_svc(rt, tool=session_tool)
    with pytest.raises(ValueError, match="'session' block"):
        _session_svc(rt, backing_tool=_tool(rt))
    with pytest.raises(ValueError, match="mode"):
        _request_svc(rt, mode="stream")
    assert len(rt.service_store) == 0


def test_session_service_takes_the_tools_protocol(rt):
    res = _session_svc(rt)
    spec = res["spec"]
    assert spec["mode"] == "session" and spec["protocol"] == SESSION["protocol"]
    assert spec["settings"] == {"model": "qwen2.5:14b"}
    assert spec["input_schema"] == SCHEMA
    assert res["price_list"]["prices"] == {"seconds": 10, "input_tokens": 3}
    assert rt.service_store.get(res["digest"]).protocol == SESSION["protocol"]


@pytest.mark.parametrize("measure", ["output_tokens", " output_tokens "])
def test_token_measures_only_when_the_tool_reports_them(rt, measure):
    with pytest.raises(ValueError, match="does not report output_tokens"):
        _session_svc(rt, measures=[{"measure": measure, "quantity": 1000}],
                     prices={"output_tokens": 1})
    assert len(rt.service_store) == 0


def test_price_list_is_validated_at_registration(rt):
    with pytest.raises(ValueError, match="exactly"):
        _request_svc(rt, prices={"seconds": 5})
    with pytest.raises(ValueError, match="voucher size above 0"):
        _request_svc(rt, voucher=0)
    with pytest.raises(ValueError, match="whole ATN"):
        _request_svc(rt, prices={"requests": "five"})
    with pytest.raises(ValueError, match="exactly"):
        _session_svc(rt, prices={"seconds": 10})
    assert len(rt.service_store) == 0
    # A request service ignores measures: it is priced per request.
    res = _request_svc(rt, measures=[{"measure": "seconds", "quantity": 1}])
    assert "measures" not in res["spec"]


def test_set_prices_bumps_seq_and_keeps_the_spec(rt):
    res = _request_svc(rt)
    seen = []
    rt.service_store.on_prices = seen.append
    pl = rt.service_store.set_prices(res["digest"], {"requests": 7})
    assert (pl["seq"], pl["prices"], pl["voucher"]) == (2, {"requests": 7}, 50)
    assert pl["service"] == res["digest"] and signed_by_provider(pl)
    pl = rt.service_store.set_prices(res["digest"], {"requests": 8}, voucher=80)
    assert (pl["seq"], pl["voucher"]) == (3, 80)
    record = rt.service_store.get(res["digest"])
    assert record.spec == res["spec"] and record.price_list == pl
    assert [r.digest for r in seen] == [res["digest"]] * 2

    with pytest.raises(ValueError, match="exactly"):
        rt.service_store.set_prices(res["digest"], {"seconds": 1})
    with pytest.raises(ValueError, match="unknown service"):
        rt.service_store.set_prices("e" * 64, {"requests": 1})
    assert rt.service_store.get(res["digest"]).price_list == pl

    # A failing hook never fails the change.
    rt.service_store.on_prices = lambda r: 1 / 0
    assert rt.service_store.set_prices(res["digest"], {"requests": 9})["seq"] == 4


def test_service_number_hook_and_persistence(rt):
    res = _session_svc(rt)
    registered = []
    rt.service_store.on_registered = registered.append
    assert rt.service_store.set_service_id(res["digest"], 9)
    assert not rt.service_store.set_service_id("e" * 64, 1)
    assert [r.service_id for r in registered] == [9]
    pl = rt.service_store.set_prices(res["digest"], {"seconds": 1, "input_tokens": 1})

    reloaded = _reload(rt).get(res["digest"])
    assert reloaded.spec == res["spec"] and reloaded.service_id == 9
    assert reloaded.price_list == pl
    assert reloaded.backing_tool == rt.service_store.get(res["digest"]).backing_tool


def test_the_same_spec_again_keeps_its_number_and_sequence(rt, monkeypatch):
    import atn.service_store as ss
    monkeypatch.setattr(ss.time, "time", lambda: 1_800_000_000)
    first = _request_svc(rt)
    rt.service_store.set_service_id(first["digest"], 3)
    again = _request_svc(rt, prices={"requests": 6})
    assert again["digest"] == first["digest"]
    record = rt.service_store.get(first["digest"])
    assert record.service_id == 3 and record.price_list["seq"] == 2
    assert record.prices == {"requests": 6}


def test_resolve_retire_and_list(rt):
    res = _request_svc(rt, name="uniq")
    store = rt.service_store
    assert store.resolve("uniq").digest == res["digest"]
    assert store.resolve(f"svc_{res['digest'][:12]}").digest == res["digest"]
    assert store.resolve(res["digest"]).digest == res["digest"]
    assert store.retire(res["digest"]) and not store.retire("nope")
    assert store.list() == [] and len(store.list(include_retired=True)) == 1
    assert store.resolve("uniq") is None
    assert _reload(rt).get(res["digest"]).retired


@pytest.mark.asyncio
async def test_spec_for(rt):
    res = _request_svc(rt)
    store = rt.service_store
    assert await store.spec_for("0x" + res["digest"]) == res["spec"]
    other = json.dumps({"kind": "service_spec", "name": "x"}).encode()
    digest = hashlib.sha256(other).hexdigest()
    assert await store.spec_for(digest) is None          # nobody holds it

    async def wrong(_d):
        return b"{}"
    store.blob_fetcher = wrong
    assert await store.spec_for(digest) is None          # digest mismatch

    async def right(_d):
        return other
    store.blob_fetcher = right
    assert (await store.spec_for(digest))["name"] == "x"
    store.blob_fetcher = None
    assert (await store.spec_for(digest))["name"] == "x"  # kept


# ---------------------------------------------------------------------------
# Request log and replay guard
# ---------------------------------------------------------------------------


def test_request_log_summary_and_rotation(rt):
    res = _request_svc(rt)
    store = rt.service_store
    store.REQUEST_LOG_MAX_BYTES = 300
    for i in range(4):
        store.record_request(res["digest"], f"r{i}", f"0x{i:040X}", ok=i != 3,
                             amount="5")
    store.record_request(res["digest"], "free", "", ok=True)
    store.count_refused(res["digest"])
    store.count_refused("e" * 64)                        # not ours: ignored
    services = rt._config.data_dir / "services"
    assert (services / "requests.1.jsonl").exists()
    entry = store.summary()[res["digest"]]
    assert (entry["count"], entry["ok_count"], entry["unique_clients"],
            entry["refused"]) == (5, 4, 4, 1)
    assert entry["success_rate"] == 0.8 and entry["service_name"] == "echo"
    rows = [json.loads(line) for name in ("requests.1.jsonl", "requests.jsonl")
            for line in (services / name).read_text().splitlines()]
    assert [r["payer"] for r in rows][-2:] == [f"0x{3:040x}", ""]

    # The totals are kept, not re-read from the rows.
    (services / "requests.jsonl").unlink()
    (services / "requests.1.jsonl").unlink()
    store._persist_summary()
    reloaded = _reload(rt)
    assert reloaded.summary()[res["digest"]] == store.summary()[res["digest"]]
    assert reloaded._request_seq == 5


def test_replay_guard_ignores_prefix_and_case(rt):
    store = rt.service_store
    rid = "8e" * 32
    store.mark_request_served("0x" + rid)
    for spelling in (rid, rid.upper(), "0X" + rid.upper(), "0x" + rid):
        assert store.has_served_request(spelling)
    assert not store.has_served_request("ab" * 32)
    assert _reload(rt).has_served_request(rid.upper())


def test_used_up_ids_are_forgotten_past_the_horizon(rt):
    import time
    store = rt.service_store
    path = rt._config.data_dir / "services" / "served_requests.jsonl"
    store.mark_request_served("01" * 32)
    store.mark_request_served("01" * 32)                 # idempotent
    store.mark_request_served("02" * 32)
    assert len(path.read_text().splitlines()) == 2
    old = time.time() - store.REPLAY_HORIZON_S - 60
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "03" * 32, "ts": old}) + "\n")
    reloaded = _reload(rt)
    assert set(reloaded._seen_requests) == {"01" * 32, "02" * 32}
    assert len(path.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# Credit ledger
# ---------------------------------------------------------------------------


def _accept(store, cid, cum, sig="0xsig"):
    return store.accept_voucher(cid, cum, sig, client="0xC", provider="0xP")


def test_vouchers_only_raise_the_running_total(rt):
    store = rt.service_store
    assert store.channel_ledger(7) is None and store.balance(7) == (0, 0)
    assert _accept(store, 7, 100, "0xa")
    assert not _accept(store, "7", 100, "0xb")           # same total
    assert not _accept(store, 7, 50, "0xc")              # lower
    assert _accept(store, 7, 150, "0xd")
    ledger = store.channel_ledger(7)
    assert (ledger["cumulative"], ledger["signature"]) == (150, "0xd")
    assert (ledger["client"], ledger["provider"]) == ("0xc", "0xp")
    assert store.kept_channel_cumulative(7) == 150 and store.balance(7) == (0, 150)


def test_consume_refund_paid_up(rt):
    store = rt.service_store
    _accept(store, 1, 100)
    store.consume(1, 60)
    assert store.paid_up(1) and store.balance(1) == (60, 100)
    store.consume(1, 60)                                 # one reply overshoots
    assert not store.paid_up(1) and store.balance(1) == (120, 100)
    _accept(store, 1, 200)                               # a new voucher covers it
    assert store.paid_up(1) and store.balance(1) == (120, 200)
    store.refund(1, 50)
    assert store.balance(1) == (70, 200)
    store.refund(1, 500)                                 # never below zero
    assert store.balance(1) == (0, 200)
    store.consume(1, 0)
    store.consume(1, -5)
    store.refund(1, -5)
    assert store.balance(1) == (0, 200)
    store.consume(2, 10)                                 # no credits held: no-op
    store.refund(2, 10)
    assert store.channel_ledger(2) is None and not store.paid_up(2)
    store.forget_channel_voucher(1)                      # closed: nothing served
    assert not store.paid_up(1)


def test_in_use_counts(rt):
    store = rt.service_store
    assert not store.has_pending_voucher(3)
    store.begin_use(3)
    store.begin_use("3")
    store.end_use(3)
    assert store.has_pending_voucher(3)
    store.end_use(3)
    assert not store.has_pending_voucher(3)
    store.end_use(3)                                     # extra end: harmless
    assert not store.has_pending_voucher(3)


def test_ledger_persistence(rt):
    store = rt.service_store
    _accept(store, 4, 100)
    _accept(store, 5, 30)
    store.begin_use(4)
    store.consume(4, 40)
    store.end_use(4)                                     # flushes the charge
    assert _reload(rt).balance(4) == (40, 100)
    store.refund(4, 10)                                  # written at once
    reloaded = _reload(rt)
    assert reloaded.balance(4) == (30, 100) and reloaded.balance(5) == (0, 30)
    assert set(reloaded.channel_vouchers()) == {"4", "5"}
    assert not reloaded.has_pending_voucher(4)           # memory only
    store.forget_channel_voucher(5)
    assert set(_reload(rt).channel_vouchers()) == {"4"}


def test_ledger_is_thread_safe(rt):
    store = rt.service_store
    _accept(store, 9, 1)

    def charge():
        for _ in range(100):
            store.begin_use(9)
            store.consume(9, 1)
            store.end_use(9)

    def raise_total(start):
        for n in range(start, 200, 4):
            _accept(store, 9, n)

    threads = ([threading.Thread(target=charge) for _ in range(4)]
               + [threading.Thread(target=raise_total, args=(s,)) for s in range(4)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.balance(9) == (400, 199)
    assert not store.has_pending_voucher(9)
    assert _reload(rt).balance(9) == (400, 199)
