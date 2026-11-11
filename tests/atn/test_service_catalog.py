"""ServiceCatalog (Decisions 2026-11-10 and 2026-11-12): services published
and found the way tools are. Specs arrive whole on gossip (with the
provider's signed price list) or by digest, and are kept only when the
chain names them; an entry counts only when its provider signed the spec;
a price list only when that provider signed it; the catalog follows the
ServiceRegistry by enumeration and logs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

import atn.on_chain as oc
from atn.service_catalog import (
    ServiceCatalog,
    announcement,
    fetch_missing,
    service_keyword_tokens,
    spec_digest,
    topic_for,
)
from atn.service_prices import build_price_list, sign_price_list
from atn.service_spec import canonical_service_bytes, service_embedding_text
from atn.tool_search import ToolSearchIndex

KEY_A = "0x" + "a1" * 32
KEY_B = "0x" + "b2" * 32
ADDR_A = Account.from_key(KEY_A).address
ADDR_B = Account.from_key(KEY_B).address
PROTOCOL = {"id": "ssh/2", "spec": "5e" * 32}


def _spec(name: str, description: str, key: str = KEY_A, mode: str = "request",
          **extra) -> dict:
    spec = {
        "kind": "service_spec", "name": name, "description": description,
        "mode": mode,
        "input_schema": {"type": "object", "properties": {"audio": {"type": "string"}}},
        "author": "agent", "author_pubkey": Account.from_key(key).address,
        "settings": {}, "image_uri": "", "version_of": None, "created_ts": 1,
    }
    if mode == "session":
        spec.update(measures=[{"measure": "seconds", "quantity": 60},
                              {"measure": "gigabytes", "quantity": 1}],
                    protocol=PROTOCOL)
    spec.update(extra)
    spec["author_sig"] = Account.sign_message(
        encode_defunct(canonical_service_bytes(spec)), private_key=key).signature.hex()
    return spec


def _prices(spec: dict, key: str = KEY_A, *, prices=None, voucher: int = 500,
            seq: int = 1, provider: str = "") -> dict:
    if prices is None:
        prices = ({"requests": 100} if spec["mode"] == "request"
                  else {"seconds": 3, "gigabytes": 40})
    pl = build_price_list(service=spec_digest(spec),
                          provider=provider or Account.from_key(key).address,
                          prices=prices, voucher=voucher, seq=seq, ts=seq)
    return sign_price_list(pl, key)


class _Runtime:
    """What the catalog reads from a runtime: the config and the store."""

    def __init__(self, own=()):
        self._config = SimpleNamespace(rpb=SimpleNamespace())
        records = {d: SimpleNamespace(digest=d, price_list=None) for d in own}
        self.service_store = SimpleNamespace(
            list=lambda include_retired=False: list(records.values()),
            get=lambda d: records.get(d))


def _catalog(tmp_path: Path | None = None, own=()) -> ServiceCatalog:
    index = ToolSearchIndex(None, embed=lambda texts: None,
                            text_of=service_embedding_text,
                            tokens_of=service_keyword_tokens)
    path = tmp_path / "services" / "catalog.json" if tmp_path else None
    return ServiceCatalog(_Runtime(own), path, index=index)


def _entry(sid: int, spec: dict, provider: str, ask: int = 0,
           active: bool = True) -> dict:
    return {"service_id": sid, "provider": provider,
            "spec_digest": spec_digest(spec), "ask_amount": ask,
            "active": active, "registered_at": 1, "updated_at": 1}


def _listed(cat: ServiceCatalog, sid: int, spec: dict, provider: str = ADDR_A,
            pl: dict | None = None, ask: int = 0) -> None:
    cat._entries[sid] = _entry(sid, spec, provider, ask)
    assert cat.accept_announcement(announcement(spec, sid, pl))


# ---------------------------------------------------------------------------
# A fake ServiceRegistry behind ServiceMarketClient
# ---------------------------------------------------------------------------

class _Call:
    def __init__(self, fn):
        self._fn = fn

    def call(self):
        return self._fn()


class _Registry:
    address = "0x" + "77" * 20

    def __init__(self):
        self.services: dict[int, tuple] = {}
        self.reads: list[int] = []
        outer = self

        class _F:
            def serviceCount(self_inner):
                return _Call(lambda: len(outer.services))

            def getService(self_inner, sid):
                outer.reads.append(int(sid))
                return _Call(lambda: outer.services.get(
                    int(sid), ("0x" + "00" * 20, b"\x00" * 32, 0, False, 0, 0)))
        self.functions = _F()

    def put(self, sid, provider, spec, ask=0, active=True):
        self.services[sid] = (provider, bytes.fromhex(spec_digest(spec)), ask,
                              active, 10, 10)


class _Eth:
    def __init__(self):
        self.block_number = 100
        self.logs: list[dict] = []
        self.refuse = False

    def get_logs(self, flt):
        if self.refuse:
            raise ValueError("logs refused")
        return [lg for lg in self.logs
                if flt["fromBlock"] <= lg["blockNumber"] <= flt["toBlock"]]


@pytest.fixture
def chain(monkeypatch):
    reg = _Registry()
    eth = _Eth()
    w3 = SimpleNamespace(eth=eth)
    monkeypatch.setattr(oc.ServiceMarketClient, "registry_available", True,
                        raising=False)
    monkeypatch.setattr(oc.ServiceMarketClient, "_get_web3", lambda self: w3)
    monkeypatch.setattr(oc.ServiceMarketClient, "_registry_contract",
                        lambda self, w3: reg)
    monkeypatch.setattr(oc.ServiceMarketClient, "__init__",
                        lambda self, config: None)
    return SimpleNamespace(reg=reg, eth=eth)


def _log(sid: int, block: int) -> dict:
    return {"blockNumber": block, "topics": [b"\x00" * 32, sid.to_bytes(32, "big")]}


# ---------------------------------------------------------------------------
# Gossip and genuineness
# ---------------------------------------------------------------------------

class TestAnnouncements:
    def test_an_announcement_carries_the_spec_and_its_price_list(self):
        spec = _spec("transcribe", "speech to text")
        pl = _prices(spec)
        assert json.loads(announcement(spec, 3, pl)) == {
            "v": 2, "service_id": 3, "spec": spec, "prices": pl}
        assert "prices" not in json.loads(announcement(spec, 3))

    def test_kept_when_the_chain_names_the_spec(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text per minute")
        _listed(cat, 1, spec, pl=_prices(spec))
        assert cat.spec(spec_digest(spec)) == spec
        assert cat.price_list(spec_digest(spec))["prices"] == {"requests": 100}
        assert [r["service_id"] for r in cat.search("speech")] == [1]

    def test_held_until_the_chain_names_it(self, chain):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        assert cat.accept_announcement(announcement(spec, 1, _prices(spec))) is False
        assert cat.spec(spec_digest(spec)) is None
        assert cat.price_list(spec_digest(spec)) is None
        chain.reg.put(1, ADDR_A, spec)
        assert cat.sync_chain()["adopted_specs"] == 1
        assert cat.spec(spec_digest(spec)) == spec
        # The price list that waited with it is kept too.
        assert cat.price_list(spec_digest(spec), ADDR_A)["voucher"] == 500

    def test_a_spec_with_removed_fields_is_refused(self):
        cat = _catalog()
        for extra in ({"unit": "one minute"}, {"endpoint_hint": ""},
                      {"ask": {"amount": "5"}}, {"hardware": "one H100"}):
            spec = _spec("old", "d", **extra)
            cat._entries[1] = _entry(1, spec, ADDR_A)
            assert cat.accept_announcement(announcement(spec, 1)) is False, extra

    def test_oversized_or_malformed_announcements_are_dropped(self):
        cat = _catalog()
        assert cat.accept_announcement(b"{not json") is False
        assert cat.accept_announcement(b"x" * (64 * 1024 + 1)) is False
        assert cat.accept_announcement(json.dumps({"spec": {"kind": "tool"}}).encode()) is False

    def test_a_copy_registered_by_another_address_is_ignored(self):
        """Anyone can register a public spec's digest: only the entry under
        the spec's signer counts."""
        cat = _catalog()
        spec = _spec("transcribe", "speech to text", key=KEY_A)
        cat._entries[1] = _entry(1, spec, ADDR_B)             # the copy
        _listed(cat, 2, spec, ask=900)                        # the provider's
        rows = cat.search("")
        assert [(r["service_id"], r["registry_price"]) for r in rows] == [(2, 900)]
        assert cat.entry_for(spec_digest(spec))["service_id"] == 2
        assert cat.entry_for(spec_digest(spec), ADDR_B) is None

    def test_a_spec_signed_by_someone_else_lists_nothing(self):
        cat = _catalog()
        spec = _spec("forged", "claims A, signed by B", key=KEY_B)
        spec["author_pubkey"] = ADDR_A   # breaks the signature
        cat._entries[1] = _entry(1, spec, ADDR_A)
        cat._specs[spec_digest(spec)] = spec
        assert cat.search("") == []

    def test_own_unsigned_spec_counts(self):
        """A dev daemon without keys publishes unsigned specs; its own
        still list."""
        spec = {"kind": "service_spec", "name": "local", "description": "d",
                "mode": "request", "input_schema": {"type": "object"},
                "author": "user", "author_pubkey": ADDR_A, "created_ts": 1}
        cat = _catalog(own=[spec_digest(spec)])
        cat._entries[1] = _entry(1, spec, ADDR_A)
        cat._specs[spec_digest(spec)] = spec
        rows = cat.search("")
        assert [r["service_id"] for r in rows] == [1] and rows[0]["owned"] is True

    def test_topic_is_per_jurisdiction(self):
        assert topic_for("1337:0xdao") == "/autonet/services/1337:0xdao"


# ---------------------------------------------------------------------------
# Price lists
# ---------------------------------------------------------------------------

class TestPriceLists:
    def test_kept_only_when_signed_by_the_registered_provider(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        _listed(cat, 1, spec)
        d = spec_digest(spec)
        # Naming the provider, signed by someone else.
        assert cat.note_prices(_prices(spec, KEY_B, provider=ADDR_A)) is False
        # Signed by its signer, for a copy entry that is not genuine.
        cat._entries[2] = _entry(2, spec, ADDR_B)
        assert cat.note_prices(_prices(spec, KEY_B)) is False
        assert cat.price_list(d) is None
        assert cat.note_prices(_prices(spec)) is True
        assert cat.price_list(d)["provider"] == ADDR_A
        assert cat.price_list(d, ADDR_B) is None

    def test_the_prices_must_name_the_specs_measures(self):
        cat = _catalog()
        spec = _spec("rent_a_box", "a shell on a machine", mode="session")
        _listed(cat, 1, spec)
        assert cat.note_prices(_prices(spec, prices={"requests": 5})) is False
        assert cat.note_prices(_prices(spec, prices={"seconds": 5})) is False
        assert cat.note_prices(_prices(spec)) is True
        # A priced list needs a voucher size.
        assert cat.note_prices(_prices(spec, voucher=0, seq=2)) is False

    def test_a_newer_sequence_number_wins(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        _listed(cat, 1, spec, pl=_prices(spec, prices={"requests": 100}, seq=5))
        d = spec_digest(spec)
        assert cat.note_prices(_prices(spec, prices={"requests": 1}, seq=4)) is False
        assert cat.price_list(d)["prices"] == {"requests": 100}
        # A prices-only announcement (no spec) changes them.
        msg = json.dumps({"v": 2, "service_id": 1,
                          "prices": _prices(spec, prices={"requests": 70}, seq=6)})
        assert cat.accept_announcement(msg.encode()) is True
        assert cat.price_list(d)["prices"] == {"requests": 70}

    def test_a_price_list_waits_for_its_spec(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        d = spec_digest(spec)
        cat._entries[1] = _entry(1, spec, ADDR_A)
        assert cat.note_prices(_prices(spec)) is False          # no spec yet
        # A forged list for the same service does not push it out.
        assert cat.note_prices(_prices(spec, KEY_B)) is False
        assert cat.note_spec(d, spec) is True
        assert cat.price_list(d, ADDR_A)["prices"] == {"requests": 100}

    def test_price_lists_survive_a_restart(self, tmp_path):
        cat = _catalog(tmp_path)
        spec = _spec("transcribe", "speech to text")
        _listed(cat, 1, spec, pl=_prices(spec, seq=3))
        again = _catalog(tmp_path)
        assert again.price_list(spec_digest(spec))["seq"] == 3
        assert again.search("speech")[0]["prices"] == {"requests": 100}

    def test_the_providers_own_services_answer_from_its_store(self):
        spec = _spec("transcribe", "speech to text")
        d = spec_digest(spec)
        cat = _catalog(own=[d])
        own = _prices(spec, seq=9)
        cat._runtime.service_store.get(d).price_list = own
        assert cat.price_list(d) == own


# ---------------------------------------------------------------------------
# Following the chain
# ---------------------------------------------------------------------------

class TestChainSync:
    def test_first_sync_enumerates_then_reads_only_new_and_changed(self, chain):
        cat = _catalog()
        s1, s2 = _spec("one", "first"), _spec("two", "second")
        chain.reg.put(1, ADDR_A, s1, 100)
        chain.reg.put(2, ADDR_A, s2, 200)
        out = cat.sync_chain()
        assert out["entries"] == 2 and cat.synced
        assert sorted(chain.reg.reads) == [1, 2]

        # Block 150: #1's registry price changes; #3 is registered.
        chain.reg.reads.clear()
        chain.reg.put(1, ADDR_A, s1, 150)
        s3 = _spec("three", "third")
        chain.reg.put(3, ADDR_A, s3, 300)
        chain.eth.block_number = 150
        chain.eth.logs.append(_log(1, 140))
        cat.sync_chain()
        assert sorted(chain.reg.reads) == [1, 3]       # #2 is not read again
        assert {e["service_id"]: e["ask_amount"] for e in cat.entries()} == \
            {1: 150, 2: 200, 3: 300}

    def test_retirement_comes_from_the_logs(self, chain):
        cat = _catalog()
        spec = _spec("one", "first")
        chain.reg.put(1, ADDR_A, spec)
        cat.sync_chain()
        cat.accept_announcement(announcement(spec, 1))
        assert len(cat.search("")) == 1
        chain.reg.put(1, ADDR_A, spec, active=False)
        chain.eth.block_number = 120
        chain.eth.logs.append(_log(1, 110))
        cat.sync_chain()
        assert cat.search("") == []
        assert len(cat.search("", include_inactive=True)) == 1

    def test_refused_logs_fall_back_to_reading_every_entry(self, chain):
        cat = _catalog()
        spec = _spec("one", "first")
        chain.reg.put(1, ADDR_A, spec, 100)
        cat.sync_chain()
        chain.reg.reads.clear()
        chain.reg.put(1, ADDR_A, spec, 999)
        chain.eth.block_number = 130
        chain.eth.refuse = True
        cat.sync_chain()
        assert chain.reg.reads == [1]
        assert cat.entries()[0]["ask_amount"] == 999

    def test_state_survives_a_restart(self, chain, tmp_path):
        cat = _catalog(tmp_path)
        spec = _spec("one", "speech to text")
        chain.reg.put(1, ADDR_A, spec)
        cat.sync_chain()
        cat.accept_announcement(announcement(spec, 1))
        again = _catalog(tmp_path)
        assert again.synced
        assert [r["service_id"] for r in again.search("speech")] == [1]


# ---------------------------------------------------------------------------
# Fetching specs no announcement brought
# ---------------------------------------------------------------------------

class TestFetch:
    @pytest.mark.asyncio
    async def test_missing_spec_is_fetched_and_retried(self, monkeypatch):
        import atn.service_catalog as sc
        now = [1000.0]
        monkeypatch.setattr(sc.time, "time", lambda: now[0])
        cat = _catalog()
        spec = _spec("one", "first")
        d = spec_digest(spec)
        cat._entries[1] = _entry(1, spec, ADDR_A)
        calls: list[str] = []

        async def offline(digest):
            calls.append(digest)
            return None
        assert await fetch_missing(cat, offline) == 0
        assert await fetch_missing(cat, offline) == 0     # leased: not again yet
        assert calls == [d]
        now[0] += sc.RETRY_BASE + 1

        async def online(digest):
            calls.append(digest)
            return spec
        assert await fetch_missing(cat, online) == 1
        assert cat.spec(d) == spec

    def test_a_fetched_spec_must_match_its_digest(self):
        cat = _catalog()
        spec = _spec("one", "first")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        other = _spec("two", "second")
        assert cat.note_spec(spec_digest(spec), other) is False
        assert cat.note_spec(spec_digest(spec), spec) is True

    def test_retired_entries_are_not_fetched(self):
        cat = _catalog()
        spec = _spec("one", "first")
        cat._entries[1] = _entry(1, spec, ADDR_A, active=False)
        assert cat.due_fetches() == []


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_rows_of_a_request_service(self):
        cat = _catalog()
        spec = _spec("chat_llama", "chat completions",
                     settings={"model": "llama3.1:70b-instruct"})
        _listed(cat, 1, spec, pl=_prices(spec), ask=90)
        row = cat.search("")[0]
        assert row["service_id"] == 1 and row["provider"] == ADDR_A
        assert row["mode"] == "request"
        assert row["settings"] == {"model": "llama3.1:70b-instruct"}
        assert row["measures"] == [{"measure": "requests", "quantity": 1}]
        assert row["prices"] == {"requests": 100} and row["voucher"] == 500
        assert row["price_list"]["sig"]
        assert row["registry_price"] == 90 and row["protocol"] is None
        assert row["input_schema"] == spec["input_schema"]
        for removed in ("unit", "hardware", "ask", "endpoint_hint"):
            assert removed not in row

    def test_rows_of_a_session_service(self):
        cat = _catalog()
        spec = _spec("rent_a_box", "a shell on a fast machine", mode="session")
        _listed(cat, 1, spec)
        row = cat.search("")[0]
        assert row["mode"] == "session" and row["protocol"] == PROTOCOL
        assert [m["measure"] for m in row["measures"]] == ["seconds", "gigabytes"]
        # Until its price list arrives a row shows no prices.
        assert row["prices"] == {} and row["voucher"] == 0 and row["price_list"] is None
        assert row["registry_price"] is None

    def test_relevance_order_and_newest_first_without_a_query(self):
        cat = _catalog()
        a = _spec("transcribe_audio", "speech to text for audio files")
        b = _spec("render_frames", "gpu rendering of video frames")
        _listed(cat, 1, a)
        _listed(cat, 2, b)
        assert cat.search("transcribe audio")[0]["service_id"] == 1
        assert [r["service_id"] for r in cat.search("")] == [2, 1]

    def test_settings_and_protocol_match_by_keyword(self):
        """With no search model loaded (keyword only), a query naming the
        model or the protocol finds the service."""
        cat = _catalog()
        llama = _spec("chat", "chat completions", settings={"model": "llama3"})
        box = _spec("rent_a_box", "a fast machine", mode="session")
        plain = _spec("transcribe_audio", "speech to text")
        for sid, sp in ((1, llama), (2, box), (3, plain)):
            _listed(cat, sid, sp)
        assert [r["service_id"] for r in cat.search("llama3")] == [1]
        assert [r["service_id"] for r in cat.search("ssh")] == [2]
        assert "llama3" in service_keyword_tokens(llama)
        assert "ssh" in service_keyword_tokens(box)

    def test_rows_carry_the_rating_of_their_provider_and_digest(self):
        """Service reviews: the latest close's rating, looked up by
        (provider, digest), so a copy never shares one."""
        cat = _catalog()
        spec = _spec("transcribe_audio", "speech to text")
        _listed(cat, 1, spec)
        assert cat.search("")[0]["rating"] is None
        asked = []

        def lookup(provider, digest):
            asked.append((provider, digest))
            return {"payers": 2, "correctness": 0.5} if provider == ADDR_A else None
        cat.rating_provider = lookup
        assert cat.search("transcribe")[0]["rating"] == {"payers": 2, "correctness": 0.5}
        assert asked[-1] == (ADDR_A, spec_digest(spec))

        def broken(provider, digest):
            raise RuntimeError("no driver")
        cat.rating_provider = broken
        assert cat.search("")[0]["rating"] is None


class TestListingContext:
    def test_rows_show_the_providers_household_and_sanction(self):
        """Search shows each listing's household now and any sanction on it,
        from the daemon's owner map (no chain read per row)."""
        from atn.sanctions import SanctionList

        house = "0x" + "cc" * 20
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        _listed(cat, 1, spec)
        sl = SanctionList(lambda: {f"sanction.{house}": "{}"},
                          {ADDR_A.lower(): house}.get)
        cat._runtime.autonet = SimpleNamespace(sanctions=sl)
        row = cat.search("speech")[0]
        assert row["provider_household"] == house
        assert row["sanction"]["household"] == house

    def test_no_chain_leaves_the_context_empty(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        _listed(cat, 1, spec)
        row = cat.search("")[0]
        assert row["provider_household"] == "" and row["sanction"] is None
