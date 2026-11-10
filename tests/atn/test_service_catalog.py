"""ServiceCatalog (Decision 2026-11-10): services published and found the
way tools are. Specs arrive whole on gossip or by digest, and are kept only
when the chain names them; an entry counts only when its provider signed
the spec; the catalog follows the ServiceRegistry by enumeration and logs.
"""
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
    spec_digest,
    topic_for,
)
from atn.service_spec import canonical_service_bytes
from atn.tool_search import ToolSearchIndex
from atn.service_spec import service_embedding_text

KEY_A = "0x" + "a1" * 32
KEY_B = "0x" + "b2" * 32
ADDR_A = Account.from_key(KEY_A).address
ADDR_B = Account.from_key(KEY_B).address


def _spec(name: str, description: str, key: str = KEY_A, **extra) -> dict:
    spec = {
        "kind": "service_spec", "name": name, "description": description,
        "input_schema": {"type": "object", "properties": {"audio": {"type": "string"}}},
        "author": "agent", "author_pubkey": Account.from_key(key).address,
        "unit": "one audio minute",
        "endpoint_hint": "", "image_uri": "", "version_of": None,
        "created_ts": 1, **extra,
    }
    spec["author_sig"] = Account.sign_message(
        encode_defunct(canonical_service_bytes(spec)), private_key=key).signature.hex()
    return spec


class _Runtime:
    """What the catalog reads from a runtime: the config and the store."""

    def __init__(self, own=()):
        self._config = SimpleNamespace(rpb=SimpleNamespace())
        self.service_store = SimpleNamespace(
            list=lambda include_retired=False: [SimpleNamespace(digest=d) for d in own])


def _catalog(tmp_path: Path | None = None, own=()) -> ServiceCatalog:
    from atn.service_catalog import service_keyword_tokens
    index = ToolSearchIndex(None, embed=lambda texts: None,
                            text_of=service_embedding_text,
                            tokens_of=service_keyword_tokens)
    path = tmp_path / "services" / "catalog.json" if tmp_path else None
    return ServiceCatalog(_Runtime(own), path, index=index)


def _entry(sid: int, spec: dict, provider: str, ask: int = 100,
           active: bool = True) -> dict:
    return {"service_id": sid, "provider": provider,
            "spec_digest": spec_digest(spec), "ask_amount": ask,
            "active": active, "registered_at": 1, "updated_at": 1}


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

    def put(self, sid, provider, spec, ask, active=True):
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
    def test_kept_when_the_chain_names_the_spec(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text per minute")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        assert cat.accept_announcement(announcement(spec, 1)) is True
        assert cat.spec(spec_digest(spec)) == spec
        assert [r["service_id"] for r in cat.search("speech")] == [1]

    def test_held_until_the_chain_names_it(self, chain):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        assert cat.accept_announcement(announcement(spec, 1)) is False
        assert cat.spec(spec_digest(spec)) is None
        chain.reg.put(1, ADDR_A, spec, 100)
        assert cat.sync_chain()["adopted_specs"] == 1
        assert cat.spec(spec_digest(spec)) == spec

    def test_a_spec_with_a_price_is_refused(self):
        cat = _catalog()
        spec = _spec("priced", "d", ask={"amount": "5"})
        cat._entries[1] = _entry(1, spec, ADDR_A)
        assert cat.accept_announcement(announcement(spec, 1)) is False

    def test_oversized_or_malformed_announcements_are_dropped(self):
        cat = _catalog()
        assert cat.accept_announcement(b"{not json") is False
        assert cat.accept_announcement(b"x" * (64 * 1024 + 1)) is False
        assert cat.accept_announcement(json.dumps({"spec": {"kind": "tool"}}).encode()) is False

    def test_a_copy_registered_by_another_address_is_ignored(self):
        """Anyone can register a public spec's digest, at price 0: only the
        entry under the spec's signer counts."""
        cat = _catalog()
        spec = _spec("transcribe", "speech to text", key=KEY_A)
        cat._entries[1] = _entry(1, spec, ADDR_B, ask=0)      # the copy
        cat._entries[2] = _entry(2, spec, ADDR_A, ask=900)    # the provider's
        cat.accept_announcement(announcement(spec, 2))
        rows = cat.search("")
        assert [(r["service_id"], r["ask_amount"]) for r in rows] == [(2, 900)]
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
                "input_schema": {}, "author": "user", "author_pubkey": ADDR_A,
                "created_ts": 1}
        cat = _catalog(own=[spec_digest(spec)])
        cat._entries[1] = _entry(1, spec, ADDR_A)
        cat._specs[spec_digest(spec)] = spec
        assert [r["service_id"] for r in cat.search("")] == [1]
        assert cat.search("")[0]["owned"] is True

    def test_topic_is_per_jurisdiction(self):
        assert topic_for("1337:0xdao") == "/autonet/services/1337:0xdao"


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

        # Block 150: #1's price changes; #3 is registered.
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
        chain.reg.put(1, ADDR_A, spec, 100)
        cat.sync_chain()
        cat.accept_announcement(announcement(spec, 1))
        assert len(cat.search("")) == 1
        chain.reg.put(1, ADDR_A, spec, 100, active=False)
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
        chain.reg.put(1, ADDR_A, spec, 100)
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
    def test_relevance_order_with_chain_metadata(self):
        cat = _catalog()
        a = _spec("transcribe_audio", "speech to text for audio files")
        b = _spec("render_frames", "gpu rendering of video frames")
        cat._entries[1] = _entry(1, a, ADDR_A, ask=10)
        cat._entries[2] = _entry(2, b, ADDR_A, ask=20)
        for sid, sp in ((1, a), (2, b)):
            cat.accept_announcement(announcement(sp, sid))
        rows = cat.search("transcribe audio")
        assert rows[0]["service_id"] == 1
        assert rows[0]["ask"] == {"amount": "10", "unit": "per_item"}
        assert rows[0]["provider"] == ADDR_A
        assert [r["service_id"] for r in cat.search("")] == [2, 1]   # newest first

    def test_rows_show_the_unit_of_work_and_hardware(self):
        """What one work item is and what it runs on travel with every row
        (Decision 2026-11-10)."""
        cat = _catalog()
        gpu = _spec("render_frames", "video frame rendering",
                    unit="one minute of RTX 5090 time",
                    hardware="one RTX 5090, 32 GB")
        plain = _spec("transcribe_audio", "speech to text")
        cat._entries[1] = _entry(1, gpu, ADDR_A)
        cat._entries[2] = _entry(2, plain, ADDR_A)
        for sid, sp in ((1, gpu), (2, plain)):
            assert cat.accept_announcement(announcement(sp, sid))
        rows = {r["service_id"]: r for r in cat.search("")}
        assert rows[1]["unit"] == "one minute of RTX 5090 time"
        assert rows[1]["hardware"] == "one RTX 5090, 32 GB"
        assert rows[2]["unit"] == "one audio minute"
        assert rows[2]["hardware"] == ""

    @pytest.mark.asyncio
    async def test_find_services_passes_unit_and_hardware(self):
        from atn.agent_tools import _find_services
        cat = _catalog()
        spec = _spec("render_frames", "frames", unit="one rendered frame",
                     hardware="one H100")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        assert cat.accept_announcement(announcement(spec, 1))
        cat._last_block = 1
        rt = SimpleNamespace(
            _config=SimpleNamespace(rpb=SimpleNamespace(network_joined=True)),
            service_catalog=cat)
        out = await _find_services(rt, {})
        row = out["services"][0]
        assert row["unit"] == "one rendered frame"
        assert row["hardware"] == "one H100"

    def test_a_spec_without_a_unit_is_not_kept(self):
        cat = _catalog()
        spec = _spec("vague", "no unit of work")
        spec.pop("unit")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        assert cat.accept_announcement(announcement(spec, 1)) is False
        assert cat.search("") == []

class TestListingContext:
    def test_rows_show_the_providers_household_and_sanction(self):
        """Search shows each listing's household now and any sanction on it,
        from the daemon's owner map (no chain read per row)."""
        from atn.sanctions import SanctionList

        house = "0x" + "cc" * 20
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        cat.accept_announcement(announcement(spec, 1))
        sl = SanctionList(lambda: {f"sanction.{house}": "{}"},
                          {ADDR_A.lower(): house}.get)
        cat._runtime.autonet = SimpleNamespace(sanctions=sl)
        row = cat.search("speech")[0]
        assert row["provider_household"] == house
        assert row["sanction"]["household"] == house

    def test_no_chain_leaves_the_context_empty(self):
        cat = _catalog()
        spec = _spec("transcribe", "speech to text")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        cat.accept_announcement(announcement(spec, 1))
        row = cat.search("")[0]
        assert row["provider_household"] == "" and row["sanction"] is None


def test_hardware_and_unit_match_by_keyword():
    """With no search model loaded (keyword only), a query naming the
    hardware finds the service."""
    cat = _catalog()
    gpu = _spec("render_frames", "video frame rendering",
                unit="one minute of GPU time", hardware="one RTX 5090")
    plain = _spec("transcribe_audio", "speech to text")
    cat._entries[1] = _entry(1, gpu, ADDR_A)
    cat._entries[2] = _entry(2, plain, ADDR_A)
    for sid, sp in ((1, gpu), (2, plain)):
        cat.accept_announcement(announcement(sp, sid))
    assert [r["service_id"] for r in cat.search("rtx 5090")] == [1]

    def test_rows_carry_the_rating_of_their_provider_and_digest(self):
        """Service reviews (Decision 2026-11-10): the latest close's rating,
        looked up by (provider, digest), so a copy never shares one."""
        cat = _catalog()
        spec = _spec("transcribe_audio", "speech to text")
        cat._entries[1] = _entry(1, spec, ADDR_A)
        cat.accept_announcement(announcement(spec, 1))
        assert cat.search("")[0]["rating"] is None
        asked = []

        def lookup(provider, digest):
            asked.append((provider, digest))
            return {"payers": 2, "correctness": 0.5} if provider == ADDR_A else None
        cat.rating_provider = lookup
        row = cat.search("transcribe")[0]
        assert row["rating"] == {"payers": 2, "correctness": 0.5}
        assert asked[-1] == (ADDR_A, spec_digest(spec))

        def broken(provider, digest):
            raise RuntimeError("no driver")
        cat.rating_provider = broken
        assert cat.search("")[0]["rating"] is None
