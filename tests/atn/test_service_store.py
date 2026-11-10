"""ServiceStore + services-market WS rail — provider-side coverage.

Design: docs/services_market.md. Covers spec build/validation, register +
reload persistence, the on-chain service number, retire, the
provider-side service_request dispatch end-to-end through a registered
echo tool, and the request log + summary the reviews later ride on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.agent_tools import execute_tool
from atn.service_spec import (
    build_service_spec,
    canonical_service_bytes,
    validate_service_spec,
)
from atn.service_store import ServiceStore


@pytest.fixture(autouse=True)
def _chain_listing(monkeypatch):
    """Each service's ServiceRegistry entry, the only reference for its price,
    its active flag and its payee (Decision 2026-11-10). By default: listed
    under the address that signs for its author, at ASK, active until
    retired. A test sets ``record.test_listing`` to describe another."""
    from atn.ws_server import WebSocketBridge

    async def _chain(self, record):
        listing = getattr(record, "test_listing", None)
        if listing is not None:
            return dict(listing)
        return {"service_id": 1,
                "provider": self._service_provider_address(record.author),
                "ask_amount": int(ASK["amount"]),
                "active": not record.retired}
    monkeypatch.setattr(WebSocketBridge, "_chain_service", _chain)


async def _stub_catalog(rt, monkeypatch, listings):
    """The daemon's service catalog holding ``listings`` as synced registry
    entries, each read fresh by ``service_client.read_service``."""
    from atn import service_client
    from atn.service_catalog import ServiceCatalog

    cat = ServiceCatalog(rt, None)
    by_id = {}
    for s in listings:
        entry = {"service_id": int(s["service_id"]),
                 "provider": s["provider"],
                 "spec_digest": str(s["spec_digest"]).lower().removeprefix("0x"),
                 "ask_amount": int(s.get("ask_amount") or 0),
                 "active": bool(s.get("active"))}
        cat._entries[entry["service_id"]] = entry
        cat._specs[entry["spec_digest"]] = {
            "kind": "service_spec", "name": f"svc{entry['service_id']}",
            "description": "d", "input_schema": SCHEMA, "author": "a",
            "author_pubkey": entry["provider"]}
        by_id[entry["service_id"]] = entry
    cat._last_block = 1
    # The stub specs stand for ones signed by their providers.
    import atn.service_catalog as sc
    monkeypatch.setattr(sc, "signed_by_author", lambda spec: True)
    rt.__dict__["service_catalog"] = cat

    async def _read(config, service_id):
        e = by_id.get(int(service_id))
        return dict(e) if e else {"error": f"no on-chain service #{service_id}"}
    monkeypatch.setattr(service_client, "read_service", _read)
    return cat


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
    return Runtime(EventBus(), data_dir=data_dir, config=config)


async def _register_agent(rt, agent_id, parent_id=None):
    defn = AgentDefinition(
        id=agent_id,
        name=agent_id,
        mode=AgentMode.COGNITIVE,
        system_prompt=f"You are {agent_id}.",
        cognitive_model="claude-sonnet-5",
        parent_id=parent_id,
    )
    await rt.register_agent(defn)
    return defn


ECHO_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'echo': args.get('x')}))\n"
)

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}

ASK = {"amount": "1000000", "unit": "per_item"}
# What one work item is (required in every spec, Decision 2026-11-10).
UNIT = "one request"


async def _register_echo_tool(rt, caller_id="child", name="echo_tool"):
    return await execute_tool(
        "register_tool",
        {"name": name, "description": "Echo x back.",
         "input_schema": SCHEMA, "code": ECHO_CODE},
        rt, caller_id=caller_id,
    )


# ---------------------------------------------------------------------------
# Spec build + validation
# ---------------------------------------------------------------------------


class TestServiceSpec:
    def test_build_valid(self):
        spec = build_service_spec(
            name="transcribe", description="Speech to text.",
            input_schema=SCHEMA, author="agent-1", unit=UNIT, created_ts=7)
        assert spec["kind"] == "service_spec"
        assert spec["name"] == "transcribe"
        assert "ask" not in spec      # the price lives on chain only
        assert spec["version_of"] is None
        assert spec["created_ts"] == 7
        assert validate_service_spec(spec) == []

    def test_missing_fields_rejected(self):
        errors = validate_service_spec({"kind": "service_spec"})
        assert any("name" in e for e in errors)
        assert any("description" in e for e in errors)
        assert any("input_schema" in e for e in errors)

    def test_spec_refuses_a_price(self):
        """The chain is the only reference for a service's price (Decision
        2026-11-10): a spec that names one is invalid."""
        errors = validate_service_spec({
            "kind": "service_spec", "name": "n", "description": "d",
            "input_schema": SCHEMA, "author": "a",
            "ask": {"amount": "5", "unit": "per_item"}})
        assert any("on chain" in e for e in errors)

    def test_unit_is_required_and_hardware_optional(self):
        """The spec declares what one work item is (the meaning of the
        on-chain price, Decision 2026-11-10); hardware is optional and left
        out when empty."""
        with pytest.raises(ValueError, match="unit"):
            build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", created_ts=1)
        with pytest.raises(ValueError, match="unit"):
            build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", unit="   ", created_ts=1)
        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA, author="a",
            unit="  one minute of RTX 5090 time ", created_ts=1)
        assert spec["unit"] == "one minute of RTX 5090 time"
        assert "hardware" not in spec
        gpu = build_service_spec(
            name="s", description="d", input_schema=SCHEMA, author="a",
            unit=UNIT, hardware=" one RTX 5090, 32 GB ", created_ts=1)
        assert gpu["hardware"] == "one RTX 5090, 32 GB"
        assert validate_service_spec(gpu) == []

    def test_unit_and_hardware_are_bounded_texts(self):
        from atn.service_spec import MAX_HARDWARE_CHARS, MAX_UNIT_CHARS
        base = {"kind": "service_spec", "name": "n", "description": "d",
                "input_schema": SCHEMA, "author": "a"}
        assert any("unit" in e for e in validate_service_spec(dict(base)))
        assert any("unit" in e for e in validate_service_spec(
            {**base, "unit": 5}))
        assert any("unit" in e for e in validate_service_spec(
            {**base, "unit": "x" * (MAX_UNIT_CHARS + 1)}))
        assert any("hardware" in e for e in validate_service_spec(
            {**base, "unit": UNIT, "hardware": ""}))
        assert any("hardware" in e for e in validate_service_spec(
            {**base, "unit": UNIT, "hardware": "x" * (MAX_HARDWARE_CHARS + 1)}))
        assert validate_service_spec({**base, "unit": UNIT}) == []

    def test_unit_and_hardware_are_signed_and_embedded(self):
        from atn.service_spec import service_embedding_text
        a = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", unit="one minute of GPU time",
                               created_ts=1)
        b = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", unit="one hour of GPU time",
                               hardware="one H100", created_ts=1)
        assert canonical_service_bytes(a) != canonical_service_bytes(b)
        assert "one minute of GPU time" in service_embedding_text(a)
        assert "one H100" in service_embedding_text(b)

    def test_image_uri_optional_and_signed_not_embedded(self):
        """image_uri is display-plane: covered by the signature (so it
        cannot be swapped after signing) but NOT by the embedding text
        (presentation is not semantics)."""
        from atn.service_spec import service_embedding_text

        bare = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", unit=UNIT, created_ts=1)
        assert bare["image_uri"] == ""            # absent -> empty, still valid
        assert validate_service_spec(bare) == []

        url = "https://example.invalid/services/s.jpg"
        withimg = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", unit=UNIT, image_uri=url, created_ts=1)
        assert withimg["image_uri"] == url
        assert validate_service_spec(withimg) == []
        # inside the signed payload
        assert url.encode() in canonical_service_bytes(withimg)
        assert canonical_service_bytes(withimg) != canonical_service_bytes(bare)
        # ...but not in the discovery text
        assert service_embedding_text(withimg) == service_embedding_text(bare)
        assert "jpg" not in service_embedding_text(withimg)

    def test_inference_signed_not_embedded(self):
        """The inference block is a commitment to the buyer (served model
        + token ceiling) so it is inside the signed bytes, but the
        semantics live in name/description — not in the embedding text."""
        from atn.service_spec import service_embedding_text

        bare = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", unit=UNIT, created_ts=1)
        assert "inference" not in bare

        from atn.service_spec import DEFAULT_MAX_INPUT_CHARS

        inf = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", unit=UNIT, created_ts=1,
            inference={"provider": "ollama", "model": "llama-3.1-8b",
                       "max_tokens_cap": 512})
        assert inf["inference"] == {"provider": "ollama",
                                    "model": "llama-3.1-8b",
                                    "max_tokens_cap": 512,
                                    "max_input_chars": DEFAULT_MAX_INPUT_CHARS}
        assert validate_service_spec(inf) == []
        assert b"llama-3.1-8b" in canonical_service_bytes(inf)
        assert canonical_service_bytes(inf) != canonical_service_bytes(bare)
        assert service_embedding_text(inf) == service_embedding_text(bare)
        assert "llama" not in service_embedding_text(inf)

    def test_inference_bounds_default_and_normalize(self):
        """The bounds are written into the spec when the author names none,
        so a buyer reads the limits it is held to."""
        from atn.service_spec import (
            DEFAULT_MAX_INPUT_CHARS, DEFAULT_MAX_TOKENS_CAP)

        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA, author="a", unit=UNIT,
            created_ts=1,
            inference={"provider": " openai ", "model": "  gpt-4o-mini  "})
        assert spec["inference"] == {"provider": "openai",
                                     "model": "gpt-4o-mini",
                                     "max_tokens_cap": DEFAULT_MAX_TOKENS_CAP,
                                     "max_input_chars": DEFAULT_MAX_INPUT_CHARS}

        # Extra keys are dropped: the shape is closed so the digest can't
        # drift on whatever a caller happened to pass.
        spec2 = build_service_spec(
            name="s", description="d", input_schema=SCHEMA, author="a", unit=UNIT,
            created_ts=1,
            inference={"provider": "ollama", "model": "m",
                       "max_tokens_cap": 8, "max_input_chars": 900,
                       "junk": True})
        assert spec2["inference"] == {"provider": "ollama", "model": "m",
                                      "max_tokens_cap": 8,
                                      "max_input_chars": 900}

    def test_bad_inference_rejected(self):
        from atn.service_spec import validate_inference

        ok = {"provider": "ollama", "model": "m"}
        assert validate_inference(ok) == []
        assert any("model" in e for e in validate_inference({"provider": "ollama"}))
        assert any("model" in e
                   for e in validate_inference({**ok, "model": 7}))
        assert any("max_tokens_cap" in e
                   for e in validate_inference({**ok, "max_tokens_cap": 0}))
        assert any("max_tokens_cap" in e
                   for e in validate_inference({**ok, "max_tokens_cap": "lots"}))
        assert any("max_input_chars" in e
                   for e in validate_inference({**ok, "max_input_chars": -5}))
        assert validate_inference("nope") == [
            "inference must be a dict of {provider, model, max_tokens_cap, "
            "max_input_chars}"]
        with pytest.raises(ValueError):
            build_service_spec(
                name="s", description="d", input_schema=SCHEMA, author="a", unit=UNIT,
                inference={"provider": "ollama", "max_tokens_cap": 10})

    def test_inference_names_its_upstream_provider(self):
        """No default provider: the upstream is named, never inferred from
        the model's name, and never one that is not this daemon's own
        stack."""
        from atn.service_spec import validate_inference

        missing = validate_inference({"model": "claude-sonnet-5"})
        assert any("inference.provider" in e for e in missing)
        assert any("inference.provider" in e
                   for e in validate_inference({"provider": "  ",
                                                "model": "llama3"}))
        for resold in ("service", "rpb", "jev"):
            errs = validate_inference({"provider": resold, "model": "m"})
            assert any("cannot serve" in e for e in errs), resold

    def test_canonical_bytes_excludes_sig(self):
        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", unit=UNIT, created_ts=1)
        base = canonical_service_bytes(spec)
        spec["author_sig"] = "0xdeadbeef"
        assert canonical_service_bytes(spec) == base  # sig excluded

    def test_canonical_bytes_deterministic_and_sorted(self):
        a = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", unit=UNIT, created_ts=1)
        b = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", unit=UNIT, created_ts=1)
        assert canonical_service_bytes(a) == canonical_service_bytes(b)


# ---------------------------------------------------------------------------
# Store: register, persist, lineage, retire
# ---------------------------------------------------------------------------


class TestServiceStore:
    @pytest.mark.asyncio
    async def test_register_and_reload(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        res = rt.service_store.register(
            name="svc", description="A service.", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool="deadbeef")
        assert len(res["digest"]) == 64
        assert res["spec"]["author"] == "child"

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        record = reloaded.get(res["digest"])
        assert record is not None
        assert record.name == "svc"
        assert record.backing_tool == "deadbeef"
        assert not record.retired

    @pytest.mark.asyncio
    async def test_service_number_survives_reload(self, tmp_path):
        """The ServiceRegistry number a registration returned is how the
        daemon finds its own entry (price, active flag, payee) on chain."""
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="numbered", description="d", input_schema=SCHEMA,
            author="user", unit=UNIT)
        assert rt.service_store.get(res["digest"]).service_id is None
        assert rt.service_store.set_service_id(res["digest"], 9)
        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.get(res["digest"]).service_id == 9

    def test_owner_services_sign_with_the_operator_key(self, tmp_path,
                                                       monkeypatch):
        """The owner surface registers, retires and is paid under the
        operator's address: its ServiceRegistry writes use the operator key
        even when no daemon key is configured."""
        rt = _make_runtime(tmp_path)
        rt._config.rpb.private_key = ""
        server = _server(rt)
        key = "0x" + "55" * 32
        monkeypatch.setattr(server, "_operator_key", lambda: key)
        assert server._service_signing_key("user") == key
        from eth_account import Account
        assert (server._service_provider_address("user")
                == Account.from_key(key).address)

    @pytest.mark.asyncio
    async def test_resolve_by_name_prefix_digest(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="uniq", description="d", input_schema=SCHEMA,
            author="user", unit=UNIT)
        store = rt.service_store
        assert store.resolve("uniq").digest == res["digest"]
        assert store.resolve(f"svc_{res['digest'][:12]}").digest == res["digest"]
        assert store.resolve(res["digest"]).digest == res["digest"]

    @pytest.mark.asyncio
    async def test_image_uri_round_trips_through_list(self, tmp_path):
        """Register with and without a banner image; both survive the
        store reload and appear on the list_services rows."""
        rt = _make_runtime(tmp_path)
        url = "https://example.invalid/services/with-image.jpg"
        withimg = rt.service_store.register(
            name="with_image", description="Has a banner.",
            input_schema=SCHEMA, author="user", unit=UNIT, image_uri=url)
        without = rt.service_store.register(
            name="no_image", description="No banner.",
            input_schema=SCHEMA, author="user", unit=UNIT)

        assert rt.service_store.get(withimg["digest"]).image_uri == url
        assert rt.service_store.get(without["digest"]).image_uri == ""

        # Survives a reload (it lives in the content-addressed spec blob).
        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.get(withimg["digest"]).image_uri == url
        assert reloaded.get(without["digest"]).image_uri == ""

        server = _server(rt)
        resp = await server._handle_message(
            {"type": "list_services", "msg_id": "m-img"}, _local_session())
        assert resp["ok"] is True
        rows = {r["name"]: r for r in resp["result"]["services"]}
        assert rows["with_image"]["image_uri"] == url
        assert rows["no_image"]["image_uri"] == ""

    @pytest.mark.asyncio
    async def test_register_inference_backed(self, tmp_path):
        """An inference-backed service round-trips through the signed spec
        blob, so the backing survives a reload with no sidecar state."""
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="cognition", description="Chat completions, per call.",
            input_schema=SCHEMA, author="user", unit=UNIT,
            inference={"provider": "ollama", "model": "qwen2.5:14b",
                       "max_tokens_cap": 1024, "max_input_chars": 5000})

        rec = rt.service_store.get(res["digest"])
        assert rec.inference == {"provider": "ollama", "model": "qwen2.5:14b",
                                 "max_tokens_cap": 1024,
                                 "max_input_chars": 5000}
        assert rec.max_tokens_cap == 1024
        assert rec.max_input_chars == 5000
        assert rec.backing_tool == ""

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.get(res["digest"]).inference["model"] == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_register_inference_cap_default(self, tmp_path):
        from atn.service_spec import DEFAULT_MAX_TOKENS_CAP

        from atn.service_spec import DEFAULT_MAX_INPUT_CHARS

        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="cognition", description="d", input_schema=SCHEMA,
            author="user", unit=UNIT, inference={"provider": "ollama", "model": "m"})
        rec = rt.service_store.get(res["digest"])
        assert rec.max_tokens_cap == DEFAULT_MAX_TOKENS_CAP
        assert rec.max_input_chars == DEFAULT_MAX_INPUT_CHARS

    @pytest.mark.asyncio
    async def test_inference_and_backing_tool_conflict(self, tmp_path):
        """One backing per service: a spec claiming both lies to the buyer
        about what they're paying for, so it's refused at registration."""
        rt = _make_runtime(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            rt.service_store.register(
                name="both", description="d", input_schema=SCHEMA,
                author="user", unit=UNIT, backing_tool="deadbeef",
                inference={"provider": "ollama", "model": "m",
                           "max_tokens_cap": 8})
        assert rt.service_store.list() == []

    @pytest.mark.asyncio
    async def test_bad_inference_rejected_at_register(self, tmp_path):
        rt = _make_runtime(tmp_path)
        with pytest.raises(ValueError, match="model"):
            rt.service_store.register(
                name="nomodel", description="d", input_schema=SCHEMA,
                author="user", unit=UNIT,
                inference={"provider": "ollama", "max_tokens_cap": 8})
        with pytest.raises(ValueError, match="max_tokens_cap"):
            rt.service_store.register(
                name="badcap", description="d", input_schema=SCHEMA,
                author="user", unit=UNIT,
                inference={"provider": "ollama", "model": "m",
                           "max_tokens_cap": -1})
        with pytest.raises(ValueError, match="inference.provider"):
            rt.service_store.register(
                name="noprovider", description="d", input_schema=SCHEMA,
                author="user", unit=UNIT, inference={"model": "claude-sonnet-5"})
        # A provider id this daemon does not know cannot serve anything.
        with pytest.raises(ValueError, match="unknown provider"):
            rt.service_store.register(
                name="nowhere", description="d", input_schema=SCHEMA,
                author="user", unit=UNIT,
                inference={"provider": "acme-llm", "model": "m"})
        assert rt.service_store.list() == []

    @pytest.mark.asyncio
    async def test_retire(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="svc", description="d", input_schema=SCHEMA,
            author="user", unit=UNIT)
        assert rt.service_store.retire(res["digest"])
        assert rt.service_store.get(res["digest"]).retired
        assert rt.service_store.list() == []
        assert len(rt.service_store.list(include_retired=True)) == 1
        assert not rt.service_store.retire("nope")

    def test_reserved_voucher_counts_but_is_kept_only_when_served(self, tmp_path):
        from atn.service_store import ServiceStore
        rt = _make_runtime(tmp_path)
        store = rt.service_store
        store.record_channel_voucher(3, 100, "0xkept", service_id=2, ask=100)
        r = store.reserve_channel_voucher(3, 200, "0xnext", service_id=2, ask=100)
        # Counts at once (a copy is refused), settles at the kept one.
        assert store.last_channel_cumulative(3) == 200
        assert store.kept_channel_cumulative(3) == 100
        assert store.has_pending_voucher(3)
        # Not persisted while pending: a restart forgets it.
        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.last_channel_cumulative(3) == 100
        store.release_channel_voucher(r)
        assert store.last_channel_cumulative(3) == 100
        assert not store.has_pending_voucher(3)
        r = store.reserve_channel_voucher(3, 200, "0xnext", service_id=2, ask=100)
        store.keep_channel_voucher(r)
        kept = store.get_channel_voucher(3)
        assert kept["cumulative"] == 200 and kept["signature"] == "0xnext"
        assert kept["service_id"] == 2 and kept["ask"] == 100
        # Releasing a handle twice, or after keeping, is a no-op.
        store.release_channel_voucher(r)
        assert store.kept_channel_cumulative(3) == 200


# ---------------------------------------------------------------------------
# Provider-side service_request dispatch (WS rail) + request log
# ---------------------------------------------------------------------------


def _server(rt):
    from atn.ws_server import WebSocketBridge
    return WebSocketBridge(rt)


def _local_session():
    from atn.ws_server import ClientSession
    return ClientSession(local=True, authed=True, owner=True,
                         root_agent_id=None, scope_ids=None)


async def _svc_request(rt, spec_digest, request_id, args, client="0xClient",
                       local=True):
    """Drive the provider-side dispatch.

    ``local=True`` models a request on the daemon's LOCAL listener, which is
    what the no-chain-config dev degrade covers. A REMOTE request with no
    chain config is refused by the gate (it cannot verify the payment), so
    these dispatch tests would otherwise never reach the dispatch they are
    about. See TestServicePaymentGateDegrade for the remote case."""
    server = _server(rt)
    return await server._handle_service_request(
        {"spec_digest": spec_digest, "request_id": request_id,
         "args": args, "client": client},
        msg_id="m1",
        local=local,
    )


class TestRegisterServiceWS:
    """The WS register_service surface accepts the optional image_uri."""

    async def _register(self, rt, msg):
        server = _server(rt)
        return await server._handle_message(
            {"type": "register_service", "msg_id": "r1", **msg},
            _local_session())

    @pytest.mark.asyncio
    async def test_register_with_and_without_image_uri(self, tmp_path):
        rt = _make_runtime(tmp_path)
        url = "https://example.invalid/services/a100-gpu-hours.jpg"
        base = {"description": "d", "input_schema": SCHEMA, "ask": ASK, "unit": UNIT}

        with_img = await self._register(
            rt, {"name": "with_img", "image_uri": url, **base})
        assert with_img["ok"] is True
        assert with_img["result"]["spec"]["image_uri"] == url

        # Omitted entirely -> empty string, not a crash.
        without = await self._register(rt, {"name": "without_img", **base})
        assert without["ok"] is True
        assert without["result"]["spec"]["image_uri"] == ""

        # Advisory: whitespace is stripped, nothing else is validated.
        loose = await self._register(
            rt, {"name": "loose", "image_uri": "  not-a-url  ", **base})
        assert loose["ok"] is True
        assert loose["result"]["spec"]["image_uri"] == "not-a-url"


class TestRegisterServiceUnit:
    """register_service (WS and agent tool) takes the unit of work
    (required) and the hardware (optional), and the store keeps them."""

    async def _ws(self, rt, msg):
        return await _server(rt)._handle_message(
            {"type": "register_service", "msg_id": "r1", **msg},
            _local_session())

    @pytest.mark.asyncio
    async def test_ws_requires_the_unit(self, tmp_path):
        rt = _make_runtime(tmp_path)
        base = {"name": "gpu", "description": "d", "input_schema": SCHEMA,
                "ask": ASK}
        missing = await self._ws(rt, base)
        assert missing["ok"] is False and "unit" in missing["error"]
        assert len(rt.service_store) == 0
        bad_hw = await self._ws(rt, {**base, "unit": UNIT, "hardware": 5})
        assert bad_hw["ok"] is False and "hardware" in bad_hw["error"]
        res = await self._ws(rt, {**base, "unit": "one minute of GPU time",
                                  "hardware": "one RTX 5090"})
        assert res["ok"] is True
        rec = rt.service_store.get(res["result"]["digest"])
        assert rec.unit == "one minute of GPU time"
        assert rec.hardware == "one RTX 5090"
        listed = await _server(rt)._handle_message(
            {"type": "list_services", "msg_id": "m"}, _local_session())
        row = listed["result"]["services"][0]
        assert row["unit"] == "one minute of GPU time"
        assert row["hardware"] == "one RTX 5090"

    @pytest.mark.asyncio
    async def test_agent_tool_requires_the_unit(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        base = {"name": "gpu", "description": "d", "ask_amount": 5,
                "input_schema": SCHEMA}
        missing = await execute_tool("register_service", dict(base), rt,
                                     caller_id="child")
        assert "unit" in missing["error"]
        assert len(rt.service_store) == 0
        # No registry here: the spec is kept locally and the chain leg fails.
        out = await execute_tool(
            "register_service", {**base, "unit": "one minute of GPU time",
                                 "hardware": "one RTX 5090"},
            rt, caller_id="child")
        rec = rt.service_store.get(out["spec_digest"])
        assert rec.unit == "one minute of GPU time"
        assert rec.hardware == "one RTX 5090"


class TestRegisterServiceWSDefaultSchema:
    """The app's Publish dialog sends a backing but no input_schema; the
    WS surface inherits the schema from the backing instead of refusing
    ("missing required field 'input_schema'", found by the live E2E)."""

    async def _register(self, rt, msg):
        server = _server(rt)
        return await server._handle_message(
            {"type": "register_service", "msg_id": "r1", **msg},
            _local_session())

    @pytest.mark.asyncio
    async def test_tool_backed_inherits_the_tool_schema(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        res = await self._register(rt, {
            "name": "from_app", "description": "d", "ask": ASK, "unit": UNIT,
            "backing_tool": tool["digest"], "tool_digest": tool["digest"]})
        assert res["ok"] is True, res
        assert res["result"]["spec"]["input_schema"] == SCHEMA

    @pytest.mark.asyncio
    async def test_inference_backed_gets_the_chat_schema(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = await self._register(rt, {
            "name": "cog", "description": "d", "ask": ASK, "unit": UNIT,
            "inference": {"provider": "ollama", "model": "echo-1"}})
        assert res["ok"] is True, res
        schema = res["result"]["spec"]["input_schema"]
        assert schema["required"] == ["messages"]

    @pytest.mark.asyncio
    async def test_explicit_schema_wins_and_unknown_tool_still_refused(
            self, tmp_path):
        rt = _make_runtime(tmp_path)
        explicit = {"type": "object", "properties": {"q": {"type": "string"}}}
        res = await self._register(rt, {
            "name": "explicit", "description": "d", "ask": ASK, "unit": UNIT,
            "input_schema": explicit, "backing_tool": "deadbeef"})
        assert res["ok"] is True
        assert res["result"]["spec"]["input_schema"] == explicit
        # No schema and a backing the daemon does not know: nothing to
        # inherit, so the validator refuses as before.
        res = await self._register(rt, {
            "name": "dead", "description": "d", "ask": ASK, "unit": UNIT,
            "backing_tool": "deadbeef"})
        assert res["ok"] is False
        assert "input_schema" in res["error"]


class TestServiceRequestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_through_backing_tool(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="Echo service.", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])

        out = await _svc_request(rt, svc["digest"], "req-1", {"x": "hi"})
        assert out["ok"] is True
        assert out["result"]["request_id"] == "req-1"
        assert out["result"]["result"] == {"echo": "hi"}

        # Provider request log recorded the served item. Nothing was paid
        # (the no-chain dev degrade), so nobody is named as its client.
        summary = rt.service_store.summary()
        entry = summary[svc["digest"]]
        assert entry["count"] == 1
        assert entry["ok_count"] == 1
        assert entry["success_rate"] == 1.0
        assert entry["unique_clients"] == 0
        assert entry["refused"] == 0

    @pytest.mark.asyncio
    async def test_unknown_and_retired_and_no_backing(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")

        miss = await _svc_request(rt, "f" * 64, "r", {"x": "1"})
        assert miss["ok"] is False and "Unknown" in miss["error"]

        # No backing tool -> refused before the payment gate.
        svc = rt.service_store.register(
            name="ghost", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT)
        out = await _svc_request(rt, svc["digest"], "r", {"x": "1"})
        assert out["ok"] is False and "backing" in out["error"]

        # Retired -> refused.
        rt.service_store.retire(svc["digest"])
        out = await _svc_request(rt, svc["digest"], "r2", {"x": "1"})
        assert out["ok"] is False and "retired" in out["error"].lower()

        # Neither was served: counted as refused, no request row.
        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 0 and entry["refused"] == 2

    @pytest.mark.asyncio
    async def test_request_log_survives_reload(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        await _svc_request(rt, svc["digest"], "req-1", {"x": "a"})
        await _svc_request(rt, svc["digest"], "req-2", {"x": "b"}, client="0xOther")

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded._request_seq == 2
        entry = reloaded.summary()[svc["digest"]]
        assert entry["count"] == 2
        # "0xOther" is only what the request claimed: never attributed.
        assert entry["unique_clients"] == 0

    @pytest.mark.asyncio
    async def test_only_the_verified_payer_is_attributed(self, tmp_path,
                                                         monkeypatch):
        """A paid request names its client by the payer_signature the gate
        verified, under the service's provider address; the request's own
        ``client`` field is ignored. Attempts the gate refuses are counted,
        never written as rows."""
        from eth_account import Account

        from atn import service_client
        from atn.ws_server import WebSocketBridge

        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        server = _server(rt)
        provider = server._service_provider_address("child")
        assert provider
        payer_key = "0x" + "71" * 32
        payer = Account.from_key(payer_key).address

        verdicts = iter([{"ok": True, "ask_amount": 5},
                         {"ok": False, "reason": "unpaid"}])

        async def _gate(self, request, record, *, local=False):
            return next(verdicts)
        monkeypatch.setattr(WebSocketBridge, "_validate_service_payment", _gate)

        rid = "0x" + "a1" * 32
        args = {"x": "hi"}
        paid = await server._handle_service_request(
            {"spec_digest": svc["digest"], "request_id": rid, "args": args,
             "client": "0xSomeoneElse",
             "payer_signature": service_client.sign_request(
                 payer_key, rid, svc["digest"], args, provider)},
            msg_id="m1")
        assert paid["ok"] is True
        refused = await server._handle_service_request(
            {"spec_digest": svc["digest"], "request_id": "0x" + "a2" * 32,
             "args": args, "client": "0xStranger"}, msg_id="m2")
        assert refused["ok"] is False

        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 1 and entry["unique_clients"] == 1
        assert entry["refused"] == 1
        rows = [json.loads(line) for line in
                (rt._config.data_dir / "services" / "requests.jsonl")
                .read_text().splitlines()]
        assert [r["payer"] for r in rows] == [payer.lower()]
        assert "0xStranger" not in json.dumps(rows)
        assert "0xSomeoneElse" not in json.dumps(rows)

    @pytest.mark.asyncio
    async def test_summary_is_kept_not_reread(self, tmp_path):
        """summary() reads no log: the totals are kept as requests are
        logged and persisted beside the rows, and the row file rotates."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        store = rt.service_store
        store.REQUEST_LOG_MAX_BYTES = 300
        for i in range(4):
            store.record_request(svc["digest"], f"r{i}", f"0x{i:040x}",
                                 ok=i != 3, amount="5")
        services = rt._config.data_dir / "services"
        assert (services / "requests.1.jsonl").exists()
        assert (services / "requests.jsonl").stat().st_size < 600
        (services / "requests.jsonl").unlink()
        (services / "requests.1.jsonl").unlink()
        entry = store.summary()[svc["digest"]]
        assert (entry["count"], entry["ok_count"], entry["unique_clients"]) \
            == (4, 3, 4)
        reloaded = ServiceStore(rt, services)
        assert reloaded.summary()[svc["digest"]] == entry
        assert reloaded._request_seq == 4

    @pytest.mark.asyncio
    async def test_replies_are_signed_by_the_paid_address(self, tmp_path):
        """Every reply about this daemon's service carries receipt_sig by
        the key that registered it; a reply altered on the way does not
        verify."""
        from atn import service_client

        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        provider = _server(rt)._service_provider_address("child")
        rid = "0x" + "b3" * 32

        out = await _svc_request(rt, svc["digest"], rid, {"x": "hi"})
        assert out["ok"] is True and out["receipt_sig"]
        assert service_client.reply_from_provider(
            json.loads(json.dumps(out)), rid, svc["digest"], provider)
        # Any spelling of the same request id verifies.
        assert service_client.reply_from_provider(
            out, rid[2:].upper(), svc["digest"], provider)
        forged = {**out, "result": {**out["result"], "result": {"echo": "no"}}}
        assert not service_client.reply_from_provider(
            forged, rid, svc["digest"], provider)
        assert not service_client.reply_from_provider(
            out, "0x" + "b4" * 32, svc["digest"], provider)
        assert not service_client.reply_from_provider(
            out, rid, svc["digest"], "0x" + "00" * 20)

        # Refusals are signed too, so a buyer can tell the provider's "no"
        # from a relay's.
        rt.service_store.retire(svc["digest"])
        no = await _svc_request(rt, svc["digest"], rid, {"x": "hi"})
        assert no["ok"] is False and service_client.reply_from_provider(
            no, rid, svc["digest"], provider)

    async def test_replay_guard_is_prefix_and_case_insensitive(self, tmp_path):
        """One payment must never buy two work items.

        Regression for the hole the cross-machine E2E found (2026-07-26): the
        seen-request set was keyed on the RAW string, and the two sides of the
        wire spell the same bytes32 differently — ``service_client`` emits
        ``0x<64 hex>``, the on-chain event decodes to bare hex. Flipping the
        prefix (or the case) walked straight past the replay guard, and the
        provider served a second real inference on one payment.
        """
        rt = _make_runtime(tmp_path)
        store = rt.service_store
        rid = "8ed3f25928f048e596d7bc5d2ba1c229b0fae15278fb4cdb8443a16bec548b9b"

        store.mark_request_served("0x" + rid)
        assert store.has_served_request("0x" + rid) is True
        # The same id, spelled the three other ways the wire produces.
        assert store.has_served_request(rid) is True
        assert store.has_served_request(rid.upper()) is True
        assert store.has_served_request("0X" + rid.upper()) is True
        # A genuinely different id is still free to be served.
        assert store.has_served_request("0x" + "ab" * 32) is False

        # And the normalization survives a reload — including a set persisted
        # by a pre-fix daemon that holds BOTH spellings of one id.
        import time as _t
        now = _t.time()
        (rt._config.data_dir / "services").mkdir(parents=True, exist_ok=True)
        (rt._config.data_dir / "services" / "served_requests.jsonl").write_text(
            json.dumps({"id": "0x" + rid, "ts": now}) + "\n"
            + json.dumps({"id": rid, "ts": now}) + "\n", encoding="utf-8")
        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert set(reloaded._seen_requests) == {rid}
        assert reloaded.has_served_request("0x" + rid.upper()) is True

    @pytest.mark.asyncio
    async def test_used_up_ids_are_appended_and_forgotten_past_the_horizon(
            self, tmp_path):
        """Marking an id appends one line (no rewrite of the whole set); an
        id older than the replay horizon is forgotten on load, because the
        gate refuses a payment that old anyway."""
        import time as _t
        rt = _make_runtime(tmp_path)
        store = rt.service_store
        path = rt._config.data_dir / "services" / "served_requests.jsonl"
        store.mark_request_served("0x" + "01" * 32)
        store.mark_request_served("0x" + "02" * 32)
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2
        old = _t.time() - store.REPLAY_HORIZON_S - 60
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": "03" * 32, "ts": old}) + "\n")
        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert set(reloaded._seen_requests) == {"01" * 32, "02" * 32}
        # The stale line is compacted away on load.
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


# ---------------------------------------------------------------------------
# Inference-backed dispatch (decision 2026-07-26)
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Stands in for whatever the daemon's provider stack resolves to.

    Records the send() kwargs so the test can assert the clamp and the
    model request, and returns a ProviderResponse-shaped object — the
    same type the sponsor rail's fulfillment path consumes."""

    def __init__(self, text="hello", model="served-model"):
        self._text = text
        self._model = model
        self.calls: list[dict] = []

    @property
    def name(self):
        return "fake"

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        from atn.providers.base import ProviderResponse, Usage
        return ProviderResponse(
            text=self._text, model=self._model, stop_reason="end_turn",
            usage=Usage(input_tokens=11, output_tokens=22),
        )


def _stub_provider_stack(rt, provider):
    """Point the daemon's fulfillment path at a fake provider.

    Patches the provider manager's resolve, the seam every agent resolves
    through, so the test exercises the real dispatch code. Records the
    (provider, model) each service asked for."""
    calls: list[tuple] = []

    def _resolve(defn):
        calls.append((defn.provider, defn.cognitive_model))
        return provider

    rt.providers.resolve_provider_with_fallback = _resolve
    return calls


def _spy_gate(monkeypatch):
    """Record each call to the payment gate (which still runs)."""
    from atn.ws_server import WebSocketBridge

    calls: list[str] = []
    real = WebSocketBridge._validate_service_payment

    async def _gate(self, request, record, *, local=False):
        calls.append(str(request.get("request_id")))
        return await real(self, request, record, local=local)
    monkeypatch.setattr(WebSocketBridge, "_validate_service_payment", _gate)
    return calls


def _inference_svc(rt, *, model="qwen2.5:14b", cap=100, name="cognition",
                   provider="ollama", max_input_chars=None):
    block = {"provider": provider, "model": model, "max_tokens_cap": cap}
    if max_input_chars is not None:
        block["max_input_chars"] = max_input_chars
    return rt.service_store.register(
        name=name, description="Chat completions, per call.",
        input_schema={"type": "object",
                      "properties": {"messages": {"type": "array"}}},
        author="user", unit=UNIT, inference=block)


class TestInferenceServiceDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_returns_completion(self, tmp_path):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        provider = _FakeProvider()
        resolved = _stub_provider_stack(rt, provider)

        out = await _svc_request(
            rt, svc["digest"], "req-1",
            {"messages": [{"role": "user", "content": "hi"}]})

        assert out["ok"] is True
        assert out["result"]["request_id"] == "req-1"
        assert out["result"]["content"] == "hello"
        assert out["result"]["model"] == "served-model"
        assert out["result"]["usage"] == {"input_tokens": 11,
                                          "output_tokens": 22}
        # The spec's declared provider and model are what we asked for.
        assert resolved[0] == ("ollama", "qwen2.5:14b")
        assert provider.calls[0]["model"] == "qwen2.5:14b"
        # No max_tokens named -> the cap is the default.
        assert provider.calls[0]["max_tokens"] == 100

        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 1 and entry["ok_count"] == 1

    @pytest.mark.asyncio
    async def test_max_tokens_clamped_to_cap(self, tmp_path):
        """The cap is what the ask was priced against, so a buyer asking
        for more than they paid for gets the cap, not an error."""
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt, cap=50)
        provider = _FakeProvider()
        _stub_provider_stack(rt, provider)

        msgs = [{"role": "user", "content": "hi"}]
        out = await _svc_request(rt, svc["digest"], "r-over",
                                 {"messages": msgs, "max_tokens": 999999})
        assert out["ok"] is True
        assert provider.calls[-1]["max_tokens"] == 50
        assert out["result"]["max_tokens"] == 50

        # Under the cap is honoured as asked.
        await _svc_request(rt, svc["digest"], "r-under",
                           {"messages": msgs, "max_tokens": 7})
        assert provider.calls[-1]["max_tokens"] == 7

        # Garbage / non-positive degrade to the cap rather than crashing.
        await _svc_request(rt, svc["digest"], "r-junk",
                           {"messages": msgs, "max_tokens": "lots"})
        assert provider.calls[-1]["max_tokens"] == 50
        await _svc_request(rt, svc["digest"], "r-zero",
                           {"messages": msgs, "max_tokens": 0})
        assert provider.calls[-1]["max_tokens"] == 1

    @pytest.mark.asyncio
    async def test_no_backing_tool_required(self, tmp_path):
        """The whole point: an inference-backed service has NO backing
        tool and must not be refused for lacking one."""
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        assert rt.service_store.get(svc["digest"]).backing_tool == ""
        _stub_provider_stack(rt, _FakeProvider())
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is True
        assert "backing" not in str(out.get("error", ""))

    @pytest.mark.asyncio
    async def test_missing_messages_refused_before_the_gate(self, tmp_path,
                                                            monkeypatch):
        """A malformed work item is refused before the payment gate, so its
        payment stays unspent."""
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        _stub_provider_stack(rt, _FakeProvider())
        gate = _spy_gate(monkeypatch)

        out = await _svc_request(rt, svc["digest"], "r-empty", {})
        assert out["ok"] is False
        assert "messages" in out["error"]
        assert "payment was not taken" in out["error"]

        out2 = await _svc_request(rt, svc["digest"], "r-blank",
                                  {"messages": []})
        assert out2["ok"] is False
        assert gate == []

        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 0 and entry["refused"] == 2

    @pytest.mark.asyncio
    async def test_oversized_request_refused_before_the_gate(self, tmp_path,
                                                             monkeypatch):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt, max_input_chars=200)
        provider = _FakeProvider()
        _stub_provider_stack(rt, provider)
        gate = _spy_gate(monkeypatch)

        big = [{"role": "user", "content": "x" * 300}]
        out = await _svc_request(rt, svc["digest"], "r-big", {"messages": big})
        assert out["ok"] is False and "too large" in out["error"]
        assert "200" in out["error"]
        # The system prompt counts toward the bound too.
        small = [{"role": "user", "content": "hi"}]
        out = await _svc_request(rt, svc["digest"], "r-sys",
                                 {"messages": small, "system": "s" * 250})
        assert out["ok"] is False and "too large" in out["error"]
        assert gate == [] and provider.calls == []

        ok = await _svc_request(rt, svc["digest"], "r-ok", {"messages": small})
        assert ok["ok"] is True and len(gate) == 1

    @pytest.mark.asyncio
    async def test_upstream_is_the_provider_the_spec_names(self, tmp_path):
        """The upstream is never guessed from the model's name: a model
        called like another vendor's still goes to the named provider."""
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt, provider="ollama", model="claude-lookalike")
        resolved = _stub_provider_stack(rt, _FakeProvider())
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is True
        assert resolved == [("ollama", "claude-lookalike")]

    @pytest.mark.asyncio
    async def test_upstream_not_buildable_refused_before_the_gate(
            self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt, provider="anthropic", model="m")
        gate = _spy_gate(monkeypatch)

        def _no_key(defn):
            raise RuntimeError("No Anthropic API key configured")
        rt.providers.resolve_provider_with_fallback = _no_key
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is False
        assert "'anthropic' is not available" in out["error"]
        assert gate == []

    @pytest.mark.asyncio
    async def test_no_provider_configured_fails_closed(self, tmp_path):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        _stub_provider_stack(rt, None)
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is False
        assert "not available" in out["error"]

    @pytest.mark.asyncio
    async def test_provider_exception_becomes_failed_request(self, tmp_path):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)

        class _Boom(_FakeProvider):
            async def send(self, **kwargs):
                raise RuntimeError("upstream 503")

        _stub_provider_stack(rt, _Boom())
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is False
        assert "upstream 503" in out["result"]["error"]
        assert rt.service_store.summary()[svc["digest"]]["ok_count"] == 0

    @pytest.mark.asyncio
    async def test_retired_inference_service_refused(self, tmp_path):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        _stub_provider_stack(rt, _FakeProvider())
        rt.service_store.retire(svc["digest"])
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is False and "retired" in out["error"].lower()


class TestRegisterInferenceServiceWS:
    async def _register(self, rt, msg):
        server = _server(rt)
        return await server._handle_message(
            {"type": "register_service", "msg_id": "r1", **msg},
            _local_session())

    @pytest.mark.asyncio
    async def test_register_and_list_surface_inference(self, tmp_path):
        rt = _make_runtime(tmp_path)
        base = {"description": "d", "input_schema": SCHEMA, "ask": ASK, "unit": UNIT}

        res = await self._register(rt, {
            "name": "cognition",
            "inference": {"provider": "ollama", "model": "qwen2.5:14b",
                          "max_tokens_cap": 256, "max_input_chars": 8000},
            **base})
        assert res["ok"] is True
        assert res["result"]["spec"]["inference"] == {
            "provider": "ollama", "model": "qwen2.5:14b",
            "max_tokens_cap": 256, "max_input_chars": 8000}

        # Without a named upstream provider there is nothing to sell.
        unnamed = await self._register(rt, {
            "name": "guess", "inference": {"model": "claude-sonnet-5"},
            **base})
        assert unnamed["ok"] is False
        assert "inference.provider" in unnamed["error"]

        tool_backed = await self._register(
            rt, {"name": "tooly", "tool_digest": "deadbeef", **base})
        assert tool_backed["ok"] is True

        server = _server(rt)
        listed = await server._handle_message(
            {"type": "list_services", "msg_id": "m"}, _local_session())
        rows = {r["name"]: r for r in listed["result"]["services"]}
        assert rows["cognition"]["inference"]["model"] == "qwen2.5:14b"
        assert rows["cognition"]["backing_tool"] == ""
        assert rows["tooly"]["inference"] == {}

    @pytest.mark.asyncio
    async def test_ws_rejects_both_backings(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = await self._register(rt, {
            "name": "both", "description": "d", "input_schema": SCHEMA,
            "ask": ASK, "unit": UNIT, "tool_digest": "deadbeef",
            "inference": {"provider": "ollama", "model": "m",
                          "max_tokens_cap": 8}})
        assert res["ok"] is False
        assert "mutually exclusive" in res["error"]

    @pytest.mark.asyncio
    async def test_ws_ignores_non_dict_inference(self, tmp_path):
        """A junk `inference` value is not a backing declaration — it must
        not turn a plain registration into a validation error."""
        rt = _make_runtime(tmp_path)
        res = await self._register(rt, {
            "name": "plain", "description": "d", "input_schema": SCHEMA,
            "ask": ASK, "unit": UNIT, "inference": "yes-please"})
        assert res["ok"] is True
        assert "inference" not in res["result"]["spec"]


# ---------------------------------------------------------------------------
# Consumer side: the owner buys one work item (invoke_service)
# ---------------------------------------------------------------------------

# The receipt keys the Flutter Services page parses. Asserted as an exact
# set so a field can never be quietly renamed or dropped underneath it.
RECEIPT_KEYS = {"paid", "degraded", "tx_hash", "amount", "token", "recipient"}


async def _invoke(rt, digest, args=None, server=None):
    server = server or _server(rt)
    return await server._handle_message(
        {"type": "invoke_service", "msg_id": "iv1", "digest": digest,
         **({"args": args} if args is not None else {})},
        _local_session())


class TestInvokeServiceLocal:
    """Local digest: pay (or degrade) then re-enter _handle_service_request."""

    @pytest.mark.asyncio
    async def test_tool_backed_happy_path_degraded(self, tmp_path):
        """No chain config -> the payment is skipped, the work is served,
        and the receipt SAYS SO (paid=false, degraded=true) rather than
        implying money moved."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="Echo service.", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])

        out = await _invoke(rt, svc["digest"], {"x": "hi"})
        assert out["ok"] is True
        result = out["result"]
        # `output` is the service's own result object verbatim — for a
        # tool-backed service that is tool_store.call's {"result": ...}
        # envelope; for an inference-backed one it is {content, model, ...}.
        # The handler does not reshape per backing.
        assert result["output"] == {"result": {"echo": "hi"}}
        # request_id is generated per invoke, not supplied by the caller.
        assert result["request_id"].startswith("0x")
        assert len(result["request_id"]) == 66

        receipt = result["receipt"]
        assert set(receipt) == RECEIPT_KEYS
        assert receipt["paid"] is False
        assert receipt["degraded"] is True
        assert receipt["tx_hash"] is None
        assert receipt["amount"] == ASK["amount"]
        assert receipt["token"] == "ATN"   # constant label; asks are ATN-only

        # It went through the PROVIDER path: the request landed on the log.
        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 1 and entry["ok_count"] == 1

    @pytest.mark.asyncio
    async def test_prepaid_proof_skips_daemon_payment(self, tmp_path):
        """Whoever purchases signs: a proof paid by the app wallet rides
        through untouched — the daemon pays nothing, the receipt carries
        the buyer's hash, and the frame reaches the gate with the
        buyer's request_id (which the gate would verify on-chain when a
        chain is configured)."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="Echo service.", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])

        server = _server(rt)
        rid = "0x" + "ab" * 32
        out = await server._handle_message(
            {"type": "invoke_service", "msg_id": "iv1",
             "digest": svc["digest"], "args": {"x": "hi"},
             "tx_hash": "0x" + "cd" * 32, "request_id": rid},
            _local_session())
        assert out["ok"] is True
        result = out["result"]
        assert result["request_id"] == rid
        receipt = result["receipt"]
        assert receipt["paid"] is True
        assert receipt["degraded"] is False
        assert receipt["tx_hash"] == "0x" + "cd" * 32

    @pytest.mark.asyncio
    async def test_prepaid_proof_requires_both_fields(self, tmp_path):
        """tx_hash without request_id (or vice versa) is a caller bug,
        refused before anything is resolved or spent."""
        rt = _make_runtime(tmp_path)
        server = _server(rt)
        out = await server._handle_message(
            {"type": "invoke_service", "msg_id": "iv1", "digest": "d" * 64,
             "tx_hash": "0x" + "cd" * 32},
            _local_session())
        assert out["ok"] is False
        assert "BOTH" in out["error"]

    @pytest.mark.asyncio
    async def test_inference_backed_happy_path(self, tmp_path):
        """An inference-backed purchase returns the completion envelope as
        `output`, with the transport's echoed request_id stripped out of it."""
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt, cap=64)
        provider = _FakeProvider()
        _stub_provider_stack(rt, provider)

        out = await _invoke(
            rt, svc["digest"],
            {"messages": [{"role": "user", "content": "hi"}]})
        assert out["ok"] is True
        output = out["result"]["output"]
        assert output["content"] == "hello"
        assert output["model"] == "served-model"
        assert output["max_tokens"] == 64
        assert "request_id" not in output          # lives at the top level
        assert out["result"]["request_id"]
        assert provider.calls[0]["max_tokens"] == 64
        assert out["result"]["receipt"]["degraded"] is True

    @pytest.mark.asyncio
    async def test_fresh_request_id_per_invoke(self, tmp_path):
        """Ids are per-invoke because the provider gate treats a reuse as a
        replay — two buys of the same service must both go through."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])

        first = await _invoke(rt, svc["digest"], {"x": "a"})
        second = await _invoke(rt, svc["digest"], {"x": "b"})
        assert first["ok"] and second["ok"]
        assert (first["result"]["request_id"]
                != second["result"]["request_id"])
        assert second["result"]["output"] == {"result": {"echo": "b"}}

    @pytest.mark.asyncio
    async def test_missing_and_bad_args(self, tmp_path):
        rt = _make_runtime(tmp_path)
        server = _server(rt)
        no_digest = await server._handle_message(
            {"type": "invoke_service", "msg_id": "m"}, _local_session())
        assert no_digest["ok"] is False
        assert "digest" in no_digest["error"]

        bad_args = await server._handle_message(
            {"type": "invoke_service", "msg_id": "m", "digest": "f" * 64,
             "args": "not-an-object"}, _local_session())
        assert bad_args["ok"] is False
        assert "'args' must be an object" in bad_args["error"]

    @pytest.mark.asyncio
    async def test_unknown_digest_without_chain(self, tmp_path):
        """A digest this daemon does not publish is FOREIGN. With no chain
        there is no registry to resolve its provider from, so it fails with
        that reason rather than a bare 'unknown'."""
        rt = _make_runtime(tmp_path)
        out = await _invoke(rt, "f" * 64, {"x": "1"})
        assert out["ok"] is False
        assert "Unknown service digest" in out["error"]
        assert "ServiceRegistry is not configured" in out["error"]

    @pytest.mark.asyncio
    async def test_retired_service_refused(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        rt.service_store.retire(svc["digest"])

        out = await _invoke(rt, svc["digest"], {"x": "1"})
        assert out["ok"] is False
        assert "retired" in out["error"].lower()
        # Refused BEFORE dispatch: nothing on the provider log.
        assert svc["digest"] not in rt.service_store.summary()

    @pytest.mark.asyncio
    async def test_no_backing_refused_before_spending(self, tmp_path):
        """An unfulfillable listing must be refused up front — taking the
        money and failing at dispatch is the one outcome to avoid."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        svc = rt.service_store.register(
            name="ghost", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT)

        out = await _invoke(rt, svc["digest"], {"x": "1"})
        assert out["ok"] is False
        assert "no backing implementation" in out["error"]
        assert svc["digest"] not in rt.service_store.summary()

    @pytest.mark.asyncio
    async def test_dispatch_error_carries_receipt(self, tmp_path):
        """When the work fails the buyer still gets the receipt: money may
        already have moved and hiding that would be dishonest."""
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)

        class _Boom(_FakeProvider):
            async def send(self, **kwargs):
                raise RuntimeError("upstream 503")

        _stub_provider_stack(rt, _Boom())
        out = await _invoke(rt, svc["digest"],
                            {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is False
        assert "upstream 503" in out["error"]
        assert set(out["result"]["receipt"]) == RECEIPT_KEYS
        assert out["result"]["request_id"]


class TestInvokeServicePayment:
    """Chain configured: the payment is real, and a failure ABORTS."""

    @staticmethod
    def _configure_chain(rt):
        rt._config.rpb.substrate_address = "0x" + "11" * 20
        rt._config.rpb.rpc_url = "http://localhost:8545"
        rt._config.rpb.private_key = "0x" + "22" * 32
        rt._config.rpb.owner_wallet = "0x" + "33" * 20

    @pytest.mark.asyncio
    async def test_payment_failure_aborts_before_dispatch(self, tmp_path,
                                                          monkeypatch):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        self._configure_chain(rt)

        from atn import service_client

        async def _fail(*a, **kw):
            return {"error": "insufficient ATN balance"}

        monkeypatch.setattr(service_client, "pay_for_service", _fail)

        out = await _invoke(rt, svc["digest"], {"x": "hi"})
        assert out["ok"] is False
        assert "Payment failed" in out["error"]
        assert "insufficient ATN balance" in out["error"]
        # ABORTED: the tool never ran, nothing is on the provider log.
        assert svc["digest"] not in rt.service_store.summary()

    @pytest.mark.asyncio
    async def test_paid_receipt_carries_tx_hash(self, tmp_path, monkeypatch):
        """With a successful payment the receipt is paid/not-degraded and the
        proof is threaded into the dispatch frame the gate verifies."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        self._configure_chain(rt)

        from atn import service_client
        seen: list[dict] = []

        async def _ok(config, key, recipient, amount, request_id=""):
            seen.append({"key": key, "recipient": recipient,
                         "amount": amount, "request_id": request_id})
            return {"tx_hash": "0xdeadbeef", "request_id": request_id}

        monkeypatch.setattr(service_client, "pay_for_service", _ok)

        server = _server(rt)
        gate_frames: list[dict] = []
        real_gate = server._validate_service_payment

        async def _spy_gate(request, record, local=False):
            gate_frames.append(dict(request))
            return {"ok": True, "reason": "stubbed"}

        server._validate_service_payment = _spy_gate

        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is True
        assert out["result"]["output"] == {"result": {"echo": "hi"}}
        receipt = out["result"]["receipt"]
        assert receipt["paid"] is True
        assert receipt["degraded"] is False
        assert receipt["tx_hash"] == "0xdeadbeef"
        assert receipt["amount"] == ASK["amount"]

        # Signed with the operator key (here the configured daemon key; not
        # the authoring agent's), for the ask, to the spec's author_pubkey —
        # the SAME recipient _validate_service_payment verifies the
        # ServicePayment event against.
        assert seen[0]["key"] == rt._config.rpb.private_key
        assert seen[0]["amount"] == int(ASK["amount"])
        assert seen[0]["recipient"] == rt.service_store.get(
            svc["digest"]).spec["author_pubkey"]
        assert seen[0]["recipient"].startswith("0x")
        assert receipt["recipient"] == seen[0]["recipient"]

        # The real gate would see the proof we just produced.
        assert gate_frames[0]["tx_hash"] == "0xdeadbeef"
        assert gate_frames[0]["request_id"] == out["result"]["request_id"]
        assert real_gate is not None

    @pytest.mark.asyncio
    async def test_owner_service_is_paid_at_its_registered_address(
            self, tmp_path, monkeypatch):
        """An owner-published service (author "user") is registered under
        the operator's address, and the chain names that address as the
        payee (Decision 2026-11-10), not the owner wallet."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="owner_svc", description="d", input_schema=SCHEMA,
            author="user", unit=UNIT, backing_tool=tool["digest"])
        # The spec names and is signed by the address it is registered
        # under: the operator's (the catalog's genuine-entry rule).
        from atn.service_spec import signed_by_author
        spec = rt.service_store.get(svc["digest"]).spec
        assert signed_by_author(spec)
        self._configure_chain(rt)

        from atn import service_client
        seen: list[str] = []

        async def _ok(config, key, recipient, amount, request_id=""):
            seen.append(recipient)
            return {"tx_hash": "0xfeed", "request_id": request_id}

        monkeypatch.setattr(service_client, "pay_for_service", _ok)
        server = _server(rt)

        async def _open_gate(request, record, local=False):
            return {"ok": True, "reason": "stubbed"}

        server._validate_service_payment = _open_gate

        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is True
        provider = server._service_provider_address("user")
        assert provider.startswith("0x")
        assert provider.lower() != rt._config.rpb.owner_wallet.lower()
        assert seen == [provider]
        assert out["result"]["receipt"]["recipient"] == provider

    @pytest.mark.asyncio
    async def test_owner_buys_as_the_operator_in_a_normal_install(
            self, tmp_path, monkeypatch):
        """No configured daemon key (a normal install): the owner surface
        pays with the vault-held operator key and buys as the operator's
        address, which also signs the request (Decision 2026-11-08)."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        self._configure_chain(rt)
        rt._config.rpb.private_key = ""
        operator_key = rt.registry.daemon_operator_key()
        from eth_account import Account
        operator = Account.from_key(operator_key).address

        from atn import service_client
        keys: list[str] = []

        async def _ok(config, key, recipient, amount, request_id=""):
            keys.append(key)
            return {"tx_hash": "0xbeef", "request_id": request_id}

        monkeypatch.setattr(service_client, "pay_for_service", _ok)
        server = _server(rt)
        frames: list[dict] = []

        async def _spy_gate(request, record, local=False):
            frames.append(dict(request))
            return {"ok": True, "reason": "stubbed", "proof": "direct"}

        server._validate_service_payment = _spy_gate
        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is True, out
        assert keys == [operator_key]
        assert frames[0]["client"] == operator
        signer = service_client.recover_request_signer(
            frames[0]["payer_signature"], frames[0]["request_id"],
            svc["digest"], {"x": "hi"}, out["result"]["receipt"]["recipient"])
        assert signer == operator

    @pytest.mark.asyncio
    async def test_no_operator_key_is_a_payment_failure(self, tmp_path,
                                                        monkeypatch):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        self._configure_chain(rt)
        rt._config.rpb.private_key = ""
        server = _server(rt)

        def _no_vault():
            raise RuntimeError("vault unavailable")
        monkeypatch.setattr(server, "_operator_key", _no_vault)

        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is False
        assert "no operator key" in out["error"]
        assert svc["digest"] not in rt.service_store.summary()

    @pytest.mark.asyncio
    async def test_owner_service_payer_and_real_gate_agree(
            self, tmp_path, monkeypatch):
        """The payer and the gate must resolve the SAME recipient: both read
        it from the service's on-chain entry. This runs the REAL gate, not
        a stub."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        self._configure_chain(rt)
        svc = rt.service_store.register(
            name="owner_svc", description="d", input_schema=SCHEMA,
            author="user", unit=UNIT, backing_tool=tool["digest"])
        server = _server(rt)
        provider = server._service_provider_address("user")
        assert provider.startswith("0x")

        from atn import on_chain, service_client
        seen: list[str] = []

        async def _ok(config, key, recipient, amount, request_id=""):
            seen.append(recipient)
            return {"tx_hash": "0xfeed", "request_id": request_id}

        verified: list[str] = []

        async def _verify(self, *, request_id, recipient, min_amount, tx_hash,
                          payer=None):
            verified.append(recipient)
            return {"verified": True}

        monkeypatch.setattr(service_client, "pay_for_service", _ok)
        monkeypatch.setattr(on_chain.OnChainService,
                            "verify_service_payment", _verify)

        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is True, out
        # Payer and gate agreed on one recipient: the on-chain provider.
        assert seen == [provider]
        assert verified == [provider]

    # ---- no chain config: local degrades open, REMOTE fails closed -------
    #
    # `service_request` is the only message dispatched PRE-AUTH on the
    # network-reachable listener, because the payment IS the credential. A
    # registry outage that left substrate_address empty used to turn that
    # into no credential at all: unlimited free work, inference included,
    # for any peer who knew the digest.

    @pytest.mark.asyncio
    async def test_remote_request_refused_without_chain_config(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        assert not rt._config.rpb.substrate_address

        out = await _svc_request(rt, svc["digest"], "req-remote",
                                 {"x": "hi"}, local=False)
        assert out["ok"] is False
        assert "no chain configuration" in out["error"]
        # The refusal is logged as a failed request, and nothing was served.
        assert rt.service_store.summary()[svc["digest"]]["ok_count"] == 0

    @pytest.mark.asyncio
    async def test_local_request_still_degrades_open(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])

        out = await _svc_request(rt, svc["digest"], "req-local",
                                 {"x": "hi"}, local=True)
        assert out["ok"] is True
        assert out["result"]["result"] == {"echo": "hi"}

    @pytest.mark.asyncio
    async def test_foreign_digest_resolves_provider_and_dials(
            self, tmp_path, monkeypatch):
        """A digest this daemon does not publish is resolved through the
        catalog to its registry entry, read fresh for its provider and ask,
        then bought cross-daemon."""
        rt = _make_runtime(tmp_path)
        self._configure_chain(rt)
        rt._config.rpb.service_registry_address = "0x" + "44" * 20
        foreign = "ab" * 32
        provider_addr = "0x" + "55" * 20

        from atn import on_chain, service_client

        async def _list(self):
            return [
                {"service_id": 1, "provider": "0x" + "66" * 20,
                 "spec_digest": "cd" * 32, "ask_amount": "5", "active": True},
                # Chain reads may return the 0x-prefixed form; the handler
                # compares in normalized space.
                {"service_id": 2, "provider": provider_addr,
                 "spec_digest": "0x" + foreign, "ask_amount": "777",
                 "active": True},
            ]

        await _stub_catalog(rt, monkeypatch, await _list(None))

        paid: list[tuple] = []

        async def _pay(config, key, recipient, amount, request_id=""):
            paid.append((recipient, amount))
            return {"tx_hash": "0xabc123", "request_id": request_id}

        async def _endpoint(self, addr):
            return f"wss://seller.invalid/{addr[:6]}"

        dialed: list[dict] = []

        async def _req(config, provider, digest, args, **kw):
            dialed.append({"provider": provider, "digest": digest,
                           "args": args, **kw})
            return {"ok": True, "result": {"request_id": kw.get("request_id"),
                                           "transcript": "done"}}

        monkeypatch.setattr(service_client, "pay_for_service", _pay)
        monkeypatch.setattr(on_chain.OnChainService, "get_agent_endpoint",
                            _endpoint)
        monkeypatch.setattr(service_client, "request_service", _req)

        out = await _invoke(rt, foreign, {"audio": "x"})
        assert out["ok"] is True
        assert out["result"]["output"] == {"transcript": "done"}
        # Paid the CHAIN ask to the CHAIN provider (not a local spec's).
        assert paid == [(provider_addr, 777)]
        receipt = out["result"]["receipt"]
        assert receipt["paid"] is True
        assert receipt["amount"] == "777"
        assert receipt["token"] == "ATN"          # chain asks are ATN-only
        assert receipt["recipient"] == provider_addr
        # The endpoint was resolved for us, so request_service doesn't
        # re-read the chain, and the payment proof rode along.
        assert dialed[0]["endpoint"] == f"wss://seller.invalid/{provider_addr[:6]}"
        assert dialed[0]["tx_hash"] == "0xabc123"
        assert dialed[0]["digest"] == foreign

    @pytest.mark.asyncio
    async def test_foreign_digest_not_on_chain_either(self, tmp_path,
                                                      monkeypatch):
        rt = _make_runtime(tmp_path)
        self._configure_chain(rt)
        rt._config.rpb.service_registry_address = "0x" + "44" * 20

        from atn import on_chain

        async def _list(self):
            return []

        await _stub_catalog(rt, monkeypatch, await _list(None))
        out = await _invoke(rt, "ab" * 32, {"x": "1"})
        assert out["ok"] is False
        assert "not registered on chain" in out["error"]

    @pytest.mark.asyncio
    async def test_foreign_digest_retired_on_chain(self, tmp_path, monkeypatch):
        rt = _make_runtime(tmp_path)
        self._configure_chain(rt)
        rt._config.rpb.service_registry_address = "0x" + "44" * 20
        foreign = "ab" * 32

        from atn import on_chain

        async def _list(self):
            return [{"service_id": 1, "provider": "0x" + "55" * 20,
                     "spec_digest": foreign, "ask_amount": "1",
                     "active": False}]

        await _stub_catalog(rt, monkeypatch, await _list(None))
        out = await _invoke(rt, foreign, {"x": "1"})
        assert out["ok"] is False
        assert "retired on chain" in out["error"]

    @pytest.mark.asyncio
    async def test_zero_ask_needs_no_payment(self, tmp_path, monkeypatch):
        """A giveaway is served without a tx (payForService rejects zero),
        but it is NOT a degrade — the chain is up, the price was zero."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="free_svc", description="d", input_schema=SCHEMA,
            author="child", unit=UNIT, backing_tool=tool["digest"])
        rt.service_store.get(svc["digest"]).test_listing = {
            "service_id": 1, "provider": "0x" + "44" * 20, "ask_amount": 0,
            "active": True}
        self._configure_chain(rt)

        from atn import service_client

        async def _never(*a, **kw):
            raise AssertionError("payment attempted for a zero ask")

        monkeypatch.setattr(service_client, "pay_for_service", _never)

        server = _server(rt)

        async def _open_gate(request, record, local=False):
            return {"ok": True, "reason": "stubbed"}

        server._validate_service_payment = _open_gate

        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is True
        assert out["result"]["output"] == {"result": {"echo": "hi"}}
        receipt = out["result"]["receipt"]
        assert receipt["tx_hash"] is None
        assert receipt["amount"] == "0"
        assert receipt["paid"] is False
        # NOT a degrade: the chain is up, the price was zero.
        assert receipt["degraded"] is False
