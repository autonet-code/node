"""ServiceStore + services-market WS rail — provider-side coverage.

Design: docs/services_market.md. Covers spec build/validation, register +
reload persistence, version_of lineage on update_ask, retire, the
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
            input_schema=SCHEMA, author="agent-1", ask=ASK, created_ts=7)
        assert spec["kind"] == "service_spec"
        assert spec["name"] == "transcribe"
        assert spec["ask"] == ASK
        assert spec["version_of"] is None
        assert spec["created_ts"] == 7
        assert validate_service_spec(spec) == []

    def test_missing_fields_rejected(self):
        errors = validate_service_spec({"kind": "service_spec"})
        assert any("name" in e for e in errors)
        assert any("description" in e for e in errors)
        assert any("input_schema" in e for e in errors)
        assert any("ask" in e for e in errors)

    def test_bad_ask_rejected(self):
        with pytest.raises(ValueError):
            build_service_spec(
                name="s", description="d", input_schema=SCHEMA,
                author="a", ask={"amount": "x"})  # non-int amount
        errors = validate_service_spec({
            "kind": "service_spec", "name": "s", "description": "d",
            "input_schema": SCHEMA, "author": "a",
            "ask": {"amount": "-5", "unit": "weird"}})
        assert any("non-negative" in e for e in errors)
        assert any("unit" in e for e in errors)

    def test_legacy_ask_token_stripped_not_rejected(self):
        """The vestigial `ask.token` was dropped 2026-07-26 (settlement is
        ATN-only). Removal is TOLERANT: an old caller still passing an
        ERC20 address is neither rejected nor allowed to poison the
        signed bytes — the field is stripped."""
        legacy = {"token": "0xC0ffee", "amount": "1000000", "unit": "per_item"}
        assert validate_service_spec({
            "kind": "service_spec", "name": "s", "description": "d",
            "input_schema": SCHEMA, "author": "a", "ask": legacy}) == []

        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", ask=legacy, created_ts=1)
        assert spec["ask"] == {"amount": "1000000", "unit": "per_item"}
        assert "token" not in spec["ask"]

    def test_ask_without_token_is_valid(self):
        """An ask is {amount, unit} — no token field required."""
        assert validate_service_spec({
            "kind": "service_spec", "name": "s", "description": "d",
            "input_schema": SCHEMA, "author": "a",
            "ask": {"amount": "5", "unit": "per_call"}}) == []

    def test_image_uri_optional_and_signed_not_embedded(self):
        """image_uri is display-plane: covered by the signature (so it
        cannot be swapped after signing) but NOT by the embedding text
        (presentation is not semantics)."""
        from atn.service_spec import service_embedding_text

        bare = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", ask=ASK, created_ts=1)
        assert bare["image_uri"] == ""            # absent -> empty, still valid
        assert validate_service_spec(bare) == []

        url = "https://example.invalid/services/s.jpg"
        withimg = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", ask=ASK, image_uri=url, created_ts=1)
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
            author="a", ask=ASK, created_ts=1)
        assert "inference" not in bare

        inf = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", ask=ASK, created_ts=1,
            inference={"model": "llama-3.1-8b", "max_tokens_cap": 512})
        assert inf["inference"] == {"model": "llama-3.1-8b",
                                    "max_tokens_cap": 512}
        assert validate_service_spec(inf) == []
        assert b"llama-3.1-8b" in canonical_service_bytes(inf)
        assert canonical_service_bytes(inf) != canonical_service_bytes(bare)
        assert service_embedding_text(inf) == service_embedding_text(bare)
        assert "llama" not in service_embedding_text(inf)

    def test_inference_cap_defaults_and_normalizes(self):
        from atn.service_spec import DEFAULT_MAX_TOKENS_CAP

        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA, author="a",
            ask=ASK, created_ts=1, inference={"model": "  gpt-4o-mini  "})
        assert spec["inference"] == {"model": "gpt-4o-mini",
                                     "max_tokens_cap": DEFAULT_MAX_TOKENS_CAP}

        # Extra keys are dropped: the shape is closed so the digest can't
        # drift on whatever a caller happened to pass.
        spec2 = build_service_spec(
            name="s", description="d", input_schema=SCHEMA, author="a",
            ask=ASK, created_ts=1,
            inference={"model": "m", "max_tokens_cap": 8, "junk": True})
        assert spec2["inference"] == {"model": "m", "max_tokens_cap": 8}

    def test_bad_inference_rejected(self):
        from atn.service_spec import validate_inference

        assert any("model" in e for e in validate_inference({}))
        assert any("model" in e for e in validate_inference({"model": 7}))
        assert any("max_tokens_cap" in e
                   for e in validate_inference({"model": "m",
                                                "max_tokens_cap": 0}))
        assert any("max_tokens_cap" in e
                   for e in validate_inference({"model": "m",
                                                "max_tokens_cap": "lots"}))
        assert validate_inference("nope") == [
            "inference must be a dict of {model, max_tokens_cap}"]
        with pytest.raises(ValueError):
            build_service_spec(
                name="s", description="d", input_schema=SCHEMA, author="a",
                ask=ASK, inference={"max_tokens_cap": 10})

    def test_canonical_bytes_excludes_sig(self):
        spec = build_service_spec(
            name="s", description="d", input_schema=SCHEMA,
            author="a", ask=ASK, created_ts=1)
        base = canonical_service_bytes(spec)
        spec["author_sig"] = "0xdeadbeef"
        assert canonical_service_bytes(spec) == base  # sig excluded

    def test_canonical_bytes_deterministic_and_sorted(self):
        a = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", ask=ASK, created_ts=1)
        b = build_service_spec(name="s", description="d", input_schema=SCHEMA,
                               author="a", ask=ASK, created_ts=1)
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
            author="child", ask=ASK, backing_tool="deadbeef")
        assert len(res["digest"]) == 64
        assert res["spec"]["author"] == "child"

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        record = reloaded.get(res["digest"])
        assert record is not None
        assert record.name == "svc"
        assert record.backing_tool == "deadbeef"
        assert not record.retired

    @pytest.mark.asyncio
    async def test_resolve_by_name_prefix_digest(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="uniq", description="d", input_schema=SCHEMA,
            author="user", ask=ASK)
        store = rt.service_store
        assert store.resolve("uniq").digest == res["digest"]
        assert store.resolve(f"svc_{res['digest'][:12]}").digest == res["digest"]
        assert store.resolve(res["digest"]).digest == res["digest"]

    @pytest.mark.asyncio
    async def test_update_ask_lineage(self, tmp_path):
        rt = _make_runtime(tmp_path)
        v1 = rt.service_store.register(
            name="svc", description="d", input_schema=SCHEMA,
            author="user", ask=ASK, backing_tool="tool-a")
        new_ask = {"amount": "2000000", "unit": "per_item"}
        v2 = rt.service_store.update_ask(v1["digest"], new_ask)

        assert v2["digest"] != v1["digest"]              # changed ask = new digest
        assert v2["spec"]["version_of"] == v1["digest"]  # lineage link
        assert v2["spec"]["ask"] == new_ask
        # backing_tool carried forward
        assert rt.service_store.get(v2["digest"]).backing_tool == "tool-a"
        # predecessor soft-retired; live list resolves to current
        assert rt.service_store.get(v1["digest"]).retired
        assert rt.service_store.resolve("svc").digest == v2["digest"]

    @pytest.mark.asyncio
    async def test_legacy_persisted_spec_with_ask_token_loads(self, tmp_path):
        """Specs written before the 2026-07-26 field removal carry
        `ask.token`. They must LOAD unchanged — the store reads blobs,
        it does not re-validate, so an old listing never disappears."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        res = rt.service_store.register(
            name="legacy_svc", description="Old.", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool="deadbeef")

        # Rewrite the persisted blob with the legacy ask shape, keeping
        # the registry row's digest (content-addressing is not the point
        # here — surviving the read is).
        blobs = rt.service_store._blob_store()
        spec = dict(blobs.get_json(res["digest"]))
        spec["ask"] = {"token": "0xC0ffee", **spec["ask"]}
        blob_path = blobs.data_dir / res["digest"]
        assert blob_path.exists()
        blob_path.write_text(json.dumps(spec), encoding="utf-8")

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        record = reloaded.get(res["digest"])
        assert record is not None
        assert record.name == "legacy_svc"
        assert record.ask["amount"] == ASK["amount"]
        assert record.ask["token"] == "0xC0ffee"   # carried, not fatal

    @pytest.mark.asyncio
    async def test_image_uri_round_trips_through_list(self, tmp_path):
        """Register with and without a banner image; both survive the
        store reload and appear on the list_services rows."""
        rt = _make_runtime(tmp_path)
        url = "https://example.invalid/services/with-image.jpg"
        withimg = rt.service_store.register(
            name="with_image", description="Has a banner.",
            input_schema=SCHEMA, author="user", ask=ASK, image_uri=url)
        without = rt.service_store.register(
            name="no_image", description="No banner.",
            input_schema=SCHEMA, author="user", ask=ASK)

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
    async def test_image_uri_carried_across_update_ask(self, tmp_path):
        rt = _make_runtime(tmp_path)
        url = "https://example.invalid/services/priced.jpg"
        v1 = rt.service_store.register(
            name="priced", description="d", input_schema=SCHEMA,
            author="user", ask=ASK, image_uri=url)
        v2 = rt.service_store.update_ask(
            v1["digest"],
            {"amount": "3000000", "unit": "per_item"})
        assert v2["spec"]["image_uri"] == url

    @pytest.mark.asyncio
    async def test_register_inference_backed(self, tmp_path):
        """An inference-backed service round-trips through the signed spec
        blob, so the backing survives a reload with no sidecar state."""
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="cognition", description="Chat completions, per call.",
            input_schema=SCHEMA, author="user", ask=ASK,
            inference={"model": "qwen2.5:14b", "max_tokens_cap": 1024})

        rec = rt.service_store.get(res["digest"])
        assert rec.inference == {"model": "qwen2.5:14b", "max_tokens_cap": 1024}
        assert rec.max_tokens_cap == 1024
        assert rec.backing_tool == ""

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded.get(res["digest"]).inference["model"] == "qwen2.5:14b"

    @pytest.mark.asyncio
    async def test_register_inference_cap_default(self, tmp_path):
        from atn.service_spec import DEFAULT_MAX_TOKENS_CAP

        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="cognition", description="d", input_schema=SCHEMA,
            author="user", ask=ASK, inference={"model": "m"})
        assert (rt.service_store.get(res["digest"]).max_tokens_cap
                == DEFAULT_MAX_TOKENS_CAP)

    @pytest.mark.asyncio
    async def test_inference_and_backing_tool_conflict(self, tmp_path):
        """One backing per service: a spec claiming both lies to the buyer
        about what they're paying for, so it's refused at registration."""
        rt = _make_runtime(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            rt.service_store.register(
                name="both", description="d", input_schema=SCHEMA,
                author="user", ask=ASK, backing_tool="deadbeef",
                inference={"model": "m", "max_tokens_cap": 8})
        assert rt.service_store.list() == []

    @pytest.mark.asyncio
    async def test_bad_inference_rejected_at_register(self, tmp_path):
        rt = _make_runtime(tmp_path)
        with pytest.raises(ValueError, match="model"):
            rt.service_store.register(
                name="nomodel", description="d", input_schema=SCHEMA,
                author="user", ask=ASK, inference={"max_tokens_cap": 8})
        with pytest.raises(ValueError, match="max_tokens_cap"):
            rt.service_store.register(
                name="badcap", description="d", input_schema=SCHEMA,
                author="user", ask=ASK,
                inference={"model": "m", "max_tokens_cap": -1})

    @pytest.mark.asyncio
    async def test_inference_carried_across_update_ask(self, tmp_path):
        rt = _make_runtime(tmp_path)
        v1 = rt.service_store.register(
            name="cognition", description="d", input_schema=SCHEMA,
            author="user", ask=ASK,
            inference={"model": "m", "max_tokens_cap": 64})
        v2 = rt.service_store.update_ask(
            v1["digest"],
            {"amount": "9", "unit": "per_call"})
        assert v2["spec"]["inference"] == {"model": "m", "max_tokens_cap": 64}
        assert rt.service_store.get(v2["digest"]).backing_tool == ""

    @pytest.mark.asyncio
    async def test_retire(self, tmp_path):
        rt = _make_runtime(tmp_path)
        res = rt.service_store.register(
            name="svc", description="d", input_schema=SCHEMA,
            author="user", ask=ASK)
        assert rt.service_store.retire(res["digest"])
        assert rt.service_store.get(res["digest"]).retired
        assert rt.service_store.list() == []
        assert len(rt.service_store.list(include_retired=True)) == 1
        assert not rt.service_store.retire("nope")


# ---------------------------------------------------------------------------
# Provider-side service_request dispatch (WS rail) + request log
# ---------------------------------------------------------------------------


def _server(rt):
    from atn.ws_server import WebSocketBridge
    return WebSocketBridge(rt)


def _local_session():
    from atn.ws_server import ClientSession
    return ClientSession(local=True, authed=True, owner=True,
                         root_agent_id="orchestrator", scope_ids=None)


async def _svc_request(rt, spec_digest, request_id, args, client="0xClient"):
    server = _server(rt)
    return await server._handle_service_request(
        {"spec_digest": spec_digest, "request_id": request_id,
         "args": args, "client": client},
        msg_id="m1",
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
        base = {"description": "d", "input_schema": SCHEMA, "ask": ASK}

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


class TestServiceRequestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_through_backing_tool(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="Echo service.", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool=tool["digest"])

        out = await _svc_request(rt, svc["digest"], "req-1", {"x": "hi"})
        assert out["ok"] is True
        assert out["result"]["request_id"] == "req-1"
        assert out["result"]["result"] == {"echo": "hi"}

        # Provider request log recorded the served item.
        summary = rt.service_store.summary()
        entry = summary[svc["digest"]]
        assert entry["count"] == 1
        assert entry["ok_count"] == 1
        assert entry["success_rate"] == 1.0
        assert entry["unique_clients"] == 1

    @pytest.mark.asyncio
    async def test_unknown_and_retired_and_no_backing(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")

        miss = await _svc_request(rt, "f" * 64, "r", {"x": "1"})
        assert miss["ok"] is False and "Unknown" in miss["error"]

        # No backing tool -> failure recorded.
        svc = rt.service_store.register(
            name="ghost", description="d", input_schema=SCHEMA,
            author="child", ask=ASK)
        out = await _svc_request(rt, svc["digest"], "r", {"x": "1"})
        assert out["ok"] is False and "backing" in out["error"]

        # Retired -> refused.
        rt.service_store.retire(svc["digest"])
        out = await _svc_request(rt, svc["digest"], "r2", {"x": "1"})
        assert out["ok"] is False and "retired" in out["error"].lower()

        # The no-backing attempt is on the provider log as a failure.
        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 1 and entry["ok_count"] == 0

    @pytest.mark.asyncio
    async def test_request_log_survives_reload(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool=tool["digest"])
        await _svc_request(rt, svc["digest"], "req-1", {"x": "a"})
        await _svc_request(rt, svc["digest"], "req-2", {"x": "b"}, client="0xOther")

        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded._request_seq == 2
        entry = reloaded.summary()[svc["digest"]]
        assert entry["count"] == 2
        assert entry["unique_clients"] == 2

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
        (rt._config.data_dir / "services").mkdir(parents=True, exist_ok=True)
        (rt._config.data_dir / "services" / "served_requests.json").write_text(
            json.dumps(["0x" + rid, rid]), encoding="utf-8")
        reloaded = ServiceStore(rt, rt._config.data_dir / "services")
        assert reloaded._seen_requests == {rid}
        assert reloaded.has_served_request("0x" + rid.upper()) is True


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

    Patches the SAME seam the sponsor rail resolves through
    (AutonetBridge._resolve_sponsor_provider), so the test exercises the
    real dispatch code rather than a parallel one."""
    calls: list[tuple] = []

    def _resolve(cfg, model):
        calls.append((cfg, model))
        return provider

    rt.autonet._resolve_sponsor_provider = _resolve
    return calls


def _inference_svc(rt, *, model="qwen2.5:14b", cap=100, name="cognition"):
    return rt.service_store.register(
        name=name, description="Chat completions, per call.",
        input_schema={"type": "object",
                      "properties": {"messages": {"type": "array"}}},
        author="user", ask=ASK,
        inference={"model": model, "max_tokens_cap": cap})


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
        # The spec's declared model is what we asked the stack for.
        assert resolved[0][1] == "qwen2.5:14b"
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
    async def test_missing_messages_rejected_and_logged(self, tmp_path):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        _stub_provider_stack(rt, _FakeProvider())

        out = await _svc_request(rt, svc["digest"], "r-empty", {})
        assert out["ok"] is False
        assert "messages" in out["result"]["error"]

        out2 = await _svc_request(rt, svc["digest"], "r-blank",
                                  {"messages": []})
        assert out2["ok"] is False

        # Failures land on the provider log like any other unserved item.
        entry = rt.service_store.summary()[svc["digest"]]
        assert entry["count"] == 2 and entry["ok_count"] == 0

    @pytest.mark.asyncio
    async def test_no_provider_configured_fails_closed(self, tmp_path):
        rt = _make_runtime(tmp_path)
        svc = _inference_svc(rt)
        _stub_provider_stack(rt, None)
        out = await _svc_request(rt, svc["digest"], "r",
                                 {"messages": [{"role": "user", "content": "x"}]})
        assert out["ok"] is False
        assert "no provider" in out["result"]["error"]

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
        base = {"description": "d", "input_schema": SCHEMA, "ask": ASK}

        res = await self._register(rt, {
            "name": "cognition",
            "inference": {"model": "qwen2.5:14b", "max_tokens_cap": 256},
            **base})
        assert res["ok"] is True
        assert res["result"]["spec"]["inference"] == {
            "model": "qwen2.5:14b", "max_tokens_cap": 256}

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
            "ask": ASK, "tool_digest": "deadbeef",
            "inference": {"model": "m", "max_tokens_cap": 8}})
        assert res["ok"] is False
        assert "mutually exclusive" in res["error"]

    @pytest.mark.asyncio
    async def test_ws_ignores_non_dict_inference(self, tmp_path):
        """A junk `inference` value is not a backing declaration — it must
        not turn a plain registration into a validation error."""
        rt = _make_runtime(tmp_path)
        res = await self._register(rt, {
            "name": "plain", "description": "d", "input_schema": SCHEMA,
            "ask": ASK, "inference": "yes-please"})
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
            author="child", ask=ASK, backing_tool=tool["digest"])

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
            author="child", ask=ASK, backing_tool=tool["digest"])

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
            author="child", ask=ASK, backing_tool=tool["digest"])

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
            author="child", ask=ASK, backing_tool=tool["digest"])
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
            author="child", ask=ASK)

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
            author="child", ask=ASK, backing_tool=tool["digest"])
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
            author="child", ask=ASK, backing_tool=tool["digest"])
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

        async def _spy_gate(request, record):
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

        # Signed with the daemon OWNER key (not the authoring agent's), for
        # the ask, to the spec's author_pubkey — the SAME recipient
        # _validate_service_payment verifies the ServicePayment event against.
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
    async def test_owner_authored_service_falls_back_to_owner_wallet(
            self, tmp_path, monkeypatch):
        """An owner-published service (author "user") has no agent identity,
        so its spec carries no author_pubkey. The seller IS this daemon's
        owner, so that wallet is the recipient — not an empty string that
        would abort the payment."""
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="owner_svc", description="d", input_schema=SCHEMA,
            author="user", ask=ASK, backing_tool=tool["digest"])
        # The key is omitted entirely, not stored empty.
        assert not rt.service_store.get(svc["digest"]).spec.get("author_pubkey")
        self._configure_chain(rt)

        from atn import service_client
        seen: list[str] = []

        async def _ok(config, key, recipient, amount, request_id=""):
            seen.append(recipient)
            return {"tx_hash": "0xfeed", "request_id": request_id}

        monkeypatch.setattr(service_client, "pay_for_service", _ok)
        server = _server(rt)

        async def _open_gate(request, record):
            return {"ok": True, "reason": "stubbed"}

        server._validate_service_payment = _open_gate

        out = await _invoke(rt, svc["digest"], {"x": "hi"}, server=server)
        assert out["ok"] is True
        assert seen == [rt._config.rpb.owner_wallet]
        assert out["result"]["receipt"]["recipient"] == rt._config.rpb.owner_wallet

    @pytest.mark.asyncio
    async def test_no_owner_key_is_a_payment_failure(self, tmp_path):
        rt = _make_runtime(tmp_path)
        await _register_agent(rt, "child")
        tool = await _register_echo_tool(rt)
        svc = rt.service_store.register(
            name="echo_svc", description="d", input_schema=SCHEMA,
            author="child", ask=ASK, backing_tool=tool["digest"])
        self._configure_chain(rt)
        rt._config.rpb.private_key = ""

        out = await _invoke(rt, svc["digest"], {"x": "hi"})
        assert out["ok"] is False
        assert "owner signing key" in out["error"]
        assert svc["digest"] not in rt.service_store.summary()

    @pytest.mark.asyncio
    async def test_foreign_digest_resolves_provider_and_dials(
            self, tmp_path, monkeypatch):
        """A digest this daemon does not publish is resolved by scanning the
        on-chain registry for its provider + ask, then bought cross-daemon."""
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

        monkeypatch.setattr(on_chain.ServiceMarketClient, "list_services", _list)

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

        monkeypatch.setattr(on_chain.ServiceMarketClient, "list_services", _list)
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

        monkeypatch.setattr(on_chain.ServiceMarketClient, "list_services", _list)
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
            author="child", ask={"amount": "0", "unit": "per_item"},
            backing_tool=tool["digest"])
        self._configure_chain(rt)

        from atn import service_client

        async def _never(*a, **kw):
            raise AssertionError("payment attempted for a zero ask")

        monkeypatch.setattr(service_client, "pay_for_service", _never)

        server = _server(rt)

        async def _open_gate(request, record):
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
