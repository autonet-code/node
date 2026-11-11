"""Publishing a service (atn/service_publish.py): the form and the publish
path the WS surface and the Agents' register_service share."""
from __future__ import annotations

import json

import pytest

from atn import on_chain
from atn.service_publish import form_from, publish_service, read_service_file

from tests.atn.test_service_store import KEY, _make_runtime, _tool


def _form(**kw):
    msg = {"name": "echo", "description": "Echo.", "mode": "request",
           "backing_tool": "echo_tool", "price": "5", "voucher": "50"}
    msg.update(kw)
    return form_from(msg)


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------


def test_request_form_takes_a_price_per_request():
    form = _form(registry_price=3)
    assert form["prices"] == {"requests": 5}
    assert (form["voucher"], form["registry_price"]) == (50, 3)
    # Both given: the prices win.
    assert _form(prices={"requests": 9})["prices"] == {"requests": 9}
    # The app sends tool_digest for the backing tool.
    assert form_from({"mode": "request", "price": 1, "tool_digest": "ab" * 32}
                     )["backing_tool"] == "ab" * 32


def test_session_form_takes_prices_per_measure():
    form = _form(mode="session", price=None,
                 prices={"seconds": "10", "output_tokens": 2000},
                 measures=[{"measure": "seconds", "quantity": 60}])
    assert form["prices"] == {"seconds": 10, "output_tokens": 2000}
    with pytest.raises(ValueError, match="name the prices"):
        _form(mode="session")                    # a per-request price only


@pytest.mark.parametrize("kw, err", [
    ({"price": None}, "name the prices"),
    ({"price": "-1"}, "must not be negative"),
    ({"price": "1.5"}, "whole number"),
    ({"price": "lots"}, "whole number"),
    ({"voucher": "-2"}, "must not be negative"),
    ({"registry_price": "x"}, "whole number"),
    ({"settings": "model=x"}, "settings must be an object"),
    ({"input_schema": []}, "input_schema must be an object"),
    ({"measures": {"seconds": 1}}, "measures must be a list"),
])
def test_bad_forms(kw, err):
    with pytest.raises(ValueError, match=err):
        _form(**kw)


def test_service_file(tmp_path):
    path = tmp_path / "autonet-services" / "chat.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "name": "chat", "description": "Chat.", "mode": "session",
        "backing_tool": "chat_server", "settings": {"model": "llama3"},
        "measures": [{"measure": "seconds", "quantity": 60}],
        "prices": {"seconds": 10}, "voucher": 100}))
    form = form_from({"path": str(path), "voucher": 200})
    assert form["name"] == "chat" and form["settings"] == {"model": "llama3"}
    assert form["prices"] == {"seconds": 10}
    assert form["voucher"] == 200                # the message overrides the file

    path.write_text(json.dumps({"name": "chat", "unit": "one minute"}))
    with pytest.raises(ValueError, match="unknown fields"):
        form_from({"path": str(path)})
    path.write_text("[1]")
    with pytest.raises(ValueError, match="JSON object"):
        read_service_file(str(path))
    with pytest.raises(ValueError, match="cannot read"):
        read_service_file(str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


@pytest.fixture
def smc(monkeypatch):
    """A ServiceMarketClient that registers without a chain; a test sets
    ``available`` and ``result`` (a reply or an exception) on it."""
    class Fake:
        available = True
        result = {"success": True, "service_id": 4, "tx_hash": "0xabc"}
        calls: list = []

        def __init__(self, config):
            pass

        @property
        def registry_available(self):
            return self.available

        async def register_service(self, key, digest, ask):
            self.calls.append((key, digest, ask))
            if isinstance(self.result, Exception):
                raise self.result
            return dict(self.result)
    monkeypatch.setattr(on_chain, "ServiceMarketClient", Fake)
    return Fake


async def _publish(rt, signing_key=KEY, **kw):
    _tool(rt)
    return await publish_service(rt, author="user", signing_key=signing_key,
                                 form=_form(registry_price=3, **kw))


@pytest.mark.asyncio
async def test_published_on_chain(tmp_path, smc):
    rt = _make_runtime(tmp_path)
    announced = []
    rt.service_store.on_registered = announced.append
    out = await _publish(rt)
    assert out["on_chain"] is True and out["service_id"] == 4
    assert out["tx_hash"] == "0xabc" and out["price_list"]["prices"] == {"requests": 5}
    assert smc.calls == [(KEY, out["digest"], 3)]
    assert rt.service_store.get(out["digest"]).service_id == 4
    assert [r.digest for r in announced] == [out["digest"]]


@pytest.mark.asyncio
async def test_offload_runs_the_chain_call(tmp_path, smc):
    rt = _make_runtime(tmp_path)
    used = []

    async def offload(factory):
        used.append(factory)
        return await factory()
    _tool(rt)
    out = await publish_service(rt, author="user", signing_key=KEY,
                                form=_form(), offload=offload)
    assert out["on_chain"] is True and len(used) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("setup, key, reason", [
    ({"available": False}, KEY, "not configured"),
    ({}, "", "No signing key"),
    ({"result": {"success": False, "error": "reverted"}}, KEY,
     "On-chain registration failed: reverted"),
    ({"result": ConnectionError("rpc down")}, KEY, "rpc down"),
])
async def test_published_locally_only(tmp_path, smc, setup, key, reason):
    for k, v in setup.items():
        setattr(smc, k, v)
    rt = _make_runtime(tmp_path)
    out = await _publish(rt, signing_key=key)
    assert out["on_chain"] is False and reason in out["reason"]
    record = rt.service_store.get(out["digest"])
    assert record is not None and record.service_id is None


@pytest.mark.asyncio
async def test_a_bad_form_publishes_nothing(tmp_path, smc):
    rt = _make_runtime(tmp_path)
    out = await _publish(rt, prices={"seconds": 1})
    assert "exactly" in out["error"] and smc.calls == []
    assert len(rt.service_store) == 0
    rt.service_store = None
    out = await publish_service(rt, author="user", signing_key=KEY, form=_form())
    assert "not available" in out["error"]
