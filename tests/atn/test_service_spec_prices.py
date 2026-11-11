"""The service spec (atn/service_spec.py) and the signed price list
(atn/service_prices.py), Decision 2026-11-12 in docs/services_market.md."""
from __future__ import annotations

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from atn.service_prices import (
    Meter,
    build_price_list,
    charge_for,
    price_list_signer,
    request_price,
    sign_price_list,
    signed_by_provider,
    validate_price_list,
    voucher_size,
)
from atn.service_spec import (
    MAX_SETTING_VALUE,
    MAX_SETTINGS,
    build_service_spec,
    canonical_service_bytes,
    service_embedding_text,
    signed_by_author,
    spec_measures,
    spec_signer,
    validate_service_spec,
)

SCHEMA = {"type": "object", "properties": {"prompt": {"type": "string"},
                                           "temperature": {"type": "number"}}}
PROTO = {"id": "openai-chat/1", "spec": "ab" * 32}
KEY = "0x" + "42" * 32
ADDR = Account.from_key(KEY).address
DIGEST = "cd" * 32


def _request(**kw):
    args = dict(name="summarize", description="Summarize a text.",
                mode="request", input_schema=SCHEMA, author="a", created_ts=1)
    args.update(kw)
    return build_service_spec(**args)


def _session(**kw):
    args = dict(name="chat", description="Chat with a model.", mode="session",
                input_schema=SCHEMA, author="a", created_ts=1,
                settings={"model": "qwen2.5:14b"}, protocol=PROTO,
                measures=[{"measure": "seconds", "quantity": 60},
                          {"measure": "output_tokens", "quantity": 1000}])
    args.update(kw)
    return build_service_spec(**args)


def _raw(**kw):
    """A valid request spec as a dict, for the validator."""
    return {**_request(), **kw}


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def test_request_spec_has_no_measures_and_no_protocol():
    spec = _request()
    assert spec["kind"] == "service_spec" and spec["mode"] == "request"
    assert spec["settings"] == {} and spec["image_uri"] == ""
    assert spec["version_of"] is None
    assert "measures" not in spec and "protocol" not in spec
    assert validate_service_spec(spec) == []
    assert spec_measures(spec) == [{"measure": "requests", "quantity": 1}]


def test_session_spec_is_normalized():
    spec = _session(
        settings={" model ": " llama3 "},
        measures=[{"measure": " seconds ", "quantity": "60"}],
        protocol={"id": " ssh/2 ", "spec": "0X" + "AB" * 32, "spec_url": "  "})
    assert spec["settings"] == {"model": "llama3"}
    assert spec["measures"] == [{"measure": "seconds", "quantity": 60}]
    assert spec["protocol"] == {"id": "ssh/2", "spec": "ab" * 32}
    assert validate_service_spec(spec) == []
    assert spec_measures(spec) == spec["measures"]


@pytest.mark.parametrize("kw, err", [
    ({"mode": "stream"}, "mode must be one of"),
    ({"mode": "request", "measures": [{"measure": "seconds", "quantity": 1}]},
     "no measures"),
    ({"mode": "request", "protocol": PROTO}, "no protocol"),
    ({"measures": None}, "lists what it charges for"),
    ({"measures": []}, "lists what it charges for"),
    ({"protocol": None}, "names the protocol"),
])
def test_modes(kw, err):
    base = _session()
    spec = {**base, **kw}
    for k, v in kw.items():
        if v is None:
            spec.pop(k)
    assert any(err in e for e in validate_service_spec(spec)), validate_service_spec(spec)


@pytest.mark.parametrize("measures, err", [
    ([{"measure": "requests", "quantity": 1}], "unknown measure"),
    ([{"measure": "watts", "quantity": 1}], "unknown measure"),
    ([{"measure": "seconds", "quantity": 1}, {"measure": "seconds", "quantity": 2}],
     "listed twice"),
    ([{"measure": "seconds", "quantity": 0}], "whole quantity"),
    ([{"measure": "seconds", "quantity": -1}], "whole quantity"),
    ([{"measure": "seconds", "quantity": True}], "whole quantity"),
    ([{"measure": "seconds", "quantity": 1.5}], "whole quantity"),
    ([{"measure": "seconds", "quantity": 1, "price": 3}], "{measure, quantity}"),
    (["seconds"], "{measure, quantity}"),
    ([{"measure": ["seconds"], "quantity": 1}], "unknown measure"),
])
def test_bad_measures(measures, err):
    errors = validate_service_spec({**_session(), "measures": measures})
    assert any(err in e for e in errors), errors


@pytest.mark.parametrize("settings, err", [
    ("model=llama", "settings must be an object"),
    ({"1model": "x"}, "short identifier"),
    ({"model": ""}, "1 to"),
    ({"model": 7}, "1 to"),
    ({"model": "x" * (MAX_SETTING_VALUE + 1)}, "1 to"),
    ({f"k{i}": "v" for i in range(MAX_SETTINGS + 1)}, f"at most {MAX_SETTINGS}"),
])
def test_bad_settings(settings, err):
    errors = validate_service_spec(_raw(settings=settings))
    assert any(err in e for e in errors), errors


@pytest.mark.parametrize("protocol, err", [
    ({"id": "openai-chat", "spec": "ab" * 32}, "protocol.id"),
    ({"id": "openai-chat/1", "spec": "ab" * 31}, "protocol.spec"),
    ({"id": "openai-chat/1", "spec": "zz" * 32}, "protocol.spec"),
    ({**PROTO, "spec_url": "http://example.com/spec"}, "https"),
    ({**PROTO, "description": "x"}, "unknown keys"),
    ("openai-chat/1", "must be an object"),
])
def test_bad_protocol(protocol, err):
    errors = validate_service_spec({**_session(), "protocol": protocol})
    assert any(err in e for e in errors), errors


@pytest.mark.parametrize("field", ["unit", "inference", "hardware", "ask",
                                   "endpoint_hint", "prices", "price"])
def test_unknown_and_removed_fields_are_refused(field):
    errors = validate_service_spec(_raw(**{field: "x"}))
    assert any("unknown fields" in e and field in e for e in errors), errors


def test_missing_required_fields():
    errors = validate_service_spec({"kind": "service_spec"})
    for f in ("name", "description", "mode", "input_schema", "author"):
        assert any(repr(f) in e for e in errors), f
    assert validate_service_spec("x") == ["service spec must be a dict"]


def test_embedding_text():
    text = service_embedding_text(_session(image_uri="https://x.invalid/a.jpg"))
    for part in ("chat", "Chat with a model.", "qwen2.5:14b", "openai-chat/1",
                 "prompt temperature"):
        assert part in text
    assert "jpg" not in text
    # A schema whose properties are not an object adds nothing (no crash).
    assert "summarize" in service_embedding_text(
        _raw(input_schema={"properties": ["x"]}))


def test_canonical_bytes_and_signature():
    spec = _session()
    base = canonical_service_bytes(spec)
    assert canonical_service_bytes({**spec, "author_sig": "0xdead"}) == base
    assert canonical_service_bytes(_session(settings={"model": "other"})) != base

    spec["author_pubkey"] = ADDR
    spec["author_sig"] = Account.sign_message(
        encode_defunct(canonical_service_bytes(spec)), private_key=KEY).signature.hex()
    assert spec_signer(spec) == ADDR
    assert signed_by_author(spec)
    assert not signed_by_author({**spec, "settings": {"model": "other"}})
    assert not signed_by_author({**spec, "author_pubkey": "0x" + "00" * 20})
    assert spec_signer({**spec, "author_sig": ""}) == ""
    assert spec_signer({**spec, "author_sig": "nonsense"}) == ""


# ---------------------------------------------------------------------------
# Price list
# ---------------------------------------------------------------------------


def _pl(**kw):
    args = dict(service=DIGEST, provider=ADDR, prices={"requests": 5},
                voucher=50, seq=1, ts=100)
    args.update(kw)
    return build_price_list(**args)


def test_price_list_valid_against_its_spec():
    assert validate_price_list(_pl(), _request()) == []
    session_pl = _pl(prices={"seconds": 10, "output_tokens": 2000})
    assert validate_price_list(session_pl, _session()) == []
    # Free: no voucher needed.
    assert validate_price_list(_pl(prices={"requests": 0}, voucher=0), _request()) == []
    # The digest is kept normalized.
    assert _pl(service="0x" + DIGEST.upper())["service"] == DIGEST


@pytest.mark.parametrize("pl, err", [
    ({"prices": {"seconds": 1}}, "exactly"),
    ({"prices": {"requests": 1, "seconds": 1}}, "exactly"),
    ({"prices": {}}, "needs prices"),
    ({"prices": {"requests": -1}}, "whole number"),
    ({"prices": {"requests": True}}, "whole number"),
    ({"prices": {"requests": "5"}}, "whole number"),
    ({"voucher": 0}, "voucher size above 0"),
    ({"voucher": "abc"}, "voucher must be"),
    ({"voucher": [1]}, "voucher must be"),
    ({"seq": -1}, "seq must be"),
    ({"service": "ab" * 31}, "spec digest"),
    ({"service": "zz" * 32}, "spec digest"),
    ({"extra": 1}, "unknown fields"),
    ({"kind": "service_spec"}, "kind"),
])
def test_bad_price_lists(pl, err):
    errors = validate_price_list({**_pl(), **pl}, _request())
    assert any(err in e for e in errors), errors


def test_price_list_signature():
    signed = sign_price_list(_pl(), KEY)
    assert price_list_signer(signed) == ADDR
    assert signed_by_provider(signed)
    assert validate_price_list(signed, _request()) == []
    assert not signed_by_provider({**signed, "prices": {"requests": 1}})
    assert not signed_by_provider({**signed, "provider": "0x" + "00" * 20})
    assert not signed_by_provider(_pl())          # unsigned
    assert price_list_signer({**signed, "sig": "junk"}) == ""


def test_request_price_and_voucher_size():
    assert request_price(_pl()) == 5 and voucher_size(_pl()) == 50
    assert request_price(None) == 0 and voucher_size(None) == 0
    assert request_price(_pl(prices={"seconds": 3})) == 0
    assert request_price({"prices": {"requests": "x"}}) == 0
    assert voucher_size({"voucher": "x"}) == 0


def test_charge_for_rounds_up():
    assert charge_for(1, 3, 1000) == 1
    assert charge_for(1000, 3, 1000) == 3
    assert charge_for(1001, 3, 1000) == 4
    assert charge_for(5, 7, 1) == 35
    assert charge_for(0, 3, 1000) == 0
    assert charge_for(-5, 3, 1000) == 0
    assert charge_for(5, 0, 1000) == 0
    assert charge_for(5, 3, 0) == 15          # a quantity below 1 counts as 1


def test_meter_only_grows():
    m = Meter(_session(), {"seconds": 10, "output_tokens": 2000, "junk": 99})
    assert m.prices == {"seconds": 10, "output_tokens": 2000}
    assert m.charges("seconds") and not m.charges("input_tokens")
    deltas = [m.add("seconds", 1),              # ceil(1*10/60) = 1
              m.add("seconds", 59),             # 10 for 60 s: +9
              m.add("output_tokens", 1),        # ceil(2000/1000) = 2
              m.add("output_tokens", 999),      # still 2000 for 1000 tokens
              m.add("seconds", 0), m.add("seconds", -5), m.add("gigabytes", 3)]
    assert deltas == [1, 9, 2, 1998, 0, 0, 0]
    assert m.charged == sum(deltas) == 10 + 2000
    assert m.usage == {"seconds": 60, "output_tokens": 1000}

    req = Meter(_request(), {"requests": 5})
    assert [req.add("requests", 1) for _ in range(3)] == [5, 5, 5]
