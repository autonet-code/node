"""Service spec: how a tool is sold as a service (docs/services_market.md,
Decision 2026-11-12).

Every service is backed by a tool registered on the provider's daemon (it
can stay unpublished, so its code never leaves the machine). The spec is a
separate document from the tool's manifest: it describes how the tool is
sold, not what it is, so a changed listing never changes the tool's digest,
and one tool can back several services (an inference server that serves
several models, one service per model).

A service is used one of two ways (``mode``):

- ``request``: each request gets one reply. The tool runs once per request
  (JSON on stdin, JSON on stdout). Priced per request.
- ``session``: the buyer keeps a two-way connection open and speaks the
  protocol the tool serves; the daemon runs the tool as a server. Priced
  per measure (``measures``: seconds open, gigabytes transferred, input
  tokens, output tokens), each per a quantity of it.

The spec carries what buyers rely on and search reads: name, description,
``settings`` (values the daemon passes to the tool for this service and
that set it apart, such as the model), what the buyer sends (the tool's
input schema for a request service, the tool's protocol for a session
service), and the measures. Prices are not in it: they are in the
provider's signed price list (``atn/service_prices.py``), which changes
without a new spec or a transaction. The registry's ``askAmount`` is an
optional price nothing enforces. A service carries no substrate standing,
no verdict claims and no mint; trust in it is behavioral.

Identity is the digest; a revision is a new spec whose ``version_of``
points at its predecessor. This module has no ``nodes.*`` dependency, so
the daemon can build and verify specs on a standalone install.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

SERVICE_SPEC_KIND = "service_spec"

MODE_REQUEST = "request"
MODE_SESSION = "session"
MODES = (MODE_REQUEST, MODE_SESSION)

# What a session service can charge for. Requests are what a request
# service charges for, one price per request.
MEASURES = ("seconds", "gigabytes", "input_tokens", "output_tokens")
MEASURE_REQUESTS = "requests"
# Measures the tool reports (the daemon cannot count them itself).
REPORTED_MEASURES = ("input_tokens", "output_tokens")
# How much of a measure the daemon counts itself, in its own units.
BYTES_PER_GIGABYTE = 1_000_000_000

# Fields excluded from the canonical signing payload: the signature itself.
_SIG_FIELD = "author_sig"

_REQUIRED_FIELDS = ("kind", "name", "description", "mode", "input_schema", "author")
_ALLOWED_FIELDS = frozenset(_REQUIRED_FIELDS + (
    "output_schema", "author_pubkey", "author_sig", "settings", "measures",
    "protocol", "image_uri", "version_of", "created_ts"))

MAX_SETTINGS = 32
MAX_SETTING_KEY = 64
MAX_SETTING_VALUE = 500
SETTING_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")

# A protocol: a name and a version ("openai-chat/1", "ssh/2"), its
# specification pinned by sha256 (copied from the tool's manifest).
PROTOCOL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}/[0-9]{1,9}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def build_service_spec(
    *,
    name: str,
    description: str,
    mode: str,
    input_schema: Dict[str, Any],
    author: str,
    author_pubkey: str = "",
    output_schema: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    measures: Optional[List[Dict[str, Any]]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    image_uri: str = "",
    version_of: Optional[str] = None,
    created_ts: int = 0,
) -> Dict[str, Any]:
    """Assemble and validate a service spec. Raises ValueError with all
    problems joined. ``created_ts`` is caller-supplied (no wall-clock reads
    here; the store stamps it)."""
    spec: Dict[str, Any] = {
        "kind": SERVICE_SPEC_KIND,
        "name": str(name or "").strip(),
        "description": str(description or "").strip(),
        "mode": str(mode or "").strip(),
        "input_schema": input_schema,
        "author": author,
        "settings": normalize_settings(settings),
        "image_uri": str(image_uri or "").strip(),
        "version_of": version_of,
        "created_ts": int(created_ts),
    }
    if output_schema is not None:
        spec["output_schema"] = output_schema
    if author_pubkey:
        spec["author_pubkey"] = author_pubkey
    if measures is not None:
        spec["measures"] = normalize_measures(measures)
    if protocol is not None:
        spec["protocol"] = normalize_protocol(protocol)
    errors = validate_service_spec(spec)
    if errors:
        raise ValueError("invalid service spec: " + "; ".join(errors))
    return spec


def normalize_settings(settings: Any) -> Any:
    """Settings as ``{name: text}``, both trimmed. Non-dicts pass through
    for the validator to refuse."""
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        return settings
    return {str(k).strip(): (v.strip() if isinstance(v, str) else v)
            for k, v in settings.items()}


def normalize_measures(measures: Any) -> Any:
    """Measures as ``[{measure, quantity}]`` in the order given; quantity
    defaults to 1. Non-lists pass through for the validator to refuse."""
    if not isinstance(measures, list):
        return measures
    out = []
    for m in measures:
        if not isinstance(m, dict):
            out.append(m)
            continue
        row = {"measure": str(m.get("measure") or "").strip(),
               "quantity": m.get("quantity", 1)}
        if isinstance(row["quantity"], str) and row["quantity"].strip().isdigit():
            row["quantity"] = int(row["quantity"])
        out.append(row)
    return out


def normalize_protocol(protocol: Any) -> Any:
    """A protocol block ``{id, spec, spec_url?}`` trimmed, the digest
    lower-case without 0x."""
    if not isinstance(protocol, dict):
        return protocol
    out = dict(protocol)
    if isinstance(out.get("id"), str):
        out["id"] = out["id"].strip()
    if isinstance(out.get("spec"), str):
        out["spec"] = out["spec"].strip().lower().removeprefix("0x")
    if isinstance(out.get("spec_url"), str):
        out["spec_url"] = out["spec_url"].strip()
        if not out["spec_url"]:
            del out["spec_url"]
    return out


def validate_protocol(protocol: Any) -> List[str]:
    if not isinstance(protocol, dict):
        return ["protocol must be an object {id, spec, spec_url?}"]
    errors: List[str] = []
    extra = sorted(set(protocol) - {"id", "spec", "spec_url"})
    if extra:
        errors.append(f"protocol has unknown keys {extra}")
    if not isinstance(protocol.get("id"), str) or not PROTOCOL_ID_RE.match(protocol["id"]):
        errors.append("protocol.id must be a name and a version number, "
                      "e.g. 'openai-chat/1' or 'ssh/2'")
    if not isinstance(protocol.get("spec"), str) or not _HEX64.match(protocol["spec"]):
        errors.append("protocol.spec must be the sha256 of the protocol's "
                      "published specification")
    url = protocol.get("spec_url")
    if url is not None and (not isinstance(url, str) or not url.startswith("https://")):
        errors.append("protocol.spec_url must be an https address")
    return errors


def validate_service_spec(payload: Dict[str, Any]) -> List[str]:
    """Return a list of validation problems (empty = valid)."""
    if not isinstance(payload, dict):
        return ["service spec must be a dict"]
    errors: List[str] = []
    if payload.get("kind") != SERVICE_SPEC_KIND:
        errors.append(f"kind must be {SERVICE_SPEC_KIND!r}")
    for field in _REQUIRED_FIELDS:
        if field != "kind" and payload.get(field) in (None, "", {}):
            errors.append(f"missing required field {field!r}")
    extra = sorted(set(payload) - _ALLOWED_FIELDS)
    if extra:
        errors.append(f"unknown fields {extra} (a spec carries no price: prices "
                      "are in the provider's price list)")

    if payload.get("input_schema") is not None and not isinstance(payload.get("input_schema"), dict):
        errors.append("input_schema must be a JSON-schema object")
    out_schema = payload.get("output_schema")
    if out_schema is not None and not isinstance(out_schema, dict):
        errors.append("output_schema must be a JSON-schema object or absent")

    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        errors.append("settings must be an object of {name: text}")
    else:
        if len(settings) > MAX_SETTINGS:
            errors.append(f"at most {MAX_SETTINGS} settings")
        for k, v in settings.items():
            if not isinstance(k, str) or not SETTING_KEY_RE.match(k):
                errors.append(f"setting name {k!r} must be a short identifier")
            if not isinstance(v, str) or not v or len(v) > MAX_SETTING_VALUE:
                errors.append(f"setting {k!r} must be a text of 1 to "
                              f"{MAX_SETTING_VALUE} characters")

    mode = payload.get("mode")
    if mode not in MODES:
        errors.append(f"mode must be one of {MODES}")
    elif mode == MODE_REQUEST:
        if "measures" in payload:
            errors.append("a request service is priced per request: it has no measures")
        if "protocol" in payload:
            errors.append("a request service has no protocol (its tool answers "
                          "one request per run)")
    else:
        errors.extend(_measure_errors(payload.get("measures")))
        if "protocol" not in payload:
            errors.append("a session service names the protocol its tool serves")
        else:
            errors.extend(validate_protocol(payload.get("protocol")))

    version_of = payload.get("version_of")
    if version_of is not None and not isinstance(version_of, str):
        errors.append("version_of must be a digest string or null")
    return errors


def _measure_errors(measures: Any) -> List[str]:
    if not isinstance(measures, list) or not measures:
        return [f"a session service lists what it charges for: measures from "
                f"{MEASURES}, each with a quantity"]
    errors: List[str] = []
    seen = set()
    for m in measures:
        if not isinstance(m, dict) or set(m) - {"measure", "quantity"}:
            errors.append("each measure is {measure, quantity}")
            continue
        name = m.get("measure")
        if name not in MEASURES:
            errors.append(f"unknown measure {name!r} (one of {MEASURES})")
        elif name in seen:
            errors.append(f"measure {name!r} is listed twice")
        else:
            seen.add(name)
        qty = m.get("quantity")
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            errors.append(f"measure {name!r} needs a whole quantity above 0")
    return errors


def spec_measures(payload: Any) -> List[Dict[str, Any]]:
    """What a service charges for: ``[{measure, quantity}]``; a request
    service charges per request."""
    if not isinstance(payload, dict):
        return []
    if payload.get("mode") == MODE_REQUEST:
        return [{"measure": MEASURE_REQUESTS, "quantity": 1}]
    m = payload.get("measures")
    return list(m) if isinstance(m, list) else []


def session_protocol(payload: Any) -> Dict[str, Any]:
    """A session service's protocol block, or ``{}``."""
    block = payload.get("protocol") if isinstance(payload, dict) else None
    return block if isinstance(block, dict) else {}


def spec_signer(payload: Dict[str, Any]) -> str:
    """The address that signed ``payload`` (``author_sig`` over
    ``canonical_service_bytes``), or "" when unsigned or invalid."""
    sig = str(payload.get(_SIG_FIELD) or "")
    if not sig:
        return ""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        return str(Account.recover_message(
            encode_defunct(canonical_service_bytes(payload)),
            signature=sig if sig.startswith("0x") else "0x" + sig))
    except Exception:  # noqa: BLE001
        return ""


def signed_by_author(payload: Dict[str, Any]) -> bool:
    """Whether the spec is signed by the address it names (``author_pubkey``)."""
    author = str(payload.get("author_pubkey") or "").lower()
    return bool(author) and spec_signer(payload).lower() == author


def is_service_spec(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("kind") == SERVICE_SPEC_KIND


def service_embedding_text(payload: Dict[str, Any]) -> str:
    """Text the search model embeds for a spec: name, description, the
    settings' values (the model, a language), the protocol and the
    input-schema property names."""
    parts = [str(payload.get("name") or ""), str(payload.get("description") or "")]
    settings = payload.get("settings")
    if isinstance(settings, dict):
        parts.append(" ".join(str(v) for _, v in sorted(settings.items())))
    proto = session_protocol(payload)
    if proto.get("id"):
        parts.append(str(proto["id"]))
    schema = payload.get("input_schema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(props, dict):
        names = sorted(str(k) for k in props)
        if names:
            parts.append(" ".join(names))
    return "\n".join(p for p in parts if p)


def canonical_service_bytes(payload: Dict[str, Any]) -> bytes:
    """Deterministic bytes for signing and verifying: everything but
    ``author_sig``, key-sorted, compact."""
    body = {k: v for k, v in payload.items() if k != _SIG_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def service_lineage(blob_store: Any, digest: str, max_depth: int = 64) -> List[str]:
    """Walk ``version_of`` links back: [digest, parent, ...], oldest last.
    Stops at missing blobs, non-spec payloads, cycles, or ``max_depth``."""
    chain: List[str] = []
    seen: set[str] = set()
    current: Optional[str] = digest
    while current and current not in seen and len(chain) < max_depth:
        chain.append(current)
        seen.add(current)
        payload = blob_store.get_json(current)
        if not is_service_spec(payload):
            break
        current = payload.get("version_of")
    return chain
