"""Service spec — the services-market's primary artifact kind.

Design: ``docs/services_market.md``. A Service is a remote API published
by an agent (ultimately by its human benefactor): general-purpose,
priced per work item in ATN. The price is NOT in the spec: the chain is
the only reference for it (Decision 2026-11-10, ``ServiceRegistry``'s
``askAmount``, changed in place by ``updateServiceAsk``), as it is for
whether the service is active and who is paid. Unlike a tool manifest, a
service spec carries NO substrate standing, NO verdict-layer claims, NO
mint — the trust basis is purely behavioral (signed identity, atomic
payment, receipt-gated reviews). It rides the same sha256-addressed blob
rail as tool manifests.

This module is the atn-side analogue of
``nodes.common.library.tool_manifest``, but deliberately
carries NO ``nodes.*`` dependency: ``canonical_service_bytes`` is
implemented locally so the ATN daemon can build/validate/sign a spec on
a standalone install. Identity is the digest; a revision is a NEW spec
whose ``version_of`` points at its predecessor. A new price is not a
revision: it changes on chain under the same digest.

The spec declares the unit of work (Decision 2026-11-10, "the unit of work;
metered on channels"): ``unit`` (required) says what one work item is, for
example "one completion up to 4,000 output tokens" or "one minute of RTX
5090 time". It is the meaning of the on-chain price, never a price itself.
``hardware`` (optional) names the hardware a compute service runs on, so a
buyer can check throughput against what that hardware documents.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

SERVICE_SPEC_KIND = "service_spec"

# Fields excluded from the canonical signing payload: the signature
# itself. Everything else — including author_pubkey — is covered so a
# spec cannot be re-attributed after signing.
_SIG_FIELD = "author_sig"

_REQUIRED_FIELDS = ("kind", "name", "description", "input_schema", "author",
                    "unit")

# Bounds on the unit-of-work texts: a sentence each, shown on every listing.
MAX_UNIT_CHARS = 200
MAX_HARDWARE_CHARS = 200

# Default output-token ceiling for an inference-backed service when the
# author names a model but no cap. Bounds the provider's own upstream
# spend per paid call (the on-chain price is set against the cap).
DEFAULT_MAX_TOKENS_CAP = 4096

# Default input bound of an inference-backed service: the characters of a
# request's messages (as compact JSON) plus its system prompt, about 16k
# tokens of text. A daemon cannot count every upstream model's tokens
# itself, so the bound is in characters. Written into the spec when the
# author names none, so a buyer reads the bound it is held to.
DEFAULT_MAX_INPUT_CHARS = 64_000

# Upstream providers an inference service cannot be served on: they are not
# this daemon's own provider stack (``service`` resells a purchase, ``rpb``
# a sponsor's grant) or generate no text (``jev``).
NOT_UPSTREAM_PROVIDERS = frozenset({"service", "rpb", "jev", "external"})

# A stream service's application protocol: a name and a version number,
# "openai-chat/1", "webrtc-sdp/1" (Decision 2026-11-10, a stream between
# buyer and seller). The network carries the bytes; what they mean is this
# protocol, the parties' business.
PROTOCOL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}/[0-9]{1,9}$")
MAX_PROTOCOL_DESCRIPTION = 2000


def inference_input_chars(messages: Any, system: str = "") -> int:
    """The size an inference request is bounded by: its messages as compact
    JSON plus its system prompt, in characters."""
    body = json.dumps(messages, separators=(",", ":"), ensure_ascii=False,
                      default=str)
    return len(body) + len(system or "")


def build_service_spec(
    *,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    author: str,
    unit: str = "",
    hardware: str = "",
    output_schema: Optional[Dict[str, Any]] = None,
    author_pubkey: str = "",
    endpoint_hint: str = "",
    image_uri: str = "",
    inference: Optional[Dict[str, Any]] = None,
    version_of: Optional[str] = None,
    created_ts: int = 0,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble and validate a service-spec payload.

    Raises ValueError with all problems joined if validation fails.
    ``created_ts`` is caller-supplied (no wall-clock reads here; the
    store stamps it).

    The spec carries no price (Decision 2026-11-10): the provider sets
    it on chain when registering and changes it there. ``unit`` (required)
    says what one work item is, the thing that price buys; ``hardware``
    (optional) names the hardware a compute service runs on. Both are
    stripped, and ``hardware`` is left out of the spec when empty.

    ``image_uri`` is display-plane only (a listing card banner), like
    ``endpoint_hint``: advisory, unvalidated, inside the signed payload
    so it cannot be swapped after signing, and deliberately NOT part of
    ``service_embedding_text`` — presentation is not semantics.

    ``inference`` (``{provider, model, max_tokens_cap, max_input_chars}``,
    decision 2026-07-26) declares that the service is backed by the
    provider daemon's OWN provider stack rather than a backing tool: a paid
    request is a chat completion served by exactly that upstream provider
    and model. It sits INSIDE the signed payload (the served model and the
    bounds are commitments to the buyer, so they must not be swappable
    post-signature) but, like ``image_uri``, is NOT part of
    ``service_embedding_text`` — the name/description carry the
    semantics, a model id is plumbing.

    ``protocol`` (``{id, description}``, Decision 2026-11-10) makes the
    service a stream: buyers open ``/autonet/service-stream/1.0.0`` to the
    provider and speak the named application protocol on it (``id`` is a
    name and a version, "openai-chat/1"; ``description`` is free text).
    """
    spec: Dict[str, Any] = {
        "kind": SERVICE_SPEC_KIND,
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "author": author,
        "unit": unit.strip() if isinstance(unit, str) else unit,
        "endpoint_hint": endpoint_hint,
        "image_uri": image_uri,
        "version_of": version_of,
        "created_ts": int(created_ts),
    }
    if output_schema is not None:
        spec["output_schema"] = output_schema
    if author_pubkey:
        spec["author_pubkey"] = author_pubkey
    if isinstance(hardware, str) and hardware.strip():
        spec["hardware"] = hardware.strip()
    elif hardware and not isinstance(hardware, str):
        spec["hardware"] = hardware
    if inference is not None:
        spec["inference"] = normalize_inference(inference)
    if protocol is not None:
        spec["protocol"] = normalize_protocol(protocol)

    errors = validate_service_spec(spec)
    if errors:
        raise ValueError("invalid service spec: " + "; ".join(errors))
    return spec


def validate_service_spec(payload: Dict[str, Any]) -> List[str]:
    """Return a list of validation problems (empty = valid)."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["service spec must be a dict"]

    if payload.get("kind") != SERVICE_SPEC_KIND:
        errors.append(f"kind must be {SERVICE_SPEC_KIND!r}")

    for field in _REQUIRED_FIELDS:
        if field == "kind":
            continue
        value = payload.get(field)
        if value in (None, "", {}):
            errors.append(f"missing required field {field!r}")

    schema = payload.get("input_schema")
    if schema is not None and not isinstance(schema, dict):
        errors.append("input_schema must be a JSON-schema dict")

    output_schema = payload.get("output_schema")
    if output_schema is not None and not isinstance(output_schema, dict):
        errors.append("output_schema must be a JSON-schema dict or absent")

    unit = payload.get("unit")
    if unit not in (None, "", {}):
        if not isinstance(unit, str) or not unit.strip():
            errors.append("unit must be a text saying what one work item is")
        elif len(unit) > MAX_UNIT_CHARS:
            errors.append(f"unit must be at most {MAX_UNIT_CHARS} characters")

    hardware = payload.get("hardware")
    if hardware is not None:
        if not isinstance(hardware, str) or not hardware.strip():
            errors.append("hardware must be a non-empty text naming the "
                          "hardware, or absent")
        elif len(hardware) > MAX_HARDWARE_CHARS:
            errors.append(f"hardware must be at most {MAX_HARDWARE_CHARS} "
                          "characters")

    if "ask" in payload:
        errors.append("a service spec carries no price: the asking price "
                      "lives on chain (ServiceRegistry askAmount)")

    version_of = payload.get("version_of")
    if version_of is not None and not isinstance(version_of, str):
        errors.append("version_of must be a digest string or null")

    inference = payload.get("inference")
    if inference is not None:
        errors.extend(validate_inference(inference))

    if "protocol" in payload:
        errors.extend(validate_protocol(payload.get("protocol")))

    return errors


def validate_protocol(protocol: Any) -> List[str]:
    """Shape-check a ``protocol`` block: ``{id, description}``, ``id`` a
    name and a version number ("openai-chat/1"), ``description`` free text
    (may be empty)."""
    if not isinstance(protocol, dict):
        return ["protocol must be a dict of {id, description}"]
    errors: List[str] = []
    pid = protocol.get("id")
    if not isinstance(pid, str) or not PROTOCOL_ID_RE.match(pid):
        errors.append("protocol.id must be a name and a version number, "
                      "e.g. 'openai-chat/1' or 'webrtc-sdp/1'")
    desc = protocol.get("description", "")
    if not isinstance(desc, str):
        errors.append("protocol.description must be text")
    elif len(desc) > MAX_PROTOCOL_DESCRIPTION:
        errors.append(f"protocol.description is longer than "
                      f"{MAX_PROTOCOL_DESCRIPTION} characters")
    extra = sorted(set(protocol) - {"id", "description"})
    if extra:
        errors.append(f"protocol has unknown keys {extra}")
    return errors


def normalize_protocol(protocol: Any) -> Any:
    """Trim a protocol block's ``id`` and ``description`` and default the
    description to "". Other keys are kept, so the validator refuses them
    instead of the digest silently dropping them."""
    if not isinstance(protocol, dict):
        return protocol
    out = dict(protocol)
    if isinstance(out.get("id"), str):
        out["id"] = out["id"].strip()
    desc = out.get("description", "")
    out["description"] = desc.strip() if isinstance(desc, str) else desc
    return out


def stream_protocol(payload: Any) -> Dict[str, Any]:
    """A stream service's application protocol block, or ``{}``."""
    block = payload.get("protocol") if isinstance(payload, dict) else None
    return block if isinstance(block, dict) else {}


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_inference(inference: Any) -> List[str]:
    """Shape-check an ``inference`` block:
    ``{provider, model, max_tokens_cap, max_input_chars}``.

    ``provider`` and ``model`` must be non-empty strings: the upstream that
    serves the buyer is a commitment chosen by the seller, never inferred
    from a model's name or defaulted (no default provider, decision
    2026-10-06). ``provider`` is a provider id of the seller's daemon, and
    not one in ``NOT_UPSTREAM_PROVIDERS``. ``max_tokens_cap`` (output
    tokens every request is clamped to) and ``max_input_chars`` (the input
    bound, see ``inference_input_chars``) are positive ints, defaulted when
    absent."""
    errors: List[str] = []
    if not isinstance(inference, dict):
        return ["inference must be a dict of {provider, model, "
                "max_tokens_cap, max_input_chars}"]
    provider = inference.get("provider")
    if not provider or not isinstance(provider, str) or not provider.strip():
        errors.append("inference.provider must name the upstream provider the "
                      "service is served on (a provider id of this daemon, "
                      "e.g. 'ollama' or 'anthropic')")
    elif provider.strip() in NOT_UPSTREAM_PROVIDERS:
        errors.append(f"inference.provider {provider.strip()!r} cannot serve "
                      "an inference service: it is not this daemon's own "
                      "provider stack, or it generates no text")
    model = inference.get("model")
    if not model or not isinstance(model, str) or not model.strip():
        errors.append("inference.model must be a non-empty model-id string")
    cap = inference.get("max_tokens_cap", DEFAULT_MAX_TOKENS_CAP)
    if cap in (None, ""):
        cap = DEFAULT_MAX_TOKENS_CAP
    if not _positive_int(cap):
        errors.append("inference.max_tokens_cap must be a positive int")
    limit = inference.get("max_input_chars", DEFAULT_MAX_INPUT_CHARS)
    if limit in (None, ""):
        limit = DEFAULT_MAX_INPUT_CHARS
    if not _positive_int(limit):
        errors.append("inference.max_input_chars must be a positive int")
    return errors


def normalize_inference(inference: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize an inference block: exactly four keys, the bounds
    defaulted.

    Keeping the shape closed means the signed bytes (and so the digest)
    don't drift on incidental extra keys a caller happened to pass."""
    provider = inference.get("provider")
    model = inference.get("model")
    cap = inference.get("max_tokens_cap", DEFAULT_MAX_TOKENS_CAP)
    if cap in (None, ""):
        cap = DEFAULT_MAX_TOKENS_CAP
    limit = inference.get("max_input_chars", DEFAULT_MAX_INPUT_CHARS)
    if limit in (None, ""):
        limit = DEFAULT_MAX_INPUT_CHARS
    out: Dict[str, Any] = {"provider": provider, "model": model,
                           "max_tokens_cap": cap, "max_input_chars": limit}
    if isinstance(provider, str):
        out["provider"] = provider.strip()
    if isinstance(model, str):
        out["model"] = model.strip()
    for key in ("max_tokens_cap", "max_input_chars"):
        if isinstance(out[key], int) and not isinstance(out[key], bool):
            out[key] = int(out[key])
    return out


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
    """Whether the spec is signed by the address it names
    (``author_pubkey``)."""
    author = str(payload.get("author_pubkey") or "").lower()
    return bool(author) and spec_signer(payload).lower() == author


def is_service_spec(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("kind") == SERVICE_SPEC_KIND


def service_embedding_text(payload: Dict[str, Any]) -> str:
    """Text a discovery index would embed for a spec: name + description
    + the unit of work and the hardware + sorted input-schema property
    names, so services are discoverable by what they do, what one work
    item is, what they run on, and their interface vocabulary."""
    name = str(payload.get("name") or "")
    description = str(payload.get("description") or "")
    unit = str(payload.get("unit") or "")
    hardware = str(payload.get("hardware") or "")
    props = payload.get("input_schema") or {}
    prop_names: List[str] = []
    if isinstance(props, dict):
        prop_names = sorted((props.get("properties") or {}).keys())
    parts = [name, description, unit, hardware]
    if prop_names:
        parts.append(" ".join(prop_names))
    return "\n".join(p for p in parts if p)


def canonical_service_bytes(payload: Dict[str, Any]) -> bytes:
    """Deterministic byte serialization for signing/verification.

    Excludes only ``author_sig``. Key-sorted, compact separators — the
    same spec dict always produces the same bytes on every daemon. Local
    twin of ``canonical_manifest_bytes`` (kept dependency-free on
    purpose; see module docstring)."""
    body = {k: v for k, v in payload.items() if k != _SIG_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def service_lineage(blob_store: Any, digest: str, max_depth: int = 64) -> List[str]:
    """Walk ``version_of`` links back through predecessors.

    Returns [digest, parent_digest, ...] oldest-last. Stops at missing
    blobs, non-spec payloads, cycles, or ``max_depth``."""
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
