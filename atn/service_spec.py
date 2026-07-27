"""Service spec — the services-market's primary artifact kind.

Design: ``docs/services_market.md``. A Service is a remote API published
by an agent (ultimately by its human benefactor): general-purpose,
priced per work item, ATN-denominated (ratified 2026-07-10 — the ask is
``{amount, unit}``, no token field). Unlike a tool manifest, a
service spec carries NO substrate standing, NO verdict-layer claims, NO
mint — the trust basis is purely behavioral (signed identity, atomic
payment, receipt-gated reviews). It rides the same sha256-addressed blob
rail as tool manifests.

This module is the atn-side analogue of
``nodes.common.world_model_substrate.tool_manifest``, but deliberately
carries NO ``nodes.*`` dependency: ``canonical_service_bytes`` is
implemented locally so the ATN daemon can build/validate/sign a spec on
a standalone install. Identity is the digest; a revision is a NEW spec
whose ``version_of`` points at its predecessor (content-addressed, so a
changed ask = a new digest — see ``ServiceStore.update_ask``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

SERVICE_SPEC_KIND = "service_spec"

# Fields excluded from the canonical signing payload: the signature
# itself. Everything else — including author_pubkey — is covered so a
# spec cannot be re-attributed after signing.
_SIG_FIELD = "author_sig"

_REQUIRED_FIELDS = ("kind", "name", "description", "input_schema", "author", "ask")

# Ask "unit" vocabulary (advisory; pricing policy is the spec's business).
_ASK_UNITS = ("per_item", "per_minute", "per_token", "per_call", "flat")

# Default output-token ceiling for an inference-backed service when the
# author names a model but no cap. Bounds the provider's own upstream
# spend per paid call (the ask is priced against the cap).
DEFAULT_MAX_TOKENS_CAP = 4096


def build_service_spec(
    *,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    author: str,
    ask: Dict[str, Any],
    output_schema: Optional[Dict[str, Any]] = None,
    author_pubkey: str = "",
    endpoint_hint: str = "",
    image_uri: str = "",
    inference: Optional[Dict[str, Any]] = None,
    version_of: Optional[str] = None,
    created_ts: int = 0,
) -> Dict[str, Any]:
    """Assemble and validate a service-spec payload.

    Raises ValueError with all problems joined if validation fails.
    ``created_ts`` is caller-supplied (no wall-clock reads here; the
    store stamps it).

    ``ask`` is normalized to ``{amount, unit}`` — a legacy ``token``
    field is stripped rather than rejected (settlement is ATN-only,
    ratified 2026-07-10), so old callers keep working and the signed
    bytes stay canonical.

    ``image_uri`` is display-plane only (a listing card banner), like
    ``endpoint_hint``: advisory, unvalidated, inside the signed payload
    so it cannot be swapped after signing, and deliberately NOT part of
    ``service_embedding_text`` — presentation is not semantics.

    ``inference`` (``{model, max_tokens_cap}``, decision 2026-07-26)
    declares that the service is backed by the provider daemon's OWN
    provider stack rather than a backing tool: a paid request is a chat
    completion. It sits INSIDE the signed payload (the served model and
    the token ceiling are commitments to the buyer, so they must not be
    swappable post-signature) but, like ``image_uri``, is NOT part of
    ``service_embedding_text`` — the name/description carry the
    semantics, a model id is plumbing.
    """
    spec: Dict[str, Any] = {
        "kind": SERVICE_SPEC_KIND,
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "author": author,
        "ask": normalize_ask(ask),
        "endpoint_hint": endpoint_hint,
        "image_uri": image_uri,
        "version_of": version_of,
        "created_ts": int(created_ts),
    }
    if output_schema is not None:
        spec["output_schema"] = output_schema
    if author_pubkey:
        spec["author_pubkey"] = author_pubkey
    if inference is not None:
        spec["inference"] = normalize_inference(inference)

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

    ask = payload.get("ask")
    if ask is not None and ask != "":
        errors.extend(_validate_ask(ask))

    version_of = payload.get("version_of")
    if version_of is not None and not isinstance(version_of, str):
        errors.append("version_of must be a digest string or null")

    inference = payload.get("inference")
    if inference is not None:
        errors.extend(validate_inference(inference))

    return errors


def validate_inference(inference: Any) -> List[str]:
    """Shape-check an ``inference`` block: ``{model, max_tokens_cap}``.

    ``model`` must be a non-empty string (the served model is a
    commitment to the buyer, not a daemon-local default), and
    ``max_tokens_cap`` a positive int (the ceiling the provider clamps
    every request to)."""
    errors: List[str] = []
    if not isinstance(inference, dict):
        return ["inference must be a dict of {model, max_tokens_cap}"]
    model = inference.get("model")
    if not model or not isinstance(model, str):
        errors.append("inference.model must be a non-empty model-id string")
    cap = inference.get("max_tokens_cap", DEFAULT_MAX_TOKENS_CAP)
    if isinstance(cap, bool) or not isinstance(cap, int):
        errors.append("inference.max_tokens_cap must be a positive int")
    elif cap <= 0:
        errors.append("inference.max_tokens_cap must be a positive int")
    return errors


def normalize_inference(inference: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize an inference block: exactly two keys, cap defaulted.

    Keeping the shape closed means the signed bytes (and so the digest)
    don't drift on incidental extra keys a caller happened to pass."""
    model = inference.get("model")
    cap = inference.get("max_tokens_cap", DEFAULT_MAX_TOKENS_CAP)
    if cap in (None, ""):
        cap = DEFAULT_MAX_TOKENS_CAP
    out: Dict[str, Any] = {"model": model, "max_tokens_cap": cap}
    if isinstance(model, str):
        out["model"] = model.strip()
    if isinstance(cap, int) and not isinstance(cap, bool):
        out["max_tokens_cap"] = int(cap)
    return out


def _validate_ask(ask: Any) -> List[str]:
    """The ask names the price: {amount, unit}. Payment/voucher
    validation is the contracts workstream's job — here we only check the
    shape so a well-formed ask reaches the payment-channel seam.

    Settlement is ATN-only (ratified 2026-07-10), so the ask carries NO
    ``token`` field — a spec is ATN-denominated by construction. Old
    callers and old persisted specs may still carry one; it is ignored
    here and stripped by ``normalize_ask``."""
    errors: List[str] = []
    if not isinstance(ask, dict):
        return ["ask must be a dict of {amount, unit}"]
    amount = ask.get("amount")
    # amount is a decimal string (base units) to avoid float rounding on
    # ATN quantities — mirror the on-chain uint representation.
    if amount in (None, ""):
        errors.append("ask.amount is required (decimal string, base units)")
    elif not isinstance(amount, (str, int)):
        errors.append("ask.amount must be a decimal string or int")
    else:
        try:
            if int(str(amount)) < 0:
                errors.append("ask.amount must be non-negative")
        except (TypeError, ValueError):
            errors.append("ask.amount must parse as an integer base-unit amount")
    unit = ask.get("unit")
    if unit is not None and unit not in _ASK_UNITS:
        errors.append(f"ask.unit must be one of {_ASK_UNITS}, got {unit!r}")
    return errors


def normalize_ask(ask: Any) -> Dict[str, Any]:
    """Canonicalize an ask to ``{amount, unit}``.

    Drops the vestigial ``token`` field (settlement is ATN-only, ratified
    2026-07-10) and any other stray key, so a caller that still passes an
    ERC20 address doesn't break AND doesn't poison the signed bytes. Not
    a dict? Returned untouched — validation reports that."""
    if not isinstance(ask, dict):
        return ask
    out: Dict[str, Any] = {"amount": ask.get("amount")}
    unit = ask.get("unit")
    if unit is not None:
        out["unit"] = unit
    return out


def is_service_spec(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("kind") == SERVICE_SPEC_KIND


def service_embedding_text(payload: Dict[str, Any]) -> str:
    """Text a discovery index would embed for a spec: name + description
    + sorted input-schema property names — so services are discoverable
    by what they do AND by their interface vocabulary."""
    name = str(payload.get("name") or "")
    description = str(payload.get("description") or "")
    props = payload.get("input_schema") or {}
    prop_names: List[str] = []
    if isinstance(props, dict):
        prop_names = sorted((props.get("properties") or {}).keys())
    parts = [name, description]
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
