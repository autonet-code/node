"""Service spec — the services-market's primary artifact kind.

Design: ``docs/services_market.md``. A Service is a remote API published
by an agent (ultimately by its human benefactor): general-purpose,
priced per work item, payable in any ERC20. Unlike a tool manifest, a
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
    version_of: Optional[str] = None,
    created_ts: int = 0,
) -> Dict[str, Any]:
    """Assemble and validate a service-spec payload.

    Raises ValueError with all problems joined if validation fails.
    ``created_ts`` is caller-supplied (no wall-clock reads here; the
    store stamps it).
    """
    spec: Dict[str, Any] = {
        "kind": SERVICE_SPEC_KIND,
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "author": author,
        "ask": ask,
        "endpoint_hint": endpoint_hint,
        "version_of": version_of,
        "created_ts": int(created_ts),
    }
    if output_schema is not None:
        spec["output_schema"] = output_schema
    if author_pubkey:
        spec["author_pubkey"] = author_pubkey

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

    return errors


def _validate_ask(ask: Any) -> List[str]:
    """The ask names the price: {token, amount, unit}. Payment/voucher
    validation is the contracts workstream's job — here we only check the
    shape so a well-formed ask reaches the escrow seam."""
    errors: List[str] = []
    if not isinstance(ask, dict):
        return ["ask must be a dict of {token, amount, unit}"]
    token = ask.get("token")
    if not token or not isinstance(token, str):
        errors.append("ask.token must be a non-empty ERC20 address string")
    amount = ask.get("amount")
    # amount is a decimal string (base units) to avoid float rounding on
    # token quantities — mirror the on-chain uint representation.
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
