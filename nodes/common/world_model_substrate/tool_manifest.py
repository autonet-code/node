"""Tool manifest — the tool-substrate's primary artifact kind.

Design: ``docs/tool_substrate.md``. A tool manifest is a JSON payload
stored on the existing blob/ArtifactIndex rail, addressed by sha256
digest. The manifest is the substrate item; claims in the verdict layer
reference it by digest, usage receipts attest invocations of it, and
epoch close mints to its author proportional to standing x usage.

Two trust classes:

  - ``pinned``   — codebase tools. Behavior is locked by ``code_digest``
                   (sha256 of the code blob). Verdicts are permanent;
                   standing compounds.
  - ``attested`` — API/endpoint tools. Behavior lives outside the
                   network, so standing decays without fresh usage
                   receipts (rug-pull pricing).

Identity is the digest. A revision is a NEW manifest whose
``version_of`` points at its predecessor — artifact lineage, no
in-place mutation, standing re-earned per version.

This module is substrate-side and dependency-light: it validates and
canonicalizes manifests but does NOT sign them. Signing/verification
happens on the ATN side where agent keys live; ``canonical_manifest_bytes``
is the byte string both sides sign/verify over.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

TOOL_MANIFEST_KIND = "tool_manifest"

TRUST_PINNED = "pinned"
TRUST_ATTESTED = "attested"
TRUST_CLASSES = (TRUST_PINNED, TRUST_ATTESTED)

# Fields excluded from the canonical signing payload: the signature
# itself (obviously) — everything else, including author_pubkey, is
# covered so a manifest cannot be re-attributed after signing.
_SIG_FIELD = "author_sig"

_REQUIRED_FIELDS = ("kind", "name", "description", "input_schema",
                    "author", "trust_class")


def build_tool_manifest(
    *,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    author: str,
    trust_class: str,
    output_schema: Optional[Dict[str, Any]] = None,
    author_pubkey: str = "",
    # pinned
    code_digest: str = "",
    entrypoint: str = "",
    runtime: str = "",
    # attested
    endpoint: str = "",
    provider: str = "",
    connector_id: str = "",
    # economics
    fee_atn: float = 0.0,
    # lineage
    version_of: Optional[str] = None,
    created_ts: int = 0,
) -> Dict[str, Any]:
    """Assemble and validate a tool-manifest payload.

    Raises ValueError with all problems joined if validation fails.
    ``created_ts`` is caller-supplied (epoch-close determinism rules:
    no wall-clock reads inside consensus-adjacent code paths).
    """
    manifest: Dict[str, Any] = {
        "kind": TOOL_MANIFEST_KIND,
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "author": author,
        "trust_class": trust_class,
        "version_of": version_of,
        "created_ts": int(created_ts),
    }
    if output_schema is not None:
        manifest["output_schema"] = output_schema
    if author_pubkey:
        manifest["author_pubkey"] = author_pubkey
    if code_digest:
        manifest["code_digest"] = code_digest
    if entrypoint:
        manifest["entrypoint"] = entrypoint
    if runtime:
        manifest["runtime"] = runtime
    if endpoint:
        manifest["endpoint"] = endpoint
    if provider:
        manifest["provider"] = provider
    if connector_id:
        manifest["connector_id"] = connector_id
    if fee_atn:
        manifest["fee_atn"] = float(fee_atn)

    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid tool manifest: " + "; ".join(errors))
    return manifest


def validate_manifest(payload: Dict[str, Any]) -> List[str]:
    """Return a list of validation problems (empty = valid)."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["manifest must be a dict"]

    if payload.get("kind") != TOOL_MANIFEST_KIND:
        errors.append(f"kind must be {TOOL_MANIFEST_KIND!r}")

    for field in _REQUIRED_FIELDS:
        if field == "kind":
            continue
        value = payload.get(field)
        if value in (None, "", {}):
            errors.append(f"missing required field {field!r}")

    schema = payload.get("input_schema")
    if schema is not None and not isinstance(schema, dict):
        errors.append("input_schema must be a JSON-schema dict")

    trust = payload.get("trust_class")
    if trust not in TRUST_CLASSES:
        errors.append(f"trust_class must be one of {TRUST_CLASSES}, got {trust!r}")
    elif trust == TRUST_PINNED:
        if not payload.get("code_digest"):
            errors.append("pinned manifest requires code_digest")
    elif trust == TRUST_ATTESTED:
        if not (payload.get("endpoint") or payload.get("connector_id")):
            errors.append("attested manifest requires endpoint or connector_id")

    version_of = payload.get("version_of")
    if version_of is not None and not isinstance(version_of, str):
        errors.append("version_of must be a digest string or null")

    return errors


def is_tool_manifest(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("kind") == TOOL_MANIFEST_KIND


def manifest_embedding_text(payload: Dict[str, Any]) -> str:
    """Text the ArtifactIndex embeds for a manifest.

    name + description + sorted input-schema property names, so tools
    are discoverable by what they do AND by their interface vocabulary.
    """
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


def canonical_manifest_bytes(payload: Dict[str, Any]) -> bytes:
    """Deterministic byte serialization for signing/verification.

    Excludes only ``author_sig``. Key-sorted, compact separators — the
    same manifest dict always produces the same bytes on every daemon.
    """
    body = {k: v for k, v in payload.items() if k != _SIG_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_lineage(blob_store: Any, digest: str, max_depth: int = 64) -> List[str]:
    """Walk ``version_of`` links back through predecessors.

    Returns [digest, parent_digest, ...] oldest-last. Stops at missing
    blobs, non-manifest payloads, cycles, or ``max_depth``.
    """
    chain: List[str] = []
    seen: set[str] = set()
    current: Optional[str] = digest
    while current and current not in seen and len(chain) < max_depth:
        chain.append(current)
        seen.add(current)
        payload = blob_store.get_json(current)
        if not is_tool_manifest(payload):
            break
        current = payload.get("version_of")
    return chain
