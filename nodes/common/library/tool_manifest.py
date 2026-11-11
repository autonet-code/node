"""Tool manifest — the tool-substrate's primary artifact kind.

Design: ``docs/tool_substrate.md``. A tool manifest is a JSON payload
stored on the existing blob/ArtifactIndex rail, addressed by sha256
digest. The manifest is the tool: ToolBond reviews rate it by digest
(correctness, simplicity, novelty, malware, the charter axes), and the
epoch close pays its author once, when those reviews conclude
(Decisions 2026-11-07 and 2026-11-09). Its declared capabilities are part
of the claim reviewers check.

Two trust classes (v2 — docs/tool_substrate.md):

  - ``pinned``   — codebase tools. Behavior is locked by ``code_digest``
                   (sha256 of the code blob), so a review rates exactly
                   that code; the only mintable class.
  - ``attested`` — connector-backed tools: they run LOCALLY with the
                   user's own credentials but lean on an external API,
                   so a review rates behavior that can change. Publishable,
                   unminted.

Endpoint-backed manifests are NOT tools — remote counterparties with an
ask price are Services (``docs/services_market.md``) and are rejected
here.

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
import re
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
    # A codebase tool (docs/tool_codebases.md): code_digest is the archive
    # blob, tree_digest the content it holds, entrypoint the file to run.
    code_format: str = "",
    tree_digest: str = "",
    # Optional provenance {repo, commit}; never the identity.
    source: Optional[Dict[str, Any]] = None,
    # attested
    endpoint: str = "",
    provider: str = "",
    connector_id: str = "",
    # economics
    fee_atn: float = 0.0,
    # composition (docs/tool_substrate.md, Composition section):
    # digests of published tools this tool calls at runtime. Doubles as
    # the sandbox call-rail ALLOWLIST and the mint fan-out DAG.
    dependencies: Optional[List[str]] = None,
    # capability manifest (docs/tool_substrate.md, Adoption section):
    # what the code needs from the host, deny-by-default. Shape:
    # {"net": bool, "fs": bool, "spawn": bool, "env": [VAR, ...],
    #  "secrets": [SERVICE, ...]}.
    # On an ADOPTING daemon this is the sandbox policy — an undeclared
    # capability is a hard runtime failure, so declarations are honest
    # by construction. Absent = needs nothing beyond stdin/stdout.
    # 'secrets' is the tool-secret binding (docs/tool_secret_binding.md):
    # NAMES ONLY, always clamped by the calling agent's own allowance, so
    # a manifest can never widen what its caller already holds.
    capabilities: Optional[Dict[str, Any]] = None,
    # lineage
    version_of: Optional[str] = None,
    created_ts: int = 0,
    # novelty claim (docs/tool_substrate.md, Decision 2026-10-13 item 2):
    # {"statement": str, "prior_art": [digest, ...]}. Optional here (peers
    # validate older manifests); REQUIRED to publish (publish_novelty_errors).
    # Inside the signed manifest, so the claim is bound to the digest.
    novelty: Optional[Dict[str, Any]] = None,
    # A tool that serves sessions (Decision 2026-11-12, services_market.md):
    # {protocol: {id, spec, spec_url?}, max_sessions, process,
    #  reports_usage}. Absent = the tool answers one request per run.
    session: Optional[Dict[str, Any]] = None,
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
    if code_format:
        manifest["code_format"] = code_format
    if tree_digest:
        manifest["tree_digest"] = tree_digest
    if source:
        manifest["source"] = {"repo": str(source.get("repo") or ""),
                              "commit": str(source.get("commit") or "")}
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
    if dependencies:
        manifest["dependencies"] = list(dependencies)
    if capabilities:
        manifest["capabilities"] = dict(capabilities)
    if novelty is not None:
        manifest["novelty"] = normalize_novelty(novelty)
    if session is not None:
        manifest["session"] = normalize_session(session)

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
        fmt = payload.get("code_format")
        if fmt not in (None, "tree"):
            errors.append(f"unknown code_format {fmt!r}")
        if fmt == "tree":
            td = str(payload.get("tree_digest") or "")
            if len(td) != 64 or any(c not in "0123456789abcdef" for c in td):
                errors.append("a tree tool needs a 64-hex tree_digest")
            if not str(payload.get("entrypoint") or "").strip():
                errors.append("a tree tool needs an entrypoint")
    elif trust == TRUST_ATTESTED:
        if not payload.get("connector_id"):
            errors.append("attested manifest requires connector_id")

    if payload.get("endpoint"):
        errors.append("endpoint-backed offerings are Services, not tools "
                      "(docs/services_market.md) — register a service instead")

    version_of = payload.get("version_of")
    if version_of is not None and not isinstance(version_of, str):
        errors.append("version_of must be a digest string or null")

    authored_via = payload.get("authored_via")
    if authored_via is not None and not isinstance(authored_via, str):
        errors.append("authored_via must be a string")

    if payload.get("novelty") is not None:
        errors.extend(novelty_errors(payload.get("novelty")))

    if payload.get("session") is not None:
        errors.extend(session_errors(payload.get("session")))
        if trust != TRUST_PINNED:
            errors.append("only a pinned tool (with code) can serve sessions")

    caps = payload.get("capabilities")
    if caps is not None:
        if not isinstance(caps, dict):
            errors.append("capabilities must be a dict")
        else:
            for key in caps:
                if key not in ("net", "fs", "spawn", "env", "secrets",
                               "provides"):
                    errors.append(f"unknown capability {key!r} (allowed: "
                                  "net, fs, spawn, env, secrets, provides)")
            for flag in ("net", "fs", "spawn"):
                if flag in caps and not isinstance(caps[flag], bool):
                    errors.append(f"capability {flag!r} must be a bool")
            env = caps.get("env")
            if env is not None and (
                not isinstance(env, list)
                or not all(isinstance(v, str) and v for v in env)
            ):
                errors.append("capability 'env' must be a list of variable "
                              "names")
            # 'secrets' (docs/tool_secret_binding.md): vault SERVICE NAMES the
            # code reads at runtime. Declaring is not being granted — the
            # daemon clamps this against the CALLING AGENT's allowance and
            # binds only the intersection to the tool's own process. Dotted
            # names are the daemon plane (app.*, agent-key.*) and are rejected
            # here as well as stripped daemon-side: a manifest that names one
            # is malformed, not merely ineffective.
            secrets = caps.get("secrets")
            if secrets is not None:
                if (not isinstance(secrets, list)
                        or not all(isinstance(v, str) and v for v in secrets)):
                    errors.append("capability 'secrets' must be a list of "
                                  "vault service names")
                elif any("." in v for v in secrets):
                    errors.append("capability 'secrets' must not name "
                                  "daemon-plane services (no dots)")
            # 'provides' names the CORE tool names this manifest implements
            # (the shell bundle). Structural check only — which names are
            # actually overridable is a daemon-local question (the resident
            # bundle differs by build), so it is enforced at registration,
            # not here where a peer validates a gossiped manifest.
            provides = caps.get("provides")
            if provides is not None and (
                not isinstance(provides, list)
                or not all(isinstance(v, str) and v for v in provides)
            ):
                errors.append("capability 'provides' must be a list of "
                              "tool names")

    deps = payload.get("dependencies")
    if deps is not None:
        if not isinstance(deps, list) or not all(
            isinstance(d, str) and len(d) == 64 for d in deps
        ):
            errors.append("dependencies must be a list of sha256 hex digests")
        elif trust != TRUST_PINNED:
            errors.append("only pinned tools may declare dependencies")
        elif len(deps) != len(set(deps)):
            errors.append("dependencies must be unique")

    return errors


# ---------------------------------------------------------------------------
# Sessions (Decision 2026-11-12, docs/services_market.md). A tool that serves
# sessions runs as a server the daemon starts: it serves buyers on the port
# in AUTONET_SERVICE_PORT, reads its service's settings from
# AUTONET_SERVICE_SETTINGS (JSON), and, when it reports usage, posts each
# connection's token counts to AUTONET_USAGE_URL. The daemon carries the
# network, the payments and the cut-off; the tool only does the work.
#
#   protocol      {id, spec, spec_url?}: what buyers speak, a name and a
#                 version ("openai-chat/1", "ssh/2"); `spec` is the sha256 of
#                 the protocol's published specification (a blob the author
#                 adds to the store, so any daemon can fetch it), `spec_url`
#                 where it is published. Required.
#   max_sessions  how many sessions at once (0 = no cap). Default 0.
#   process       "shared": one process for every session (a model server);
#                 "per_session": a fresh process per buyer (a rented
#                 machine). Default "shared".
#   reports_usage the measures the tool counts and reports, a subset of
#                 MEASURES_REPORTED. Only these can be priced in tokens.
# ---------------------------------------------------------------------------

PROTOCOL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}/[0-9]{1,9}$")
SESSION_PROCESSES = ("shared", "per_session")
MEASURES_REPORTED = ("input_tokens", "output_tokens")
_SESSION_KEYS = ("protocol", "max_sessions", "process", "reports_usage")
_PROTOCOL_KEYS = ("id", "spec", "spec_url")


def normalize_session(session: Any) -> Any:
    """Canonical form of a session block: defaults filled, the spec digest
    lower-case without 0x, reports_usage sorted. Non-dicts pass through for
    validation to reject."""
    if not isinstance(session, dict):
        return session
    out = dict(session)
    proto = out.get("protocol")
    if isinstance(proto, dict):
        proto = dict(proto)
        if isinstance(proto.get("id"), str):
            proto["id"] = proto["id"].strip()
        if isinstance(proto.get("spec"), str):
            proto["spec"] = proto["spec"].strip().lower().removeprefix("0x")
        if isinstance(proto.get("spec_url"), str):
            proto["spec_url"] = proto["spec_url"].strip()
            if not proto["spec_url"]:
                del proto["spec_url"]
        out["protocol"] = proto
    out.setdefault("max_sessions", 0)
    out.setdefault("process", "shared")
    reported = out.get("reports_usage", [])
    if isinstance(reported, list):
        out["reports_usage"] = sorted({str(m) for m in reported})
    return out


def session_errors(session: Any) -> List[str]:
    """Structure check of a session block (see above)."""
    if not isinstance(session, dict):
        return ["session must be an object {protocol, max_sessions, process, "
                "reports_usage}"]
    errors: List[str] = []
    extra = sorted(set(session) - set(_SESSION_KEYS))
    if extra:
        errors.append(f"session has unknown keys {extra}")
    proto = session.get("protocol")
    if not isinstance(proto, dict):
        errors.append("session.protocol is required: {id, spec, spec_url?}")
    else:
        extra = sorted(set(proto) - set(_PROTOCOL_KEYS))
        if extra:
            errors.append(f"session.protocol has unknown keys {extra}")
        if not isinstance(proto.get("id"), str) or not PROTOCOL_ID_RE.match(proto["id"]):
            errors.append("session.protocol.id must be a name and a version "
                          "number, e.g. 'openai-chat/1' or 'ssh/2'")
        if not isinstance(proto.get("spec"), str) or not _HEX64.match(proto["spec"]):
            errors.append("session.protocol.spec must be the sha256 (64 hex) of "
                          "the protocol's published specification")
        url = proto.get("spec_url")
        if url is not None and (not isinstance(url, str) or not url.startswith("https://")):
            errors.append("session.protocol.spec_url must be an https address")
    cap = session.get("max_sessions", 0)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        errors.append("session.max_sessions must be a whole number (0 = no cap)")
    if session.get("process", "shared") not in SESSION_PROCESSES:
        errors.append(f"session.process must be one of {SESSION_PROCESSES}")
    reported = session.get("reports_usage", [])
    if not isinstance(reported, list) or any(m not in MEASURES_REPORTED for m in reported):
        errors.append(f"session.reports_usage must list measures from "
                      f"{MEASURES_REPORTED}")
    return errors


# ---------------------------------------------------------------------------
# Novelty claim (Decision 2026-10-13 item 2). The author states what is new
# and names the closest existing tools (found via probe_tools) by digest.
# Reviewers score the claim on `correctness`; this layer checks structure
# only. An empty prior_art list is allowed only when the statement says
# outright that nothing comparable exists, so "no neighbours" is an explicit,
# reviewable claim rather than an omission.
# ---------------------------------------------------------------------------

NOVELTY_MAX_STATEMENT = 4000
NOVELTY_MAX_PRIOR_ART = 32
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NOTHING_COMPARABLE = re.compile(
    r"\b(no|nothing|none|not\s+any)\b[^.;:]{0,60}?"
    r"\b(comparable|similar|equivalent|existing|prior|alternative|like\s+(this|it))\b"
    r"|\b(comparable|similar|equivalent|existing)\b[^.;]{0,60}?"
    r"\b(none|do(es)?\s*n[o']t\s+exist)\b",
    re.IGNORECASE)


def normalize_novelty(novelty: Any) -> Any:
    """Canonical form: stripped statement, lowercase prior_art digests
    (0x stripped, order kept), missing prior_art as []. Non-dicts pass
    through for validation to reject."""
    if not isinstance(novelty, dict):
        return novelty
    out = dict(novelty)
    if isinstance(out.get("statement"), str):
        out["statement"] = out["statement"].strip()
    prior = out.get("prior_art")
    if prior is None:
        out["prior_art"] = []
    elif isinstance(prior, list):
        out["prior_art"] = [p.strip().lower().removeprefix("0x")
                            if isinstance(p, str) else p for p in prior]
    return out


def states_nothing_comparable(statement: str) -> bool:
    """True when ``statement`` asserts that no comparable tool exists."""
    return bool(_NOTHING_COMPARABLE.search(str(statement or "")))


def novelty_errors(novelty: Any) -> List[str]:
    """Structural problems with a novelty claim (empty = well-formed)."""
    if not isinstance(novelty, dict):
        return ["novelty must be an object {statement, prior_art}"]
    errors: List[str] = []
    extra = set(novelty) - {"statement", "prior_art"}
    if extra:
        errors.append("novelty has unknown field(s): " + ", ".join(sorted(extra)))
    statement = novelty.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        errors.append("novelty.statement must be a non-empty string")
    elif len(statement) > NOVELTY_MAX_STATEMENT:
        errors.append(f"novelty.statement exceeds {NOVELTY_MAX_STATEMENT} chars")
    prior = novelty.get("prior_art", [])
    if not isinstance(prior, list) or not all(
        isinstance(p, str) and _HEX64.match(p) for p in prior
    ):
        errors.append("novelty.prior_art must be a list of 64-hex manifest digests")
    elif len(prior) != len(set(prior)):
        errors.append("novelty.prior_art must be unique")
    elif len(prior) > NOVELTY_MAX_PRIOR_ART:
        errors.append(f"novelty.prior_art lists more than {NOVELTY_MAX_PRIOR_ART} tools")
    elif (not prior and isinstance(statement, str) and statement.strip()
          and not states_nothing_comparable(statement)):
        errors.append("novelty.prior_art is empty: name the closest existing "
                      "tools (probe_tools), or state explicitly that nothing "
                      "comparable exists")
    return errors


def publish_novelty_errors(payload: Dict[str, Any]) -> List[str]:
    """What blocks publishing ``payload`` on the novelty rule: the claim is
    required to publish, and must be well-formed."""
    novelty = payload.get("novelty") if isinstance(payload, dict) else None
    if novelty is None:
        return ["publishing requires a novelty claim {statement, prior_art}: "
                "what is new, naming the closest existing tools by digest"]
    return novelty_errors(novelty)


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


# The kind of the gossiped tool registration (Decision 2026-11-08): the one
# consensus event a published tool produces. The close reads its digest
# and metadata; nothing else about a tool rides the wire.
TOOL_REGISTERED = "tool_registered"


def registration_event(manifest: Dict[str, Any], digest: str, *,
                       agent_id: str) -> Dict[str, Any]:
    """The ``tool_registered`` event for a published manifest.

    ``manifest_meta`` carries what the close needs so a mint never depends
    on blob replication: trust_class, author, the declared dependencies
    (comma-joined, the composition fan-out) and ``version_of`` when set,
    plus ``manifest``, the manifest's canonical bytes (sorted-key JSON, the
    bytes its digest is the sha256 of). A chain-backed close reads the
    fields from those bytes only (fundamentals audit G2), so no copy of
    the event gossiped first can change them. Values are strings on the
    wire.
    """
    meta = {
        "trust_class": str(manifest.get("trust_class", "")),
        "author": str(manifest.get("author", "")),
        "deps": ",".join(manifest.get("dependencies") or []),
        "manifest": json.dumps(manifest, sort_keys=True),
    }
    if manifest.get("version_of"):
        meta["version_of"] = str(manifest["version_of"])
    return {
        "kind": TOOL_REGISTERED,
        "seq": 1,
        "author_agent": str(agent_id),
        "artifact_digest": str(digest),
        "manifest_meta": meta,
    }
