"""ToolStore — daemon-side registry of substrate tool manifests.

Design: ``docs/tool_substrate.md``. Registered tools are agent-authored
(or owner-authored) tool manifests persisted under ``data_dir/tools/``.
Manifests and pinned code blobs live in a local content-addressed blob
store (sha256 of key-sorted JSON — identical digests to the network
rail, so feeding a manifest to the substrate later re-produces the same
identity).

Scoping — authorship is the security primitive:

  A registered tool is visible/callable by its author and the author's
  ancestor chain (direct superiors), plus any agents the OWNER has
  explicitly granted. Cross-lineage granting is owner-only and lives on
  the WS surface, never in an agent tool — the same structural pattern
  as clone_agent. Enforcement happens both at listing time (visibility)
  and at call time (``ToolRegistry.call_tool``).

The substrate package (``nodes.*``) is imported lazily; on a standalone
atn install registration fails loudly instead of degrading silently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger(__name__)

# Author id recorded for owner-registered tools (WS surface, caller "").
OWNER_AUTHOR = "user"

# Publish gate (ruling 2026-07-24): rewards must never be unclaimable, so a
# manifest may only enter consensus with an author that has an on-chain
# claim path — an agent's 0x identity or the owner wallet. Address-less
# setups keep full PRIVATE tool capability; publishing is what's gated.
_ORPHAN_PUBLISH_ERROR = (
    "publishing requires a claimable author identity (agent 0x address or "
    "owner wallet) — register the agent on-chain or configure the owner "
    "wallet, then re-register the tool")


def _is_claimable_identity(author: str) -> bool:
    """True when ``author`` can claim epoch mint on-chain (0x address)."""
    return len(author) == 42 and author.startswith("0x")

_PINNED_EXEC_TIMEOUT_S = 120
_MAX_CODE_BYTES = 512 * 1024
# Composition (docs/tool_substrate.md — Composition, COMPOSITE_MAX_DEPTH):
# a composite may nest dep calls this many levels deep. Exceeding it is a
# runtime error frame, not a crash — guards runaway recursion / cycles.
_COMPOSITE_MAX_DEPTH = 4


def _authorized_hosts_for(services) -> set[str]:
    """Union of ``authorized_hosts`` across ``services`` (keystore sidecar).

    Fail-OPEN by omission, deliberately: a service with no configured hosts
    contributes nothing, and an empty union means the guard leaves net
    unrestricted. Configuring hosts is how an owner NARROWS a tool; absence is
    not a claim that the tool may go nowhere, it is the absence of a claim.
    Any keystore fault yields an empty set (same outcome: no narrowing).
    """
    try:
        from atn._vendor.kevin import keystore as _ks
    except Exception:  # noqa: BLE001
        return set()
    out: set[str] = set()
    for svc in services or ():
        try:
            for h in _ks.get_authorized_hosts(svc) or ():
                if isinstance(h, str) and h.strip():
                    out.add(h.strip().lower())
        except Exception:  # noqa: BLE001 — one bad service must not blind the rest
            log.debug("authorized_hosts lookup failed for %s", svc, exc_info=True)
    return out


@dataclass
class ToolRecord:
    """One registered tool: manifest digest + daemon-local state.

    Two author identities, deliberately distinct (E2E seam #3,
    docs/local_e2e.md): ``author`` (from the manifest) is the CONSENSUS
    identity — the agent's 0x address, globally unique and chain-
    claimable, what mint attribution and the damper's owner map key on.
    ``author_id`` is the LOCAL agent id used for daemon-side scoping
    (lineage walks over parent_id need registry ids, and local ids are
    meaningless network-wide — every daemon has an "assistant").
    """
    digest: str
    manifest: dict[str, Any]
    grants: set[str] = field(default_factory=set)   # owner-granted agent ids
    enabled: bool = True
    # Three tiers (tool_substrate.md v2): private (default) vs published.
    # Private tools are pure local capability — never pushed to the
    # substrate, no consensus footprint. Publishing is a deliberate act.
    published: bool = False
    # Local author agent id (scoping). Empty on pre-address rows —
    # author_id falls back to the manifest author. For ADOPTED records
    # this is the adopting agent — adoption scopes to the adopter's
    # lineage while manifest.author (the original 0x) keeps earning.
    local_author: str = ""
    registered_ts: int = 0
    # "authored" (default) | "adopted". Adopted records carry foreign
    # code: they execute ONLY under the capability guard
    # (atn/tool_guard.py — scrubbed env, sandbox cwd, deny-by-default
    # audit hook) and can never be re-published by the adopter.
    origin: str = "authored"

    @property
    def author(self) -> str:
        """Consensus author identity (0x address when available)."""
        return str(self.manifest.get("author") or "")

    @property
    def author_id(self) -> str:
        """Local agent id for scoping; manifest author as fallback."""
        return self.local_author or self.author

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or "")

    @property
    def trust_class(self) -> str:
        return str(self.manifest.get("trust_class") or "")

    def to_row(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "grants": sorted(self.grants),
            "enabled": self.enabled,
            "published": self.published,
            "local_author": self.local_author,
            "registered_ts": self.registered_ts,
            "origin": self.origin,
        }


class ToolStore:
    """Persistent registry of tool manifests with author-lineage scoping."""

    def __init__(self, runtime: Runtime, tools_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(tools_dir)
        self._registry_path = self._dir / "registry.jsonl"
        self._receipts_path = self._dir / "receipts.jsonl"
        self._attestations_path = self._dir / "attestations.jsonl"
        self._records: dict[str, ToolRecord] = {}
        self._blobs: Any = None  # lazy; None until substrate package needed
        self._embedder: Any = None  # lazy usefulness embedder; injectable for tests
        self._receipt_seq = 0
        # Optional consensus sinks: when the autonet WorldService is up,
        # event_sink submits ToolUsed events onto the substrate event
        # rail (gossip + epoch buffer) and manifest_sink registers the
        # manifest on the verdict layer (submit_tool_manifest). Local
        # state is ALWAYS recorded first; sinks are best-effort
        # federation — and idempotent, so backfill on late wiring is
        # safe (manifest claims are content-addressed by digest).
        self.event_sink: Any = None
        self.manifest_sink: Any = None
        # Optional network blob fetch (async callable: digest -> bytes
        # or None). Wired to the libp2p blob resolver when the autonet
        # host is up; lets validators vet (and later adopt) manifests
        # that were published from OTHER daemons. Fetched blobs are
        # digest-verified (content addressing IS the auth) and cached
        # into the local blob store.
        self.blob_fetcher: Any = None
        # Optional vet-status lookup (callable: digest -> dict | None)
        # reading the federated close's vetting carry-over — feeds the
        # provenance block on adoption proposals. Absent = "unknown".
        self.vet_status_provider: Any = None
        # Optional support-post sink (callable: (con_node_id, claim) ->
        # dict). Wired to WorldService.submit_support so a validator that
        # replays an evidence-CON and CONFIRMS it can post a PRO support
        # sprout under the CON in one flow (docs/tool_substrate.md —
        # Evidence). Absent = replay is diagnostic-only (no support post).
        self.support_sink: Any = None
        # Adoption proposals (docs/tool_substrate.md — Adoption rail):
        # agent proposes, OWNER approves per-tool. The queue is the one
        # legitimate approval queue in the tool economy.
        self._proposals_path = self._dir / "adoption_proposals.json"
        self._proposals: dict[str, dict[str, Any]] = self._load_proposals()
        # Active harness distro digest (docs/tool_substrate.md — "Resident
        # tools, loadouts, distros"). Set by bootstrap_reference_distro at
        # runtime init; stamped onto cognitive attestations so adoption
        # accrues to the distro. Empty until a distro is bootstrapped.
        self.active_loadout: str = ""
        self._load()
        # Seq counter spans BOTH tiers so seqs stay unique within the store
        # across a reload (mechanical receipts + cognitive attestations both
        # draw from it).
        self._receipt_seq = self._count_receipts() + self._count_attestations()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        author: str,
        code: str = "",
        entrypoint: str = "",
        provider: str = "",
        connector_id: str = "",
        version_of: str | None = None,
        publish: bool = False,
        dependencies: list[str] | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build, sign, store, and index a tool manifest. Returns
        ``{"digest", "manifest"}`` or raises ValueError on bad input.

        Trust class is derived, not chosen: ``code`` present → pinned
        (behavior hash-locked by the code blob digest); otherwise
        attested (connector-backed). Endpoint-backed offerings are
        Services and are rejected upstream (docs/services_market.md).

        ``publish=False`` (default) keeps the tool PRIVATE: local
        capability only, no substrate push, no consensus footprint.
        Publishing is a deliberate act.

        ``dependencies`` (docs/tool_substrate.md — Composition): digests
        of published tools this tool may call at runtime. Each declared
        dep MUST already exist in this store and be enabled — you cannot
        declare what you cannot call. The manifest layer additionally
        enforces pinned-only + 64-hex + uniqueness.
        """
        from nodes.common.world_model_substrate.tool_manifest import (
            build_tool_manifest,
        )

        blobs = self._blob_store()

        # Tool-secret declaration (docs/tool_secret_binding.md). Normalized at
        # registration so the stored manifest is what the binding reads: dotted
        # daemon-plane names stripped, deduped, sorted (stable content hash).
        # Declaring is NOT being granted — the binding clamps against the
        # CALLER's allowance at call time, so a manifest can never widen reach.
        if capabilities is not None:
            capabilities = dict(capabilities)
            if "secrets" in capabilities:
                from .runtime.tool_secrets import declared_tool_secrets
                declared = declared_tool_secrets({"capabilities": capabilities})
                if declared:
                    capabilities["secrets"] = sorted(declared)
                else:
                    capabilities.pop("secrets", None)
            if "provides" in capabilities:
                # 'provides' claims authority over CORE tool names (the shell
                # bundle). Normalize to a sorted unique list and drop anything
                # outside the overridable set — a manifest cannot invent a name
                # to provide, and cannot claim a framework tool that is not
                # swappable. Declaring is not being granted in any case: the
                # swap is gated off (runtime/shell_provider.py), so today this
                # field is descriptive only.
                try:
                    from .shell_tools import SHELL_TOOL_EXECUTORS as _SHELL
                    allowed = frozenset(_SHELL)
                except Exception:  # noqa: BLE001 — fail closed
                    allowed = frozenset()
                raw = capabilities.get("provides")
                claimed = sorted({
                    s.strip() for s in (raw if isinstance(raw, (list, tuple, set))
                                        else [])
                    if isinstance(s, str) and s.strip() in allowed
                })
                if claimed:
                    capabilities["provides"] = claimed
                else:
                    capabilities.pop("provides", None)
            if not capabilities:
                capabilities = None

        code_digest = ""
        if code:
            raw = code.encode("utf-8")
            if len(raw) > _MAX_CODE_BYTES:
                raise ValueError(
                    f"code exceeds {_MAX_CODE_BYTES} bytes; store large tools "
                    "as packages, not inline blobs")
            code_digest = blobs.add_bytes(raw)

        trust_class = "pinned" if code_digest else "attested"

        # Composition guard (docs/tool_substrate.md — Composition rule 1):
        # you cannot declare a dependency you cannot call. Each declared
        # digest must resolve to an existing, enabled record in THIS store.
        # (build_tool_manifest enforces the structural rules: pinned-only,
        # 64-hex, uniqueness.)
        deps = list(dependencies) if dependencies else []
        for dep_digest in deps:
            dep_record = self._records.get(dep_digest)
            if dep_record is None:
                raise ValueError(
                    f"declared dependency {dep_digest[:16]}... is not "
                    "registered in this store")
            if not dep_record.enabled:
                raise ValueError(
                    f"declared dependency {dep_digest[:16]}... is disabled")

        # Consensus author = the 0x address (chain-claimable, globally
        # unique — mint keyed by a local id has no on-chain claim path).
        # The local id stays on the record for daemon-side scoping.
        # Resolved BEFORE the idempotency match: if the author's identity
        # changed since an old record was baked (agent registered, owner
        # wallet configured), same-content re-registration must mint a
        # FRESH record with the claimable author, not resurrect the
        # orphan-authored one.
        consensus_author = self._consensus_identity(author)

        if publish and not _is_claimable_identity(consensus_author):
            raise ValueError(_ORPHAN_PUBLISH_ERROR)

        # Idempotency — content-addressed in SPIRIT, not just bytes: the
        # manifest bakes ``created_ts``, so re-registering byte-identical
        # content would mint a fresh digest every time (observed live: the
        # harness bootstrap re-registered per boot → 13 copies of every
        # atn_* module). Match the identity-defining fields instead and
        # return the existing record.
        for existing in self._records.values():
            m = existing.manifest
            if (existing.local_author == author
                    and (m.get("author") or "") == consensus_author
                    and existing.origin == "authored"
                    and m.get("name") == name
                    and m.get("trust_class") == trust_class
                    and (m.get("code_digest") or "") == code_digest
                    and (m.get("entrypoint") or "") == entrypoint
                    and (m.get("provider") or "") == provider
                    and (m.get("connector_id") or "") == connector_id
                    and m.get("description") == description
                    and list(m.get("dependencies") or []) == deps
                    and (m.get("capabilities") or None) == (capabilities or None)
                    and m.get("version_of") == version_of
                    and m.get("input_schema") == input_schema):
                if publish and not existing.published:
                    self.set_published(existing.digest, True)
                log.debug("register: content-identical to %s; reusing",
                          existing.digest[:16])
                return {"digest": existing.digest,
                        "manifest": existing.manifest,
                        "published": existing.published,
                        "existing": True}

        author_pubkey = self._author_address(author)

        manifest = build_tool_manifest(
            name=name,
            description=description,
            input_schema=input_schema,
            author=consensus_author,
            trust_class=trust_class,
            author_pubkey=author_pubkey,
            code_digest=code_digest,
            entrypoint=entrypoint,
            runtime="python3" if code_digest else "",
            provider=provider,
            connector_id=connector_id,
            version_of=version_of,
            dependencies=deps or None,
            capabilities=capabilities,
            created_ts=int(time.time()),
        )
        self._sign(author, manifest)

        digest = blobs.add_json(manifest)
        record = ToolRecord(
            digest=digest,
            manifest=manifest,
            published=bool(publish),
            local_author=author,
            registered_ts=int(time.time()),
        )
        self._records[digest] = record
        self._persist()
        log.info("registered tool %r (%s, %s, %s) by %s -> %s",
                 name, trust_class, "code" if code_digest else connector_id,
                 "published" if publish else "private", author, digest[:16])
        if record.published and self.manifest_sink is not None:
            try:
                # Consensus author (0x) — same identity the sprout's
                # author_agent and manifest_meta carry.
                self.manifest_sink(manifest, consensus_author)
            except Exception as exc:
                log.warning("manifest sink failed for %s: %s", name, exc)
        return {"digest": digest, "manifest": manifest,
                "published": record.published}

    def prune_superseded_local(
        self, *, name: str, author: str, keep_digest: str,
    ) -> int:
        """Drop UNPUBLISHED locally-authored records that share ``name`` +
        ``author`` with ``keep_digest`` but carry a different digest — the
        stale re-registrations left behind by pre-idempotency boots and by
        dev code drift (each source change mints a new digest with no
        version chain). Grants on pruned records migrate to the kept one.

        Local-plane only, deliberately conservative: published records are
        never pruned (their substrate registration is forward-only) and
        adopted records are not ours to touch. Blobs stay (content-
        addressed, harmless). Returns the number of records removed.
        """
        keep = self._records.get(keep_digest)
        if keep is None:
            return 0
        stale = [
            record for digest, record in self._records.items()
            if digest != keep_digest
            and record.local_author == author
            and record.manifest.get("name") == name
            and not record.published
            and record.origin == "authored"
        ]
        if not stale:
            return 0
        for record in stale:
            keep.grants.update(record.grants)
            self._records.pop(record.digest, None)
        self._persist()
        log.info("pruned %d superseded local record(s) of %r (kept %s)",
                 len(stale), name, keep_digest[:16])
        return len(stale)

    def set_published(self, digest: str, published: bool) -> bool:
        """Owner-gated publish/unpublish. Publishing pushes the manifest
        to the substrate; unpublishing only stops future pushes (the
        substrate is forward-only — existing claims stand). Adopted
        records are not ours to publish — the original author's
        publication stands."""
        record = self._records.get(digest)
        if record is None:
            return False
        if record.origin == "adopted":
            return False
        if published and not _is_claimable_identity(
                str(record.manifest.get("author") or "")):
            # Baked orphan author (pre-identity registration). Blocked —
            # re-registering the tool re-stamps it once an identity exists.
            raise ValueError(_ORPHAN_PUBLISH_ERROR)
        record.published = bool(published)
        self._persist()
        if record.published and self.manifest_sink is not None:
            try:
                self.manifest_sink(record.manifest, record.author)
            except Exception as exc:
                log.warning("manifest sink failed for %s: %s",
                            record.name, exc)
        return True

    def push_all_manifests(self) -> int:
        """Re-submit every PUBLISHED manifest through manifest_sink.

        Called when the substrate comes up after tools were registered
        offline. Idempotent on the world side (content-addressed claim
        node per digest). Private tools never leave the daemon. Returns
        the number pushed."""
        if self.manifest_sink is None:
            return 0
        pushed = 0
        for record in self._records.values():
            if not record.published:
                continue
            author = str(record.manifest.get("author") or "")
            if not _is_claimable_identity(author):
                # Legacy orphan-authored publication (pre-gate). Its
                # substrate claims are forward-only where they already
                # landed, but don't re-seed fresh worlds with rewards
                # nobody can claim.
                log.warning("skipping backfill of orphan-authored tool %r "
                            "(author=%r)", record.name, author)
                continue
            try:
                self.manifest_sink(record.manifest, record.author)
                pushed += 1
            except Exception as exc:
                log.warning("manifest backfill failed for %s: %s",
                            record.name, exc)
        return pushed

    # ------------------------------------------------------------------
    # Scoping
    # ------------------------------------------------------------------

    def allowed(self, caller_id: str | None, record: ToolRecord) -> bool:
        """True if ``caller_id`` may see/call ``record``.

        Owner callers always may. Agents may when they are the author,
        an ancestor of the author, or explicitly granted by the owner.
        """
        from .orchestrator import is_owner_caller
        if is_owner_caller(caller_id):
            return True
        assert caller_id is not None
        if not record.enabled:
            return False
        if caller_id == record.author_id:
            return True
        if caller_id in record.grants:
            return True
        return caller_id in self._ancestors(record.author_id)

    def visible_to(self, caller_id: str | None) -> list[ToolRecord]:
        return [r for r in self._records.values() if self.allowed(caller_id, r)]

    def get(self, digest: str) -> ToolRecord | None:
        return self._records.get(digest)

    def resolve(self, name_or_digest: str) -> ToolRecord | None:
        """Resolve by full digest, ``reg_<digest-prefix>``, or manifest
        name (None when the name is ambiguous across records)."""
        if name_or_digest in self._records:
            return self._records[name_or_digest]
        if name_or_digest.startswith("reg_"):
            prefix = name_or_digest[len("reg_"):]
            matches = [r for d, r in self._records.items() if d.startswith(prefix)]
            return matches[0] if len(matches) == 1 else None
        named = [r for r in self._records.values() if r.name == name_or_digest]
        return named[0] if len(named) == 1 else None

    def grant(self, digest: str, agent_id: str) -> bool:
        record = self._records.get(digest)
        if record is None:
            return False
        record.grants.add(agent_id)
        self._persist()
        return True

    def revoke(self, digest: str, agent_id: str) -> bool:
        record = self._records.get(digest)
        if record is None:
            return False
        record.grants.discard(agent_id)
        self._persist()
        return True

    def set_enabled(self, digest: str, enabled: bool) -> bool:
        record = self._records.get(digest)
        if record is None:
            return False
        record.enabled = enabled
        self._persist()
        return True

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def call(
        self,
        record: ToolRecord,
        arguments: dict[str, Any],
        *,
        caller_id: str | None = None,
        via: str = "",
        _depth: int = 0,
    ) -> dict[str, Any]:
        """Execute a registered tool. Scoping is the CALLER's job
        (``ToolRegistry.call_tool``) — this only dispatches.

        ``via`` tags the receipt with a composite's digest when this call
        was dispatched from a composite's sandbox call-rail (telemetry;
        mechanical receipts mint nothing). ``_depth`` is the composition
        nesting counter (``_COMPOSITE_MAX_DEPTH`` guard).
        """
        manifest = record.manifest
        if manifest.get("code_digest"):
            deps = manifest.get("dependencies") or []
            if deps:
                # Interactive (composition) path — opt-in on declared deps.
                result = await self._call_pinned_interactive(
                    record, arguments, caller_id=caller_id, depth=_depth)
            else:
                # Sealed path — byte-for-byte legacy contract.
                result = await self._call_pinned(
                    record, arguments, caller_id=caller_id)
        elif manifest.get("connector_id"):
            result = await self._call_connector(record, arguments)
        else:
            return {"error": f"Tool {record.name!r} has no executable backing"}

        self._record_receipt(record, arguments, caller_id,
                             ok="error" not in result, via=via)
        return result

    def _exec_spec(
        self, record: ToolRecord, script: Path,
        secrets: frozenset[str] | None = None,
    ) -> tuple[list[str], dict[str, str] | None, str | None]:
        """(argv, env, cwd) for a pinned tool subprocess.

        ADOPTED tools run fully contained (docs/tool_substrate.md — Adoption):

          - argv wraps the script in atn/tool_guard.py, whose audit
            hook hard-fails undeclared net/fs/spawn use;
          - env is scrubbed to the minimum Python needs plus ONLY the
            variables the capability manifest declares — secrets in
            the daemon's environment never reach foreign code;
          - cwd is a per-tool sandbox directory under the tool store.

        AUTHORED tools historically ran BARE — no guard, no sandbox, and
        ``env=None`` meaning full inheritance of the daemon environment
        ("the author judged their own code"). But the author is a model,
        and register_tool is therefore a general code-execution primitive
        (docs/tool_secret_binding.md — change 3). They now run under the
        guard by default, with one deliberate concession: ENV INHERITANCE
        IS PRESERVED. Authored tools depend on reading daemon config vars,
        and scrubbing them would break working tools for a benefit the
        audit hook already delivers. Tightening env is a separate step,
        gated on a survey of what actually reads what.

        Set ``ATN_AUTHORED_TOOLS_BARE=1`` to restore the old bare exec —
        an escape hatch for a daemon whose authored tools break under the
        hook, not a recommended posture.

        ``secrets`` is the RESOLVED tool-secret binding (L_agent ∩
        declared, docs/tool_secret_binding.md). It is advertised to the
        subprocess as ATN_TOOL_SECRETS — NAMES ONLY, never values. The
        tool reads each value from its own broker session, which is
        bound to its PID after spawn. Advertising the names is not a
        grant: the broker authorizes from the session, not this env var.
        """
        import sys
        secret_names = ",".join(sorted(secrets or ()))
        guard = Path(__file__).with_name("tool_guard.py")

        if record.origin != "adopted":
            if os.environ.get("ATN_AUTHORED_TOOLS_BARE") == "1":
                env = None
                if secret_names:
                    env = dict(os.environ)
                    env["ATN_TOOL_SECRETS"] = secret_names
                return [sys.executable, str(script)], env, None

            # Guarded, but env-inheriting (see docstring). The policy is the
            # author's own declaration; absent capabilities keep the historical
            # permissive posture (net/fs/spawn all allowed) so existing tools
            # keep working — the hook's value here is the DESTINATION check on
            # a secret-bound tool, not a sudden deny-by-default flip.
            caps = record.manifest.get("capabilities") or {}
            declared_any = any(k in caps for k in ("net", "fs", "spawn"))
            env = dict(os.environ)
            policy: dict[str, Any] = {
                "net": bool(caps.get("net")) if declared_any else True,
                "fs": bool(caps.get("fs")) if declared_any else True,
                "spawn": bool(caps.get("spawn")) if declared_any else True,
            }
            if secret_names:
                env["ATN_TOOL_SECRETS"] = secret_names
                hosts = _authorized_hosts_for(secrets or ())
                if hosts:
                    policy["hosts"] = sorted(hosts)
            env["ATN_TOOL_POLICY"] = json.dumps(policy)
            return ([sys.executable, str(guard), str(script)], env, None)

        caps = record.manifest.get("capabilities") or {}
        sandbox = self._dir / "sandbox" / record.digest[:16]
        sandbox.mkdir(parents=True, exist_ok=True)

        env: dict[str, str] = {
            "PATH": os.path.dirname(sys.executable),
            "TEMP": str(sandbox),
            "TMP": str(sandbox),
            "PYTHONIOENCODING": "utf-8",
        }
        # Windows: python/network runtime needs SystemRoot.
        for keep in ("SYSTEMROOT", "SystemRoot", "WINDIR"):
            if keep in os.environ:
                env[keep] = os.environ[keep]
        for name in caps.get("env") or []:
            if name in os.environ:
                env[name] = os.environ[name]
        policy: dict[str, Any] = {
            "net": bool(caps.get("net")),
            "fs": bool(caps.get("fs")),
            "spawn": bool(caps.get("spawn")),
        }
        if secret_names:
            env["ATN_TOOL_SECRETS"] = secret_names
            # docs/tool_secret_binding.md (change 2): a tool holding a
            # credential AND unrestricted egress is the shape worth narrowing.
            # The destinations come from each bound secret's authorized_hosts
            # — the field that until now was written and displayed but
            # consulted by nothing. An empty union leaves net unrestricted
            # (no hosts configured => no claim about where it may go).
            hosts = _authorized_hosts_for(secrets or ())
            if hosts:
                policy["hosts"] = sorted(hosts)
        env["ATN_TOOL_POLICY"] = json.dumps(policy)
        return ([sys.executable, str(guard), str(script)], env, str(sandbox))

    def _tool_secret_session(self, record: ToolRecord,
                             caller_id: str | None) -> tuple[Any, frozenset[str]]:
        """(session_or_None, resolved_services) for one pinned tool call.

        docs/tool_secret_binding.md — the secret is bound to the TOOL's PID,
        not the agent's, so the agent never holds a handle to it. Returns a
        minted-but-unbound session (bind after spawn) or None when the tool
        declared nothing / the clamp is empty / the tripwire is down.

        Never raises: a fault here means the tool runs WITHOUT secrets, which
        is the safe outcome. The secret is gated, not the execution.
        """
        try:
            from .runtime.tool_secrets import (
                ToolSecretSession, resolve_tool_secrets,
            )
            services = resolve_tool_secrets(
                record.manifest, self._runtime, caller_id or "")
            if not services:
                return None, frozenset()
            session = ToolSecretSession(
                self._runtime, services, agent_id=caller_id or "",
                tool_digest=record.digest)
            if not session.mint():
                return None, frozenset()
            return session, services
        except Exception:  # noqa: BLE001 — no secrets is always safe
            log.debug("tool secret binding failed; running without secrets",
                      exc_info=True)
            return None, frozenset()

    async def _call_pinned(self, record: ToolRecord,
                           arguments: dict[str, Any],
                           *, caller_id: str | None = None) -> dict[str, Any]:
        """Run the pinned code blob as a subprocess: JSON args on stdin,
        JSON result on stdout. The blob is materialized to a cache file
        named by its digest, so what runs is exactly what was judged.

        If the manifest declares secrets AND the caller's allowance covers
        them, a short-lived broker session is bound to the subprocess PID for
        the duration of the call (docs/tool_secret_binding.md)."""
        code_digest = record.manifest["code_digest"]
        blobs = self._blob_store()
        raw = blobs.get_bytes(code_digest)
        if raw is None:
            return {"error": f"code blob {code_digest[:16]}... not in local store"}

        cache_dir = self._dir / "code_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        script = cache_dir / f"{code_digest}.py"
        if not script.exists():
            script.write_bytes(raw)

        session, services = self._tool_secret_session(record, caller_id)
        argv, env, cwd = self._exec_spec(record, script, services)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env, cwd=cwd,
            )
            # Bind BEFORE writing stdin: the tool may request its secret the
            # instant it starts, and an unbound PID is not a broker session.
            if session is not None:
                session.bind(proc.pid)
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(arguments).encode("utf-8")),
                timeout=_PINNED_EXEC_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"error": f"tool timed out after {_PINNED_EXEC_TIMEOUT_S}s"}
        except Exception as exc:
            return {"error": f"tool subprocess failed to start: {exc}"}
        finally:
            # ALWAYS: drops policies + push sink and unlinks every staged file
            # for the PID, so no raw value outlives the subprocess.
            if session is not None:
                session.release()

        out_text = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            return {"error": f"tool exited {proc.returncode}: {err_text[:2000]}"}
        try:
            return {"result": json.loads(out_text)}
        except json.JSONDecodeError:
            return {"result": out_text[:8000]}

    async def _call_pinned_interactive(
        self,
        record: ToolRecord,
        arguments: dict[str, Any],
        *,
        caller_id: str | None,
        depth: int,
    ) -> dict[str, Any]:
        """Composition sandbox (docs/tool_substrate.md — Composition).

        Line-framed JSON protocol over a KEPT-OPEN stdin:

          - we write ``json.dumps(arguments) + "\\n"`` and keep stdin open;
          - the tool emits line-framed JSON on stdout:
              * ``{"call": <digest|name>, "args": {...}}`` — invoke a dep;
                the target MUST be in this manifest's declared dependency
                allowlist AND callable by the ORIGINAL caller (nested calls
                run under the original caller's authority, never the
                composite author's). We write the result back as one JSON
                line on stdin.
              * ``{"return": <result>}`` — final result.
          - undecodable / non-frame lines are tolerated (logged) and, at
            process exit without a return frame, parsed like the sealed
            path's stdout blob.

        Nested ``self.call`` carries ``via=<composite digest>`` so the
        nested mechanical receipt self-records tagged to this composite,
        and ``_depth=depth+1`` so ``_COMPOSITE_MAX_DEPTH`` bounds recursion.
        """
        code_digest = record.manifest["code_digest"]
        declared: set[str] = set(record.manifest.get("dependencies") or [])
        blobs = self._blob_store()
        raw = blobs.get_bytes(code_digest)
        if raw is None:
            return {"error": f"code blob {code_digest[:16]}... not in local store"}

        if depth >= _COMPOSITE_MAX_DEPTH:
            return {"error": f"composition depth exceeded {_COMPOSITE_MAX_DEPTH}"}

        cache_dir = self._dir / "code_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        script = cache_dir / f"{code_digest}.py"
        if not script.exists():
            script.write_bytes(raw)

        # Same tool-secret binding as the sealed path. Nested dep calls run
        # under the ORIGINAL caller's authority (see the frame handler below),
        # so each tool in a composition gets its own clamp against the same
        # L_agent — a composite cannot lend its secrets to a dependency.
        session, services = self._tool_secret_session(record, caller_id)
        argv, env, cwd = self._exec_spec(record, script, services)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env, cwd=cwd,
            )
            if session is not None:
                session.bind(proc.pid)
        except Exception as exc:
            if session is not None:
                session.release()
            return {"error": f"tool subprocess failed to start: {exc}"}

        async def _pump() -> dict[str, Any]:
            # Feed arguments as the first line; keep stdin open for the
            # dep-result round-trips (the tool blocks on readline()).
            proc.stdin.write((json.dumps(arguments) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            leftover: list[str] = []  # non-frame stdout for the exit fallback
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break  # EOF: process exited without a return frame
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    leftover.append(line)  # junk / plain output — tolerate
                    continue
                if not isinstance(frame, dict):
                    leftover.append(line)
                    continue

                if "return" in frame:
                    # Close stdin and reap the process so its pipes/
                    # transports are released (Windows Proactor loop leaks
                    # ResourceWarnings otherwise).
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except (asyncio.TimeoutError, Exception):
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                    return {"result": frame["return"]}

                if "call" in frame:
                    reply = await self._dispatch_dep_call(
                        frame, declared, record,
                        caller_id=caller_id, depth=depth)
                    proc.stdin.write((json.dumps(reply) + "\n").encode("utf-8"))
                    await proc.stdin.drain()
                    continue

                # A dict that is neither a call nor a return: treat as
                # trailing output (legacy-compatible best effort).
                leftover.append(line)

            # No return frame: fall back to the sealed path's parsing of
            # accumulated non-frame stdout, respecting exit code.
            await proc.wait()
            if proc.returncode != 0:
                err = (await proc.stderr.read()).decode(
                    "utf-8", errors="replace").strip()
                return {"error": f"tool exited {proc.returncode}: {err[:2000]}"}
            out_text = "\n".join(leftover).strip()
            if not out_text:
                return {"error": "composite exited without a return frame"}
            try:
                return {"result": json.loads(out_text)}
            except json.JSONDecodeError:
                return {"result": out_text[:8000]}

        try:
            return await asyncio.wait_for(_pump(), timeout=_PINNED_EXEC_TIMEOUT_S)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"error": f"tool timed out after {_PINNED_EXEC_TIMEOUT_S}s"}
        except Exception as exc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"error": f"composite sandbox failed: {exc}"}
        finally:
            # ALWAYS, incl. the kill paths: unlink staged files + drop the
            # session so no raw value outlives the composite subprocess.
            if session is not None:
                session.release()

    async def _dispatch_dep_call(
        self,
        frame: dict[str, Any],
        declared: set[str],
        composite: ToolRecord,
        *,
        caller_id: str | None,
        depth: int,
    ) -> dict[str, Any]:
        """Service one ``{"call": ..., "args": ...}`` frame from a composite.

        Two gates, in order:
          1. DECLARED allowlist — the resolved target's digest must be in
             the composite manifest's ``dependencies`` (an undeclared call
             is impossible by construction). Reject → error frame.
          2. ORIGINAL-caller scoping — the nested call runs under the agent
             that invoked the composite, never the composite author. A
             composite must not launder access its caller lacks. Reject →
             error frame.
        """
        target = frame.get("call")
        if not isinstance(target, str) or not target:
            return {"error": "malformed call frame"}
        dep_record = self.resolve(target)
        if dep_record is None:
            return {"error": "undeclared dependency"}
        if dep_record.digest not in declared:
            return {"error": "undeclared dependency"}
        if not self.allowed(caller_id, dep_record):
            return {"error": "caller not authorized for dependency"}
        args = frame.get("args")
        if not isinstance(args, dict):
            args = {}
        return await self.call(
            dep_record, args,
            caller_id=caller_id,
            via=composite.digest,
            _depth=depth + 1,
        )

    async def _call_connector(self, record: ToolRecord,
                              arguments: dict[str, Any]) -> dict[str, Any]:
        """Attested connector-backed tool: the manifest name is the
        connector operation name."""
        connector_id = record.manifest["connector_id"]
        try:
            await self._runtime.connectors.ensure_started([connector_id])
        except Exception as exc:
            return {"error": f"failed to start connector {connector_id!r}: {exc}"}
        return await self._runtime.connectors.call_tool(
            connector_id, record.name, arguments)

    # ------------------------------------------------------------------
    # Receipts + fee ledger (docs/tool_substrate.md — usage receipts)
    # ------------------------------------------------------------------

    def _record_receipt(
        self,
        record: ToolRecord,
        arguments: dict[str, Any],
        caller_id: str | None,
        *,
        ok: bool,
        via: str = "",
    ) -> None:
        """Append a usage receipt and (best-effort) emit the consensus
        ``tool_used`` event. Local ledger first — never lost to a
        substrate outage; the sink failure is logged, not raised.

        ``via`` (composition telemetry): when non-empty, this call was
        dispatched from a composite's sandbox call-rail; the digest is
        recorded on the receipt row and the emitted event so nested
        mechanical receipts stay attributable to the composite (mechanical
        receipts still mint nothing — docs/tool_substrate.md, Composition
        rule 2)."""
        import hashlib

        caller = caller_id if caller_id else OWNER_AUTHOR
        fee = float(record.manifest.get("fee_atn") or 0.0)
        args_digest = hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode("utf-8")).hexdigest()
        self._receipt_seq += 1
        receipt = {
            "seq": self._receipt_seq,
            "ts": int(time.time()),
            "manifest_digest": record.digest,
            "tool_name": record.name,
            "tool_author": record.author,
            "caller": caller,
            "arguments_digest": args_digest,
            "ok": ok,
            "fee_atn": fee,
        }
        if via:
            receipt["via"] = via
        receipt_digest = ""
        try:
            receipt_digest = self._blob_store().add_json(receipt)
        except RuntimeError:
            pass  # standalone install: ledger row still recorded below
        receipt["receipt_digest"] = receipt_digest

        self._dir.mkdir(parents=True, exist_ok=True)
        with self._receipts_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt) + "\n")

        if self.event_sink is not None:
            event = {
                "kind": "tool_used",
                "seq": self._receipt_seq,
                # Consensus rail carries 0x identities (mint/damper/
                # owner-map key space); local ids stay in the jsonl
                # rows for display.
                "author_agent": self._consensus_identity(caller),
                "manifest_digest": record.digest,
                "tool_author": record.author,
                "receipt_digest": receipt_digest,
                "ok": ok,
                "fee_atn": fee,
            }
            if via:
                event["via"] = via
            try:
                self.event_sink(event)
            except Exception as exc:
                log.warning("tool_used event sink failed: %s", exc)

    def balances(self) -> dict[str, Any]:
        """Off-chain fee ledger, derived from the receipts log.

        Earned accrues to tool authors, spent to callers, on successful
        invocations of fee-bearing tools. Settlement on-chain (labeled
        ATN transfer at epoch close via payForService) is deliberately
        NOT wired yet.
        """
        earned: dict[str, float] = {}
        spent: dict[str, float] = {}
        usage: dict[str, dict[str, Any]] = {}
        for receipt in self._iter_receipts():
            digest = receipt.get("manifest_digest", "")
            entry = usage.setdefault(digest, {
                "tool_name": receipt.get("tool_name", ""),
                "tool_author": receipt.get("tool_author", ""),
                "count": 0, "ok_count": 0, "fee_total": 0.0,
            })
            entry["count"] += 1
            if receipt.get("ok"):
                entry["ok_count"] += 1
                fee = float(receipt.get("fee_atn") or 0.0)
                if fee > 0:
                    author = receipt.get("tool_author", "")
                    caller = receipt.get("caller", "")
                    entry["fee_total"] = round(entry["fee_total"] + fee, 10)
                    earned[author] = round(earned.get(author, 0.0) + fee, 10)
                    spent[caller] = round(spent.get(caller, 0.0) + fee, 10)
        return {
            "earned": dict(sorted(earned.items())),
            "spent": dict(sorted(spent.items())),
            "usage": dict(sorted(usage.items())),
        }

    def _iter_receipts(self):
        if not self._receipts_path.exists():
            return
        try:
            lines = self._receipts_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("ToolStore could not read receipts: %s", exc)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("ToolStore skipping corrupt receipt row")

    def _count_receipts(self) -> int:
        return sum(1 for _ in self._iter_receipts())

    # ------------------------------------------------------------------
    # Cognitive attestation (docs/tool_substrate.md — two receipt tiers)
    # ------------------------------------------------------------------

    def _usefulness_embedder(self):
        """Lazy usefulness embedder (dim=1024), cached on the instance.

        Injectable for tests: pre-set ``self._embedder`` to skip the heavy
        subprocess/torch worker. Returns None if the substrate package is
        unavailable — callers must degrade to ``problem_coords=[]``."""
        if self._embedder is not None:
            return self._embedder
        try:
            from nodes.common.world_model_substrate.usefulness_coords import (
                default_usefulness_embedder,
            )
        except ImportError:
            return None
        self._embedder = default_usefulness_embedder(dim=1024)
        return self._embedder

    def attest_usage(
        self,
        caller_id: str | None,
        judgments: list[dict[str, Any]],
        context_text: str,
    ) -> dict[str, Any]:
        """Record COGNITIVE attestations for a closed work item.

        Unlike mechanical receipts (per call, worthless to mint), an
        attestation is a deliberate reflection step: the caller judges which
        registered tools served the work it just finished. This is the only
        usage the mint counts (docs/tool_substrate.md — two receipt tiers).

        ``judgments`` — list of ``{"tool", "ok", "score"?, "note"?}``. Each
        tool is resolved via ``self.resolve`` and scoping-checked; a tool the
        caller may not use (or can't resolve) is skipped and reported, never
        failing the batch. ``context_text`` (the work item) is embedded ONCE
        into ``problem_coords``; if the substrate package is unavailable the
        rows still record locally with ``problem_coords=[]`` — never crash the
        agent."""
        caller = caller_id if caller_id else OWNER_AUTHOR

        # Embed the work-item context once. Degrade to [] if the substrate
        # package (embedder or blob store) is unavailable — never crash.
        problem_coords: list[float] = []
        embedder = self._usefulness_embedder()
        if embedder is not None and context_text:
            try:
                problem_coords = [float(v) for v in embedder(context_text)]
            except Exception as exc:
                log.warning("attest_usage embed failed: %s", exc)

        try:
            blobs: Any = self._blob_store()
        except RuntimeError:
            blobs = None

        attested = 0
        skipped: list[dict[str, Any]] = []
        ts = int(time.time())
        self._dir.mkdir(parents=True, exist_ok=True)

        for judgment in judgments:
            tool_ref = str(judgment.get("tool") or "")
            record = self.resolve(tool_ref)
            if record is None:
                skipped.append({"tool": tool_ref, "error": "tool not found"})
                continue
            if not self.allowed(caller, record):
                skipped.append({"tool": tool_ref,
                                "error": "not in author lineage"})
                continue

            ok = bool(judgment.get("ok"))
            raw_score = judgment.get("score")
            score = 0.0
            if raw_score is not None:
                try:
                    score = max(0.0, min(1.0, float(raw_score)))
                except (TypeError, ValueError):
                    score = 0.0
            # v3 per-charter-axis signed review scores (spec Decision
            # 2026-07-08): {axis_id: [-1, +1]}, unknown/garbage entries
            # dropped, axes the reviewer didn't score simply absent.
            axes: dict[str, float] = {}
            raw_axes = judgment.get("axes")
            if isinstance(raw_axes, dict):
                for axis_id, value in raw_axes.items():
                    try:
                        axes[str(axis_id)] = max(-1.0, min(1.0, float(value)))
                    except (TypeError, ValueError):
                        continue
            note = judgment.get("note")

            review_digest = ""
            if note and blobs is not None:
                review = {
                    "kind": "tool_review",
                    "manifest_digest": record.digest,
                    "caller": caller,
                    "ok": ok,
                    "score": score,
                    "note": str(note),
                    "context": context_text[:2000],
                    "ts": ts,
                }
                if axes:
                    review["axes"] = dict(sorted(axes.items()))
                try:
                    review_digest = blobs.add_json(review) or ""
                except Exception as exc:
                    log.warning("attestation review blob failed: %s", exc)

            self._receipt_seq += 1
            row = {
                "seq": self._receipt_seq,
                "ts": ts,
                "manifest_digest": record.digest,
                "tool_name": record.name,
                "tool_author": record.author,
                "caller": caller,
                "ok": ok,
                "score": score,
                "review_digest": review_digest,
            }
            if axes:
                row["axes"] = dict(sorted(axes.items()))
            # Stamp the active harness distro (docs/tool_substrate.md —
            # loadout digest is atomic with the attestation). Only when set.
            if self.active_loadout:
                row["loadout"] = self.active_loadout
            receipt_digest = ""
            if blobs is not None:
                try:
                    receipt_digest = blobs.add_json(row) or ""
                except Exception as exc:
                    log.warning("attestation blob failed: %s", exc)
            row["receipt_digest"] = receipt_digest

            with self._attestations_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

            if self.event_sink is not None:
                event = {
                    "kind": "tool_used",
                    "seq": self._receipt_seq,
                    # 0x identity on the consensus rail (see
                    # _record_receipt).
                    "author_agent": self._consensus_identity(caller),
                    "manifest_digest": record.digest,
                    "tool_author": record.author,
                    "receipt_digest": receipt_digest,
                    "ok": ok,
                    "fee_atn": 0.0,
                    "attested": True,
                    "score": score,
                    "problem_coords": list(problem_coords),
                    "review_digest": review_digest,
                }
                if axes:
                    event["axes"] = dict(sorted(axes.items()))
                if self.active_loadout:
                    event["loadout"] = self.active_loadout
                try:
                    self.event_sink(event)
                except Exception as exc:
                    log.warning("attested tool_used event sink failed: %s", exc)

            attested += 1

        return {"attested": attested, "skipped": skipped}

    def recent_attestations(
        self, digest: str, *, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """The most recent local attestation rows for one manifest digest
        (newest first), with review-note text resolved from the blob
        store when available. Feeds the Substrate view's review drawer
        (v3). Local-plane only — this daemon's attestations.jsonl."""
        digest = str(digest or "").strip().lower()
        if not digest or not self._attestations_path.exists():
            return []
        try:
            limit = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            limit = 50
        rows: list[dict[str, Any]] = []
        try:
            with self._attestations_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("manifest_digest") != digest:
                        continue
                    rows.append(row)
        except OSError:
            return []
        rows = rows[-limit:][::-1]  # newest first

        blobs: Any = None
        try:
            blobs = self._blob_store()
        except RuntimeError:
            blobs = None
        out: list[dict[str, Any]] = []
        for row in rows:
            entry: dict[str, Any] = {
                "ts": row.get("ts"),
                "caller": row.get("caller", ""),
                "ok": bool(row.get("ok")),
                "score": row.get("score"),
                "axes": row.get("axes") or {},
                "note": "",
            }
            review_digest = str(row.get("review_digest") or "")
            if review_digest and blobs is not None:
                try:
                    review = blobs.get_json(review_digest)
                    if isinstance(review, dict):
                        entry["note"] = str(review.get("note") or "")[:2000]
                except Exception:  # noqa: BLE001 — note is best-effort
                    pass
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Adoption (docs/tool_substrate.md — Adoption rail)
    # ------------------------------------------------------------------

    async def propose_adoption(
        self, caller_id: str | None, digest: str, reason: str = "",
    ) -> dict[str, Any]:
        """Agent-side half of the install rail: PROPOSE, never install.

        Adoption is foreign code entering the host — the one place a
        real approval queue is legitimate. The proposal fetches and
        digest-verifies the manifest, then packages what the owner
        needs to decide: declared capabilities (the sandbox policy it
        will run under), signature presence, and the tool's inspection
        activity (v4.1: inspection is a review, not a gate — the carried
        vetting fields now count code reads, they do not greenlight or
        unlock anything) when the close state is available. Nothing
        executes until approve_adoption.
        """
        caller = caller_id if caller_id else OWNER_AUTHOR
        digest = digest.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef"
                                    for c in digest):
            return {"error": "digest must be the 64-hex manifest digest "
                             "(probe the substrate to find one)"}
        if digest in self._records:
            return {"error": "already available locally"}
        existing = self._proposals.get(digest)
        if existing is not None and existing.get("status") == "pending":
            return {"status": "pending", "digest": digest,
                    "note": "already proposed; awaiting owner approval"}

        payload = await self.fetch_payload(digest)
        from nodes.common.world_model_substrate.tool_manifest import (
            is_tool_manifest,
        )
        if payload is None or not is_tool_manifest(payload):
            return {"error": f"cannot resolve manifest {digest[:16]}: not "
                             "fetchable from the network"}
        if str(payload.get("trust_class") or "") != "pinned":
            return {"error": "only pinned tools can be adopted — attested "
                             "tools lean on connectors/credentials that "
                             "don't transfer between daemons"}

        vet_status: dict[str, Any] | None = None
        if self.vet_status_provider is not None:
            try:
                vet_status = self.vet_status_provider(digest)
            except Exception as exc:
                log.warning("vet status lookup failed: %s", exc)

        caps = dict(payload.get("capabilities") or {})
        proposal = {
            "digest": digest,
            "name": str(payload.get("name") or ""),
            "description": str(payload.get("description") or ""),
            "author": str(payload.get("author") or ""),
            "proposed_by": caller,
            "reason": str(reason or ""),
            "ts": int(time.time()),
            "status": "pending",
            "capabilities": caps,
            "provenance": {
                "signed": self._verify_manifest_sig(payload),
                "greenlit": (bool(vet_status.get("greenlit"))
                             if vet_status else None),
                "busted": (bool(vet_status.get("busted"))
                           if vet_status else None),
                "vets": (len(vet_status.get("vets") or {})
                         if vet_status else None),
                "dependencies": len(payload.get("dependencies") or []),
            },
        }
        self._proposals[digest] = proposal
        self._save_proposals()
        log.info("adoption proposed: %s (%s) by %s",
                 proposal["name"], digest[:16], caller)
        return {"status": "pending", "digest": digest,
                "capabilities": caps, "provenance": proposal["provenance"],
                "note": "proposed; awaiting owner approval"}

    def list_adoption_proposals(
        self, status: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = sorted(self._proposals.values(), key=lambda r: -r.get("ts", 0))
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return [dict(r) for r in rows]

    async def approve_adoption(self, digest: str) -> dict[str, Any]:
        """Owner-side half (WS surface, never an agent tool): verify,
        install as an ADOPTED record — original author preserved on the
        manifest (their mint keeps flowing from our attestations),
        adopter-lineage scoping via local_author, contained execution
        via origin="adopted"."""
        proposal = self._proposals.get(digest)
        if proposal is None or proposal.get("status") != "pending":
            return {"error": "no pending proposal for that digest"}
        payload = await self.fetch_payload(digest)
        from nodes.common.world_model_substrate.tool_manifest import (
            is_tool_manifest,
        )
        if payload is None or not is_tool_manifest(payload):
            return {"error": "manifest blob no longer resolvable"}
        code_digest = str(payload.get("code_digest") or "")
        code_raw = await self.fetch_bytes(code_digest) if code_digest else None
        if code_raw is None:
            return {"error": "code blob not fetchable — refusing to install "
                             "a manifest whose code can't be pinned locally"}

        # Provenance is LOAD-BEARING here, not decoration. propose_adoption
        # assembles {signed, greenlit, busted, vets} for the owner to read,
        # but approval used to consult NONE of it — a manifest carrying a
        # WRONG signature installed exactly as readily as a valid one.
        #
        # Re-verify against the payload we just re-fetched (not the stored
        # proposal, which was written when the blob may have differed):
        #   signed is False -> the signature is PRESENT but does not recover
        #     to the claimed author. _verify_manifest_sig calls this what it
        #     is: a re-attribution attempt. Refuse.
        #   signed is None  -> unsigned / verification unavailable. ALLOWED,
        #     deliberately: most tools are unsigned today and refusing them
        #     would break the rail. Absence of a claim is not a false claim.
        # A CON-busted tool is likewise refused: the network has already
        # produced reproducible evidence against it.
        if self._verify_manifest_sig(payload) is False:
            return {"error": "manifest signature does not recover to its "
                             "claimed author — refusing to install "
                             "(re-attribution attempt)"}
        prov = (proposal.get("provenance") or {}) if isinstance(
            proposal.get("provenance"), dict) else {}
        if prov.get("busted") is True:
            return {"error": "tool is CON-busted on the substrate — refusing "
                             "to install; clear the evidence claim first"}

        record = ToolRecord(
            digest=digest,
            manifest=payload,
            local_author=str(proposal.get("proposed_by") or ""),
            registered_ts=int(time.time()),
            origin="adopted",
        )
        self._records[digest] = record
        proposal["status"] = "approved"
        proposal["resolved_ts"] = int(time.time())
        self._persist()
        self._save_proposals()
        log.info("adoption approved: %s (%s) for %s",
                 record.name, digest[:16], record.local_author)
        return {"digest": digest, "name": record.name,
                "origin": "adopted", "author": record.author,
                "adopted_for": record.local_author}

    def reject_adoption(self, digest: str, reason: str = "") -> dict[str, Any]:
        proposal = self._proposals.get(digest)
        if proposal is None or proposal.get("status") != "pending":
            return {"error": "no pending proposal for that digest"}
        proposal["status"] = "rejected"
        proposal["resolved_ts"] = int(time.time())
        if reason:
            proposal["reject_reason"] = str(reason)
        self._save_proposals()
        return {"digest": digest, "status": "rejected"}

    def _verify_manifest_sig(self, manifest: dict[str, Any]) -> bool | None:
        """True = signature recovers to the manifest author OR its
        author_pubkey; False = present but WRONG (red flag:
        re-attribution attempt); None = no signature / verification
        unavailable. Content addressing already guarantees integrity —
        this proves the AUTHORSHIP claim. author_pubkey matters when
        the author is the owner WALLET (unregistered agent's tool,
        household-claimed rewards): the authoring agent's key signs,
        its address is stamped as author_pubkey inside the signed
        payload, so recovery to it still binds code to signer."""
        sig = manifest.get("author_sig")
        if not sig:
            return None
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            from nodes.common.world_model_substrate.tool_manifest import (
                canonical_manifest_bytes,
            )
            recovered = Account.recover_message(
                encode_defunct(canonical_manifest_bytes(manifest)),
                signature=bytes.fromhex(str(sig).removeprefix("0x")),
            ).lower()
            return recovered in (
                str(manifest.get("author") or "").lower(),
                str(manifest.get("author_pubkey") or "").lower(),
            )
        except Exception as exc:
            log.debug("manifest sig verification unavailable: %s", exc)
            return None

    def _load_proposals(self) -> dict[str, dict[str, Any]]:
        if not self._proposals_path.exists():
            return {}
        try:
            data = json.loads(self._proposals_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()
                        if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("adoption proposals unreadable: %s", exc)
        return {}

    def _save_proposals(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._proposals_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._proposals, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self._proposals_path)

    # ------------------------------------------------------------------
    # Vetting (docs/tool_substrate.md — Vetting section)
    # ------------------------------------------------------------------

    async def fetch_payload(self, digest: str) -> dict[str, Any] | None:
        """JSON blob by digest: local blob store first, then the network
        fetcher. Network bytes are digest-verified and cached locally."""
        raw = await self.fetch_bytes(digest)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    async def fetch_bytes(self, digest: str) -> bytes | None:
        import hashlib

        try:
            blobs = self._blob_store()
        except RuntimeError:
            return None
        raw = blobs.get_bytes(digest)
        if raw is not None:
            return raw
        if self.blob_fetcher is None:
            return None
        try:
            fetched = await self.blob_fetcher(digest)
        except Exception as exc:
            log.warning("blob fetch failed for %s: %s", digest[:16], exc)
            return None
        if not fetched:
            return None
        if hashlib.sha256(fetched).hexdigest() != digest:
            log.warning("blob fetch for %s returned wrong content", digest[:16])
            return None
        blobs.add_bytes(fetched)
        return fetched

    async def vet_tool(
        self,
        caller_id: str | None,
        tool_ref: str,
        verdict: str | None = None,
        report: str = "",
        axes: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Inspection-review flow (v4.1 gradient trust — memory
        tool-economy-v4-gradient-trust), two steps in one rail:

        - ``verdict=None`` — INSPECT: resolve the manifest (locally or
          fetched from the network by digest) and return it with the
          pinned source code, so the validator can actually read what
          it is judging.
        - ``verdict="pass"|"fail"`` — ATTEST: record the inspection
          locally and emit the consensus ``tool_used`` event with
          ``vet=True``. ``report`` (required) is blob-stored as the
          inspection report — inspectors show their work; the report is
          what a later exploit CON gets argued against. Optional
          per-charter-axis scores (``axes`` — ``{axis_id: [-1,1]}``,
          same shape as ``attest_usage``) ride the event so the
          inspection moves the tool's public position/rating exactly
          like a usage review — weighted at close by the inspector's
          reputation share × credibility. An inspection MINTS NOTHING;
          it only moves position/rating.

        Self-vetting is rejected here for the local-author case; the
        close voids it structurally anyway (fleet + wire exclusions).
        An inspection is not usage and never touches the receipt ledgers.
        """
        caller = caller_id if caller_id else OWNER_AUTHOR

        record = self.resolve(tool_ref)
        manifest: dict[str, Any] | None = None
        digest = ""
        if record is not None:
            manifest, digest = record.manifest, record.digest
        elif len(tool_ref) == 64 and all(
            c in "0123456789abcdef" for c in tool_ref
        ):
            payload = await self.fetch_payload(tool_ref)
            from nodes.common.world_model_substrate.tool_manifest import (
                is_tool_manifest,
            )
            if payload is not None and is_tool_manifest(payload):
                manifest, digest = payload, tool_ref
        if manifest is None:
            return {"error": f"cannot resolve manifest {tool_ref[:16]}: not "
                             "registered here and not fetchable from the "
                             "network"}

        if str(manifest.get("trust_class") or "") != "pinned":
            return {"error": "only pinned tools are vetted — attested/"
                             "connector tools have no hash-locked code to "
                             "read (they are also not mint-eligible)"}

        code_digest = str(manifest.get("code_digest") or "")
        code_raw = await self.fetch_bytes(code_digest) if code_digest else None

        if verdict is None:
            return {
                "digest": digest,
                "manifest": manifest,
                "code": (code_raw.decode("utf-8", errors="replace")
                         if code_raw is not None else None),
                "note": ("review the code against the manifest, then call "
                         "again with verdict 'pass' or 'fail' and a report"),
            }

        if verdict not in ("pass", "fail"):
            return {"error": "verdict must be 'pass' or 'fail'"}
        if not report.strip():
            return {"error": "a vet requires a report — state what you "
                             "checked and what you found"}
        if record is not None and record.author_id == caller:
            return {"error": "you cannot vet your own tool"}
        consensus_caller = self._consensus_identity(caller)
        if consensus_caller == str(manifest.get("author") or ""):
            return {"error": "you cannot vet your own tool"}
        if code_raw is None:
            return {"error": "code blob unavailable — a vet without the "
                             "code read is worthless; retry when the blob "
                             "is fetchable"}

        ok = verdict == "pass"
        ts = int(time.time())
        # Per-charter-axis inspection scores (v4.1): {axis_id: [-1,1]},
        # same normalization as attest_usage — garbage entries dropped,
        # unscored axes simply absent. These ride the vet event so the
        # inspection moves position/rating like a usage review.
        norm_axes: dict[str, float] = {}
        if isinstance(axes, dict):
            for axis_id, value in axes.items():
                try:
                    norm_axes[str(axis_id)] = max(-1.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
        review_digest = ""
        try:
            blobs: Any = self._blob_store()
        except RuntimeError:
            blobs = None
        if blobs is not None:
            try:
                report_blob = {
                    "kind": "tool_vet_report",
                    "manifest_digest": digest,
                    "code_digest": code_digest,
                    "validator": consensus_caller,
                    "ok": ok,
                    "report": str(report),
                    "ts": ts,
                }
                if norm_axes:
                    report_blob["axes"] = dict(sorted(norm_axes.items()))
                review_digest = blobs.add_json(report_blob) or ""
            except Exception as exc:
                log.warning("vet report blob failed: %s", exc)

        self._receipt_seq += 1
        row = {
            "seq": self._receipt_seq,
            "ts": ts,
            "manifest_digest": digest,
            "tool_name": str(manifest.get("name") or ""),
            "tool_author": str(manifest.get("author") or ""),
            "caller": caller,
            "ok": ok,
            "vet": True,
            "review_digest": review_digest,
        }
        if norm_axes:
            row["axes"] = dict(sorted(norm_axes.items()))
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._attestations_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        if self.event_sink is not None:
            event = {
                "kind": "tool_used",
                "seq": self._receipt_seq,
                "author_agent": consensus_caller,
                "manifest_digest": digest,
                "tool_author": str(manifest.get("author") or ""),
                "receipt_digest": review_digest,
                "ok": ok,
                "fee_atn": 0.0,
                "vet": True,
                "review_digest": review_digest,
            }
            if norm_axes:
                event["axes"] = dict(sorted(norm_axes.items()))
            try:
                self.event_sink(event)
            except Exception as exc:
                log.warning("vet event sink failed: %s", exc)

        return {"digest": digest, "verdict": verdict,
                "review_digest": review_digest}

    # ------------------------------------------------------------------
    # Evidence-replay (docs/tool_substrate.md — Evidence section)
    # ------------------------------------------------------------------

    @staticmethod
    def _result_digest(result: dict[str, Any]) -> str:
        """sha256 of a tool result's canonical ``{"result": ...}`` payload.

        The SAME canonicalization a CON author used to compute
        ``expected_digest`` / ``actual_digest`` — key-sorted JSON of the
        inner result value only (errors are compared textually, never by
        digest, since an error message is not a stable success output)."""
        import hashlib

        payload = result.get("result")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True,
                       default=str).encode("utf-8")).hexdigest()

    async def replay_evidence(
        self,
        con_evidence: dict[str, Any],
        manifest_digest: str,
    ) -> dict[str, Any]:
        """Re-run a pinned tool with a CON's evidence args and compare.

        DAEMON-LOCAL and VOLUNTARY (docs/tool_substrate.md — Evidence):
        a validator replays the failing invocation carried on an
        evidence-bearing CON to decide whether to SUPPORT it. This never
        touches the close math — a confirmed replay is grounds for the
        validator to post a normal support sprout under the CON, and the
        deterministic close prices that post like any other. Evidence
        recruits verification; it does not weight standing.

        ``con_evidence`` — ``{"args_json", "expected_digest" |
        "expected_error", "actual_digest"?}``. The tool runs through the
        SAME execution path as a normal call (``self.call`` → pinned
        subprocess), so an ADOPTED tool replays under its capability
        guard exactly as it would in production — the guard's own
        hard-fail IS reproducible evidence.

        Returns ``{"confirmed": bool, "kind", "manifest_digest",
        "observed_digest"?, "observed_error"?, "expected"..., "note"}``.
        ``confirmed`` is True when the replay reproduces the CON's
        claimed failure:

          - expected_error: the replay errored (any error confirms a
            "this input breaks it" claim — the specific text is advisory);
          - expected_digest: the replay SUCCEEDED but produced a
            DIFFERENT result digest than the manifest's contract claims
            (wrong-answer evidence: expected == the correct digest the
            CON says the tool fails to produce).
        """
        record = self.resolve(manifest_digest) or self._records.get(
            manifest_digest)
        if record is None:
            return {"confirmed": False, "error": "tool not resolvable locally "
                    "— adopt or register it before replaying its evidence"}
        if not record.manifest.get("code_digest"):
            return {"confirmed": False,
                    "error": "only pinned tools have replayable evidence"}

        raw_args = con_evidence.get("args_json")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                return {"confirmed": False,
                        "error": "evidence args_json is not valid JSON"}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            return {"confirmed": False,
                    "error": "evidence must carry args_json (object or "
                             "JSON string)"}

        expected_error = con_evidence.get("expected_error")
        expected_digest = con_evidence.get("expected_digest")
        if not expected_error and not expected_digest:
            return {"confirmed": False,
                    "error": "evidence must state expected_error or "
                             "expected_digest to be checkable"}

        # Replay through the ordinary call path (guard applies for adopted
        # code). caller_id=None runs as owner-sanctioned local verification;
        # scoping is the CON reader's concern, not the replay's.
        result = await self.call(record, arguments, caller_id=None)

        if "error" in result:
            observed_error = str(result["error"])
            # An error confirms an "input breaks it" CON regardless of the
            # exact message (messages vary by platform; the failure does
            # not). A wrong-answer CON (expected_digest) is NOT confirmed
            # by an error — that is a different, unclaimed failure mode.
            confirmed = bool(expected_error)
            return {
                "confirmed": confirmed,
                "kind": "error",
                "manifest_digest": record.digest,
                "observed_error": observed_error[:2000],
                "expected_error": (str(expected_error)[:2000]
                                   if expected_error else None),
                "note": ("replay errored as the CON claims"
                         if confirmed else
                         "replay errored, but the CON claimed a wrong-answer "
                         "(expected_digest) — not the failure observed"),
            }

        observed_digest = self._result_digest(result)
        if expected_error:
            # CON claimed the input errors, but the replay succeeded.
            return {
                "confirmed": False,
                "kind": "result",
                "manifest_digest": record.digest,
                "observed_digest": observed_digest,
                "note": "replay succeeded — the CON's claimed error did "
                        "not reproduce",
            }

        # Wrong-answer CON: confirmed when the replay's digest differs
        # from the correct digest the CON says the tool fails to produce.
        confirmed = observed_digest != str(expected_digest)
        # If the CON recorded the buggy actual_digest it observed, a
        # matching replay is stronger corroboration (fully reproducible).
        actual_digest = con_evidence.get("actual_digest")
        reproduced = (bool(actual_digest)
                      and observed_digest == str(actual_digest))
        return {
            "confirmed": confirmed,
            "kind": "result",
            "manifest_digest": record.digest,
            "observed_digest": observed_digest,
            "expected_digest": str(expected_digest),
            "reproduced_actual": reproduced,
            "note": ("replay produced a different result than the correct "
                     "digest — wrong-answer CON confirmed" if confirmed else
                     "replay matched the expected-correct digest — the CON "
                     "does not reproduce"),
        }

    async def check_evidence(
        self,
        caller_id: str | None,
        manifest_digest: str,
        evidence: dict[str, Any],
        *,
        con_node_id: str = "",
        support: bool = True,
    ) -> dict[str, Any]:
        """Validator verify-then-support flow (docs/tool_substrate.md —
        Evidence), one call:

          1. REPLAY the CON's evidence against the pinned tool
             (``replay_evidence``).
          2. If it CONFIRMS and ``support`` is set and a ``con_node_id``
             and a wired ``support_sink`` are present, post a normal PRO
             support sprout under the CON — recruiting the validator's
             standing behind a dispute they personally reproduced.

        No close-side math changes: the support sprout is priced by the
        deterministic close exactly like any author post. Evidence
        recruits verification; it never weights standing directly. When
        the replay does NOT confirm, nothing is posted — the validator's
        standing is not spent on a claim they could not reproduce.

        Returns the replay verdict plus, when a support post fired, a
        ``supported`` block with the new PRO node id.
        """
        verdict = await self.replay_evidence(evidence, manifest_digest)
        out: dict[str, Any] = dict(verdict)
        out["supported"] = None
        if not verdict.get("confirmed"):
            return out
        if not support or not con_node_id or self.support_sink is None:
            out["note"] = (str(verdict.get("note") or "")
                           + " — confirmed; no support posted "
                           "(support disabled or no CON node / sink)")
            return out
        claim = ("reproduced the CON's failing invocation locally "
                 f"({verdict.get('kind', 'failure')})")
        try:
            posted = self.support_sink(con_node_id, claim)
            out["supported"] = posted
        except Exception as exc:
            log.warning("evidence support post failed: %s", exc)
            out["supported"] = {"error": str(exc)}
        return out

    def attestation_summary(self) -> dict[str, dict[str, Any]]:
        """Per-digest attestation aggregates from ``attestations.jsonl``:
        ``{attested_count, ok_count, avg_score, last_ts}``. For the Tools
        screen — the mechanical receipt ledger stays separate. Vet rows
        are a different flavor and are excluded (see ``vet_summary``)."""
        summary: dict[str, dict[str, Any]] = {}
        score_totals: dict[str, float] = {}
        for row in self._iter_attestations():
            if row.get("vet"):
                continue
            digest = row.get("manifest_digest", "")
            entry = summary.setdefault(digest, {
                "attested_count": 0, "ok_count": 0,
                "avg_score": 0.0, "last_ts": 0,
            })
            entry["attested_count"] += 1
            if row.get("ok"):
                entry["ok_count"] += 1
            score_totals[digest] = score_totals.get(digest, 0.0) + float(
                row.get("score") or 0.0)
            ts = int(row.get("ts") or 0)
            if ts > entry["last_ts"]:
                entry["last_ts"] = ts
        for digest, entry in summary.items():
            count = entry["attested_count"]
            entry["avg_score"] = round(
                score_totals.get(digest, 0.0) / count, 10) if count else 0.0
        return summary

    def vet_summary(self) -> dict[str, dict[str, Any]]:
        """Per-digest vet aggregates from the local attestation log:
        ``{vet_count, pass_count, last_ts}``. Local view only — the
        authoritative candidate/greenlit state is the federated close's
        vetting carry-over."""
        summary: dict[str, dict[str, Any]] = {}
        for row in self._iter_attestations():
            if not row.get("vet"):
                continue
            digest = row.get("manifest_digest", "")
            entry = summary.setdefault(digest, {
                "vet_count": 0, "pass_count": 0, "last_ts": 0,
            })
            entry["vet_count"] += 1
            if row.get("ok"):
                entry["pass_count"] += 1
            ts = int(row.get("ts") or 0)
            if ts > entry["last_ts"]:
                entry["last_ts"] = ts
        return summary

    def _iter_attestations(self):
        if not self._attestations_path.exists():
            return
        try:
            lines = self._attestations_path.read_text(
                encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("ToolStore could not read attestations: %s", exc)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("ToolStore skipping corrupt attestation row")

    def _count_attestations(self) -> int:
        return sum(1 for _ in self._iter_attestations())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_owner(self, caller_id: str | None) -> bool:
        from .orchestrator import is_owner_caller
        return is_owner_caller(caller_id)

    def _ancestors(self, agent_id: str) -> set[str]:
        """The author's ancestor chain via parent_id (cycle-guarded)."""
        out: set[str] = set()
        current = agent_id
        for _ in range(64):
            defn = self._runtime.get_agent(current)
            parent = getattr(defn, "parent_id", None) if defn else None
            if not parent or parent in out:
                break
            out.add(parent)
            current = parent
        return out

    def _author_address(self, author: str) -> str:
        defn = self._runtime.get_agent(author)
        identity = getattr(defn, "identity", None) if defn else None
        return str(getattr(identity, "address", "") or "")

    def _owner_wallet(self) -> str:
        return str(getattr(
            getattr(getattr(self._runtime, "_config", None),
                    "autonet", None), "owner_wallet", "") or "")

    def _consensus_identity(self, local_id: str | None) -> str:
        """Map a local id to its consensus identity (0x address).

        Ruling 2026-07-24 (never unclaimable rewards): an agent's own
        address counts only once the agent is REGISTERED ON-CHAIN —
        every agent has a local keypair from birth, but Substrate's
        recordTrainingForEpoch reverts AgentNotActive for addresses it
        doesn't know, so mint keyed to an unregistered keypair is
        stranded until (if ever) the agent registers. Unregistered
        agents' tools are authored by the owner wallet instead — the
        household is the claimant. The owner ("user") → the owner
        wallet when configured. The local-id fallback only remains
        for wallet-less setups, and those manifests are PRIVATE-ONLY:
        publishing is gated on a claimable identity (see register /
        set_published)."""
        if not local_id:
            return OWNER_AUTHOR
        if local_id == OWNER_AUTHOR:
            return self._owner_wallet() or OWNER_AUTHOR
        defn = self._runtime.get_agent(local_id)
        identity = getattr(defn, "identity", None) if defn else None
        addr = str(getattr(identity, "address", "") or "")
        if addr and bool(getattr(identity, "registered_on_chain", False)):
            return addr
        return self._owner_wallet() or local_id

    def _sign(self, author: str, manifest: dict[str, Any]) -> None:
        """Sign the canonical manifest bytes with the author agent's key.

        Best-effort: owner/vendor authors have no agent key, and the
        eth_account dependency is optional — an unsigned manifest is
        valid, it just carries no cryptographic authorship proof."""
        key = None
        registry = getattr(self._runtime, "registry", None)
        if registry is not None and hasattr(registry, "get_agent_key"):
            key = registry.get_agent_key(author)
        if not key:
            return
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            from nodes.common.world_model_substrate.tool_manifest import (
                canonical_manifest_bytes,
            )
            signed = Account.sign_message(
                encode_defunct(canonical_manifest_bytes(manifest)),
                private_key=key,
            )
            manifest["author_sig"] = signed.signature.hex()
        except Exception as exc:  # optional-dep path; never block registration
            log.debug("manifest signing skipped for %s: %s", author, exc)

    def _blob_store(self) -> Any:
        if self._blobs is None:
            try:
                from nodes.common.blob_store import BlobStore
            except ImportError as exc:
                raise RuntimeError(
                    "tool registration requires the autonet substrate "
                    "package (nodes.*); it is not importable here"
                ) from exc
            self._blobs = BlobStore(data_dir=str(self._dir / "blobs"))
        return self._blobs

    # ---- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            lines = self._registry_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("ToolStore could not read %s: %s", self._registry_path, exc)
            return
        try:
            blobs = self._blob_store()
        except RuntimeError as exc:
            log.warning("ToolStore: %d persisted tools unusable: %s",
                        len(lines), exc)
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                digest = row["digest"]
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("ToolStore skipping corrupt registry row")
                continue
            manifest = blobs.get_json(digest)
            if manifest is None:
                log.warning("ToolStore: manifest blob %s... missing, dropping",
                            digest[:16])
                continue
            self._records[digest] = ToolRecord(
                digest=digest,
                manifest=manifest,
                grants=set(row.get("grants") or []),
                enabled=bool(row.get("enabled", True)),
                published=bool(row.get("published", False)),
                local_author=str(row.get("local_author") or ""),
                registered_ts=int(row.get("registered_ts") or 0),
                origin=str(row.get("origin") or "authored"),
            )

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in self._records.values():
                fh.write(json.dumps(record.to_row()) + "\n")
        os.replace(tmp, self._registry_path)
