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


def _is_address(value: str) -> bool:
    v = str(value or "")
    if len(v) != 42 or not v.lower().startswith("0x"):
        return False
    try:
        int(v[2:], 16)
    except ValueError:
        return False
    return True


MALWARE_EVIDENCE_REQUIRED = (
    "a malware flag needs evidence: say what the code does and point into "
    "it (file and line numbers, or the quoted code)")


def _charter_scores(raw: Any) -> dict[str, float]:
    """A review's charter scores, ``{axis_id: [-1, 1]}``: garbage dropped,
    unscored axes simply absent (ToolBond counts them at the default)."""
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for axis_id, value in raw.items():
            if isinstance(value, bool):
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v == v:
                out[str(axis_id)] = max(-1.0, min(1.0, v))
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
    # ``published`` is for the CURRENT jurisdiction; ``published_in`` keeps
    # every jurisdiction it was published in. Publishing in another
    # jurisdiction is a fresh act there (docs/multi_jurisdiction.md 6).
    published: bool = False
    published_in: set[str] = field(default_factory=set)
    # For ADOPTED records: the jurisdiction they were adopted from. They
    # are usable only there; switching away parks them.
    jurisdiction: str = ""
    # Local author agent id (scoping). Empty on pre-address rows —
    # author_id falls back to the manifest author. For ADOPTED records
    # this is the adopting agent — adoption scopes to the adopter's
    # lineage while manifest.author (the original 0x) keeps earning.
    local_author: str = ""
    registered_ts: int = 0
    # "authored" (default) | "adopted". Adopted records carry another
    # household's code and can never be re-published by the adopter.
    origin: str = "authored"
    # Core-replacement acknowledgment (Decision 2026-10-12). True only when
    # the OWNER approved this adoption with the explicit core_replacement
    # flag, i.e. saw "this tool replaces <names> and will see every command
    # and file your agent touches". A record whose manifest claims
    # capabilities.provides over core built-ins can never swap in without it.
    core_replacement_ack: bool = False
    # Digest of the newest signed evidence trace this tool produced while
    # replacing a core built-in (blob store). Surfaced so an accuser can copy
    # it into ToolBond.accuse(digest, evidenceCid). Telemetry, not consensus.
    last_evidence_digest: str = ""
    # Where the author's codebase lives on this machine (local state, never
    # in the manifest): registering it again snapshots a new version.
    source_path: str = ""

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
            "published_in": sorted(self.published_in),
            **({"jurisdiction": self.jurisdiction} if self.jurisdiction else {}),
            "local_author": self.local_author,
            "source_path": self.source_path,
            "registered_ts": self.registered_ts,
            "origin": self.origin,
            **({"core_replacement_ack": True}
               if self.core_replacement_ack else {}),
            **({"last_evidence_digest": self.last_evidence_digest}
               if self.last_evidence_digest else {}),
        }


# Code seeding (#46). A tool's code lives only with its author and the
# daemons that adopted it, like a torrent. An approval made while none of
# them is online is kept as approved_pending and retried: after
# ADOPTION_RETRY_BASE_S, doubling up to ADOPTION_RETRY_MAX_S, and at once when
# a new peer connects (checked every ADOPTION_PEER_POLL_S), for at most
# ADOPTION_RETRY_WINDOW_S. After that the proposal goes back to pending and
# the owner can approve it again.
ADOPTION_APPROVED_PENDING = "approved_pending"
ADOPTION_RETRY_BASE_S = 120.0
ADOPTION_RETRY_MAX_S = 600.0
ADOPTION_RETRY_WINDOW_S = 24 * 3600.0
ADOPTION_PEER_POLL_S = 15.0
_NO_CODE_HOLDER = "no holder of this tool's code is online"


class ToolStore:
    """Persistent registry of tool manifests with author-lineage scoping."""

    def __init__(self, runtime: Runtime, tools_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(tools_dir)
        self._registry_path = self._dir / "registry.jsonl"
        self._receipts_path = self._dir / "receipts.jsonl"
        self._attestations_path = self._dir / "attestations.jsonl"
        self._records: dict[str, ToolRecord] = {}
        # Adopted records from another jurisdiction: kept on disk, invisible
        # and unusable until the daemon switches back to that jurisdiction.
        self._parked: list[ToolRecord] = []
        self._blobs: Any = None  # lazy; None until substrate package needed
        self._receipt_seq = 0
        # Optional sink wired by the autonet bridge when the library service
        # is up: manifest_sink sends the published manifest's
        # tool_registered event (LibraryService.submit_tool_manifest).
        # Tool use is NOT gossiped (Decision 2026-11-07: the mint reads
        # staked ToolBond review rounds only, and use events would reveal
        # which household used which tool), and since 2026-11-08 it feeds
        # no ranking either: search orders by relevance alone.
        self.manifest_sink: Any = None
        # Optional blob announcer (callable: list of digests -> None,
        # blocking, bounded; wired to the libp2p host's DHT provide). A
        # published tool's manifest and code are announced BEFORE its
        # registration event is sent, so peers that react to the event can
        # find a holder; adopted tools are announced too (every holder
        # serves). docs/p2p_networking.md, "Blobs".
        self.blob_announcer: Any = None
        # Optional network blob fetch (async callable: digest -> bytes
        # or None). Wired to the libp2p blob resolver when the autonet
        # host is up; lets validators vet (and later adopt) manifests
        # that were published from OTHER daemons. Fetched blobs are
        # digest-verified (content addressing IS the auth) and cached
        # into the local blob store.
        self.blob_fetcher: Any = None
        # digest -> the tool's rating as of the latest close (plain means,
        # federated_reconcile.tool_rating) or None; wired by the autonet
        # service, read by search and the adopt replies.
        self.rating_provider: Any = None
        # A pruned tool's blobs stay on disk only for what this household
        # installed (Decision 2026-11-10): what to run is its own policy.
        from nodes.common import pruned_tools
        pruned_tools.add_keeper(self._keeps_blob)
        # Optional sanction lookup (callable: address -> entry | None, may
        # block on a chain read): an agent may not propose adopting a tool
        # whose author's household the DAO has sanctioned; the owner sees
        # the sanction on the proposal instead (docs/accountability.md).
        self.sanction_check: Any = None
        # Adoption proposals (docs/tool_substrate.md — Adoption rail):
        # agent proposes, OWNER approves per-tool. The queue is the one
        # legitimate approval queue in the tool economy.
        self._proposals_path = self._dir / "adoption_proposals.json"
        self._proposals: dict[str, dict[str, Any]] = self._load_proposals()
        # Approved adoptions waiting for their code (#46): a background
        # retry with backoff, also woken when a new peer connects.
        # peer_lister (callable -> iterable of connected peer ids) is wired
        # by the autonet service; absent = timer retries only.
        self.peer_lister: Any = None
        self.adoption_retry_base_s = ADOPTION_RETRY_BASE_S
        self.adoption_retry_max_s = ADOPTION_RETRY_MAX_S
        self.adoption_retry_window_s = ADOPTION_RETRY_WINDOW_S
        self.adoption_poll_s = ADOPTION_PEER_POLL_S
        self._adopt_lock: asyncio.Lock | None = None
        self._retry_task: asyncio.Task | None = None
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
        novelty: dict[str, Any] | None = None,
        archive: bytes | None = None,
        tree_digest: str = "",
        source: dict[str, Any] | None = None,
        source_path: str = "",
        session: dict[str, Any] | None = None,
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

        ``novelty`` (Decision 2026-10-13): the ``{statement, prior_art}``
        claim publishing requires. It is part of the signed manifest, so it
        is supplied here (or stamped later by ``restamp_novelty``).
        ``version_of`` must name a predecessor by the SAME author.
        """
        from nodes.common.library.tool_manifest import (
            build_tool_manifest, normalize_novelty, normalize_session,
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
                # swappable. Declaring is not being granted: a swap needs the
                # core-replacement approval, a ToolBond bond and a per-agent
                # loadout binding (runtime/shell_provider.py, 2026-10-12).
                try:
                    from .shell_tools import SHELL_TOOL_EXECUTORS as _SHELL
                    from .harness_policy import POLICY_HOOKS as _HOOKS
                    allowed = frozenset(_SHELL) | frozenset(_HOOKS)
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
        code_format = ""
        if archive is not None:
            # A codebase tool (atn/tool_tree.py): the archive is the blob,
            # the tree digest names its content.
            if code:
                raise ValueError("give either code or a codebase path, not both")
            if not tree_digest or not entrypoint:
                raise ValueError("a codebase tool needs its tree digest and entrypoint")
            code_digest = blobs.add_bytes(archive)
            code_format = "tree"
        elif code:
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

        if version_of:
            err = self.successor_error(version_of, consensus_author, author)
            if err:
                raise ValueError(err)
        if novelty is not None:
            novelty = normalize_novelty(novelty)
            # A new version is always compared against the one it replaces:
            # its predecessor joins prior_art (Decision 2026-11-07).
            pred = str(version_of or "").strip().lower().removeprefix("0x")
            if pred and isinstance(novelty, dict) and pred not in novelty.get("prior_art", []):
                novelty = {**novelty, "prior_art": [pred, *novelty.get("prior_art", [])]}

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
                    and (m.get("tree_digest") or "") == (tree_digest if code_format else "")
                    and (m.get("source") or None) == (source or None)
                    and (m.get("entrypoint") or "") == entrypoint
                    and (m.get("provider") or "") == provider
                    and (m.get("connector_id") or "") == connector_id
                    and m.get("description") == description
                    and list(m.get("dependencies") or []) == deps
                    and (m.get("capabilities") or None) == (capabilities or None)
                    and m.get("version_of") == version_of
                    and (m.get("novelty") or None) == (novelty or None)
                    and (m.get("session") or None) == (
                        normalize_session(session) if session else None)
                    and m.get("input_schema") == input_schema):
                if publish and not existing.published:
                    self.set_published(existing.digest, True)
                if source_path and existing.source_path != source_path:
                    existing.source_path = source_path
                    self._persist()
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
            code_format=code_format,
            tree_digest=tree_digest if code_format else "",
            source=source if code_format else None,
            provider=provider,
            connector_id=connector_id,
            version_of=version_of,
            dependencies=deps or None,
            capabilities=capabilities,
            created_ts=int(time.time()),
            novelty=novelty,
            session=session,
        )
        self._sign(author, manifest)

        digest = blobs.add_json(manifest)
        record = ToolRecord(
            digest=digest,
            manifest=manifest,
            published=bool(publish),
            published_in=({self._jurisdiction} if publish and self._jurisdiction
                          else set()),
            local_author=author,
            registered_ts=int(time.time()),
            source_path=source_path,
        )
        self._records[digest] = record
        self._persist()
        log.info("registered tool %r (%s, %s, %s) by %s -> %s",
                 name, trust_class, "code" if code_digest else connector_id,
                 "published" if publish else "private", author, digest[:16])
        if record.published and self.manifest_sink is not None:
            # Consensus author (0x): same identity the tool_registered
            # event's author_agent and manifest_meta carry.
            self._register_published(digest, manifest, consensus_author, name)
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

    def successor_error(self, version_of: str, consensus_author: str,
                        local_author: str = "") -> str:
        """Why ``version_of`` may not be claimed by ``consensus_author``
        ("" when it may). Decision 2026-10-13 item 6: only the original
        author publishes a successor (which carries the predecessor's
        reviews and ranking); everyone else publishes a new tool with its
        own novelty claim. Compared on the CONSENSUS author (0x), the
        identity peers can verify. Vetted status never carries: that is
        ToolBond state keyed by digest, so a new digest starts unvetted."""
        record = self.resolve(version_of) or self._records.get(version_of)
        payload = record.manifest if record is not None else None
        if payload is None:
            try:
                payload = self._blob_store().get_json(version_of)
            except Exception:  # noqa: BLE001
                payload = None
        if not isinstance(payload, dict) or payload.get("kind") != "tool_manifest":
            return (f"version_of {str(version_of)[:16]}... is not a tool "
                    "manifest known to this daemon; its author cannot be "
                    "checked")
        pred_author = str(payload.get("author") or "")
        if pred_author.lower() != str(consensus_author or "").lower():
            return ("version_of may only name your own tool: "
                    f"{str(version_of)[:16]}... is authored by {pred_author!r}. "
                    "Publish a new tool with its own novelty claim instead")
        return ""

    def restamp_novelty(self, digest: str,
                        novelty: dict[str, Any]) -> dict[str, Any]:
        """The manifest of ``digest`` with ``novelty`` stamped in, as its own
        record. The claim is inside the signed manifest, so stamping it is a
        NEW digest (same code, schema, author and created_ts). The private
        original is retired (grants migrate) unless another record depends
        on it. Deterministic: re-stamping the same claim reuses the record.
        Returns ``{"digest", "manifest", "replaced"}`` or raises ValueError.
        """
        from nodes.common.library.tool_manifest import (
            normalize_novelty, validate_manifest,
        )
        return self._restamp(digest, {"novelty": normalize_novelty(novelty)},
                             "novelty claim")

    def restamp_author(self, digest: str) -> dict[str, Any]:
        """The manifest of ``digest`` re-authored as its local agent's
        current consensus identity (its 0x address, once registered on
        chain), so a tool registered before the agent was registered can be
        published without registering it again. Same rules as
        ``restamp_novelty``."""
        record = self._records.get(digest)
        if record is None:
            raise ValueError(f"Unknown tool: {digest[:16]}")
        author = self._consensus_identity(record.local_author)
        updates: dict[str, Any] = {"author": author}
        pubkey = self._author_address(record.local_author)
        if pubkey:
            updates["author_pubkey"] = pubkey
        return self._restamp(digest, updates, "author")

    def _restamp(self, digest: str, updates: dict[str, Any],
                 what: str) -> dict[str, Any]:
        from nodes.common.library.tool_manifest import (
            validate_manifest,
        )
        record = self._records.get(digest)
        if record is None:
            raise ValueError(f"Unknown tool: {digest[:16]}")
        if record.origin == "adopted":
            raise ValueError("adopted tools cannot be re-published")
        if record.published:
            raise ValueError("a published manifest is immutable; register a "
                             "successor (version_of) with the change")
        manifest = {k: v for k, v in record.manifest.items()
                    if k != "author_sig"}
        manifest.update(updates)
        errors = validate_manifest(manifest)
        if errors:
            raise ValueError("invalid tool manifest: " + "; ".join(errors))
        self._sign(record.local_author, manifest)
        new_digest = self._blob_store().add_json(manifest)
        if new_digest == digest:
            return {"digest": digest, "manifest": record.manifest,
                    "replaced": ""}
        existing = self._records.get(new_digest)
        if existing is None:
            existing = ToolRecord(
                digest=new_digest, manifest=manifest,
                grants=set(record.grants), enabled=record.enabled,
                local_author=record.local_author,
                registered_ts=int(time.time()), origin=record.origin,
                source_path=record.source_path)
            self._records[new_digest] = existing
        referenced = any(digest in (r.manifest.get("dependencies") or [])
                         for r in self._records.values())
        replaced = ""
        if not referenced:
            existing.grants.update(record.grants)
            self._records.pop(digest, None)
            replaced = digest
        self._persist()
        log.info("stamped %s on %r: %s -> %s", what, record.name,
                 digest[:16], new_digest[:16])
        return {"digest": new_digest, "manifest": existing.manifest,
                "replaced": replaced}

    def reaches_connector(self, digest: str | None) -> bool:
        """True if ``digest`` is connector-backed or (transitively, bounded)
        declares a connector-backed dependency."""
        seen: set[str] = set()
        frontier = [digest] if digest else []
        for _ in range(8):                    # composition depth is <= 4
            nxt: list[str] = []
            for d in frontier:
                rec = self.resolve(d)
                if rec is None or rec.digest in seen:
                    continue
                seen.add(rec.digest)
                manifest = rec.manifest or {}
                if manifest.get("connector_id"):
                    return True
                nxt.extend(x for x in (manifest.get("dependencies") or [])
                           if isinstance(x, str))
            if not nxt:
                break
            frontier = nxt
        return False

    @property
    def _jurisdiction(self) -> str:
        """The jurisdiction this process runs ("<chain_id>:<dao>"), or "" in
        chain-free runs."""
        return str(getattr(self._runtime, "jurisdiction", "") or "")

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
        if self._jurisdiction:
            if record.published:
                record.published_in.add(self._jurisdiction)
            else:
                record.published_in.discard(self._jurisdiction)
        self._persist()
        if record.published and self.manifest_sink is not None:
            self._register_published(record.digest, record.manifest, record.author,
                                     record.name)
        return True

    @staticmethod
    def blob_digests(digest: str, manifest: dict[str, Any]) -> list[str]:
        """The blobs a peer needs to run a tool: its manifest and code."""
        out = [str(digest)]
        code = str(manifest.get("code_digest") or "")
        if code:
            out.append(code)
        return out

    def held_blob_digests(self) -> list[str]:
        """Every tool blob this daemon holds and serves (published and
        adopted tools), for the host's provider records."""
        out: list[str] = []
        try:
            blobs = self._blob_store()
        except RuntimeError:
            return out
        for record in self._records.values():
            if not (record.published or record.origin == "adopted"):
                continue
            for d in self.blob_digests(record.digest, record.manifest):
                if d not in out and blobs.get_bytes(d) is not None:
                    out.append(d)
        return out

    def _announce(self, digests: list[str]) -> None:
        if self.blob_announcer is None or not digests:
            return
        try:
            self.blob_announcer(list(digests))
        except Exception as exc:  # noqa: BLE001 - announcing is best effort
            log.debug("blob announce failed: %s", exc)

    def _register_published(self, digest: str, manifest: dict[str, Any],
                            author: str, name: str) -> None:
        """Send a published tool's registration (manifest_sink), its blobs
        announced first when an announcer is wired. With a running event
        loop the announce runs in a worker thread and the registration
        follows it, so the caller is not held up; without one, inline."""
        def register() -> None:
            try:
                self.manifest_sink(manifest, author)
            except Exception as exc:
                log.warning("manifest sink failed for %s: %s", name, exc)
        if self.blob_announcer is None:
            register()
            return
        digests = self.blob_digests(digest, manifest)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            self._announce(digests)
            register()
            return

        async def announce_then_register() -> None:
            await asyncio.to_thread(self._announce, digests)
            register()
        loop.create_task(announce_then_register())

    def push_all_manifests(self, skip: Any = None) -> int:
        """Re-submit every PUBLISHED manifest through manifest_sink.

        Called when the network comes up after tools were registered
        offline. ``skip(digest)`` True leaves a manifest out (the caller
        passes the anchored registrations, which need no second event).
        Private tools never leave the daemon. Returns the number pushed."""
        if self.manifest_sink is None:
            return 0
        pushed = 0
        for record in self._records.values():
            if not record.published:
                continue
            if skip is not None and skip(record.digest):
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
        an ancestor of the author, or explicitly granted by the owner. An
        external agent also may for a tool any external agent of this
        daemon adopted (``shared_adoption``).
        """
        from .agent_tools import is_owner_caller
        if is_owner_caller(caller_id):
            return True
        assert caller_id is not None
        if not record.enabled:
            return False
        if caller_id == record.author_id:
            return True
        if caller_id in record.grants:
            return True
        if self.shared_adoption(caller_id, record):
            return True
        return caller_id in self._ancestors(record.author_id)

    def _is_external(self, agent_id: str | None) -> bool:
        from .provider_identity import is_external
        if not agent_id:
            return False
        try:
            defn = self._runtime.get_agent(agent_id)
        except Exception:  # noqa: BLE001
            return False
        return defn is not None and is_external(defn)

    def shared_adoption(self, caller_id: str | None, record: ToolRecord) -> bool:
        """Adopted tools are shared across the household's external
        sessions (Decision 2026-11-09, docs/tool_substrate.md): every
        external session acts on chain as the operator (one identity, one
        review batch) and an adoption is a checkout on this machine, so a
        tool one session adopted is installed for all of them to use, check
        out and review. Native agents keep their lineage scoping: it is a
        boundary between native lineages, and a native adopter's tools stay
        its lineage's."""
        return (record.origin == "adopted"
                and self._is_external(caller_id)
                and self._is_external(record.local_author))

    def visible_to(self, caller_id: str | None) -> list[ToolRecord]:
        return [r for r in self._records.values() if self.allowed(caller_id, r)]

    # ------------------------------------------------------------------
    # Loadouts / distros (Decision 2026-10-12)
    # ------------------------------------------------------------------

    def loadout_for(self, agent_id: str | None) -> str:
        """The distro digest an attestation by ``agent_id`` must carry: the
        agent's OWN bound loadout (AgentDefinition.loadout) when set, else the
        daemon's reference distro (``active_loadout``). Never raises."""
        if agent_id:
            try:
                defn = self._runtime.get_agent(agent_id)
            except Exception:  # noqa: BLE001
                defn = None
            bound = str(getattr(defn, "loadout", "") or "") if defn else ""
            if bound:
                return bound
        return self.active_loadout

    def list_distros(self) -> list[dict[str, Any]]:
        """Distro manifests available on this daemon: pinned records whose
        code blob is a harness distro config (atn/harness_policy.py). The
        reference distro is first and flagged."""
        from .harness_policy import parse_distro_config, reference_policy
        try:
            blobs = self._blob_store()
        except RuntimeError:
            return []
        ref = reference_policy().to_dict()
        out: list[dict[str, Any]] = []
        for rec in self._records.values():
            code_digest = str(rec.manifest.get("code_digest") or "")
            if not code_digest or not rec.manifest.get("dependencies"):
                continue
            try:
                cfg = parse_distro_config(blobs.get_bytes(code_digest))
            except Exception:  # noqa: BLE001
                cfg = None
            if cfg is None:
                continue
            policy = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
            out.append({
                "digest": rec.digest,
                "name": rec.name,
                "description": str(rec.manifest.get("description") or ""),
                "author": rec.author,
                "origin": rec.origin,
                "enabled": rec.enabled,
                "is_reference": rec.digest == self.active_loadout,
                "modules": list(rec.manifest.get("dependencies") or []),
                "policy_overrides": sorted(
                    k for k, v in policy.items() if ref.get(k) != v),
                "core_providers": self._core_providers_in(rec),
            })
        out.sort(key=lambda d: (not d["is_reference"], d["name"], d["digest"]))
        return out

    def _core_providers_in(self, distro: ToolRecord) -> list[dict[str, Any]]:
        """Deps of ``distro`` that claim to provide core built-in names."""
        from .runtime.shell_provider import core_claims
        rows = []
        for dep in distro.manifest.get("dependencies") or []:
            rec = self.get(str(dep))
            if rec is None:
                continue
            claims = core_claims(rec.manifest)
            if claims and not (rec.name == "atn_shell" and rec.origin != "adopted"):
                rows.append({"digest": rec.digest, "name": rec.name,
                             "provides": claims,
                             "core_replacement_ack": rec.core_replacement_ack})
        return rows

    def get(self, digest: str) -> ToolRecord | None:
        return self._records.get(digest)

    def adopted_by(self, agent_id: str) -> list[ToolRecord]:
        """The tools ``agent_id`` adopted from other households."""
        return [r for r in self._records.values()
                if r.origin == "adopted" and r.local_author == agent_id]

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
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a registered tool. Scoping is the CALLER's job
        (``ToolRegistry.call_tool``) — this only dispatches.

        ``via`` tags the receipt with a composite's digest when this call
        was dispatched from a composite's sandbox call-rail (telemetry;
        mechanical receipts mint nothing). ``_depth`` is the composition
        nesting counter (``_COMPOSITE_MAX_DEPTH`` guard).

        There is no run gate and no containment (Decision 2026-11-09): a
        tool runs in its own subprocess with a scrubbed environment and only
        the secrets it declares (``_exec_spec``). Every execution path
        (agents, composite deps, service sales, evidence replay, core swaps)
        comes through here.

        ``extra_env`` is set by the daemon itself, never by a tool: a
        service passes its settings this way (``AUTONET_SERVICE_SETTINGS``,
        atn/service_runner.py). Pinned tools only.
        """
        manifest = record.manifest
        if manifest.get("code_digest"):
            deps = manifest.get("dependencies") or []
            if deps:
                # Interactive (composition) path — opt-in on declared deps.
                result = await self._call_pinned_interactive(
                    record, arguments, caller_id=caller_id, depth=_depth,
                    extra_env=extra_env)
            else:
                # Sealed path — byte-for-byte legacy contract.
                result = await self._call_pinned(
                    record, arguments, caller_id=caller_id,
                    extra_env=extra_env)
        elif manifest.get("connector_id"):
            result = await self._call_connector(record, arguments)
        else:
            return {"error": f"Tool {record.name!r} has no executable backing"}

        self._record_receipt(record, arguments, caller_id,
                             ok="error" not in result, via=via)
        return result

    async def start_server(
        self, record: ToolRecord, *, extra_env: dict[str, str],
        log_path: Path,
    ) -> Any:
        """Start a pinned tool as a long-running server, for a service that
        buyers keep open (atn/service_runner.py): the same extracted code,
        scrubbed environment and secret binding as ``call``, plus the
        daemon's ``extra_env`` (the port to serve on, where to report usage,
        the service's settings). stdout and stderr go to ``log_path``.

        Returns ``(process, secret_session)``; stop it with ``stop_server``.
        Raises ValueError when the tool has no code here."""
        if not record.manifest.get("code_digest"):
            raise ValueError(f"Tool {record.name!r} has no code to run as a server")
        script, code_root = self._materialize_code(record)
        session, services = self._tool_secret_session(record, None)
        argv, env, cwd = self._exec_spec(record, script, services,
                                         code_root=code_root,
                                         extra_env=extra_env)
        from . import contained_pids
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")  # noqa: SIM115 - closed in stop_server
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.DEVNULL,
                stdout=log_file, stderr=log_file, env=env, cwd=cwd)
        except Exception:
            log_file.close()
            if session is not None:
                session.release()
            raise
        contained_pids.register(proc.pid, f"{record.origin} service tool")
        if session is not None:
            session.bind(proc.pid)
        proc._autonet_log = log_file  # type: ignore[attr-defined]
        return proc, session

    @staticmethod
    async def stop_server(proc: Any, session: Any = None,
                          grace_s: float = 5.0) -> None:
        """Stop a server ``start_server`` started: terminate, then kill after
        ``grace_s``. Releases its secret session and its log file."""
        from . import contained_pids
        try:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=grace_s)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
        except ProcessLookupError:
            pass
        finally:
            contained_pids.unregister(proc.pid)
            if session is not None:
                session.release()
            log_file = getattr(proc, "_autonet_log", None)
            if log_file is not None:
                log_file.close()

    def checkout(self, digest: str, dest: Path) -> dict[str, Any]:
        """Write a pinned tool's code into ``dest`` (new or empty), like a
        clone, for an agent to read, set up and run itself
        (docs/tool_codebases.md). A codebase comes out as its files; a
        single script as ``main.py`` plus an ``autonet-tool.json`` built from
        its manifest, so either can be registered again from that folder."""
        import hashlib
        from . import tool_tree
        record = self._records.get(digest)
        if record is None:
            return {"error": "not installed here: adopt it first"}
        manifest = record.manifest
        code_digest = str(manifest.get("code_digest") or "")
        raw = self._blob_store().get_bytes(code_digest) if code_digest else None
        if raw is None:
            return {"error": "this tool has no code here to check out"}
        if manifest.get("code_format") == "tree":
            files = tool_tree.unpack(raw)
            if tool_tree.tree_digest_of(files) != manifest.get("tree_digest"):
                return {"error": "tool code does not match its tree digest"}
        else:
            if hashlib.sha256(raw).hexdigest() != code_digest:
                return {"error": "tool code does not match its digest"}
            meta = {k: manifest[k] for k in (
                "name", "description", "input_schema", "output_schema",
                "capabilities", "dependencies", "novelty", "session")
                if k in manifest}
            meta["entrypoint"] = "main.py"
            files = {"main.py": raw,
                     tool_tree.MANIFEST_FILE: json.dumps(meta, indent=2).encode("utf-8")}
        dest = Path(dest)
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            return {"error": f"{dest} exists and is not an empty directory"}
        for rel, content in sorted(files.items()):
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
        return {"path": str(dest), "files": len(files),
                "bytes": sum(len(c) for c in files.values()),
                "entrypoint": str(manifest.get("entrypoint") or "main.py")}

    def _materialize_code(self, record: ToolRecord) -> tuple[Path, Path]:
        """``(script, code root)`` for a pinned tool: its code extracted into
        a directory of its own under ``code_cache`` and verified against its
        digest before every run, so no tool can change another's code (or
        its own between runs)."""
        import hashlib
        manifest = record.manifest
        code_digest = str(manifest.get("code_digest") or "")
        cache_dir = self._dir / "code_cache"
        if manifest.get("code_format") == "tree":
            from . import tool_tree
            tree = str(manifest.get("tree_digest") or "")
            root = cache_dir / "trees" / tree
            if not (root.is_dir() and tool_tree.verify_dir(root, tree)):
                raw = self._blob_store().get_bytes(code_digest)
                if raw is None:
                    raise ValueError(f"code blob {code_digest[:16]}... not in local store")
                root = tool_tree.materialize(raw, tree, cache_dir / "trees")
            entry = str(manifest.get("entrypoint") or "").replace("\\", "/")
            script = (root / entry).resolve()
            if root.resolve() not in script.parents:
                raise ValueError("entrypoint escapes the tool's code")
            return script, root
        root = cache_dir / "scripts" / code_digest
        script = root / "tool.py"
        ok = False
        if script.is_file():
            try:
                ok = (hashlib.sha256(script.read_bytes()).hexdigest() == code_digest
                      and [p.name for p in root.iterdir()] == ["tool.py"])
            except OSError:
                ok = False
        if not ok:
            raw = self._blob_store().get_bytes(code_digest)
            if raw is None:
                raise ValueError(f"code blob {code_digest[:16]}... not in local store")
            import shutil
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)
            script.write_bytes(raw)
        return script, root

    def _exec_spec(
        self, record: ToolRecord, script: Path,
        secrets: frozenset[str] | None = None,
        code_root: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[list[str], dict[str, str] | None, str | None]:
        """(argv, env, cwd) for a pinned tool subprocess.

        Not containment (Decision 2026-11-09: the framework rates tools, it
        does not try to make running them safe). The daemon only gives a
        tool nothing it did not ask for:

          - env is scrubbed to the minimum Python needs plus ONLY the
            variables the manifest declares in ``capabilities.env``: the
            daemon's own environment (provider keys, BROKER_OWNER_SECRET)
            never reaches tool code;
          - ``secrets`` is the RESOLVED tool-secret binding (L_agent ∩
            declared, docs/tool_secret_binding.md), advertised as
            ATN_TOOL_SECRETS — NAMES ONLY, never values. The tool reads each
            value from its own broker session, bound to its PID after
            spawn; advertising the names is not a grant;
          - ``code_root`` (the tool's own extracted code) goes first on
            PYTHONPATH, so a codebase imports its own modules;
          - an adopted tool runs in a per-tool directory under the tool
            store; an authored one in the daemon's.
        """
        import sys
        secret_names = ",".join(sorted(secrets or ()))
        caps = record.manifest.get("capabilities") or {}
        sandbox = self._dir / "sandbox" / record.digest[:16]
        sandbox.mkdir(parents=True, exist_ok=True)
        env = self._contained_env(caps, sandbox, secret_names)
        if code_root is not None:
            env["PYTHONPATH"] = str(code_root)
        for name, value in (extra_env or {}).items():
            env[str(name)] = str(value)
        cwd = str(sandbox) if record.origin == "adopted" else None
        return [sys.executable, str(script)], env, cwd

    @staticmethod
    def _contained_env(caps: dict[str, Any], sandbox: Path,
                       secret_names: str) -> dict[str, str]:
        """The scrubbed environment every pinned (authored or adopted) tool
        runs with: the minimum Python needs plus ONLY the declared ``env``
        names. Secrets in the daemon's environment never reach tool code."""
        import sys
        env: dict[str, str] = {
            "PATH": os.path.dirname(sys.executable),
            "TEMP": str(sandbox),
            "TMP": str(sandbox),
            "PYTHONIOENCODING": "utf-8",
            # Bytecode caches live in the tool's sandbox, never beside its
            # code: Python would load a .pyc planted there instead of the
            # verified source.
            "PYTHONPYCACHEPREFIX": str(sandbox / "pycache"),
        }
        # Windows: python/network runtime needs SystemRoot.
        for keep in ("SYSTEMROOT", "SystemRoot", "WINDIR"):
            if keep in os.environ:
                env[keep] = os.environ[keep]
        for name in caps.get("env") or []:
            if isinstance(name, str) and name in os.environ:
                env[name] = os.environ[name]
        if secret_names:
            env["ATN_TOOL_SECRETS"] = secret_names
            # Where the tool's broker session lives (a path, not a secret).
            for name in ("VAULT_BROKER_SOCK", "ATN_VAULT_BROKER_SOCK"):
                if name in os.environ:
                    env[name] = os.environ[name]
        return env

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
                           *, caller_id: str | None = None,
                           extra_env: dict[str, str] | None = None,
                           ) -> dict[str, Any]:
        """Run the pinned code blob as a subprocess: JSON args on stdin,
        JSON result on stdout. The blob is materialized to a cache file
        named by its digest, so what runs is exactly what was judged.

        If the manifest declares secrets AND the caller's allowance covers
        them, a short-lived broker session is bound to the subprocess PID for
        the duration of the call (docs/tool_secret_binding.md)."""
        try:
            script, code_root = self._materialize_code(record)
        except (ValueError, OSError) as exc:
            return {"error": str(exc)}

        session, services = self._tool_secret_session(record, caller_id)
        argv, env, cwd = self._exec_spec(record, script, services,
                                         code_root=code_root,
                                         extra_env=extra_env)
        from . import contained_pids
        tracked_pid: int | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env, cwd=cwd,
            )
            # The owner listener refuses this PID and its descendants for the
            # life of the run (atn/contained_pids.py).
            tracked_pid = proc.pid
            contained_pids.register(tracked_pid, f"{record.origin} tool")
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
            if tracked_pid is not None:
                contained_pids.unregister(tracked_pid)

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
        extra_env: dict[str, str] | None = None,
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
        declared: set[str] = set(record.manifest.get("dependencies") or [])

        if depth >= _COMPOSITE_MAX_DEPTH:
            return {"error": f"composition depth exceeded {_COMPOSITE_MAX_DEPTH}"}

        timeout = float(_PINNED_EXEC_TIMEOUT_S)
        from . import contained_pids
        tracked_pid: int | None = None
        try:
            script, code_root = self._materialize_code(record)
        except (ValueError, OSError) as exc:
            return {"error": str(exc)}

        # Same tool-secret binding as the sealed path. Nested dep calls
        # run under the ORIGINAL caller's authority (see the frame handler
        # below), so each tool in a composition gets its own clamp against
        # the same L_agent — a composite cannot lend its secrets to a
        # dependency.
        session, services = self._tool_secret_session(record, caller_id)
        argv, env, cwd = self._exec_spec(record, script, services,
                                         code_root=code_root,
                                         extra_env=extra_env)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env, cwd=cwd,
            )
            tracked_pid = proc.pid
            contained_pids.register(tracked_pid, f"{record.origin} tool")
            if session is not None:
                session.bind(proc.pid)
        except Exception as exc:
            if session is not None:
                session.release()
            if tracked_pid is not None:
                contained_pids.unregister(tracked_pid)
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
            return await asyncio.wait_for(_pump(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"error": f"tool timed out after {timeout:g}s"}
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
            if tracked_pid is not None:
                contained_pids.unregister(tracked_pid)

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
        """Append a usage receipt to the local ledger. Nothing is gossiped
        (Decision 2026-11-07): receipts are daemon-local stats.

        ``via`` (composition telemetry): when non-empty, this call was
        dispatched from a composite's sandbox call-rail; the digest is
        recorded on the receipt row so nested receipts stay attributable
        to the composite."""
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

    def attest_usage(
        self,
        caller_id: str | None,
        judgments: list[dict[str, Any]],
        context_text: str,
    ) -> dict[str, Any]:
        """Record the reviews of a closed work item (Decision 2026-11-09).

        ``judgments`` — list of ``{"tool", "correctness", "simplicity",
        "novelty", "malware", "evidence"?, "axes"?, "ok"?, "note"?}``.
        Every review carries the four fixed fields; a malware flag carries
        ``evidence`` (free text pointing into the code), stored as a
        content-addressed blob whose sha256 is the review's evidence hash.
        Each tool is resolved via ``self.resolve`` and scoping-checked; a
        tool the caller may not use, can't resolve, or an invalid review is
        skipped and reported, never failing the batch. ``context_text``
        (the work item) is kept with the note blob.

        Nothing is gossiped: each review comes back under ``_reviews`` for
        the caller to queue on ToolBond; the local row keeps it for
        history, the UI and ``review_due``."""
        from nodes.common.library.review_fields import parse_fields
        caller = caller_id if caller_id else OWNER_AUTHOR

        try:
            blobs: Any = self._blob_store()
        except RuntimeError:
            blobs = None

        attested = 0
        skipped: list[dict[str, Any]] = []
        reviews: list[dict[str, Any]] = []
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
            fields, why = parse_fields(judgment)
            if fields is None:
                skipped.append({"tool": tool_ref, "error": why})
                continue
            evidence_text = str(judgment.get("evidence") or "").strip()
            if fields["malware"] and not evidence_text:
                skipped.append({"tool": tool_ref, "error": MALWARE_EVIDENCE_REQUIRED})
                continue
            axes = _charter_scores(judgment.get("axes"))
            ok = (bool(judgment["ok"]) if isinstance(judgment.get("ok"), bool)
                  else fields["correctness"] > 0 and not fields["malware"])
            note = judgment.get("note")

            review_digest = ""
            if note and blobs is not None:
                review = {
                    "kind": "tool_review",
                    "manifest_digest": record.digest,
                    "caller": caller,
                    "ok": ok,
                    **fields,
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
            evidence_hash = review_digest
            if fields["malware"]:
                evidence_hash = self.store_evidence(
                    record.digest, record.manifest, caller, evidence_text, ts)
                if not evidence_hash:
                    skipped.append({"tool": tool_ref,
                                    "error": "the evidence could not be stored"})
                    continue

            self._receipt_seq += 1
            row = {
                "seq": self._receipt_seq,
                "ts": ts,
                "manifest_digest": record.digest,
                "tool_name": record.name,
                "tool_author": record.author,
                "caller": caller,
                "ok": ok,
                "score": round(max(0.0, fields["correctness"]), 6),
                **fields,
                "review_digest": review_digest,
            }
            if axes:
                row["axes"] = dict(sorted(axes.items()))
            if fields["malware"]:
                row["evidence_hash"] = evidence_hash
            # Stamp the CALLER's harness distro (docs/tool_substrate.md —
            # loadout digest is atomic with the attestation; Decision
            # 2026-10-12: per-agent binding, not the daemon-wide default).
            _loadout = self.loadout_for(caller)
            if _loadout:
                row["loadout"] = _loadout
            receipt_digest = ""
            if blobs is not None:
                try:
                    receipt_digest = blobs.add_json(row) or ""
                except Exception as exc:
                    log.warning("attestation blob failed: %s", exc)
            row["receipt_digest"] = receipt_digest

            with self._attestations_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

            attested += 1
            reviews.append({"digest": record.digest, **fields, "axes": dict(axes),
                            "evidence_hash": evidence_hash,
                            "author": str(record.manifest.get("author") or "")})

        return {"attested": attested, "skipped": skipped, "_reviews": reviews}

    def store_evidence(self, digest: str, manifest: dict[str, Any], caller: str,
                       text: str, ts: int | None = None) -> str:
        """Keep a malware flag's evidence as a content-addressed blob on this
        daemon and return its sha256 (the review's evidence hash). Keeping it
        available is the reviewer's job: peers fetch it from here
        (``fetch_evidence``). "" when the blob store is unavailable."""
        try:
            blobs: Any = self._blob_store()
        except RuntimeError:
            return ""
        body = {"kind": "malware_evidence", "manifest_digest": digest,
                "code_digest": str(manifest.get("code_digest") or ""),
                "reviewer": self._consensus_identity(caller),
                "evidence": str(text)[:20_000],
                "ts": int(ts if ts is not None else time.time())}
        try:
            evidence_hash = blobs.add_json(body) or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("malware evidence blob failed: %s", exc)
            return ""
        self._announce_evidence(evidence_hash)
        return evidence_hash

    def _announce_evidence(self, evidence_hash: str) -> None:
        """Provide a malware flag's evidence in the DHT (``blob_announcer``),
        so any daemon finds this one as its holder and fetches it while this
        daemon is online. Malware evidence only; no copy goes to relays.
        Not awaited: the review goes on whether or not a DHT peer took the
        record (the host keeps re-providing it)."""
        if self.blob_announcer is None or not evidence_hash:
            return
        try:
            asyncio.get_running_loop().run_in_executor(
                None, self._announce, [evidence_hash])
        except RuntimeError:
            self._announce([evidence_hash])

    def held_evidence_digests(self) -> list[str]:
        """The evidence blobs of this daemon's own malware flags that it
        still holds (from its attestation log), for the host's provider
        records at start. Other review blobs (notes, reports) are not
        announced."""
        out: list[str] = []
        if not self._attestations_path.exists():
            return out
        try:
            blobs = self._blob_store()
            lines = self._attestations_path.read_text(encoding="utf-8").splitlines()
        except (RuntimeError, OSError):
            return out
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("malware") is not True:
                continue
            h = str(row.get("evidence_hash") or "")
            if len(h) == 64 and h not in out and blobs.has(h):
                out.append(h)
        return out

    async def fetch_evidence(self, evidence_hash: str) -> dict[str, Any]:
        """A review's evidence by its hash: this daemon's blob store, then
        the network (``blob_fetcher``: the connected peers and, through the
        DHT, the hash's providers, the reviewer's daemon among them, dialled
        directly or through a relay circuit). The bytes are checked against
        the hash."""
        import hashlib
        h = str(evidence_hash or "").strip().lower().removeprefix("0x")
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            return {"error": "evidence_hash must be 64 hex characters"}
        if h == "0" * 64:
            return {"error": "this review carries no evidence"}
        raw = await self.fetch_bytes(h)
        if raw is None:
            return {"evidence_hash": h, "found": False,
                    "note": "no daemon holding this evidence is reachable; "
                            "keeping it online is the reviewer's job"}
        if hashlib.sha256(raw).hexdigest() != h:
            return {"error": "the fetched bytes do not match the hash"}
        try:
            body: Any = json.loads(raw)
        except ValueError:
            body = raw.decode("utf-8", errors="replace")[:20_000]
        return {"evidence_hash": h, "found": True, "evidence": body}

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
                "correctness": row.get("correctness"),
                "simplicity": row.get("simplicity"),
                "novelty": row.get("novelty"),
                "malware": row.get("malware"),
                "axes": row.get("axes") or {},
                "evidence_hash": row.get("evidence_hash", ""),
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

    def _keeps_blob(self, digest: str) -> bool:
        """True for the manifest or code blob of a tool installed here."""
        if digest in self._records:
            return True
        return any(str(r.manifest.get("code_digest") or "") == digest
                   for r in list(self._records.values()))

    @staticmethod
    def _pruned_refusal(digest: str, what: str = "") -> dict[str, Any] | None:
        """The refusal for adopting a pruned tool (Decision 2026-11-10): a
        malware verdict, its own or a dependency's, or a refused
        registration. None when the tool is not pruned."""
        from nodes.common.pruned_tools import is_pruned
        if not is_pruned(digest):
            return None
        label = what or digest
        return {"error": f"cannot adopt: {label} has a malware verdict (its own "
                         "or a dependency's) and the network has pruned it",
                "pruned": digest}

    async def propose_adoption(
        self, caller_id: str | None, digest: str, reason: str = "",
    ) -> dict[str, Any]:
        """Agent-side half of the install rail: PROPOSE, never install.

        Adoption is foreign code entering the host — the one place a
        real approval queue is legitimate. The proposal fetches and
        digest-verifies the manifest, then packages what the owner
        needs to decide: declared capabilities (the sandbox policy it
        will run under) and signature presence. Nothing executes until
        approve_adoption.
        """
        caller = caller_id if caller_id else OWNER_AUTHOR
        digest = digest.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef"
                                    for c in digest):
            return {"error": "digest must be the 64-hex manifest digest "
                             "(probe the substrate to find one)"}
        if digest in self._records:
            return {"error": "already available locally"}
        refused = self._pruned_refusal(digest)
        if refused is not None:
            return refused
        existing = self._proposals.get(digest)
        if existing is not None and existing.get("status") == "pending":
            return {"status": "pending", "digest": digest,
                    "note": "already proposed; awaiting owner approval"}
        if (existing is not None
                and existing.get("status") == ADOPTION_APPROVED_PENDING):
            return {"status": ADOPTION_APPROVED_PENDING, "digest": digest,
                    "note": "already approved; waiting for a holder of its "
                            "code to come online"}

        payload = await self.fetch_payload(digest)
        from nodes.common.library.tool_manifest import (
            is_tool_manifest,
        )
        if payload is None or not is_tool_manifest(payload):
            return {"error": f"cannot resolve manifest {digest[:16]}: not "
                             "fetchable from the network"}
        if str(payload.get("trust_class") or "") != "pinned":
            return {"error": "only pinned tools can be adopted — attested "
                             "tools lean on connectors/credentials that "
                             "don't transfer between daemons"}

        sanction: dict[str, Any] | None = None
        author = str(payload.get("author") or "")
        if self.sanction_check is not None and author:
            try:
                sanction = await asyncio.to_thread(self.sanction_check, author)
            except Exception as exc:
                log.warning("sanction lookup failed: %s", exc)
        if sanction is not None and caller != OWNER_AUTHOR:
            from .sanctions import refusal
            return refusal(author, sanction, "adopt the tools of")

        # A composite brings its declared dependencies (#37): every one not
        # installed here joins this proposal, so one owner approval installs
        # the set and the composite can actually run.
        bundle, dep_err = await self._collect_missing_deps(payload, caller)
        if dep_err is not None:
            return dep_err

        signed = await asyncio.to_thread(self._verify_manifest_sig, payload)
        caps = dict(payload.get("capabilities") or {})
        from .runtime.shell_provider import core_claims
        replaces = core_claims(payload)
        for dep in bundle:
            replaces = replaces + [c for c in dep["core_replacement"]
                                   if c not in replaces]
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
            # Non-empty => approving installs a CORE REPLACEMENT candidate:
            # the owner must approve with core_replacement_ack (a distinct,
            # explicit act; the UI shows an extra warning).
            "core_replacement": replaces,
            # The author's household is under a DAO sanction (owner-made
            # proposals only; agents are refused above).
            "sanction": sanction,
            "provenance": {
                "signed": signed,
                "dependencies": len(payload.get("dependencies") or []),
            },
            # Declared dependencies not installed here, installed together
            # with this tool on approval (each re-checked like an adoption).
            "bundle": bundle,
        }
        self._proposals[digest] = proposal
        self._save_proposals()
        log.info("adoption proposed: %s (%s) by %s, %d dependencies",
                 proposal["name"], digest[:16], caller, len(bundle))
        note = "proposed; awaiting owner approval"
        if bundle:
            note += (" (includes dependencies: "
                     + ", ".join(d["name"] or d["digest"][:16] for d in bundle)
                     + ")")
        return {"status": "pending", "digest": digest,
                "capabilities": caps, "provenance": proposal["provenance"],
                "core_replacement": replaces,
                "dependencies": [{"digest": d["digest"], "name": d["name"]}
                                 for d in bundle],
                "note": note}

    async def _collect_missing_deps(
        self, payload: dict[str, Any], caller: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Walk a manifest's declared dependencies (recursively, at most
        ``_COMPOSITE_MAX_DEPTH`` levels, the same bound composition runs
        under) and return the ones not installed here, each fetched and
        checked like a proposal: fetchable manifest, pinned, author not
        sanctioned (agent callers). Returns ``(bundle, None)`` or
        ``([], {"error": ...})`` naming the dependency that failed."""
        from nodes.common.library.tool_manifest import (
            is_tool_manifest,
        )
        from .runtime.shell_provider import core_claims
        bundle: list[dict[str, Any]] = []
        seen: set[str] = set()
        frontier = [str(d).strip().lower()
                    for d in (payload.get("dependencies") or [])]
        for _level in range(_COMPOSITE_MAX_DEPTH):
            nxt: list[str] = []
            for dep in frontier:
                if not dep or dep in seen:
                    continue
                seen.add(dep)
                refused = self._pruned_refusal(dep, f"its dependency {dep}")
                if refused is not None:
                    refused["dependency"] = dep
                    return [], refused
                if dep in self._records:
                    continue  # already installed here
                dep_payload = await self.fetch_payload(dep)
                if dep_payload is None or not is_tool_manifest(dep_payload):
                    return [], {"error": f"cannot adopt: dependency {dep} "
                                         "is not fetchable from the network "
                                         "(its author may not have published "
                                         "it)", "dependency": dep}
                name = str(dep_payload.get("name") or "")
                label = f"{name} ({dep[:16]})" if name else dep
                if str(dep_payload.get("trust_class") or "") != "pinned":
                    return [], {"error": f"cannot adopt: dependency {label} is "
                                         "not a pinned tool", "dependency": dep}
                author = str(dep_payload.get("author") or "")
                sanction: dict[str, Any] | None = None
                if self.sanction_check is not None and author:
                    try:
                        sanction = await asyncio.to_thread(
                            self.sanction_check, author)
                    except Exception as exc:
                        log.warning("sanction lookup failed: %s", exc)
                if sanction is not None and caller != OWNER_AUTHOR:
                    from .sanctions import refusal
                    out = refusal(author, sanction, "adopt the tools of")
                    out["dependency"] = dep
                    return [], out
                bundle.append({
                    "digest": dep,
                    "name": name,
                    "author": author,
                    "capabilities": dict(dep_payload.get("capabilities") or {}),
                    "core_replacement": core_claims(dep_payload),
                    "sanction": sanction,
                    "provenance": {
                        "signed": await asyncio.to_thread(
                            self._verify_manifest_sig, dep_payload),
                    },
                })
                nxt.extend(str(d).strip().lower()
                           for d in (dep_payload.get("dependencies") or []))
            frontier = nxt
            if not frontier:
                break
        return bundle, None

    def list_adoption_proposals(
        self, status: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = sorted(self._proposals.values(), key=lambda r: -r.get("ts", 0))
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return [dict(r) for r in rows]

    async def approve_adoption(
        self, digest: str, core_replacement_ack: bool = False,
    ) -> dict[str, Any]:
        """Owner-side half (WS surface, never an agent tool): verify,
        install as an ADOPTED record — original author preserved on the
        manifest (their mint keeps flowing from our attestations),
        adopter-lineage scoping via local_author, contained execution
        via origin="adopted".

        Code seeds like a torrent (#46): only the author and adopters hold
        a tool's code. When none of them is online the approval is kept
        (status ``approved_pending``) and the code fetch retried in the
        background; see ``_adoption_retry_loop``."""
        async with self._adoption_lock():
            return await self._approve_locked(digest, core_replacement_ack)

    def _adoption_lock(self) -> asyncio.Lock:
        # One install at a time: the owner's approval and a background
        # retry must not both install the same proposal.
        if self._adopt_lock is None:
            self._adopt_lock = asyncio.Lock()
        return self._adopt_lock

    async def _approve_locked(
        self, digest: str, core_replacement_ack: bool,
    ) -> dict[str, Any]:
        proposal = self._proposals.get(digest)
        if proposal is None or proposal.get("status") not in (
                "pending", ADOPTION_APPROVED_PENDING):
            return {"error": "no pending proposal for that digest"}
        # A verdict may have landed since the proposal (Decision 2026-11-10).
        for d in [digest] + [b["digest"] for b in proposal.get("bundle") or []]:
            refused = self._pruned_refusal(d)
            if refused is not None:
                proposal["status"] = "refused"
                self._save_proposals()
                return refused
        # A retry keeps the owner's original acknowledgment.
        core_replacement_ack = bool(core_replacement_ack
                                    or proposal.get("core_replacement_ack"))

        # A composite's proposal carries its missing dependencies (#37).
        # Every member passes the same install checks; any failure installs
        # NOTHING, so the composite never lands without what it calls.
        members: list[tuple[str, dict[str, Any]]] = []
        replaces: list[str] = []
        for dep in proposal.get("bundle") or []:
            dep_digest = str(dep.get("digest") or "")
            if not dep_digest or dep_digest in self._records:
                continue
            payload, err = await self._install_check(
                dep_digest, dep.get("provenance") or {})
            if err == _NO_CODE_HOLDER:
                return self._defer_for_code(digest, proposal, dep_digest,
                                            core_replacement_ack)
            if err is not None:
                label = (f"{dep.get('name')} ({dep_digest[:16]})"
                         if dep.get("name") else dep_digest)
                return {"error": f"dependency {label}: {err}",
                        "dependency": dep_digest}
            members.append((dep_digest, payload))
        prov = (proposal.get("provenance") or {}) if isinstance(
            proposal.get("provenance"), dict) else {}
        payload, err = await self._install_check(digest, prov)
        if err == _NO_CODE_HOLDER:
            return self._defer_for_code(digest, proposal, digest,
                                        core_replacement_ack)
        if err is not None:
            return {"error": err}
        members.append((digest, payload))

        from .runtime.shell_provider import core_claims
        for _d, p in members:
            replaces += [c for c in core_claims(p) if c not in replaces]
        # Core-replacement privilege class (Decision 2026-10-12). A tool that
        # claims to provide core built-in names would see every command and
        # file body the agent touches once swapped in, so approving it is a
        # visibly different act: the approval must carry the explicit
        # acknowledgment, and is rejected without it. One ack covers the
        # bundle; the proposal listed every claim.
        if replaces and not core_replacement_ack:
            return {"error": "core_replacement_ack required: this tool "
                             "replaces " + ", ".join(replaces) + " and will "
                             "see every command and file your agent touches",
                    "core_replacement": replaces}

        now = int(time.time())
        local_author = str(proposal.get("proposed_by") or "")
        for d, p in members:
            self._records[d] = ToolRecord(
                digest=d,
                manifest=p,
                local_author=local_author,
                registered_ts=now,
                origin="adopted",
                core_replacement_ack=bool(core_claims(p)
                                          and core_replacement_ack),
                jurisdiction=self._jurisdiction,
            )
            if d != digest:
                # A standalone pending proposal for a dependency is settled
                # by this install.
                other = self._proposals.get(d)
                if other is not None and other.get("status") in (
                        "pending", ADOPTION_APPROVED_PENDING):
                    other.pop("retry", None)
                    other["status"] = "approved"
                    other["resolved_ts"] = now
                    other["installed_with"] = digest
        record = self._records[digest]
        proposal["status"] = "approved"
        proposal.pop("retry", None)
        if replaces and core_replacement_ack:
            proposal["core_replacement_ack"] = True
        proposal["resolved_ts"] = now
        self._persist()
        self._save_proposals()
        deps_installed = [{"digest": d, "name": str(p.get("name") or "")}
                          for d, p in members if d != digest]
        log.info("adoption approved: %s (%s) for %s, %d dependencies",
                 record.name, digest[:16], record.local_author,
                 len(deps_installed))
        # Every holder serves: announce the adopted blobs (not awaited).
        adopted = [x for d, p in members for x in self.blob_digests(d, p)]
        if self.blob_announcer is not None and adopted:
            try:
                asyncio.get_running_loop().run_in_executor(None, self._announce, adopted)
            except RuntimeError:
                self._announce(adopted)
        return {"digest": digest, "name": record.name,
                "origin": "adopted", "author": record.author,
                "adopted_for": record.local_author,
                "dependencies_installed": deps_installed,
                "core_replacement": replaces,
                "core_replacement_ack": bool(replaces and core_replacement_ack)}

    async def _install_check(
        self, digest: str, prov: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """The install-time checks one adopted tool must pass, against a
        freshly fetched manifest: resolvable, pinned, code fetchable,
        signature not wrong. Ratings never refuse an install (Decision
        2026-11-09): what to adopt is the household's call.
        Returns ``(payload, None)`` or ``(None, reason)``."""
        payload = await self.fetch_payload(digest)
        from nodes.common.library.tool_manifest import (
            is_tool_manifest,
        )
        if payload is None or not is_tool_manifest(payload):
            return None, "manifest blob no longer resolvable"
        if str(payload.get("trust_class") or "") != "pinned":
            return None, "only pinned tools can be adopted"
        code_digest = str(payload.get("code_digest") or "")
        code_raw = await self.fetch_bytes(code_digest) if code_digest else None
        if code_raw is None:
            # Not a refusal: nobody holding the code is online right now.
            # approve_adoption keeps the approval and retries (#46).
            return None, _NO_CODE_HOLDER
        if payload.get("code_format") == "tree":
            # A codebase: the archive must unpack safely and hold exactly
            # the content its tree digest names.
            from . import tool_tree
            try:
                files = tool_tree.unpack(code_raw)
            except (tool_tree.TreeError, ValueError, OSError) as exc:
                return None, f"tool code is not a valid snapshot: {exc}"
            if tool_tree.tree_digest_of(files) != str(payload.get("tree_digest") or ""):
                return None, "tool code does not match its tree digest"

        # Provenance is LOAD-BEARING here, not decoration. propose_adoption
        # assembles {signed, dependencies} for the owner to read,
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
        if await asyncio.to_thread(self._verify_manifest_sig, payload) is False:
            return None, ("manifest signature does not recover to its "
                          "claimed author — refusing to install "
                          "(re-attribution attempt)")
        return payload, None

    def _defer_for_code(
        self, digest: str, proposal: dict[str, Any], missing: str,
        core_replacement_ack: bool,
    ) -> dict[str, Any]:
        """Keep an approval whose code no online daemon holds (#46) and
        schedule the background retry. Nothing is installed."""
        now = int(time.time())
        if proposal.get("status") != ADOPTION_APPROVED_PENDING:
            proposal["status"] = ADOPTION_APPROVED_PENDING
            proposal["approved_ts"] = now
        if core_replacement_ack:
            proposal["core_replacement_ack"] = True
        retry = dict(proposal.get("retry") or {})
        attempts = int(retry.get("attempts") or 0) + 1
        delay = min(self.adoption_retry_base_s * (2 ** (attempts - 1)),
                    self.adoption_retry_max_s)
        retry.update({"attempts": attempts, "last_ts": now,
                      "next_ts": now + delay, "awaiting_code_of": missing})
        proposal["retry"] = retry
        self._save_proposals()
        self._ensure_adoption_retry()
        whose = ("this tool's" if missing == digest
                 else f"its dependency {missing[:16]}'s")
        return {"status": ADOPTION_APPROVED_PENDING, "digest": digest,
                "name": str(proposal.get("name") or ""),
                "awaiting_code_of": missing,
                "note": (f"No holder of {whose} code is online (its author "
                         "or a daemon that adopted it). The adoption stays "
                         "approved and pending: this daemon retries the "
                         "fetch in the background and when a new peer "
                         "connects, and installs it then.")}

    def resume_adoption_retries(self) -> None:
        """Start the retry loop if approved adoptions are waiting for code
        (called at runtime start, so a restart keeps retrying)."""
        self._ensure_adoption_retry()

    async def stop_adoption_retries(self) -> None:
        task, self._retry_task = self._retry_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def _awaiting_code(self) -> list[str]:
        return [d for d, p in self._proposals.items()
                if p.get("status") == ADOPTION_APPROVED_PENDING]

    def _ensure_adoption_retry(self) -> None:
        if self._retry_task is not None and not self._retry_task.done():
            return
        if not self._awaiting_code():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._retry_task = loop.create_task(self._adoption_retry_loop())

    def _peer_snapshot(self) -> set[str] | None:
        if self.peer_lister is None:
            return None
        try:
            return {str(p) for p in self.peer_lister()}
        except Exception:  # noqa: BLE001 — host down: timer retries only
            return None

    async def _adoption_retry_loop(self) -> None:
        seen = self._peer_snapshot()
        while True:
            waiting = self._awaiting_code()
            if not waiting:
                return
            peers = self._peer_snapshot()
            new_peer = bool(peers is not None and seen is not None
                            and peers - seen)
            if peers is not None:
                seen = peers
            now = time.time()
            for digest in waiting:
                proposal = self._proposals.get(digest)
                if (proposal is None
                        or proposal.get("status") != ADOPTION_APPROVED_PENDING):
                    continue
                started = float(proposal.get("approved_ts") or now)
                if now - started > self.adoption_retry_window_s:
                    await self._expire_adoption(digest, proposal)
                    continue
                due = float((proposal.get("retry") or {}).get("next_ts") or 0)
                if new_peer or now >= due:
                    try:
                        await self.retry_adoption(digest)
                    except Exception:  # noqa: BLE001 — keep retrying others
                        log.warning("adoption retry crashed for %s",
                                    digest[:16], exc_info=True)
            await asyncio.sleep(self.adoption_poll_s)

    async def retry_adoption(self, digest: str) -> dict[str, Any]:
        """One background attempt for an approved_pending adoption. It
        runs the owner's approval path unchanged (same verification,
        digest pinning) and reports the outcome as a
        TOOL_ADOPTION event."""
        async with self._adoption_lock():
            proposal = self._proposals.get(digest)
            if (proposal is None
                    or proposal.get("status") != ADOPTION_APPROVED_PENDING):
                return {"error": "no approved adoption waiting for code"}
            out = await self._approve_locked(digest, False)
            if out.get("status") == ADOPTION_APPROVED_PENDING:
                return out
            if "error" in out:
                # The code arrived but a later check refused it (denied,
                # wrong signature): the approval ends here.
                proposal["status"] = "failed"
                proposal["fail_reason"] = str(out["error"])
                proposal["resolved_ts"] = int(time.time())
                proposal.pop("retry", None)
                self._save_proposals()
                log.warning("adoption retry refused %s: %s",
                            digest[:16], out["error"])
        if "error" in out:
            await self._emit_adoption(digest, proposal, "failed",
                                      error=str(out["error"]))
        else:
            await self._emit_adoption(digest, proposal, "installed",
                                      result=out)
        return out

    async def _expire_adoption(self, digest: str,
                               proposal: dict[str, Any]) -> None:
        proposal["status"] = "pending"
        proposal["expired_ts"] = int(time.time())
        proposal.pop("retry", None)
        self._save_proposals()
        log.info("adoption of %s found no online code holder in %ds: back "
                 "to the owner's queue", digest[:16],
                 int(self.adoption_retry_window_s))
        await self._emit_adoption(
            digest, proposal, "expired",
            note="no holder of its code came online; approve it again to "
                 "retry")

    async def _emit_adoption(self, digest: str, proposal: dict[str, Any],
                             status: str, **extra: Any) -> None:
        events = getattr(self._runtime, "events", None)
        if events is None:
            return
        from .events import Event, EventType
        try:
            await events.emit(Event(
                type=EventType.TOOL_ADOPTION,
                source="tool_store",
                data={"digest": digest,
                      "name": str(proposal.get("name") or ""),
                      "proposed_by": str(proposal.get("proposed_by") or ""),
                      "status": status, **extra},
            ))
        except Exception:  # noqa: BLE001 — reporting never breaks install
            log.debug("adoption event emit failed", exc_info=True)

    def reject_adoption(self, digest: str, reason: str = "") -> dict[str, Any]:
        proposal = self._proposals.get(digest)
        if proposal is None or proposal.get("status") not in (
                "pending", ADOPTION_APPROVED_PENDING):
            return {"error": "no pending proposal for that digest"}
        proposal.pop("retry", None)
        proposal["status"] = "rejected"
        proposal["resolved_ts"] = int(time.time())
        if reason:
            proposal["reject_reason"] = str(reason)
        self._save_proposals()
        return {"digest": digest, "status": "rejected"}

    def _verify_manifest_sig(self, manifest: dict[str, Any]) -> bool | None:
        """Does the signature prove the manifest's ``author`` claim?

        True  = the signature recovers to ``author``, or to a signer the
                chain binds to ``author`` (see below);
        False = present but WRONG: it recovers to neither ``author`` nor
                ``author_pubkey``, or to an ``author_pubkey`` the chain binds
                to a DIFFERENT owner (re-attribution attempt);
        None  = no signature, verification unavailable, or authorship not
                provable. Treated like an unsigned manifest.

        Content addressing already guarantees integrity; this proves the
        AUTHORSHIP claim. Both ``author`` and ``author_pubkey`` sit inside the
        signed payload, so a signature recovering to ``author_pubkey`` proves
        only that the signer wrote the bytes, not that ``author`` did: any key
        could stamp itself as ``author_pubkey`` and claim any ``author``.

        The legitimate case for that shape: an agent not registered on-chain
        authors tools as the owner WALLET (_consensus_identity), and its own
        key signs (_sign), so ``author`` = wallet and ``author_pubkey`` = the
        agent. That counts as proof only when the chain binds the signer to
        ``author`` (Substrate ``agentOwner(signer) == author``). An unbound
        signer (an unregistered agent, which is exactly the case above) or an
        unreachable chain yields None: authorship unproven, fail closed for
        the claim, same standing as an unsigned tool.

        Blocking (a chain read on that path): async callers run it in a
        thread."""
        sig = manifest.get("author_sig")
        if not sig:
            return None
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            from nodes.common.library.tool_manifest import (
                canonical_manifest_bytes,
            )
            recovered = Account.recover_message(
                encode_defunct(canonical_manifest_bytes(manifest)),
                signature=bytes.fromhex(str(sig).removeprefix("0x")),
            ).lower()
        except Exception as exc:
            log.debug("manifest sig verification unavailable: %s", exc)
            return None
        author = str(manifest.get("author") or "").lower()
        if recovered == author:
            return True
        if recovered != str(manifest.get("author_pubkey") or "").lower():
            return False
        if not _is_address(author):
            return None  # a local id / "user" author is never chain-bound
        owner = self._bound_owner(recovered)
        if not owner or owner == "0x" + "00" * 20:
            return None  # unbound signer or chain unreachable: unproven
        return True if owner == author else False

    def _bound_owner(self, signer: str) -> str | None:
        """The owner wallet the chain binds ``signer`` to (lowercase 0x; the
        zero address when unbound), or None when the chain is not configured
        or unreachable. ``agent_owner_lookup`` (address -> owner) overrides
        the default Substrate read, for tests and alternate chains."""
        lookup = getattr(self, "agent_owner_lookup", None)
        try:
            if lookup is None:
                from .on_chain import OnChainService
                cfg = getattr(getattr(self, "_runtime", None), "_config", None)
                autonet = getattr(cfg, "autonet", None)
                if autonet is None or not getattr(autonet, "enabled", False):
                    return None
                svc = OnChainService(autonet)
                if not svc.available:
                    return None
                status = svc.read_binding_status(signer)
                owner = status.get("owner") if status.get("registered") else None
                owner = owner or "0x" + "00" * 20
            else:
                owner = lookup(signer)
            owner = str(owner or "").lower()
            return owner if owner.startswith("0x") else None
        except Exception as exc:  # noqa: BLE001 — unreachable => unproven
            log.debug("agent owner lookup failed for %s: %s", signer, exc)
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
        review: dict[str, Any] | None = None,
        report: str = "",
    ) -> dict[str, Any]:
        """Inspection review (Decision 2026-11-09), two steps in one rail:

        - ``review=None`` — INSPECT: resolve the manifest (locally or
          fetched from the network by digest) and return it with the
          pinned source code, so the reviewer can read what it is judging.
        - ``review`` given — REVIEW: the four fixed fields (correctness,
          simplicity, novelty, malware), optional charter ``axes``, and for
          a malware flag its ``evidence``. ``report`` (required) is
          blob-stored as the inspection report; a malware flag's evidence
          is its own blob, whose sha256 is the review's evidence hash
          (otherwise the report's). The caller queues the review on
          ToolBond; no consensus event is gossiped.

        Self-review is rejected. An inspection is not usage and never
        touches the receipt ledgers.
        """
        from nodes.common.library.review_fields import parse_fields
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
            from nodes.common.library.tool_manifest import (
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

        if review is None:
            out = {
                "digest": digest,
                "manifest": manifest,
                "code": (code_raw.decode("utf-8", errors="replace")
                         if code_raw is not None else None),
                "note": ("review the code against the manifest and its "
                         "declared capabilities, then call again with "
                         "correctness, simplicity, novelty, malware (and "
                         "evidence for a malware flag) and a report"),
            }
            if manifest.get("code_format") == "tree" and code_raw is not None:
                # A codebase: its files, not the archive bytes. Large or
                # binary files are listed with their hash only.
                import hashlib
                from . import tool_tree
                try:
                    files = tool_tree.unpack(code_raw)
                except tool_tree.TreeError as exc:
                    return {"error": f"tool code is not a valid snapshot: {exc}"}
                shown: dict[str, Any] = {}
                for rel, content in sorted(files.items()):
                    text = None
                    if len(content) <= 256 * 1024:
                        try:
                            text = content.decode("utf-8")
                        except UnicodeDecodeError:
                            text = None
                    shown[rel] = text if text is not None else {
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content)}
                out["code"] = None
                out["files"] = shown
                out["note"] = ("a codebase: 'files' holds every file (the "
                               "entrypoint is the manifest's). " + out["note"])
            return out

        fields, why = parse_fields(review)
        if fields is None:
            return {"error": why}
        if not report.strip():
            return {"error": "a review requires a report — state what you "
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
        evidence_text = str(review.get("evidence") or "").strip()
        if fields["malware"] and not evidence_text:
            return {"error": MALWARE_EVIDENCE_REQUIRED}

        ts = int(time.time())
        norm_axes = _charter_scores(review.get("axes"))
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
                    **fields,
                    "report": str(report),
                    "ts": ts,
                }
                if norm_axes:
                    report_blob["axes"] = dict(sorted(norm_axes.items()))
                review_digest = blobs.add_json(report_blob) or ""
            except Exception as exc:
                log.warning("vet report blob failed: %s", exc)
        evidence_hash = review_digest
        if fields["malware"]:
            evidence_hash = self.store_evidence(digest, manifest, caller,
                                                evidence_text, ts)
            if not evidence_hash:
                return {"error": "the evidence could not be stored"}

        self._receipt_seq += 1
        row = {
            "seq": self._receipt_seq,
            "ts": ts,
            "manifest_digest": digest,
            "tool_name": str(manifest.get("name") or ""),
            "tool_author": str(manifest.get("author") or ""),
            "caller": caller,
            "ok": fields["correctness"] > 0 and not fields["malware"],
            "vet": True,
            **fields,
            "review_digest": review_digest,
        }
        if norm_axes:
            row["axes"] = dict(sorted(norm_axes.items()))
        if fields["malware"]:
            row["evidence_hash"] = evidence_hash
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._attestations_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        return {"digest": digest, "review_digest": review_digest,
                "review": {"digest": digest, **fields, "axes": dict(norm_axes),
                           "evidence_hash": evidence_hash,
                           "author": str(manifest.get("author") or "")},
                "tool_author": str(manifest.get("author") or "")}

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
        subprocess), so it replays exactly as it runs in production.

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

        # Replay through the ordinary call path. caller_id=None runs as
        # owner-sanctioned local verification;
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
        ``{vet_count, pass_count, last_ts}``. Local view only."""
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
        from .agent_tools import is_owner_caller
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
        every agent has a local keypair from birth, but an unregistered
        keypair has no proven owner, so its mint would key to an address
        no household claims (the close keys mint by the agent's on-chain
        owner, Decision 2026-10-27). Unregistered
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
            from nodes.common.library.tool_manifest import (
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
            cur = self._jurisdiction
            published_in = set(row.get("published_in") or [])
            if "published_in" not in row and row.get("published") and cur:
                # Pre-2026-10-17 row: published in the jurisdiction this
                # daemon ran then, the packaged one.
                published_in = {cur}
            origin = str(row.get("origin") or "authored")
            if origin not in ("authored", "adopted"):
                # Guest ("integration") records from the removed integration
                # listener: their code was written for the guest sandbox and
                # must never run on the authored path. Dropped.
                log.warning("ToolStore: dropping %s record %s... (origin no "
                            "longer supported)", origin, digest[:16])
                continue
            record = ToolRecord(
                digest=digest,
                manifest=manifest,
                grants=set(row.get("grants") or []),
                enabled=bool(row.get("enabled", True)),
                published=((cur in published_in) if cur
                           else bool(row.get("published", False))),
                published_in=published_in,
                jurisdiction=str(row.get("jurisdiction")
                                 or (cur if origin == "adopted" else "")),
                local_author=str(row.get("local_author") or ""),
                registered_ts=int(row.get("registered_ts") or 0),
                source_path=str(row.get("source_path") or ""),
                origin=origin,
                core_replacement_ack=bool(row.get("core_replacement_ack")),
                last_evidence_digest=str(row.get("last_evidence_digest") or ""),
            )
            if origin == "adopted" and cur and record.jurisdiction != cur:
                self._parked.append(record)
                continue
            self._records[digest] = record

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in [*self._records.values(), *self._parked]:
                fh.write(json.dumps(record.to_row()) + "\n")
        os.replace(tmp, self._registry_path)
