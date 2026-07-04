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

_PINNED_EXEC_TIMEOUT_S = 120
_MAX_CODE_BYTES = 512 * 1024
# Composition (docs/tool_substrate.md — Composition, COMPOSITE_MAX_DEPTH):
# a composite may nest dep calls this many levels deep. Exceeding it is a
# runtime error frame, not a crash — guards runaway recursion / cycles.
_COMPOSITE_MAX_DEPTH = 4


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
    # author_id falls back to the manifest author.
    local_author: str = ""
    registered_ts: int = 0

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

        author_pubkey = self._author_address(author)
        # Consensus author = the 0x address (chain-claimable, globally
        # unique — mint keyed by a local id has no on-chain claim path).
        # The local id stays on the record for daemon-side scoping.
        consensus_author = self._consensus_identity(author)

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

    def set_published(self, digest: str, published: bool) -> bool:
        """Owner-gated publish/unpublish. Publishing pushes the manifest
        to the substrate; unpublishing only stops future pushes (the
        substrate is forward-only — existing claims stand)."""
        record = self._records.get(digest)
        if record is None:
            return False
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
                result = await self._call_pinned(record, arguments)
        elif manifest.get("connector_id"):
            result = await self._call_connector(record, arguments)
        else:
            return {"error": f"Tool {record.name!r} has no executable backing"}

        self._record_receipt(record, arguments, caller_id,
                             ok="error" not in result, via=via)
        return result

    async def _call_pinned(self, record: ToolRecord,
                           arguments: dict[str, Any]) -> dict[str, Any]:
        """Run the pinned code blob as a subprocess: JSON args on stdin,
        JSON result on stdout. The blob is materialized to a cache file
        named by its digest, so what runs is exactly what was judged."""
        import sys
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

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
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
        import sys
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

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
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
        ATN transfer at epoch close) is deliberately NOT wired yet —
        payForService vs widened payForInference is an open decision.
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
                try:
                    self.event_sink(event)
                except Exception as exc:
                    log.warning("attested tool_used event sink failed: %s", exc)

            attested += 1

        return {"attested": attested, "skipped": skipped}

    def attestation_summary(self) -> dict[str, dict[str, Any]]:
        """Per-digest attestation aggregates from ``attestations.jsonl``:
        ``{attested_count, ok_count, avg_score, last_ts}``. For the Tools
        screen — the mechanical receipt ledger stays separate."""
        summary: dict[str, dict[str, Any]] = {}
        score_totals: dict[str, float] = {}
        for row in self._iter_attestations():
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

    def _consensus_identity(self, local_id: str | None) -> str:
        """Map a local id to its consensus identity (0x address).

        Agents → their identity address; the owner ("user") → the
        owner wallet when configured. Falls back to the local id so
        nothing breaks on address-less setups — those mints simply
        have no chain claim path (documented E2E seam #3)."""
        if not local_id:
            return OWNER_AUTHOR
        if local_id == OWNER_AUTHOR:
            owner = getattr(
                getattr(getattr(self._runtime, "_config", None),
                        "autonet", None), "owner_wallet", "") or ""
            return owner or OWNER_AUTHOR
        return self._author_address(local_id) or local_id

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
            )

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in self._records.values():
                fh.write(json.dumps(record.to_row()) + "\n")
        os.replace(tmp, self._registry_path)
