# Tool substrate v2: tools as the primary substrate item

Status: DESIGN v2 — ratified in discussion 2026-07-04. Supersedes v1
(same day): the **attested trust class and standing decay are retired**
from the substrate; remote/endpoint-backed offerings moved to the
Services market (`docs/services_market.md`). Tools and Services are
separate economies unified only at the agent's interface (one inference
probe, one MCP-shaped surface).

## The line: ground truth

The substrate's verdict layer only holds items with **executable ground
truth** — a CON can attach a reproducible failing invocation and the
dispute settles itself. Pinned code has that property. A remote
endpoint does not (its behavior is unknowable in principle), so it
never enters the verdict layer. This is the lesson of phase8 applied
forward: debates without ground truth are weak debates.

| | Tools (this doc) | Services (services_market.md) |
|---|---|---|
| Execution | local — your daemon, your data | remote — provider's machine |
| Trust basis | code digest; the network KNOWS it | receipts + reviews; the network TRADES with it |
| Verdicts | permanent claims in the substrate | none — market history only |
| Monetization | epoch mint (emission pays for commons) | per-work-item fees, any ERC20 |
| Boundary case | connector-backed tools: run locally with the user's own credentials → tools (evidence-grade marker `attested`, no mint) | anything with an ask price and a counterparty |

## Three tiers of tools (daemon standpoint)

1. **Private** — registered by an agent purely for local capability
   (MCP-style control of something). Author-lineage scoped, never
   leaves the daemon, zero consensus footprint. This is the DEFAULT.
2. **Published** — deliberately pushed to the substrate
   (`register_tool(..., publish=true)`, or later via the owner UI).
   Blob + ArtifactIndex + one verdict-layer claim + gossip. Publishing
   is also the future on-chain act (`ToolRegistered(agent, digest)` —
   see On-chain section).
3. (Remote offerings are not tools — they're Services.)

## Manifest (unchanged from v1 except trust semantics)

Blob-store JSON, sha256-addressed, `version_of` lineage, canonical
signing surface (`canonical_manifest_bytes`, excludes `author_sig`).
Trust classes:

- `pinned` — code blob, behavior hash-locked. Full substrate citizen:
  permanent verdicts, mintable.
- `attested` — connector-backed (external API via the user's OWN
  credentials, no counterparty ask). Publishable for discovery,
  debatable, but **mints nothing** and carries an evidence-grade
  marker: its CONs are timestamps, not permanent proofs. NO DECAY —
  decay was a v1 patch for endpoint tools that no longer live here.
- endpoint-backed manifests: REMOVED → register a Service instead.

## Utility: claimed → demonstrated → verified

Codebases replaced claims; **receipts are the new observations**. The
substrate rhythm is unchanged — durable node + flowing signal — one
level up:

1. **Claimed** — the manifest interface (name/description/schema).
   Sets the tool's initial embedding position. This is an ASK.
2. **Demonstrated** — `ToolUsed` receipts. Each ok invocation is a
   (problem-instance, tool, success) datum with skin in the game.
   Receipts carry a **problem-context embedding** (`problem_coords`:
   what the caller was trying to do, embedded in the same usefulness
   space) so a tool's effective position drifts from where its author
   SAYS it lives toward the centroid of problems it has ACTUALLY
   solved. Anti-SEO: self-description proposes, usage disposes.
3. **Verified** — debate. PRO/CON claims about the manifest; the
   canonical CON is a failing-invocation artifact, replayable forever
   against the pinned code digest.

**Inference probe** (`mode="artifacts"` over manifests): rank by
cosine against demonstrated coverage (blend of claimed embedding and
receipt problem-coords centroid), re-rank by standing. Returns tools
AND services (see services_market.md) in one answer; the agent chooses
by judgment and wallet.

## Consensus mechanics (v1 implementation, still current)

- `ToolUsed` consensus event: caller-attested, gossiped, epoch-
  buffered, graph-neutral (replay skips it). Aggregated
  deterministically by `tool_usage.py`.
- Registration sprout carries `manifest_meta` ({trust_class, author})
  so epoch close never depends on blob replication.
- Charter space UNCHANGED: manifests enter with zeroed 6-dim charter
  head + embedding tail; alignment placement emerges from debate; the
  violator-pays gate reads charter CONs as always.
- Mint (`compute_tool_mint`): **pinned only** now. Author earns
  `f(standing, usage)` anchored on the manifest claim node so the gate
  prices it. Cross-epoch carry-over: `tool_registrations` +
  `tool_receipt_history` params (both derived from canonical events).
- Wash-trading dampers (OPEN — sims decide): pinned calls are free, so
  receipts are free mint-pump. Candidates: count only out-of-lineage
  callers, caller-diversity weighting, per-receipt burn. The Python
  epoch simulator sweeps these.

## On-chain (with the Services contract work)

`ToolRegistered(agent, manifestDigest)` on Substrate.sol — msg.sender
is the agent key, so authorship becomes chain-verified. Chain = truth,
blob = storage, indexer mirrors to Firestore `tools` collection for
the web2 surface. The gossiped `manifest_meta` demotes to a cache; a
mismatch vs chain is a slashable/CON-able inconsistency. The federated
close keeps reading gossip (stays chain-free and deterministic); chain
is the dispute arbiter.

## ATN surface (implemented)

- `register_tool` (agent-callable, `toolsmith` bundle): author derived
  from caller, never accepted as input. Pinned = inline Python blob
  run as digest-named subprocess; attested = connector-backed.
  TODO v2: `publish: bool = false` (private default), owner-gated
  publish path.
- Scoping enforced twice (listing + call time); cross-lineage grants
  owner-only via WS (`grant_tool`/`revoke_tool`/`set_tool_enabled`).
- Receipts: local ledger always (`receipts.jsonl`, `tool_earnings`
  WS); consensus event when the substrate is up. `fee_atn` on tools is
  deprecated in v2 — fees belong to Services.

## v1 → v2 migration (code deltas)

1. `register_tool`: reject `endpoint=` (point to Services); add
   `publish` flag; manifest_sink only fires for published tools.
2. `compute_tool_mint`: skip non-pinned trust classes; drop
   decay/effective_standing from the mint path (`tool_standing.py`
   keeps `manifest_standing`; decay fns retired).
3. `ToolUsed` gains optional `problem_coords` (same serialize-only-
   when-present back-compat rule as every optional event field).
4. Retrieval blend in `_infer_artifacts` for manifests.

## Open knobs (decided by sims, then blessed by user)

1. Mint curve + wash-trading damper choice.
2. Emission share of tool mint vs work-unit mint.
3. Whether `attested` (connector) tools appear in cross-daemon
   discovery or stay lineage-visible only.
