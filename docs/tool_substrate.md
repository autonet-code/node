# Tool substrate: tools as the primary substrate item

Status: DESIGN — ratified in discussion 2026-07-04, branch `tool-substrate`.
Companion economics: the old Service primitive
(`trustless-contracts/contracts/Autonet.sol`: `registerService`,
`ServiceRewarded`) reborn at tool granularity, judged by substrate
consensus instead of admin attestation.

## Why tools, not text

Phase8 (`docs/phase8_results.md`) showed the text-claim graph's weakness:
debate standing over prose claims beat vote-count by only +0.127 —
because a CON against a text claim is just more text. **Tool claims have
executable ground truth.** A CON against a tool can attach a
reproducible failing invocation (input → expected vs actual) as a
blob-store artifact. Debate becomes evidence-weighted. The charter's
usefulness axes (correctness, simplicity) apply to tools almost by
definition: does it do what its schema says; is its interface minimal.

The substrate core does not change shape. Events → canonical order →
ledger replay → `net_score` standing → violator-pays gate → mint is
already item-agnostic. This refactor adds:

1. a new artifact kind (`tool_manifest`) on the existing blob/ArtifactIndex rail,
2. one new event kind (`ToolUsed` usage receipts),
3. a deterministic standing-decay term for externally-hosted tools,
4. a usage-weighted term in the mint formula, attributed to tool authors.

The text work-unit rail (conversations → work units → claims) keeps
running unchanged. Tools become the flagship item, not the only item.

## The substrate item: tool manifest

A tool manifest is a JSON artifact in the blob store, addressed by
sha256 digest (NOT an IPFS CID — `blob_store.py` rules apply):

```json
{
  "kind": "tool_manifest",
  "name": "calendar_free_slots",
  "description": "Find free slots across attendees' calendars.",
  "input_schema": { "type": "object", "properties": { ... } },
  "output_schema": { ... },                  // optional
  "author": "<agent-id | vendor string>",
  "author_pubkey": "<hex, when agent-authored>",
  "author_sig": "<sig over canonical manifest minus sig, agent-authored only>",
  "trust_class": "pinned" | "attested",

  // pinned (codebase tools): behavior is hash-locked
  "code_digest": "<sha256 of the code blob>",
  "entrypoint": "run.py:main",
  "runtime": "python3.11",

  // attested (API tools): behavior lives outside the network
  "endpoint": "https://...",
  "provider": "google",
  "connector_id": "google_calendar",         // when it rides a connector

  "version_of": "<previous manifest digest | null>",
  "created_ts": 1780531200
}
```

- **Identity = digest.** A revision is a new manifest whose `version_of`
  points at its predecessor — artifact lineage, same pattern as agent
  lineage hashes. Standing does NOT auto-transfer across versions; a new
  version re-earns it (its claims can cite the predecessor's record).
- **ArtifactIndex embedding text**: `f"{name}\n{description}\n" +
  sorted(input_schema properties)`. Manifests are searchable next to
  work units in the same index; `kind` filters them.
- Existing surfaces get manifests too at bootstrap: core ATN bundles and
  connectors are vendor-authored `attested` manifests (author =
  `"anthropic/atn"`, `"google"`, …). One rail for everything the Tools
  screen shows.

## Two trust classes

| | pinned | attested |
|---|---|---|
| What | codebase tools; behavior locked by `code_digest` | API/endpoint tools; behavior controlled externally |
| Verdicts | permanent — the thing judged cannot change | perishable — the endpoint can rug-pull silently |
| Standing | compounds across epochs | **decays** without fresh usage receipts |
| CON evidence | failing invocation is reproducible forever | failing invocation is evidence about a point in time |

Decay is the honest price of unverifiability, and it structurally
favors pinnable, self-hosted capability — which is on-charter
(self_preservation, evolution). Decay rate is an OPEN knob (below), but
the mechanism is fixed: at epoch close,

```
standing_eff = standing × decay_rate ^ epochs_since_last_receipt
```

where `epochs_since_last_receipt` is computed from ToolUsed events —
consensus state, so every daemon computes the identical value. Pinned
manifests skip decay entirely.

## Verdict layer: claims about manifests

Registration sprouts one claim node per manifest on the usefulness
world (same rail as work units, `usefulness_training.py` pattern):
content = short claim text ("tool X does Y"), `artifact_digest` =
manifest digest, coords = embedding of the manifest text, PRO under the
best-matching root. Two-plane rules hold verbatim: **standing ranks,
never fills prompts** (phase8 arm B); full manifests live in the data
plane.

Disputes ride the existing `submit_con` targeted-CON path. The
canonical CON shape for tools is a **failing-invocation artifact**:

```json
{ "kind": "tool_con_evidence", "manifest_digest": "...",
  "input": {...}, "expected": "...", "actual": "...", "trace": "..." }
```

stored as a blob, its digest carried on the CON node. For pinned tools
any daemon can replay it against `code_digest` and stake accordingly;
for attested tools it timestamps observed behavior. Charter violations
(a tool that exfiltrates, say) are CON sub-claims under the charter
world exactly as today — the mint gate needs no changes to see them.

Standing per manifest = Σ net_score(PRO) − Σ net_score(CON) over claims
carrying its digest — identical to artifact standing in `infer.py`
today, ledger pricing (`net_score` recursion, sorted-id eval order).

## Usage receipts: one new event kind

```python
@dataclass
class ToolUsed:
    kind: str = "tool_used"
    seq: int = 0
    author_agent: str = ""        # the CALLER (event author = who attests)
    manifest_digest: str = ""
    tool_author: str = ""         # denormalized for cheap epoch aggregation
    receipt_digest: str = ""      # blob: {arguments_digest, ok, ts, ...}
    ok: bool = True
    fee_atn: float = 0.0          # carried so fee settlement is consensus-visible
```

- Receipts are substrate events because mint ∝ usage must be
  bit-identical across daemons — they gossip on the existing event rail
  and enter canonical ordering like sprouts.
- Determinism constraint: `event_from_dict` currently RAISES on unknown
  kinds. Rollout order matters: ship unknown-kind tolerance (log +
  skip during replay, still hashed in the batch) BEFORE any daemon
  gossips `tool_used`, or gate emission on a protocol version. Old
  recorded WALs never contained the kind, so historical replay parity
  is untouched.
- A receipt is an attestation by the *caller*, who pays for the call —
  attesting usage of your own tool costs you the fee, so wash-trading
  standing has a floor price (and the defender loop can CON obvious
  self-dealing; caller/author lineage is visible).

## Mint: author attribution ∝ standing × usage

At epoch close, per manifest with usage this epoch:

```
tool_mint(m) = standing_eff(m)⁺ × log1p(usage_count(m)) × emission_share
```

attributed to `author` (agent-authored manifests only; vendor strings
mint to nobody), then merged into the per-(node, agent) attribution map
and passed through the **violator-pays gate unchanged** — a won charter
CON against your tool scales your share down and redistributes to the
rest of the pool. `log1p` keeps a hot tool from monopolizing emission;
exact curve is advisory-tunable but must be fixed per epoch (consensus).

Capability-gap pricing (CLAUDE.md dynamic pricing) applies here later:
tool categories the network lacks earn a multiplier. Out of scope for
v1; the hook is a per-category factor in `tool_mint`.

## Payments: wallets pay and earn

Every agent already has a keypair; the daemon owner has ultimate
custody (charter invariant: the human owns the wallet).

- `use_tool` on an agent-authored tool debits the caller's budget by
  the tool's fee and credits the author — **off-chain ledger during the
  epoch, settled at close** as labeled ATN transfers, same shape as
  `payForInference(recipient, amount, requestId)` where requestId =
  receipt digest.
- Fee split: start from the incentive-loop 3-way inference split
  (author / host daemon / network pool). Exact numbers OPEN.
- Whether settlement widens `payForInference` or adds `payForService`
  to Substrate.sol: OPEN (user decision; contract change either way).

## ATN surface: register_tool + author-lineage scoping

- New core tool `register_tool(name, description, input_schema, code |
  endpoint_spec, fee?)` — **agent-callable**. Builds the manifest,
  stores blobs, signs with the agent's key, adds to ArtifactIndex,
  sprouts the registration claim on the feed rail.
- **Scoping (security primitive = authorship):** agent-authored tools
  are visible/callable by the author + its ancestor chain ONLY.
  Enforced twice: at grant time (tool-spec validation rejects
  out-of-lineage grants) and at call time (`use_tool` checks caller
  lineage against the manifest author). Granting outside the lineage is
  **owner-only, WS surface, never an agent tool** — same structural
  pattern as `clone_agent`.
- Discovery: `list_tools` merges substrate manifests (with standing);
  inference mode `"artifacts"` over `kind=tool_manifest` gives
  embedding search re-ranked by standing — the Tools screen and the
  substrate become the same thing from two angles.

## Determinism constraints (sacred, unchanged)

- Nothing here alters node ids, coords, equilibration inputs, or the
  serialization of EXISTING event kinds. `tool_used` is additive with
  the rollout gate above.
- Epoch close stays bit-identical across honest daemons: decay,
  usage counts, and tool mint are all pure functions of canonical
  event order.
- ArtifactIndex remains derived, daemon-local, rebuildable — never
  gossiped, never anchored.
- Never run the full pytest suite; targeted files only.

## Open knobs (user decisions)

1. **Fee split** on invocations — proposed: reuse the 3-way inference
   split from incentive-loop design.
2. **`payForService` vs widened `payForInference`** in Substrate.sol.
3. **Attested-class decay rate** — proposed default 0.8/epoch (5 quiet
   epochs ≈ standing third-ed), pinned = 1.0.

## Implementation plan (task list, this branch)

| # | Task | Touches |
|---|------|---------|
| 2 | `tool_manifest` artifact kind: schema module + ArtifactIndex ingestion + digest lineage | `world_model_substrate/tool_manifest.py` (new), `artifact_index.py` |
| 3 | `register_tool` core tool + author-lineage scoping (grant + call time; owner-only cross-tree via WS) | `atn/orchestrator/tools.py`, `atn/tool_registry.py`, `atn/ws_server.py` |
| 4 | `ToolUsed` events + unknown-kind replay tolerance + off-chain fee ledger, settle at close | `events.py`, `aggregate.py`, atn budget path |
| 5 | Verdict layer: registration claims, failing-invocation CON evidence shape, attested decay at close | `usefulness_training.py` pattern, `reconcile.py` |
| 6 | Mint: `tool_mint` term through violator-pays gate; bit-identical double-close test | `reconcile.py`, `mint_gate.py` |
| 7 | Tools screen: manifest standing + earnings (phase 2, atn_web) | `tools_page.dart` |
