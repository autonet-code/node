# Two-plane substrate: consensus plane vs data plane

Decision (2026-07-03): the claim graph is demoted from knowledge
representation to **verdict layer**. Artifacts (full work-unit payloads)
live in the blob store and a daemon-local embedding index; graph nodes
are debated judgments *about* artifacts, referenced by sha256 digest.
Inference retrieves artifacts by embedding similarity and re-ranks by
graph standing. Equilibration continues to run only on the claim layer.

Rationale: experiments showed the graph-as-retrieval-index loses to
plain embedding retrieval (experiment_utility.py, May 2026) and the
locate→render path returned labels, not payloads (phase6 artifact,
May 2026). The substrate's differentiator is the trust picture it
attaches to a tool, not retrieval mechanics. (This was written as
"debate-priced verdicts"; the debate rail retired at v3 and the vetting
gate at v4.1 — the live differentiator is now the rep-weighted review
scores that drift a tool's position and rank discovery. See
`docs/tool_substrate.md`, Decision 2026-07-09.)

## Components

### 1. `nodes/common/world_model_substrate/artifact_index.py` (new)

```python
class ArtifactIndex:
    def __init__(self, blob_store, embedder=None, index_path=None, dim=DEFAULT_DIM): ...
    def add_artifact(self, payload: Dict[str, Any]) -> str:
        """blob_store.add_json(payload) -> digest; embed payload text;
        append (digest, embedding, label) to index_path (jsonl).
        Idempotent per digest. Returns the sha256 hex digest."""
    def search(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Top-k (digest, cosine) by embedding similarity."""
    def has(self, digest: str) -> bool: ...
    def rebuild(self) -> int:
        """Re-embed every blob-store JSON payload that looks like an
        artifact (has 'problem' or 'text' key). Returns count."""
```

- Embedding text for a work-unit payload: `f"{problem}\n\n{resolution}"`;
  for generic payloads, the `text` field.
- Embedder default: `default_usefulness_embedder(dim=dim)` from
  `usefulness_coords`.
- The index is **derived, daemon-local state** — NOT consensus state.
  It must be rebuildable from blobs. Never gossiped, never anchored.
- Persistence: jsonl at `index_path`; loaded eagerly on construction;
  in-memory numpy matrix for search.

### 2. Event change: `SubClaimSprouted.artifact_digest` (optional field)

- New optional field `artifact_digest: str = ""` on `SubClaimSprouted`
  in `events.py`. Serialized in `to_dict()` only when non-empty;
  `event_from_dict` tolerates absence (back-compat with recorded WALs
  and gossiped events from older daemons — replay of old logs MUST
  produce identical worlds).
- Federated replay: the digest rides the event; peers fetch the blob
  lazily via the existing p2p blob protocol when rendering. The digest
  does not participate in node ids, coords, or equilibration —
  bit-identical epoch close is unaffected.

### 3. Ingestion (`world_service.submit_work_units`, `usefulness_training`)

- Full `(problem, resolution, outcome)` payload →
  `{"problem": ..., "resolution": ..., "outcome": outcome_dict}` →
  `ArtifactIndex.add_artifact` → digest.
- The per-work-unit claim node keeps its claim-text content and gains
  `artifact_digest` (stored on the node as attribute and on the
  emitted SubClaimSprouted event).
  **[SUPERSEDED by v3 — see Addendum 2026-07-08: work units no longer
  sprout claim nodes at all; ingestion stops at `add_artifact`.]**
- `submit_con` unchanged (disputes target claim nodes, which now
  transitively price artifacts). **[STALE as of v3: `submit_con` /
  `submit_support` are retired from the live path.]**

### 4. Inference (`infer.py`)

New mode `"artifacts"`:

```python
def _infer_artifacts(input_data, world, artifact_index, blob_store,
                     k=5) -> Dict[str, Any]
```

- Query text: `input_data["text"]` (or `query`/`question`).
- `artifact_index.search(query, k=k*3)` → candidates.
- Standing per digest: over all claim nodes in the world with
  `artifact_digest == digest`:
  `standing = Σ net_score(PRO-positioned) − Σ net_score(CON-positioned)`.
- Re-rank: `final = cosine * (1.0 + tanh(standing))`, keep top k.
- Return shape:
  `{"mode": "artifacts", "artifacts": [{digest, cosine, standing,
    final, payload, claims: [{content, position, net_score}]}, ...]}`
  where `payload = blob_store.get_json(digest)` (None tolerated if the
  blob hasn't replicated yet — include the digest anyway).
- `infer_with_world_model` gains optional `artifact_index=`,
  `blob_store=` kwargs and dispatches `mode="artifacts"` when both are
  provided and the input isn't turn-shaped. Existing alignment/general
  modes untouched.

### 5. WorldService wiring

- `WorldService.__init__` constructs an `ArtifactIndex` next to its
  WAL (`<state_dir>/artifact_index.jsonl`) sharing the daemon's
  blob store when available (else a local `BlobStore` under
  `<state_dir>/blobs/`).
- New method `infer_artifacts(query_text, k=5)` → `_infer_artifacts`
  under the service lock.

## Constraints (all agents)

- Digests are sha256 hex from `blob_store.py` — NOT IPFS CIDs.
- Determinism of federated epoch close is sacred: nothing in this
  refactor may alter node ids, coords, equilibration inputs, or event
  ordering for existing event kinds.
- Never run the full pytest suite. Targeted files only.
- The scoring/re-rank formula is advisory (daemon-local inference),
  not consensus — exact constants may be iterated freely.

## Out of scope for this pass

- Scoped/per-tendency equilibration (the O(N²) wall) — separate work.
- Any change to mint, reconcile, or the gate.
- P2P blob replication improvements.

## Addendum: phase8 outcome (2026-07-03)

Phase 8 (`docs/phase8_prereg.md`, `docs/phase8_results.md`) contested this
design end-to-end. Two findings feed back into this document:

- **Retrieval validated.** Plain artifact retrieval (payloads only, no
  verdict text) beat no-context by +0.367 (Holm p=0.012) — the two-plane
  data path (artifacts in blob store + `ArtifactIndex`) is the product
  surface, confirmed on a real, unsaturated domain.
- **Verdicts-in-context hurt.** Injecting claim/verdict text into the
  prompt alongside payloads cost −0.280 (Holm p=0.012) relative to
  payloads-only. `render_context()` in `infer.py` therefore ships the
  arm-B format: standing-ranked payload text only (problem + resolution,
  proportional truncation), no claims/verdict text in the rendered
  prompt. Claims remain available in the structured result for
  programmatic consumers. Standing ranks and prices; it does not fill
  context. See `docs/ledger_pricing.md` (Change 3) for the exact
  `render_context` / `infer_artifacts(render=...)` contract.
- **Equilibration demoted.** The re-rank geometry contrast (equilibrated
  vs vote-count standing) cleared significance but not the pre-registered
  magnitude bar, so epoch-close pricing now defaults to the ledger mode
  (`net_score` tree recursion) described in `docs/ledger_pricing.md`;
  equilibration is an opt-in experimental kernel pending
  `docs/phase9_depth_experiment.md`. This does not change the artifact
  plane / verdict plane split above — it changes how standing on the
  verdict plane is computed by default.

## Addendum: substrate v3 (2026-07-08)

Per the v3 decision (`docs/tool_substrate.md`, Decision 2026-07-08),
the verdict plane narrows to TOOL MANIFESTS only:

- **Work units leave consensus.** `submit_work_units` stops sprouting
  claim nodes; ingestion is `add_artifact` only (the ArtifactIndex was
  always daemon-local and never gossiped — §1 stands unchanged).
  Retrieval over work-unit artifacts becomes pure embedding ranking,
  which is exactly the arm phase8 validated (+0.367, payloads-only).
- **Re-rank formula (§4) changes for tool manifests**: standing is
  retired; `final = cosine × (1 + tanh(mean(head[correctness],
  head[simplicity])))` where `head` is the tool's review-drifted
  charter head. Work-unit artifacts rank by pure cosine (their claim
  nodes no longer exist). The coverage-density blend is unchanged.
- `submit_con` / `submit_support` retire from the live path; the
  arm-B render rule (standing ranks, never fills the prompt) carries
  over as "ratings rank, never fill the prompt".
