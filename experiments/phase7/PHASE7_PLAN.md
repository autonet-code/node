# Phase 7: corrected substrate-vs-RAG contest (two-plane inference)

## Why phase 7 exists — the phase 6 bug

Phase 6 (`phase6/run_contest.py:290`) called:

```python
probe = infer_with_world_model({"text": query_text}, mode="general")
```

It never passed the built `world`. With `world=None`, `_infer_general`
builds a fresh **charter world** (4 alignment axes, empty of toolz
knowledge) and locates the query in *that*. `probe_region_size` was 0 on
every row, and the "substrate" arm was formatted from an empty region —
i.e. it was actually **no-context haiku**. Phase 6's "substrate vs RAG"
was really "bare haiku vs RAG". Invalid.

## The fix

Phase 7 uses the new **two-plane inference** path
(`docs/two_plane_inference.md`): artifacts (full work-unit payloads) live
in a blob store + embedding index (`ArtifactIndex`); claim graph nodes are
debated verdicts *about* artifacts, referenced by sha256 digest.
Retrieval is embedding similarity re-ranked by graph standing:

```python
probe = infer_with_world_model(
    {"text": query_text},
    mode="artifacts",
    world=world,                 # the BUILT world, not None
    artifact_index=index,
    blob_store=store,
    k=5,
)
```

The world is threaded through explicitly, so retrieval runs over the
toolz corpus we actually built. The harness **asserts** every substrate
probe returns non-empty artifacts with non-None payloads and prints a
loud FAIL otherwise — the exact regression phase 6 shipped.

## Substrate build (N=10, all tendencies)

Same skeleton as phase 6's `build_world_for_n`, plus the data plane:

1. Ten tendencies (`correctness … documentation_fidelity`), anchors as in
   phase 6 (head one-hot + 64-dim embedding tail).
2. For each train entry:
   - **Artifact**: payload
     `{"problem": f"{name}\n\n{docstring}", "resolution": impl_full_source,
     "text": name}` → `ArtifactIndex.add_artifact` → `digest`.
     BlobStore rooted at `phase7/blobs/`, index at
     `phase7/artifact_index.jsonl`.
   - **Work-unit claim node**: sprout the observation node under the first
     tendency exactly as phase 6, then set
     `node.artifact_digest = digest` (plain attribute).
   - **Judge sub-claims**: sprout each cached judge claim under the
     work-unit node per active axis (as phase 6), and **also** set the
     same `artifact_digest` on every judge claim node so their verdicts
     price the artifact during `_infer_artifacts`' standing walk.
3. `equilibrate(world, max_rounds=4, tolerance=1e-3)` once at the end.

## Three contest arms (same haiku prompts / system as phase 6)

1. **none** — no context. The explicit baseline (this is what phase 6's
   broken "substrate" arm actually measured).
2. **rag** — phase 6's top-k-by-embedding context, made *fair*: each
   retrieved entry's judge claims (from `judge_cache`) are included as
   text under its source, so RAG sees the same distilled knowledge the
   substrate's verdict layer carries.
3. **substrate** — `infer_with_world_model(mode="artifacts", …)`. Context
   formatted from returned artifacts: each artifact's payload
   (problem + full-source resolution), its claim list with `net_score`
   and position, and a standing/final note. Total context capped to a
   char budget comparable to RAG.

## Scoring & outputs

- Doctest scoring via phase 5's `doctest_harness.grade_implementation`,
  identical to phase 6.
- `contest7_N10.jsonl` — one row per test problem, all three arms'
  impls + scores, plus substrate probe metadata (per-artifact digest /
  cosine / standing / final / payload-present, region size).
- `contest7_progress.jsonl` — appended per problem, mid-run readable.
- `aggregate7.json` — per-arm means and pairwise deltas
  (substrate−none, substrate−rag, rag−none).

## Offline smoke — `--mock`

`--mock` swaps the bridge call for a deterministic stub returning
`def <name>(...):\n    raise NotImplementedError()` (scores 0 — fine; the
smoke exercises the *pipeline*, not the LLM). It runs corpus →
substrate build → retrieval → context formatting → scoring →
aggregation fully offline, makes **zero** bridge/API calls, and asserts
the substrate probe invariant on every problem.

Smoke command:

```
python run_contest.py --mock --out contest7_mock.jsonl
```

## Real run (user-triggered later)

```
python run_contest.py --out contest7_N10.jsonl
```
