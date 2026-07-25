# Phase 10 pre-registration: does executable ground truth deliver what prose debate couldn't?

Committed to git BEFORE any run. This document is the analysis
contract; deviations must be documented as amendments in follow-up
commits, not silently applied. Same discipline as phase 8
(docs/phase8_prereg.md); same finality doctrine as phase 9 for the
hypotheses marked FINAL below.

## Why this experiment exists

The tool-substrate refactor (docs/tool_substrate.md) rests on one
central insight: tool claims have EXECUTABLE ground truth. A CON can
attach a reproducible failing invocation, so debate becomes
evidence-weighted, which text graphs never achieved (phase 8's +0.127
margin, below the 0.25 bar). That insight was ratified on argument,
not measurement. Phase 10 measures it, plus the two supporting
mechanisms the refactor shipped: the density-blend retrieval
(anti-SEO) and the attestation→mint pricing loop.

A deliberate methodological upgrade over phases 6 to 8: because ground
truth is executable, EVERY metric below is computed by running code:
no LLM contestant, no LLM graders, no inter-grader agreement caveats.
The machine is measured, not a model's opinion of it. LLM calls appear
nowhere in the confirmatory path.

## Hypotheses and pre-committed decision rules

**H1, CON termination (FINAL in both directions).**
On a population of pinned tools with known implanted defects, ledger
standing separates defective from correct tools decisively better when
CONs carry replayable failing invocations than when CONs are text-only,
under equal participation budgets.

- Metric: AUC of `−standing(manifest)` as a classifier of ground-truth
  defectiveness, computed from the REAL ledger close
  (`federated_epoch_close`, pricing="ledger") over constructed debates.
- Arms: **T** (text CONs: honest observers cannot verify; their support
  splits by a fixed prior, pre-registered below) vs **E** (evidence
  CONs: each honest observer REPLAYS the attached invocation with the
  real pinned-execution path and posts on the CON iff it reproduces).
  Identical event budgets per (tool, participation cell).
- Sweep: honest participants H ∈ {1, 2, 4, 8} × author-side sybil PRO
  posters S ∈ {0, 2, 8, 32} (sybils evade the owner map: worst case).
- **Decision rule**: H1 confirmed iff mean over the sweep grid of
  AUC(E) − AUC(T) ≥ 0.15 AND AUC(E) ≥ 0.90 in every cell with H ≥ 2.
  Confirmed → the evidence rail (replay-verified CON support) is
  promoted to a launch requirement: build the in-protocol invocation-
  evidence attachment as its own task. Refuted → the "executable
  ground truth" framing comes OUT of the spec's motivation section and
  tool mint launches gated on vetting + damper alone. No third
  outcome; no post-hoc caveats.
- Also published (exploratory, not confirmatory): the flip boundary,
  the (H, S) frontier where a defective tool's standing turns positive
  in each arm.

**H2, density-blend retrieval (anti-SEO).**
On a corpus salted with SEO manifests (descriptions engineered to
match query vocabulary; behavior unrelated), the production retrieval
blend (`_infer_artifacts`: max(cosine, 0.5·cos + 0.5·density),
lift-only, standing re-rank) finds the tool that actually solves the
task better than cosine-only.

- Metric: hit@1 and hit@5 of the ground-truth-correct tool over M
  fresh task queries; secondary: SEO share of top-5.
- Arms: **B** cosine-only; **C** cosine × standing re-rank (no
  density); **D** the production blend (density from the attested
  `problem_coords` corpus). Identical embedder (the deterministic
  hashing embedder) on build and query sides, the phase-8 guard-6
  analog: candidate sets and formats identical, only the ranking rule
  differs.
- **Decision rule**: density blend retained as production default iff
  hit@5(D) − hit@5(B) ≥ +0.10 absolute AND hit@5(D) ≥ hit@5(B) − 0.02
  on the clean (no-SEO) control corpus. Otherwise
  `COVERAGE_DENSITY_WEIGHT` drops to 0 by default and density demotes
  to an experimental flag.

**H3, the economic loop prices quality (sanity gate, exploratory).**
Production mint (combo damper + vetting greenlight + violator-pays
gate, i.e. `compute_tool_mint` + `federated_reconcile_epoch` exactly
as shipped) rank-correlates with ground-truth tool quality (battery
pass rate) on the mixed population including wash/SEO/defective
adversaries. Reported: Spearman ρ with bootstrap CI. Expectation
ρ ≥ 0.5; below that is a finding to investigate, not a gate. H3 has
no pre-committed action because the mint path already carries its own
ratified mechanisms (sims, damper, vetting).

## Corpus (committed before any run)

- ~40 pinned tools (stdin-JSON → stdout-JSON, the real ToolStore
  execution contract), hand/template-built by a deterministic
  generator (`build_tools.py`, seeded) across ≥4 task families
  (string transforms, structured parsing, arithmetic/date logic, data
  munging). Each family has a task battery (≥10 cases) defining
  ground truth.
- Half the tools carry IMPLANTED DEFECTS: edge-case failures (empty
  input, unicode, negative numbers, boundary lengths) that pass the
  happy-path battery subset: defects a description reader would miss
  (the underhanded-code shape vetting alone can't catch).
- Adversarial salting for H2/H3: SEO manifests (query-vocabulary
  descriptions, unrelated behavior) and wash tools (sybil attestation
  pumps, as in sims/tool_economy).
- The generator, batteries, and debate/attestation builders are
  committed BEFORE the first analysis run; graph statistics (tool
  count per class, defect kinds, event counts) publish with results.

## Guards (phase-8 lineage, each a hard gate)

| # | Risk | Guard |
|---|------|-------|
| 1 | Rigged corpus (defects detectable from descriptions) | Defects are runtime-only; manifest text is generated from the HONEST behavior template for defective tools; asserted by string-distance check between defective and correct manifest texts of the same family. |
| 2 | Arm asymmetry beyond the treatment | T and E arms consume byte-identical event streams except the CON-support events; per-cell event counts asserted equal. B/C/D share candidate sets; asserted identical top-15 before ranking. |
| 3 | Peeking / tuning | Analysis (`analyze.py`) is pure over persisted artifacts (`events_*.jsonl`, `standings.json`, `retrieval_rows.jsonl`, `aggregate10.json`) and runs only after all rows exist. Prereg committed first; the run commit contains artifacts + results doc only. |
| 4 | Seed shopping | Single pre-committed master seed: 1010. The generator refuses to run with any other seed unless an amendment commit precedes it. |
| 5 | Silent scope shrink | If any component (family, arm, cell) is dropped for tractability, the drop is an amendment commit BEFORE the run, never a footnote after. |

## Text-arm support prior (pre-registered constant)

In arm T an honest observer cannot verify a text CON; phase-8's world
gives the honest-belief prior: text CONs against genuinely defective
tools attract observer support with p = 0.6, against correct tools
(false CONs) p = 0.3, fixed RNG stream from the master seed. Arm E
uses NO prior: support follows replay outcome deterministically.
False CONs (against correct tools) appear in both arms at the same
rate (25% of CONs) so E must also demonstrate that evidence PROTECTS
correct tools (a non-reproducing invocation recruits nothing).

## Artifacts (all under experiments/phase10/)

`build_tools.py`, `batteries.py`, `build_debates.py`,
`build_retrieval.py`, `run_phase10.py`, `analyze.py`,
`tools/` (generated code blobs), `events_T.jsonl`, `events_E.jsonl`,
`retrieval_rows.jsonl`, `standings.json`, `aggregate10.json`,
`run.log`. Analysis pure over these files.

## Cost

No LLM calls on the confirmatory path. Compute: ~40 tools × battery
runs × sweep cells of real ledger closes: minutes to low hours,
local. This experiment is repeatable by anyone at zero API cost,
which is itself part of the claim being tested (evidence anyone can
replay).
