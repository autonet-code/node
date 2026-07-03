# Phase 8 results (2026-07-03)

Prereg: docs/phase8_prereg.md (d0f3f82 + amendments 191d804, aff6c42,
77486af — all committed before any contest call). Raw artifacts in
substrate_experiment/phase8/ (contest_rows.jsonl, grades_contest.jsonl,
aggregate8.json, label_map.jsonl, llm_cache/).

Setup: 200-unit corpus from real autonet session traces; 25 questions
selected where bare Haiku fails (calibration mean 1.12/5, gate #3
passed); 5 arms; two blind graders (Opus 4.8, Sonnet 5; r=0.57, clears
the 0.4 gate); 4% null grades (under the 10% abort gate); no guard
aborts; context parity held.

## Per-arm means (95% bootstrap CI, n=25)

| Arm | Mean |
|---|---|
| A bare | 1.220 [1.113, 1.340] |
| B rag (payloads only) | **1.587** [1.373, 1.827] |
| C ledger (payloads + verdict claims, vote re-rank) | 1.307 [1.153, 1.507] |
| D256 substrate (equilibrated re-rank) | 1.433 [1.253, 1.653] |
| D64 | 1.387 [1.233, 1.567] |

## Contrasts (Holm-corrected)

- **B−A = +0.367, p=0.012** — retrieval genuinely helps on an unknown
  domain. First positive evidence for the substrate value prop's
  retrieval leg (phase 7's negative was a saturated-domain artifact).
- **C−B = −0.280, p=0.012** — injecting verdict claims INTO the context
  hurts. Verdicts belong in ranking, not in the prompt.
- **Primary: D256−C = +0.127, Holm p=0.054, CI [+0.03, +0.24]** —
  equilibrated standing beats vote-count standing by a small, probably
  real, margin that fails the pre-registered bar (p<0.05 AND diff≥0.25).
- D64−D256 = −0.05, n.s. — embedding dim 64 vs 256 doesn't matter here.

## Pre-registered decision

**DEMOTE equilibration.** Expansion rule not triggered (CI excludes 0).
Per the gate: mint pricing moves toward explicit debate standing
(economics change requires user ratification); equilibration becomes an
experimental kernel; quantum-inference path parked.

Honest nuance recorded: the primary effect is positive with a CI
excluding zero — equilibration is not worthless, it is small. It failed
the magnitude bar that was set, before data, as "earns its complexity."

## Design implications beyond the gate

1. Retrieval arm (payloads only) was the best overall — the two-plane
   data path is validated as the product surface.
2. Standing should re-rank and price, never fill context (C−B).
3. All context arms remain far from a knowledgeable ceiling (~1.6/5) —
   retrieval quality / corpus depth, not standing math, is the
   binding constraint on answer quality.
