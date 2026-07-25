# Phase 10 results (2026-07-05)

Prereg: docs/phase10_prereg.md (6da0264, committed before any run; no
amendments were needed: no scope drops, master seed 1010 used as
committed). Raw artifacts in experiments/phase10/ (corpus.json, tools/,
events_T.jsonl, events_E.jsonl, retrieval_rows.jsonl, mint_rows.jsonl,
standings.json, aggregate10.json, run.log). Builders + guard harness
committed before the confirmatory run; analyze.py is pure over the
persisted rows. 9/9 guard tests pass. No LLM calls anywhere on the
confirmatory path: every number below is computed by running code.

Setup: 79 pinned tools from the seeded generator (36 correct, 31 with
runtime-only implanted defects, 6 SEO, 6 wash) across 4 task families;
defectiveness MEASURED by battery pass rate through the real subprocess
execution contract. H1 population = the 67 correct+defective tools;
39 CONs fired (8 false, 25% of the 36 correct tools per the
pre-registered rate). H1 sweep = 4 H x 4 S x 2 arms = 1072 debates per
arm (14,231 / 14,812 events), each standing read from the real ledger
replay. Full run: 132 s.

## H1, CON termination (FINAL in both directions): **REFUTED**

AUC of −standing classifying ground-truth defectiveness, per (H, S) cell:

| cell | AUC(T) | AUC(E) | gap |
|---|---|---|---|
| H1_S0 | 0.918 | 1.000 | +0.082 |
| H1_S2 | 0.922 | 1.000 | +0.078 |
| H1_S8 | 0.936 | 1.000 | +0.064 |
| H1_S32 | 0.932 | 1.000 | +0.068 |
| H2_S0 | 0.938 | 1.000 | +0.062 |
| H2_S2 | 0.921 | 1.000 | +0.079 |
| H2_S8 | 0.959 | 1.000 | +0.041 |
| H2_S32 | 0.929 | 1.000 | +0.071 |
| H4_S0 | 0.986 | 1.000 | +0.014 |
| H4_S2 | 0.975 | 1.000 | +0.025 |
| H4_S8 | 0.948 | 1.000 | +0.052 |
| H4_S32 | 0.968 | 1.000 | +0.032 |
| H8_S0 | 0.976 | 1.000 | +0.024 |
| H8_S2 | 0.983 | 1.000 | +0.017 |
| H8_S8 | 0.970 | 1.000 | +0.030 |
| H8_S32 | 0.966 | 1.000 | +0.034 |

- **Mean AUC(E) − AUC(T) = +0.0483, which fails the pre-committed 0.15 bar.**
- AUC(E) floor: 1.000 in every cell (bar 0.90 in H≥2 cells: held, and
  then some).

**Pre-registered decision: H1 REFUTED.** The consequence fires as
committed: the "executable ground truth" framing comes OUT of
docs/tool_substrate.md's motivation section, and tool mint launches
gated on vetting + damper alone. The evidence rail is NOT promoted to a
launch requirement. (Spec edit is the follow-up action; this doc is the
trigger.)

Honest nuance recorded, phase-8 style: evidence separation is
*perfect* (AUC(E) = 1.000 in all 16 cells, including 32-sybil worst
cases), and the realized gap (+0.0483) equals the structural ceiling
(1 − mean AUC(T)) exactly. E extracted every point of headroom text
debate left, and it still was not 0.15, because the text arm with the
phase-8-derived honest-belief priors (0.6 vs 0.3) already ranks this
population at AUC ≈ 0.95. The bar was set on the *delta* over text
debate; for it to have been reachable at all, text debate would have
had to rank at AUC ≤ 0.85. That interaction between the pre-registered
prior and the pre-registered bar was not noticed at prereg time; it is
recorded here as a design lesson, not re-litigated: FINAL means final.

What the AUC hides (visible in the raw rows): text debate *convicts
innocents*. At H=8, S=0 the 8 correct tools carrying false CONs sit at
mean standing −5.5 in arm T (honest observers recruited by prior alone)
vs −1.0 in arm E (a non-reproducing invocation recruits nobody). AUC is
a ranking metric and both arms rank correct above defective; but in
absolute terms T's false-CON damage is real and E's protection of
correct tools is exactly the property the refactor argued for. It
simply is not what the pre-registered metric measured.

### Exploratory: the flip boundary (defective standing turns positive)

Fraction of the 31 defective tools whose ABSOLUTE standing is positive
(the sybil PRO flood beat the CON), per cell:

| cell | T flip | E flip |
|---|---|---|
| H1_S2 | 1.00 | 1.00 |
| H2_S2 | 0.52 | **0.00** |
| H2_S8 | 1.00 | 1.00 |
| H4_S2 | 0.13 | **0.00** |
| H4_S8 | 1.00 | 1.00 |
| H8_S2 | 0.00 | 0.00 |
| H8_S8 | 0.90 | **0.00** |
| H8_S32 | 1.00 | 1.00 |

(all S=0 cells: 0.00 both arms; omitted cells are 1.00 both arms)

Evidence turns the flip boundary from probabilistic to deterministic:
E holds every defective tool negative up to roughly S < 2H (all
observers replay, so the CON's weight is exactly 1+2H), where T leaks
through at rates set by the prior. At H=8, S=8, text lets 90% of
defective tools stand positive; evidence lets none. Past S ≈ 2H both
arms drown: standing is linear in sybil count, and no debate mechanism
at fixed honest participation survives an unbounded PRO flood. That
frontier is the damper/owner-map's job, not the debate layer's.

## H2, density-blend retrieval (anti-SEO): **RETAINED**

108 queries per corpus (36 tasks x 3 paraphrases), hashing embedder both
sides, identical candidate sets across arms (asserted by guard test):

| arm | salted hit@1 | salted hit@5 | salted SEO-share | clean hit@5 |
|---|---|---|---|---|
| B cosine-only | 0.093 | 0.213 | 0.924 | 0.241 |
| C cosine x standing | 0.093 | 0.213 | 0.924 | n/a |
| D production blend | **0.306** | **0.361** | 0.861 | 0.398 |

- **hit@5(D) − hit@5(B) = +0.148 ≥ +0.10** (salted): bar cleared.
- Clean-corpus control: D − B = +0.157, no regression (tolerance −0.02);
  density HELPS even without SEO salting, because demonstrated coverage
  rescues tools whose manifest vocabulary happens to miss the query.
- hit@1 more than triples (0.093 → 0.306).

**Pre-registered decision: density blend retained as production
default.** COVERAGE_DENSITY_WEIGHT stays 0.5.

C == B exactly: by design, H2's world gives every manifest identical
standing (one author post), isolating the density lever; the standing
re-rank is H1's lever, not H2's. Absolute hit rates are low across all
arms: the trigram hashing embedder is the pre-registered, deterministic
floor, not a product-quality retriever; the confirmatory quantity is the
delta, which is large and one-directional.

## H3, the economic loop prices quality (exploratory sanity gate)

Production mint (real `federated_epoch_close`: compute_tool_mint, combo
damper, VET_QUORUM=2 distinct-fleet greenlight, violator-pays gate) over
the full 79-tool mixed population:

- **Spearman ρ(mint, battery pass rate) = 0.4925, bootstrap 95% CI
  [0.298, 0.662]**, a hair under the 0.5 expectation; CI comfortably
  excludes 0.
- Mean mint by class: correct 33.9 > defective 11.8 > wash 8.4 > SEO 1.1.
  The ordering is right; the correlation is dragged under 0.5 by wash
  tools (pass rate ~0, mint ~8: sybil callers on distinct wire keys with
  an EMPTY owner map (the documented degraded mode) still buy log1p
  breadth per sybil). This is the sims' known result reproduced through
  the real close: the combo damper *bounds* wash mint (≈ ¼ of honest
  mean, at 10-30x the receipt volume) but only the owner map *zeroes*
  it. No pre-committed action fires (per prereg, H3 has none); the
  finding matches the already-ratified mechanism stack and sharpens the
  case for sponsored registrations shipping early.

## Graph statistics

79 tools (36 correct / 31 defective / 6 SEO / 6 wash); 4 families x 9
tasks; batteries 10-14 cases per task (10 happy + hand-built edge cases;
defect kinds: boundary 19, negative 11, unicode 1); defective battery
pass rates 0.08-0.93 (all < 1.0, measured); wash 0.0-0.23; SEO 1.0 (they
behave, they just lie). H1: 2,144 debates, 29,043 events total, every
standing from the real ledger replay. H2: 540 retrieval rows. H3: one
real close over 1,519 canonical events, 79/79 greenlit, total mint
1643.1.

## Limitations

1. **The H1 bar interacted with the H1 prior.** With text-arm priors
   0.6/0.3 and n=67, AUC(T) lands ≈ 0.95, capping the achievable gap at
   ≈ 0.05, a third of the bar. The refuted verdict is therefore partly
   a statement about the bar's construction, not only about the
   mechanism. Recorded as a prereg-design lesson (bars on deltas need a
   pre-computed ceiling check); the verdict stands, FINAL both ways.
2. The text-arm prior itself is a model (phase-8 honest-belief rates
   applied uniformly). Real text debate could rank better or worse; arm
   T is a simulation of crowd belief, while arm E is the real replay
   path end to end.
3. One CON per tool, one critic, one debate depth. Deep contested
   graphs (CON-on-CON) are phase 9's still-unrun territory.
4. The hashing embedder floors H2's absolute numbers; a semantic
   embedder could compress or widen the density delta. The prereg fixed
   the embedder deliberately (determinism, zero cost): the delta is
   confirmatory, the absolute rates are not.
5. H3's population and receipt volumes are one seeded draw (79 tools,
   fixed adversary mix); ρ's CI reflects resampling of that population,
   not variation across corpus designs.
6. Sybil PRO posters in H1 attack only with direct manifest-node PRO
   posts; richer attacks (sybil CONs on the CON) were out of scope.

## Bottom line

The tool-substrate refactor's central *argument*, that evidence beats
prose, shows up as perfect separation and deterministic sybil resistance in
arm E, but it REFUTED its own pre-registered bar because text debate
under honest-majority priors was already strong on ranking. Per the
committed rule: the "executable ground truth" framing comes out of the
spec's motivation; launch gates on vetting + damper. The density blend
earned its default (only mechanism of the three to clear its bar
outright). The mint loop prices quality directionally (ρ ≈ 0.49) with
the known owner-map gap as the residual leak.
