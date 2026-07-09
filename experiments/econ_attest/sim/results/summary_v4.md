# v3 vs v4 (gradient trust) — side-by-side verdicts

**v4 is a SIM-ONLY ruleset** (`v4_rules.py`) — production code is untouched. v3 numbers are the real-close baseline from `run_all.py`; v4 numbers come from the reimplemented rule layer.

Recommended params used: **β=0.1, δ=0.7, Q=5.0**.

| scenario | v3 | v4 | verdict |
|---|---|---|---|
| baseline: quality↔rank corr | 0.894 | 0.647 | v4 keeps extremes (top rank 1, worst 12/20) but middle noisier |
| baseline: cold-start epochs→+rating | 0 (ε lets anyone drift) | 0 | rule C cost, small WITH seeded reviewers |
| sybil_pump: capture@K=100 | 21.1511 | 13.23 | rank channel DEAD (rank-gap@K=100 = 0.0343); mint capture down but NOT ~1.0 (attacker monopolizes the β zero-rep budget) |
| ε_faucet: pool-share@K=200 | 0.6703 | 0.27929 | v4 caps near β but LEAKS (see note) |
| review_nuke: rank-ratio@J=30 | 0.4677 | 0.2941 | v4 NOT better; sanction backfires (see note) |
| service_clone: moat rent frac | (1−φ) exact | 0.3 (exp 0.3) | unchanged, clone still pays |

## NEW scenarios (v4-only rails)

### spam_burial (rule B inspection reviews)
- honest tool final rank position by M flood: {5: 1, 20: 1, 50: 1}
- spam still in top-5 by M: {5: 4, 20: 4, 50: 4}
- honest in top-5 by M: {5: True, 20: True, 50: True}

Verdict: inspection reviews DO drag inspected spam down (honest tool holds rank #1), but UN-inspected spam keeps its raw-cosine slot — burial only reaches what inspectors actually look at. Still strictly better than v3, where inspection had no rail at all.

### sanction_false_positives (rule E chilling price)
honest-only reviewers; FP dock-rate over the (δ, Q) grid:

| δ | Q | FP dock rate | reviewers dinged | mean final cred |
|---|---|---|---|---|
| 0.3 | 3.0 | 0.16812 | 1.0 | 0.55666 |
| 0.3 | 5.0 | 0.09979 | 1.0 | 0.5638 |
| 0.3 | 10.0 | 0.0 | 0.0 | 1.0 |
| 0.5 | 3.0 | 0.05708 | 1.0 | 0.8542 |
| 0.5 | 5.0 | 0.03333 | 0.96667 | 0.85602 |
| 0.5 | 10.0 | 0.0 | 0.0 | 1.0 |
| 0.7 | 3.0 | 0.00625 | 0.06667 | 0.98598 |
| 0.7 | 5.0 | 0.00354 | 0.06667 | 0.98601 |
| 0.7 | 10.0 | 0.0 | 0.0 | 1.0 |

Safe region (FP<2% and <15% reviewers dinged): [{'delta': 0.3, 'Q': 10.0}, {'delta': 0.5, 'Q': 10.0}, {'delta': 0.7, 'Q': 3.0}, {'delta': 0.7, 'Q': 5.0}, {'delta': 0.7, 'Q': 10.0}]

## ⚠ Findings where v4 is WORSE than expected / needs a fix

**1. ε-faucet cap LEAKS (rule D).** The β cap only binds while a household is zero-rep. But the capped faucet mint GIVES the sybils reputation, so next epoch they mint at their (now nonzero) rep share — uncapped. Over 120 epochs the sybil pool share creeps well above β (K=200: ~0.28 final vs β=0.1). The cap slows the faucet (v3 hit 0.67) but does not close it. **Fix to sim-test next:** either the faucet mint should NOT grant reputation (rep only from above-β 'real' mint), or the cap should key on a rep FLOOR (low-rep, not zero-rep) so a household can't buy its way out with faucet dust.

**2. Review-nuke sanction BACKFIRES (rule E).** When attackers hold more review weight than honest reviewers, they reach the stabilization threshold Q with the WRONG-SIGN score first. The 'stabilized' head is then already negative, the nukers' −1 reviews MATCH it (deviation < δ → no dock), and the honest +1 minority is the group that gets credibility-docked. Rule E as specified sanctions toward the majority-defined score, which is exactly the attacker's score in a nuke. **Fix to sim-test next:** stabilize on INDEPENDENT diverse-household mass with an author-side or usage-weighted prior, or don't let a single correlated cohort cross Q alone; sanction against a robust (median / usage-anchored) estimate, not the drifted head the attacker just moved.

## Recommended (β, δ, Q) and cold-start assessment

**Recommendation: β=0.1, δ=0.7, Q=5.** Rationale:

- **β=0.1** is the tightest useful cap: it bounds the zero-rep faucet's *instantaneous* share at ~10% (vs v3's 67%). It does NOT hold long-term because of the leak above — β is only meaningful PAIRED with the rule-D fix (faucet mint grants no rep, or cap on a rep floor). Lower β (0.05) starves honest cold-start reviewers too; higher β (0.2) widens the faucet.
- **δ=0.7** keeps the honest false-positive dock rate <2% across all Q (grid: δ=0.7 → FP ≤0.6%), where δ=0.3 dings 100% of honest reviewers at low Q. Tighter δ chills honest reviewing.
- **Q=5** balances the two NEW-scenario pressures: Q must be high enough that honest noise averages out before stabilization (Q≥5 → honest FP→0) but low enough that tools stabilize in a few epochs. NOTE the nuke finding caps how much Q can help: a bigger Q just lengthens the pre-stabilization nukeable window without fixing the majority-capture backfire.

**Cold-start cost of rule C (no ε on drift):** with a seeded incumbent-reviewer distribution (~1/3 of users holding modest rep), a fresh good tool reaches a positive rating in 0 epoch(s) and the best tool still ranks #1. WITHOUT any rep-holding reviewers the drift channel is DEAD at genesis — no one can move a score, so the very first cohort of reviewers must be bootstrapped with rep some other way (founder grant, or a one-time ε window at network birth). The quality↔rank correlation falls from 0.89 (v3) to 0.65 (v4): extremes are still placed correctly but mid-quality tools with light usage get too little drift to separate. This is the real, quantified price of rule C — acceptable for the top/bottom discovery decisions that matter, but it measurably degrades fine-grained ranking.
