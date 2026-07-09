# Substrate v3 tool-economy attack sim

An agent-based simulation harness that runs the **real** close-time
economics of the Substrate v3 tool economy over synthetic agent
populations for hundreds of epochs, fast (pure Python, no libp2p, no
chain). It empirically tests the economic claims and the attacks
catalogued in `../attacks.md`.

## Fidelity: what is REAL vs STUBBED

**Real (imported from production, never reimplemented):**

- **Mint** — `compute_tool_mint` / `federated_epoch_close` in
  `nodes/common/federated_reconcile.py`: household collapse, log1p usage
  damper, voice weighting, owner-map + wire-key sybil exclusions,
  composition fan-out over the declared-dep DAG.
- **Vetting greenlight** — distinct-fleet quorum (`VET_QUORUM=2.0`),
  royalty split, `weight = 1/(1+busts)`. Same function.
- **Position drift** — the per-axis mint-weighted running centroid of
  review scores with author-prior mass 1.0 (v3 `tool_positions`). Same
  function.
- **Emission pool** — `apply_emission_pool` via
  `federated_epoch_close(emission_pool=...)`: total minted ATN per epoch
  is normalized to a fixed pool (`BASE_EMISSION_PER_EPOCH = 100` +
  recycled fees), which makes mint **zero-sum among authors**. A
  `conservation_check` asserts `total_mint <= pool` every epoch.
- **Discovery ranking lift** — `tanh(mean(correctness, simplicity))` on
  the **drifted** head. Replicated *verbatim* from
  `infer._infer_artifacts` (lines 321-327) in `harness.rank_score`,
  because the production ranker needs a live `artifact_index` +
  `blob_store` we do not stand up. The formula is real; only its input
  base is stubbed (next point).
- **Voice weight formula** — `ε + household_reputation / reputationSupply`,
  computed exactly as `voice_state.read_voice_state` does, but from the
  sim's reputation ledger instead of on-chain checkpoints.

**Stubbed (thin, documented):**

- **Retrieval cosine** — each tool carries a scalar `topic_match ∈ [0,1]`
  standing in for `artifact_index` cosine relevance to a query. The
  ranking *lift* applied to it is the real formula. (The brief explicitly
  permits this simplification.)
- **Reputation** = cumulative minted ATN (1:1), matching the dual-token
  lockstep (`Substrate.sol` mints reputation == ATN per training event).
  Reputation feeds the next epoch's voice weights.
- **Event stream** — we fabricate the canonical `sub_claim_sprouted` /
  `tool_used` events in the exact shapes `tests/test_tool_mint.py` uses,
  then drive the real close over them. Signing-key assignment reproduces
  the wire-dedup (co-hosted sybils) and distinct-fleet (vetter)
  semantics.

**Not exercisable through the close (noted, not faked):**

- The **violator-pays gate** and **CON-triggered bust** are DORMANT in v3
  (`apply_gate=False` default); we don't drive them. Vetting *quorum* is
  checked inside `compute_tool_mint`, so we exercise greenlight, but the
  bust/claw-back rail has no live trigger — consistent with attacks.md's
  note that it is dormant.

## How to run

```bash
cd experiments/econ_attest/sim
python run_all.py            # full run, ~30s, writes results/*.json + summary.md
python run_all.py --quick    # short epoch counts for a fast smoke check
```

Each scenario is also directly callable from `scenarios.py` with its own
parameters (epochs, seeds, sweep values). Everything is seeded and
deterministic.

## The discovery feedback loop

Each epoch, agents pick which tool to use with probability proportional
to its current discovery `rank_score` (real lift on the drifted head).
Usage + axis reviews feed the real mint and drift functions; drift
updates the head; the new head changes next epoch's ranks. This closes
the loop **reviews → drift → ranking → usage → mint** through production
code.

## Per-scenario verdicts (full run, see results/summary.md for numbers)

- **baseline_honest** (C1, C3): quality vs cumulative-mint corr ≈ 0.92,
  quality vs rank corr ≈ 0.89; worst-quality tool sinks to last rank.
  **C1 + C3 supported** — mint and discovery track true quality; bad
  tools are buried.
- **sybil_pump** (attack 1, C2): capture ratio (attacker/control cum
  mint) ≈ 1.0 at K=0, rising to ~21× at K=100. The pumped tool's drifted
  head is driven to +0.97 vs +0.29 for an equal-quality honestly-reviewed
  control — **discovery pollution is the cheaper, higher-leverage half of
  the ring**, exactly as attacks.md predicts. Mint capture grows with K
  (self-bootstrapping) rather than being ε-capped per operator.
- **epsilon_faucet** (attack 6): sybil share of the fixed pool grows
  roughly linearly in K (0.05 → 0.17 → 0.34 → 0.51 → 0.67 for
  K=5…200). **Attack 6 confirmed** — ε·K is unbounded in K; the fixed
  pool is drained pro-rata to the number of dust identities. This is the
  most policy-relevant number: it directly prices "how low must ε be, or
  how expensive must identity creation be, to keep the faucet closed."
- **review_nuke** (attack 5): victim/control rank ratio degrades with
  nuker count (0.98 at J=1 → 0.47 at J=30); the victim survives J≤3 and
  sinks at J≥10. **Holds for low-mass (young) tools**, as attacks.md
  predicts.
- **service_clone** (core hypothesis): surviving service revenue decays
  to **exactly the (1−φ) moat rent** (φ=0.3→0.70, 0.7→0.30, 1.0→0.00).
  The free clone pays back its rediscovery cost in every swept regime.
  Fee recycling is **directionally coupled** (clone cumulative mint
  higher with recycling ON) but second-order at a ~1.25% recycle rate —
  it enlarges the pool (and so the clone's absolute payout) without
  changing the clone's pool *share*.

## Surprises (where the code behaved differently than attacks.md framed)

1. **The honest reviewer voice problem amplifies attack 6.** attacks.md
   frames ε·K against "the real economy." But in the sim, honest
   *reviewers* (users who use tools but never author one) never mint, so
   they never accrue reputation, so their voice is stuck at ε forever —
   just like the sybils. Honest voice mass is therefore dominated by the
   handful of *authors*, and K sybils overwhelm it faster than a naive
   "sybils vs whole economy" reading suggests. The defensible voice mass
   is only the authoring subset, not the whole honest population.

2. **The sybil pump's mint advantage is real but its *rank* advantage is
   the bigger effect**, and it appears almost immediately (rank-cross at
   epoch 0 in the summary), because the drift denominator's only inertia
   is the author prior mass of 1.0. A handful of +1 reviews outweighs
   honest *noisy* reviews (which average below +1), so the pumped head
   climbs to ≈ +0.97 while an equal-quality honest tool sits at ≈ +0.29.
   The economics protect *mint* far better than they protect *discovery
   rank* — matching attacks.md but the size of the gap is stark.

3. **service revenue decay lands *exactly* on (1−φ)**, not approximately.
   Because the switch is a clean fraction of exogenous demand and the
   moat is the un-expressible remainder, the surviving-revenue = moat-rent
   identity is exact in the model. The real economics don't distort it —
   the clone's mint comes from the pool, entirely decoupled from the
   service's fee revenue except through the small recycling term.

## Files (v3 — real-close baseline)

- `harness.py` — population model, event fabrication, per-epoch call into
  the real close, reputation ledger → voice weights, discovery ranking.
- `scenarios.py` — the five parameterized scenarios.
- `run_all.py` — runs all five with fixed seeds; writes `results/*.json`
  and `results/summary.md`.
- `results/` — per-scenario JSON (trajectories thinned to every 5th epoch
  for readability) + the summary table.

## v4 "gradient trust" layer (SIM-ONLY — proposed rules, not production)

The user ratified a redesign to be sim-validated BEFORE any production
build. Because these rules do not exist in the codebase, the v4 layer
**deliberately deviates from the import-the-real-functions fidelity
rule**: it RE-IMPLEMENTS the close math (a faithful port of the v3
algorithm) with six labeled changes. v3 stays the comparison baseline;
production code is untouched. Run `python run_all.py` first, then
`python run_v4.py`.

- `v4_rules.py` — the sim-only reimplemented close, with each deviation
  tagged `[v4-A]`…`[v4-E]`:
  - **A** no vet gate (mint from first attested use)
  - **B** vetting → inspection reviews (drift, no usage, no mint)
  - **C** drift weight = rep/supply with NO ε floor (zero-rep → zero drift)
  - **D** mint ε-weight of zero-rep households capped at aggregate β,
    pro-rata
  - **E** cross-epoch credibility sanctions on reviewers who deviate from
    a stabilized score
- `harness_v4.py` — `V4Simulation` driver over `v4_rules`.
- `scenarios_v4.py` — the 5 comparison scenarios + 2 NEW rails
  (`spam_burial` for rule B, `sanction_false_positives` for rule E).
- `run_v4.py` — runs all 7; writes `results/v4/*.json` and the
  side-by-side `results/summary_v4.md`.

### v4 headline verdicts (see `results/summary_v4.md`)

- **baseline** (cold-start, rule C): with seeded incumbent reviewers a
  fresh good tool rates positive in 0 epochs and the best tool still
  ranks #1, BUT quality↔rank correlation falls 0.89 → 0.65 — extremes
  stay correct, mid-quality ranking degrades. Without any rep-holding
  reviewers the drift channel is dead at genesis (needs a bootstrap).
- **sybil_pump**: the *rank* channel dies (rank-gap ≈ 0.03 at all K), the
  coordinator's specific prediction — CONFIRMED. But *mint* capture only
  drops 21× → 13× (not to ~1.0): the attacker monopolizes the whole β
  zero-rep budget.
- **epsilon_faucet**: share drops 0.67 → 0.28 at K=200 — the β cap binds
  per-epoch but **LEAKS** (finding below).
- **review_nuke**: NOT improved (rank-ratio 0.47 → 0.29) — the sanction
  **BACKFIRES** (finding below).
- **spam_burial** (NEW): inspection reviews hold the honest tool at rank
  #1, but only *inspected* spam sinks; un-inspected spam keeps its
  raw-cosine slot. Strictly better than v3 (which had no inspection rail)
  but not a full fix.
- **sanction_false_positives** (NEW): safe (δ, Q) region is **δ≥0.7 with
  Q≥3, or any δ with Q=10** — honest FP dock-rate <2%; tight δ (0.3) dings
  100% of honest reviewers at low Q.
- **service_clone**: unchanged — moat rent = (1−φ) exact, clone still pays
  (no vet-royalty drag).

### Two places v4 is WORSE than expected (both need a further sim-tested fix)

1. **The β mint cap leaks (rule D):** capped faucet mint grants sybils
   reputation, which lifts them out of the zero-rep bucket so they mint
   uncapped next epoch. Fix candidates: faucet mint grants no rep, or cap
   on a rep *floor* not zero-rep.
2. **The credibility sanction backfires (rule E):** when an attacking
   cohort out-weighs honest reviewers it reaches the stabilization
   threshold Q with the wrong-sign score first; the honest minority then
   gets docked for disagreeing with the attacker-defined "stable" score.
   Fix candidates: stabilize on independent diverse-household mass; anchor
   sanctions to a robust/usage-weighted estimate, not the just-moved head.

### Recommended params

**β=0.1, δ=0.7, Q=5** — but β is only meaningful once the rule-D leak is
fixed, and Q cannot be raised to fix the rule-E backfire (a bigger Q just
lengthens the pre-stabilization nukeable window). Full rationale +
cold-start cost assessment in `results/summary_v4.md`.

## v4.1 layer (SIM-ONLY — ratified after the v4 findings)

Two rule revisions + a metric reframe, sim-validated before any build.
Run `python run_v4_1.py` (after `run_all.py` and `run_v4.py`).

- `v4_1_rules.py` — the sim-only reimplemented close with:
  - **D'** (replaces D): drop the β cap. Zero-rep callers' usage mints ATN
    but grants **no reputation** — an author's rep increment is only the
    rep-holder-attributable portion of their mint. A dust ring skims ATN
    forever but never gains voice.
  - **E'** (replaces E): continuous, reversal-aware credibility. Every
    epoch each household's reviews are re-scored against the CURRENT moving
    head — deviation > δ docks, deviation shrinking restores (symmetric).
    No stabilization moment; a mass floor (≥3) stops thin tools from
    sanctioning anyone. A captured score that later reverses retroactively
    docks the capturers.
  - A, B, C carried from v4 unchanged.
- `harness_v4_1.py` — `V41Simulation`; accrues reputation from the
  decoupled `rep_increment`, not total mint.
- `scenarios_v4_1.py` — baseline, epsilon_faucet (D'), sybil_pump (D'+E'),
  `consensus_capture` (reframed nuke: capture-cost curves + reversal path),
  `niche_discovery` (NEW, assert-style surfacing property).
- `run_v4_1.py` — runs all 5; writes `results/v4_1/*.json` and the
  three-way `results/summary_v4_1.md`.

### Metric reframe

Per the user's paradigm decision, the rep-weighted consensus **is** a
tool's quality — no external ground truth, no "honest minority."
Adversarial scenarios report **consensus-capture cost**, not "did truth
win."

### v4.1 headline verdicts (see `results/summary_v4_1.md`)

- **D' kills the voice leak completely** — sybil VOICE share flatlines at
  exactly 0 over 200 epochs (v4 leaked 0.10→0.28). Trade-off: the ATN skim
  is now uncapped (sybil ATN share ~0.72 at K=200) — by design (ATN is
  money, not voice), but the magnitude is a user decision.
- **E' reversal works** — a captured score recovers when rep-backed
  adopters keep using + up-reviewing the tool, and the early capturers are
  retroactively docked to the credibility floor. The v4 nuke backfire is
  fixed. Reversal isn't free: it needs a loyal rep-backed cohort to
  out-weight the adverse mass over time.
- **Continuous-E' honest false-positive dock rate ≈ 0.02%** at δ=0.7.
- **Consensus-capture cost is high (rep-share ≈ 0.94-0.96 to move a score
  0.3-0.6) and nearly mass-independent** — the binding cost is the ongoing
  rep-weight FLOW, not a static mass stock.
- **niche_discovery property holds** — badly-scored tool surfaces top-1
  when alone (multiplicative lift never zeroes) but is buried in a crowd.
- **Recommendation: adopt D' + E', δ=0.7.** Open user question: whether to
  add a rep-independent soft ATN cap to bound the dust-ring ATN skim.

### Where v4.1 is worse than v4

Only the uncapped ATN skim (0.72 vs v4's capped ~0.28). v4.1 dominates v4
on voice protection, the nuke/reversal fix, and honest FP rate.
