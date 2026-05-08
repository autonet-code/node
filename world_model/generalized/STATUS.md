# Generalized world model — status

A non-LLM, coordinate-space implementation of the formalization in
`docs/FORMALIZATION.md` and `docs/THEORY.md`. Built in one session
on 2026-05-01, replacing the earlier flat-bipartite-graph drift.

## Where we are

**Three demos pass cleanly. One benchmark shows real signal above random.**

  - `demo_two_tendencies.py`: 1D adversarial setting. Two tendencies
    on opposite sides of an axis. Balanced evidence ties them; tilted
    evidence produces a score gap. Confirms basic adversarial
    dynamics.

  - `demo_constraint_satisfaction.py`: hand-built SAT. **10/10 cases
    pass**, including AND, NOT, OR, IMPLICATION, two-hop and three-hop
    chained reasoning, UNSAT detection, mixed 4-variable constraints.
    Single-pass equilibration; no backtracking.

  - `demo_satlib.py`: SATLIB uf20-91 benchmark (20-variable phase-
    transition 3-SAT, 100 published instances). **94.1% average
    clause-satisfaction across 20 instances tested.** Random baseline
    = 87.5%. Best instance = 97.8%. Full-SAT rate = 0% (single-pass
    architectural ceiling).

## Architecture (implemented)

  - **CoordinateFrame** (coordinate_frame.py): a non-LLM
    ReferenceFrame. Two similarity modes:
    - `use_dim_overlap=True` (default): sparse-friendly. Sim = fraction
      of anchor's nonzero dims that obs touches. Stance = sign-
      agreement count between obs and polarity axis on shared dims.
    - `use_dim_overlap=False`: original Gaussian-distance mode for
      dense low-dim cases.

  - **CoordinateProbe**: NoveltyProbe with the formalized terminations
    (INTEGRATED / CONTRADICTS_ROOT / DISRUPTS / ORTHOGONAL /
    MAX_ITERATIONS).

  - **GeneralizedTendency** (tendency.py): owns Tree + Frame + Probe
    + budget. Acts by running the probe on observations and on other
    tendencies' nodes, then translating terminations into signed
    stakes. Includes joint-satisfaction discount: stake from an obs
    is reduced by how much that obs is already satisfied through
    independently-resolved variables (this is what enables chained
    reasoning).

  - **World** (world.py): coalition + observation stream + stake
    graph.

  - **equilibrate / equilibrate_with_growth** (equilibrate.py).

  - **propose_growth** (grow.py): exists but rarely fires; needs
    rework for tied-equilibrium-based contention.

## Open research threads

In rough order of value:

1. **Full-SAT via depth-on-demand growth.** The architecture finds
   94% of clauses in one pass; closing to 100% needs the growth rule
   to fire on tied equilibria and sprout sub-claims that condition
   on neighboring resolved variables. This is the user's
   "complexity is handled by going deeper" property in operation.
   Gate: pass at least one full SATLIB instance.

2. **Performance.** SATLIB instances take ~10s on the unbatched
   Python implementation. The joint-satisfaction loop is the
   bottleneck. NumPy-batched calibration (NOTES.md) would help but
   is deferred until clearly needed.

3. **Other reasoning benchmarks.** Knights & Knaves at n=5-8, Zebra
   puzzle (25-var CSP), bAbI logical-deduction. Test whether the
   architecture generalizes beyond random 3-SAT structure.

4. **Decentralization writeup.** The architecture has properties no
   neural network has: no backprop, local updates, statistics merge
   cleanly, lineage attestation. Once we have a clearly-useful
   capability (full-SAT, even at 20 vars), write the case for
   decentralized inference.

## What this validates against the formalization

The implementation honors all axioms in docs/FORMALIZATION.md:

  - **A1 reference dependence**: same observation has different
    novelty against different tendencies (verified by stance
    asymmetry between v_T and v_F tendencies).
  - **A2 absorption monotonicity**: absorbed observations are
    INTEGRATED at full sim on re-encounter.
  - **A6 stance asymmetry**: CON terminations dominate INTEGRATED
    in driving down root scores.
  - **A7 topical relevance**: dim-overlap gate ensures off-topic
    observations don't fire stance.

The formalization is a domain-agnostic specification; the
coordinate-space implementation makes it concrete without an LLM.

## File map

```
world_model/generalized/
├── __init__.py            -- public API
├── observation.py         -- Observation data class
├── coordinate_frame.py    -- CoordinateClaim, CoordinateFrame, CoordinateProbe
├── tendency.py            -- GeneralizedTendency + joint-satisfaction
├── world.py               -- World coalition
├── equilibrate.py         -- equilibrate, equilibrate_with_growth
├── grow.py                -- propose_growth (currently dormant)
└── STATUS.md              -- this file

demos:
demo_two_tendencies.py            -- 1D adversarial demo
demo_constraint_satisfaction.py   -- hand-built SAT (10/10)
demo_satlib.py                    -- SATLIB uf20-91 (94.1% > 87.5% random)
```

## Where the LLM substitutes live

`docs/FORMALIZATION.md` defined sim and σ (stance) abstractly. The
existing `world_model/novelty/` already had three concrete LLM-backed
implementations (Wikidata, NLI, Hybrid). This module adds:

  - **CoordinateFrame.find_claims**: replaces semantic-similarity
    with dimensional overlap.
  - **CoordinateFrame.detect_stance**: replaces NLI with sign-
    agreement count over shared dims.
  - **CoordinateFrame.contains**: replaces near-paraphrase detection
    with epsilon-distance match.

The probe loop, termination conditions, and the four-component
novelty score are reused unchanged from `world_model.novelty.core`.
