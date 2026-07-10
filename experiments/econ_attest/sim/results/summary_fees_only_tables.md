<!-- DATA TABLES auto-rendered by summary_fees_only.py from results/fees_only/*.json. The narrative sections above the '--- machine tables ---' marker are hand-authored; do not let them contradict these tables. -->

# fees-only + REP-from-earnings — machine tables

## S1 honest baseline
- quality↔ATN-earnings corr: **0.722**
- quality↔REP corr: **0.722**
- author income as frac of service GMV: 0.01243
- burn as frac of GMV (should ≈ 0.0125): 0.0125
- dead-start transition clean: **True** (pool 0 during dead: True, >0 after: True)

## S2 usage-flood ring — THE loop
- ANY cell compounds (earnings→REP→weight loop grows): **False**
- max TRANSITION pool capture (first funded epoch, one-shot): **0.6307** (worst cell: genesis/K100/k_houses)
- max LATE pool capture (steady state): **0.0**
- max ring REP-share (of supply): **5.1e-05**

| cell (stage/K/topology) | transition cap | peak cap | late cap | final ring REP-share | compounds |
|---|---|---|---|---|---|
| genesis/K5/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| genesis/K5/k_houses | 0.0805 | 0.0805 | 0.0 | 7e-06 | False |
| genesis/K20/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| genesis/K20/k_houses | 0.257 | 0.257 | 0.0 | 2e-05 | False |
| genesis/K100/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| genesis/K100/k_houses | 0.6307 | 0.6307 | 0.0 | 5.1e-05 | False |
| young/K5/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| young/K5/k_houses | 0.0499 | 0.0499 | 0.0 | 4e-06 | False |
| young/K20/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| young/K20/k_houses | 0.1707 | 0.1707 | 0.0 | 1.4e-05 | False |
| young/K100/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| young/K100/k_houses | 0.5156 | 0.5156 | 0.0 | 4.3e-05 | False |
| mature/K5/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| mature/K5/k_houses | 0.0421 | 0.0421 | 0.0 | 3e-06 | False |
| mature/K20/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| mature/K20/k_houses | 0.1536 | 0.1536 | 0.0 | 1.2e-05 | False |
| mature/K100/one_house | 0.0 | 0.0 | 0.0 | 0.0 | False |
| mature/K100/k_houses | 0.4684 | 0.4684 | 0.0 | 3.7e-05 | False |

### S2 with `service_rep_only=True` (fix candidate: REP only on service revenue)
- ANY cell compounds: **False**
- max transition capture: **0.6307**
- max ring REP-share: **0.0**

## S3 wash trading
- ring fee paid: 600.0 | pool reclaimed: 0.86 | net ATN cost: **599.14**
- strict-loss holds (ring loses ATN net): **True** (reclaim = 0.0014 of fee paid)
- ring REP gained from wash: 23400.86
- wash voice-per-dollar: **39.0574** vs honest voice-per-dollar: **1.0**
- washing buys voice cheaper than honest service: **True**

## S4 whale spender
- whale REP: 0.0 → earns zero REP: **True** (supply share 0.0)
- author REP with/without whale: 892.5 / 148.75 (uplift 743.75)
- provider REP with/without whale: 70200.0 / 11700.0

## S5 retroactivity (same-epoch vs carried dead-period usage)
Transition-epoch (first funded epoch) ring pool capture, by dead-period demand regime:

| dead-period regime | same-epoch | carried | amplification | retro worse? |
|---|---|---|---|---|
| honest users BUSY | 0.3362 | 0.2616 | ×0.778 | False |
| honest users IDLE (only ring pre-farms) | 0.3552 | 0.5679 | ×1.599 | **True** |
- steady-state capture (post-transition, both): ~0.0 (collapses)

## S6 β/S0 relevance under demand-backed REP
- β still load-bearing (uncapped ring capture materially > capped): **False** (max uncapped ring capture 0.0)
- any single S0 robust across all fee-growth curves: **False**

| curve | S0 | uncapped cap | capped cap | uncapped corr | capped corr | corr drop |
|---|---|---|---|---|---|---|
| dead_slow | S0_10 | 0.0 | 0.0 | 0.8996 | 0.0 | 0.8996 |
| dead_slow | S0_50 | 0.0 | 0.0 | 0.8996 | 0.0 | 0.8996 |
| dead_slow | S0_200 | 0.0 | 0.0 | 0.8996 | 0.0 | 0.8996 |
| dead_hot | S0_10 | 0.0 | 0.0 | 0.9883 | 0.0 | 0.9883 |
| dead_hot | S0_50 | 0.0 | 0.0 | 0.9883 | 0.0 | 0.9883 |
| dead_hot | S0_200 | 0.0 | 0.0 | 0.9883 | 0.0 | 0.9883 |
| hot_from_genesis | S0_10 | 0.0 | 0.0 | 0.9862 | 0.0 | 0.9862 |
| hot_from_genesis | S0_50 | 0.0 | 0.0 | 0.9862 | 0.0 | 0.9862 |
| hot_from_genesis | S0_200 | 0.0 | 0.0 | 0.9862 | 0.0 | 0.9862 |

### S0 robustness across curves
| S0 | robust | worst ring capture | worst corr drop |
|---|---|---|---|
| S0_10 | False | 0.0 | 0.9883 |
| S0_50 | False | 0.0 | 0.9883 |
| S0_200 | False | 0.0 | 0.9883 |
