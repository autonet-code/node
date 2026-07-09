# Substrate v3 tool-economy attack sim — verdict summary

All mint runs through the REAL `federated_epoch_close` with `apply_emission_pool` (fixed pool = 100 ATN/epoch + recycled fees), so total minted ATN per epoch == pool (conservation asserted). Mint is therefore zero-sum among authors.

## baseline_honest  (claims C1, C3)

- quality vs cumulative-mint correlation: **0.924**
- quality vs final discovery-rank correlation: **0.894**
- highest-true-quality tool's discovery rank (1=top): **2**
- lowest-true-quality tool's discovery rank: **20**

Verdict: mint share and discovery rank track true quality; the worst tool sinks to the bottom (C1 + C3 SUPPORTED).

## sybil_pump  (attack 1, claim C2)

| K sybils | capture ratio (atk/ctrl cum mint) | rank-cross epoch |
|---|---|---|
| 0 | 0.9882 | 0 |
| 3 | 1.4807 | 0 |
| 10 | 3.1262 | 0 |
| 30 | 7.1119 | 0 |
| 100 | 21.1511 | 0 |

Verdict: capture ratio rises with K (self-bootstrapping ring); each sybil is ε-capped but K of them are not.

## epsilon_faucet  (attack 6)

| K dust identities | final sybil pool share | share growth |
|---|---|---|
| 0 | 0.0 | 0.0 |
| 5 | 0.04782 | 0.00707 |
| 20 | 0.17224 | 0.00708 |
| 50 | 0.33927 | 0.02267 |
| 100 | 0.5115 | 0.03947 |
| 200 | 0.6703 | 0.01334 |

Verdict: sybil pool share grows ~linearly in K and is NOT bounded by a single ε — the fixed pool is drained pro-rata to the count of dust identities (attack 6 CONFIRMED).

## review_nuke  (attack 5)

| J nukers | victim/ctrl rank ratio | victim survived |
|---|---|---|
| 0 | 0.997 | True |
| 1 | 0.9767 | True |
| 3 | 0.9092 | True |
| 10 | 0.7397 | False |
| 30 | 0.4677 | False |

Verdict: a young tool's rank degrades with nuker count; heavy nuking sinks it (attack 5 holds for low-mass tools).

## service_clone  (core hypothesis)

clone pays (cum mint > rediscovery cost)?

- phi=0.3,rcost=0.0: **True**
- phi=0.3,rcost=5.0: **True**
- phi=0.3,rcost=20.0: **True**
- phi=0.3,rcost=50.0: **True**
- phi=0.7,rcost=0.0: **True**
- phi=0.7,rcost=5.0: **True**
- phi=0.7,rcost=20.0: **True**
- phi=0.7,rcost=50.0: **True**
- phi=1.0,rcost=0.0: **True**
- phi=1.0,rcost=5.0: **True**
- phi=1.0,rcost=20.0: **True**
- phi=1.0,rcost=50.0: **True**

surviving service revenue fraction by φ (moat rent ≈ 1−φ):
- φ=0.3: surviving rev frac **0.7** (expected ≈ 0.7)
- φ=0.7: surviving rev frac **0.3** (expected ≈ 0.3)
- φ=1.0: surviving rev frac **0.0** (expected ≈ 0.0)

- fee-recycling payback epoch (recycle ON): **58**
- fee-recycling payback epoch (recycle OFF): **58**
- clone cumulative mint (recycle ON): **3621.399**
- clone cumulative mint (recycle OFF): **3607.87**

Verdict: the free clone captures φ of demand and service revenue decays to exactly the (1−φ) moat rent. Fee recycling IS directionally coupled (clone cum mint is higher with recycling ON) but the effect is second-order at a ~1.25% recycle rate — it lifts the clone's absolute payout via a bigger pool without changing its pool SHARE, too small to move the discrete payback epoch here.
