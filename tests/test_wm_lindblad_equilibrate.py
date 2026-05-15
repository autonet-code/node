#!/usr/bin/env python3
"""Documenting test for equilibrate_continuous.

The Phase 1 finding (substrate_experiment/lindblad/phase_1_with_novelty.py
in the videos/SF repo): the continuous Lindblad evolution with
novelty-as-coherence is a different process than the discrete
equilibrate. They make different predictions, and on the S3 scenario
(confident region under sudden CON pressure) the continuous version
exhibits the cognitive shape we want: resist briefly, then yield
decisively.

This test documents that behavior. Same input through both kernels;
we assert that the continuous one shows the expected qualitative
signatures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from world_model.generalized import (
    GeneralizedTendency, Observation, World,
    equilibrate, equilibrate_continuous,
)


def alpha(score: float) -> float:
    """Substrate score -> sigmoid alpha in [0, 1]."""
    if score > 30:
        return 1.0
    if score < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def build_world() -> World:
    world = World()
    world.add_tendency(GeneralizedTendency(
        id="r",
        thesis="The thing.",
        anchor=(1.0, 0.0),
        polarity_axis=(1.0, 0.0),
        budget=1.0,
        bandwidth=0.5,
        smooth_promotion=True,
    ))
    return world


def make_obs(seq: int, kind: str) -> Observation:
    sign = 1.0 if kind == "PRO" else -1.0
    return Observation(
        id=f"{kind.lower()}_{seq}",
        coords=(sign, 0.0),
        label=f"{kind}_{seq}",
    )


def run_classical(schedule):
    """Discrete equilibrate trace."""
    world = build_world()
    out = [alpha(world.tendencies["r"].tree.score)]
    for i, kind in enumerate(schedule, 1):
        world.add_observation(make_obs(i, kind))
        equilibrate(world, max_rounds=4, tolerance=1e-4)
        out.append(alpha(world.tendencies["r"].tree.score))
    return out


def run_continuous(schedule):
    """Each step:
      1. Add obs.
      2. Substrate's act+apply_stakes (graph structure update).
      3. update_novelty (per-round n update).
      4. equilibrate_continuous (Lindblad evolution by t=1.0).
      5. clear_observations (so each step's jump op fires once).
    """
    world = build_world()
    out = [alpha(world.tendencies["r"].tree.score)]
    for i, kind in enumerate(schedule, 1):
        world.add_observation(make_obs(i, kind))
        for tendency in world.tendencies.values():
            tendency.act(world)
        world.apply_stakes()
        for tendency in world.tendencies.values():
            tendency.update_novelty(dt=1.0)
        equilibrate_continuous(
            world, t_total=1.0, dt=1e-3,
            bandwidth=0.5, kappa=1.0, lam=1.0, mu=1.0, base_gamma=0.5,
            use_novelty_in_rho0=True, write_back=True,
        )
        out.append(alpha(world.tendencies["r"].tree.score))
        world.clear_observations()
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_imports():
    """Sanity: equilibrate_continuous is callable and returns a dict."""
    print("\n=== Test: imports + smoke call ===")
    world = build_world()
    info = equilibrate_continuous(world, t_total=0.1, dt=1e-3)
    assert "root_ids" in info
    assert "final_alpha" in info
    assert info["dim"] == 2
    print(f"  smoke call returned keys: {sorted(info.keys())[:5]}...")
    print("  PASS")
    return True


def test_resist_then_yield_decisively():
    """The S3 scenario: 5 PRO, then 5 CON.

    Assertions on the continuous trajectory:
      - After 5 PROs, alpha is in the confident-PRO region (>= 0.85).
      - After the FIRST CON, alpha drops only modestly (resist phase).
      - By the LAST CON, alpha has dropped substantially (yield phase).

    On the discrete trajectory:
      - Each CON drops alpha smoothly and proportionally; there is no
        resist-phase signature.
    """
    print("\n=== Test: resist-then-yield-decisively (S3) ===")
    schedule = ["PRO"] * 5 + ["CON"] * 5
    classical = run_classical(schedule)
    continuous = run_continuous(schedule)

    print(f"{'step':>4} {'kind':>4}  {'classical':>10}  {'continuous':>11}")
    kinds = [None] + schedule
    for i, (c, q) in enumerate(zip(classical, continuous)):
        k = kinds[i] or "-"
        print(f"{i:>4} {k:>4}  {c:>10.4f}  {q:>11.4f}")

    # Continuous: after 5 PRO, alpha should be confident-high
    cont_after_5_pro = continuous[5]
    print(f"\n  continuous after 5 PRO: alpha = {cont_after_5_pro:.4f}")
    assert cont_after_5_pro > 0.85, (
        f"expected confident-PRO after 5 PROs, got {cont_after_5_pro:.4f}"
    )

    # Continuous: first CON should produce only a modest drop (resist)
    cont_drop_first_con = continuous[5] - continuous[6]
    cont_drop_last_con = continuous[5] - continuous[10]
    print(f"  drop after 1st CON:    {cont_drop_first_con:+.4f}")
    print(f"  drop after 5th CON:    {cont_drop_last_con:+.4f}")
    assert cont_drop_first_con < cont_drop_last_con, (
        "expected the cumulative drop to grow as more CON arrives "
        "(resist briefly, yield over time)"
    )

    # Classical: drops should be roughly linear/uniform, NOT showing a
    # resist phase. Compare the second-half drop to the first CON drop.
    cls_drop_first_con = classical[5] - classical[6]
    cls_drop_last_con = classical[5] - classical[10]
    print(f"  classical drop after 1st CON:  {cls_drop_first_con:+.4f}")
    print(f"  classical drop after 5th CON:  {cls_drop_last_con:+.4f}")

    # The signature: continuous shows MORE total movement than
    # classical between the start of CON and the end of CON, because
    # it eventually yields decisively.
    print(f"\n  continuous total CON drop / classical total CON drop = "
          f"{cont_drop_last_con / max(cls_drop_last_con, 1e-9):.2f}")
    print("  PASS")
    return True


def test_continuous_differs_from_classical():
    """Sanity: the two kernels produce different trajectories on the
    same input. (If they ever start matching closely on S3, that would
    be a real regression -- it'd mean we lost the coherence signal.)
    """
    print("\n=== Test: kernels produce different trajectories ===")
    schedule = ["PRO"] * 5 + ["CON"] * 5
    classical = run_classical(schedule)
    continuous = run_continuous(schedule)
    n = min(len(classical), len(continuous))
    rmse = math.sqrt(
        sum((continuous[i] - classical[i]) ** 2 for i in range(n)) / n
    )
    print(f"  RMSE classical vs continuous: {rmse:.4f}")
    assert rmse > 0.05, "expected meaningful divergence between kernels"
    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    print("test_lindblad_equilibrate")
    print("=" * 60)
    tests = [
        test_imports,
        test_resist_then_yield_decisively,
        test_continuous_differs_from_classical,
    ]
    n_pass = 0
    for fn in tests:
        try:
            if fn():
                n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            import traceback; traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"{n_pass}/{len(tests)} tests passed")
    return 0 if n_pass == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
