"""Tests for the vetting-knob sweep (sims/tool_economy/vet_sweep.py).

Run:  python -m pytest sims/tool_economy/test_vet_sweep.py -q

Covers (closed-form wherever the real close admits one):
  1. Determinism — identical seed => identical sweep rows.
  2. Greenlight mechanics through the REAL compute_tool_mint:
     owner-map exclusion, distinct-fleet counting, slip iff
     evading socks >= quorum, late vets earn nothing.
  3. Royalty closed form: per-validator income, author dilution,
     conservation, calendar window ticking.
  4. Bust: royalty zeroed, weight halved, pivotal lockout.
  5. Collusion: royalty is an intra-family transfer (conserved).
"""

from __future__ import annotations

import logging
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

logging.disable(logging.INFO)

from sims.tool_economy import vet_sweep as vs  # noqa: E402

LN2 = math.log1p(1)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_collusion_sweep_deterministic():
    a = vs.sweep_collusion(9, evasions=(0.3,), quorums=(2,),
                           epochs=4, k=4, n_seeds=2)
    b = vs.sweep_collusion(9, evasions=(0.3,), quorums=(2,),
                           epochs=4, k=4, n_seeds=2)
    assert a == b


def test_quorum_sweep_deterministic():
    kw = dict(n_honest=3, n_validators=3, n_colluders=1,
              max_epochs=6, n_seeds=2)
    a = vs.sweep_quorum(5, quorums=(2,), evasions=(1.0,), **kw)
    b = vs.sweep_quorum(5, quorums=(2,), evasions=(1.0,), **kw)
    assert a == b


# ---------------------------------------------------------------------------
# 2. Greenlight mechanics (real compute_tool_mint)
# ---------------------------------------------------------------------------

def _wash_family(quorum, n_evading, n_caught, *, share=0.1, k=4):
    """Author + socks; ``n_evading`` socks unknown to the owner map,
    ``n_caught`` socks sharing the author's registered owner."""
    h = vs.VetHarness(seed=77, quorum=float(quorum), share=share,
                      royalty_epochs=k)
    h.add_owner("washa", "own_fam")
    d = h.register("washa")
    socks = []
    for j in range(n_evading):
        socks.append(f"evade_{j}")
    for j in range(n_caught):
        s = f"caught_{j}"
        h.add_owner(s, "own_fam")
        socks.append(s)
    for s in socks:
        h.vet(d, s)
    return h, d, socks


def test_slip_iff_evading_socks_reach_quorum():
    # 2 evading socks: quorum 2 greenlights, quorum 3 does not.
    h, d, _ = _wash_family(2, n_evading=2, n_caught=0)
    h.close()
    assert h.greenlit(d)

    h, d, _ = _wash_family(3, n_evading=2, n_caught=0)
    h.close()
    assert not h.greenlit(d)


def test_owner_map_excludes_caught_socks_entirely():
    # 5 caught socks cannot pass even quorum 1: the merge pass drops
    # same-registered-owner vets before any weighing.
    h, d, _ = _wash_family(1, n_evading=0, n_caught=5)
    h.close()
    assert not h.greenlit(d)


def test_self_vet_never_counts():
    h = vs.VetHarness(seed=3, quorum=1.0)
    h.add_owner("alice", "own_alice")
    d = h.register("alice")
    h.vet(d, "alice")
    h.close()
    assert not h.greenlit(d)


def test_same_fleet_counts_once():
    # Two vetters sharing one registered owner are ONE fleet: their
    # best weight (1.0) counts once, so quorum 2 is not reached.
    h = vs.VetHarness(seed=4, quorum=2.0)
    h.add_owner("auth", "own_auth")
    d = h.register("auth")
    for v in ("v1", "v2"):
        h.add_owner(v, "own_shared")
        h.vet(d, v)
    h.close()
    assert not h.greenlit(d)


def test_late_vet_earns_nothing():
    h = vs.VetHarness(seed=5, quorum=2.0, share=0.2, royalty_epochs=4)
    h.add_owner("auth", "own_auth")
    d = h.register("auth")
    h.vet(d, "v1")
    h.vet(d, "v2")
    h.close()                      # greenlight, validators frozen
    assert h.greenlit(d)
    h.vet(d, "v3")                 # late vet
    h.attest(d, "user_0")
    h.close()
    assert h.earned.get("v3", 0.0) == 0.0
    assert h.earned.get("v1", 0.0) > 0.0


# ---------------------------------------------------------------------------
# 3. Royalty closed form
# ---------------------------------------------------------------------------

def test_royalty_closed_form_and_conservation():
    share, k, users, sups = 0.2, 3, 5, 3
    h = vs.VetHarness(seed=6, quorum=2.0, share=share, royalty_epochs=k)
    h.add_owner("alice", "own_alice")
    d = h.register("alice", n_supporters=sups)
    for v in ("val_a", "val_b"):
        h.vet(d, v)
    epochs = 2 * k
    for _ in range(epochs):
        for u in range(users):
            h.attest(d, f"user_{u}")
        h.close()

    # mint/epoch = standing * usage_term = (1+sups) * users*log1p(1)
    mint = (1 + sups) * users * LN2
    total = sum(sum(m.values()) for m in h.per_epoch)
    assert math.isclose(total, epochs * mint, rel_tol=1e-9)

    # each validator: share * mint * K / 2; author keeps the rest
    per_val = share * mint * k / 2
    assert math.isclose(h.earned["val_a"], per_val, rel_tol=1e-9)
    assert math.isclose(h.earned["val_b"], per_val, rel_tol=1e-9)
    assert math.isclose(h.earned["alice"],
                        epochs * mint - 2 * per_val, rel_tol=1e-9)
    # author retained fraction over 2K epochs = 1 - share/2
    assert math.isclose(h.earned["alice"] / total, 1 - share / 2,
                        rel_tol=1e-9)


def test_royalty_window_is_calendar_not_usage():
    # Greenlight at close 0, then K closes with NO usage: the window
    # ticks away regardless; usage afterwards pays no royalty.
    share, k = 0.5, 2
    h = vs.VetHarness(seed=8, quorum=2.0, share=share, royalty_epochs=k)
    h.add_owner("auth", "own_auth")
    d = h.register("auth")
    h.vet(d, "v1")
    h.vet(d, "v2")
    h.close()                                    # greenlight, tick 1
    h.close()                                    # tick 2 -> window gone
    h.attest(d, "user_0")
    h.close()
    assert h.earned.get("v1", 0.0) == 0.0
    assert h.earned.get("auth", 0.0) > 0.0       # author still mints


def test_factory_matches_closed_form():
    # Steady-state factory income = K * share * mint_med / 2 per epoch;
    # R* = mint_good / that = n_val * (standing ratio) / (K * share).
    share, k = 0.2, 4
    rows = vs.royalty_factory(11, shares=(share,), ks=(k,))
    r = rows[0]
    mint_med = 1 * 5 * LN2
    mint_good = 4 * 5 * LN2
    assert math.isclose(r["factory_income_per_epoch"],
                        k * share * mint_med / 2, rel_tol=1e-6)
    assert math.isclose(r["good_author_income_per_epoch"], mint_good,
                        rel_tol=1e-6)
    assert math.isclose(r["vets_per_epoch_to_beat_author"],
                        2 * 4 / (k * share), rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 4. Bust: zeroed royalty, halved weight, pivotal lockout
# ---------------------------------------------------------------------------

def test_bust_zeroes_royalty_and_locks_out_pivotal_vetter():
    share, k = 0.2, 6
    h = vs.VetHarness(seed=13, quorum=2.0, share=share, royalty_epochs=k)
    h.add_owner("auth_a", "own_a")
    da = h.register("auth_a")
    h.vet(da, "vx")
    h.vet(da, "nv1")
    h.attest(da, "user_0")
    h.close()                                    # greenlight + royalty
    assert h.greenlit(da)
    paid_before = h.earned.get("vx", 0.0)
    assert paid_before > 0.0

    h.bust(da)                                   # winning exploit CON
    h.attest(da, "user_0")
    h.close()
    # remaining royalty zeroed, bust recorded, weight halved
    assert h.vetting["manifests"][da]["busted"] is True
    assert h.vetting["manifests"][da]["royalty_left"] == 0
    assert h.busts_of("vx") == 1
    assert h.earned.get("vx", 0.0) == paid_before   # no post-bust income

    # vx (weight 0.5) + one clean partner (1.0) = 1.5 < 2: locked out.
    h.add_owner("auth_b", "own_b")
    db = h.register("auth_b")
    h.vet(db, "vx")
    h.vet(db, "nv2")
    h.close()
    assert not h.greenlit(db)


def test_rubber_stamp_wins_when_redundant():
    # THE perverse-incentive flag: with redundant quorum, 1/(1+busts)
    # never touches income, so sheer volume beats care at every p_bust.
    rows = vs.sweep_bust(21, p_busts=(0.2,), epochs=20, n_seeds=2)
    by_regime = {r["regime"]: r for r in rows}
    assert by_regime["redundant"]["rubber_wins"]
    # pivotal regime: busts gate future greenlights, cutting the
    # rubber-stamper's pipeline (weight < 1 can no longer form quorum
    # with a single partner).
    assert by_regime["pivotal"]["rubber_final_weight"] < 1.0


# ---------------------------------------------------------------------------
# 5. Collusion: royalty is an intra-family transfer
# ---------------------------------------------------------------------------

def test_collusion_royalty_is_internal_transfer():
    # Family total is identical with the royalty on or off: the split
    # is conserved (taken from the author), so socks gain nothing the
    # damper had not already leaked.
    kw = dict(m_socks=3, epochs=6, k=4, pump=5)
    a = vs._run_collusion_cell(31, 1.0, 2, 0.1, **kw)
    b = vs._run_collusion_cell(31, 1.0, 2, 0.0, **kw)
    assert a["greenlit"] and b["greenlit"]
    assert math.isclose(a["family_mint"], b["family_mint"], rel_tol=1e-9)
    assert a["sock_royalty"] > 0.0
    assert b["sock_royalty"] == 0.0


def test_collusion_zero_evasion_mints_nothing():
    r = vs._run_collusion_cell(33, 0.0, 2, 0.1, m_socks=4, epochs=4,
                               k=4, pump=5)
    assert not r["greenlit"]
    assert r["family_mint"] == 0.0
