"""Equilibration: run tendencies' actions until budgets stabilize.

Each round:
  1. Each tendency calls .act(world) -> populates last_stakes.
  2. World.apply_stakes() writes them onto nodes (own + cross).
  3. Convergence check: have last_stakes dictionaries stabilized
     (max abs delta < tolerance) compared to previous round?

Until convergence, repeat. Returns the number of rounds.

Note: the staking policy is deterministic given world state, so the
convergence is from the *interaction* between tendencies (each one's
absorption of new observations changes its frame, which changes
everyone else's staking next round).

Also exposes:
  - equilibrate_with_growth: equilibrate -> grow -> equilibrate -> ...
  - equilibrate_continuous: parallel kernel that runs Lindblad
    master-equation evolution between observations, capturing the
    resist-then-yield-decisively dynamics that the discrete update
    approximates lossily.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .grow import propose_growth
from .world import World
from . import lindblad as _lb


def equilibrate(
    world: World,
    max_rounds: int = 20,
    tolerance: float = 1e-3,
    scope: Optional[Iterable[str]] = None,
) -> int:
    """Run rounds of (act, apply_stakes) until intents stabilize.

    Returns the number of rounds executed.

    Scope: if provided, only the tendencies whose ids are in `scope`
    run `act` each round and contribute to the convergence check. Their
    posts are written via the scoped `apply_stakes`. Out-of-scope
    tendencies are untouched (their last_stakes, tree, and per-node
    novelty are preserved as-is). When `scope` is None the behavior is
    identical to the pre-scope kernel: every tendency in the world is
    iterated.
    """
    scope_set: Optional[Set[str]] = set(scope) if scope is not None else None
    active = (
        [t for tid, t in world.tendencies.items() if tid in scope_set]
        if scope_set is not None
        else list(world.tendencies.values())
    )
    prev_intents: Dict[str, Dict[Tuple[str, str], float]] = {
        t.id: dict(t.last_stakes) for t in active
    }
    for round_idx in range(1, max_rounds + 1):
        for tendency in active:
            tendency.act(world)
        world.apply_stakes(scope=scope_set)
        for tendency in active:
            tendency.update_novelty(dt=1.0)
        max_delta = 0.0
        for tendency in active:
            old = prev_intents.get(tendency.id, {})
            new = tendency.last_stakes
            keys = set(old.keys()) | set(new.keys())
            for k in keys:
                d = abs(old.get(k, 0.0) - new.get(k, 0.0))
                if d > max_delta:
                    max_delta = d
        if max_delta < tolerance and round_idx >= 2:
            return round_idx
        prev_intents = {t.id: dict(t.last_stakes) for t in active}
    return max_rounds


def equilibrate_with_growth(
    world: World,
    max_outer: int = 5,
    max_rounds: int = 20,
    tolerance: float = 1e-3,
    contention_threshold: float = 0.15,
    offset: float = 0.5,
    scope: Optional[Iterable[str]] = None,
) -> Tuple[int, int]:
    """Outer loop: equilibrate, grow, equilibrate, ... until no growth.

    Returns (total_rounds, total_new_nodes).
    """
    total_rounds = 0
    total_new = 0
    for outer in range(max_outer):
        rounds = equilibrate(
            world, max_rounds=max_rounds, tolerance=tolerance, scope=scope,
        )
        total_rounds += rounds
        new_nodes = propose_growth(
            world,
            contention_threshold=contention_threshold,
            offset=offset,
        )
        total_new += new_nodes
        if new_nodes == 0:
            break
    return total_rounds, total_new


# ---------------------------------------------------------------------------
# Continuous-time equilibration (Lindblad)
# ---------------------------------------------------------------------------


def _extract_subclaims(
    world: World,
) -> Dict[str, List[Tuple[float, float, Tuple[float, ...]]]]:
    """For each tendency, list (signed_stake, capacity, anchor_coords)
    for each direct child of the root. Stakes are signed by position
    (PRO positive, CON negative); magnitude is the subtree's net_score
    so the full weight of evidence below the sub-claim is captured.
    """
    out: Dict[str, List[Tuple[float, float, Tuple[float, ...]]]] = {}
    for tid, tendency in world.tendencies.items():
        items: List[Tuple[float, float, Tuple[float, ...]]] = []
        root = tendency.tree.root_node
        for child in root.all_children:
            sign = +1.0 if child.position.value == "pro" else -1.0
            stake_magnitude = abs(child.net_score)
            stake = sign * stake_magnitude
            cap = tendency.node_capacity.get(child.id, 0.0)
            claim = tendency._node_to_claim.get(child.id)
            if claim is None or not claim.anchor:
                coords = tendency.anchor
            else:
                coords = claim.anchor
            items.append((stake, cap, tuple(coords)))
        out[tid] = items
    return out


def _alpha_from_score(score: float) -> float:
    """Sigmoid-mapped substrate score -> population alpha in [0, 1]."""
    if score > 30:
        return 1.0
    if score < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def _score_from_alpha(alpha: float) -> float:
    """Inverse: alpha -> logit-mapped substrate score."""
    eps = 1e-6
    a = max(eps, min(1.0 - eps, alpha))
    return math.log(a / (1.0 - a))


def _avg_child_n(world: World, tid: str) -> float:
    """Average n across the root's direct sub-claims. Used to seed
    initial coherence in rho_0. Returns 0 (fully decohered) if no
    sub-claims exist.
    """
    tendency = world.tendencies[tid]
    root = tendency.tree.root_node
    children = root.all_children
    if not children:
        return 0.0
    return sum(c.n for c in children) / len(children)


def _build_initial_rho(
    world: World,
    root_ids: List[str],
    use_novelty: bool = True,
) -> np.ndarray:
    """Joint rho_0 as the tensor product of per-root marginals.

    Each marginal carries:
      alpha_i = sigmoid(net_score_i)              (population in PRO)
      c_i     = avg_n_i * sqrt(alpha_i*(1-alpha_i))  (coherence)

    The c formula keeps rho positive semi-definite by construction:
    |c|^2 <= alpha*(1-alpha) since 0 <= avg_n <= 1.

    use_novelty=False sets c_i = 0 (decohered classical mixture). Use
    that for tests that want to isolate the population dynamics from
    the coherence contribution.
    """
    populations = [_alpha_from_score(world.tendencies[r].tree.score) for r in root_ids]
    if use_novelty:
        coherences = [
            _avg_child_n(world, r) * math.sqrt(max(0.0, a * (1.0 - a)))
            for r, a in zip(root_ids, populations)
        ]
    else:
        coherences = [0.0] * len(root_ids)
    return _lb.joint_rho_from_marginals(populations, coherences)


def _observation_jump_ops(
    world: World,
    root_ids: List[str],
    bandwidth: float,
    base_gamma: float,
) -> List[Tuple[np.ndarray, float]]:
    """Build Lindblad jump operators for the world's pending observations.

    For each (observation, root) pair:
      - distance = ||obs.coords - tendency.anchor||
      - locality_kernel = gauss(distance; bandwidth)
      - polarity_dot = obs.coords . tendency.polarity_axis
      - L = single-qubit RAISE_TO_PRO if dot > 0 else LOWER_TO_CON
      - gamma = base_gamma * locality_kernel
    """
    N = len(root_ids)
    ops: List[Tuple[np.ndarray, float]] = []
    for obs in world.observations.values():
        for idx, tid in enumerate(root_ids):
            tendency = world.tendencies[tid]
            anchor = tendency.anchor
            polarity = tendency.polarity_axis
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs.coords, anchor)))
            kernel = math.exp(-d * d / (2.0 * bandwidth * bandwidth + 1e-12))
            gamma = base_gamma * kernel
            if gamma < 1e-4:
                continue
            dot = sum(o * p for o, p in zip(obs.coords, polarity))
            if dot > 0:
                L = _lb.single_qubit_op(N, idx, _lb.RAISE_TO_PRO)
            elif dot < 0:
                L = _lb.single_qubit_op(N, idx, _lb.LOWER_TO_CON)
            else:
                continue
            ops.append((L, gamma))
    return ops


def equilibrate_continuous_exploration(
    world: World,
    *,
    mu: float = 2.5,
    t_total: float = 8.0,
    dt: float = 1e-2,
    bandwidth: float = 0.5,
    kappa: float = 1.0,
    lam: float = 1.0,
    base_gamma: float = 0.5,
    use_novelty_in_rho0: bool = True,
    write_back: bool = True,
    cross_link_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Slow-path Lindblad pass tuned for cross-domain exploration.

    Wraps `equilibrate_continuous` with boosted coupling (mu=2.5 vs
    default 1.0) and longer evolution time (t_total=8.0 vs 1.0). Two
    effects:

      - Higher mu amplifies J_ab coupling between roots, so weak
        cross-domain correlations that the discrete kernel's locality
        gate skips have a chance to show up as entanglement.
      - Longer t_total gives zz-coupling time to accumulate effect on
        marginal populations.

    Bandwidth is NOT widened (default 0.5). Boosting bandwidth would
    also rescale the observation jump operators, which is the wrong
    knob — we want to dial up coupling between root degrees of
    freedom, not change how observations apply.

    `cross_link_threshold`: when |J_ab| exceeds this after evolution,
    the highest-coupled sub-claim pair (i in root_a, j in root_b)
    gets a `_lindblad_cross_link` stake posted on each. Threading
    that into the discrete kernel is Phase 3.2.
    """
    return equilibrate_continuous(
        world,
        t_total=t_total,
        dt=dt,
        bandwidth=bandwidth,
        kappa=kappa,
        lam=lam,
        mu=mu,
        base_gamma=base_gamma,
        use_novelty_in_rho0=use_novelty_in_rho0,
        write_back=write_back,
        emit_cross_link_stakes=True,
        cross_link_threshold=cross_link_threshold,
    )


def equilibrate_continuous(
    world: World,
    t_total: float = 1.0,
    dt: float = 1e-3,
    bandwidth: float = 0.5,
    kappa: float = 1.0,
    lam: float = 1.0,
    mu: float = 1.0,
    base_gamma: float = 0.5,
    use_novelty_in_rho0: bool = True,
    write_back: bool = True,
    emit_cross_link_stakes: bool = False,
    cross_link_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Continuous-time equilibration via Lindblad master-equation evolution.

    Parallel to `equilibrate`: instead of running discrete (act,
    apply_stakes) rounds, this kernel constructs a Hamiltonian H from
    the current sub-claim configuration, builds a joint rho_0 from
    per-root (population, coherence), and integrates the master
    equation forward by t_total.

    The result: per-root populations alpha_i are read out of the final
    rho and (if write_back) written back to the substrate as net_score
    via the logit map.

    The sub-claim graph topology, the ledger, and capacity state are
    NOT modified by this kernel. It only updates per-root scores. Use
    it BETWEEN observation events; let `tendency.act + apply_stakes`
    run the graph-structure updates (sprouts, prunes, capacity).

    Args:
        world: substrate world.
        t_total: simulated time interval to evolve.
        dt: integration step.
        bandwidth: locality kernel bandwidth (also the substrate's).
        kappa, lam, mu: Hamiltonian-construction constants for omega,
                        zeta, J respectively. Each parameter lives in
                        a bounded range scaled by these.
        base_gamma: base rate for observation jump operators.
        use_novelty_in_rho0: if True, seed off-diagonal coherence from
                             avg child n. If False, start from a
                             decohered classical mixture.
        write_back: if True, mutate world's net_scores to match the
                    Lindblad final state.

    Returns:
        dict with diagnostic info: root_ids, omegas, zetas, Js,
        initial_alpha, final_alpha, n_observations, n_jump_ops, dim.
    """
    subs = _extract_subclaims(world)
    root_ids, omegas, zetas, Js = _lb.hamiltonian_params_from_subclaims(
        subs, bandwidth=bandwidth, kappa=kappa, lam=lam, mu=mu,
    )
    H = _lb.assemble_hamiltonian(omegas, zetas, Js)
    jump_ops = _observation_jump_ops(world, root_ids, bandwidth, base_gamma)
    rho_0 = _build_initial_rho(world, root_ids, use_novelty=use_novelty_in_rho0)

    initial_alpha = [_alpha_from_score(world.tendencies[r].tree.score)
                     for r in root_ids]
    rho_t = _lb.evolve(rho_0, H, jump_ops, t_total=t_total, dt=dt)

    N = len(root_ids)
    final_alpha: List[float] = []
    for idx in range(N):
        rho_i = _lb.marginal(rho_t, N=N, keep=idx)
        final_alpha.append(_lb.population_pro(rho_i))

    # When running in exploration mode (emit_cross_link_stakes=True), the
    # root-score writeback is suppressed by default. The Lindblad kernel
    # produces equilibrium populations bounded in [0, 1] (logit-mapped to
    # ~[-30, +30] root scores); the discrete kernel accumulates posts
    # without that bound. Writing back from the bounded Lindblad value
    # would overwrite ~99% of the discrete kernel's accumulated magnitude
    # every time the slow pass fires. In exploration mode we want the
    # Lindblad math to surface cross-domain coupling via stakes, not to
    # re-decide the discrete kernel's verdict.
    #
    # In non-exploration mode (the original equilibrate_continuous call
    # path for tests that compare classical-vs-Lindblad trajectories on
    # a fresh world), write_back is honored as before.
    do_root_writeback = write_back and not emit_cross_link_stakes
    if do_root_writeback:
        for idx, tid in enumerate(root_ids):
            tendency = world.tendencies[tid]
            target_score = _score_from_alpha(final_alpha[idx])
            current = tendency.tree.root_node.net_score
            delta = target_score - current
            if abs(delta) > 1e-6:
                tendency.tree.root_node.add_stake(
                    agent_id="_lindblad_continuous", weight=delta,
                )

    cross_links: List[Dict[str, Any]] = []
    if emit_cross_link_stakes:
        cross_links = _emit_cross_link_stakes(
            world, root_ids, subs, Js, bandwidth, cross_link_threshold,
        )

    return {
        "root_ids": root_ids,
        "omegas": omegas,
        "zetas": zetas,
        "Js": {f"{a}-{b}": v for (a, b), v in Js.items()},
        "n_observations": len(world.observations),
        "n_jump_ops": len(jump_ops),
        "dim": 2 ** N,
        "initial_alpha": initial_alpha,
        "final_alpha": final_alpha,
        "t_total": t_total,
        "dt": dt,
        "cross_links": cross_links,
    }


def _emit_cross_link_stakes(
    world: World,
    root_ids: List[str],
    subs: Dict[str, List[Tuple[float, float, Tuple[float, ...]]]],
    Js: Dict[Tuple[int, int], float],
    bandwidth: float,
    threshold: float,
) -> List[Dict[str, Any]]:
    """For each (root_a, root_b) pair with |J_ab| >= threshold, find
    the dominant sub-claim pair (i, j) and post a unit-weight
    `_lindblad_cross_link` stake on each. Returns descriptors of every
    link emitted, for callers/tests to inspect.

    "Dominant" = the pair that contributed most to the raw J_ab sum
    in `hamiltonian_params_from_subclaims`: signed-stake product
    weighted by capacities and the Gaussian locality kernel.
    """
    emitted: List[Dict[str, Any]] = []
    for (a_idx, b_idx), J in Js.items():
        if abs(J) < threshold:
            continue
        tid_a, tid_b = root_ids[a_idx], root_ids[b_idx]
        sc_a, sc_b = subs[tid_a], subs[tid_b]
        if not sc_a or not sc_b:
            continue

        children_a = world.tendencies[tid_a].tree.root_node.all_children
        children_b = world.tendencies[tid_b].tree.root_node.all_children
        if len(children_a) != len(sc_a) or len(children_b) != len(sc_b):
            # Tree mutated between subclaim extraction and now; skip.
            continue

        best_score = 0.0
        best_pair: Optional[Tuple[int, int]] = None
        for i, (s_i, c_i, coords_i) in enumerate(sc_a):
            for j, (s_j, c_j, coords_j) in enumerate(sc_b):
                d = math.sqrt(sum(
                    (x - y) ** 2 for x, y in zip(coords_i, coords_j)
                ))
                kern = math.exp(-d * d / (2.0 * bandwidth * bandwidth + 1e-12))
                contrib = abs(s_i * s_j) * c_i * c_j * kern
                if contrib > best_score:
                    best_score = contrib
                    best_pair = (i, j)
        if best_pair is None or best_score <= 0.0:
            continue

        node_a = children_a[best_pair[0]]
        node_b = children_b[best_pair[1]]
        node_a.add_stake(agent_id="_lindblad_cross_link", weight=1.0)
        node_b.add_stake(agent_id="_lindblad_cross_link", weight=1.0)
        emitted.append({
            "root_a": tid_a, "root_b": tid_b, "J": J,
            "node_a": node_a.id, "node_b": node_b.id,
            "contribution": best_score,
        })
    return emitted
