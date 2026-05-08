"""Lindblad master-equation kernel for continuous-time score evolution.

The substrate's discrete `equilibrate` is one specific approximation
to a deeper continuous-time process. This module provides the
continuous version: a Lindblad master equation that evolves a joint
density matrix over the substrate's roots, capturing dynamics the
discrete update misses (resist-then-yield-decisively under sudden
contradiction, tilted steady states, coherent oscillation under high
tension).

Mathematical scaffold
---------------------

For a substrate with N root tendencies, the joint state is a 2^N
density matrix rho. Per-root sub-claim configurations determine three
Hamiltonian parameter sets:

    omega_a = bias toward PRO/CON for root a
              (direction of net stake among root a's sub-claims, in
              [-1, 1] after normalization)

    zeta_a  = transverse-field tension on root a
              (how much PRO and CON sub-claims simultaneously pull, in
              [0, 1])

    J_ab    = ferromagnetic coupling between roots a and b
              (Gaussian-kernel-weighted overlap of nearby sub-claims)

Combined Hamiltonian:

    H = sum_a (-omega_a sigma_z^a + zeta_a sigma_x^a)
      + sum_{a<b} (-J_ab) sigma_z^a tensor sigma_z^b

Observations apply Lindblad jump operators:

    L_PRO_a = sqrt(gamma) * |PRO_a><CON_a|    (drives population to PRO)
    L_CON_a = sqrt(gamma) * |CON_a><PRO_a|    (drives population to CON)

with gamma the per-observation rate, modulated by the locality kernel
between the obs coords and the tendency's anchor, and (optionally)
by the tendency's confidence weight (heavier roots resist).

Master equation:

    drho/dt = -i [H, rho] + sum_j gamma_j * D[L_j](rho)

where D[L](rho) = L rho L^dag - 0.5 (L^dag L rho + rho L^dag L).

The kernel here is engine-internal: pure numpy, no substrate-specific
imports. Callers (equilibrate_continuous in this same package) build
the H and L_j from substrate state.

See also
--------

  equilibrate.equilibrate_continuous: the engine-side public entry
  point that runs Lindblad evolution on a World between observations.

  Phase 1 finding (substrate_experiment/lindblad/phase_1_with_novelty.py):
  the continuous evolution and the discrete equilibrate are genuinely
  different processes; the continuous-with-coherence form is the
  cognitive shape we want.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Pauli operators, in the {|PRO> = |0>, |CON> = |1>} basis
# ---------------------------------------------------------------------------

I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

# Transition operators (amplitude damping)
RAISE_TO_PRO = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
LOWER_TO_CON = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# Multi-qubit operator embedding
# ---------------------------------------------------------------------------


def kron_chain(*matrices: np.ndarray) -> np.ndarray:
    """Iterated Kronecker product of 2x2 matrices."""
    out = matrices[0]
    for m in matrices[1:]:
        out = np.kron(out, m)
    return out


def single_qubit_op(N: int, idx: int, op: np.ndarray) -> np.ndarray:
    """Embed a 2x2 op at qubit idx in an N-qubit register."""
    factors = [I2] * N
    factors[idx] = op
    return kron_chain(*factors)


def two_qubit_zz(N: int, a: int, b: int) -> np.ndarray:
    """sigma_z^a tensor sigma_z^b in an N-qubit register."""
    factors = [I2] * N
    factors[a] = SIGMA_Z
    factors[b] = SIGMA_Z
    return kron_chain(*factors)


# ---------------------------------------------------------------------------
# Master equation pieces
# ---------------------------------------------------------------------------


def _commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def _anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def dissipator(L: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """The Lindblad dissipator D[L](rho) = L rho L^dag - 0.5(L^dag L rho + rho L^dag L)."""
    Ldag = L.conj().T
    LdagL = Ldag @ L
    return L @ rho @ Ldag - 0.5 * _anticommutator(LdagL, rho)


def lindblad_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    """Right-hand side of the master equation.

    jump_ops is a list of (L_j, gamma_j). The Lindblad rate appears as
    a multiplier on the dissipator (equivalent to absorbing sqrt(gamma)
    into L; we keep them separate for clarity).
    """
    drho = -1j * _commutator(H, rho)
    for L, gamma in jump_ops:
        if gamma == 0.0:
            continue
        drho = drho + gamma * dissipator(L, rho)
    return drho


# ---------------------------------------------------------------------------
# Time evolution (RK4)
# ---------------------------------------------------------------------------


def evolve(
    rho_0: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[tuple[np.ndarray, float]],
    t_total: float,
    dt: float = 1e-3,
) -> np.ndarray:
    """Integrate the master equation from t=0 to t=t_total via RK4.

    Numerical hygiene: re-Hermitize and renormalize trace at every
    step to suppress accumulating floating-point error.
    """
    rho = rho_0.astype(complex).copy()
    n_steps = int(round(t_total / dt))
    for _ in range(n_steps):
        k1 = lindblad_rhs(rho, H, jump_ops)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, jump_ops)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, jump_ops)
        k4 = lindblad_rhs(rho + dt * k3, H, jump_ops)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Re-Hermitize and renormalize.
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if tr > 0:
            rho = rho / tr
    return rho


# ---------------------------------------------------------------------------
# Marginal extraction (per-root readout)
# ---------------------------------------------------------------------------


def marginal(rho: np.ndarray, N: int, keep: int) -> np.ndarray:
    """Partial-trace out all qubits except `keep`. Returns 2x2.

    Used to extract per-root populations (alpha = <PRO|rho_keep|PRO>)
    from the joint state.
    """
    if rho.shape != (2 ** N, 2 ** N):
        raise ValueError(f"rho must be {2**N}x{2**N}, got {rho.shape}")
    shape = (2,) * N + (2,) * N
    rho_t = rho.reshape(shape)
    out = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            idx_row = [slice(None)] * N
            idx_col = [slice(None)] * N
            idx_row[keep] = i
            idx_col[keep] = j
            sub = rho_t[tuple(idx_row + idx_col)]
            sub_flat = sub.reshape((2 ** (N - 1), 2 ** (N - 1)))
            out[i, j] = np.trace(sub_flat)
    return out


def population_pro(rho: np.ndarray) -> float:
    """<PRO|rho|PRO> for a 2x2 marginal."""
    return rho[0, 0].real


# ---------------------------------------------------------------------------
# Sub-claim view (engine-side; substrate-aware but not substrate-specific)
# ---------------------------------------------------------------------------


def _direction(stakes_caps: Sequence[tuple[float, float]]) -> float:
    """omega_dir in [-1, 1]: signed_weight / total_weight, both
    capacity-weighted. Returns 0 for empty input.
    """
    EPS = 1e-9
    total = sum(abs(s) * c for s, c in stakes_caps)
    signed = sum(s * c for s, c in stakes_caps)
    return signed / (total + EPS)


def _normalized_tension(stakes_caps: Sequence[tuple[float, float]]) -> float:
    """zeta_norm in [0, 1]: 2*sqrt(p*c) where p, c are PRO and CON
    fractions of total weight. 0 when only one pole present; 1 when
    PRO and CON each carry exactly half.
    """
    EPS = 1e-9
    total = sum(abs(s) * c for s, c in stakes_caps)
    if total < EPS:
        return 0.0
    P = sum(max(s, 0.0) * c for s, c in stakes_caps)
    C = sum(max(-s, 0.0) * c for s, c in stakes_caps)
    p, q = P / total, C / total
    return 2.0 * math.sqrt(p * q)


def _gaussian_kernel(d: float, bandwidth: float) -> float:
    return math.exp(-d * d / (2.0 * bandwidth * bandwidth + 1e-12))


def hamiltonian_params_from_subclaims(
    subclaims_per_root: dict[str, list[tuple[float, float, tuple[float, ...]]]],
    bandwidth: float = 0.5,
    kappa: float = 1.0,
    lam: float = 1.0,
    mu: float = 1.0,
) -> tuple[list[str], list[float], list[float], dict[tuple[int, int], float]]:
    """Compute (root_ids, omegas, zetas, Js) from per-root sub-claim
    configs. Each sub-claim is (signed_stake, capacity, coords).

    Returns:
      root_ids: stable ordering for the qubit register
      omegas:   list of omega_a (one per root, in [-kappa, kappa])
      zetas:    list of zeta_a  (one per root, in [0, lam])
      Js:       dict (i,j) -> J_ab (i<j); only non-zero entries.
    """
    EPS = 1e-9
    root_ids = sorted(subclaims_per_root.keys())
    omegas: list[float] = []
    zetas: list[float] = []
    weights: list[float] = []
    for r in root_ids:
        sc = subclaims_per_root[r]
        stakes_caps = [(s, c) for (s, c, _coords) in sc]
        omegas.append(kappa * _direction(stakes_caps))
        zetas.append(lam * _normalized_tension(stakes_caps))
        weights.append(sum(abs(s) * c for s, c in stakes_caps))
    Js: dict[tuple[int, int], float] = {}
    for i in range(len(root_ids)):
        for j in range(i + 1, len(root_ids)):
            sc_i = subclaims_per_root[root_ids[i]]
            sc_j = subclaims_per_root[root_ids[j]]
            W_i, W_j = weights[i], weights[j]
            if W_i < EPS or W_j < EPS:
                continue
            raw = 0.0
            for s_i, c_i, coords_i in sc_i:
                for s_j, c_j, coords_j in sc_j:
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(coords_i, coords_j)))
                    raw += (math.copysign(1, s_i) * math.copysign(1, s_j)
                            * abs(s_i * s_j) * c_i * c_j
                            * _gaussian_kernel(d, bandwidth))
            J = mu * raw / math.sqrt(W_i * W_j * (W_i + W_j))
            if abs(J) > 1e-9:
                Js[(i, j)] = J
    return root_ids, omegas, zetas, Js


def assemble_hamiltonian(
    omegas: Sequence[float],
    zetas: Sequence[float],
    Js: dict[tuple[int, int], float],
) -> np.ndarray:
    """H = sum_a (-omega_a sigma_z^a + zeta_a sigma_x^a)
        + sum_{a<b} (-J_ab) sigma_z^a tensor sigma_z^b
    """
    N = len(omegas)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for a in range(N):
        H = H + (-omegas[a]) * single_qubit_op(N, a, SIGMA_Z)
        H = H + zetas[a] * single_qubit_op(N, a, SIGMA_X)
    for (a, b), J in Js.items():
        if a > b:
            a, b = b, a
        H = H + (-J) * two_qubit_zz(N, a, b)
    return H


# ---------------------------------------------------------------------------
# Density-matrix construction from per-root (population, coherence)
# ---------------------------------------------------------------------------


def joint_rho_from_marginals(
    populations: Sequence[float],
    coherences: Sequence[float],
) -> np.ndarray:
    """Build a joint density matrix as the tensor product of per-root
    2x2 marginals with given population alpha and coherence c. Each
    c is real-valued (no phase tracking needed for the substrate's
    use case).

    Each marginal:
      rho_i = [[alpha,    c],
               [c,    1-alpha]]
    must satisfy |c|^2 <= alpha*(1-alpha) for positivity. Caller is
    responsible; typically c = n * sqrt(alpha*(1-alpha)) keeps it
    legal automatically.
    """
    if len(populations) != len(coherences):
        raise ValueError("populations and coherences must have same length")
    rho = None
    for a, c in zip(populations, coherences):
        rho_i = np.array([[a, c], [c, 1 - a]], dtype=complex)
        rho = rho_i if rho is None else np.kron(rho, rho_i)
    return rho
