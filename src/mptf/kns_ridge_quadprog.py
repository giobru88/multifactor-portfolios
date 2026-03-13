# -*- coding: utf-8 -*-
"""
kns_ridge_quadprog.py

KNS ridge-penalized QP solver.

Port of MATLAB kns_ridge_quadprog2:
    H = 2*(Sigma*Sigma + gamma*Sigma);
    f = -2*(Sigma*mu');
    b = quadprog(H, f, A, bineq, Aeq, beq, lb, ub, [], opts);

Solver: quadprog (Goldfarb-Idnani dual active-set) — the same
algorithm used by MATLAB's quadprog for small/medium problems.

Replaces the previous CVXOPT implementation which produced suboptimal
solutions when gamma is very small (near-singular H), because CVXOPT's
interior-point barrier method lost precision at small objective scales.

Install: pip install quadprog

Author: Giovanni Bruno (Python port)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


def _is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return True
    if isinstance(x, np.ndarray) and x.size == 0:
        return True
    return False


def _as_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Matrix input must be 2D.")
    return x


def kns_ridge_quadprog(
    mu,
    Sigma,
    gamma: float,
    A=None,
    bineq=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    qpOpts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve the KNS ridge-penalized QP:

        min  0.5 * b' H b + f' b
        s.t. A b <= bineq
             Aeq b = beq
             lb <= b <= ub

    where H = 2*(Sigma @ Sigma + gamma * Sigma) and f = -2*(Sigma @ mu).

    Uses the Goldfarb-Idnani dual active-set algorithm (quadprog package),
    which is the same algorithm MATLAB's quadprog uses for small/medium QPs.

    Parameters
    ----------
    mu : array_like, shape (N,)
    Sigma : array_like, shape (N, N)
    gamma : float — ridge penalty
    A, bineq : inequality constraints A @ b <= bineq
    Aeq, beq : equality constraints Aeq @ b = beq
    lb, ub : bounds on b
    qpOpts : dict — unused (kept for API compatibility)

    Returns
    -------
    b : (N,) ndarray — optimal SDF coefficients
    qpOut : dict — solver diagnostics
    """
    import quadprog as qp_module

    Sigma = _as_2d(Sigma)
    mu = _as_1d(mu)

    n = Sigma.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be NxN. Got {Sigma.shape}")
    if mu.shape[0] != n:
        raise ValueError(f"mu must have length N={n}. Got {mu.shape[0]}")
    if not np.isfinite(mu).all():
        raise ValueError("mu contains NaN/Inf.")
    if not np.isfinite(Sigma).all():
        raise ValueError("Sigma contains NaN/Inf.")

    # ── QP matrices (MATLAB convention) ───────────────────────────
    # MATLAB: min 0.5 x'Hx + f'x  s.t. Ax <= bineq, Aeq x = beq
    H = 2.0 * (Sigma @ Sigma + gamma * Sigma)
    H = 0.5 * (H + H.T)  # enforce symmetry
    f = -2.0 * (Sigma @ mu)

    # ── Convert to quadprog convention ────────────────────────────
    # quadprog: min 0.5 x'Gx - a'x  s.t. C'x >= b
    #   G = H,  a = -f
    #   A x <= bineq  →  -A' x >= -bineq
    #   Aeq x = beq   →  first meq rows (equalities)
    #   lb <= x        →  I' x >= lb
    #   x <= ub        →  -I' x >= -ub
    G = H
    a_vec = -f

    # Build constraint matrix C (n × m) and vector b (m,)
    # quadprog expects C'x >= b, with first meq rows being equalities
    C_blocks = []
    b_blocks = []
    meq = 0

    # Equality constraints first (quadprog requires equalities at the front)
    A_eq = None if _is_empty(Aeq) else _as_2d(Aeq)
    b_eq = None if _is_empty(beq) else _as_1d(beq)
    if A_eq is not None:
        if b_eq is None:
            raise ValueError("Aeq provided but beq missing.")
        C_blocks.append(A_eq.T)         # (n, meq)
        b_blocks.append(b_eq)
        meq = A_eq.shape[0]

    # Inequality constraints: A x <= bineq  →  -A' x >= -bineq
    A_ineq = None if _is_empty(A) else _as_2d(A)
    b_ineq = None if _is_empty(bineq) else _as_1d(bineq)
    if A_ineq is not None:
        if b_ineq is None:
            raise ValueError("A provided but bineq missing.")
        C_blocks.append(-A_ineq.T)      # (n, m_ineq)
        b_blocks.append(-b_ineq)

    # Lower bounds: x >= lb  →  I' x >= lb
    lbv = None if _is_empty(lb) else _as_1d(lb)
    if lbv is not None:
        if lbv.size == 1:
            lbv = np.full(n, float(lbv[0]))
        finite_lb = np.isfinite(lbv)
        if finite_lb.any():
            I_lb = np.eye(n)[:, finite_lb]
            C_blocks.append(I_lb)
            b_blocks.append(lbv[finite_lb])

    # Upper bounds: x <= ub  →  -I' x >= -ub
    ubv = None if _is_empty(ub) else _as_1d(ub)
    if ubv is not None:
        if ubv.size == 1:
            ubv = np.full(n, float(ubv[0]))
        finite_ub = np.isfinite(ubv)
        if finite_ub.any():
            I_ub = -np.eye(n)[:, finite_ub]
            C_blocks.append(I_ub)
            b_blocks.append(-ubv[finite_ub])

    # ── Call quadprog ─────────────────────────────────────────────
    if len(C_blocks) > 0:
        C = np.hstack(C_blocks)          # (n, m_total)
        b_vec = np.concatenate(b_blocks)  # (m_total,)
        x, val, xu, iters, lagr, iact = qp_module.solve_qp(
            G, a_vec, C, b_vec, meq
        )
    else:
        x, val, xu, iters, lagr, iact = qp_module.solve_qp(G, a_vec)

    # ── Build MATLAB-like output dict ─────────────────────────────
    qpOut = {
        "exitflag": 1,
        "output": {
            "solver": "quadprog (Goldfarb-Idnani)",
            "iterations": iters,
            "objective": val,
        },
        "lambda": {
            "ineqlin": lagr if A_ineq is not None else None,
            "eqlin": None,
            "lower": None,
            "upper": None,
        },
        "iact": iact,
        "mu": mu,
        "Sigma": Sigma,
        "gamma": gamma,
    }

    return x, qpOut
