# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:38:23 2026

@author: G_BRUNO
"""

# -*- coding: utf-8 -*-
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
    if x.ndim == 2 and 1 in x.shape:
        return x.reshape(-1)
    return x.reshape(-1)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Matrix input must be 2D.")
    return x


def markowitz_quadprog(
    mu,
    Sigma,
    A=None,
    bineq=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    qpOpts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standard Markowitz QP (tangency direction) with linear constraints:

        minimize   0.5*b' Sigma b  -  mu' b
        s.t.       A b <= bineq
                   Aeq b = beq
                   lb <= b <= ub

    Unconstrained FOC => Sigma b = mu => b = Sigma^{-1} mu.

    Returns (b, qpOut) with a MATLAB-like diagnostics dict (same style as kns_ridge_quadprog).
    """
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

    # QP matrices for cvxopt: 0.5 x' P x + q' x
    P = 0.5 * (Sigma + Sigma.T)   # enforce symmetry
    q = -mu

    # Optional linear constraints
    A = None if _is_empty(A) else _as_2d(A)
    bineq = None if _is_empty(bineq) else _as_1d(bineq)
    Aeq = None if _is_empty(Aeq) else _as_2d(Aeq)
    beq = None if _is_empty(beq) else _as_1d(beq)

    if A is not None and bineq is None:
        raise ValueError("A provided but bineq missing.")
    if Aeq is not None and beq is None:
        raise ValueError("Aeq provided but beq missing.")

    # Bounds: [] => unbounded
    if _is_empty(lb):
        lbv = -np.inf * np.ones(n)
    else:
        lbv = _as_1d(lb)
        if lbv.size == 1:
            lbv = np.full(n, float(lbv[0]))
        if lbv.size != n:
            raise ValueError("lb must be scalar or length N")

    if _is_empty(ub):
        ubv = np.inf * np.ones(n)
    else:
        ubv = _as_1d(ub)
        if ubv.size == 1:
            ubv = np.full(n, float(ubv[0]))
        if ubv.size != n:
            raise ValueError("ub must be scalar or length N")

    qpOpts = {} if qpOpts is None else dict(qpOpts)
    show_progress = bool(qpOpts.get("verbose", False))
    abstol = float(qpOpts.get("abstol", 1e-10))
    reltol = float(qpOpts.get("reltol", 1e-10))
    feastol = float(qpOpts.get("feastol", 1e-10))
    maxiters = int(qpOpts.get("maxiters", 500))
    jitter = float(qpOpts.get("jitter", 0.0))

    if jitter > 0:
        P = P + jitter * np.eye(n)

    try:
        from cvxopt import matrix, solvers
    except ImportError as e:
        raise ImportError("cvxopt not installed. Install: conda install -c conda-forge cvxopt") from e

    P_cvx = matrix(P)
    q_cvx = matrix(q)

    # Inequalities: include A b <= bineq and bounds
    G_blocks, h_blocks = [], []
    m_A = m_lb = m_ub = 0

    if A is not None:
        if A.shape[1] != n or bineq.shape[0] != A.shape[0]:
            raise ValueError("A and bineq have incompatible shapes.")
        G_blocks.append(A)
        h_blocks.append(bineq)
        m_A = A.shape[0]

    I = np.eye(n)
    lb_mask = np.isfinite(lbv)
    ub_mask = np.isfinite(ubv)

    if lb_mask.any():
        G_blocks.append((-I)[lb_mask, :])
        h_blocks.append((-lbv)[lb_mask])
        m_lb = int(lb_mask.sum())

    if ub_mask.any():
        G_blocks.append(I[ub_mask, :])
        h_blocks.append(ubv[ub_mask])
        m_ub = int(ub_mask.sum())

    if len(G_blocks) > 0:
        G_cvx = matrix(np.vstack(G_blocks))
        h_cvx = matrix(np.concatenate(h_blocks))
    else:
        G_cvx = None
        h_cvx = None

    if Aeq is not None:
        if Aeq.shape[1] != n or beq.shape[0] != Aeq.shape[0]:
            raise ValueError("Aeq and beq have incompatible shapes.")
        A_cvx = matrix(Aeq)
        b_cvx = matrix(beq)
    else:
        A_cvx = None
        b_cvx = None

    old_opts = dict(solvers.options)
    solvers.options["show_progress"] = show_progress
    solvers.options["abstol"] = abstol
    solvers.options["reltol"] = reltol
    solvers.options["feastol"] = feastol
    solvers.options["maxiters"] = maxiters

    try:
        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    finally:
        solvers.options.clear()
        solvers.options.update(old_opts)

    status = sol.get("status", "")
    success = (status == "optimal")
    exitflag = 1 if success else 0

    b = np.array(sol["x"], dtype=float).reshape(-1)

    # Multipliers (same unpacking logic as kns_ridge_quadprog)
    z = np.array(sol["z"]).reshape(-1) if sol.get("z", None) is not None else None
    y = np.array(sol["y"]).reshape(-1) if sol.get("y", None) is not None else None

    lam_ineqlin = None
    lam_lower = np.zeros(n, dtype=float)
    lam_upper = np.zeros(n, dtype=float)

    if z is not None:
        idx = 0
        if m_A > 0:
            lam_ineqlin = z[idx:idx + m_A].copy()
            idx += m_A
        if m_lb > 0:
            lam_lb_active = z[idx:idx + m_lb].copy()
            lam_lower[lb_mask] = lam_lb_active
            idx += m_lb
        if m_ub > 0:
            lam_ub_active = z[idx:idx + m_ub].copy()
            lam_upper[ub_mask] = lam_ub_active
            idx += m_ub

    qpOut = {
        "exitflag": exitflag,
        "output": {
            "solver": "CVXOPT.qp",
            "status": status,
            "iterations": sol.get("iterations", None),
            "primal_objective": sol.get("primal objective", None),
            "dual_objective": sol.get("dual objective", None),
        },
        "lambda": {
            "ineqlin": lam_ineqlin,
            "eqlin": y,
            "lower": lam_lower,
            "upper": lam_upper,
        },
        "mu": mu,
        "Sigma": Sigma,
    }
    return b, qpOut