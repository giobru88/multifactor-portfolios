# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:15:44 2026

@author: G_BRUNO
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
    if x.ndim == 2 and 1 in x.shape:
        return x.reshape(-1)
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
    CVXOPT-only analogue of MATLAB kns_ridge_quadprog2:

        H = 2*(Sigma*Sigma + gamma*Sigma);
        H = (H+H')/2;
        f = -2*(Sigma*(mu'));

        [b,~,exitflag,output,lambda] = quadprog(H,f,A,bineq,Aeq,beq,lb,ub,[],opts);

    Solve:
        min 0.5*b'Hb + f'b
        s.t. A b <= bineq
             Aeq b = beq
             lb <= b <= ub

    Returns
    -------
    b : (N,) ndarray
    qpOut : dict with keys
        exitflag : 1 if optimal, else 0
        output   : dict of solver diagnostics
        lambda   : dict with fields {'ineqlin','eqlin','lower','upper'}
        mu, Sigma, gamma : echoes
    """
#    if gamma is None or gamma <= 0:
#        raise ValueError("Provide gamma > 0")

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

    # --- MATLAB quadprog setup ---
    H = 2.0 * (Sigma @ Sigma + gamma * Sigma)
    H = 0.5 * (H + H.T)  # enforce symmetry numerically
    f = -2.0 * (Sigma @ mu)

    if not np.isfinite(H).all():
        raise ValueError("H contains NaN/Inf. Check Sigma/gamma.")
    if not np.isfinite(f).all():
        raise ValueError("f contains NaN/Inf. Check mu/Sigma.")

    # Optional linear constraints
    A = None if _is_empty(A) else _as_2d(A)
    bineq = None if _is_empty(bineq) else _as_1d(bineq)
    Aeq = None if _is_empty(Aeq) else _as_2d(Aeq)
    beq = None if _is_empty(beq) else _as_1d(beq)

    if A is not None and bineq is None:
        raise ValueError("A provided but bineq missing.")
    if Aeq is not None and beq is None:
        raise ValueError("Aeq provided but beq missing.")

    # Bounds: MATLAB [] => unbounded
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

    # CVXOPT solve options
    qpOpts = {} if qpOpts is None else dict(qpOpts)
    show_progress = bool(qpOpts.get("verbose", False))
    abstol = float(qpOpts.get("abstol", 1e-10))
    reltol = float(qpOpts.get("reltol", 1e-10))
    feastol = float(qpOpts.get("feastol", 1e-10))
    maxiters = int(qpOpts.get("maxiters", 500))

    # Optional jitter (useful if CVXOPT complains about semidefinite P)
    # This does NOT change your MATLAB definition unless you choose to use it.
    jitter = float(qpOpts.get("jitter", 0.0))
    if jitter > 0:
        H = H + jitter * np.eye(n)

    # --- Build CVXOPT matrices ---
    try:
        from cvxopt import matrix, solvers
    except ImportError as e:
        raise ImportError("cvxopt not installed. Install: conda install -c conda-forge cvxopt") from e

    P = matrix(H)
    q = matrix(f)

    # Inequalities Gx <= h include:
    #   (i) A x <= bineq
    #   (ii) bounds: x >= lb => -I x <= -lb  (finite only)
    #              : x <= ub =>  I x <=  ub  (finite only)
    G_blocks = []
    h_blocks = []
    m_A = 0
    m_lb = 0
    m_ub = 0

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
        G = matrix(np.vstack(G_blocks))
        h = matrix(np.concatenate(h_blocks))
    else:
        G = None
        h = None

    # Equalities Ax = b
    if Aeq is not None:
        if Aeq.shape[1] != n or beq.shape[0] != Aeq.shape[0]:
            raise ValueError("Aeq and beq have incompatible shapes.")
        A_cvx = matrix(Aeq)
        b_cvx = matrix(beq)
    else:
        A_cvx = None
        b_cvx = None

    # --- Solve with CVXOPT (interior-point QP) ---
    # solvers.options is global; set and restore to avoid side-effects.
    old_opts = dict(solvers.options)
    solvers.options["show_progress"] = show_progress
    solvers.options["abstol"] = abstol
    solvers.options["reltol"] = reltol
    solvers.options["feastol"] = feastol
    solvers.options["maxiters"] = maxiters

    try:
        sol = solvers.qp(P, q, G, h, A_cvx, b_cvx)
    finally:
        solvers.options.clear()
        solvers.options.update(old_opts)

    status = sol.get("status", "")
    success = (status == "optimal")
    exitflag = 1 if success else 0

    b = np.array(sol["x"], dtype=float).reshape(-1)

    # --- Build MATLAB-like lambda struct ---
    # CVXOPT returns multipliers:
    #   z : multipliers for Gx <= h (stacked)
    #   y : multipliers for Aeq x = beq
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

    lambda_struct = {
        "ineqlin": lam_ineqlin,
        "eqlin": y,
        "lower": lam_lower,
        "upper": lam_upper,
    }

    qpOut = {
        "exitflag": exitflag,
        "output": {
            "solver": "CVXOPT.qp",
            "status": status,
            "iterations": sol.get("iterations", None),
            "primal_objective": sol.get("primal objective", None),
            "dual_objective": sol.get("dual objective", None),
        },
        "lambda": lambda_struct,
        "mu": mu,
        "Sigma": Sigma,
        "gamma": gamma,
    }

    return b, qpOut