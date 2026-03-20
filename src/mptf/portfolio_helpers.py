# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 18:02:54 2026

@author: G_BRUNO
"""

# -*- coding: utf-8 -*-
"""
portfolio_helpers.py

KNS-style Bayes SDF coefficients + covariance regularization utilities.

This module includes:
- regcov_det : deterministic covariance shrinkage consistent with your MATLAB regcov.m
- regcov_LW  : original Python placeholder (Ledoit–Wolf-ish diagonal shrinkage)
- sdfcoefficients_bayes_kns : Bayes/KNS coefficients that can use either covariance estimator

Author: (ported for Giovanni Bruno)
"""

from typing import Callable, Literal, Optional, Tuple, Union
import numpy as np


Frequency = Literal["monthly", "weekly", "daily"]
CovMethod = Literal["det", "lw","sample"]


# -----------------------------------------------------------------------------
# Covariance estimators
# -----------------------------------------------------------------------------
def regcov_det(r: np.ndarray) -> np.ndarray:
    """
    Exact Python equivalent of your MATLAB regcov.m:

        function [X] = regcov(r)
            X = cov(r);
            [T, n] = size(r);
            a = n / (n + T);
            X = a*trace(X)/n*eye(n) + (1-a)*X;
        end

    Notes:
    - MATLAB cov(r) uses 1/(T-1) normalization and treats columns as variables.
    - Target is (trace(X)/n) * I  (scaled identity).
    - Shrinkage intensity is deterministic: a = n/(n+T).

    Parameters
    ----------
    r : array_like, shape (T, n)
        Observations in rows, variables in columns.

    Returns
    -------
    X_reg : ndarray, shape (n, n)
        Regularized covariance matrix.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("regcov_det: input must be 2D (T x n).")
    if not np.isfinite(r).all():
        raise ValueError("regcov_det: input contains NaN/Inf. Handle missing data upstream.")

    T, n = r.shape
    if T < 2:
        raise ValueError("regcov_det: need at least 2 observations (T>=2).")

    # MATLAB cov(r): columns=variables, unbiased normalization 1/(T-1)
    X = np.cov(r, rowvar=False, bias=False)  # (n, n)

    a = n / (n + T)
    target = (np.trace(X) / n) * np.eye(n, dtype=np.float64)

    X_reg = a * target + (1.0 - a) * X
    X_reg = 0.5 * (X_reg + X_reg.T)  # enforce symmetry numerically
    return X_reg


def regcov_LW(Z: np.ndarray) -> np.ndarray:
    """
    Original Python placeholder covariance regularization (Ledoit–Wolf-ish diagonal shrinkage).

    This is NOT MATLAB-consistent. It shrinks toward diag(S) with a data-driven intensity.

    Implementation details (as originally provided):
    - Uses 1/T scaling for the raw covariance S = (X'X)/T
    - Target is diagonal diag(S)
    - Shrinkage intensity delta is estimated from second moments (heuristic LW-like)

    Parameters
    ----------
    Z : array_like, shape (T, N)

    Returns
    -------
    Sigma : ndarray, shape (N, N)
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("regcov_LW: Z must be 2D (T x N).")
    if not np.isfinite(Z).all():
        raise ValueError("regcov_LW: Z contains NaN/Inf; handle missingness upstream.")

    T, N = Z.shape
    if T < 2:
        raise ValueError("regcov_LW: Need at least 2 observations to estimate covariance.")

    # Center
    X = Z - Z.mean(axis=0, keepdims=True)

    # Biased sample covariance with 1/T scaling
    S = (X.T @ X) / T  # N x N

    # Target: diagonal of S
    F = np.diag(np.diag(S))

    # Heuristic LW-style shrinkage intensity toward diagonal
    X2 = X * X
    B = (X2.T @ X2) / T
    pi_hat = (B - S * S).sum() / T

    gamma_hat = ((S - F) * (S - F)).sum()
    if gamma_hat <= 0:
        delta = 0.0
    else:
        delta = float(np.clip(pi_hat / gamma_hat, 0.0, 1.0))

    Sigma = (1.0 - delta) * S + delta * F
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma

def regcov_sample(Z: np.ndarray) -> np.ndarray:
    """
    Standard sample covariance matrix using the same convention as MATLAB cov(Z):
    - columns = variables, rows = observations
    - unbiased normalization (divide by T-1)
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("regcov_sample: Z must be 2D (T x N).")
    if not np.isfinite(Z).all():
        raise ValueError("regcov_sample: Z contains NaN/Inf; handle missingness upstream.")
    T, N = Z.shape
    if T < 2:
        raise ValueError("regcov_sample: need at least 2 observations (T>=2).")

    Sigma = np.cov(Z, rowvar=False, bias=False)  # (N,N), divide by T-1
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma



# Backward-compatible alias (optional, but helpful if other modules call regcov())
def regcov(Z: np.ndarray) -> np.ndarray:
    """Default regcov: MATLAB-consistent deterministic shrinkage."""
    return regcov_det(Z)


# -----------------------------------------------------------------------------
# Bayes/KNS SDF coefficients
# -----------------------------------------------------------------------------
def sdfcoefficients_bayes_kns(
    frequency: Frequency,
    Z: np.ndarray,
    kappa: float = 0.5,
    normalize: Union[str, bool] = "YES",
    A: Optional[np.ndarray] = None,
    bineq: Optional[np.ndarray] = None,
    cov_method: Union[CovMethod, Callable[[np.ndarray], np.ndarray]] = "det",
    qp_solver: str = "quadprog",
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Port of MATLAB sdfcoefficients_BayesKNS, with selectable covariance estimator.

    MATLAB logic replicated:
      - kappa_F = kappa / sqrt(F)
      - Sigma   = regcov(Z)   [here selectable via cov_method]
      - tau     = trace(Sigma)
      - gamma   = tau / ((kappa_F^2) * T)
      - V       = (1/(T*gamma)) * Sigma
      - Phi     = Sigma + V
      - beta    = kns_ridge_quadprog(mean(Z), Sigma, gamma, A, bineq)
      - beta0   = kns_ridge_quadprog(mean(Z), Sigma, gamma)  (unconstrained)
      - post_mu = Sigma @ beta0
      - optional L1 normalization: beta /= sum(abs(beta))
      - post_SRPtf = (beta' post_mu) / sqrt(beta' Sigma beta)
      - post_muPtf = post_mu (vector)

    Parameters
    ----------
    frequency : {'monthly','weekly','daily'}
        Used only for kappa scaling.
    Z : ndarray, shape (T, N)
        Factor returns, rows=time.
    kappa : float
        Prior on max achievable annualized Sharpe ratio.
    normalize : 'YES'/'NO' or bool
        If YES/True: L1-normalize beta by sum(abs(beta)).
    A, bineq : optional
        Linear inequality constraints A @ beta <= bineq (passed to kns_ridge_quadprog).
    cov_method : {'det','lw'} or callable
        - 'det' (default): regcov_det (MATLAB-consistent)
        - 'lw'           : regcov_LW  (original Python placeholder)
        - callable       : custom covariance estimator function(Z)->Sigma
    qp_solver : str
        - 'quadprog' (default): Goldfarb-Idnani active-set (matches MATLAB)
        - 'cvxopt'            : interior-point (legacy, can fail with small gamma)

    Returns
    -------
    beta : ndarray, shape (N,)
        SDF weights (possibly constrained + normalized).
    post_SRPtf : float
        Posterior expected Sharpe ratio (per-period, not annualized).
    post_muPtf : ndarray, shape (N,)
        Vector post_mu = Sigma @ beta0 (matches your MATLAB code).
    Phi : ndarray, shape (N, N)
        Phi = Sigma + (1/(T*gamma))*Sigma
    """
    freq = str(frequency).lower()
    if freq == "monthly":
        F = 12.0
    elif freq == "weekly":
        F = 52.0
    elif freq == "daily":
        F = 252.0
    else:
        raise ValueError("frequency must be one of: 'monthly', 'weekly', 'daily'.")

    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (T x N).")
    if not np.isfinite(Z).all():
        raise ValueError("Z contains NaN/Inf (clean upstream for replication).")

    T, N = Z.shape
    if T < 2:
        raise ValueError("Z must have at least 2 rows (T>=2).")

    # Annualized -> per-period kappa
    kappa_F = float(kappa) / np.sqrt(F)

    # Mean vector (N,)
    mu = Z.mean(axis=0)
    # ----- Covariance estimator selection -----
    if callable(cov_method):
       Sigma = cov_method(Z)
    else:
        cm = str(cov_method).strip().lower()
        if cm == "det":
           Sigma = regcov_det(Z)          # MATLAB-consistent
        elif cm == "lw":
           Sigma = regcov_LW(Z)           # placeholder
        elif cm in {"sample", "cov", "standard"}:
           Sigma = regcov_sample(Z)       # standard sample cov (T-1)
        else:
           raise ValueError(
            "cov_method must be 'det', 'lw', 'sample' (or 'cov'/'standard'), "
            "or a callable(Z)->Sigma."
        )

    # (optional but recommended) sanity checks:
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.shape != (N, N):
       raise ValueError(f"Covariance estimator returned shape {Sigma.shape}, expected {(N, N)}.")
    Sigma = 0.5 * (Sigma + Sigma.T)


    tau = float(np.trace(Sigma))

    # KNS gamma map (note: MATLAB uses T here even though cov uses 1/(T-1) inside cov())
    gamma = tau / ((kappa_F ** 2) * T)

    # Estimation error matrix and Phi
    V = (1.0 / (T * gamma)) * Sigma
    Phi = Sigma + V

    # Solve constrained and unconstrained
    _solver = qp_solver.strip().lower() if isinstance(qp_solver, str) else "quadprog"
    if _solver == "cvxopt":
        from .kns_ridge_quadprog_cvxopt import kns_ridge_quadprog as _qp_solve
    else:
        from .kns_ridge_quadprog import kns_ridge_quadprog as _qp_solve

    beta, _ = _qp_solve(mu, Sigma, gamma, A=A, bineq=bineq)
    beta0, _ = _qp_solve(mu, Sigma, gamma)

    # Posterior mean vector
    post_mu = Sigma @ beta0

    # Optional L1 normalization
    do_norm = (normalize is True) or (isinstance(normalize, str) and normalize.strip().lower() == "yes")
    if do_norm:
        s = float(np.sum(np.abs(beta)))
        if s > 0:
            beta = beta / s

    # Posterior Sharpe (per-period)
    denom = float(np.sqrt(beta @ (Sigma @ beta)))
    post_SRPtf = float((beta @ post_mu) / denom) if denom > 0 else np.nan

    post_muPtf = post_mu
    return beta, post_SRPtf, post_muPtf, Phi