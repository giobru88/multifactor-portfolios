# -*- coding: utf-8 -*-
"""
sharpe_test.py

Ledoit & Wolf (2008) robust Sharpe ratio testing via studentized
time-series block bootstrap (Politis & Romano, 1994).

Port of MATLAB functions from the sharpe-ratio-testing repository:
  - bootInference          : test H0: SR1 = SR2
  - bootInference_diffZero : test H0: SR = 0

Full dependency tree ported:
  sharpeRatioDiff, sharpeRatioDiffZero,
  sharpeHACnoOut, sharpeHACdiffZero,
  computeSE, computeSEdiffZero, computeSEpw, computeSEpwDiffZero,
  computeVhat, computeVhatDiffZero,
  computePSI, computePSIdiffZero,
  computeAlpha, GammaHat, kernelType,
  ar2, mlag, trimr, regnoint, cbbSequence

Author: Giovanni Bruno (Python port from Michael Wolf's MATLAB code)
References:
  - Ledoit, O. & Wolf, M. (2008). "Robust performance hypothesis testing
    with the Sharpe ratio." Journal of Empirical Finance, 15, 850-859.
  - Politis, D.N. & Romano, J.P. (1994). "The stationary bootstrap."
    JASA, 89, 1303-1313.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════

def trimr(x: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Strip first n1 and last n2 rows from x."""
    n = x.shape[0]
    if n1 + n2 >= n:
        raise ValueError("Attempting to trim too much in trimr")
    return x[n1: n - n2] if n2 > 0 else x[n1:]


def mlag(x: np.ndarray, nlag: int, init: float = 0.0) -> np.ndarray:
    """Generate matrix of nlag lags from x (nobs × nvar)."""
    x = np.atleast_2d(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    nobs, nvar = x.shape
    xlag = np.full((nobs, nvar * nlag), init)
    icnt = 0
    for i in range(nvar):
        for j in range(nlag):
            xlag[j + 1:nobs, icnt + j] = x[0:nobs - j - 1, i]
        icnt += nlag
    return xlag


def regnoint(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS without intercept: y = X @ coef + resi."""
    coef = np.linalg.solve(X.T @ X, X.T @ y)
    resi = y - X @ coef
    return coef, resi


def ar2(y: np.ndarray, nlag: int, const: int = 1) -> dict:
    """AR(nlag) model via OLS. Returns dict with beta, sige, yhat, x."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if const == 1:
        x = np.column_stack([np.ones(n), mlag(y[:, np.newaxis], nlag)])
    else:
        x = mlag(y[:, np.newaxis], nlag)
    x = trimr(x, nlag, 0)
    y_trim = trimr(y[:, np.newaxis], nlag, 0).ravel()
    b0 = np.linalg.solve(x.T @ x, x.T @ y_trim)
    p = nlag + const
    sige = float((y_trim - x @ b0).T @ (y_trim - x @ b0)) / (n - p + 1)
    return {"beta": b0, "sige": sige, "yhat": x @ b0, "x": x}


# ═══════════════════════════════════════════════════════════════════
#  KERNEL & HAC BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════

def kernel_type(x: float, ktype: str = "G") -> float:
    """Parzen-Gallant ('G') or Quadratic-Spectral ('QS') kernel weight."""
    if ktype == "G":
        x = abs(x)
        if x < 0.5:
            return 1.0 - 6.0 * x**2 + 6.0 * x**3
        elif x < 1.0:
            return 2.0 * (1.0 - x)**3
        else:
            return 0.0
    elif ktype == "QS":
        term = 6.0 * np.pi * x / 5.0
        return 25.0 * (np.sin(term) / term - np.cos(term)) / (12.0 * np.pi**2 * x**2)
    else:
        raise ValueError(f"Unknown kernel type: {ktype}")


def gamma_hat(vhat: np.ndarray, j: int) -> np.ndarray:
    """Autocovariance matrix at lag j: (1/T) sum_{t=j+1}^T v_t v_{t-j}'."""
    T, p = vhat.shape
    if j >= T:
        raise ValueError("j must be smaller than T")
    G = np.zeros((p, p))
    for i in range(j, T):
        G += np.outer(vhat[i], vhat[i - j])
    return G / T


def compute_alpha(vhat: np.ndarray) -> float:
    """Andrews (1991) bandwidth parameter for HAC kernel."""
    T, p = vhat.shape
    numerator = 0.0
    denominator = 0.0
    for i in range(p):
        res = ar2(vhat[:, i], 1)
        rhohat = res["beta"][1]  # AR(1) coefficient (index 1 because const is index 0)
        sighat = np.sqrt(res["sige"])
        numerator += 4.0 * rhohat**2 * sighat**4 / (1.0 - rhohat)**8
        denominator += sighat**4 / (1.0 - rhohat)**4
    return numerator / denominator


# ═══════════════════════════════════════════════════════════════════
#  PSI (long-run variance) ESTIMATORS
# ═══════════════════════════════════════════════════════════════════

def compute_psi(vhat: np.ndarray, ktype: str = "G") -> np.ndarray:
    """HAC long-run covariance (for 2-strategy SR difference)."""
    T = vhat.shape[0]
    alpha_hat = compute_alpha(vhat)
    Sstar = 2.6614 * (alpha_hat * T)**0.2
    psi = gamma_hat(vhat, 0)
    j = 1
    while j < Sstar:
        G = gamma_hat(vhat, j)
        psi += kernel_type(j / Sstar, ktype) * (G + G.T)
        j += 1
    psi = (T / (T - 4)) * psi
    return psi


def compute_psi_diff_zero(vhat: np.ndarray, ktype: str = "G") -> np.ndarray:
    """HAC long-run covariance (for single-strategy SR ≠ 0 test)."""
    T = vhat.shape[0]
    alpha_hat = compute_alpha(vhat)
    Sstar = 2.6614 * (alpha_hat * T)**0.2
    psi = gamma_hat(vhat, 0)
    j = 1
    while j < Sstar:
        G = gamma_hat(vhat, j)
        psi += kernel_type(j / Sstar, ktype) * (G + G.T)
        j += 1
    psi = (T / (T - 2)) * psi  # note: T-2 instead of T-4
    return psi


# ═══════════════════════════════════════════════════════════════════
#  Vhat (centered moment vectors)
# ═══════════════════════════════════════════════════════════════════

def compute_vhat(ret: np.ndarray) -> np.ndarray:
    """Centered moment vector for 2-strategy case: [r1-mu1, r2-mu2, r1²-γ1, r2²-γ2]."""
    r1, r2 = ret[:, 0], ret[:, 1]
    nu = np.array([r1.mean(), r2.mean(), (r1**2).mean(), (r2**2).mean()])
    return np.column_stack([r1 - nu[0], r2 - nu[1], r1**2 - nu[2], r2**2 - nu[3]])


def compute_vhat_diff_zero(ret: np.ndarray) -> np.ndarray:
    """Centered moment vector for single-strategy case: [r1-mu1, r1²-γ1]."""
    r1 = ret[:, 0] if ret.ndim == 2 else ret.ravel()
    nu = np.array([r1.mean(), (r1**2).mean()])
    return np.column_stack([r1 - nu[0], r1**2 - nu[1]])


# ═══════════════════════════════════════════════════════════════════
#  STANDARD ERROR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def compute_se(ret: np.ndarray, ktype: str = "G") -> float:
    """HAC SE for difference of two Sharpe ratios."""
    r1, r2 = ret[:, 0], ret[:, 1]
    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    grad = np.array([
        g1 / (g1 - mu1**2)**1.5,
        -g2 / (g2 - mu2**2)**1.5,
        -0.5 * mu1 / (g1 - mu1**2)**1.5,
        0.5 * mu2 / (g2 - mu2**2)**1.5,
    ])
    vhat = compute_vhat(ret)
    psi = compute_psi(vhat, ktype)
    return float(np.sqrt(grad @ psi @ grad / T))


def compute_se_diff_zero(ret: np.ndarray, ktype: str = "G") -> float:
    """HAC SE for single Sharpe ratio (H0: SR = 0)."""
    r1 = ret[:, 0] if ret.ndim == 2 else ret.ravel()
    T = len(r1)
    mu1 = r1.mean()
    g1 = (r1**2).mean()
    grad = np.array([
        g1 / (g1 - mu1**2)**1.5,
        -0.5 * mu1 / (g1 - mu1**2)**1.5,
    ])
    vhat = compute_vhat_diff_zero(ret)
    psi = compute_psi_diff_zero(vhat, ktype)
    return float(np.sqrt(grad @ psi @ grad / T))


def compute_se_pw(ret: np.ndarray, ktype: str = "G") -> float:
    """Prewhitened HAC SE for difference of two Sharpe ratios."""
    r1, r2 = ret[:, 0], ret[:, 1]
    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    grad = np.array([
        g1 / (g1 - mu1**2)**1.5,
        -g2 / (g2 - mu2**2)**1.5,
        -0.5 * mu1 / (g1 - mu1**2)**1.5,
        0.5 * mu2 / (g2 - mu2**2)**1.5,
    ])
    vhat = compute_vhat(ret)
    # Prewhiten via VAR(1) without intercept
    p = 4
    Als = np.zeros((p, p))
    Vstar = np.zeros((T - 1, p))
    X = vhat[:-1, :]
    for j in range(p):
        coef, resi = regnoint(X, vhat[1:, j])
        Als[j, :] = coef.ravel()
        Vstar[:, j] = resi.ravel()
    # Clamp singular values
    U, s, Vt = np.linalg.svd(Als)
    s = np.clip(s, -0.97, 0.97)
    Ahat = U @ np.diag(s) @ Vt
    D = np.linalg.inv(np.eye(p) - Ahat)
    for j in range(p):
        Vstar[:, j] = vhat[1:, j] - (Ahat[j, :] @ X.T)
    psi = compute_psi(Vstar, ktype)
    psi = D @ psi @ D.T
    return float(np.sqrt(grad @ psi @ grad / T))


def compute_se_pw_diff_zero(ret: np.ndarray, ktype: str = "G") -> float:
    """Prewhitened HAC SE for single Sharpe ratio (H0: SR = 0)."""
    r1 = ret[:, 0] if ret.ndim == 2 else ret.ravel()
    T = len(r1)
    mu1 = r1.mean()
    g1 = (r1**2).mean()
    grad = np.array([
        g1 / (g1 - mu1**2)**1.5,
        -0.5 * mu1 / (g1 - mu1**2)**1.5,
    ])
    vhat = compute_vhat_diff_zero(ret)
    p = 2
    Als = np.zeros((p, p))
    Vstar = np.zeros((T - 1, p))
    X = vhat[:-1, :]
    for j in range(p):
        coef, resi = regnoint(X, vhat[1:, j])
        Als[j, :] = coef.ravel()
        Vstar[:, j] = resi.ravel()
    U, s, Vt = np.linalg.svd(Als)
    s = np.clip(s, -0.97, 0.97)
    Ahat = U @ np.diag(s) @ Vt
    D = np.linalg.inv(np.eye(p) - Ahat)
    for j in range(p):
        Vstar[:, j] = vhat[1:, j] - (Ahat[j, :] @ X.T)
    psi = compute_psi_diff_zero(Vstar, ktype)
    psi = D @ psi @ D.T
    return float(np.sqrt(grad @ psi @ grad / T))


# ═══════════════════════════════════════════════════════════════════
#  SHARPE RATIO FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def sharpe_ratio_diff(ret: np.ndarray) -> float:
    """SR1 - SR2 from a T×2 return matrix. Uses MATLAB convention: var() with ddof=1."""
    r1, r2 = ret[:, 0], ret[:, 1]
    sr1 = r1.mean() / np.sqrt(np.var(r1, ddof=1))
    sr2 = r2.mean() / np.sqrt(np.var(r2, ddof=1))
    return float(sr1 - sr2)


def sharpe_ratio_diff_zero(ret: np.ndarray) -> float:
    """SR from a T×1 return vector. Uses MATLAB convention: var() with ddof=1."""
    r1 = ret[:, 0] if ret.ndim == 2 else ret.ravel()
    return float(r1.mean() / np.sqrt(np.var(r1, ddof=1)))


# ═══════════════════════════════════════════════════════════════════
#  HAC WRAPPERS (matching MATLAB sharpeHACnoOut / sharpeHACdiffZero)
# ═══════════════════════════════════════════════════════════════════

def sharpe_hac_no_out(ret: np.ndarray, ktype: str = "G") -> Tuple[float, float, float, float]:
    """HAC inference for SR difference (no printing). Returns (se, pval, sePw, pvalPw)."""
    r1, r2 = ret[:, 0], ret[:, 1]
    sr1 = r1.mean() / np.sqrt(np.var(r1, ddof=1))
    sr2 = r2.mean() / np.sqrt(np.var(r2, ddof=1))
    diff = sr1 - sr2
    se = compute_se(ret, ktype)
    se_pw = compute_se_pw(ret, ktype)
    from scipy.stats import norm
    pval = 2.0 * norm.cdf(-abs(diff) / se) if se > 0 else np.nan
    pval_pw = 2.0 * norm.cdf(-abs(diff) / se_pw) if se_pw > 0 else np.nan
    return se, pval, se_pw, pval_pw


def sharpe_hac_diff_zero(ret: np.ndarray, ktype: str = "G") -> Tuple[float, float, float, float]:
    """HAC inference for SR ≠ 0 (no printing). Returns (se, pval, sePw, pvalPw)."""
    r1 = ret[:, 0] if ret.ndim == 2 else ret.ravel()
    sr = r1.mean() / np.sqrt(np.var(r1, ddof=1))
    se = compute_se_diff_zero(ret, ktype)
    se_pw = compute_se_pw_diff_zero(ret, ktype)
    from scipy.stats import norm
    pval = 2.0 * norm.cdf(-abs(sr) / se) if se > 0 else np.nan
    pval_pw = 2.0 * norm.cdf(-abs(sr) / se_pw) if se_pw > 0 else np.nan
    return se, pval, se_pw, pval_pw


# ═══════════════════════════════════════════════════════════════════
#  CIRCULAR BLOCK BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════

def cbb_sequence(T: int, b: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Circular block bootstrap index sequence (0-based)."""
    if rng is None:
        rng = np.random.default_rng()
    l = T // b + 1
    index_seq = np.concatenate([np.arange(T), np.arange(b)])
    start_points = rng.integers(0, T, size=l)
    seq = np.empty(l * b, dtype=int)
    for j in range(l):
        start = start_points[j]
        seq[j * b: (j + 1) * b] = index_seq[start: start + b]
    return seq[:T]


# ═══════════════════════════════════════════════════════════════════
#  MAIN BOOTSTRAP INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def boot_inference(
    ret: np.ndarray,
    b: int,
    M: int = 4999,
    se_type: str = "G",
    pw: bool = True,
    delta_null: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[float, float, float, int, float]:
    """
    Bootstrap test for equality of two Sharpe ratios.

    H0: SR1 - SR2 = delta_null

    Parameters
    ----------
    ret : (T, 2) array of excess returns
    b : block size for circular block bootstrap
    M : number of bootstrap repetitions (default 4999)
    se_type : HAC kernel type, 'G' (Parzen-Gallant) or 'QS'
    pw : use prewhitened HAC SE (default True)
    delta_null : hypothesized SR difference (default 0)
    seed : random seed for reproducibility

    Returns
    -------
    p_value : bootstrap p-value
    delta_hat : observed SR difference
    d : original test statistic
    b : block size used
    se : HAC standard error
    """
    ret = np.asarray(ret, dtype=float)
    assert ret.ndim == 2 and ret.shape[1] == 2

    rng = np.random.default_rng(seed)

    # Observed SR difference
    delta_hat = sharpe_ratio_diff(ret)

    # HAC standard error
    se_raw, _, se_pw_val, _ = sharpe_hac_no_out(ret, se_type)
    se = se_pw_val if pw else se_raw

    # Original test statistic
    d = abs(delta_hat - delta_null) / se

    T = ret.shape[0]
    b_root = np.sqrt(b)
    l = T // b
    T_adj = l * b

    p_value = 1
    for m in range(M):
        # Bootstrap pseudo-data
        idx = cbb_sequence(T_adj, b, rng)
        ret_star = ret[idx, :]

        # Bootstrap SR difference
        delta_hat_star = sharpe_ratio_diff(ret_star)

        # Bootstrap gradient & moments
        r1s, r2s = ret_star[:, 0], ret_star[:, 1]
        mu1s, mu2s = r1s.mean(), r2s.mean()
        g1s, g2s = (r1s**2).mean(), (r2s**2).mean()

        grad = np.array([
            g1s / (g1s - mu1s**2)**1.5,
            -g2s / (g2s - mu2s**2)**1.5,
            -0.5 * mu1s / (g1s - mu1s**2)**1.5,
            0.5 * mu2s / (g2s - mu2s**2)**1.5,
        ])

        y_star = np.column_stack([
            r1s - mu1s, r2s - mu2s, r1s**2 - g1s, r2s**2 - g2s
        ])

        # Bootstrap SE via block means
        psi_star = np.zeros((4, 4))
        for j in range(l):
            zeta = b_root * y_star[j * b: (j + 1) * b, :].mean(axis=0)
            psi_star += np.outer(zeta, zeta)
        psi_star /= l

        se_star = np.sqrt(grad @ psi_star @ grad / T_adj)

        # Bootstrap test statistic
        d_star = abs(delta_hat_star - delta_hat) / se_star if se_star > 0 else 0.0
        if d_star >= d:
            p_value += 1

    p_value = p_value / (M + 1)
    return p_value, delta_hat, d, b, se


def boot_inference_diff_zero(
    ret: np.ndarray,
    b: int,
    M: int = 4999,
    se_type: str = "G",
    pw: bool = True,
    delta_null: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[float, float, float, int, float]:
    """
    Bootstrap test that a single Sharpe ratio equals zero.

    H0: SR = delta_null

    Parameters
    ----------
    ret : (T,) or (T, 1) array of excess returns
    b : block size for circular block bootstrap
    M : number of bootstrap repetitions (default 4999)
    se_type : HAC kernel type, 'G' (Parzen-Gallant) or 'QS'
    pw : use prewhitened HAC SE (default True)
    delta_null : hypothesized SR (default 0)
    seed : random seed for reproducibility

    Returns
    -------
    p_value : bootstrap p-value
    delta_hat : observed SR
    d : original test statistic
    b : block size used
    se : HAC standard error
    """
    ret = np.asarray(ret, dtype=float)
    if ret.ndim == 1:
        ret = ret[:, np.newaxis]

    rng = np.random.default_rng(seed)

    # Observed SR
    delta_hat = sharpe_ratio_diff_zero(ret)

    # HAC standard error
    se_raw, _, se_pw_val, _ = sharpe_hac_diff_zero(ret, se_type)
    se = se_pw_val if pw else se_raw

    # Original test statistic
    d = abs(delta_hat - delta_null) / se

    T = ret.shape[0]
    b_root = np.sqrt(b)
    l = T // b
    T_adj = l * b

    p_value = 1
    for m in range(M):
        idx = cbb_sequence(T_adj, b, rng)
        ret_star = ret[idx, :]

        delta_hat_star = sharpe_ratio_diff_zero(ret_star)

        r1s = ret_star[:, 0]
        mu1s = r1s.mean()
        g1s = (r1s**2).mean()

        grad = np.array([
            g1s / (g1s - mu1s**2)**1.5,
            -0.5 * mu1s / (g1s - mu1s**2)**1.5,
        ])

        y_star = np.column_stack([r1s - mu1s, r1s**2 - g1s])

        psi_star = np.zeros((2, 2))
        for j in range(l):
            zeta = b_root * y_star[j * b: (j + 1) * b, :].mean(axis=0)
            psi_star += np.outer(zeta, zeta)
        psi_star /= l

        se_star = np.sqrt(grad @ psi_star @ grad / T_adj)

        d_star = abs(delta_hat_star - delta_hat) / se_star if se_star > 0 else 0.0
        if d_star >= d:
            p_value += 1

    p_value = p_value / (M + 1)
    return p_value, delta_hat, d, b, se
