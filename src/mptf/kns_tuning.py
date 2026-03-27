# -*- coding: utf-8 -*-
"""
kns_tuning.py

Cross-validation for KNS kappa (shrinkage) parameter tuning.

Port of MATLAB ``kns_rolling_cv_kappa.m`` with the following improvements:
  - Eigendecomposition-based vectorized ridge solve across all kappas
  - Per-window NaN-column handling (factors with NaN in train or val are excluded)
  - Market-beta adjustment extracted as reusable helper
  - Extensible OOS metric dispatch
  - Optional parallelism placeholder (n_jobs)
  - Consistent covariance method throughout (no train/refit inconsistency)

Author: G_BRUNO (ported & improved from MATLAB)
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

from .portfolio_helpers import regcov_det, regcov_LW, regcov_sample


# Type aliases
CovMethod = Literal["det", "lw", "sample"]
OOSMetric = Literal["csr2", "hjd", "sr"]


# ================================================================== #
#  Market-beta adjustment (reusable in mptf_calculation.py too)       #
# ================================================================== #

def market_beta_adjust(
    R: np.ndarray,
    mkt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hedge market exposure from factor returns and rescale to match
    market volatility.

    Port of the MATLAB logic inside ``kns_rolling_cv_kappa.m`` and
    ``mptfcalculation.m``::

        mktbeta = [ones, mkt] \\ R       % OLS regression
        mktbeta = mktbeta(2,:)           % keep slopes
        R_adj   = R - mktbeta .* mkt     % hedge
        Lev     = std(mkt)./std(R_adj)   % leverage ratio
        R_adj   = R_adj .* Lev           % lever

    Parameters
    ----------
    R : ndarray, shape (T, N)
        Factor returns.
    mkt : ndarray, shape (T,) or (T, 1)
        Market excess returns.

    Returns
    -------
    R_adj : ndarray (T, N)
        Hedged and levered factor returns.
    betas : ndarray (N,)
        Market betas (OLS slopes).
    lev : ndarray (N,)
        Leverage factors (std_mkt / std_residual per factor).
    """
    R = np.asarray(R, dtype=float)
    mkt = np.asarray(mkt, dtype=float).ravel()
    T, N = R.shape

    # OLS: [ones, mkt] \ R  →  (2, N)
    X = np.column_stack([np.ones(T), mkt])
    coeff = np.linalg.lstsq(X, R, rcond=None)[0]  # (2, N)
    betas = coeff[1, :]                             # (N,)

    # Hedge: subtract market component
    R_adj = R - betas[np.newaxis, :] * mkt[:, np.newaxis]

    # Lever: rescale each residual to match market volatility
    # MATLAB std() uses ddof=1
    std_mkt = np.std(mkt, ddof=1)
    std_adj = np.std(R_adj, axis=0, ddof=1)
    zero_std = std_adj < 1e-15
    if np.any(zero_std):
        R_adj[:, zero_std] = np.nan
    std_adj_safe = np.where(zero_std, 1.0, std_adj)
    lev = np.where(zero_std, np.nan, std_mkt / std_adj_safe)

    R_adj = R_adj * lev[np.newaxis, :]

    return R_adj, betas, lev


def apply_market_beta(
    R: np.ndarray,
    mkt: np.ndarray,
    betas: np.ndarray,
    lev: np.ndarray,
) -> np.ndarray:
    """
    Apply pre-fitted market-beta adjustment to new data.

    Use betas and lev from :func:`market_beta_adjust` estimated on the
    training window and apply them to validation (or OOS) data — no
    look-ahead.

    Parameters
    ----------
    R : ndarray (T_new, N)
    mkt : ndarray (T_new,)
    betas : ndarray (N,) — from training window
    lev   : ndarray (N,) — from training window

    Returns
    -------
    R_adj : ndarray (T_new, N)
    """
    R = np.asarray(R, dtype=float)
    mkt = np.asarray(mkt, dtype=float).ravel()
    R_adj = R - betas[np.newaxis, :] * mkt[:, np.newaxis]
    R_adj = R_adj * lev[np.newaxis, :]
    return R_adj


# ================================================================== #
#  Covariance dispatch                                                #
# ================================================================== #

def _get_cov_func(
    cov_method: Union[CovMethod, Callable[[np.ndarray], np.ndarray]],
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve string or callable to a covariance-estimation function."""
    if callable(cov_method):
        return cov_method
    _map = {"det": regcov_det, "lw": regcov_LW, "sample": regcov_sample}
    key = cov_method.strip().lower()
    if key not in _map:
        raise ValueError(f"Unknown cov_method='{cov_method}'. Use {list(_map)}.")
    return _map[key]


# ================================================================== #
#  Vectorized ridge solve via eigendecomposition                      #
# ================================================================== #

def _solve_ridge_all_kappas(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tau: float,
    T_train: int,
    kappa_grid_ann: np.ndarray,
    F: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve  b_k = (Σ + γ_k I)^{-1} μ  for all kappas simultaneously.

    Instead of calling ``np.linalg.solve`` nK times, we decompose
    Σ = Q Λ Qᵀ once, then:

        b_k = Q  diag(1 / (λ_i + γ_k))  Qᵀ μ

    This reduces the inner loop from O(nK × N³) to O(N³ + nK × N²).

    Parameters
    ----------
    mu : (N,) — training mean returns
    Sigma : (N, N) — training covariance (regularized)
    tau : float — trace(Sigma)
    T_train : int — number of training observations
    kappa_grid_ann : (nK,) — annualized kappa grid
    F : int — frequency (252 daily, 12 monthly)

    Returns
    -------
    B : (nK, N) — SDF coefficient vectors, one row per kappa
    gammas : (nK,) — penalty parameters corresponding to each kappa
    """
    eigenvalues, Q = np.linalg.eigh(Sigma)  # λ sorted ascending
    z = Q.T @ mu  # (N,) — projected mean

    kappa_per = kappa_grid_ann / np.sqrt(F)
    gammas = tau / (kappa_per ** 2 * T_train)  # (nK,)

    # (nK, N):  z_i / (λ_i + γ_k)
    z_scaled = z[np.newaxis, :] / (
        eigenvalues[np.newaxis, :] + gammas[:, np.newaxis]
    )

    # B[k, :] = Q @ z_scaled[k, :]   →   (nK, N)
    B = z_scaled @ Q.T

    return B, gammas


# ================================================================== #
#  OOS metric functions  (vectorized across kappas)                   #
# ================================================================== #

def _oos_csr2_vec(
    B: np.ndarray,
    mu_va: np.ndarray,
    Sig_va: np.ndarray,
) -> np.ndarray:
    """
    Cross-sectional R²:  1 - ||μ_va - Σ_va b||² / ||μ_va||²

    Eq. (30) in Kozak, Nagel, Santosh (2020).

    Parameters
    ----------
    B : (nK, N_valid) — SDF coefficients per kappa
    mu_va, Sig_va : validation moments

    Returns
    -------
    r2 : (nK,)
    """
    denom = float(mu_va @ mu_va)
    if denom <= np.finfo(float).eps:
        return np.full(B.shape[0], -np.inf)
    fitted = B @ Sig_va                          # (nK, N) — Σ_va is symmetric
    resid = mu_va[np.newaxis, :] - fitted        # (nK, N)
    ss_resid = np.sum(resid ** 2, axis=1)        # (nK,)
    return 1.0 - ss_resid / denom


def _oos_hjd_vec(
    B: np.ndarray,
    mu_va: np.ndarray,
    Sig_va: np.ndarray,
) -> np.ndarray:
    """
    Negative Hansen-Jagannathan distance (squared):
        -(μ - Σb)ᵀ Σ⁻¹ (μ - Σb)

    Maximising this is equivalent to minimising the HJ distance.

    Returns (nK,).
    """
    fitted = B @ Sig_va                          # (nK, N)
    resid = mu_va[np.newaxis, :] - fitted        # (nK, N)
    try:
        Sig_va_inv = np.linalg.inv(Sig_va)
    except np.linalg.LinAlgError:
        Sig_va_inv = np.linalg.pinv(Sig_va)
    resid_scaled = resid @ Sig_va_inv            # (nK, N)
    return -np.sum(resid * resid_scaled, axis=1) # (nK,)


def _oos_sr_vec(
    B: np.ndarray,
    mu_va: np.ndarray,
    Sig_va: np.ndarray,
) -> np.ndarray:
    """
    OOS Sharpe ratio of the SDF portfolio: (μᵀb) / √(bᵀΣb)

    Returns (nK,).
    """
    numer = B @ mu_va                            # (nK,)
    tmp = B @ Sig_va                             # (nK, N)
    var = np.sum(B * tmp, axis=1)                # (nK,) = diag(B Σ Bᵀ)
    var = np.maximum(var, np.finfo(float).eps)
    return numer / np.sqrt(var)


_OOS_DISPATCH: Dict[str, Callable] = {
    "csr2": _oos_csr2_vec,
    "hjd":  _oos_hjd_vec,
    "sr":   _oos_sr_vec,
}


# ================================================================== #
#  Frequency helper                                                   #
# ================================================================== #

def _parse_frequency(frequency: Union[str, int, float]) -> int:
    """Convert frequency specification to observations per year (F)."""
    if isinstance(frequency, str):
        _map = {"daily": 252, "monthly": 12, "weekly": 52}
        key = frequency.strip().lower()
        if key not in _map:
            raise ValueError(f"Unknown frequency='{frequency}'. Use {list(_map)}.")
        return _map[key]
    F = int(frequency)
    if F <= 0:
        raise ValueError("Numeric frequency must be positive.")
    return F


# ================================================================== #
#  Main function                                                      #
# ================================================================== #

def kns_tuning_kappa(
    R: np.ndarray,
    mkt: np.ndarray,
    *,
    frequency: Union[str, int, float] = "monthly",
    output_type: str = "CSR2",
    train_years: float = 10,
    val_years: float = 5,
    skip_years: float = 1,
    train_obs: Optional[int] = None,
    val_obs: Optional[int] = None,
    skip_obs: Optional[int] = None,
    kappa_grid_ann: Optional[np.ndarray] = None,
    cov_method: Union[CovMethod, Callable] = "det",
    market_adjust: bool = True,
    min_valid_cols: int = 2,
    verbose: bool = True,
) -> Tuple[float, np.ndarray, float, dict]:
    """
    Rolling-window cross-validation to find the optimal KNS kappa.

    For each rolling window, trains SDF coefficients on the training
    portion and evaluates OOS fit on the validation portion, across a
    grid of annualized kappa values. Kappa controls shrinkage intensity
    (higher = less shrinkage).

    **Improvements over MATLAB version:**

    - Eigendecomposition-based vectorized solve (all kappas at once)
    - Per-window NaN-column handling: factors with any NaN in the
      train or validation slice are excluded for that window
    - Market-beta adjustment is optional and extracted as a helper
    - Consistent covariance method throughout (train windows AND refit)
    - Extensible OOS metric dispatch

    Parameters
    ----------
    R : ndarray, shape (T, N)
        Factor returns. May contain NaN columns that start/end at
        different dates; per-window column filtering is applied.
    mkt : ndarray, shape (T,) or (T, 1)
        Market excess returns (must not contain NaN).
    frequency : str or int
        ``'daily'`` (252), ``'monthly'`` (12), ``'weekly'`` (52), or
        numeric F.
    output_type : str
        OOS metric: ``'CSR2'``, ``'HJD'``, or ``'SR'``
        (case-insensitive).
    train_years, val_years, skip_years : float
        Window lengths in years (converted to obs via F). Ignored if
        the corresponding ``*_obs`` parameter is provided.
    train_obs, val_obs, skip_obs : int, optional
        Direct observation counts (override year-based calculation).
    kappa_grid_ann : ndarray, optional
        Annualized kappa grid. Default: ``logspace(-1, log10(10), 201)``.
    cov_method : str or callable
        ``'det'`` (default, MATLAB-consistent), ``'lw'``, ``'sample'``,
        or a custom ``func(Z) -> Sigma``.
    market_adjust : bool
        If True (default), hedge market beta and lever residuals inside
        each CV window. Set False to skip.
    min_valid_cols : int
        Minimum number of NaN-free columns required to process a window.
        Windows with fewer valid columns are skipped. Default: 2.
    verbose : bool
        Print progress information.

    Returns
    -------
    kappa_opt_ann : float
        CV-optimal annualized kappa.
    b_opt : ndarray, shape (N,)
        SDF coefficients estimated on the FULL sample at kappa_opt_ann.
        Contains ``np.nan`` for columns that had any NaN in the full sample.
    metric_opt : float
        Best average OOS metric across windows.
    out : dict
        Diagnostics::

            kappa_grid_ann   : (nK,)  annualized kappa grid
            frequency_F      : int    observations per year
            output_type      : str
            n_windows        : int    number of valid windows processed
            n_windows_total  : int    total windows constructed
            metric_by_window : (W, nK) per-window OOS metric (NaN for
                               skipped windows)
            avg_metric       : (nK,) average across valid windows
            gamma_full       : float  penalty at kappa_opt on full sample
            idx_best         : int    index into kappa_grid_ann
            valid_cols_full  : (N,) bool — columns used in full-sample refit
    """
    # ── Input validation ──────────────────────────────────────────
    R = np.asarray(R, dtype=float)
    mkt = np.asarray(mkt, dtype=float).ravel()
    if R.ndim != 2:
        raise ValueError("R must be 2-D (T × N).")
    T, N = R.shape
    if mkt.shape[0] != T:
        raise ValueError(
            f"mkt length ({mkt.shape[0]}) must equal R row count ({T})."
        )
    if np.any(np.isnan(mkt)):
        raise ValueError("mkt must not contain NaN.")

    # ── Parse options ─────────────────────────────────────────────
    F = _parse_frequency(frequency)
    cov_func = _get_cov_func(cov_method)

    otype = output_type.strip().lower()
    if otype not in _OOS_DISPATCH:
        raise ValueError(
            f"Unknown output_type='{output_type}'. Use {list(_OOS_DISPATCH)}."
        )
    metric_func = _OOS_DISPATCH[otype]

    # ── Window sizes (observations) ───────────────────────────────
    n_train = train_obs if train_obs is not None else max(1, round(train_years * F))
    n_val   = val_obs   if val_obs   is not None else max(1, round(val_years * F))
    n_skip  = skip_obs  if skip_obs  is not None else max(1, round(skip_years * F))
    if n_train + n_val > T:
        raise ValueError(
            f"train ({n_train}) + val ({n_val}) = {n_train + n_val} > T ({T})."
        )

    # ── Kappa grid ────────────────────────────────────────────────
    if kappa_grid_ann is None:
        kappa_grid_ann = np.logspace(-1, np.log10(10), 201)
    else:
        kappa_grid_ann = np.asarray(kappa_grid_ann, dtype=float).ravel()
    nK = len(kappa_grid_ann)

    # ── Build rolling windows ─────────────────────────────────────
    windows = []
    t0 = 0  # Python 0-based indexing
    while (t0 + n_train + n_val) <= T:
        tr = slice(t0, t0 + n_train)
        va = slice(t0 + n_train, t0 + n_train + n_val)
        windows.append((tr, va))
        t0 += n_skip
    W_total = len(windows)
    if W_total == 0:
        raise ValueError(
            "No rolling windows could be constructed. "
            "Reduce train/val lengths or increase T."
        )
    if verbose:
        print(
            f"Rolling CV: {nK} kappas × {W_total} windows  |  "
            f"F={F}, n_train={n_train}, n_val={n_val}, n_skip={n_skip}"
        )

    # ── Main loop over windows ────────────────────────────────────
    metric_by_window = np.full((W_total, nK), np.nan)

    for widx, (tr_sl, va_sl) in enumerate(windows):
        R_tr = R[tr_sl, :]   # (n_train, N)
        R_va = R[va_sl, :]   # (n_val, N)
        mkt_tr = mkt[tr_sl]
        mkt_va = mkt[va_sl]

        # ── Per-window NaN column filtering ───────────────────────
        valid_cols = (
            ~np.any(np.isnan(R_tr), axis=0)
            & ~np.any(np.isnan(R_va), axis=0)
        )  # (N,) bool
        n_valid = int(valid_cols.sum())
        if n_valid < min_valid_cols:
            if verbose:
                print(f"  Window {widx + 1}/{W_total}: skipped "
                      f"({n_valid} valid cols < {min_valid_cols})")
            continue

        R_tr_v = R_tr[:, valid_cols]  # (n_train, n_valid)
        R_va_v = R_va[:, valid_cols]  # (n_val,   n_valid)

        # ── Market-beta adjustment ────────────────────────────────
        if market_adjust:
            R_tr_v, betas_w, lev_w = market_beta_adjust(R_tr_v, mkt_tr)
            R_va_v = apply_market_beta(R_va_v, mkt_va, betas_w, lev_w)

        # ── Training moments ──────────────────────────────────────
        mu_tr = np.mean(R_tr_v, axis=0)  # (n_valid,)
        Sig_tr = cov_func(R_tr_v)        # (n_valid, n_valid)
        tau_tr = float(np.trace(Sig_tr))
        T_tr = R_tr_v.shape[0]

        # ── Vectorized ridge solve across all kappas ──────────────
        B_w, _ = _solve_ridge_all_kappas(
            mu_tr, Sig_tr, tau_tr, T_tr, kappa_grid_ann, F
        )  # (nK, n_valid)

        # ── Validation moments ────────────────────────────────────
        mu_va = np.mean(R_va_v, axis=0)  # (n_valid,)
        Sig_va = cov_func(R_va_v)        # (n_valid, n_valid)

        # ── OOS metric for all kappas ─────────────────────────────
        metric_by_window[widx, :] = metric_func(B_w, mu_va, Sig_va)

        if verbose and (widx + 1) % max(1, W_total // 5) == 0:
            print(f"  Window {widx + 1}/{W_total} done")

    # ── Aggregate across windows ──────────────────────────────────
    valid_windows = ~np.all(np.isnan(metric_by_window), axis=1)
    n_valid_windows = int(valid_windows.sum())
    if n_valid_windows == 0:
        raise RuntimeError(
            "All windows were skipped (too many NaN columns). "
            "Check data or lower min_valid_cols."
        )

    avg_metric = np.nanmean(metric_by_window[valid_windows, :], axis=0)
    idx_best = int(np.nanargmax(avg_metric))
    kappa_opt_ann = float(kappa_grid_ann[idx_best])
    metric_opt = float(avg_metric[idx_best])

    if verbose:
        print(
            f"Optimal kappa (ann): {kappa_opt_ann:.4f}  |  "
            f"Avg OOS {otype.upper()}: {metric_opt:.4f}  |  "
            f"Valid windows: {n_valid_windows}/{W_total}"
        )

    # ── Refit on FULL sample at kappa_opt_ann ─────────────────────
    valid_cols_full = ~np.any(np.isnan(R), axis=0)  # (N,) bool
    n_valid_full = int(valid_cols_full.sum())

    b_opt = np.full(N, np.nan)

    if n_valid_full >= min_valid_cols:
        R_full = R[:, valid_cols_full]
        mkt_full = mkt

        if market_adjust:
            R_full, _, _ = market_beta_adjust(R_full, mkt_full)

        mu_full = np.mean(R_full, axis=0)
        Sig_full = cov_func(R_full)
        tau_full = float(np.trace(Sig_full))

        kappa_opt_per = kappa_opt_ann / np.sqrt(F)
        gamma_full = tau_full / (kappa_opt_per ** 2 * R_full.shape[0])

        # Single solve at optimal kappa
        I_n = np.eye(n_valid_full)
        try:
            b_full = np.linalg.solve(Sig_full + gamma_full * I_n, mu_full)
        except np.linalg.LinAlgError:
            b_full = np.linalg.lstsq(
                Sig_full + gamma_full * I_n, mu_full, rcond=None
            )[0]

        b_opt[valid_cols_full] = b_full
    else:
        gamma_full = np.nan

    # ── Pack diagnostics ──────────────────────────────────────────
    out = {
        "kappa_grid_ann":   kappa_grid_ann,
        "frequency_F":      F,
        "output_type":      otype,
        "n_windows":        n_valid_windows,
        "n_windows_total":  W_total,
        "metric_by_window": metric_by_window,
        "avg_metric":       avg_metric,
        "gamma_full":       gamma_full,
        "idx_best":         idx_best,
        "valid_cols_full":  valid_cols_full,
        "n_train":          n_train,
        "n_val":            n_val,
        "n_skip":           n_skip,
    }

    return kappa_opt_ann, b_opt, metric_opt, out