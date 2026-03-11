"""
regression_helpers.py — OLS with robust standard errors.

Port of MATLAB regrobustse_GB.m.  Supports:
  - 'ordinary'  : classical OLS standard errors
  - 'white'     : White (1980) heteroskedasticity-consistent (HC0)
  - 'nw'        : Newey-West (1987) HAC with Bartlett kernel
  - 'clustered' : cluster-robust (one-way)

All functions return a dict with:
    coeff      (k,)   OLS coefficients
    se         (k,)   standard errors
    tstat      (k,)   t-statistics  (coeff / se)
    residuals  (T,)   OLS residuals
    V          (k,k)  variance-covariance matrix of coefficients

NaN guard: if y or X contain any NaN, all numeric outputs are NaN.

Author: Giovanni Bruno (Python port)
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Bandwidth selection helpers
# ═══════════════════════════════════════════════════════════════════

def _ar1_mle_rho(z):
    """Exact (unconditional) MLE for AR(1) with intercept.

    Fits: z_t = c + rho * z_{t-1} + u_t   via profile MLE.

    Parameters
    ----------
    z : (T,) array

    Returns
    -------
    rho   : float, AR(1) coefficient, clamped to (-0.99, 0.99)
    sig2  : float, innovation variance (MLE estimate)
    """
    from scipy.optimize import minimize_scalar

    T = len(z)
    if T < 3:
        return 0.0, np.var(z)

    z_lag = z[:-1]
    z_cur = z[1:]

    def neg_profile_loglik(rho):
        rho2 = rho * rho
        if rho2 >= 0.9999:
            return 1e15
        # Conditional mean: c = mean(z) * (1 - rho)
        c = np.mean(z) * (1.0 - rho)
        ss_cond = np.sum((z_cur - c - rho * z_lag) ** 2)
        # First obs contribution
        mu1 = c / (1.0 - rho)
        ss_first = (1.0 - rho2) * (z[0] - mu1) ** 2
        sig2 = (ss_first + ss_cond) / T
        if sig2 <= 1e-30:
            return 1e15
        return T * np.log(sig2) - np.log(1.0 - rho2)

    result = minimize_scalar(neg_profile_loglik, bounds=(-0.99, 0.99), method="bounded")
    rho = np.clip(result.x, -0.99, 0.99)

    # Compute sig2 at optimal rho
    c = np.mean(z) * (1.0 - rho)
    ss_cond = np.sum((z_cur - c - rho * z_lag) ** 2)
    mu1 = c / (1.0 - rho)
    ss_first = (1.0 - rho ** 2) * (z[0] - mu1) ** 2
    sig2 = (ss_first + ss_cond) / T

    return rho, sig2


def _nw_auto_bandwidth(X, e, T, k, has_intercept=True):
    """Newey-West (1994) / Andrews (1991) data-dependent bandwidth for Bartlett kernel.

    Matches MATLAB's hac() with 'type','HAC','weights','BT','Bandwidth','AR1MLE'.

    For each score series z_t = e_t * x_{t,j}, fits AR(1) via MLE to get
    rho_j and sigma_j^2. Then computes alpha_1 (for Bartlett kernel):

        num0_j  = 4 * rho_j^2 * sigma_j^4
        num1_j  = (1 - rho_j)^6 * (1 + rho_j)^2
        den_j   = sigma_j^4 / (1 - rho_j)^4

        alpha_1 = sum(w_j * num0_j / num1_j) / sum(w_j * den_j)
        c*      = 1.1447 * (alpha_1 * T)^(1/3)

    where w_j = 0 for the intercept column and 1 otherwise.

    Parameters
    ----------
    X : (T, k) regressor matrix
    e : (T,) OLS residuals
    T : int, number of observations
    k : int, number of regressors
    has_intercept : bool, whether column 0 of X is an intercept

    Returns
    -------
    float : optimal bandwidth c* (continuous)
    """
    Xe = X * e[:, np.newaxis]  # T x k — score matrix V

    # Alpha weights: zero for intercept, one for slopes (matches MATLAB)
    w_alpha = np.ones(k)
    if has_intercept:
        w_alpha[0] = 0.0

    num_sum = 0.0
    den_sum = 0.0

    for j in range(k):
        if w_alpha[j] == 0.0:
            continue
        z = Xe[:, j]
        rho, sig2 = _ar1_mle_rho(z)
        sig4 = sig2 ** 2

        # Andrews (1991) formula — alpha_1 for Bartlett kernel
        num0 = 4.0 * rho ** 2 * sig4
        num1 = (1.0 - rho) ** 6 * (1.0 + rho) ** 2
        den = sig4 / (1.0 - rho) ** 4

        num_sum += w_alpha[j] * num0 / num1
        den_sum += w_alpha[j] * den

    if den_sum < 1e-30:
        return 1.0

    alpha1 = num_sum / den_sum
    c_star = 1.1447 * (alpha1 * T) ** (1.0 / 3.0)
    c_star = min(c_star, float(T - 1))
    return c_star


def _nw_simple_bandwidth(T):
    """Simple plug-in rule: floor(4 * (T/100)^(2/9)).

    Parameters
    ----------
    T : int, number of observations

    Returns
    -------
    float : lag truncation L (as float for consistency)
    """
    return float(int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))


def _nw_bandwidth(arg4, X, e, T, k, has_intercept=True):
    """Resolve NW bandwidth from user input.

    Parameters
    ----------
    arg4 : int, str, or None
        int      -> fixed lag (used as integer bandwidth = arg4 + 1)
        'auto'   -> Newey-West (1994) data-dependent (default)
        'simple' -> plug-in rule floor(4*(T/100)^(2/9)), used as integer
        None     -> same as 'auto'
    has_intercept : bool, whether first column of X is an intercept

    Returns
    -------
    float : bandwidth c (continuous for 'auto', integer+1 for fixed lag)
    """
    if arg4 is None or (isinstance(arg4, str) and arg4.lower() == "auto"):
        return _nw_auto_bandwidth(X, e, T, k, has_intercept)
    elif isinstance(arg4, str) and arg4.lower() == "simple":
        # Simple rule returns L; bandwidth = L + 1
        return _nw_simple_bandwidth(T) + 1.0
    elif isinstance(arg4, (int, float, np.integer, np.floating)):
        # Integer → interpreted as lag count L, bandwidth = L + 1
        # Float with decimals → interpreted as direct bandwidth c
        if isinstance(arg4, (float, np.floating)) and arg4 != int(arg4):
            return float(arg4)  # direct bandwidth
        else:
            return float(int(arg4)) + 1.0  # lag → bandwidth
    else:
        raise ValueError(
            f"arg4 for 'nw' must be int, 'auto', 'simple', or None. Got: {arg4}"
        )


# ═══════════════════════════════════════════════════════════════════
# Main regression function
# ═══════════════════════════════════════════════════════════════════


def regrobustse(y, X, se_type="ordinary", arg4=None):
    """OLS regression with selectable standard-error type.

    Parameters
    ----------
    y : ndarray, shape (T,)
        Dependent variable.
    X : ndarray, shape (T, k)
        Regressors.  Include a column of ones for intercept if desired.
    se_type : str
        One of 'ordinary', 'white', 'nw', 'clustered'.
    arg4 : int or str or ndarray or None
        - If se_type == 'nw':
            int       -> fixed lag L (number of autocovariance lags).
            'auto'    -> Newey-West (1994) data-dependent bandwidth (default).
            'simple'  -> simple plug-in rule: floor(4*(T/100)^(2/9)).
            None      -> same as 'auto'.
        - If se_type == 'clustered': array of cluster IDs, shape (T,).

    Returns
    -------
    dict with keys: coeff, se, tstat, residuals, V
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    T, k = X.shape

    # ── NaN guard ─────────────────────────────────────────────────
    if np.any(np.isnan(y)) or np.any(np.isnan(X)):
        nan_k = np.full(k, np.nan)
        return {
            "coeff": nan_k,
            "se": nan_k,
            "tstat": nan_k,
            "residuals": np.full(T, np.nan),
            "V": np.full((k, k), np.nan),
        }

    # ── OLS ───────────────────────────────────────────────────────
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b

    # ── Variance-covariance matrix ────────────────────────────────
    se_type = se_type.lower().strip()

    if se_type == "ordinary":
        s2 = (e @ e) / (T - k)
        V = s2 * XtX_inv

    elif se_type == "white":
        # HC0 with small-sample correction: T/(T-k)
        Xe = X * e[:, np.newaxis]          # T x k
        meat = Xe.T @ Xe                   # k x k
        meat *= T / (T - k)
        V = XtX_inv @ meat @ XtX_inv

    elif se_type == "nw":
        # Detect intercept: first column is all ones
        has_intercept = np.allclose(X[:, 0], 1.0) if k > 0 else False

        # Bandwidth c (continuous); lags with positive weight: j = 1, ..., floor(c-eps)
        c = _nw_bandwidth(arg4, X, e, T, k, has_intercept)
        max_lag = int(np.floor(c - 1e-10))  # largest j with w_j > 0
        max_lag = max(min(max_lag, T - 1), 0)

        Xe = X * e[:, np.newaxis]          # T x k
        S = Xe.T @ Xe                      # Gamma_0

        for j in range(1, max_lag + 1):
            w = 1.0 - j / c                # Bartlett kernel: w_j = 1 - j/c
            Gj = Xe[j:].T @ Xe[:-j]       # k x k
            S += w * (Gj + Gj.T)

        # Small-sample correction (matches MATLAB hac default)
        S *= T / (T - k)

        V = XtX_inv @ S @ XtX_inv

    elif se_type == "clustered":
        if arg4 is None:
            raise ValueError("For 'clustered' SEs, provide cluster IDs as arg4.")
        cluster_ids = np.asarray(arg4).ravel()
        unique_cl = np.unique(cluster_ids)
        G = len(unique_cl)

        Xe = X * e[:, np.newaxis]
        meat = np.zeros((k, k))
        for cl in unique_cl:
            idx = cluster_ids == cl
            Xe_cl = Xe[idx].sum(axis=0, keepdims=True)  # 1 x k
            meat += Xe_cl.T @ Xe_cl

        # Small-sample correction: G/(G-1) * (T-1)/(T-k)
        correction = (G / (G - 1.0)) * ((T - 1.0) / (T - k))
        V = correction * XtX_inv @ meat @ XtX_inv

    else:
        raise ValueError(
            f"Unknown se_type '{se_type}'. Use 'ordinary', 'white', 'nw', or 'clustered'."
        )

    se = np.sqrt(np.diag(V))
    tstat = b / se

    return {
        "coeff": b,
        "se": se,
        "tstat": tstat,
        "residuals": e,
        "V": V,
    }