# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 16:32:02 2026

@author: G_BRUNO
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union
import numpy as np
from .portfolio_helpers import sdfcoefficients_bayes_kns, regcov_det, regcov_LW, regcov_sample
from .markowitz_quadprog import markowitz_quadprog


def _get(I: Any, key: str, default: Any = None) -> Any:
    """Support both dict-like and attribute-like inputs."""
    if isinstance(I, Mapping):
        return I.get(key, default)
    return getattr(I, key, default)


def _bayes_kns_handler(
    I: Any,
    *,
    cov_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "det",
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    mptf_type = str(_get(I, "mptf_type")).strip().lower()

    R = _get(I, "returns")
    if R is None:
        raise ValueError("_bayes_kns_handler: I.returns is required.")
    R = np.asarray(R, dtype=float)

    frequency = _get(I, "est_freq")
    if frequency is None:
        raise ValueError("_bayes_kns_handler: I.est_freq is required for BayesKNS.")

    normalizewgt = _get(I, "normalizewgt", "YES")
    A = _get(I, "A", None)
    bineq = _get(I, "bineq", None)

    if mptf_type == "bayes_kns_naive":
        kappa = 50.0
    elif mptf_type == "bayes_kns_heu":
        kappa = float(_get(I, "kappa_heu"))
    elif mptf_type == "bayes_kns_extsmall":
        kappa = 0.01
    elif mptf_type == "bayes_kns_datacsr2":
        kappa = float(_get(I, "kappa_ddcsr2"))
    elif mptf_type == "bayes_kns_datahjd":
        kappa = float(_get(I, "kappa_ddhjd"))
    elif mptf_type == "bayes_kns_freeprior":
        kappa = float(_get(I, "kappa_freeprior"))
    else:
        raise NotImplementedError(f"_bayes_kns_handler: unsupported mptf_type='{mptf_type}'")

    beta, post_SR, post_mu, Phi = sdfcoefficients_bayes_kns(
        frequency=frequency,
        Z=R,
        kappa=kappa,
        normalize=normalizewgt,
        A=A,
        bineq=bineq,
        cov_method=cov_method,  # default "det" (MATLAB-consistent)
    )
    return beta, post_SR, post_mu, Phi


def _ew_handler(
    I: Any,
    *,
    mode: str,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Equal-weight handler (GROSS-normalized):
      - sign from prior
      - equal absolute weights across selected factors
      - sum(abs(weights)) == 1

    Modes:
      - 'meanret'   : sign_i = sign(mean_i) from estimation sample
      - 'freeprior' : sign_i = sign(ewprior_i) provided in I.ewprior

    Returns:
      beta, post_SR (nan), post_mu (None), Phi (None)
    """
    R = _get(I, "returns")
    if R is None:
        raise ValueError("_ew_handler: I.returns is required.")
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("_ew_handler: I.returns must be 2D (T x N).")

    T, N = R.shape
    mode_l = mode.strip().lower()

    if mode_l == "meanret":
        mu = np.nanmean(R, axis=0)   # (N,)
        s = np.sign(mu)              # -1, 0, +1
    elif mode_l == "freeprior":
        ewprior = _get(I, "ewprior")
        if ewprior is None:
            raise ValueError("_ew_handler(mode='cntrprior'): missing I.ewprior.")
        ewprior = np.asarray(ewprior, dtype=float).reshape(-1)
        if ewprior.shape[0] != N:
            raise ValueError(f"_ew_handler: ewprior length {ewprior.shape[0]} != N {N}.")
        s = np.sign(ewprior)
    else:
        raise NotImplementedError(f"_ew_handler: unknown mode='{mode}'")

    # Select non-zero signs
    sel = s != 0
    if not np.any(sel):
        raise ValueError(f"_ew_handler: prior produced empty selection (mode='{mode_l}').")

    s_sel = s[sel].astype(float)          # now only ±1
    K = int(s_sel.size)

    # Gross equal-weight: each selected factor gets ±1/K
    w_sel = s_sel / K                     # sum(abs(w_sel)) == 1

    beta = np.zeros(N, dtype=float)
    beta[sel] = w_sel

    return beta, float("nan"), None, None


def _markowitz_handler(
    I: Any,
    *,
    cov_kind: str,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Standard Markowitz tangency direction under constraints:
        min 0.5 b' Sigma b - mu' b  s.t. constraints
    """
    R = _get(I, "returns")
    if R is None:
        raise ValueError("_markowitz_handler: I.returns is required.")
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("_markowitz_handler: I.returns must be 2D (T x N).")

    mu = np.nanmean(R, axis=0)

    ck = cov_kind.strip().lower()
    if ck in {"naive", "sample", "cov"}:
        Sigma = regcov_sample(R)
    elif ck in {"det", "detcov"}:
        Sigma = regcov_det(R)
    elif ck in {"lw", "lwcov"}:
        Sigma = regcov_LW(R)
    else:
        raise ValueError(f"_markowitz_handler: unknown cov_kind='{cov_kind}'")

    A = _get(I, "A", None)
    bineq = _get(I, "bineq", None)
    Aeq = _get(I, "Aeq", None)
    beq = _get(I, "beq", None)
    lb = _get(I, "lb", None)
    ub = _get(I, "ub", None)
    qpOpts = _get(I, "qpOpts", None)

    # If there are no constraints/bounds, solve directly (faster, no cvxopt).
    has_lin = (A is not None) or (Aeq is not None)
    has_bnd = (lb is not None) or (ub is not None)

    if not has_lin and not has_bnd:
        try:
            beta = np.linalg.solve(Sigma, mu)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Sigma, mu, rcond=None)[0]
        qpOut = None
    else:
        beta, qpOut = markowitz_quadprog(mu, Sigma, A=A, bineq=bineq, Aeq=Aeq, beq=beq, lb=lb, ub=ub, qpOpts=qpOpts)

    # Optional normalization (match your convention: L1 if normalizewgt == YES)
    normalizewgt = _get(I, "normalizewgt", "YES")
    do_norm = (normalizewgt is True) or (isinstance(normalizewgt, str) and normalizewgt.strip().lower() == "yes")
    if do_norm:
        s = float(np.sum(np.abs(beta)))
        if s > 0:
            beta = beta / s

    # Ex-ante SR using the same moments used to compute weights
    denom = float(np.sqrt(beta @ (Sigma @ beta)))
    post_SR = float((beta @ mu) / denom) if denom > 0 else float("nan")

    # Return mu and Sigma as "post_mu" and "Phi" so downstream ex-ante formulas can reuse them
    return beta, post_SR, mu, Sigma


# -----------------------------------------------------------------------------
# Dispatch map
# -----------------------------------------------------------------------------
_DISPATCH: Dict[str, Callable[..., Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]]] = {
    # Bayes/KNS family
    "bayes_kns_naive": _bayes_kns_handler,
    "bayes_kns_heu": _bayes_kns_handler,
    "bayes_kns_extsmall": _bayes_kns_handler,
    "bayes_kns_datacsr2": _bayes_kns_handler,
    "bayes_kns_datahjd": _bayes_kns_handler,
    "bayes_kns_freeprior": _bayes_kns_handler,

    # Equal-weight family
    "ew_meanret": lambda I, **kwargs: _ew_handler(I, mode="meanret"),
    "ew_freeprior": lambda I, **kwargs: _ew_handler(I, mode="freeprior"),
     
    # Markowitz family (standard MVE, no ridge)
    "markowitz_naivecov": lambda I, **kwargs: _markowitz_handler(I, cov_kind="sample"),
    "markowitz_detcov":   lambda I, **kwargs: _markowitz_handler(I, cov_kind="det"),
    "markowitz_lwcov":    lambda I, **kwargs: _markowitz_handler(I, cov_kind="lw"),    
}


def call_mptfweights(
    I: Any,
    *,
    cov_method: Union[str, Callable[[np.ndarray], np.ndarray]] = "det",
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Dispatcher for portfolio weights.

    Returns (beta, post_SR, post_mu, Phi). For non-Bayes methods, the last
    three outputs are (nan, None, None) unless you later define them.
    """
    mptf_type = str(_get(I, "mptf_type")).strip().lower()
    if mptf_type not in _DISPATCH:
        raise NotImplementedError(f"call_mptfweights: unknown mptf_type='{mptf_type}'")

    # Pass cov_method to handlers that use it; EW lambdas ignore it via **kwargs.
    return _DISPATCH[mptf_type](I, cov_method=cov_method)