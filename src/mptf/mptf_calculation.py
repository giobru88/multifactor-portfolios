"""
mptf_calculation.py — Main orchestrator for multifactor portfolio OOS/IS evaluation.

Port of MATLAB mptfcalculation.m.  Given a configuration dict P, this module:
  1. Builds factor-selection masks and constraint matrices.
  2. Runs a rolling OOS loop: at each OOS date it re-estimates weights
     (conditional on the update schedule) and records single-period OOS returns.
  3. Runs an IS calculation on the full OOS window for comparison.

Returns a nested dict (MPTF) mirroring the MATLAB struct.

Author: Giovanni Bruno (Python port)
"""

import re

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from mptf.factor_selection import factorselection
from mptf.call_mptfweights import call_mptfweights
from mptf.kns_tuning import kns_tuning_kappa
from mptf.helpers import previous_reb_gb
from mptf.selection_helpers import parse_maxmin_suffix, pick_by_k
from mptf.portfolio_helpers import regcov_det
from mptf.regression_helpers import regrobustse


# Private column names used after merging market into factor DataFrames.
# These guarantee row-level alignment between factor and market data.
_MKT_COL = "__MKT__"
_MKTd_COL = "__MKTd__"


# ═══════════════════════════════════════════════════════════════════
# Helper: update-tuning flag
# ═══════════════════════════════════════════════════════════════════

def _update_tuning(upd_tun, upd_tun_pre):
    """Return True if tuning date has advanced."""
    return upd_tun > upd_tun_pre


# ═══════════════════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════════════════

def mptfcalculation(P):
    """Run multifactor portfolio OOS and IS evaluation.

    Parameters
    ----------
    P : dict with keys (mirrors MATLAB struct P):
        PTF        : SimpleNamespace with .F (DataFrame, Date + factors), .pname, .dates
        PTFd       : same as PTF but daily frequency
        MKT        : DataFrame with columns ['Date', 'FF6_MktMinusRF'] (monthly)
        MKTd       : same but daily
        mptf_type  : list of str — portfolio types
        fctselid   : list of str — factor-set identifiers (possibly with _maxK/_minK suffix)
        fctselid_cv: list of str — factor set for kappa cross-validation
        oos_start  : datetime or pd.Timestamp
        oos_end    : datetime or pd.Timestamp
        T_est      : int — estimation window in months
        normalizewgt : str 'YES'/'NO'
        est_freq   : str 'monthly' or 'daily'
        updateFreq : str 'annual', 'monthly', or 'custom'
        updmonth   : int or None
        upddates   : list of datetime (used if updateFreq == 'custom')
        CONTAB     : list of lists or None — constraint table (see notes below)
        cv_start   : str or pd.Timestamp — start of CV training window
                     (default '1977-01-01'; set in main script)
        cv_freq    : str — frequency for kappa CV ('monthly' or 'daily')
                     (default: same as est_freq)

    Returns
    -------
    MPTF : dict with keys
        pname, rets_oos, rets_is, wgt_oos, wgt_is,
        kappa_heu, kappa_ddcsr2, kappa_ddhjd,
        kappa_heu_is, kappa_ddcsr2_is, kappa_ddhjd_is,
        exanteSR_oos, exanteMu_oos
    """

    # ── Defaults ──────────────────────────────────────────────────
    cv_freq = P.get("cv_freq", P.get("est_freq", "monthly"))
    cv_start = pd.Timestamp(P.get("cv_start", "1977-01-01"))
    update_freq = P["updateFreq"]
    updmonth = P.get("updmonth", 7) or 7
    T_est = P["T_est"]              # estimation window in months
    est_freq = P["est_freq"]        # 'monthly' or 'daily'

    # ── Factor / market data references ───────────────────────────
    ptf = P["PTF"]
    ptfd = P["PTFd"]
    pname = ptf.pname               # list of factor names
    N = len(pname)

    # Merge market returns INTO factor DataFrames (guarantees row-level
    # alignment everywhere — eliminates cross-DataFrame mask bugs).
    df_f = ptf.F.copy()
    df_f = df_f.merge(
        P["MKT"][["Date", "FF6_MktMinusRF"]].rename(
            columns={"FF6_MktMinusRF": _MKT_COL}
        ),
        on="Date", how="inner",
    ).reset_index(drop=True)

    df_fd = ptfd.F.copy()
    df_fd = df_fd.merge(
        P["MKTd"][["Date", "FF6_MktMinusRF"]].rename(
            columns={"FF6_MktMinusRF": _MKTd_COL}
        ),
        on="Date", how="inner",
    ).reset_index(drop=True)

    # Factor-only column names (everything except Date and the merged market col)
    _factor_cols = [c for c in df_f.columns if c not in ("Date", _MKT_COL)]
    _factor_cols_d = [c for c in df_fd.columns if c not in ("Date", _MKTd_COL)]

    # ── OOS date mask ─────────────────────────────────────────────
    oos_start = pd.Timestamp(P["oos_start"])
    oos_end = pd.Timestamp(P["oos_end"])

    oos_mask_f = (df_f["Date"] >= oos_start) & (df_f["Date"] <= oos_end)
    F_oos = df_f.loc[oos_mask_f].reset_index(drop=True)
    MKT_oos = F_oos[_MKT_COL].to_numpy(dtype=float)
    oos_dates = F_oos["Date"].values
    T_oos = len(oos_dates)

    # ── Parse factor-selection IDs ────────────────────────────────
    fctselid = P["fctselid"]
    fctselid_cv = P["fctselid_cv"]
    conditional_sel = parse_maxmin_suffix(fctselid)
    fctselid_complete = list(fctselid)
    # Strip suffix: 'GFDAIQ_max2' -> 'GFDAIQ'
    fctselid_clean = [re.sub(r"_.*", "", s) for s in fctselid]

    S = len(fctselid_clean)  # number of factor-set specifications

    # ── Selection matrices (boolean) ──────────────────────────────
    selmat = np.column_stack(
        [factorselection(pname, fctselid_clean[s]) for s in range(S)]
    )
    selcv = factorselection(pname, fctselid_cv[0])

    C = len(P["mptf_type"])  # number of portfolio types

    # ── Constraint matrices ───────────────────────────────────────
    CON = {}
    contab = P.get("CONTAB", None)
    for s in range(S):
        key = fctselid_clean[s]
        if contab is None:
            CON[key] = {"A": None, "bineq": None}
        else:
            # contab is list-of-lists, shape (Ncon, S)
            Ncon = len(contab)
            col_s = [contab[l][s] for l in range(Ncon)]
            has_valid = any(
                c is not None and not (isinstance(c, float) and np.isnan(c))
                for c in col_s
            )
            if has_valid:
                Nsel = int(selmat[:, s].sum())
                pname_sel = [pname[i] for i in range(N) if selmat[i, s]]
                A_s = np.zeros((Ncon, Nsel))
                bineq_s = np.zeros(Ncon)
                for l in range(Ncon):
                    A_s[l, :] = factorselection(pname_sel, col_s[l])
                CON[key] = {"A": -A_s.astype(float), "bineq": bineq_s}
            else:
                CON[key] = {"A": None, "bineq": None}

    # ── Initialize output containers ──────────────────────────────
    MPTF = {
        "pname": pname,
        "rets_oos": {},
        "rets_is": {},
        "wgt_oos": {},
        "wgt_is": {},
        "exanteSR_oos": {},
        "exanteMu_oos": {},
        "kappa_heu": [],
        "kappa_ddcsr2": [],
        "kappa_ddhjd": [],
        "kappa_heu_is": np.nan,
        "kappa_ddcsr2_is": np.nan,
        "kappa_ddhjd_is": np.nan,
    }

    for p in range(C):
        mtype = P["mptf_type"][p]
        MPTF["rets_oos"][mtype] = np.full((T_oos, S), np.nan)
        MPTF["rets_is"][mtype] = np.full((T_oos, S), np.nan)
        MPTF["wgt_oos"][mtype] = np.full((T_oos, N, S), np.nan)
        MPTF["wgt_is"][mtype] = np.full((1, N, S), np.nan)
        MPTF["exanteSR_oos"][mtype] = np.full((T_oos, S), np.nan)
        MPTF["exanteMu_oos"][mtype] = np.full((T_oos, S), np.nan)

    # Column labels for output DataFrames
    MPTF["fctselid_complete"] = fctselid_complete
    MPTF["oos_dates"] = oos_dates

    # ── Shared estimation-input dict ──────────────────────────────
    I = {
        "normalizewgt": P["normalizewgt"],
        "est_freq": est_freq,
    }

    # ── Persistent kappa values (carried forward between updates) ─
    kappa_heu_current = np.nan
    kappa_ddcsr2_current = np.nan
    kappa_ddhjd_current = np.nan
    upd_tun_pre = cv_start

    # ==============================================================
    #  OOS LOOP
    # ==============================================================
    for m in range(T_oos):
        if (m + 1) % 50 == 0 or m == 0:
            print(f"Date {m + 1} of {T_oos}")

        oos_date_m = pd.Timestamp(oos_dates[m])

        # ── Determine update / tuning dates ───────────────────────
        if update_freq == "monthly":
            upddate = oos_date_m
            upd_tun = previous_reb_gb(oos_date_m, updmonth)
        elif update_freq == "annual":
            upddate = previous_reb_gb(oos_date_m, updmonth)
            upd_tun = previous_reb_gb(oos_date_m, updmonth)
        elif update_freq == "custom":
            valid = [d for d in P["upddates"] if pd.Timestamp(d) <= oos_date_m]
            upddate = pd.Timestamp(valid[-1])
            upd_tun = pd.Timestamp(valid[-1])
        else:
            raise ValueError(f"Unknown updateFreq: {update_freq}")

        do_tune = _update_tuning(upd_tun, upd_tun_pre)
        upd_tun_pre = upd_tun

        # ── Estimation window dates ───────────────────────────────
        date_start_est = pd.Timestamp(upddate - relativedelta(months=T_est))
        date_end_est = pd.Timestamp(upddate - relativedelta(months=1))
        # Shift to end of month
        date_end_est = date_end_est + pd.offsets.MonthEnd(0)

        # Rolling masks (applied to df_f / df_fd which contain merged market)
        m_roll = (df_f["Date"] >= date_start_est) & (df_f["Date"] <= date_end_est)
        m_exp = (df_f["Date"] >= cv_start) & (df_f["Date"] <= date_end_est)
        d_roll = (df_fd["Date"] >= date_start_est) & (df_fd["Date"] <= date_end_est)

        # ── Estimation returns (Z) ────────────────────────────────
        # All data extracted from the SAME DataFrame → guaranteed alignment
        if est_freq == "daily":
            Z = df_fd.loc[d_roll, _factor_cols_d].to_numpy(dtype=float)
            mkt_est_roll = df_fd.loc[d_roll, _MKTd_COL].to_numpy(dtype=float)
        else:  # monthly
            Z = df_f.loc[m_roll, _factor_cols].to_numpy(dtype=float)
            mkt_est_roll = df_f.loc[m_roll, _MKT_COL].to_numpy(dtype=float)

        # Monthly Z for market-beta estimation (always monthly betas)
        Z_mon = df_f.loc[m_roll, _factor_cols].to_numpy(dtype=float)
        mkt_mon = df_f.loc[m_roll, _MKT_COL].to_numpy(dtype=float)

        # ── Market-beta estimation (OLS on monthly data) ──────────
        T_mon = len(mkt_mon)
        X_mkt = np.column_stack([np.ones(T_mon), mkt_mon])
        # mktbeta_est = (X'X)^{-1} X'Z, take row index 1 (slope)
        mktbeta_coeff = np.linalg.lstsq(X_mkt, Z_mon, rcond=None)[0]
        mktbeta_est = mktbeta_coeff[1, :]  # (N,)

        # ── Alpha t-stats via Newey-West ──────────────────────────
        T_est_obs = Z.shape[0]
        X_reg = np.column_stack([np.ones(T_est_obs), mkt_est_roll])
        alphatstat = np.zeros(N)
        for i in range(N):
            res = regrobustse(Z[:, i], X_reg, "nw")
            alphatstat[i] = res["tstat"][0]  # intercept t-stat

        # ── Market-beta hedge + leverage ──────────────────────────
        Zadj = Z - mktbeta_est[np.newaxis, :] * mkt_est_roll[:, np.newaxis]
        Lev = np.std(mkt_est_roll, ddof=1) / np.std(Zadj, axis=0, ddof=1)
        Zadj = Zadj * Lev[np.newaxis, :]

        # ── OOS single-period return (market-adjusted) ────────────
        Z_oos_raw = F_oos.loc[m, _factor_cols].to_numpy(dtype=float)  # (N,)
        Z_oos_adj = (Z_oos_raw - mktbeta_est * MKT_oos[m]) * Lev

        # ── Expanding-window market returns (for heuristic kappa) ─
        mkt_est_exp = df_f.loc[m_exp, _MKT_COL].to_numpy(dtype=float)
        I["mktrets"] = mkt_est_exp

        # ══════════════════════════════════════════════════════════
        # Loop over portfolio types
        # ══════════════════════════════════════════════════════════
        for p in range(C):
            mtype = P["mptf_type"][p]
            I["mptf_type"] = mtype

            # ── Kappa tuning (only on tuning-update dates) ────────
            if do_tune:
                if mtype == "bayes_kns_heu":
                    mu_mkt = np.mean(mkt_est_exp)
                    std_mkt = np.std(mkt_est_exp, ddof=1)
                    kappa_heu_current = 2.0 * (mu_mkt / std_mkt) * np.sqrt(12.0)
                    MPTF["kappa_heu"].append((oos_date_m, kappa_heu_current))

                if mtype in ("bayes_kns_dataCSR2", "bayes_kns_dataHJD"):
                    # Expanding window for CV
                    if cv_freq == "daily":
                        cv_mask = (df_fd["Date"] >= cv_start) & (
                            df_fd["Date"] <= date_end_est
                        )
                        Zcv = df_fd.loc[cv_mask, _factor_cols_d].to_numpy(
                            dtype=float
                        )
                        mkt_cv = df_fd.loc[cv_mask, _MKTd_COL].to_numpy(
                            dtype=float
                        )
                    else:
                        cv_mask = (df_f["Date"] >= cv_start) & (
                            df_f["Date"] <= date_end_est
                        )
                        Zcv = df_f.loc[cv_mask, _factor_cols].to_numpy(
                            dtype=float
                        )
                        mkt_cv = df_f.loc[cv_mask, _MKT_COL].to_numpy(
                            dtype=float
                        )

                    cv_otype = "CSR2" if mtype == "bayes_kns_dataCSR2" else "HJD"
                    kopt, _, _, _ = kns_tuning_kappa(
                        Zcv[:, selcv],
                        mkt_cv,
                        frequency=cv_freq,
                        train_years=T_est // 12,
                        val_years=5,
                        skip_years=1,
                        output_type=cv_otype,
                        verbose=False,
                    )
                    if mtype == "bayes_kns_dataCSR2":
                        kappa_ddcsr2_current = kopt
                        MPTF["kappa_ddcsr2"].append((oos_date_m, kopt))
                    else:
                        kappa_ddhjd_current = kopt
                        MPTF["kappa_ddhjd"].append((oos_date_m, kopt))

            # Pass current kappa values into I
            I["kappa_heu"] = kappa_heu_current
            I["kappa_ddcsr2"] = kappa_ddcsr2_current
            I["kappa_ddhjd"] = kappa_ddhjd_current

            # ── Loop over factor-set specifications ───────────────
            for s in range(S):
                sel_s = selmat[:, s].astype(bool)
                I["A"] = CON[fctselid_clean[s]]["A"]
                I["bineq"] = CON[fctselid_clean[s]]["bineq"]
                I["returns"] = Zadj[:, sel_s]
                I["alphatstat"] = alphatstat[sel_s]
                idFlg_s = sel_s.copy()

                # Conditional factor selection (max/min K)
                if abs(conditional_sel[s]) > 0.01 and contab is not None:
                    Ncon = len(contab)
                    K_adj = conditional_sel[s] / Ncon
                    pname_sel = [pname[i] for i in range(N) if sel_s[i]]
                    sel_alpha = pick_by_k(
                        I["alphatstat"], K_adj,
                        pname_sel, contab[0][s], contab[1][s],
                    )
                    # Update idFlg_s: among the True positions, keep only sel_alpha
                    true_idx = np.where(idFlg_s)[0]
                    idFlg_s[true_idx[~sel_alpha]] = False
                    I["returns"] = I["returns"][:, sel_alpha]
                    if I["A"] is not None:
                        I["A"] = I["A"][:, sel_alpha]

                # ── Compute weights ───────────────────────────────
                result = call_mptfweights(I)
                beta, post_SR, post_mu, Phi = result

                # ── Ex-ante SR (from same call — same dimensions) ─
                if post_mu is not None and Phi is not None:
                    denom = beta @ Phi @ beta
                    if denom > 0:
                        exante_sr = (beta @ post_mu / np.sqrt(denom)) * np.sqrt(252)
                        exante_mu = (beta @ post_mu) * 252
                    else:
                        exante_sr = np.nan
                        exante_mu = np.nan
                else:
                    exante_sr = np.nan
                    exante_mu = np.nan

                # ── Store OOS outputs ─────────────────────────────
                MPTF["wgt_oos"][mtype][m, idFlg_s, s] = beta
                MPTF["rets_oos"][mtype][m, s] = Z_oos_adj[idFlg_s] @ beta
                MPTF["exanteSR_oos"][mtype][m, s] = exante_sr
                MPTF["exanteMu_oos"][mtype][m, s] = exante_mu

    # ==============================================================
    #  IN-SAMPLE CALCULATION
    # ==============================================================
    print("Computing in-sample portfolios...")

    is_mask_m = (df_f["Date"] >= oos_start) & (df_f["Date"] <= oos_end)
    is_mask_m_exp = df_f["Date"] <= oos_end

    if est_freq == "daily":
        is_mask_d = (df_fd["Date"] >= oos_start) & (df_fd["Date"] <= oos_end)
        Z_is_est = df_fd.loc[is_mask_d, _factor_cols_d].to_numpy(dtype=float)
        mkt_is_est = df_fd.loc[is_mask_d, _MKTd_COL].to_numpy(dtype=float)
    else:
        Z_is_est = df_f.loc[is_mask_m, _factor_cols].to_numpy(dtype=float)
        mkt_is_est = df_f.loc[is_mask_m, _MKT_COL].to_numpy(dtype=float)

    # Market-beta (NW) on estimation-freq data
    T_is = Z_is_est.shape[0]
    X_is = np.column_stack([np.ones(T_is), mkt_is_est])
    mktbeta_is = np.full((2, N), np.nan)
    for i in range(N):
        res = regrobustse(Z_is_est[:, i], X_is, "nw")
        mktbeta_is[:, i] = res["coeff"]
    mktbeta_is_slope = mktbeta_is[1, :]

    # IS alphatstat: MATLAB uses the slope coefficients (not t-stats).
    # See MATLAB mptfcalculation.m IS section:
    #   mktbeta_est = mktbeta_est(2,:);      % overwrite to slopes
    #   alphatstat  = mktbeta_est(1,:)';      % this IS the slopes
    alphatstat_is = mktbeta_is_slope.copy()

    # Hedge + lever
    Zadj_is = Z_is_est - mktbeta_is_slope[np.newaxis, :] * mkt_is_est[:, np.newaxis]
    Lev_is = np.std(mkt_is_est, ddof=1) / np.std(Zadj_is, axis=0, ddof=1)
    Zadj_is = Zadj_is * Lev_is[np.newaxis, :]

    # IS returns (monthly, market-adjusted)
    Z_is_mon = df_f.loc[is_mask_m, _factor_cols].to_numpy(dtype=float)
    mkt_is_mon = df_f.loc[is_mask_m, _MKT_COL].to_numpy(dtype=float)
    Z_is_adj = (
        (Z_is_mon - mktbeta_is_slope[np.newaxis, :] * mkt_is_mon[:, np.newaxis])
        * Lev_is[np.newaxis, :]
    )

    mkt_is_exp = df_f.loc[is_mask_m_exp, _MKT_COL].to_numpy(dtype=float)
    I["mktrets"] = mkt_is_exp

    for p in range(C):
        mtype = P["mptf_type"][p]
        I["mptf_type"] = mtype

        # IS kappa: average of OOS kappas
        if mtype == "bayes_kns_heu" and len(MPTF["kappa_heu"]) > 0:
            MPTF["kappa_heu_is"] = np.mean([k[1] for k in MPTF["kappa_heu"]])
            I["kappa_heu"] = MPTF["kappa_heu_is"]
        if mtype == "bayes_kns_dataCSR2" and len(MPTF["kappa_ddcsr2"]) > 0:
            MPTF["kappa_ddcsr2_is"] = np.mean([k[1] for k in MPTF["kappa_ddcsr2"]])
            I["kappa_ddcsr2"] = MPTF["kappa_ddcsr2_is"]
        if mtype == "bayes_kns_dataHJD" and len(MPTF["kappa_ddhjd"]) > 0:
            MPTF["kappa_ddhjd_is"] = np.mean([k[1] for k in MPTF["kappa_ddhjd"]])
            I["kappa_ddhjd"] = MPTF["kappa_ddhjd_is"]

        for s in range(S):
            sel_s = selmat[:, s].astype(bool)
            I["A"] = CON[fctselid_clean[s]]["A"]
            I["bineq"] = CON[fctselid_clean[s]]["bineq"]
            I["returns"] = Zadj_is[:, sel_s]
            I["alphatstat"] = alphatstat_is[sel_s]
            idFlg_s = sel_s.copy()

            # Conditional selection
            if abs(conditional_sel[s]) > 0.01 and contab is not None:
                Ncon = len(contab)
                K_adj = conditional_sel[s] / Ncon
                pname_sel = [pname[i] for i in range(N) if sel_s[i]]
                sel_alpha = pick_by_k(
                    I["alphatstat"], K_adj,
                    pname_sel, contab[0][s], contab[1][s],
                )
                true_idx = np.where(idFlg_s)[0]
                idFlg_s[true_idx[~sel_alpha]] = False
                I["returns"] = I["returns"][:, sel_alpha]
                if I["A"] is not None:
                    I["A"] = I["A"][:, sel_alpha]

            result = call_mptfweights(I)
            beta = result[0] if isinstance(result, tuple) else result

            MPTF["wgt_is"][mtype][0, idFlg_s, s] = beta
            MPTF["rets_is"][mtype][:, s] = Z_is_adj[:, idFlg_s] @ beta

    print("Done.")
    return MPTF