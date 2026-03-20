# -*- coding: utf-8 -*-
"""
Test mptf_calculation.py with constraints (CONTAB).
Mirrors MATLAB's MainV6_GFD_kns_MptfOnly.m settings.

Start with bayes_kns_naive only + monthly est_freq for speed.
Once validated, switch to est_freq='daily' and add more mptf_types.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from mptf.data_io import getptfreturns_intprj
from mptf.mptf_calculation import mptfcalculation

# ══════════════════════════════════════════════════════════════════
#  1) SETTINGS — mirror MATLAB's MainV6_GFD_kns_MptfOnly.m
# ══════════════════════════════════════════════════════════════════

base_dir    = Path(r"C:\Users\GiovanniBruno\OneDrive - SCIENTIFIC BETA SAS"
                   r"\ResProj\AIQ\MPTF_Project\Portfolios")

ptf_set     = ["FF6", "GFD"]
ptf_type    = "ls"
full_start  = "1971-01-01"
full_end    = "2023-12-31"

# ── Full MATLAB settings ─────────────────────────────────────────
mptf_type   = ["bayes_kns_naive"]           # start with naive only (fast)
# mptf_type = ["bayes_kns_naive", "bayes_kns_heu", "bayes_kns_dataHJD",
#              "bayes_kns_dataCSR2", "bayes_kns_extsmall"]  # full set

fctselid    = ["GFDSPRQFF", "GFDSPRQHXZ", "GFDAIQ_max2", "GFDAIQ"]
fctselid_cv = ["GFDAIQ"]
oos_start   = "1996-07-01"
oos_end     = "2023-12-31"
T_est       = 120
normalizewgt = "YES"
est_freq    = "monthly"                     # use 'daily' to match MATLAB exactly
updateFreq  = "annual"
updmonth    = 7

# ── CONTAB: constraint table (Ncon=2 rows × S=4 columns) ─────────
# MATLAB: [{'GFDPROFFF';'GFDINVFF'}, {'GFDPROFHXZ';'GFDINVHXZ'},
#           {'GFDPROFAIQ';'GFDINVAIQ'}, {'GFDPROFAIQ';'GFDINVAIQ'}]
#
# Python: list of lists, indexed as contab[row][col]
#   contab[0][s] = profitability group for factor set s
#   contab[1][s] = investment group for factor set s
CONTAB = [
    ["GFDPROFFF", "GFDPROFHXZ", "GFDPROFAIQ", "GFDPROFAIQ"],  # row 0
    ["GFDINVFF",  "GFDINVHXZ",  "GFDINVAIQ",  "GFDINVAIQ"],   # row 1
]

# ══════════════════════════════════════════════════════════════════
#  2) LOAD DATA
# ══════════════════════════════════════════════════════════════════

print("Loading monthly returns...")
PTF = getptfreturns_intprj(
    base_dir=str(base_dir), frequency="monthly",
    ptf_set=ptf_set, ptf_type=ptf_type,
    date_start=full_start, date_end=full_end,
)

print("Loading daily returns...")
PTFd = getptfreturns_intprj(
    base_dir=str(base_dir), frequency="daily",
    ptf_set=ptf_set, ptf_type=ptf_type,
    date_start=full_start, date_end=full_end,
)

# Drop rows with any NaN (matches MATLAB rmmissing on monthly only)
PTF.F = PTF.F.dropna().reset_index(drop=True)
PTF.dates = PTF.F["Date"].values

# Daily dates from the DAILY DataFrame (not monthly!)
PTFd.dates = PTFd.F["Date"].values

print(f"Monthly: {len(PTF.F)} obs, {len(PTF.pname)} factors")
print(f"Daily:   {len(PTFd.F)} obs")
print(f"Monthly sample: {PTF.F['Date'].iloc[0]} to {PTF.F['Date'].iloc[-1]}")
print(f"Daily sample:   {PTFd.F['Date'].iloc[0]} to {PTFd.F['Date'].iloc[-1]}")

# Market returns — monthly aligned to PTF dates, daily to PTFd dates
MKT = getptfreturns_intprj(
    base_dir=str(base_dir), frequency="monthly",
    ptf_set=["FF6"], ptf_type=ptf_type,
    date_start=str(PTF.dates[0])[:10], date_end=str(PTF.dates[-1])[:10],
)
MKT_df = MKT.F[["Date", "FF6_MktMinusRF"]].copy()

MKTd = getptfreturns_intprj(
    base_dir=str(base_dir), frequency="daily",
    ptf_set=["FF6"], ptf_type=ptf_type,
    date_start=str(PTFd.dates[0])[:10], date_end=str(PTFd.dates[-1])[:10],
)
MKTd_df = MKTd.F[["Date", "FF6_MktMinusRF"]].copy()

print(f"MKT monthly: {len(MKT_df)} obs")
print(f"MKT daily:   {len(MKTd_df)} obs")

# ══════════════════════════════════════════════════════════════════
#  3) BUILD CONFIG DICT P
# ══════════════════════════════════════════════════════════════════

P = {
    "PTF":          PTF,
    "PTFd":         PTFd,
    "MKT":          MKT_df,
    "MKTd":         MKTd_df,
    "mptf_type":    mptf_type,
    "fctselid":     fctselid,
    "fctselid_cv":  fctselid_cv,
    "oos_start":    oos_start,
    "oos_end":      oos_end,
    "T_est":        T_est,
    "normalizewgt": normalizewgt,
    "est_freq":     est_freq,
    "updateFreq":   updateFreq,
    "updmonth":     updmonth,
    "CONTAB":       CONTAB,
    "cv_start":     "1977-01-01",
}

# ══════════════════════════════════════════════════════════════════
#  4) RUN
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"  Running mptfcalculation")
print(f"  mptf_types:  {mptf_type}")
print(f"  fctselid:    {fctselid}")
print(f"  est_freq:    {est_freq}")
print(f"  CONTAB:      {len(CONTAB)} constraints × {len(CONTAB[0])} sets")
print("=" * 60 + "\n")

MPTF = mptfcalculation(P)

# ══════════════════════════════════════════════════════════════════
#  5) INSPECT RESULTS
# ══════════════════════════════════════════════════════════════════

oos_dates = MPTF["oos_dates"]
T_oos_total = len(oos_dates)
print(f"\nOOS period: {oos_dates[0]} to {oos_dates[-1]}  ({T_oos_total} months)")
print(f"Factor sets: {MPTF['fctselid_complete']}")

for mtype in mptf_type:
    print(f"\n{'─' * 60}")
    print(f"  Portfolio type: {mtype}")
    print(f"{'─' * 60}")

    for s, fsel in enumerate(fctselid):
        rets_oos = MPTF["rets_oos"][mtype][:, s]
        rets_is  = MPTF["rets_is"][mtype][:, s]

        valid_oos = ~np.isnan(rets_oos)
        valid_is  = ~np.isnan(rets_is)

        print(f"\n  Factor set: {fsel}")
        print(f"    Valid OOS: {valid_oos.sum()} / {T_oos_total}")
        print(f"    Valid IS:  {valid_is.sum()} / {len(rets_is)}")

        if valid_oos.sum() > 0:
            r = rets_oos[valid_oos]
            mu = np.mean(r) * 12
            sd = np.std(r, ddof=1) * np.sqrt(12)
            sr = mu / sd if sd > 0 else np.nan
            print(f"    OOS  Mean={mu:.4f}  Std={sd:.4f}  SR={sr:.4f}")

        if valid_is.sum() > 0:
            r = rets_is[valid_is]
            mu = np.mean(r) * 12
            sd = np.std(r, ddof=1) * np.sqrt(12)
            sr = mu / sd if sd > 0 else np.nan
            print(f"    IS   Mean={mu:.4f}  Std={sd:.4f}  SR={sr:.4f}")

        # Weights summary at first valid OOS date
        wgt = MPTF["wgt_oos"][mtype]
        if valid_oos.sum() > 0:
            first_valid = valid_oos.argmax()
            w = wgt[first_valid, :, s]
            n_nonzero = np.sum(np.abs(w) > 1e-10)
            print(f"    Weights (first valid): sum|w|={np.nansum(np.abs(w)):.4f}, "
                  f"nonzero={n_nonzero}")

print(f"\n{'═' * 60}")
print("Done.")