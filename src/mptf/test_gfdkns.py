# -*- coding: utf-8 -*-
"""
MainV6_GFD_kns.py

Python replication of MATLAB MainV6_GFD_kns.m

Steps:
  1. Load data, build config
  2. Run mptfcalculation (bayes_kns_naive, monthly, with CONTAB constraints)
  3. Sharpe ratio test: H0: SR = 0 for each of the 4 factor sets
  4. Sharpe ratio difference test: H0: SR(GFDAIQ) = SR(benchmark)
     for benchmarks = {GFDSPRQFF, GFDSPRQHXZ, GFDAIQ_max2}

Usage:
    cd to your project root
    python MainV6_GFD_kns.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time

from mptf.data_io import getptfreturns_intprj
from mptf.mptf_calculation import mptfcalculation
from mptf.sharpe_test import boot_inference, boot_inference_diff_zero

# ═══════════════════════════════════════════════════════════════════
#  1) SETTINGS — mirror MATLAB MainV6_GFD_kns.m
# ═══════════════════════════════════════════════════════════════════

base_dir = Path(
    r"C:\Users\GiovanniBruno\OneDrive - SCIENTIFIC BETA SAS"
    r"\ResProj\AIQ\MPTF_Project\Portfolios"
)

ptf_set       = ["FF6", "GFD"]
ptf_type      = "ls"
full_start    = "1971-01-01"
full_end      = "2023-12-31"

mptf_type     = ["bayes_kns_dataHJD"]       # start with naive only
fctselid      = ["GFDSPRQFF", "GFDSPRQHXZ", "GFDAIQ_max2", "GFDAIQ"]
fctselid_cv   = ["GFDAIQ"]
oos_start     = "1996-07-01"
oos_end       = "2023-12-31"
T_est         = 120
normalizewgt  = "YES"
est_freq      = "daily"                  # monthly for speed; switch to 'daily' later
updateFreq    = "annual"
updmonth      = 7

CONTAB = [
    ["GFDPROFFF", "GFDPROFHXZ", "GFDPROFAIQ", "GFDPROFAIQ"],
    ["GFDINVFF",  "GFDINVHXZ",  "GFDINVAIQ",  "GFDINVAIQ"],
]

# Bootstrap settings (matching MATLAB)
b_block = 10                               # block size for circular block bootstrap
M_boot  = 4999                             # number of bootstrap repetitions
seed    = 42                               # for reproducibility (MATLAB uses parfor with no seed)

# Test setup (matching MATLAB)
testptfid      = "GFDAIQ"                  # portfolio to test
benchmarkselid = ["GFDSPRQFF", "GFDSPRQHXZ", "GFDAIQ_max2"]  # benchmarks

# ═══════════════════════════════════════════════════════════════════
#  2) LOAD DATA
# ═══════════════════════════════════════════════════════════════════

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

PTF.F = PTF.F.dropna().reset_index(drop=True)
PTF.dates = PTF.F["Date"].values
PTFd.dates = PTFd.F["Date"].values

MKT = getptfreturns_intprj(
    base_dir=str(base_dir), frequency="monthly",
    ptf_set=["FF6"], ptf_type=ptf_type,
    date_start=str(PTF.dates[0])[:10], date_end=str(PTF.dates[-1])[:10],
)
MKTd = getptfreturns_intprj(
    base_dir=str(base_dir), frequency="daily",
    ptf_set=["FF6"], ptf_type=ptf_type,
    date_start=str(PTFd.dates[0])[:10], date_end=str(PTFd.dates[-1])[:10],
)

print(f"Monthly: {len(PTF.F)} obs, {len(PTF.pname)} factors")
print(f"Daily:   {len(PTFd.F)} obs")

# ═══════════════════════════════════════════════════════════════════
#  3) BUILD CONFIG & RUN MPTFCALCULATION
# ═══════════════════════════════════════════════════════════════════

P = {
    "PTF":          PTF,
    "PTFd":         PTFd,
    "MKT":          MKT.F[["Date", "FF6_MktMinusRF"]].copy(),
    "MKTd":         MKTd.F[["Date", "FF6_MktMinusRF"]].copy(),
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

print(f"\n{'='*70}")
print(f"  Running mptfcalculation")
print(f"  mptf_types:  {mptf_type}")
print(f"  est_freq:    {est_freq}")
print(f"  fctselid:    {fctselid}")
print(f"{'='*70}\n")

t0 = time.time()
MPTF = mptfcalculation(P)
print(f"mptfcalculation completed in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════
#  4) DISPLAY OOS RESULTS
# ═══════════════════════════════════════════════════════════════════

oos_dates = MPTF["oos_dates"]
mtype = mptf_type[0]

print(f"\n{'='*70}")
print(f"  OOS RESULTS: {mtype}, est_freq={est_freq}")
print(f"  OOS period: {oos_dates[0]} to {oos_dates[-1]}")
print(f"{'='*70}")
print(f"\n{'Factor set':>20}  {'Mean':>8}  {'Std':>8}  {'SR':>8}  {'SR_ann':>8}")
print("-" * 60)

# Store returns for testing
oos_rets = {}
for s, fs in enumerate(fctselid):
    r = MPTF["rets_oos"][mtype][:, s]
    valid = ~np.isnan(r)
    rv = r[valid]
    oos_rets[fs] = rv

    mu_ann = np.mean(rv) * 12
    sd_ann = np.std(rv, ddof=1) * np.sqrt(12)
    sr_ann = mu_ann / sd_ann if sd_ann > 0 else np.nan
    sr_pm  = np.mean(rv) / np.std(rv, ddof=1)  # per-month SR (for reference)

    print(f"{fs:>20}  {mu_ann:8.4f}  {sd_ann:8.4f}  {sr_pm:8.4f}  {sr_ann:8.4f}")

# ═══════════════════════════════════════════════════════════════════
#  5) SHARPE RATIO TESTS: H0: SR = 0 (for each factor set)
#     Matches MATLAB:
#       [pValue, DeltaHat, ~, ~, se_sr, z_sr] = bootInference_diffZero(rets, b);
#       SR(row, col) = DeltaHat * sqrt(12);  % annualized
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  SHARPE RATIO TEST: H0: SR = 0")
print(f"  Bootstrap: b={b_block}, M={M_boot}")
print(f"{'='*70}")
print(f"\n{'Factor set':>20}  {'SR (ann)':>10}  {'SE':>10}  {'p-value':>10}  {'Signif':>8}")
print("-" * 65)

sr_results = {}
for fs in fctselid:
    t0 = time.time()
    pval, delta_hat, d, b_used, se = boot_inference_diff_zero(
        oos_rets[fs], b=b_block, M=M_boot, seed=seed
    )
    elapsed = time.time() - t0
    sr_ann = delta_hat * np.sqrt(12)
    se_ann = se * np.sqrt(12)

    signif = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    print(f"{fs:>20}  {sr_ann:10.4f}  {se_ann:10.4f}  {pval:10.4f}  {signif:>8}  ({elapsed:.1f}s)")

    sr_results[fs] = {"sr_ann": sr_ann, "se_ann": se_ann, "pval": pval}

# ═══════════════════════════════════════════════════════════════════
#  6) SHARPE RATIO DIFFERENCE TESTS:
#     H0: SR(GFDAIQ) - SR(benchmark) = 0
#     Matches MATLAB:
#       rets = [ptf, ben];
#       [pValue, DeltaHat, ~, ~, se_sr, z_sr] = bootInference(rets, b);
#       SR(row, col) = DeltaHat * sqrt(12);
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  SHARPE RATIO DIFFERENCE TEST: H0: SR({testptfid}) = SR(benchmark)")
print(f"  Bootstrap: b={b_block}, M={M_boot}")
print(f"{'='*70}")
print(f"\n{'Test vs Benchmark':>30}  {'SR_ptf':>8}  {'SR_ben':>8}  {'Diff':>8}  {'p-value':>10}  {'Signif':>8}")
print("-" * 85)

diff_results = {}
ptf_rets = oos_rets[testptfid]

for ben_id in benchmarkselid:
    ben_rets = oos_rets[ben_id]

    # Align lengths (should be identical, but just in case)
    T_min = min(len(ptf_rets), len(ben_rets))
    ret_2col = np.column_stack([ptf_rets[:T_min], ben_rets[:T_min]])

    t0 = time.time()
    pval, delta_hat, d, b_used, se = boot_inference(
        ret_2col, b=b_block, M=M_boot, seed=seed
    )
    elapsed = time.time() - t0

    sr_ptf_ann = sr_results[testptfid]["sr_ann"]
    sr_ben_ann = sr_results[ben_id]["sr_ann"]
    diff_ann   = delta_hat * np.sqrt(12)

    signif = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""

    label = f"{testptfid} vs {ben_id}"
    print(f"{label:>30}  {sr_ptf_ann:8.4f}  {sr_ben_ann:8.4f}  {diff_ann:8.4f}  {pval:10.4f}  {signif:>8}  ({elapsed:.1f}s)")

    diff_results[ben_id] = {"diff_ann": diff_ann, "pval": pval}

# ═══════════════════════════════════════════════════════════════════
#  7) SUMMARY TABLE (matching MATLAB SR matrix layout)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  SUMMARY TABLE (MATLAB SR matrix format)")
print(f"  Row 1: annualized SR or SR difference")
print(f"  Row 2: bootstrap p-value")
print(f"{'='*70}")

# Build the table matching MATLAB's layout:
# Column 0: SR of test portfolio (GFDAIQ)
# Columns 1-2: SR of benchmark / SR diff for GFDSPRQFF
# Columns 3-4: SR of benchmark / SR diff for GFDSPRQHXZ
# Columns 5-6: SR of benchmark / SR diff for GFDAIQ_max2

header = f"{'':>12}  {testptfid:>12}"
for ben_id in benchmarkselid:
    header += f"  {ben_id:>12}  {'Diff':>12}"
print(header)
print("-" * len(header))

# Row 1: values
row1 = f"{'SR/Diff':>12}  {sr_results[testptfid]['sr_ann']:12.4f}"
for ben_id in benchmarkselid:
    row1 += f"  {sr_results[ben_id]['sr_ann']:12.4f}  {diff_results[ben_id]['diff_ann']:12.4f}"
print(row1)

# Row 2: p-values
row2 = f"{'p-value':>12}  {sr_results[testptfid]['pval']:12.4f}"
for ben_id in benchmarkselid:
    row2 += f"  {sr_results[ben_id]['pval']:12.4f}  {diff_results[ben_id]['pval']:12.4f}"
print(row2)

print(f"\n{'='*70}")
print("Done.")
print(f"{'='*70}")