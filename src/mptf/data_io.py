# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:30:49 2026

@author: G_BRUNO
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union, Optional, Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


DateLike = Union[str, pd.Timestamp, dt.datetime, np.datetime64, int, float]


@dataclass
class PtfReturns:
    F: pd.DataFrame           # merged returns table, Date + prefixed columns
    dates: pd.Series          # Date column (datetime64)
    pname: List[str]          # list of prefixed portfolio names (columns excluding Date)
    set: List[str]            # list of set labels, one per pname entry


def _matlab_datenum_to_datetime(dn: pd.Series) -> pd.Series:
    """
    Convert MATLAB datenum to pandas datetime.

    MATLAB datenum counts days since 0000-01-00 (effectively),
    and 1970-01-01 corresponds to datenum 719529.
    """
    dn = pd.to_numeric(dn, errors="coerce")
    # days since unix epoch:
    return pd.to_datetime(dn - 719529, unit="D", origin="unix")


def _parse_date_col(date_col: pd.Series, frequency: str) -> pd.Series:
    """
    Robust Date parsing:
    - datetime -> keep
    - numeric:
        * if looks like YYYYMMDD -> parse with format
        * else assume MATLAB datenum
    - string -> to_datetime
    Then if monthly -> month end.
    """
    if pd.api.types.is_datetime64_any_dtype(date_col):
        dt = pd.to_datetime(date_col)
    elif pd.api.types.is_numeric_dtype(date_col):
        x = pd.to_numeric(date_col, errors="coerce")

        # Heuristic: 8-digit integers like 19720131 => YYYYMMDD
        x_max = float(np.nanmax(x.values)) if len(x) else np.nan
        if x_max >= 10_000_000:  # 8+ digits
            dt = pd.to_datetime(x.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            # Typical in your MATLAB tables: ~720000 (MATLAB datenum)
            dt = _matlab_datenum_to_datetime(x)
    else:
        dt = pd.to_datetime(date_col, errors="coerce")

    if frequency.lower() == "monthly":
        dt = dt + MonthEnd(0)

    return dt


def _parse_date_input(x: DateLike, frequency: str) -> pd.Timestamp:
    """
    Accepts:
    - datetime-like strings ('2005-01-01')
    - pd.Timestamp
    - MATLAB datenum as int/float (e.g. 720471)
    - YYYYMMDD as int (e.g. 19720131)
    Returns Timestamp (shifted to month end if monthly).
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        if x >= 10_000_000:  # YYYYMMDD
            ts = pd.to_datetime(str(int(x)), format="%Y%m%d")
        else:  # assume MATLAB datenum
            ts = pd.to_datetime(float(x) - 719529, unit="D", origin="unix")
    else:
        ts = pd.to_datetime(x)

    if frequency.lower() == "monthly":
        ts = ts + MonthEnd(0)

    return ts


def _load_one_parquet(
    base_dir: Path,
    frequency: str,
    ptf_set: str,
    ptf_type: str,
) -> pd.DataFrame:
    """
    Load one parquet file <base_dir>/<ptf_set>/Portfolios_<ptf_type>_<frequency>.parquet
    and return a cleaned DataFrame with a proper Date column.
    """
    fpath = base_dir / ptf_set / f"Portfolios_{ptf_type}_{frequency}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing parquet file: {fpath}")

    df = pd.read_parquet(fpath)

    # Handle cases where Date came in as index or as an unnamed index column
    if "Date" not in df.columns:
        if df.index.name == "Date":
            df = df.reset_index()
        elif "__index_level_0__" in df.columns and "Date" in df.columns:  # very rare
            pass

    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {fpath}. Columns: {list(df.columns)}")

    df = df.copy()
    df["Date"] = _parse_date_col(df["Date"], frequency=frequency)

    # Drop rows with unparseable dates
    df = df.loc[df["Date"].notna()].sort_values("Date").reset_index(drop=True)

    return df


def getptfreturns_intprj(
    base_dir: Union[str, Path],
    frequency: str,
    ptf_set: Iterable[str],
    ptf_type: str,
    date_start: DateLike,
    date_end: DateLike,
) -> PtfReturns:
    """
    Python equivalent of MATLAB getptfreturns_intprj, but reading parquet.

    Parameters
    ----------
    base_dir : root folder that contains subfolders GFD / OAP / RF / FF6 ...
    frequency : 'daily' or 'monthly'
    ptf_set : list of set names (e.g., ['GFD','OAP'])
    ptf_type : e.g. 'ls'
    date_start, date_end : datetime-like or MATLAB datenum or YYYYMMDD

    Returns
    -------
    PtfReturns(F, dates, pname, set)
    """
    base_dir = Path(base_dir)
    frequency = frequency.lower()
    ptf_set = list(ptf_set)

    d0 = _parse_date_input(date_start, frequency)
    d1 = _parse_date_input(date_end, frequency)

    # --- Load first set
    Ftab = _load_one_parquet(base_dir, frequency, ptf_set[0], ptf_type)

    # Date range filter (inclusive)
    Ftab = Ftab.loc[(Ftab["Date"] >= d0) & (Ftab["Date"] <= d1)].copy()

    # Prefix columns
    first_cols = [c for c in Ftab.columns if c != "Date"]
    rename_map = {c: f"{ptf_set[0]}_{c}" for c in first_cols}
    Ftab.rename(columns=rename_map, inplace=True)

    pname: List[str] = list(rename_map.values())
    fct_set: List[str] = [ptf_set[0]] * len(pname)

    # --- Load and join additional sets (inner join on Date)
    for s in ptf_set[1:]:
        FCT = _load_one_parquet(base_dir, frequency, s, ptf_type)
        FCT = FCT.loc[(FCT["Date"] >= d0) & (FCT["Date"] <= d1)].copy()

        cols = [c for c in FCT.columns if c != "Date"]
        rmap = {c: f"{s}_{c}" for c in cols}
        FCT.rename(columns=rmap, inplace=True)

        pname.extend(list(rmap.values()))
        fct_set.extend([s] * len(rmap))

        Ftab = Ftab.merge(FCT, on="Date", how="inner")

    # Optional: enforce float64 on return columns (good for exact MATLAB-ish numerics)
    ret_cols = [c for c in Ftab.columns if c != "Date"]
    Ftab[ret_cols] = Ftab[ret_cols].astype("float64")

    return PtfReturns(
        F=Ftab,
        dates=Ftab["Date"],
        pname=pname,
        set=fct_set,
    )



