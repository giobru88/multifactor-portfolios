# -*- coding: utf-8 -*-
"""
selection_helpers.py

Conditional factor-selection utilities ported from nested MATLAB functions
inside mptfcalculation.m:
  - parse_maxmin_suffix  →  parse_maxmin_suffix()
  - pick_by_K            →  pick_by_k()

Author: G_BRUNO (ported to Python)
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .factor_selection import factorselection


# --------------------------------------------------------------------------- #
#  parse_maxmin_suffix
# --------------------------------------------------------------------------- #

_SUFFIX_RE = re.compile(r"_(max|min)(\d+)$", re.IGNORECASE)


def parse_maxmin_suffix(
    fctselid: Union[str, Sequence[str]],
) -> np.ndarray:
    """
    Port of MATLAB ``parse_maxmin_suffix(fctselid)``.

    For each element in *fctselid*, extract the trailing ``_maxK`` or
    ``_minK`` suffix and return a signed integer:

    * ``_maxK``  →  +K
    * ``_minK``  →  -K
    * no match   →   0

    Parameters
    ----------
    fctselid : str or sequence of str
        Factor-selection IDs, e.g. ``["GFDAIQ_max2", "GFDSPRQFF"]``.

    Returns
    -------
    kvec : np.ndarray, shape (len(fctselid),)
        Signed integers.

    Examples
    --------
    >>> parse_maxmin_suffix(["GFDAIQ_max2", "GFDSPRQFF", "OAPAIQ_min3"])
    array([ 2,  0, -3])
    """
    if isinstance(fctselid, str):
        fctselid = [fctselid]

    kvec = np.zeros(len(fctselid), dtype=int)
    for i, s in enumerate(fctselid):
        m = _SUFFIX_RE.search(s)
        if m:
            sign = 1 if m.group(1).lower() == "max" else -1
            kvec[i] = sign * int(m.group(2))
    return kvec


def clean_suffix(fctselid: Union[str, Sequence[str]]) -> List[str]:
    """
    Strip ``_maxK`` / ``_minK`` suffixes, returning the "clean" IDs.

    Port of MATLAB ``regexprep(fctselid, '_.*', '')``.

    Parameters
    ----------
    fctselid : str or sequence of str

    Returns
    -------
    list of str
        Cleaned IDs, e.g. ``["GFDAIQ_max2"]`` → ``["GFDAIQ"]``.

    Notes
    -----
    The MATLAB code uses ``regexprep(fctselid, '_.*', '')`` which strips
    everything after the *first* underscore.  However the actual intent
    (confirmed by usage) is to strip only the ``_max/_min`` tail, since
    IDs like ``GFDSPRQFF`` have no underscore and IDs like ``GFDAIQ_max2``
    should become ``GFDAIQ``.  We replicate the MATLAB regex exactly:
    remove from the first underscore onward.
    """
    if isinstance(fctselid, str):
        fctselid = [fctselid]
    # MATLAB: regexprep(fctselid, '_.*', '')  →  strip from first '_'
    return [s.split("_")[0] if "_" in s else s for s in fctselid]


# --------------------------------------------------------------------------- #
#  pick_by_k  (two modes)
# --------------------------------------------------------------------------- #

def _pick_topk_overall(
    alphatstat: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Mode 1 (original MATLAB behavior with 2 arguments):

    * ``K == 0``  →  all True
    * ``K > 0``   →  top-K by largest value
    * ``K < 0``   →  bottom-|K| by smallest value

    NaNs are never selected.

    Returns boolean mask of length ``len(alphatstat)``.
    """
    n = len(alphatstat)
    sel = np.zeros(n, dtype=bool)

    if k == 0:
        sel[:] = True
        return sel

    valid = ~np.isnan(alphatstat)
    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        return sel

    # Sort ascending (NaNs already excluded)
    order = np.argsort(alphatstat[idx_valid])  # ascending
    idx_sorted = idx_valid[order]

    k_eff = min(abs(k), len(idx_sorted))
    if k > 0:
        chosen = idx_sorted[-k_eff:]           # largest
    else:
        chosen = idx_sorted[:k_eff]            # smallest

    sel[chosen] = True
    return sel


def _pick_topk_within_group(
    alphatstat: np.ndarray,
    mask: np.ndarray,
    k: int,
    exclude: np.ndarray,
) -> np.ndarray:
    """
    Pick top-*k* indices within *mask* (ignoring NaNs and *exclude*).

    Returns 1-D integer array of chosen indices (may be empty).
    """
    if k <= 0:
        return np.array([], dtype=int)

    candidates = np.where(mask & ~np.isnan(alphatstat) & ~exclude)[0]
    if candidates.size == 0:
        return np.array([], dtype=int)

    vals = alphatstat[candidates]
    order = np.argsort(vals)[::-1]             # descending
    k_eff = min(k, len(order))
    return candidates[order[:k_eff]]


def pick_by_k(
    alphatstat: np.ndarray,
    k: Union[int, float],
    pname: Optional[Union[Sequence[str], pd.Index, np.ndarray]] = None,
    group1: Optional[str] = None,
    group2: Optional[str] = None,
) -> np.ndarray:
    """
    Port of MATLAB ``pick_by_K(alphatstat, K [, pname, group1, group2])``.

    **Mode 1** (2 arguments — ``pname``, ``group1``, ``group2`` all None):

    * ``K == 0``  →  select all
    * ``K > 0``   →  top-K overall
    * ``K < 0``   →  bottom-|K| overall

    **Mode 2** (5 arguments):

    Select the *K* largest ``alphatstat`` **within group1** and the *K*
    largest **within group2**.  Group membership is determined by
    ``factorselection(pname, group_id)``.  Overlaps are assigned to
    group1 first; group2 then fills remaining slots (excluding indices
    already selected by group1).

    Parameters
    ----------
    alphatstat : array-like, shape (N,)
        Alpha t-statistics (or any ranking metric) for each factor.
    k : int or float
        Number of factors to pick per group (Mode 2) or overall (Mode 1).
        Converted to ``int(round(...))``.
    pname : sequence of str, optional
        Factor names aligned with *alphatstat*.  Required for Mode 2.
    group1, group2 : str, optional
        Selection IDs passed to ``factorselection()``.  Required for Mode 2.

    Returns
    -------
    sel : np.ndarray[bool], shape (N,)
        Boolean selection mask.

    Notes
    -----
    * NaNs in *alphatstat* are never selected.
    * If K exceeds the number of valid elements in a group, it caps silently.

    Examples
    --------
    >>> # Mode 1 – top 3 overall
    >>> pick_by_k(np.array([1.2, 0.5, 2.3, 1.8, np.nan]), k=3)
    array([True, False, True, True, False])

    >>> # Mode 2 – top 2 per group
    >>> pick_by_k(tstats, k=2, pname=names, group1="GFDPROFAIQ", group2="GFDINVAIQ")
    """
    alphatstat = np.asarray(alphatstat, dtype=float).ravel()
    k = int(round(k))
    n = len(alphatstat)

    # ── Mode 1: simple top/bottom-K ──────────────────────────────────
    if pname is None and group1 is None and group2 is None:
        return _pick_topk_overall(alphatstat, k)

    # ── Mode 2: two-group selection ──────────────────────────────────
    if pname is None or group1 is None or group2 is None:
        raise ValueError(
            "pick_by_k Mode 2 requires all of: pname, group1, group2."
        )
    if k < 0:
        raise ValueError("For two-group mode, k must be >= 0.")

    # Convert pname to list for factorselection
    if isinstance(pname, np.ndarray):
        pname_list = pname.tolist()
    elif isinstance(pname, pd.Index):
        pname_list = pname.tolist()
    else:
        pname_list = list(pname)

    if len(pname_list) != n:
        raise ValueError(
            f"pname length ({len(pname_list)}) must match "
            f"alphatstat length ({n})."
        )

    g1 = factorselection(pname_list, group1)  # bool mask
    g2 = factorselection(pname_list, group2)  # bool mask

    sel = np.zeros(n, dtype=bool)

    # First: pick K from group1
    chosen1 = _pick_topk_within_group(alphatstat, g1, k, exclude=np.zeros(n, dtype=bool))
    sel[chosen1] = True

    # Then: pick K from group2, excluding anything already taken by group1
    chosen2 = _pick_topk_within_group(alphatstat, g2, k, exclude=sel)
    sel[chosen2] = True

    return sel
