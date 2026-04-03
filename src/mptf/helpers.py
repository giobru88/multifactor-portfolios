# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:32:41 2026

@author: G_BRUNO
"""

import calendar
from datetime import date, datetime
from typing import Any, Union
import pandas as pd

def _to_date_any(d: Any) -> date:
    # Handles pandas Timestamp, numpy datetime64, strings, Python date/datetime, etc.
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()

    ts = pd.to_datetime(d, errors="raise")

    # If timezone-aware, drop tz (date comparison is calendar-based here)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)

    return ts.date()

def previous_reb_gb(d: Any, m: int) -> date:
    if not (1 <= m <= 12):
        raise ValueError("m must be an integer in [1, 12].")

    d_date = _to_date_any(d)
    y = d_date.year

    last_day = calendar.monthrange(y, m)[1]
    candidate = date(y, m, last_day)

    if candidate > d_date:
        y -= 1
        last_day = calendar.monthrange(y, m)[1]
        candidate = date(y, m, last_day)

    return pd.Timestamp(candidate)


_FREQ_MAP = {"daily": 252, "monthly": 12, "weekly": 52}

def parse_frequency(frequency: Union[str, int, float]) -> int:
    """Convert frequency specification to observations per year (F).

    Accepts 'daily' (252), 'monthly' (12), 'weekly' (52), or a positive int.
    """
    if isinstance(frequency, str):
        key = frequency.strip().lower()
        if key not in _FREQ_MAP:
            raise ValueError(f"Unknown frequency='{frequency}'. Use {list(_FREQ_MAP)}.")
        return _FREQ_MAP[key]
    F = int(frequency)
    if F <= 0:
        raise ValueError("Numeric frequency must be positive.")
    return F