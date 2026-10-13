"""Timestamp helpers.

pandas 3 picks the resolution of a datetime column from its source
(microseconds by default, nanoseconds from some constructors), and
operations such as merge_asof refuse to join columns of different
resolutions; int64 casts silently change meaning with the unit. Everything
that joins or converts timestamps goes through here.
"""

from __future__ import annotations

import pandas as pd


def utc_us(s: pd.Series) -> pd.Series:
    """Timezone-aware UTC datetimes at microsecond resolution."""
    return pd.to_datetime(s, utc=True).dt.as_unit("us")
