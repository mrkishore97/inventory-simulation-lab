"""Demand-history reader for the Policy Advisor.

Reads the ``demand`` column out of a parquet — an M5 archetype's bundled history or a user-supplied
path — into a 1-D ``float64`` array for :func:`recommender.advisor.recommend`. Mirrors the parquet
contract that :class:`demand.empirical.EmpiricalDemand` enforces at engine init (a ``demand``
column, at least one row, all values finite and non-negative), but for the read-only advisor path:
no engine, no RNG, just load-and-validate. Streamlit-free, so it is unit-testable in isolation; the
page is the glue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


def read_demand_column(path: str | Path) -> npt.NDArray[np.float64]:
    """Load the ``demand`` column from the parquet at ``path`` as a 1-D ``float64`` array.

    Raises :class:`ValueError` (fail-loud, with a user-facing message) when the file has no
    ``demand`` column, is empty, or holds non-finite or negative values — the same contract
    :class:`EmpiricalDemand` applies, so a history that classifies here would run in the engine too.
    Parquet read errors (missing file, bad format) propagate from ``pandas`` / ``pyarrow``.
    """
    frame = pd.read_parquet(path)
    if "demand" not in frame.columns:
        raise ValueError("parquet has no 'demand' column")
    demand: npt.NDArray[np.float64] = np.asarray(frame["demand"].to_numpy(), dtype=np.float64)
    if demand.size == 0:
        raise ValueError("'demand' column is empty")
    if not np.isfinite(demand).all():
        raise ValueError("'demand' column has non-finite values (NaN or inf)")
    if (demand < 0.0).any():
        raise ValueError("'demand' column has negative values")
    return demand
