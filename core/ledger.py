"""Pre-allocated, append-free run buffer for one simulation run.

The :class:`Ledger` is the single sink for per-period state. The engine writes
one row per period, but never appends to a Pandas DataFrame inside the inner
loop — that pattern is O(n²) and would break the 100 ms single-run target at
horizon=365. Instead, the Ledger pre-allocates one
NumPy array per declared column, exposes an O(1) :meth:`record`, and
materializes a DataFrame once in :meth:`to_dataframe` at end-of-run.

The schema is fixed at class level. The engine writes to known columns; an
unknown name fails loudly with :class:`KeyError`. Adding a new column requires
extending ``_SCHEMA`` *at the end* so existing column-position tests stay
stable.

The Ledger is dumb storage. It does not enforce the cost accounting
conventions or within-period event sequence —
those are engine contracts, verified by integration tests when M1 bullet 5
lands the engine.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd


class Ledger:
    _SCHEMA: Final[tuple[str, ...]] = (
        "on_hand",
        "on_order",
        "inventory_position",
        "demand",
        "sales",
        "backorders",
        "lost_sales",
        "backorders_cleared",
        "order_placed",
        "order_received",
        "holding_cost",
        "ordering_cost",
        "purchase_cost",
        "stockout_cost",
    )
    _SCHEMA_SET: Final[frozenset[str]] = frozenset(_SCHEMA)

    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self._horizon = horizon
        self._buffers: dict[str, npt.NDArray[np.float64]] = {
            name: np.zeros(horizon, dtype=np.float64) for name in self._SCHEMA
        }

    def record(self, t: int, **fields: float) -> None:
        if not 0 <= t < self._horizon:
            raise IndexError(f"period {t} out of range for horizon {self._horizon}")
        for name, value in fields.items():
            if name not in self._SCHEMA_SET:
                raise KeyError(
                    f"unknown ledger field {name!r}. Known fields: {sorted(self._SCHEMA_SET)}"
                )
            self._buffers[name][t] = value

    def row(self, t: int) -> dict[str, float]:
        """Return all column values at period t as a dict.

        Strict indexing: negative t or t >= horizon raises IndexError. No
        ``-1`` lazy lookup; callers must compute the period explicitly.
        """
        if not 0 <= t < self._horizon:
            raise IndexError(f"period {t} out of range for horizon {self._horizon}")
        return {name: float(self._buffers[name][t]) for name in self._SCHEMA}

    def to_dataframe(self) -> pd.DataFrame:
        data: dict[str, npt.NDArray[Any]] = {"period": np.arange(self._horizon, dtype=np.int64)}
        for name in self._SCHEMA:
            data[name] = self._buffers[name].copy()
        return pd.DataFrame(data)
