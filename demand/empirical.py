"""Empirical demand — IID resampling from a stored parquet history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import EmpiricalDemandConfig
from demand.base import Demand


class EmpiricalDemand(Demand):
    """Empirical demand on a finite sample history (e.g., a curated M5 SKU).

    The first **non-parametric** arm of the demand chassis. Where prior
    arms (Normal, Poisson, NegBin, Gamma, Lognormal) carry parameters and
    call ``rng.normal(...)`` / ``rng.poisson(...)`` / etc. at each
    ``draw()``, EmpiricalDemand stores a sample **array** at
    construction time and calls ``rng.choice(history)`` per draw — the
    "distribution" is implicit in the data, not parameterized.

    Sampling semantics: **random IID** (``numpy.random.Generator.choice``).
    Each draw is independent of every other; serial correlation in the
    source history is NOT preserved. This is the locked chassis default;
    sequential replay / block-bootstrap variants may land as additional
    config fields in future bullets.

    Construction-time file I/O + validation. The :class:`EmpiricalDemandConfig`
    carries a ``history_path`` (relative to CWD); ``__init__`` reads the
    parquet, extracts the ``demand`` column, validates it, and stores
    the resulting ``np.ndarray`` on the instance. **Six failure modes**
    are raised loudly as ``ValueError`` (all locked by unit tests):

    1. **File not found** — ``history_path`` does not exist on disk.
    2. **Missing column** — parquet has no ``demand`` column.
    3. **Empty history** — parquet has zero rows.
    4. **NaN value** — any element is ``NaN``.
    5. **+inf / -inf** — any element is non-finite (other than NaN).
    6. **Negative value** — any element is < 0.

    No runtime clip in ``draw()`` — the history is pre-validated
    non-negative, so every draw is guaranteed ≥ 0 by construction.
    Distinct from :class:`NormalDemand` (which clips at draw time
    because Normal can produce negatives).

    Determinism: deterministic under ``(history, rng_seed)``. The
    config hash includes ``history_path`` but NOT the file contents,
    so mutating the bundled parquet between runs silently changes the
    demand stream without changing the hash. Users treat bundled
    parquets as immutable artifacts (a future bullet may add an
    optional ``history_sha256`` fingerprint field).

    Extra columns in the parquet are ignored (e.g., a ``date`` column
    is silently dropped).
    """

    def __init__(self, config: EmpiricalDemandConfig, rng: np.random.Generator) -> None:
        try:
            df = pd.read_parquet(config.history_path)
        except FileNotFoundError as exc:
            raise ValueError(f"Empirical demand history not found: {config.history_path}") from exc

        if "demand" not in df.columns:
            raise ValueError(
                f"Empirical demand parquet missing required 'demand' column "
                f"at {config.history_path}; got columns {list(df.columns)}"
            )

        history = df["demand"].to_numpy(dtype=np.float64)
        if len(history) == 0:
            raise ValueError(
                f"Empirical demand history is empty at {config.history_path}; need at least one row"
            )
        if not np.all(np.isfinite(history)):
            raise ValueError(
                f"Empirical demand history at {config.history_path} contains "
                f"non-finite values (NaN, +inf, or -inf)"
            )
        if not np.all(history >= 0.0):
            raise ValueError(
                f"Empirical demand history at {config.history_path} contains "
                f"negative values; demand must be non-negative"
            )

        self._history: np.ndarray = history
        self._rng: np.random.Generator = rng

    def draw(self) -> float:
        return float(self._rng.choice(self._history))
