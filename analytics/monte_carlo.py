"""Monte Carlo runner — full KPI distributions across seeded replications.

:func:`monte_carlo` runs one :class:`~core.config.RunConfig`
``n_replications`` times, each with an independent deterministic seed, and returns
a DataFrame of per-replication KPIs — the full outcome distribution that the risk
module (VaR/CVaR) and the M4 violin/box plots consume.

Per-replication seeds are spawned from ``SeedSequence(config.master_seed)`` — the
same cascade the engine's ``SeedManager`` uses (``core/seeding.py``), one level up:
spawn at the *replication* level, then each replication's engine spawns its own
component streams. Because a replication's seed is fixed by its index (not by
execution order), the result is deterministic and **independent of ``n_jobs``** — a
parallel run equals the serial run. Reusing :func:`experiments.single_run.run`
keeps the engine untouched (the per-rep seed rides a ``config.model_copy``), so the
runner is non-invasive.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analytics.kpis import compute_kpis
from core.config import RunConfig
from experiments import single_run


def monte_carlo(config: RunConfig, n_replications: int, n_jobs: int = -1) -> pd.DataFrame:
    """Run ``config`` ``n_replications`` times under independent seeds; return per-rep KPIs.

    Returns a DataFrame with one row per replication and columns ``replication``,
    ``seed``, and the 12 KPI fields (see :class:`analytics.kpis.KPIs`). The KPI
    columns are the full outcome distribution — summary stats, VaR/CVaR, and the
    distribution plots are computed downstream from this frame, never collapsed
    here. Each row's ``seed`` reproduces that replication exactly via
    ``single_run.run(config.model_copy(update={"master_seed": seed}))``.

    Per-replication seeds are spawned from ``config.master_seed`` (the same
    SeedSequence cascade as :class:`core.seeding.SeedManager`), so two runs with the
    same config produce identical frames and the result is independent of ``n_jobs``
    (a rep's seed is fixed by its index, not by execution order). ``n_jobs`` is
    forwarded to :class:`joblib.Parallel` (default ``-1`` = all cores).
    """
    if n_replications < 1:
        raise ValueError(f"n_replications must be >= 1, got {n_replications}")

    parent = np.random.SeedSequence(config.master_seed)
    # 53-bit seeds: high-quality + independent, fit a clean int64 column, AND stay
    # exactly representable as float64 — so the seed survives a DataFrame row Series
    # (df.iloc[i]) upcast to float and still reproduces its replication.
    seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0]) >> 11
        for child in parent.spawn(n_replications)
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_replication)(config, i, seed) for i, seed in enumerate(seeds)
    )
    return pd.DataFrame(results)


def _run_replication(config: RunConfig, replication: int, seed: int) -> dict[str, float | int]:
    """Run one replication at ``seed``; return its KPIs plus ``replication``/``seed`` metadata."""
    frame = single_run.run(config.model_copy(update={"master_seed": seed}))
    return {"replication": replication, "seed": seed, **asdict(compute_kpis(frame))}
