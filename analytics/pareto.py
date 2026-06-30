"""Pareto sweep — the service-vs-cost efficient frontier across a policy grid.

:func:`pareto_sweep`
runs the Monte Carlo runner at each point of a policy-parameter grid, reduces
each point to its mean KPIs, and flags the **efficient frontier** — the
non-dominated (low-cost, high-service) points. The M4 UI renders the cloud + the
frontier and lets users click a point to load its parameters.

Common random numbers: every grid point keeps ``config.master_seed``, so the Monte
Carlo runner derives the *same* per-replication seeds at every point — each policy
faces the same demand/lead-time realizations, which cancels sampling noise and
yields a cleaner, monotone frontier (the same-stream spirit).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from analytics.monte_carlo import monte_carlo
from core.config import RunConfig

_META_COLUMNS = ["replication", "seed"]


def pareto_sweep(
    config: RunConfig,
    param_grid: Mapping[str, Sequence[float | int]],
    n_replications: int = 100,
    n_jobs: int = -1,
    cost_kpi: str = "total_cost",
    service_kpi: str = "fill_rate",
) -> pd.DataFrame:
    """Sweep a policy-parameter grid; return per-point mean KPIs + the efficient frontier.

    ``param_grid`` maps policy-config field names to value lists; the sweep is their
    Cartesian product. For each grid point a policy variant is built and validated
    (``config.policy.model_validate(...)`` — field types, cross-field invariants like
    ``(s,S)``'s ``s < S``, and unknown-name rejection all fail loud), then
    :func:`analytics.monte_carlo.monte_carlo` runs ``n_replications`` seeded
    replications and the 12 KPIs are averaged.

    The returned frame has one row per grid point: the swept param columns, the 12
    KPI **means** (same names as :class:`analytics.kpis.KPIs`), and ``on_frontier`` —
    ``True`` for the non-dominated points (minimizing ``cost_kpi``, maximizing
    ``service_kpi``). Every grid point shares ``config.master_seed`` (common random
    numbers), so all points face the same demand/lead-time realizations.
    """
    if not param_grid:
        raise ValueError("param_grid must be non-empty")

    names = list(param_grid)
    rows: list[dict[str, float]] = []
    for combo in itertools.product(*(param_grid[name] for name in names)):
        updates = dict(zip(names, combo, strict=True))
        # model_validate re-runs full validation (field types, cross-field invariants,
        # extra="forbid"); model_copy is then safe because RunConfig has no
        # @model_validator coupling policy to other fields and the base is already valid.
        policy = config.policy.model_validate({**config.policy.model_dump(), **updates})
        variant = config.model_copy(update={"policy": policy})
        means = (
            monte_carlo(variant, n_replications, n_jobs=n_jobs).drop(columns=_META_COLUMNS).mean()
        )
        rows.append({**updates, **means.to_dict()})

    df = pd.DataFrame(rows)
    for col in (cost_kpi, service_kpi):
        if col not in df.columns:
            raise ValueError(f"{col!r} is not a KPI column; got {list(df.columns)}")
    df["on_frontier"] = _pareto_mask(df[cost_kpi].to_numpy(), df[service_kpi].to_numpy())
    return df


def _pareto_mask(
    cost: npt.NDArray[np.float64], service: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Non-dominated mask for minimize-cost / maximize-service.

    Point ``i`` is dominated when another point is no worse on both axes and strictly
    better on at least one. Tied points (equal cost AND equal service) are both kept —
    neither strictly dominates the other. O(n²), fine for grid-sized inputs.
    """
    on_frontier = np.ones(len(cost), dtype=np.bool_)
    for i in range(len(cost)):
        dominated = (
            (cost <= cost[i])
            & (service >= service[i])
            & ((cost < cost[i]) | (service > service[i]))
        )
        on_frontier[i] = not bool(dominated.any())
    return on_frontier
