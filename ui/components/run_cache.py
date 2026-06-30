"""Cached run helpers — config → results frame, memoized on the canonical JSON.

Streamlit reruns the whole page script on every interaction, so a naive page would
re-simulate on each widget change. :func:`run_cached` wraps :func:`experiments.single_run.run`
in ``st.cache_data`` keyed on ``config.to_json()`` — the canonical sorted JSON that *is* the
content identity behind ``RunConfig.config_hash`` (a ``RunConfig`` object is not hashable for
``st.cache_data``; the JSON string is, and reusing it keeps the cache key aligned with the
reproducibility hash). Only the expensive ``single_run.run`` is cached; ``compute_kpis``
is a cheap pure reduction and stays uncached.

:func:`sweep_cached` (the Pareto Explorer) applies the same JSON-key pattern to the much
costlier :func:`analytics.pareto.pareto_sweep`: it adds the param grid (canonical JSON) and the
axis/replication knobs to the cache key. ``n_jobs`` is fixed to ``-1`` (and kept out of the key)
because the runner is parallel-invariant — the number of workers must not change the result or
fragment the cache.

:func:`mc_cached` (the Monte Carlo & Risk page) is the same pattern over
:func:`analytics.monte_carlo.monte_carlo`, keyed on the config JSON + the replication count
(``n_jobs=-1`` likewise fixed and out of the key). The page then re-derives ``risk_metrics`` over
the cached per-replication frame on every live control change — a pure reduction, no re-run.

:func:`sensitivity_cached` (the Sensitivity page) is the same pattern over
:func:`analytics.sensitivity.sensitivity_oat`: it adds the ``{path: (low, high)}`` spec (canonical
JSON, preserving each value's int/float distinction) and the output-KPI + replication count to the
key (``n_jobs=-1`` fixed and out of the key, as ever).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from analytics.monte_carlo import monte_carlo
from analytics.pareto import pareto_sweep
from analytics.sensitivity import sensitivity_oat
from core.config import RunConfig
from experiments import single_run


@st.cache_data(show_spinner="Running simulation…")
def _cached_run(config_json: str) -> pd.DataFrame:
    return single_run.run(RunConfig.from_json(config_json))


def run_cached(config: RunConfig) -> pd.DataFrame:
    """Run ``config`` (memoized on its canonical JSON) and return the Ledger frame."""
    result: pd.DataFrame = _cached_run(config.to_json())
    return result


@st.cache_data(show_spinner="Sweeping policy grid…")
def _cached_sweep(
    config_json: str, grid_json: str, n_replications: int, cost_kpi: str, service_kpi: str
) -> pd.DataFrame:
    return pareto_sweep(
        RunConfig.from_json(config_json),
        json.loads(grid_json),
        n_replications=n_replications,
        n_jobs=-1,
        cost_kpi=cost_kpi,
        service_kpi=service_kpi,
    )


def sweep_cached(
    config: RunConfig,
    param_grid: Mapping[str, Sequence[float | int]],
    n_replications: int,
    cost_kpi: str = "total_cost",
    service_kpi: str = "fill_rate",
) -> pd.DataFrame:
    """Pareto-sweep ``param_grid`` for ``config`` (memoized), returning the per-point frame.

    The grid is canonicalized to sorted JSON for the cache key (which also preserves the int /
    float distinction of each grid value, so an int field stays integer-valued). See
    :func:`analytics.pareto.pareto_sweep` for the returned frame's columns.
    """
    grid_json = json.dumps({k: list(v) for k, v in param_grid.items()}, sort_keys=True)
    result: pd.DataFrame = _cached_sweep(
        config.to_json(), grid_json, n_replications, cost_kpi, service_kpi
    )
    return result


@st.cache_data(show_spinner="Running Monte Carlo…")
def _cached_mc(config_json: str, n_replications: int) -> pd.DataFrame:
    return monte_carlo(RunConfig.from_json(config_json), n_replications, n_jobs=-1)


def mc_cached(config: RunConfig, n_replications: int) -> pd.DataFrame:
    """Monte-Carlo ``config`` for ``n_replications`` (memoized), returning the per-rep KPI frame.

    The frame (one row per replication: ``replication`` / ``seed`` + the 12 KPIs) is the input to
    both :func:`analytics.risk.risk_metrics` and the distribution histogram; the page re-reduces it
    live without re-running the simulation.
    """
    result: pd.DataFrame = _cached_mc(config.to_json(), n_replications)
    return result


@st.cache_data(show_spinner="Running sensitivity…")
def _cached_sensitivity(
    config_json: str, specs_json: str, n_replications: int, output_kpi: str
) -> pd.DataFrame:
    return sensitivity_oat(
        RunConfig.from_json(config_json),
        json.loads(specs_json),
        output_kpi=output_kpi,
        n_replications=n_replications,
        n_jobs=-1,
    )


def sensitivity_cached(
    config: RunConfig,
    param_specs: Mapping[str, Sequence[float | int]],
    n_replications: int,
    output_kpi: str = "total_cost",
) -> pd.DataFrame:
    """One-at-a-time sensitivity for ``config`` (memoized), returning the tornado-ranked frame.

    ``param_specs`` maps a ``"section.field"`` path to its ``(low, high)`` pair; it is canonicalized
    to sorted JSON for the cache key (preserving each value's int / float distinction, so an int
    field stays integer-valued). See :func:`analytics.sensitivity.sensitivity_oat` for the returned
    frame's columns. ``n_jobs=-1`` is fixed and out of the key (parallel-invariant).
    """
    specs_json = json.dumps({k: list(v) for k, v in param_specs.items()}, sort_keys=True)
    result: pd.DataFrame = _cached_sensitivity(
        config.to_json(), specs_json, n_replications, output_kpi
    )
    return result
