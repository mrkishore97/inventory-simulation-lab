"""Sensitivity analysis — one-at-a-time (OAT) perturbation + the tornado ranking.

:func:`sensitivity_oat` answers "which input matters
most?": it perturbs one config parameter at a time around its baseline (low / high),
runs the Monte Carlo runner at each setting, and ranks the parameters by the
swing they induce in an output KPI. The returned frame, sorted by ``abs_swing``
descending, is the tornado-chart data (widest bar on top).

``swing = kpi_high - kpi_low`` — **positive** means the KPI *rises* as the parameter
moves low -> high, **negative** means it falls; the ranking uses ``abs_swing``.

Common random numbers: every run (baseline + the two per parameter) shares
``config.master_seed``, so the Monte Carlo runner derives the same per-replication
seeds — cost-parameter perturbations face identical demand/lead-time realizations,
sharpening the swing estimate. Non-invasive: composes ``monte_carlo`` + ``model_copy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import pandas as pd
from pydantic import BaseModel

from analytics.kpis import KPIs
from analytics.monte_carlo import monte_carlo
from core.config import RunConfig

_KPI_NAMES = frozenset(f.name for f in fields(KPIs))


def sensitivity_oat(
    config: RunConfig,
    param_specs: Mapping[str, tuple[float | int, float | int]],
    output_kpi: str = "total_cost",
    n_replications: int = 100,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """One-at-a-time sensitivity → a tornado-ranked DataFrame.

    ``param_specs`` maps a ``"section.field"`` config path (e.g. ``"demand.mean"``,
    ``"costs.ordering_fixed"``, ``"policy.reorder_point"``) to a ``(low, high)`` pair.
    For each parameter the output KPI is averaged (via :func:`monte_carlo`) with the
    parameter held at ``low`` and at ``high`` — all others at baseline — and the swing
    ``kpi_high - kpi_low`` recorded.

    Returns one row per parameter — ``parameter``, ``baseline`` (its config value),
    ``low``, ``high``, ``kpi_low`` / ``kpi_baseline`` / ``kpi_high``, ``swing``, and
    ``abs_swing`` — **sorted by ``abs_swing`` descending** (the tornado order). A
    positive ``swing`` means the KPI rises as the parameter moves low -> high.
    """
    if not param_specs:
        raise ValueError("param_specs must be non-empty")
    if output_kpi not in _KPI_NAMES:
        raise ValueError(f"{output_kpi!r} is not a KPI; choose from {sorted(_KPI_NAMES)}")

    def mean_kpi(cfg: RunConfig) -> float:
        return float(monte_carlo(cfg, n_replications, n_jobs=n_jobs)[output_kpi].mean())

    kpi_baseline = mean_kpi(config)
    rows: list[dict[str, object]] = []
    for path, (low, high) in param_specs.items():
        kpi_low = mean_kpi(_apply_override(config, path, low))
        kpi_high = mean_kpi(_apply_override(config, path, high))
        swing = kpi_high - kpi_low
        rows.append(
            {
                "parameter": path,
                "baseline": _get_param(config, path),
                "low": low,
                "high": high,
                "kpi_low": kpi_low,
                "kpi_baseline": kpi_baseline,
                "kpi_high": kpi_high,
                "swing": swing,
                "abs_swing": abs(swing),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_swing", ascending=False, ignore_index=True)


def _resolve(config: RunConfig, path: str) -> tuple[str, BaseModel, str]:
    """Split ``"section.field"`` into ``(section, sub-config, field)``, validating fail-loud.

    Both names are checked here so :func:`_get_param` and :func:`_apply_override` share one
    consistent error path (a clear ``ValueError`` instead of a raw ``AttributeError``).
    """
    if "." not in path:
        raise ValueError(f"param path must be 'section.field', got {path!r}")
    section, field = path.split(".", 1)
    if section not in type(config).model_fields:
        raise ValueError(f"unknown config section {section!r}")
    sub = getattr(config, section)
    if field not in type(sub).model_fields:
        raise ValueError(f"unknown field {field!r} in section {section!r}")
    return section, sub, field


def _get_param(config: RunConfig, path: str) -> object:
    """Return the current (baseline) value at ``"section.field"``."""
    _, sub, field = _resolve(config, path)
    return getattr(sub, field)


def _apply_override(config: RunConfig, path: str, value: float | int) -> RunConfig:
    """Return a copy of ``config`` with ``"section.field"`` set to ``value``.

    The section is re-validated (field type + cross-field invariants, e.g. ``(s,S)``'s
    ``s < S``) via ``model_validate``; ``model_copy`` is then safe because ``RunConfig``
    has no ``@model_validator`` coupling its sections.
    """
    section, sub, field = _resolve(config, path)
    new_sub = sub.model_validate({**sub.model_dump(), field: value})
    return config.model_copy(update={section: new_sub})
