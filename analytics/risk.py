"""Risk module — Value-at-Risk and Conditional VaR over a Monte Carlo distribution.

:func:`risk_metrics` is a **pure reduction over an existing Monte Carlo
distribution frame** — the same role :func:`analytics.kpis.compute_kpis` plays
over a Ledger frame. It does *not* take a config and re-run the simulation (the
:mod:`analytics.pareto` / :mod:`analytics.sensitivity` pattern); the
:func:`analytics.monte_carlo.monte_carlo` output already holds the full outcome
distribution and risk just summarizes its tail. Usage::

    risk_metrics(monte_carlo(config, 1000))                       # cost tail (default)
    risk_metrics(mc_frame, column="fill_rate", tail="lower")      # service tail

For a confidence level ``alpha`` (default 0.95):

- **Upper (loss) tail** — the default, for cost: ``VaR = quantile(x, alpha)`` (the
  cost not exceeded ``alpha`` of the time, e.g. the 95th percentile);
  ``CVaR = mean(x | x >= VaR)`` (the expected cost in the worst ``1 - alpha``
  fraction of replications). Ordering: ``mean <= VaR <= CVaR``.
- **Lower (gain) tail** — for service KPIs (``fill_rate``, ...):
  ``VaR = quantile(x, 1 - alpha)`` (e.g. the 5th-percentile service outcome);
  ``CVaR = mean(x | x <= VaR)``. Ordering: ``CVaR <= VaR <= mean``.

CVaR uses the empirical tail mask (``x >= VaR`` / ``x <= VaR``) — the standard
sample estimator, and exactly what the M4 distribution plot draws (a VaR marker
line plus CVaR as the mean of the points beyond it). Since ``np.quantile`` returns
a value in ``[min(x), max(x)]``, the mask always selects at least one replication,
so CVaR is never ``nan`` (this matters at ``n == 1``).

Non-invasive: composes :func:`analytics.monte_carlo.monte_carlo` output; no engine
or config change. v1 assumes a ``nan``-free column — ``total_cost`` (the default)
and every cost / service KPI qualify; only ``inventory_turns`` / ``days_of_supply``
can be ``nan`` (zero on-hand / demand) and are out of the default path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskMetrics:
    """Tail-risk summary of one KPI column over a Monte Carlo distribution.

    A self-describing bundle (it carries ``column`` / ``confidence_level`` /
    ``tail_probability`` / ``tail`` so a CLI, the M4 UI, or an export needs no
    extra arithmetic to label it — e.g. "95% VaR" / "worst 5% average").

    - ``mean``: the distribution mean (the figure VaR/CVaR contrast against).
    - ``value_at_risk`` (VaR): the ``confidence_level`` quantile (upper tail) or
      the ``1 - confidence_level`` quantile (lower tail) of ``column``.
    - ``conditional_value_at_risk`` (CVaR): the mean of the replications in the
      tail at or beyond VaR — the expected outcome given a tail event.
    - ``confidence_level`` (``alpha``): the tail confidence, in ``(0, 1)``.
    - ``tail_probability``: ``1 - confidence_level`` (the worst-``(1 - alpha)``
      fraction; the "5%" in "worst 5% average").
    - ``tail``: ``"upper"`` (loss tail — high values are bad, e.g. cost) or
      ``"lower"`` (gain tail — low values are bad, e.g. a service level).
    """

    column: str
    mean: float
    value_at_risk: float
    conditional_value_at_risk: float
    confidence_level: float
    tail_probability: float
    tail: str


def risk_metrics(
    distribution: pd.DataFrame,
    column: str = "total_cost",
    confidence_level: float = 0.95,
    tail: Literal["upper", "lower"] = "upper",
) -> RiskMetrics:
    """Value-at-Risk and Conditional VaR for one KPI column of a Monte Carlo frame.

    ``distribution`` is a per-replication DataFrame (the
    :func:`analytics.monte_carlo.monte_carlo` output); ``column`` selects the KPI
    to assess (default ``"total_cost"``, the use). ``confidence_level``
    (``alpha``) is the tail confidence in ``(0, 1)``. ``tail`` picks which tail is
    the *risky* one: ``"upper"`` for a loss like cost (high is bad — VaR is the
    ``alpha`` quantile), ``"lower"`` for a gain like a service level (low is bad —
    VaR is the ``1 - alpha`` quantile). See the module docstring for the formulas.

    Returns a :class:`RiskMetrics`. This is a pure reduction — two calls on the
    same frame return identical results.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if tail not in ("upper", "lower"):
        raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}")
    if column not in distribution.columns:
        raise ValueError(f"{column!r} is not a column; got {list(distribution.columns)}")
    if len(distribution) == 0:
        raise ValueError("distribution is empty")

    x = distribution[column].to_numpy(dtype=float)
    if tail == "upper":  # loss tail: the risk is in the high (expensive) values
        var = float(np.quantile(x, confidence_level))
        cvar = float(x[x >= var].mean())  # >= keeps the tail non-empty -> never nan
    else:  # gain tail: the risk is in the low (poor-service) values
        var = float(np.quantile(x, 1.0 - confidence_level))
        cvar = float(x[x <= var].mean())

    return RiskMetrics(
        column=column,
        mean=float(x.mean()),
        value_at_risk=var,
        conditional_value_at_risk=cvar,
        confidence_level=confidence_level,
        tail_probability=1.0 - confidence_level,
        tail=tail,
    )
