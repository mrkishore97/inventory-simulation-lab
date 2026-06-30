"""Integration: risk module — VaR / CVaR over a Monte Carlo distribution."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from analytics.monte_carlo import monte_carlo
from analytics.risk import RiskMetrics, risk_metrics
from core.config import RunConfig


def _known_frame() -> pd.DataFrame:
    """A frame whose ``total_cost`` column is 0..99 — VaR/CVaR are hand-computable."""
    return pd.DataFrame({"total_cost": np.arange(100, dtype=float)})


def test_exact_upper_tail() -> None:
    x = np.arange(100, dtype=float)
    rm = risk_metrics(_known_frame(), confidence_level=0.95)  # tail="upper" default
    # VaR: reference the library's own default-method quantile (not a brittle 94.05
    # literal a NumPy default change could shift) -> guards the wiring (alpha, column, tail).
    assert rm.value_at_risk == pytest.approx(np.quantile(x, 0.95))
    # CVaR: the independent hand-computed check -> mean(95..99) == 97.0.
    assert rm.conditional_value_at_risk == pytest.approx(97.0)
    assert rm.mean == pytest.approx(49.5)
    assert rm.tail_probability == pytest.approx(0.05)
    assert rm.column == "total_cost"
    assert rm.tail == "upper"


def test_exact_lower_tail() -> None:
    x = np.arange(100, dtype=float)
    rm = risk_metrics(_known_frame(), confidence_level=0.95, tail="lower")
    assert rm.value_at_risk == pytest.approx(np.quantile(x, 0.05))
    # CVaR (gain tail): mean(0..4) == 2.0.
    assert rm.conditional_value_at_risk == pytest.approx(2.0)
    assert rm.tail == "lower"


def test_ordering_upper_tail_primary(make_config: Callable[..., RunConfig]) -> None:
    # The version-independent mathematical invariant: mean <= VaR <= CVaR (loss tail).
    df = monte_carlo(make_config(), 64, n_jobs=1)
    rm = risk_metrics(df, "total_cost", confidence_level=0.95)
    assert rm.mean <= rm.value_at_risk <= rm.conditional_value_at_risk


def test_ordering_lower_tail(make_config: Callable[..., RunConfig]) -> None:
    # Gain tail: CVaR <= VaR <= mean for a service KPI.
    df = monte_carlo(make_config(), 64, n_jobs=1)
    rm = risk_metrics(df, "fill_rate", confidence_level=0.95, tail="lower")
    assert rm.conditional_value_at_risk <= rm.value_at_risk <= rm.mean


def test_var_monotone_in_confidence(make_config: Callable[..., RunConfig]) -> None:
    df = monte_carlo(make_config(), 64, n_jobs=1)
    deep = risk_metrics(df, "total_cost", confidence_level=0.99)
    shallow = risk_metrics(df, "total_cost", confidence_level=0.90)
    assert deep.value_at_risk >= shallow.value_at_risk


def test_composition_smoke(make_config: Callable[..., RunConfig]) -> None:
    rm = risk_metrics(monte_carlo(make_config(), 50, n_jobs=1))
    assert np.isfinite(rm.value_at_risk)
    assert rm.value_at_risk > 0.0  # total_cost is positive
    assert rm.conditional_value_at_risk >= rm.value_at_risk


def test_determinism_pure_reduction(make_config: Callable[..., RunConfig]) -> None:
    df = monte_carlo(make_config(), 32, n_jobs=1)
    assert risk_metrics(df) == risk_metrics(df)  # frozen dataclass equality


def test_single_replication(make_config: Callable[..., RunConfig]) -> None:
    df = monte_carlo(make_config(), 1, n_jobs=1)
    rm = risk_metrics(df, "total_cost")
    assert rm.mean == rm.value_at_risk == rm.conditional_value_at_risk
    assert np.isfinite(rm.conditional_value_at_risk)  # tail mask non-empty -> no nan


def test_returns_risk_metrics_instance() -> None:
    assert isinstance(risk_metrics(_known_frame()), RiskMetrics)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_bad_confidence_level_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence_level must be in"):
        risk_metrics(_known_frame(), confidence_level=bad)


def test_bad_tail_raises() -> None:
    with pytest.raises(ValueError, match="tail must be"):
        risk_metrics(_known_frame(), tail="sideways")  # type: ignore[arg-type]


def test_unknown_column_raises() -> None:
    with pytest.raises(ValueError, match="is not a column"):
        risk_metrics(_known_frame(), column="bogus")


def test_empty_distribution_raises() -> None:
    with pytest.raises(ValueError, match="distribution is empty"):
        risk_metrics(pd.DataFrame({"total_cost": []}))
