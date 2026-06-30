"""Integration: sensitivity analysis — OAT perturbation + tornado ranking."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from analytics.sensitivity import _apply_override, _get_param, sensitivity_oat
from core.config import RunConfig


def test_shape_and_columns(make_config: Callable[..., RunConfig]) -> None:
    df = sensitivity_oat(
        make_config(),
        {"costs.ordering_fixed": (10.0, 40.0), "demand.std": (1.0, 5.0)},
        n_replications=6,
        n_jobs=1,
    )
    assert len(df) == 2
    assert list(df.columns) == [
        "parameter",
        "baseline",
        "low",
        "high",
        "kpi_low",
        "kpi_baseline",
        "kpi_high",
        "swing",
        "abs_swing",
    ]


def test_tornado_sorted_by_abs_swing(make_config: Callable[..., RunConfig]) -> None:
    df = sensitivity_oat(
        make_config(),
        {
            "costs.ordering_fixed": (5.0, 80.0),
            "demand.std": (1.0, 3.0),
            "policy.reorder_point": (10.0, 30.0),
        },
        n_replications=8,
        n_jobs=1,
    )
    assert df["abs_swing"].is_monotonic_decreasing


def test_known_direction_ordering_cost(make_config: Callable[..., RunConfig]) -> None:
    df = sensitivity_oat(
        make_config(),
        {"costs.ordering_fixed": (10.0, 50.0)},
        output_kpi="total_cost",
        n_replications=8,
        n_jobs=1,
    )
    row = df.iloc[0]
    assert row["parameter"] == "costs.ordering_fixed"
    # Higher fixed ordering cost -> higher total cost (same order pattern under CRN).
    assert row["kpi_high"] > row["kpi_low"]
    assert row["swing"] > 0


def test_apply_override_changes_only_target(make_config: Callable[..., RunConfig]) -> None:
    config = make_config()
    variant = _apply_override(config, "demand.mean", 99.0)
    assert _get_param(variant, "demand.mean") == 99.0
    assert variant.costs == config.costs
    assert variant.policy == config.policy
    assert _get_param(variant, "demand.std") == _get_param(config, "demand.std")


def test_determinism(make_config: Callable[..., RunConfig]) -> None:
    specs: dict[str, tuple[float | int, float | int]] = {
        "costs.ordering_fixed": (10.0, 40.0),
        "demand.std": (1.0, 4.0),
    }
    a = sensitivity_oat(make_config(), specs, n_replications=6, n_jobs=1)
    b = sensitivity_oat(make_config(), specs, n_replications=6, n_jobs=1)
    pd.testing.assert_frame_equal(a, b)


def test_empty_specs_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="param_specs must be non-empty"):
        sensitivity_oat(make_config(), {}, n_replications=3, n_jobs=1)


def test_bad_output_kpi_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="is not a KPI"):
        sensitivity_oat(
            make_config(),
            {"demand.std": (1.0, 3.0)},
            output_kpi="bogus",
            n_replications=3,
            n_jobs=1,
        )


def test_path_without_dot_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="section.field"):
        sensitivity_oat(make_config(), {"demand": (1.0, 3.0)}, n_replications=3, n_jobs=1)


def test_unknown_section_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="unknown config section"):
        sensitivity_oat(make_config(), {"nope.x": (1.0, 3.0)}, n_replications=3, n_jobs=1)


def test_unknown_field_raises(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ValueError, match="unknown field"):
        sensitivity_oat(make_config(), {"demand.nope": (1.0, 3.0)}, n_replications=3, n_jobs=1)
