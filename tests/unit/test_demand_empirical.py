"""Tests for ``demand.empirical.EmpiricalDemand``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from core.config import EmpiricalDemandConfig
from demand.empirical import EmpiricalDemand


def _write_parquet(
    path: Path,
    demand: np.ndarray,
    extra_columns: dict[str, object] | None = None,
) -> Path:
    """Helper: write a parquet with a ``demand`` column to ``path``."""
    data: dict[str, object] = {"demand": demand}
    if extra_columns:
        data.update(extra_columns)
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


class TestEmpiricalDemandDraw:
    """Happy-path draw behavior.

    The chassis arm of the non-parametric demand surface. Each draw is
    ``rng.choice(history)`` — uniformly random from the stored history
    array. These tests pin: float-type output, value-from-history
    membership, non-negativity (by pre-validation, not clip), RNG state
    advance, marginal distribution preservation (sample mean ≈ history
    mean; KS test passes), and determinism under same seed.
    """

    def test_draw_returns_float_type(self, tmp_path: Path) -> None:
        # numpy's Generator.choice returns the array's dtype scalar
        # (np.float64 here); the float(...) upcast at the seam preserves
        # the Demand.draw() -> float contract — same idiom as Poisson /
        # NegBin / Gamma / Lognormal.
        path = _write_parquet(tmp_path / "h.parquet", np.array([1.0, 2.0, 3.0]))
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        assert isinstance(demand.draw(), float)

    def test_draw_returns_value_from_history(self, tmp_path: Path) -> None:
        history = np.linspace(0.0, 100.0, 1000)
        path = _write_parquet(tmp_path / "h.parquet", history)
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        history_set = set(history.tolist())
        for _ in range(1000):
            assert demand.draw() in history_set

    def test_draw_returns_non_negative(self, tmp_path: Path) -> None:
        # History is pre-validated non-negative at __init__; every draw
        # is guaranteed >= 0 by construction (no runtime clip needed).
        path = _write_parquet(tmp_path / "h.parquet", np.array([0.0, 0.5, 1.0, 5.0, 10.0, 100.0]))
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        for _ in range(1000):
            assert demand.draw() >= 0.0

    def test_draw_advances_rng_state(self, tmp_path: Path) -> None:
        # Counterpart to the deterministic-pattern tests (Stationary /
        # Seasonal / Trending) where rng state must NOT advance. Empirical
        # IS stochastic: bit_generator.state MUST change across draws.
        path = _write_parquet(tmp_path / "h.parquet", np.arange(1.0, 100.0))
        cfg = EmpiricalDemandConfig(history_path=path)
        rng = np.random.default_rng(42)
        demand = EmpiricalDemand(cfg, rng)
        state_before = rng.bit_generator.state
        for _ in range(100):
            demand.draw()
        state_after = rng.bit_generator.state
        assert state_before != state_after

    def test_draw_distribution_matches_history_mean(self, tmp_path: Path) -> None:
        # Over 10_000 draws, sample mean is within ±3% of the history
        # mean. Statistical sanity check; loose bound to avoid flakiness.
        rng_gen = np.random.default_rng(123)
        history = rng_gen.uniform(0.0, 20.0, size=2000)
        path = _write_parquet(tmp_path / "h.parquet", history)
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        samples = np.array([demand.draw() for _ in range(10_000)])
        history_mean = history.mean()
        sample_mean = samples.mean()
        assert abs(sample_mean - history_mean) / history_mean < 0.03, (
            f"empirical mean {sample_mean:.4f} differs from history mean "
            f"{history_mean:.4f} by more than 3%"
        )

    def test_draw_distribution_matches_history_via_ks_test(self, tmp_path: Path) -> None:
        # Kolmogorov-Smirnov two-sample test: empirical draws should be
        # drawn from the same distribution as the history. Very loose
        # p-value gate (> 0.01) — chassis correctness, not statistical
        # power. A wildly-off implementation (e.g., always returning the
        # first element) would fail this catastrophically.
        rng_gen = np.random.default_rng(456)
        history = rng_gen.normal(10.0, 2.0, size=5000)
        history = np.maximum(0.0, history)  # match the chassis non-neg invariant
        path = _write_parquet(tmp_path / "h.parquet", history)
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        samples = np.array([demand.draw() for _ in range(5000)])
        result = stats.ks_2samp(history, samples)
        assert result.pvalue > 0.01, (
            f"KS test failed (p={result.pvalue:.4f}); empirical draws look "
            f"unlike the source history"
        )

    def test_determinism_under_same_seed(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "h.parquet", np.arange(1.0, 100.0))
        cfg = EmpiricalDemandConfig(history_path=path)
        a = EmpiricalDemand(cfg, np.random.default_rng(42))
        b = EmpiricalDemand(cfg, np.random.default_rng(42))
        for _ in range(100):
            assert a.draw() == b.draw()


class TestEmpiricalDemandInit:
    """File-loading and validation behavior.

    Six failure modes — all locked here. EmpiricalDemand is the first
    demand arm with non-trivial init failure modes; the engine error
    surface needs each path to fail loudly with a clear ``ValueError``
    that mentions the offending file path.
    """

    def test_init_loads_demand_column(self, tmp_path: Path) -> None:
        history = np.arange(0.0, 50.0)
        path = _write_parquet(tmp_path / "h.parquet", history)
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        # Read the private attribute to confirm the load worked
        assert len(demand._history) == len(history)
        np.testing.assert_array_equal(demand._history, history)

    def test_file_not_found_raises_ValueError(self, tmp_path: Path) -> None:
        cfg = EmpiricalDemandConfig(history_path=tmp_path / "does_not_exist.parquet")
        with pytest.raises(ValueError, match="not found"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_missing_demand_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "no_demand.parquet"
        pd.DataFrame({"value": [1.0, 2.0, 3.0]}).to_parquet(path, index=False)
        cfg = EmpiricalDemandConfig(history_path=path)
        with pytest.raises(ValueError, match="'demand' column"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_empty_parquet_raises(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "empty.parquet", np.array([], dtype=np.float64))
        cfg = EmpiricalDemandConfig(history_path=path)
        with pytest.raises(ValueError, match="empty"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_nan_value_in_history_raises(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "nan.parquet", np.array([1.0, np.nan, 2.0]))
        cfg = EmpiricalDemandConfig(history_path=path)
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_positive_inf_value_in_history_raises(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "inf.parquet", np.array([1.0, np.inf, 2.0]))
        cfg = EmpiricalDemandConfig(history_path=path)
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_negative_inf_value_in_history_raises(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "ninf.parquet", np.array([1.0, -np.inf, 2.0]))
        cfg = EmpiricalDemandConfig(history_path=path)
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_negative_value_in_history_raises(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "neg.parquet", np.array([1.0, -0.5, 2.0]))
        cfg = EmpiricalDemandConfig(history_path=path)
        with pytest.raises(ValueError, match="negative"):
            EmpiricalDemand(cfg, np.random.default_rng(0))

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        # Files with extra columns (e.g., a 'date' column) load fine; the
        # extra columns are silently ignored. Locks the "demand-column-only"
        # contract — future bullets may opt-in to date-aware replay.
        history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        path = _write_parquet(
            tmp_path / "extra.parquet",
            history,
            extra_columns={"date": ["a", "b", "c", "d", "e"], "store": [1, 2, 3, 4, 5]},
        )
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        assert len(demand._history) == 5
        np.testing.assert_array_equal(demand._history, history)


class TestEmpiricalDemandProperty:
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    @given(
        history=arrays(
            np.float64,
            shape=st.integers(min_value=1, max_value=1000),
            elements=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_draw_returns_non_negative_finite_float(
        self, tmp_path: Path, history: np.ndarray
    ) -> None:
        # Property: any valid history (length 1-1000, finite non-negative
        # values up to 1e6) produces draws that are non-negative finite
        # floats. Bounds rationale:
        #   * length [1, 1000] covers single-element edge case and
        #     reasonable horizons.
        #   * values [0, 1e6] is the demand non-negativity contract;
        #     1e6 covers high-volume SKUs.
        path = _write_parquet(tmp_path / "h.parquet", history)
        cfg = EmpiricalDemandConfig(history_path=path)
        demand = EmpiricalDemand(cfg, np.random.default_rng(0))
        result = demand.draw()
        assert isinstance(result, float)
        assert result >= 0.0
        assert math.isfinite(result)
