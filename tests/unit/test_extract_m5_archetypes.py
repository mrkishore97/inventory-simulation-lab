"""Unit tests for the selection-signal helpers in ``scripts/extract_m5_archetypes.py``.

The script is not an importable package, so it is loaded by path via ``importlib``. Only the
pure, I/O-free helpers are exercised here (the data-loading ``main()`` glue is validated against
the real M5 CSVs in Phase 2). The Syntetos-Boylan classification metrics (ADI / CV² / quadrant /
leading-zero trim) were lifted into ``recommender.classify`` and are tested in
``tests/unit/test_classify.py``; this file now locks the archetype-selection signals
(autocorrelation / seasonal-strength / promo-intensity).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_m5_archetypes.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_m5_archetypes", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m5 = _load_script()


class TestAutocorr:
    def test_periodic_series_high_at_period_lag(self) -> None:
        series = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        assert m5._autocorr(series, 3) == pytest.approx(1.0)

    def test_constant_series_zero(self) -> None:
        assert m5._autocorr(np.ones(10), 1) == pytest.approx(0.0)

    def test_lag_too_large_zero(self) -> None:
        assert m5._autocorr(np.array([1.0, 2.0, 3.0]), 5) == pytest.approx(0.0)

    def test_insufficient_periods_returns_zero(self) -> None:
        # len < 2*lag: too few full periods to trust the signal (a short tail
        # would otherwise fake a near-perfect correlation). 10 < 2*6 -> 0.0.
        assert m5._autocorr(np.arange(10.0), 6) == pytest.approx(0.0)


class TestSeasonalStrength:
    def test_weekly_periodic_is_strong(self) -> None:
        one_week = np.array([5.0, 1.0, 1.0, 2.0, 3.0, 8.0, 9.0])
        series = np.tile(one_week, 10)  # 10 identical weeks -> lag-7 autocorr ~ 1
        assert m5._seasonal_strength(series) == pytest.approx(1.0, abs=1e-9)

    def test_constant_series_weak(self) -> None:
        assert m5._seasonal_strength(np.ones(400)) == pytest.approx(0.0)


class TestPromoIntensity:
    def test_discount_with_lift_is_positive(self) -> None:
        # Median price 10; the two 8.0 days are discounted (< 9). Demand lifts
        # from 1 (full price) to 5 (discounted): freq=2/6, lift=5/1-1=4.
        demand = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0])
        price = np.array([10.0, 10.0, 10.0, 10.0, 8.0, 8.0])
        assert m5._promo_intensity(demand, price) == pytest.approx((2.0 / 6.0) * 4.0)

    def test_no_discount_is_zero(self) -> None:
        demand = np.array([1.0, 2.0, 3.0, 4.0])
        price = np.array([10.0, 10.0, 10.0, 10.0])
        assert m5._promo_intensity(demand, price) == pytest.approx(0.0)

    def test_all_unknown_price_is_zero(self) -> None:
        demand = np.array([1.0, 2.0, 3.0])
        price = np.array([np.nan, np.nan, np.nan])
        assert m5._promo_intensity(demand, price) == pytest.approx(0.0)

    def test_discount_without_lift_is_zero(self) -> None:
        # Discounts occur but demand does not respond -> lift 0 -> intensity 0.
        demand = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        price = np.array([10.0, 10.0, 10.0, 10.0, 8.0, 8.0])
        assert m5._promo_intensity(demand, price) == pytest.approx(0.0)
