"""Unit tests for the Syntetos-Boylan demand classifier (``recommender.classify``).

The four bundled M5 archetype SKUs are a built-in oracle: classifying each one's history must
reproduce the Syntetos-Boylan quadrant recorded in the manifest by the curation script. The
manifest is read directly (not via the ``ui`` loader) so this compute module's tests stay
independent of the UI layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommender.classify import (
    SB_ADI_CUTOFF,
    SB_CV2_CUTOFF,
    SBClassification,
    adi,
    classify,
    classify_metrics,
    cv_squared,
    trim_leading_zeros,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = json.loads((_REPO_ROOT / "data/m5/archetypes_manifest.json").read_text())["archetypes"]


class TestTrimLeadingZeros:
    def test_drops_pre_introduction_run(self) -> None:
        out = trim_leading_zeros(np.array([0.0, 0.0, 3.0, 0.0, 5.0]))
        np.testing.assert_array_equal(out, np.array([3.0, 0.0, 5.0]))

    def test_all_zero_returns_empty(self) -> None:
        assert trim_leading_zeros(np.zeros(5)).shape[0] == 0

    def test_no_leading_zeros_unchanged(self) -> None:
        series = np.array([1.0, 0.0, 2.0])
        np.testing.assert_array_equal(trim_leading_zeros(series), series)


class TestADI:
    def test_basic_interval(self) -> None:
        # 5 periods, 2 non-zero -> 2.5
        assert adi(np.array([0.0, 0.0, 3.0, 0.0, 5.0])) == pytest.approx(2.5)

    def test_dense_series_near_one(self) -> None:
        assert adi(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)

    def test_all_zero_is_inf(self) -> None:
        assert np.isinf(adi(np.zeros(5)))


class TestCVSquared:
    def test_constant_sizes_zero(self) -> None:
        assert cv_squared(np.array([3.0, 3.0, 3.0])) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # sizes [2, 4]: mean 3, population std 1 -> (1/3)^2
        assert cv_squared(np.array([2.0, 4.0])) == pytest.approx((1.0 / 3.0) ** 2)

    def test_ignores_zeros(self) -> None:
        assert cv_squared(np.array([0.0, 2.0, 0.0, 4.0])) == pytest.approx((1.0 / 3.0) ** 2)

    def test_single_nonzero_is_zero(self) -> None:
        assert cv_squared(np.array([0.0, 7.0, 0.0])) == pytest.approx(0.0)


class TestClassifyMetrics:
    def test_smooth(self) -> None:
        assert classify_metrics(1.0, 0.2) == "smooth"

    def test_erratic(self) -> None:
        assert classify_metrics(1.0, 0.6) == "erratic"

    def test_intermittent(self) -> None:
        assert classify_metrics(2.0, 0.2) == "intermittent"

    def test_lumpy(self) -> None:
        assert classify_metrics(2.0, 0.6) == "lumpy"

    def test_cutoffs_inclusive_on_high_side(self) -> None:
        # exactly at both cutoffs -> both axes "high" -> lumpy; just below both -> smooth
        assert classify_metrics(SB_ADI_CUTOFF, SB_CV2_CUTOFF) == "lumpy"
        assert classify_metrics(SB_ADI_CUTOFF - 0.01, SB_CV2_CUTOFF - 0.01) == "smooth"


class TestClassify:
    def test_returns_bundle_with_metrics(self) -> None:
        result = classify(np.array([2.0, 4.0, 2.0, 4.0]))  # adi=1.0, cv2=(1/3)^2 -> low/low
        assert isinstance(result, SBClassification)
        assert result.sb_class == "smooth"
        assert result.adi == pytest.approx(1.0)
        assert result.cv2 == pytest.approx((1.0 / 3.0) ** 2)
        assert result.n_periods == 4
        assert result.n_nonzero == 4

    def test_trims_leading_zeros(self) -> None:
        # pre-introduction zeros must change neither the classification nor the metrics
        assert classify(np.array([0.0, 0.0, 3.0, 0.0, 5.0])) == classify(np.array([3.0, 0.0, 5.0]))

    def test_accepts_plain_list(self) -> None:
        assert classify([1.0, 2.0, 3.0, 4.0]).sb_class == "smooth"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            classify(np.array([]))

    def test_all_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="all zeros"):
            classify(np.zeros(10))


@pytest.mark.parametrize("name", list(_MANIFEST))
class TestManifestOracle:
    """Each bundled SKU must classify to the quadrant the curation script recorded."""

    def test_classify_history_matches_manifest(self, name: str) -> None:
        entry = _MANIFEST[name]
        demand = pd.read_parquet(_REPO_ROOT / "data/m5" / entry["parquet"])["demand"].to_numpy()
        assert classify(demand).sb_class == entry["sb_quadrant"]

    def test_classify_metrics_matches_manifest(self, name: str) -> None:
        entry = _MANIFEST[name]
        assert classify_metrics(entry["adi"], entry["cv2"]) == entry["sb_quadrant"]
