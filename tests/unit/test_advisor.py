"""Unit tests for the heuristic Policy Advisor (``recommender.advisor``).

Three layers of confidence that the Advisor is formula-driven, not a lookup table:

1. **synthetic quadrants** — four hand-built histories, one per Syntetos-Boylan quadrant, whose
   recommended parameters are checked against the values recomputed independently through
   ``analytics.classical`` (proving the numbers are derived, not hardcoded);
2. **manifest oracle** — the four bundled M5 archetype SKUs must each recommend the policy kind the
   quadrant maps to (end-to-end on real data); and
3. **RunConfig embedding** — every recommended policy must validate inside a real ``RunConfig``,
   proving the advisor's output is actually runnable.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics import classical
from core.config import (
    BaseStockPolicyConfig,
    PeriodicReviewPolicyConfig,
    RunConfig,
    SQPolicyConfig,
    SSPolicyConfig,
)
from recommender.advisor import (
    DEFAULT_SERVICE_LEVEL,
    FEASIBILITY_WARNING_SQ,
    DemandStats,
    Recommendation,
    recommend,
    summarize,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = json.loads((_REPO_ROOT / "data/m5/archetypes_manifest.json").read_text())["archetypes"]

# A valid business context shared across tests (matches example.yaml's lead time + costs).
_CTX = {"lead_time": 3.0, "ordering_cost": 20.0, "holding_cost": 0.5}

# Synthetic histories, one per Syntetos-Boylan quadrant (see test_classify for the cutoffs).
_SMOOTH = np.array([10.0, 12.0, 8.0, 10.0, 12.0, 8.0, 10.0, 12.0, 8.0, 10.0])
_ERRATIC = np.array([1.0, 20.0, 2.0, 18.0, 1.0, 25.0, 3.0, 15.0, 2.0, 22.0])
_INTERMITTENT = np.array([5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0])
_LUMPY = np.array([1.0, 0.0, 0.0, 50.0, 0.0, 0.0, 3.0, 0.0, 0.0, 40.0, 0.0, 0.0])

# A high-volume smooth SKU (μ ≈ 130): EOQ ≈ √(2·130·20/0.5) ≈ 102 < μ, so the once-per-period
# (s, Q) batch is below one period's demand — the EOQ-feasibility case the guardrail must flag
# (mirrors the M5 smooth archetype's ~131/period demand). Still smooth: demand every period
# (ADI 1.0), tiny CV².
_SMOOTH_HIGH_VOLUME = np.array(
    [130.0, 132.0, 128.0, 131.0, 129.0, 133.0, 127.0, 130.0, 132.0, 128.0]
)

# Quadrant → expected recommended policy kind (the one-policy-per-quadrant map).
_QUADRANT_KIND = {"smooth": "sQ", "erratic": "sS", "intermittent": "RS", "lumpy": "base_stock"}


class TestSummarize:
    def test_moments_over_active_history(self) -> None:
        stats = summarize(_SMOOTH)
        assert stats.mean == pytest.approx(float(np.mean(_SMOOTH)))
        assert stats.std == pytest.approx(float(np.std(_SMOOTH)))

    def test_trims_leading_zeros(self) -> None:
        # pre-introduction zeros must not change the moments
        raw = np.array([0.0, 0.0, 3.0, 0.0, 5.0])
        trimmed = np.array([3.0, 0.0, 5.0])
        assert summarize(raw) == summarize(trimmed)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize(np.array([]))

    def test_all_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="all zeros"):
            summarize(np.zeros(8))


class TestRecommendSmooth:
    def test_recommends_sQ_with_eoq_and_rop(self) -> None:
        rec = recommend(_SMOOTH, **_CTX)
        stats = summarize(_SMOOTH)
        z = classical.z_for_service_level(DEFAULT_SERVICE_LEVEL)
        policy = rec.policy
        assert isinstance(policy, SQPolicyConfig)
        assert rec.classification.sb_class == "smooth"
        assert policy.order_quantity == max(1, round(classical.eoq(stats.mean, 20.0, 0.5)))
        assert policy.reorder_point == float(
            round(classical.reorder_point(stats.mean, stats.std, 3.0, z))
        )
        assert rec.stats == stats

    def test_rationale_is_number_rich_and_per_period(self) -> None:
        rec = recommend(_SMOOTH, **_CTX)
        policy = rec.policy
        assert isinstance(policy, SQPolicyConfig)
        assert "smooth" in rec.rationale
        assert "(s, Q)" in rec.rationale
        assert "per period" in rec.rationale  # the EOQ unit clarification the review required
        assert str(policy.order_quantity) in rec.rationale


class TestFeasibilityGuardrail:
    """The EOQ-feasibility stopgap: warn when a smooth (s, Q) batch is below one period's demand.

    A once-per-period (s, Q) cannot reorder sub-period, so an EOQ batch ``Q < μ`` may order every
    period and still fall behind. The Advisor flags this rather than silently recommending it.
    """

    def test_high_volume_smooth_warns(self) -> None:
        rec = recommend(_SMOOTH_HIGH_VOLUME, **_CTX)
        policy = rec.policy
        assert isinstance(policy, SQPolicyConfig)
        assert rec.classification.sb_class == "smooth"
        # The EOQ batch is below mean demand — the infeasible-cycle case the guardrail flags.
        assert policy.order_quantity < summarize(_SMOOTH_HIGH_VOLUME).mean
        assert rec.feasibility_warning == FEASIBILITY_WARNING_SQ

    def test_low_volume_smooth_has_no_warning(self) -> None:
        rec = recommend(_SMOOTH, **_CTX)
        policy = rec.policy
        assert isinstance(policy, SQPolicyConfig)
        # EOQ (≈28) exceeds μ (≈10), so the once-per-period batch is feasible — no warning.
        assert policy.order_quantity >= summarize(_SMOOTH).mean
        assert rec.feasibility_warning is None

    @pytest.mark.parametrize("history", [_ERRATIC, _INTERMITTENT, _LUMPY])
    def test_non_sq_recommendations_never_warn(self, history: np.ndarray) -> None:
        # The guardrail is specific to the fixed-batch (s, Q) failure; the order-up-to policies
        # absorb size variability and carry no EOQ-feasibility warning.
        rec = recommend(history, **_CTX)
        assert not isinstance(rec.policy, SQPolicyConfig)
        assert rec.feasibility_warning is None


class TestRecommendErratic:
    def test_recommends_sS_with_order_up_to_above_reorder(self) -> None:
        rec = recommend(_ERRATIC, **_CTX)
        stats = summarize(_ERRATIC)
        z = classical.z_for_service_level(DEFAULT_SERVICE_LEVEL)
        s = float(round(classical.reorder_point(stats.mean, stats.std, 3.0, z)))
        batch = max(1, round(classical.eoq(stats.mean, 20.0, 0.5)))
        policy = rec.policy
        assert isinstance(policy, SSPolicyConfig)
        assert rec.classification.sb_class == "erratic"
        assert policy.reorder_point == s
        assert policy.order_up_to == max(s + batch, 1.0)
        assert policy.reorder_point < policy.order_up_to


class TestRecommendIntermittent:
    def test_recommends_RS_with_review_period_from_adi(self) -> None:
        rec = recommend(_INTERMITTENT, **_CTX)
        policy = rec.policy
        assert isinstance(policy, PeriodicReviewPolicyConfig)
        assert rec.classification.sb_class == "intermittent"
        assert policy.review_period == max(1, round(rec.classification.adi))
        assert policy.order_up_to >= 1.0


class TestRecommendLumpy:
    def test_recommends_base_stock_with_positive_target(self) -> None:
        rec = recommend(_LUMPY, **_CTX)
        policy = rec.policy
        assert isinstance(policy, BaseStockPolicyConfig)
        assert rec.classification.sb_class == "lumpy"
        assert policy.target_level >= 1.0

    def test_rationale_flags_conservative_starting_point(self) -> None:
        # the review required the lumpy recommendation to disclaim optimality
        rec = recommend(_LUMPY, **_CTX)
        assert "conservative starting point, not a guaranteed optimum" in rec.rationale


class TestLowVolumeFloors:
    def test_tiny_demand_level_is_floored_to_one(self) -> None:
        # a fractional, very low-volume intermittent SKU: μ(R+L)+zσ√(R+L) rounds to 0, so the
        # PositiveFloat order-up-to must be floored to 1.0 (the safeguard the review required).
        history = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        rec = recommend(history, **_CTX)
        policy = rec.policy
        assert isinstance(policy, PeriodicReviewPolicyConfig)
        assert policy.order_up_to == 1.0


class TestContextValidation:
    def test_nonpositive_lead_time_raises(self) -> None:
        with pytest.raises(ValueError, match="lead_time"):
            recommend(_SMOOTH, lead_time=0.0, ordering_cost=20.0, holding_cost=0.5)

    def test_negative_ordering_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="ordering_cost"):
            recommend(_SMOOTH, lead_time=3.0, ordering_cost=-1.0, holding_cost=0.5)

    def test_nonpositive_holding_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="holding_cost"):
            recommend(_SMOOTH, lead_time=3.0, ordering_cost=20.0, holding_cost=0.0)

    def test_service_level_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="service_level"):
            recommend(_SMOOTH, **_CTX, service_level=1.5)


class TestHistoryValidation:
    def test_empty_history_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            recommend(np.array([]), **_CTX)

    def test_all_zero_history_raises(self) -> None:
        with pytest.raises(ValueError, match="all zeros"):
            recommend(np.zeros(10), **_CTX)


@pytest.mark.parametrize("name", list(_MANIFEST))
class TestManifestOracle:
    """Each bundled SKU must recommend the policy kind its quadrant maps to (real-data oracle)."""

    def test_recommended_kind_matches_quadrant(self, name: str) -> None:
        entry = _MANIFEST[name]
        demand = pd.read_parquet(_REPO_ROOT / "data/m5" / entry["parquet"])["demand"].to_numpy()
        rec = recommend(demand, **_CTX)
        assert rec.policy.kind == _QUADRANT_KIND[entry["sb_quadrant"]]


@pytest.mark.parametrize(
    "history", [_SMOOTH, _ERRATIC, _INTERMITTENT, _LUMPY], ids=list(_QUADRANT_KIND)
)
class TestRunConfigEmbedding:
    """Every recommended policy must validate inside a real RunConfig — i.e. be runnable."""

    def test_policy_validates_inside_run_config(self, history: np.ndarray) -> None:
        base = RunConfig.from_yaml((_REPO_ROOT / "data/scenarios/example.yaml").read_text())
        rec = recommend(history, **_CTX)
        validated = RunConfig.model_validate(
            {**base.model_dump(mode="json"), "policy": rec.policy.model_dump(mode="json")}
        )
        assert validated.policy == rec.policy


class TestRecommendationImmutability:
    def test_recommendation_is_frozen(self) -> None:
        rec = recommend(_SMOOTH, **_CTX)
        assert isinstance(rec, Recommendation)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.rationale = "mutated"  # type: ignore[misc]

    def test_demand_stats_is_frozen(self) -> None:
        stats = summarize(_SMOOTH)
        assert isinstance(stats, DemandStats)
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.mean = 0.0  # type: ignore[misc]
