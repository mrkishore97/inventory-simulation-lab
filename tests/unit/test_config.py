"""Tests for core.config — validation, serialization roundtrip, and hashing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.config import (
    BaseStockPolicyConfig,
    CostConfig,
    DeterministicLeadTimeConfig,
    DisruptionWindow,
    EmpiricalDemandConfig,
    GammaDemandConfig,
    GammaLeadTimeConfig,
    IntermittentPatternConfig,
    LognormalDemandConfig,
    LognormalLeadTimeConfig,
    LumpyPatternConfig,
    NegativeBinomialDemandConfig,
    NoDisruptionConfig,
    NormalDemandConfig,
    NormalLeadTimeConfig,
    PeriodicReviewPolicyConfig,
    PoissonDemandConfig,
    RunConfig,
    ScheduledDisruptionConfig,
    SeasonalPatternConfig,
    SimulationConfig,
    SQPolicyConfig,
    SSPolicyConfig,
    StationaryPatternConfig,
    TrendingPatternConfig,
)


def _valid_run() -> RunConfig:
    return RunConfig(
        simulation=SimulationConfig(horizon=365, initial_on_hand=100),
        demand=NormalDemandConfig(mean=10.0, std=2.0),
        lead_time=DeterministicLeadTimeConfig(lead_time=3),
        policy=SQPolicyConfig(reorder_point=20.0, order_quantity=50),
        costs=CostConfig(
            unit_cost=5.0,
            holding_per_unit_per_period=0.1,
            ordering_fixed=20.0,
            backorder_per_unit_per_period=1.0,
        ),
        master_seed=42,
        name="test",
    )


class TestValidation:
    def test_valid_config_instantiates(self) -> None:
        cfg = _valid_run()
        assert cfg.simulation.horizon == 365
        assert cfg.policy.kind == "sQ"

    def test_negative_unit_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CostConfig(unit_cost=-1, holding_per_unit_per_period=0.1, ordering_fixed=20)

    def test_zero_horizon_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfig(horizon=0, initial_on_hand=0)

    def test_negative_std_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalDemandConfig(mean=10, std=-1)

    def test_zero_order_quantity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SQPolicyConfig(reorder_point=10, order_quantity=0)

    def test_extra_field_at_top_level_rejected(self) -> None:
        d = _valid_run().model_dump(mode="json")
        d["typo_field"] = "oops"
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_extra_field_in_nested_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationConfig.model_validate({"horizon": 100, "initial_on_hand": 0, "foo": "bar"})


class TestRoundtrip:
    def test_json_roundtrip(self) -> None:
        cfg = _valid_run()
        assert RunConfig.from_json(cfg.to_json()) == cfg

    def test_yaml_roundtrip(self) -> None:
        cfg = _valid_run()
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_sS(self) -> None:
        cfg = _valid_run().model_copy(
            update={"policy": SSPolicyConfig(reorder_point=15.0, order_up_to=55.0)}
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_base_stock(self) -> None:
        cfg = _valid_run().model_copy(update={"policy": BaseStockPolicyConfig(target_level=55.0)})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_RS(self) -> None:
        cfg = _valid_run().model_copy(
            update={"policy": PeriodicReviewPolicyConfig(review_period=7, order_up_to=55.0)}
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_poisson_demand(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": PoissonDemandConfig(rate=8.5)})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_negative_binomial_demand(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": NegativeBinomialDemandConfig(n=7.5, p=0.4)})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_gamma_demand(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": GammaDemandConfig(shape=4.0, scale=2.5)})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_lognormal_demand(self) -> None:
        cfg = _valid_run().model_copy(
            update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)}
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_empirical_demand(self) -> None:
        # RunConfig with explicit EmpiricalDemandConfig round-trips
        # through YAML preserving the history_path field (Path is
        # serialized as string by Pydantic; loads back as Path).
        cfg = _valid_run().model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_pattern_stationary(self) -> None:
        # The Pydantic ``default_factory`` for ``RunConfig.pattern`` already
        # populates an implicit Stationary. Explicit construction must also
        # round-trip through YAML preserving the ``kind: stationary`` field.
        cfg = _valid_run().model_copy(update={"pattern": StationaryPatternConfig()})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_pattern_seasonal(self) -> None:
        # RunConfig with explicit SeasonalPatternConfig round-trips through
        # YAML preserving all three fields (amplitude, period, phase) plus
        # the kind discriminator.
        cfg = _valid_run().model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_pattern_trending(self) -> None:
        # RunConfig with explicit TrendingPatternConfig round-trips through
        # YAML preserving the single slope field plus the kind discriminator.
        cfg = _valid_run().model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_normal_leadtime(self) -> None:
        # RunConfig with explicit NormalLeadTimeConfig round-trips through YAML
        # preserving mean + std + the kind discriminator.
        cfg = _valid_run().model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_gamma_leadtime(self) -> None:
        # RunConfig with explicit GammaLeadTimeConfig round-trips through YAML
        # preserving shape + scale + the kind discriminator.
        cfg = _valid_run().model_copy(
            update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)}
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg

    def test_yaml_roundtrip_lognormal_leadtime(self) -> None:
        # RunConfig with explicit LognormalLeadTimeConfig round-trips through YAML
        # preserving mu + sigma + the kind discriminator.
        cfg = _valid_run().model_copy(
            update={"lead_time": LognormalLeadTimeConfig(mu=1.045931, sigma=0.324593)}
        )
        assert RunConfig.from_yaml(cfg.to_yaml()) == cfg


class TestHash:
    def test_hash_deterministic(self) -> None:
        cfg = _valid_run()
        assert cfg.config_hash() == cfg.config_hash()

    def test_hash_changes_with_seed(self) -> None:
        cfg1 = _valid_run()
        cfg2 = cfg1.model_copy(update={"master_seed": cfg1.master_seed + 1})
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_invariant_to_yaml_field_order(self) -> None:
        cfg = _valid_run()
        d = yaml.safe_load(cfg.to_yaml())
        reordered_yaml = yaml.safe_dump(dict(reversed(list(d.items()))), sort_keys=False)
        assert RunConfig.from_yaml(reordered_yaml).config_hash() == cfg.config_hash()

    def test_hash_changes_with_policy_kind(self) -> None:
        # sQ vs sS at superficially-similar params hash differently — proves
        # the discriminator field flows into the canonical JSON, not just the
        # numeric params.
        base = _valid_run()
        sS_cfg = base.model_copy(
            update={"policy": SSPolicyConfig(reorder_point=20.0, order_up_to=60.0)}
        )
        assert base.config_hash() != sS_cfg.config_hash()

    def test_hash_changes_for_base_stock_kind(self) -> None:
        # Same discriminator-into-hash check, base-stock variant.
        base = _valid_run()
        bs_cfg = base.model_copy(update={"policy": BaseStockPolicyConfig(target_level=60.0)})
        assert base.config_hash() != bs_cfg.config_hash()

    def test_hash_changes_for_RS_kind(self) -> None:
        # Same discriminator-into-hash check, (R,S) variant.
        base = _valid_run()
        rs_cfg = base.model_copy(
            update={"policy": PeriodicReviewPolicyConfig(review_period=7, order_up_to=60.0)}
        )
        assert base.config_hash() != rs_cfg.config_hash()

    def test_hash_changes_for_poisson_demand_kind(self) -> None:
        # Same discriminator-into-hash check on the demand axis.
        base = _valid_run()
        poisson_cfg = base.model_copy(update={"demand": PoissonDemandConfig(rate=10.0)})
        assert base.config_hash() != poisson_cfg.config_hash()

    def test_hash_changes_for_negative_binomial_demand_kind(self) -> None:
        # Discriminator flows into the canonical JSON / SHA-256 hash. Two
        # NegBin configs with different params hash differently; NegBin
        # vs Poisson with the same numerical mean also hash differently
        # (the discriminator + field shape drives differentiation).
        base = _valid_run()
        negbin_cfg = base.model_copy(update={"demand": NegativeBinomialDemandConfig(n=10.0, p=0.5)})
        negbin_other_params = base.model_copy(
            update={"demand": NegativeBinomialDemandConfig(n=5.0, p=0.25)}
        )
        poisson_same_mean = base.model_copy(update={"demand": PoissonDemandConfig(rate=10.0)})
        assert base.config_hash() != negbin_cfg.config_hash()
        assert negbin_cfg.config_hash() != negbin_other_params.config_hash()
        assert negbin_cfg.config_hash() != poisson_same_mean.config_hash()

    def test_hash_changes_for_gamma_demand_kind(self) -> None:
        # Same discriminator-into-hash check, Gamma variant. Two Gamma
        # configs with different (shape, scale) hash differently; Gamma
        # vs NegBin at the same first two moments (mean=10, var=20)
        # hash differently — discriminator + field shape drives
        # differentiation even when the resulting distributions have
        # matching moments.
        base = _valid_run()
        gamma_cfg = base.model_copy(update={"demand": GammaDemandConfig(shape=5.0, scale=2.0)})
        gamma_other_params = base.model_copy(
            update={"demand": GammaDemandConfig(shape=10.0, scale=1.0)}
        )
        negbin_same_moments = base.model_copy(
            update={"demand": NegativeBinomialDemandConfig(n=10.0, p=0.5)}
        )
        assert base.config_hash() != gamma_cfg.config_hash()
        assert gamma_cfg.config_hash() != gamma_other_params.config_hash()
        assert gamma_cfg.config_hash() != negbin_same_moments.config_hash()

    def test_hash_changes_for_lognormal_demand_kind(self) -> None:
        # Same discriminator-into-hash check, Lognormal variant. Two
        # Lognormal configs with different (mu, sigma) hash differently;
        # Lognormal vs Gamma at matched first two moments (mean=10,
        # variance≈20) hash differently — discriminator + field shape
        # drives differentiation even when the resulting distributions
        # have matching moments. This closes the canonical-hash
        # uniqueness check across the full M2 demand quartet.
        base = _valid_run()
        lognormal_cfg = base.model_copy(
            update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)}
        )
        lognormal_other_params = base.model_copy(
            update={"demand": LognormalDemandConfig(mu=0.0, sigma=1.0)}
        )
        gamma_same_moments = base.model_copy(
            update={"demand": GammaDemandConfig(shape=5.0, scale=2.0)}
        )
        assert base.config_hash() != lognormal_cfg.config_hash()
        assert lognormal_cfg.config_hash() != lognormal_other_params.config_hash()
        assert lognormal_cfg.config_hash() != gamma_same_moments.config_hash()

    def test_hash_changes_for_empirical_kind(self) -> None:
        # Empirical demand chassis. RunConfigs differing only
        # in demand.kind (Empirical vs Normal vs Lognormal) hash
        # differently after the 5→6 widening. Extends the
        # discriminator-into-canonical-JSON flow.
        base = _valid_run()
        empirical_cfg = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )
        lognormal_cfg = base.model_copy(
            update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)}
        )
        assert base.config_hash() != empirical_cfg.config_hash()
        assert lognormal_cfg.config_hash() != empirical_cfg.config_hash()

    def test_hash_changes_for_empirical_path(self) -> None:
        # Two EmpiricalDemandConfigs with different history_path values
        # produce different hashes. Confirms the path field flows into
        # canonical JSON; the determinism caveat (path-in-hash;
        # file-content-NOT-in-hash) is consistent with this behavior.
        base = _valid_run()
        cfg_a = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )
        cfg_b = base.model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_other.parquet"))
            }
        )
        assert cfg_a.config_hash() != cfg_b.config_hash()

    def test_hash_matches_for_implicit_and_explicit_stationary(self) -> None:
        # The Pydantic ``default_factory`` produces a ``StationaryPatternConfig``
        # whenever ``pattern`` is omitted. The implicit and explicit forms must
        # serialize to identical canonical JSON and therefore identical hashes —
        # the cornerstone of the backward-compat thesis.
        implicit = _valid_run()
        explicit = _valid_run().model_copy(update={"pattern": StationaryPatternConfig()})
        assert implicit.config_hash() == explicit.config_hash()

    def test_hash_changes_when_pattern_kind_changes(self) -> None:
        # Closes the deferred slot from with two pattern arms
        # registered (Stationary + Seasonal), the discriminator flows into
        # the canonical JSON / SHA-256 hash. RunConfigs differing only in
        # ``pattern.kind`` produce different hashes.
        stationary = _valid_run()
        seasonal = _valid_run().model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        assert stationary.config_hash() != seasonal.config_hash()

    def test_hash_changes_for_seasonal_params(self) -> None:
        # Two SeasonalPatternConfigs with different (amplitude, period,
        # phase) produce different hashes. Confirms all three fields flow
        # into the canonical JSON, not just the discriminator.
        base = _valid_run()
        cfg_a = base.model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        cfg_b = base.model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=24, phase=0.0)}
        )
        cfg_c = base.model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.3, period=12, phase=0.0)}
        )
        cfg_d = base.model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=1.0)}
        )
        assert cfg_a.config_hash() != cfg_b.config_hash()  # period differs
        assert cfg_a.config_hash() != cfg_c.config_hash()  # amplitude differs
        assert cfg_a.config_hash() != cfg_d.config_hash()  # phase differs

    def test_hash_changes_for_trending_kind(self) -> None:
        # Pins the Trending discriminator's contribution to the canonical
        # hash after the 2→3 widening. RunConfigs differing only in
        # ``pattern.kind`` (Stationary vs Trending, Seasonal vs Trending)
        # produce different hashes.
        stationary = _valid_run()
        trending = _valid_run().model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        seasonal = _valid_run().model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        assert stationary.config_hash() != trending.config_hash()
        assert seasonal.config_hash() != trending.config_hash()

    def test_hash_changes_for_trending_slope(self) -> None:
        # Two TrendingPatternConfigs with different slope values produce
        # different hashes. Confirms the single field flows into the
        # canonical JSON.
        base = _valid_run()
        cfg_a = base.model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        cfg_b = base.model_copy(update={"pattern": TrendingPatternConfig(slope=0.05)})
        cfg_c = base.model_copy(update={"pattern": TrendingPatternConfig(slope=-0.01)})
        assert cfg_a.config_hash() != cfg_b.config_hash()
        assert cfg_a.config_hash() != cfg_c.config_hash()
        assert cfg_b.config_hash() != cfg_c.config_hash()

    def test_hash_changes_for_intermittent_probability(self) -> None:
        # Two IntermittentPatternConfigs with different probabilities produce different hashes.
        base = _valid_run()
        cfg_a = base.model_copy(
            update={"pattern": IntermittentPatternConfig(occurrence_probability=0.6)}
        )
        cfg_b = base.model_copy(
            update={"pattern": IntermittentPatternConfig(occurrence_probability=0.3)}
        )
        assert cfg_a.config_hash() != cfg_b.config_hash()

    def test_hash_changes_for_lumpy_params(self) -> None:
        # Lumpy configs differing in EITHER occurrence_probability OR burst_multiplier hash
        # differently — both fields flow into the canonical JSON.
        base = _valid_run()
        cfg_a = base.model_copy(
            update={"pattern": LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)}
        )
        cfg_b = base.model_copy(
            update={"pattern": LumpyPatternConfig(occurrence_probability=0.5, burst_multiplier=5.0)}
        )
        cfg_c = base.model_copy(
            update={"pattern": LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=8.0)}
        )
        assert cfg_a.config_hash() != cfg_b.config_hash()  # probability differs
        assert cfg_a.config_hash() != cfg_c.config_hash()  # multiplier differs

    def test_hash_changes_for_leadtime_kind(self) -> None:
        # LeadTimeConfig widened 1→2 (Deterministic + Normal).
        # RunConfigs differing only in lead_time.kind hash differently.
        base = _valid_run()
        normal_lt = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        assert base.config_hash() != normal_lt.config_hash()

    def test_hash_changes_for_normal_leadtime_params(self) -> None:
        # mean and std both flow into the canonical JSON / hash.
        base = _valid_run()
        cfg_a = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        cfg_b = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=5.0, std=1.0)})
        cfg_c = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=2.0)})
        assert cfg_a.config_hash() != cfg_b.config_hash()
        assert cfg_a.config_hash() != cfg_c.config_hash()

    def test_hash_changes_for_gamma_leadtime_kind_and_params(self) -> None:
        # LeadTimeConfig widened 2→3. Gamma kind hashes differently
        # from deterministic + normal; shape and scale both flow into the hash.
        base = _valid_run()
        normal_lt = base.model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        gamma_a = base.model_copy(update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=0.5)})
        gamma_b = base.model_copy(update={"lead_time": GammaLeadTimeConfig(shape=4.0, scale=0.5)})
        gamma_c = base.model_copy(update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=0.8)})
        assert base.config_hash() != gamma_a.config_hash()
        assert normal_lt.config_hash() != gamma_a.config_hash()
        assert gamma_a.config_hash() != gamma_b.config_hash()
        assert gamma_a.config_hash() != gamma_c.config_hash()

    def test_hash_changes_for_lognormal_leadtime_kind_and_params(self) -> None:
        # LeadTimeConfig widened 3→4. Lognormal kind hashes differently
        # from the other arms; mu and sigma both flow into the hash.
        base = _valid_run()
        gamma_lt = base.model_copy(update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=0.5)})
        ln_a = base.model_copy(update={"lead_time": LognormalLeadTimeConfig(mu=1.05, sigma=0.32)})
        ln_b = base.model_copy(update={"lead_time": LognormalLeadTimeConfig(mu=1.50, sigma=0.32)})
        ln_c = base.model_copy(update={"lead_time": LognormalLeadTimeConfig(mu=1.05, sigma=0.50)})
        assert base.config_hash() != ln_a.config_hash()
        assert gamma_lt.config_hash() != ln_a.config_hash()
        assert ln_a.config_hash() != ln_b.config_hash()
        assert ln_a.config_hash() != ln_c.config_hash()

    def test_hash_matches_for_implicit_and_explicit_no_disruption(self) -> None:
        # The Pydantic ``default_factory`` produces a ``NoDisruptionConfig``
        # whenever ``disruption`` is omitted. The implicit and explicit forms
        # must serialize to identical canonical JSON and therefore identical
        # hashes — the backward-compat thesis (mirrors the
        # Stationary-pattern default).
        implicit = _valid_run()
        explicit = _valid_run().model_copy(update={"disruption": NoDisruptionConfig()})
        assert implicit.config_hash() == explicit.config_hash()

    def test_hash_changes_for_disruption_kind_and_params(self) -> None:
        # A scheduled disruption hashes differently from the no-disruption
        # default; the window fields (start, duration, multiplier) all flow into
        # the canonical JSON.
        base = _valid_run()
        sched = base.model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=20, multiplier=2.0)]
                )
            }
        )
        other_mult = base.model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=20, multiplier=3.0)]
                )
            }
        )
        assert base.config_hash() != sched.config_hash()
        assert sched.config_hash() != other_mult.config_hash()


class TestFrozen:
    def test_mutation_rejected(self) -> None:
        cfg = _valid_run()
        with pytest.raises(ValidationError):
            cfg.master_seed = 99


class TestSSPolicyConfig:
    def test_basic_construction(self) -> None:
        cfg = SSPolicyConfig(reorder_point=20.0, order_up_to=60.0)
        assert cfg.kind == "sS"
        assert cfg.reorder_point == 20.0
        assert cfg.order_up_to == 60.0

    def test_s_equal_to_S_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reorder_point < order_up_to"):
            SSPolicyConfig(reorder_point=60.0, order_up_to=60.0)

    def test_s_greater_than_S_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reorder_point < order_up_to"):
            SSPolicyConfig(reorder_point=70.0, order_up_to=60.0)

    def test_negative_S_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SSPolicyConfig(reorder_point=10.0, order_up_to=-1.0)

    def test_zero_S_rejected(self) -> None:
        # PositiveFloat is strict (> 0), so zero must be rejected.
        with pytest.raises(ValidationError):
            SSPolicyConfig(reorder_point=-10.0, order_up_to=0.0)

    def test_negative_s_with_positive_S_allowed(self) -> None:
        # Backorder mode can have negative reorder_point; only s < S is required.
        cfg = SSPolicyConfig(reorder_point=-50.0, order_up_to=10.0)
        assert cfg.reorder_point == -50.0
        assert cfg.order_up_to == 10.0


class TestBaseStockPolicyConfig:
    def test_basic_construction(self) -> None:
        cfg = BaseStockPolicyConfig(target_level=60.0)
        assert cfg.kind == "base_stock"
        assert cfg.target_level == 60.0

    def test_kind_defaults_to_base_stock(self) -> None:
        # YAML files that omit ``kind`` cannot benefit from the default
        # because the discriminated-union dispatch needs ``kind`` to pick
        # the arm. But constructing the class directly should still default.
        cfg = BaseStockPolicyConfig(target_level=60.0)
        assert cfg.kind == "base_stock"

    def test_zero_target_level_rejected(self) -> None:
        # PositiveFloat is strict (> 0); zero must be rejected.
        with pytest.raises(ValidationError):
            BaseStockPolicyConfig(target_level=0.0)

    def test_negative_target_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseStockPolicyConfig(target_level=-1.0)


class TestPeriodicReviewPolicyConfig:
    def test_basic_construction(self) -> None:
        cfg = PeriodicReviewPolicyConfig(review_period=7, order_up_to=60.0)
        assert cfg.kind == "RS"
        assert cfg.review_period == 7
        assert cfg.order_up_to == 60.0

    def test_zero_review_period_rejected(self) -> None:
        # PositiveInt is strict (>= 1).
        with pytest.raises(ValidationError):
            PeriodicReviewPolicyConfig(review_period=0, order_up_to=60.0)

    def test_negative_review_period_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PeriodicReviewPolicyConfig(review_period=-1, order_up_to=60.0)

    def test_zero_order_up_to_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PeriodicReviewPolicyConfig(review_period=7, order_up_to=0.0)

    def test_negative_order_up_to_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PeriodicReviewPolicyConfig(review_period=7, order_up_to=-1.0)

    def test_review_period_one_allowed(self) -> None:
        # Corner case: R=1 is the degenerate "every period is a review"
        # configuration. Functionally equivalent to base-stock but is a
        # valid (R,S) instance.
        cfg = PeriodicReviewPolicyConfig(review_period=1, order_up_to=60.0)
        assert cfg.review_period == 1


class TestPolicyDiscriminator:
    def test_sQ_kind_resolves_to_SQPolicyConfig(self) -> None:
        cfg = _valid_run()
        assert isinstance(cfg.policy, SQPolicyConfig)
        assert cfg.policy.kind == "sQ"

    def test_sS_kind_resolves_to_SSPolicyConfig(self) -> None:
        cfg = _valid_run().model_copy(
            update={"policy": SSPolicyConfig(reorder_point=20.0, order_up_to=60.0)}
        )
        assert isinstance(cfg.policy, SSPolicyConfig)
        assert cfg.policy.kind == "sS"

    def test_base_stock_kind_resolves_to_BaseStockPolicyConfig(self) -> None:
        cfg = _valid_run().model_copy(update={"policy": BaseStockPolicyConfig(target_level=60.0)})
        assert isinstance(cfg.policy, BaseStockPolicyConfig)
        assert cfg.policy.kind == "base_stock"

    def test_RS_kind_resolves_to_PeriodicReviewPolicyConfig(self) -> None:
        cfg = _valid_run().model_copy(
            update={"policy": PeriodicReviewPolicyConfig(review_period=7, order_up_to=60.0)}
        )
        assert isinstance(cfg.policy, PeriodicReviewPolicyConfig)
        assert cfg.policy.kind == "RS"

    def test_yaml_dispatches_via_discriminator(self) -> None:
        cfg = _valid_run().model_copy(
            update={"policy": SSPolicyConfig(reorder_point=15.0, order_up_to=50.0)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.policy, SSPolicyConfig)
        assert roundtripped.policy.reorder_point == 15.0
        assert roundtripped.policy.order_up_to == 50.0

    def test_yaml_dispatches_to_base_stock(self) -> None:
        cfg = _valid_run().model_copy(update={"policy": BaseStockPolicyConfig(target_level=42.5)})
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.policy, BaseStockPolicyConfig)
        assert roundtripped.policy.target_level == 42.5

    def test_yaml_dispatches_to_RS(self) -> None:
        cfg = _valid_run().model_copy(
            update={"policy": PeriodicReviewPolicyConfig(review_period=14, order_up_to=42.5)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.policy, PeriodicReviewPolicyConfig)
        assert roundtripped.policy.review_period == 14
        assert roundtripped.policy.order_up_to == 42.5

    def test_unknown_kind_rejected(self) -> None:
        cfg = _valid_run()
        d = cfg.model_dump(mode="json")
        d["policy"] = {"kind": "bogus", "reorder_point": 15.0, "order_up_to": 50.0}
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)


class TestPoissonDemandConfig:
    def test_basic_construction(self) -> None:
        cfg = PoissonDemandConfig(rate=10.0)
        assert cfg.kind == "poisson"
        assert cfg.rate == 10.0

    def test_zero_rate_rejected(self) -> None:
        # PositiveFloat is strict (> 0); zero must be rejected.
        # Degenerate λ=0 ("always zero demand") is not a Poisson; it would
        # belong to a separate ConstantDemandConfig if ever needed.
        with pytest.raises(ValidationError):
            PoissonDemandConfig(rate=0.0)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PoissonDemandConfig(rate=-1.0)

    def test_high_rate_allowed(self) -> None:
        # No upper bound — high-volume SKUs get high λ; the simulator
        # must accept the full positive range.
        cfg = PoissonDemandConfig(rate=1e6)
        assert cfg.rate == 1e6


class TestNegativeBinomialDemandConfig:
    def test_basic_construction(self) -> None:
        cfg = NegativeBinomialDemandConfig(n=10.0, p=0.5)
        assert cfg.kind == "negative_binomial"
        assert cfg.n == 10.0
        assert cfg.p == 0.5

    def test_zero_n_rejected(self) -> None:
        # PositiveFloat is strict (> 0).
        with pytest.raises(ValidationError):
            NegativeBinomialDemandConfig(n=0.0, p=0.5)

    def test_negative_n_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NegativeBinomialDemandConfig(n=-1.0, p=0.5)

    def test_high_n_allowed(self) -> None:
        # No upper bound on n; high-overdispersion regimes are allowed.
        cfg = NegativeBinomialDemandConfig(n=1e6, p=0.5)
        assert cfg.n == 1e6

    def test_float_n_allowed_via_gamma_poisson_mixture(self) -> None:
        # Non-integer n is the Gamma-Poisson mixture interpretation;
        # numpy accepts it and the config must too.
        cfg = NegativeBinomialDemandConfig(n=2.5, p=0.5)
        assert cfg.n == 2.5

    def test_p_at_zero_rejected(self) -> None:
        # Open interval (0, 1) — exclusive lower bound.
        with pytest.raises(ValidationError):
            NegativeBinomialDemandConfig(n=10.0, p=0.0)

    def test_p_at_one_rejected(self) -> None:
        # Open interval (0, 1) — exclusive upper bound.
        with pytest.raises(ValidationError):
            NegativeBinomialDemandConfig(n=10.0, p=1.0)

    def test_p_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NegativeBinomialDemandConfig(n=10.0, p=1.5)

    def test_p_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NegativeBinomialDemandConfig(n=10.0, p=-0.1)

    def test_p_near_boundaries_accepted(self) -> None:
        # Tight values inside the open interval are accepted; the bounds
        # are STRICTLY open but anything strictly inside is fine.
        cfg_low = NegativeBinomialDemandConfig(n=10.0, p=0.001)
        cfg_high = NegativeBinomialDemandConfig(n=10.0, p=0.999)
        assert cfg_low.p == 0.001
        assert cfg_high.p == 0.999


class TestGammaDemandConfig:
    def test_basic_construction(self) -> None:
        cfg = GammaDemandConfig(shape=5.0, scale=2.0)
        assert cfg.kind == "gamma"
        assert cfg.shape == 5.0
        assert cfg.scale == 2.0

    def test_zero_shape_rejected(self) -> None:
        # PositiveFloat is strict (> 0).
        with pytest.raises(ValidationError):
            GammaDemandConfig(shape=0.0, scale=2.0)

    def test_negative_shape_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaDemandConfig(shape=-1.0, scale=2.0)

    def test_zero_scale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaDemandConfig(shape=5.0, scale=0.0)

    def test_negative_scale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaDemandConfig(shape=5.0, scale=-1.0)

    def test_high_shape_allowed(self) -> None:
        # No upper bound on shape; high-shape regimes (where Gamma
        # converges to Normal) are valid.
        cfg = GammaDemandConfig(shape=1e6, scale=1.0)
        assert cfg.shape == 1e6

    def test_high_scale_allowed(self) -> None:
        # No upper bound on scale.
        cfg = GammaDemandConfig(shape=1.0, scale=1e6)
        assert cfg.scale == 1e6

    def test_small_values_accepted(self) -> None:
        # No lower bound except strict positivity; very small values
        # (mode-at-zero regime) are valid.
        cfg = GammaDemandConfig(shape=1e-6, scale=1e-6)
        assert cfg.shape == 1e-6
        assert cfg.scale == 1e-6


class TestLognormalDemandConfig:
    def test_basic_construction(self) -> None:
        cfg = LognormalDemandConfig(mu=2.2114, sigma=0.4271)
        assert cfg.kind == "lognormal"
        assert cfg.mu == 2.2114
        assert cfg.sigma == 0.4271

    def test_negative_mu_allowed(self) -> None:
        # First demand config field that accepts negative values.
        # Mathematically correct: Lognormal supports any real mu (mean
        # = exp(mu + sigma²/2) is positive for any real mu). Slow-moving
        # SKUs with mean < 1 require mu < 0 (e.g., mu = -2 gives
        # mean ≈ 0.135).
        cfg = LognormalDemandConfig(mu=-2.0, sigma=0.5)
        assert cfg.mu == -2.0

    def test_zero_mu_allowed(self) -> None:
        # mu = 0 is valid: resulting mean = exp(sigma²/2) > 1 for any
        # sigma > 0.
        cfg = LognormalDemandConfig(mu=0.0, sigma=1.0)
        assert cfg.mu == 0.0

    def test_large_mu_allowed(self) -> None:
        # mu = 10 gives mean ≈ exp(10) ≈ 22026; numerically valid.
        cfg = LognormalDemandConfig(mu=10.0, sigma=0.5)
        assert cfg.mu == 10.0

    def test_zero_sigma_rejected(self) -> None:
        # PositiveFloat is strict (> 0). sigma = 0 is the degenerate
        # point mass at exp(mu) — not a Lognormal.
        with pytest.raises(ValidationError):
            LognormalDemandConfig(mu=2.0, sigma=0.0)

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalDemandConfig(mu=2.0, sigma=-0.5)

    def test_large_sigma_allowed(self) -> None:
        # No upper bound on sigma; sigma = 3.0 is within Hypothesis
        # bounds and produces a well-defined (if heavy-tailed) Lognormal.
        cfg = LognormalDemandConfig(mu=0.0, sigma=3.0)
        assert cfg.sigma == 3.0

    def test_small_sigma_accepted(self) -> None:
        # No lower bound except strict positivity; very small sigma
        # (highly concentrated near exp(mu)) is valid.
        cfg = LognormalDemandConfig(mu=2.0, sigma=1e-6)
        assert cfg.sigma == 1e-6


class TestEmpiricalDemandConfig:
    """First non-parametric demand arm. Single field ``history_path: Path``
    pointing at a parquet with a ``demand`` column. File-content
    validation happens at engine ``__init__`` time, not config-load time
    — Pydantic stays I/O-free.
    """

    def test_basic_construction(self) -> None:
        cfg = EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
        assert cfg.kind == "empirical"
        assert cfg.history_path == Path("data/m5/sample_smooth.parquet")

    def test_string_path_coerced_to_Path(self) -> None:
        # Pydantic v2 coerces str → Path natively for Path-typed fields.
        cfg = EmpiricalDemandConfig.model_validate(
            {"kind": "empirical", "history_path": "data/m5/sample_smooth.parquet"}
        )
        assert isinstance(cfg.history_path, Path)
        assert cfg.history_path == Path("data/m5/sample_smooth.parquet")

    def test_relative_and_absolute_path_both_accepted(self) -> None:
        rel = EmpiricalDemandConfig(history_path=Path("a/b.parquet"))
        absolute = EmpiricalDemandConfig(history_path=Path("/tmp/x.parquet"))
        assert rel.history_path == Path("a/b.parquet")
        assert absolute.history_path == Path("/tmp/x.parquet")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EmpiricalDemandConfig.model_validate(
                {
                    "kind": "empirical",
                    "history_path": "data/m5/sample_smooth.parquet",
                    "unknown": 1,
                }
            )

    def test_frozen_immutable(self) -> None:
        cfg = EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
        with pytest.raises(ValidationError):
            # Re-assigning even the same value triggers the frozen check.
            cfg.history_path = Path("data/m5/sample_smooth.parquet")

    def test_kind_field_defaults_from_yaml(self) -> None:
        # Omitting ``kind`` in YAML produces ``kind == "empirical"`` via
        # the Literal default. Same convention as the parametric arms.
        cfg = EmpiricalDemandConfig.model_validate(
            {"history_path": "data/m5/sample_smooth.parquet"}
        )
        assert cfg.kind == "empirical"


class TestDemandDiscriminator:
    """DemandConfig widened from a single alias to a 2-arm union; then to
    3 arms (Normal + Poisson + NegBin); then to 4 arms (+ Gamma); then to
    the full 5-arm quartet (Normal + Poisson + NegBin +
    Gamma + Lognormal). These tests pin the discriminator routing on
    the demand axis.
    """

    def test_normal_kind_resolves_to_NormalDemandConfig(self) -> None:
        cfg = _valid_run()
        assert isinstance(cfg.demand, NormalDemandConfig)
        assert cfg.demand.kind == "normal"

    def test_poisson_kind_resolves_to_PoissonDemandConfig(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": PoissonDemandConfig(rate=10.0)})
        assert isinstance(cfg.demand, PoissonDemandConfig)
        assert cfg.demand.kind == "poisson"

    def test_negative_binomial_kind_resolves_to_NegativeBinomialDemandConfig(
        self,
    ) -> None:
        cfg = _valid_run().model_copy(
            update={"demand": NegativeBinomialDemandConfig(n=10.0, p=0.5)}
        )
        assert isinstance(cfg.demand, NegativeBinomialDemandConfig)
        assert cfg.demand.kind == "negative_binomial"

    def test_gamma_kind_resolves_to_GammaDemandConfig(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": GammaDemandConfig(shape=5.0, scale=2.0)})
        assert isinstance(cfg.demand, GammaDemandConfig)
        assert cfg.demand.kind == "gamma"

    def test_lognormal_kind_resolves_to_LognormalDemandConfig(self) -> None:
        cfg = _valid_run().model_copy(
            update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)}
        )
        assert isinstance(cfg.demand, LognormalDemandConfig)
        assert cfg.demand.kind == "lognormal"

    def test_empirical_kind_resolves_to_EmpiricalDemandConfig(self) -> None:
        # first non-parametric arm. Pydantic discriminator
        # dispatches `kind: empirical` to the right config class after
        # the 5→6 widening.
        cfg = _valid_run().model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )
        assert isinstance(cfg.demand, EmpiricalDemandConfig)
        assert cfg.demand.kind == "empirical"

    def test_yaml_dispatches_to_poisson(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": PoissonDemandConfig(rate=42.5)})
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.demand, PoissonDemandConfig)
        assert roundtripped.demand.rate == 42.5

    def test_yaml_dispatches_to_negative_binomial(self) -> None:
        cfg = _valid_run().model_copy(
            update={"demand": NegativeBinomialDemandConfig(n=12.5, p=0.3)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.demand, NegativeBinomialDemandConfig)
        assert roundtripped.demand.n == 12.5
        assert roundtripped.demand.p == 0.3

    def test_yaml_dispatches_to_gamma(self) -> None:
        cfg = _valid_run().model_copy(update={"demand": GammaDemandConfig(shape=4.0, scale=2.5)})
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.demand, GammaDemandConfig)
        assert roundtripped.demand.shape == 4.0
        assert roundtripped.demand.scale == 2.5

    def test_yaml_dispatches_to_lognormal(self) -> None:
        cfg = _valid_run().model_copy(
            update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.demand, LognormalDemandConfig)
        assert roundtripped.demand.mu == 2.2114
        assert roundtripped.demand.sigma == 0.4271

    def test_yaml_dispatches_to_empirical(self) -> None:
        # YAML containing `demand: {kind: empirical, history_path: ...}`
        # parses through the 6-arm discriminated union to an
        # EmpiricalDemandConfig with the right path value.
        cfg = _valid_run().model_copy(
            update={
                "demand": EmpiricalDemandConfig(history_path=Path("data/m5/sample_smooth.parquet"))
            }
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.demand, EmpiricalDemandConfig)
        assert roundtripped.demand.history_path == Path("data/m5/sample_smooth.parquet")

    def test_6way_widening_preserves_all_parametric_arms(self) -> None:
        # Backward-compat regression: after the 5→6 widening (Empirical
        # was added), all 5 parametric arms still parse correctly.
        # Locks the chassis-additivity contract.
        normal_cfg = _valid_run()
        assert isinstance(RunConfig.from_yaml(normal_cfg.to_yaml()).demand, NormalDemandConfig)

        poisson_cfg = _valid_run().model_copy(update={"demand": PoissonDemandConfig(rate=10.0)})
        assert isinstance(RunConfig.from_yaml(poisson_cfg.to_yaml()).demand, PoissonDemandConfig)

        negbin_cfg = _valid_run().model_copy(
            update={"demand": NegativeBinomialDemandConfig(n=10.0, p=0.5)}
        )
        assert isinstance(
            RunConfig.from_yaml(negbin_cfg.to_yaml()).demand, NegativeBinomialDemandConfig
        )

        gamma_cfg = _valid_run().model_copy(
            update={"demand": GammaDemandConfig(shape=5.0, scale=2.0)}
        )
        assert isinstance(RunConfig.from_yaml(gamma_cfg.to_yaml()).demand, GammaDemandConfig)

        lognormal_cfg = _valid_run().model_copy(
            update={"demand": LognormalDemandConfig(mu=2.2114, sigma=0.4271)}
        )
        assert isinstance(
            RunConfig.from_yaml(lognormal_cfg.to_yaml()).demand, LognormalDemandConfig
        )

    def test_yaml_dispatches_to_all_four_existing_kinds_after_5way_widening(
        self,
    ) -> None:
        # Backward-compat regression guard: existing Normal, Poisson,
        # NegBin AND Gamma scenarios MUST continue to validate after the
        # fifth arm (Lognormal) lands. Locks against accidental
        # discriminator-routing breakage. Closes the M2 demand quartet
        # regression-guard chain.
        normal_cfg = _valid_run()
        normal_rt = RunConfig.from_yaml(normal_cfg.to_yaml())
        assert isinstance(normal_rt.demand, NormalDemandConfig)
        assert normal_rt.demand.kind == "normal"
        assert normal_rt == normal_cfg

        poisson_cfg = _valid_run().model_copy(update={"demand": PoissonDemandConfig(rate=10.0)})
        poisson_rt = RunConfig.from_yaml(poisson_cfg.to_yaml())
        assert isinstance(poisson_rt.demand, PoissonDemandConfig)
        assert poisson_rt.demand.kind == "poisson"
        assert poisson_rt == poisson_cfg

        negbin_cfg = _valid_run().model_copy(
            update={"demand": NegativeBinomialDemandConfig(n=10.0, p=0.5)}
        )
        negbin_rt = RunConfig.from_yaml(negbin_cfg.to_yaml())
        assert isinstance(negbin_rt.demand, NegativeBinomialDemandConfig)
        assert negbin_rt.demand.kind == "negative_binomial"
        assert negbin_rt == negbin_cfg

        gamma_cfg = _valid_run().model_copy(
            update={"demand": GammaDemandConfig(shape=5.0, scale=2.0)}
        )
        gamma_rt = RunConfig.from_yaml(gamma_cfg.to_yaml())
        assert isinstance(gamma_rt.demand, GammaDemandConfig)
        assert gamma_rt.demand.kind == "gamma"
        assert gamma_rt == gamma_cfg

    def test_unknown_demand_kind_rejected(self) -> None:
        cfg = _valid_run()
        d = cfg.model_dump(mode="json")
        d["demand"] = {"kind": "bogus", "rate": 5.0}
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)


class TestStationaryPatternConfig:
    """The chassis arm of the pattern engine.

    Identity is fully stateless — the config carries only the discriminator
    field. No moments to set, no amplitude / period / phase / probability;
    those land later with the Seasonal / Trending /
    Intermittent / Lumpy arms.
    """

    def test_basic_construction(self) -> None:
        cfg = StationaryPatternConfig()
        assert cfg.kind == "stationary"

    def test_extra_fields_rejected(self) -> None:
        # frozen + extra="forbid"; the locked _FROZEN config catches typos.
        with pytest.raises(ValidationError):
            StationaryPatternConfig.model_validate({"kind": "stationary", "unknown": 1})

    def test_frozen_immutable(self) -> None:
        cfg = StationaryPatternConfig()
        with pytest.raises(ValidationError):
            # Re-assigning even the same literal value triggers the
            # frozen check — Pydantic rejects any field write on a
            # frozen model regardless of value.
            cfg.kind = "stationary"

    def test_explicit_kind_value(self) -> None:
        # Literal["stationary"] is strict: explicitly passing the literal
        # works; any other string is rejected.
        cfg = StationaryPatternConfig(kind="stationary")
        assert cfg.kind == "stationary"
        with pytest.raises(ValidationError):
            StationaryPatternConfig(kind="seasonal")  # type: ignore[arg-type]


class TestSeasonalPatternConfig:
    """First concrete arm of ``PatternConfig`` — sinusoidal multiplicative
    seasonal modulation. Three independent fields (amplitude, period, phase);
    no cross-field invariant. Locks the mixed-bounds field pattern
    ``Annotated[float, Field(ge=0.0, lt=1.0)]`` (closed at 0, open at 1).
    """

    def test_basic_construction(self) -> None:
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        assert cfg.kind == "seasonal"
        assert cfg.amplitude == 0.5
        assert cfg.period == 12
        assert cfg.phase == 0.0

    def test_zero_amplitude_allowed(self) -> None:
        # amplitude=0 is DEGENERATE-TO-IDENTITY (apply(base, t) == base).
        # Useful for A/B comparisons against Stationary; the bound is
        # closed at 0 (``ge=0.0``) precisely to allow this.
        cfg = SeasonalPatternConfig(amplitude=0.0, period=12, phase=0.0)
        assert cfg.amplitude == 0.0

    def test_negative_amplitude_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SeasonalPatternConfig(amplitude=-0.1, period=12, phase=0.0)

    def test_amplitude_at_one_rejected(self) -> None:
        # Field has ``lt=1.0`` (EXCLUSIVE upper bound). amplitude=1.0 would
        # produce zero-demand troughs (degenerate intermittent behavior);
        # the Intermittent pattern is the right tool for that.
        with pytest.raises(ValidationError):
            SeasonalPatternConfig(amplitude=1.0, period=12, phase=0.0)

    def test_amplitude_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SeasonalPatternConfig(amplitude=1.5, period=12, phase=0.0)

    def test_amplitude_just_below_one_accepted(self) -> None:
        # Pins the open-at-1 boundary: 0.999... is inside the bound.
        cfg = SeasonalPatternConfig(amplitude=0.999, period=12, phase=0.0)
        assert cfg.amplitude == 0.999

    def test_zero_period_rejected(self) -> None:
        # PositiveInt is strict (>= 1).
        with pytest.raises(ValidationError):
            SeasonalPatternConfig(amplitude=0.5, period=0, phase=0.0)

    def test_negative_period_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SeasonalPatternConfig(amplitude=0.5, period=-1, phase=0.0)

    def test_period_one_allowed(self) -> None:
        # period=1 is technically valid but degenerate: sin(2π·t + phase)
        # = sin(phase) for all integer t (since sin has period 2π and t·2π
        # is a full-period shift). The factor is constant across all t.
        # Users who pick period=1 are signaling intent.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=1, phase=0.0)
        assert cfg.period == 1

    def test_high_period_allowed(self) -> None:
        cfg = SeasonalPatternConfig(amplitude=0.5, period=10_000, phase=0.0)
        assert cfg.period == 10_000

    def test_negative_phase_allowed(self) -> None:
        # phase is any real (no bounds): allows shifts in either direction.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=-1.5)
        assert cfg.phase == -1.5

    def test_positive_phase_allowed(self) -> None:
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=math.pi)
        assert cfg.phase == math.pi

    def test_zero_phase_default(self) -> None:
        # phase has a default of 0.0; consumers can omit it.
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12)
        assert cfg.phase == 0.0

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SeasonalPatternConfig.model_validate(
                {"kind": "seasonal", "amplitude": 0.5, "period": 12, "phase": 0.0, "unknown": 1}
            )

    def test_frozen_immutable(self) -> None:
        cfg = SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)
        with pytest.raises(ValidationError):
            # Re-assigning even the same value triggers the frozen check.
            cfg.amplitude = 0.5


class TestTrendingPatternConfig:
    """Second concrete arm of ``PatternConfig`` — linear-multiplicative
    trending modulation. Single field ``slope: FiniteFloat`` (any finite
    real; NaN, +inf, -inf rejected at config-load time as a numeric sanity
    gate, NOT a domain bound). Locks the FiniteFloat pattern for
    unbounded-but-finite numeric fields.
    """

    def test_basic_construction(self) -> None:
        cfg = TrendingPatternConfig(slope=0.02)
        assert cfg.kind == "trending"
        assert cfg.slope == 0.02

    def test_zero_slope_allowed(self) -> None:
        # slope=0 is DEGENERATE-TO-IDENTITY (apply(base, t) = max(0, base * 1)
        # = base for any base >= 0). Useful for A/B comparisons against
        # Stationary; FiniteFloat admits 0 naturally (it is finite).
        cfg = TrendingPatternConfig(slope=0.0)
        assert cfg.slope == 0.0

    def test_positive_slope_allowed(self) -> None:
        cfg = TrendingPatternConfig(slope=0.05)
        assert cfg.slope == 0.05

    def test_negative_slope_allowed(self) -> None:
        # Negative slope captures declining SKUs (sunsetting products).
        # Clip-at-source in apply() handles the steep-negative-slope edge.
        cfg = TrendingPatternConfig(slope=-0.05)
        assert cfg.slope == -0.05

    def test_high_positive_slope_allowed(self) -> None:
        # No upper magnitude bound — finite is the only constraint.
        cfg = TrendingPatternConfig(slope=10.0)
        assert cfg.slope == 10.0

    def test_steep_negative_slope_allowed(self) -> None:
        # No lower magnitude bound — finite is the only constraint.
        # Clip handles the runtime; config admits the value.
        cfg = TrendingPatternConfig(slope=-10.0)
        assert cfg.slope == -10.0

    def test_nan_slope_rejected(self) -> None:
        # FiniteFloat = Annotated[float, Field(allow_inf_nan=False)] — NaN
        # is rejected at config-load. Protects engine's no-NaN/inf ledger
        # invariant from silent corruption.
        with pytest.raises(ValidationError):
            TrendingPatternConfig(slope=float("nan"))

    def test_positive_inf_slope_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrendingPatternConfig(slope=float("inf"))

    def test_negative_inf_slope_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrendingPatternConfig(slope=float("-inf"))

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrendingPatternConfig.model_validate({"kind": "trending", "slope": 0.02, "unknown": 1})

    def test_frozen_immutable(self) -> None:
        cfg = TrendingPatternConfig(slope=0.02)
        with pytest.raises(ValidationError):
            # Re-assigning even the same value triggers the frozen check.
            cfg.slope = 0.02


class TestIntermittentPatternConfig:
    """Third concrete arm of ``PatternConfig`` — the first stochastic arm. Single field
    ``occurrence_probability: Annotated[float, Field(gt=0.0, le=1.0)]`` (open at 0, closed at 1).
    Locks the bounded-probability field pattern.
    """

    def test_basic_construction(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        assert cfg.kind == "intermittent"
        assert cfg.occurrence_probability == 0.6

    def test_probability_one_allowed(self) -> None:
        # p=1 is DEGENERATE-TO-IDENTITY (every period kept; apply == base). Allowed for A/B
        # parity with amplitude=0 / slope=0; le=1.0 is the CLOSED upper bound.
        cfg = IntermittentPatternConfig(occurrence_probability=1.0)
        assert cfg.occurrence_probability == 1.0

    def test_small_positive_probability_allowed(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.01)
        assert cfg.occurrence_probability == 0.01

    def test_zero_probability_rejected(self) -> None:
        # gt=0.0 — p=0 would zero ALL demand forever (no orders ever). Pathological; rejected.
        with pytest.raises(ValidationError):
            IntermittentPatternConfig(occurrence_probability=0.0)

    def test_negative_probability_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntermittentPatternConfig(occurrence_probability=-0.1)

    def test_probability_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntermittentPatternConfig(occurrence_probability=1.5)

    def test_nan_probability_rejected(self) -> None:
        # The bounded Field rejects NaN intrinsically (NaN fails gt/le); no FiniteFloat needed.
        with pytest.raises(ValidationError):
            IntermittentPatternConfig(occurrence_probability=float("nan"))

    def test_inf_probability_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntermittentPatternConfig(occurrence_probability=float("inf"))

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntermittentPatternConfig.model_validate(
                {"kind": "intermittent", "occurrence_probability": 0.6, "unknown": 1}
            )

    def test_frozen_immutable(self) -> None:
        cfg = IntermittentPatternConfig(occurrence_probability=0.6)
        with pytest.raises(ValidationError):
            cfg.occurrence_probability = 0.6


class TestLumpyPatternConfig:
    """Fourth concrete arm of ``PatternConfig`` — closes the pattern set. Two fields:
    ``occurrence_probability`` (as Intermittent) + ``burst_multiplier``
    (``Annotated[float, Field(gt=1.0, allow_inf_nan=False)]`` — strictly > 1, finite).
    """

    def test_basic_construction(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        assert cfg.kind == "lumpy"
        assert cfg.occurrence_probability == 0.3
        assert cfg.burst_multiplier == 5.0

    def test_multiplier_just_above_one_allowed(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=1.01)
        assert cfg.burst_multiplier == 1.01

    def test_large_multiplier_allowed(self) -> None:
        # No upper bound — a 100x burst is a valid (severe) lumpy config.
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=100.0)
        assert cfg.burst_multiplier == 100.0

    def test_multiplier_one_rejected(self) -> None:
        # gt=1.0 (strict) — m=1 collapses Lumpy into Intermittent; use that arm instead.
        with pytest.raises(ValidationError):
            LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=1.0)

    def test_multiplier_below_one_rejected(self) -> None:
        # m<1 would shrink demand (anti-lumpy), not burst it.
        with pytest.raises(ValidationError):
            LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=0.5)

    def test_multiplier_nan_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=float("nan"))

    def test_multiplier_inf_rejected(self) -> None:
        # allow_inf_nan=False — gt=1.0 alone admits +inf (inf > 1.0 is True), which would
        # corrupt the ledger via base * inf.
        with pytest.raises(ValidationError):
            LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=float("inf"))

    def test_zero_probability_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LumpyPatternConfig(occurrence_probability=0.0, burst_multiplier=5.0)

    def test_probability_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LumpyPatternConfig(occurrence_probability=1.5, burst_multiplier=5.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LumpyPatternConfig.model_validate(
                {"kind": "lumpy", "occurrence_probability": 0.3, "burst_multiplier": 5.0, "x": 1}
            )

    def test_frozen_immutable(self) -> None:
        cfg = LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)
        with pytest.raises(ValidationError):
            cfg.burst_multiplier = 5.0


class TestPatternDiscriminator:
    """Pattern discriminator dispatch on the ``kind`` field.

    ``PatternConfig`` began as a single-arm alias for
    ``StationaryPatternConfig``, then widened to a proper
    ``Annotated[StationaryPatternConfig | SeasonalPatternConfig,
    Field(discriminator="kind")]`` discriminated union. These tests pin
    the dispatch routing on the pattern axis; Bullets 12-14 extend the
    union additively without changing the discriminator field name.
    """

    def test_stationary_kind_resolves_to_StationaryPatternConfig(self) -> None:
        cfg = _valid_run()
        assert isinstance(cfg.pattern, StationaryPatternConfig)
        assert cfg.pattern.kind == "stationary"

    def test_yaml_dispatches_to_stationary(self) -> None:
        cfg = _valid_run().model_copy(update={"pattern": StationaryPatternConfig()})
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.pattern, StationaryPatternConfig)
        assert roundtripped.pattern.kind == "stationary"

    def test_yaml_without_pattern_defaults_to_stationary(self) -> None:
        # The critical backward-compat test for a YAML that
        # omits the ``pattern:`` block must still produce a Pydantic-valid
        # RunConfig whose ``pattern`` is a ``StationaryPatternConfig``.
        # All 8 existing scenario YAMLs rely on this property.
        yaml_str = """
master_seed: 42
simulation:
  horizon: 10
  initial_on_hand: 50
demand:
  kind: normal
  mean: 5.0
  std: 1.0
lead_time:
  kind: deterministic
  lead_time: 3
policy:
  kind: sQ
  reorder_point: 10.0
  order_quantity: 20
costs:
  unit_cost: 1.0
  holding_per_unit_per_period: 0.1
  ordering_fixed: 5.0
"""
        cfg = RunConfig.from_yaml(yaml_str)
        assert isinstance(cfg.pattern, StationaryPatternConfig)
        assert cfg.pattern.kind == "stationary"

    def test_seasonal_kind_resolves_to_SeasonalPatternConfig(self) -> None:
        # Pydantic discriminator dispatches `kind: seasonal` to the right
        # config class. Pins the union routing after the 1→2 widening.
        cfg = _valid_run().model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        assert isinstance(cfg.pattern, SeasonalPatternConfig)
        assert cfg.pattern.kind == "seasonal"

    def test_yaml_dispatches_to_seasonal(self) -> None:
        # YAML containing `pattern: {kind: seasonal, amplitude: 0.5, period:
        # 12, phase: 0.0}` parses through the discriminated union to a
        # SeasonalPatternConfig with the right field values.
        cfg = _valid_run().model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.pattern, SeasonalPatternConfig)
        assert roundtripped.pattern.amplitude == 0.5
        assert roundtripped.pattern.period == 12
        assert roundtripped.pattern.phase == 0.0

    def test_yaml_dispatches_to_stationary_after_2way_widening(self) -> None:
        # Backward-compat regression: pattern: {kind: stationary} still
        # parses correctly after PatternConfig widens from 1 to 2 arms.
        # Plus the no-pattern-block default-factory branch.
        explicit_cfg = _valid_run().model_copy(update={"pattern": StationaryPatternConfig()})
        roundtripped = RunConfig.from_yaml(explicit_cfg.to_yaml())
        assert isinstance(roundtripped.pattern, StationaryPatternConfig)
        # Implicit default still resolves correctly post-widening:
        default_cfg = _valid_run()
        assert isinstance(default_cfg.pattern, StationaryPatternConfig)

    def test_trending_kind_resolves_to_TrendingPatternConfig(self) -> None:
        # Pydantic discriminator dispatches `kind: trending` to the right
        # config class. Pins the union routing after the 2→3 widening.
        cfg = _valid_run().model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        assert isinstance(cfg.pattern, TrendingPatternConfig)
        assert cfg.pattern.kind == "trending"

    def test_yaml_dispatches_to_trending(self) -> None:
        # YAML containing `pattern: {kind: trending, slope: 0.02}` parses
        # through the 3-arm discriminated union to a TrendingPatternConfig
        # with the right slope value.
        cfg = _valid_run().model_copy(update={"pattern": TrendingPatternConfig(slope=0.02)})
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.pattern, TrendingPatternConfig)
        assert roundtripped.pattern.slope == 0.02

    def test_3way_widening_preserves_stationary_and_seasonal(self) -> None:
        # Backward-compat regression: after PatternConfig widens from 2 to
        # 3 arms (adding Trending), both Stationary and Seasonal pattern
        # blocks still parse correctly. Plus the no-pattern-block
        # default-factory branch.
        stationary_cfg = _valid_run().model_copy(update={"pattern": StationaryPatternConfig()})
        seasonal_cfg = _valid_run().model_copy(
            update={"pattern": SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0)}
        )
        assert isinstance(
            RunConfig.from_yaml(stationary_cfg.to_yaml()).pattern, StationaryPatternConfig
        )
        assert isinstance(
            RunConfig.from_yaml(seasonal_cfg.to_yaml()).pattern, SeasonalPatternConfig
        )
        # Implicit default still resolves correctly post-3way-widening:
        default_cfg = _valid_run()
        assert isinstance(default_cfg.pattern, StationaryPatternConfig)

    def test_intermittent_kind_resolves_to_IntermittentPatternConfig(self) -> None:
        # Pydantic discriminator dispatches `kind: intermittent` after the 3→4 widening.
        cfg = _valid_run().model_copy(
            update={"pattern": IntermittentPatternConfig(occurrence_probability=0.6)}
        )
        assert isinstance(cfg.pattern, IntermittentPatternConfig)
        assert cfg.pattern.kind == "intermittent"

    def test_yaml_dispatches_to_intermittent(self) -> None:
        cfg = _valid_run().model_copy(
            update={"pattern": IntermittentPatternConfig(occurrence_probability=0.6)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.pattern, IntermittentPatternConfig)
        assert roundtripped.pattern.occurrence_probability == 0.6

    def test_4way_widening_preserves_earlier_arms(self) -> None:
        # Backward-compat: after PatternConfig widens 3→4 (adding Intermittent), the Stationary /
        # Seasonal / Trending blocks and the no-pattern default still resolve correctly.
        for pattern in (
            StationaryPatternConfig(),
            SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0),
            TrendingPatternConfig(slope=0.02),
        ):
            cfg = _valid_run().model_copy(update={"pattern": pattern})
            assert type(RunConfig.from_yaml(cfg.to_yaml()).pattern) is type(pattern)
        assert isinstance(_valid_run().pattern, StationaryPatternConfig)

    def test_lumpy_kind_resolves_to_LumpyPatternConfig(self) -> None:
        # Pydantic discriminator dispatches `kind: lumpy` after the 4→5 widening.
        cfg = _valid_run().model_copy(
            update={"pattern": LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)}
        )
        assert isinstance(cfg.pattern, LumpyPatternConfig)
        assert cfg.pattern.kind == "lumpy"

    def test_yaml_dispatches_to_lumpy(self) -> None:
        cfg = _valid_run().model_copy(
            update={"pattern": LumpyPatternConfig(occurrence_probability=0.3, burst_multiplier=5.0)}
        )
        roundtripped = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(roundtripped.pattern, LumpyPatternConfig)
        assert roundtripped.pattern.occurrence_probability == 0.3
        assert roundtripped.pattern.burst_multiplier == 5.0

    def test_5way_widening_preserves_earlier_arms(self) -> None:
        # Backward-compat: after PatternConfig widens 4→5 (adding Lumpy), all earlier arms and
        # the no-pattern default still resolve correctly.
        for pattern in (
            StationaryPatternConfig(),
            SeasonalPatternConfig(amplitude=0.5, period=12, phase=0.0),
            TrendingPatternConfig(slope=0.02),
            IntermittentPatternConfig(occurrence_probability=0.6),
        ):
            cfg = _valid_run().model_copy(update={"pattern": pattern})
            assert type(RunConfig.from_yaml(cfg.to_yaml()).pattern) is type(pattern)
        assert isinstance(_valid_run().pattern, StationaryPatternConfig)


class TestRunConfigPatternField:
    """Semantics of the new ``RunConfig.pattern`` field."""

    def test_default_factory_produces_stationary(self) -> None:
        # No explicit pattern argument produces a ``StationaryPatternConfig``
        # instance — never None.
        cfg = _valid_run()
        assert isinstance(cfg.pattern, StationaryPatternConfig)

    def test_default_factory_produces_distinct_instances(self) -> None:
        # ``Field(default_factory=lambda: StationaryPatternConfig())`` is a
        # FACTORY, not a singleton. Two RunConfig constructions must yield
        # different ``pattern`` instances. Even though both are frozen, the
        # rule is locked: factory creates a fresh instance per call. This
        # protects against any future code path that might mutate ``_pattern``.
        cfg_a = _valid_run()
        cfg_b = _valid_run()
        assert cfg_a.pattern is not cfg_b.pattern
        # ... while still being equal in content (semantic equivalence)
        assert cfg_a.pattern == cfg_b.pattern

    def test_explicit_stationary_matches_default(self) -> None:
        # Explicit ``StationaryPatternConfig()`` and implicit default produce
        # equal ``pattern`` fields and equal RunConfigs (locked at the model
        # level — the canonical-hash equivalence is tested separately in
        # ``TestHash.test_hash_matches_for_implicit_and_explicit_stationary``).
        implicit = _valid_run()
        explicit = _valid_run().model_copy(update={"pattern": StationaryPatternConfig()})
        assert implicit.pattern == explicit.pattern
        assert implicit == explicit


class TestDeterministicLeadTimeConfig:
    """The L>=1 rule moved from engine-init to config-load:
    ``lead_time`` is now a ``PositiveInt`` (was ``NonNegativeInt``), so
    ``lead_time=0`` fails at config validation with a ``ValidationError``.
    """

    def test_positive_lead_time_accepted(self) -> None:
        cfg = DeterministicLeadTimeConfig(lead_time=3)
        assert cfg.kind == "deterministic"
        assert cfg.lead_time == 3

    def test_zero_lead_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DeterministicLeadTimeConfig(lead_time=0)

    def test_negative_lead_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DeterministicLeadTimeConfig(lead_time=-1)


class TestNormalLeadTimeConfig:
    """first stochastic lead-time arm. ``mean: PositiveFloat`` (an
    average lead time of 0 is invalid); ``std: NonNegativeFloat`` (0 is a valid
    near-deterministic degenerate). Rounding/clip happens at ``sample()``, not
    config-load.
    """

    def test_basic_construction(self) -> None:
        cfg = NormalLeadTimeConfig(mean=3.0, std=1.0)
        assert cfg.kind == "normal"
        assert cfg.mean == 3.0
        assert cfg.std == 1.0

    def test_kind_field_defaults_from_yaml(self) -> None:
        cfg = NormalLeadTimeConfig.model_validate({"mean": 3.0, "std": 1.0})
        assert cfg.kind == "normal"

    def test_zero_mean_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalLeadTimeConfig(mean=0.0, std=1.0)

    def test_negative_mean_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalLeadTimeConfig(mean=-1.0, std=1.0)

    def test_zero_std_accepted(self) -> None:
        cfg = NormalLeadTimeConfig(mean=3.0, std=0.0)
        assert cfg.std == 0.0

    def test_negative_std_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalLeadTimeConfig(mean=3.0, std=-1.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalLeadTimeConfig.model_validate(
                {"kind": "normal", "mean": 3.0, "std": 1.0, "unknown": 1}
            )

    def test_frozen_immutable(self) -> None:
        cfg = NormalLeadTimeConfig(mean=3.0, std=1.0)
        with pytest.raises(ValidationError):
            cfg.mean = 5.0


class TestGammaLeadTimeConfig:
    """second stochastic lead-time arm. ``shape, scale`` both
    ``PositiveFloat`` (numpy-native naming, matching ``GammaDemandConfig``).
    Rounding/clip happens at ``sample()`` via the shared helper, not at
    config-load.
    """

    def test_basic_construction(self) -> None:
        cfg = GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)
        assert cfg.kind == "gamma"
        assert cfg.shape == 9.0
        assert cfg.scale == 1.0 / 3.0

    def test_kind_field_defaults_from_yaml(self) -> None:
        cfg = GammaLeadTimeConfig.model_validate({"shape": 9.0, "scale": 0.5})
        assert cfg.kind == "gamma"

    def test_zero_shape_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaLeadTimeConfig(shape=0.0, scale=1.0)

    def test_negative_shape_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaLeadTimeConfig(shape=-1.0, scale=1.0)

    def test_zero_scale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaLeadTimeConfig(shape=9.0, scale=0.0)

    def test_negative_scale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaLeadTimeConfig(shape=9.0, scale=-1.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GammaLeadTimeConfig.model_validate(
                {"kind": "gamma", "shape": 9.0, "scale": 0.5, "unknown": 1}
            )

    def test_frozen_immutable(self) -> None:
        cfg = GammaLeadTimeConfig(shape=9.0, scale=0.5)
        with pytest.raises(ValidationError):
            cfg.shape = 4.0


class TestLognormalLeadTimeConfig:
    """third/final stochastic lead-time arm. ``mu: FiniteFloat``
    (any finite real, incl. negative; NaN/±inf rejected — the first FiniteFloat
    field on the lead-time axis, applying the lock). ``sigma: PositiveFloat``
    (σ=0 degenerate). ``mu/sigma`` are the underlying Normal's params (foot-gun).
    """

    def test_basic_construction(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=1.045931, sigma=0.324593)
        assert cfg.kind == "lognormal"
        assert cfg.mu == 1.045931
        assert cfg.sigma == 0.324593

    def test_kind_field_defaults_from_yaml(self) -> None:
        cfg = LognormalLeadTimeConfig.model_validate({"mu": 1.05, "sigma": 0.32})
        assert cfg.kind == "lognormal"

    def test_negative_mu_accepted(self) -> None:
        # mu < 0 is valid: gives a sub-1 mean lead time (mostly clips to 1).
        cfg = LognormalLeadTimeConfig(mu=-2.0, sigma=0.3)
        assert cfg.mu == -2.0

    def test_nan_mu_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalLeadTimeConfig(mu=float("nan"), sigma=0.3)

    def test_inf_mu_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalLeadTimeConfig(mu=float("inf"), sigma=0.3)

    def test_negative_inf_mu_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalLeadTimeConfig(mu=float("-inf"), sigma=0.3)

    def test_zero_sigma_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalLeadTimeConfig(mu=1.05, sigma=0.0)

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalLeadTimeConfig(mu=1.05, sigma=-1.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LognormalLeadTimeConfig.model_validate(
                {"kind": "lognormal", "mu": 1.05, "sigma": 0.32, "unknown": 1}
            )

    def test_frozen_immutable(self) -> None:
        cfg = LognormalLeadTimeConfig(mu=1.05, sigma=0.32)
        with pytest.raises(ValidationError):
            cfg.mu = 2.0


class TestLeadTimeDiscriminator:
    """LeadTimeConfig widened from a single alias to a 2-arm
    discriminated union (Deterministic + Normal); then to 3 arms
    (+ Gamma); then to 4 arms (+ Lognormal). Pins the routing on
    the lead-time axis.
    """

    def test_deterministic_kind_resolves_to_DeterministicLeadTimeConfig(self) -> None:
        cfg = _valid_run()
        assert isinstance(cfg.lead_time, DeterministicLeadTimeConfig)
        assert cfg.lead_time.kind == "deterministic"

    def test_normal_kind_resolves_to_NormalLeadTimeConfig(self) -> None:
        cfg = _valid_run().model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        assert isinstance(cfg.lead_time, NormalLeadTimeConfig)
        assert cfg.lead_time.kind == "normal"

    def test_yaml_dispatches_to_normal_leadtime(self) -> None:
        cfg = _valid_run().model_copy(update={"lead_time": NormalLeadTimeConfig(mean=4.0, std=2.0)})
        loaded = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(loaded.lead_time, NormalLeadTimeConfig)
        assert loaded.lead_time.mean == 4.0
        assert loaded.lead_time.std == 2.0

    def test_gamma_kind_resolves_to_GammaLeadTimeConfig(self) -> None:
        cfg = _valid_run().model_copy(
            update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)}
        )
        assert isinstance(cfg.lead_time, GammaLeadTimeConfig)
        assert cfg.lead_time.kind == "gamma"

    def test_yaml_dispatches_to_gamma_leadtime(self) -> None:
        cfg = _valid_run().model_copy(
            update={"lead_time": GammaLeadTimeConfig(shape=4.0, scale=0.75)}
        )
        loaded = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(loaded.lead_time, GammaLeadTimeConfig)
        assert loaded.lead_time.shape == 4.0
        assert loaded.lead_time.scale == 0.75

    def test_3way_widening_preserves_deterministic_and_normal_routing(self) -> None:
        # Backward-compat after the 2→3 widening: both prior arms still route.
        det = _valid_run().model_copy(
            update={"lead_time": DeterministicLeadTimeConfig(lead_time=5)}
        )
        nrm = _valid_run().model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        det_loaded = RunConfig.from_yaml(det.to_yaml())
        nrm_loaded = RunConfig.from_yaml(nrm.to_yaml())
        assert isinstance(det_loaded.lead_time, DeterministicLeadTimeConfig)
        assert det_loaded.lead_time.lead_time == 5
        assert isinstance(nrm_loaded.lead_time, NormalLeadTimeConfig)
        assert nrm_loaded.lead_time.mean == 3.0

    def test_lognormal_kind_resolves_to_LognormalLeadTimeConfig(self) -> None:
        cfg = _valid_run().model_copy(
            update={"lead_time": LognormalLeadTimeConfig(mu=1.045931, sigma=0.324593)}
        )
        assert isinstance(cfg.lead_time, LognormalLeadTimeConfig)
        assert cfg.lead_time.kind == "lognormal"

    def test_yaml_dispatches_to_lognormal_leadtime(self) -> None:
        cfg = _valid_run().model_copy(
            update={"lead_time": LognormalLeadTimeConfig(mu=1.2, sigma=0.4)}
        )
        loaded = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(loaded.lead_time, LognormalLeadTimeConfig)
        assert loaded.lead_time.mu == 1.2
        assert loaded.lead_time.sigma == 0.4

    def test_4way_widening_preserves_deterministic_normal_gamma_routing(self) -> None:
        # Backward-compat after the 3→4 widening: all three prior arms route.
        det = _valid_run().model_copy(
            update={"lead_time": DeterministicLeadTimeConfig(lead_time=5)}
        )
        nrm = _valid_run().model_copy(update={"lead_time": NormalLeadTimeConfig(mean=3.0, std=1.0)})
        gam = _valid_run().model_copy(
            update={"lead_time": GammaLeadTimeConfig(shape=9.0, scale=1.0 / 3.0)}
        )
        assert isinstance(RunConfig.from_yaml(det.to_yaml()).lead_time, DeterministicLeadTimeConfig)
        assert isinstance(RunConfig.from_yaml(nrm.to_yaml()).lead_time, NormalLeadTimeConfig)
        assert isinstance(RunConfig.from_yaml(gam.to_yaml()).lead_time, GammaLeadTimeConfig)


class TestNoDisruptionConfig:
    """Identity arm of ``DisruptionConfig`` — the chassis default."""

    def test_kind_defaults_to_none(self) -> None:
        cfg = NoDisruptionConfig()
        assert cfg.kind == "none"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NoDisruptionConfig.model_validate({"kind": "none", "unknown": 1})

    def test_frozen_immutable(self) -> None:
        cfg = NoDisruptionConfig()
        with pytest.raises(ValidationError):
            cfg.kind = "none"


class TestDisruptionWindow:
    """The ``(start, duration, multiplier)`` triple and its field bounds."""

    def test_basic_construction(self) -> None:
        w = DisruptionWindow(start=30, duration=20, multiplier=2.0)
        assert (w.start, w.duration, w.multiplier) == (30, 20, 2.0)

    def test_start_zero_allowed(self) -> None:
        # A disruption may begin at period 0 (NonNegativeInt).
        assert DisruptionWindow(start=0, duration=5, multiplier=2.0).start == 0

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DisruptionWindow(start=-1, duration=5, multiplier=2.0)

    def test_zero_duration_rejected(self) -> None:
        # PositiveInt — a zero-length window is meaningless.
        with pytest.raises(ValidationError):
            DisruptionWindow(start=0, duration=0, multiplier=2.0)

    def test_zero_multiplier_rejected(self) -> None:
        # PositiveFloat — a 0 multiplier would collapse the lead time.
        with pytest.raises(ValidationError):
            DisruptionWindow(start=0, duration=5, multiplier=0.0)

    def test_negative_multiplier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DisruptionWindow(start=0, duration=5, multiplier=-1.0)

    def test_fractional_multiplier_allowed(self) -> None:
        # 0 < multiplier < 1 expedites (the result clips to >= 1 at runtime); a
        # multiplier > 1 delays. Both are valid PositiveFloat values.
        assert DisruptionWindow(start=0, duration=5, multiplier=0.5).multiplier == 0.5
        assert DisruptionWindow(start=0, duration=5, multiplier=2.5).multiplier == 2.5

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DisruptionWindow.model_validate(
                {"start": 0, "duration": 5, "multiplier": 2.0, "unknown": 1}
            )

    def test_frozen_immutable(self) -> None:
        w = DisruptionWindow(start=0, duration=5, multiplier=2.0)
        with pytest.raises(ValidationError):
            w.start = 1


class TestScheduledDisruptionConfig:
    """Deterministic schedule arm of ``DisruptionConfig``."""

    def test_basic_construction(self) -> None:
        cfg = ScheduledDisruptionConfig(
            windows=[DisruptionWindow(start=30, duration=20, multiplier=2.0)]
        )
        assert cfg.kind == "scheduled"
        assert len(cfg.windows) == 1

    def test_multiple_windows_allowed(self) -> None:
        cfg = ScheduledDisruptionConfig(
            windows=[
                DisruptionWindow(start=10, duration=10, multiplier=2.0),
                DisruptionWindow(start=12, duration=6, multiplier=1.5),
            ]
        )
        assert len(cfg.windows) == 2

    def test_empty_windows_rejected(self) -> None:
        # min_length=1: an empty schedule is redundant with NoDisruption and is
        # rejected so the union has one canonical "no disruption" spelling.
        with pytest.raises(ValidationError):
            ScheduledDisruptionConfig(windows=[])

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScheduledDisruptionConfig.model_validate(
                {
                    "kind": "scheduled",
                    "windows": [{"start": 0, "duration": 5, "multiplier": 2.0}],
                    "unknown": 1,
                }
            )

    def test_frozen_immutable(self) -> None:
        cfg = ScheduledDisruptionConfig(
            windows=[DisruptionWindow(start=0, duration=5, multiplier=2.0)]
        )
        with pytest.raises(ValidationError):
            cfg.kind = "scheduled"


class TestDisruptionDiscriminator:
    """Disruption discriminator dispatch on the ``kind`` field.

    ``DisruptionConfig`` is a discriminated union of
    ``NoDisruptionConfig`` (default) and ``ScheduledDisruptionConfig``. These
    tests pin the dispatch routing and the omitted-block default — the
    backward-compat thesis (mirrors the pattern axis).
    """

    def test_default_disruption_resolves_to_NoDisruptionConfig(self) -> None:
        cfg = _valid_run()
        assert isinstance(cfg.disruption, NoDisruptionConfig)
        assert cfg.disruption.kind == "none"

    def test_yaml_without_disruption_defaults_to_none(self) -> None:
        # The backward-compat test for a YAML that omits the
        # ``disruption:`` block must still produce a Pydantic-valid RunConfig
        # whose ``disruption`` is a ``NoDisruptionConfig``. Every existing
        # scenario YAML relies on this property.
        yaml_str = """
master_seed: 42
simulation:
  horizon: 10
  initial_on_hand: 50
demand:
  kind: normal
  mean: 5.0
  std: 1.0
lead_time:
  kind: deterministic
  lead_time: 3
policy:
  kind: sQ
  reorder_point: 10.0
  order_quantity: 20
costs:
  unit_cost: 1.0
  holding_per_unit_per_period: 0.1
  ordering_fixed: 5.0
"""
        cfg = RunConfig.from_yaml(yaml_str)
        assert isinstance(cfg.disruption, NoDisruptionConfig)
        assert cfg.disruption.kind == "none"

    def test_scheduled_kind_resolves_to_ScheduledDisruptionConfig(self) -> None:
        cfg = _valid_run().model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=20, multiplier=2.0)]
                )
            }
        )
        assert isinstance(cfg.disruption, ScheduledDisruptionConfig)
        assert cfg.disruption.kind == "scheduled"

    def test_yaml_dispatches_to_scheduled(self) -> None:
        cfg = _valid_run().model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[DisruptionWindow(start=30, duration=20, multiplier=2.0)]
                )
            }
        )
        loaded = RunConfig.from_yaml(cfg.to_yaml())
        assert isinstance(loaded.disruption, ScheduledDisruptionConfig)
        assert loaded.disruption.kind == "scheduled"
        assert loaded.disruption.windows[0].start == 30
        assert loaded.disruption.windows[0].duration == 20
        assert loaded.disruption.windows[0].multiplier == 2.0

    def test_roundtrip_preserves_multiple_windows(self) -> None:
        cfg = _valid_run().model_copy(
            update={
                "disruption": ScheduledDisruptionConfig(
                    windows=[
                        DisruptionWindow(start=10, duration=10, multiplier=2.0),
                        DisruptionWindow(start=50, duration=5, multiplier=1.5),
                    ]
                )
            }
        )
        loaded = RunConfig.from_yaml(cfg.to_yaml())
        assert loaded == cfg
        assert isinstance(loaded.disruption, ScheduledDisruptionConfig)
        assert len(loaded.disruption.windows) == 2
