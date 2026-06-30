"""Integration: the named built-in stress-scenario presets.

These presets ship the feasible subset of the named stress scenarios — the ones the
static-config + lead-time-only-disruption model can faithfully express. Each preset is the
example.yaml baseline world with a single stressor, runnable as-is from Page 5 (Scenario Library).
The demand-windowed scenarios (demand_surge / demand_collapse / recovery_test / the demand half of
double_whammy) are deferred to the future demand-disruption overlay and are NOT shipped here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analytics.kpis import compute_kpis
from core.config import RunConfig, SeasonalPatternConfig
from experiments.single_run import run

_SCENARIOS = Path("data/scenarios")
_EXAMPLE = _SCENARIOS / "example.yaml"
_PRESETS = ["lead_time_spike", "supplier_outage", "seasonal_shock"]


def _load(name: str) -> RunConfig:
    return RunConfig.from_yaml((_SCENARIOS / f"{name}.yaml").read_text())


def _baseline_cost() -> float:
    return compute_kpis(run(RunConfig.from_yaml(_EXAMPLE.read_text()))).total_cost


@pytest.mark.parametrize("name", _PRESETS)
def test_preset_validates_and_runs(name: str) -> None:
    cfg = _load(name)
    df = run(cfg)
    assert len(df) == cfg.simulation.horizon
    assert df["demand"].notna().all()


def test_lead_time_spike_is_scheduled_disruption_above_baseline() -> None:
    cfg = _load("lead_time_spike")
    assert cfg.disruption.kind == "scheduled"
    assert compute_kpis(run(cfg)).total_cost > _baseline_cost()


def test_supplier_outage_causes_backorders_above_baseline() -> None:
    cfg = _load("supplier_outage")
    assert cfg.disruption.kind == "scheduled"
    df = run(cfg)
    # In-window orders never arrive, so on-hand drains into backorder.
    assert df["backorders"].max() > 0.0
    assert compute_kpis(df).total_cost > _baseline_cost()


def test_seasonal_shock_is_the_intended_seasonal_pattern() -> None:
    cfg = _load("seasonal_shock")
    # Prove it's the intended SHOCK, not the gentle example_pattern_seasonal demo.
    assert isinstance(cfg.pattern, SeasonalPatternConfig)
    assert cfg.pattern.amplitude == 0.7
    assert cfg.pattern.period == 30


def test_all_shipped_scenarios_validate_and_hash() -> None:
    # Pre-M6 library safety net: every YAML in data/scenarios/ parses via from_yaml and its
    # config hash computes. Validation-only — no behavioral assertions over the old scenarios.
    yamls = sorted(_SCENARIOS.glob("*.yaml"))
    assert yamls, "no scenario YAMLs found"
    for path in yamls:
        cfg = RunConfig.from_yaml(path.read_text())
        assert len(cfg.config_hash()) == 64  # SHA-256 hex digest
