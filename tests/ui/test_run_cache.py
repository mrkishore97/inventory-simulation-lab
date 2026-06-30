"""Integration: the cached single-run helper."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SQPolicyConfig,
)
from experiments import single_run
from ui.components import run_cache
from ui.components.run_cache import run_cached


def _config(seed: int = 42) -> RunConfig:
    return RunConfig(
        simulation=SimulationConfig(horizon=30, initial_on_hand=100),
        demand=NormalDemandConfig(mean=10.0, std=2.0),
        lead_time=DeterministicLeadTimeConfig(lead_time=3),
        policy=SQPolicyConfig(reorder_point=20.0, order_quantity=30),
        costs=CostConfig(unit_cost=5.0, holding_per_unit_per_period=0.5, ordering_fixed=20.0),
        master_seed=seed,
    )


def test_run_cached_matches_single_run() -> None:
    cfg = _config()
    pd.testing.assert_frame_equal(run_cached(cfg), single_run.run(cfg))


def test_run_cached_is_deterministic() -> None:
    cfg = _config()
    pd.testing.assert_frame_equal(run_cached(cfg), run_cached(cfg))


def test_run_cached_caches_on_config_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cache hit = the underlying single_run.run is not re-invoked for an identical config.
    run_cache._cached_run.clear()
    calls = {"n": 0}
    real: Callable[[RunConfig], pd.DataFrame] = single_run.run

    def counting(config: RunConfig) -> pd.DataFrame:
        calls["n"] += 1
        return real(config)

    monkeypatch.setattr(single_run, "run", counting)

    cfg_a, cfg_b = _config(1), _config(2)
    run_cached(cfg_a)
    run_cached(cfg_a)  # identical JSON key -> cache hit, underlying not called again
    run_cached(cfg_b)  # different key -> miss
    assert calls["n"] == 2
