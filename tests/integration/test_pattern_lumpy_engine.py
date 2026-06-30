"""Integration: the Lumpy pattern composes through the engine; it equals Intermittent × m.

Lumpy(p, m) and Intermittent(p) both draw one uniform from the seeded ``"pattern"`` stream per
period, so for a fixed ``master_seed`` they fire on exactly the same periods — Lumpy's kept demand
is just Intermittent's scaled by the burst multiplier. Verified end-to-end via
``experiments.single_run.run``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.config import IntermittentPatternConfig, LumpyPatternConfig, RunConfig
from experiments.single_run import run

_EXAMPLE = Path("data/scenarios/example.yaml")  # Stationary-pattern Normal(10, 2) baseline


def _run_with(pattern: Any) -> pd.DataFrame:
    base = RunConfig.from_yaml(_EXAMPLE.read_text())
    return run(base.model_copy(update={"pattern": pattern}))


def test_lumpy_demand_equals_intermittent_times_multiplier() -> None:
    p, m = 0.3, 5.0
    intermittent = _run_with(IntermittentPatternConfig(occurrence_probability=p))
    lumpy = _run_with(LumpyPatternConfig(occurrence_probability=p, burst_multiplier=m))
    # Same "pattern" stream + same p → same fire positions; kept demand scaled by m.
    for i_demand, l_demand in zip(intermittent["demand"], lumpy["demand"], strict=True):
        assert l_demand == i_demand * m
    # Bursts actually fire (some zeros) but not every period.
    n_zero = int((lumpy["demand"] == 0.0).sum())
    assert 0 < n_zero < len(lumpy)


def test_lumpy_p1_is_the_stationary_stream_amplified() -> None:
    # p=1 → every period a burst → demand is exactly the Stationary stream * m (the pattern RNG is
    # consumed each period but draws from a stream independent of demand).
    m = 5.0
    stationary = run(RunConfig.from_yaml(_EXAMPLE.read_text()))  # Stationary default
    lumpy_full = _run_with(LumpyPatternConfig(occurrence_probability=1.0, burst_multiplier=m))
    for s_demand, l_demand in zip(stationary["demand"], lumpy_full["demand"], strict=True):
        assert l_demand == s_demand * m
