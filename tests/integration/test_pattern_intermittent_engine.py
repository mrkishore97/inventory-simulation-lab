"""Integration: the Intermittent pattern composes correctly through the engine.

The pattern overlay draws from the seeded ``"pattern"`` stream, which is independent of the
``"demand"`` stream. So for a fixed ``master_seed`` an intermittent run's base draws match the
Stationary baseline's, and the intermittent ``demand`` column is exactly that sequence with some
periods masked to zero — verified end-to-end through ``experiments.single_run.run``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import IntermittentPatternConfig, RunConfig
from experiments.single_run import run

_EXAMPLE = Path("data/scenarios/example.yaml")  # Stationary-pattern Normal(10, 2) baseline


def _stationary_and_intermittent(p: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = RunConfig.from_yaml(_EXAMPLE.read_text())  # no pattern block → Stationary default
    stationary = run(base)
    intermittent_cfg = base.model_copy(
        update={"pattern": IntermittentPatternConfig(occurrence_probability=p)}
    )
    return stationary, run(intermittent_cfg)


def test_intermittent_masks_the_stationary_demand_stream() -> None:
    stationary, intermittent = _stationary_and_intermittent(0.6)
    # Independent streams → identical base draws; each intermittent demand is 0 or the base.
    for s, i in zip(stationary["demand"], intermittent["demand"], strict=True):
        assert i == 0.0 or i == s
    # The mask actually fires (some zeros) but does not zero everything.
    n_zero = int((intermittent["demand"] == 0.0).sum())
    assert 0 < n_zero < len(intermittent)


def test_probability_one_reproduces_the_stationary_demand_stream() -> None:
    # p=1 keeps every period → demand column identical to the Stationary run, even though the
    # pattern RNG is consumed each period (it draws from a stream independent of demand).
    stationary, intermittent = _stationary_and_intermittent(1.0)
    assert list(intermittent["demand"]) == list(stationary["demand"])
