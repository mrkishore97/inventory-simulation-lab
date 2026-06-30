"""Unit + smoke tests for the M5 demand-archetype loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import EmpiricalDemandConfig, RunConfig
from experiments import single_run
from ui.components.archetypes import load_archetypes

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_loads_four_in_manifest_order() -> None:
    assert list(load_archetypes()) == ["smooth", "intermittent", "seasonal", "promotional"]


def test_history_path_repo_relative_and_exists() -> None:
    archs = load_archetypes()
    assert archs["smooth"].history_path == "data/m5/archetype_smooth.parquet"
    for a in archs.values():
        assert not Path(a.history_path).is_absolute()  # repo-relative -> portable config hash
        assert (_REPO_ROOT / a.history_path).exists()  # the bundled parquet is really there


def test_stats_and_sb_quadrant_divergence() -> None:
    archs = load_archetypes()
    smooth = archs["smooth"]
    assert smooth.item_id == "FOODS_3_090"
    assert smooth.sb_quadrant == "smooth"
    assert smooth.cv2 == pytest.approx(0.370897, abs=1e-4)
    assert smooth.adi == pytest.approx(1.231017, abs=1e-4)
    assert smooth.zero_fraction == pytest.approx(0.187663, abs=1e-4)
    # business archetype name != statistical Syntetos-Boylan class (the teaching point)
    assert archs["intermittent"].sb_quadrant == "intermittent"
    assert archs["seasonal"].sb_quadrant == "erratic"
    assert archs["promotional"].sb_quadrant == "lumpy"


def test_archetype_runs_end_to_end() -> None:
    """The real I/O gate: a bundled archetype path resolves and simulates at engine init."""
    smooth = load_archetypes()["smooth"]
    base = RunConfig.from_yaml((_REPO_ROOT / "data/scenarios/example.yaml").read_text())
    config = base.model_copy(
        update={"demand": EmpiricalDemandConfig(history_path=smooth.history_path)}
    )
    result = single_run.run(config)
    assert len(result) == config.simulation.horizon
