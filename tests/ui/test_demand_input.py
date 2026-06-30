"""Unit tests for the Policy Advisor demand reader (``ui.components.demand_input``).

``ui/`` is coverage-omitted, but this pure reader carries real validation logic, so it is
unit-tested directly (the "page = glue, logic = importable component" pattern).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ui.components.demand_input import read_demand_column

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_reads_bundled_archetype() -> None:
    arr = read_demand_column(_REPO_ROOT / "data/m5/archetype_smooth.parquet")
    assert arr.ndim == 1
    assert arr.size > 0
    assert np.isfinite(arr).all()


def test_reads_tmp_parquet(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"demand": [1.0, 0.0, 3.0]}).to_parquet(p)
    np.testing.assert_array_equal(read_demand_column(p), np.array([1.0, 0.0, 3.0]))


def test_ignores_other_columns(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"price": [9.0, 9.0], "demand": [2.0, 4.0]}).to_parquet(p)
    np.testing.assert_array_equal(read_demand_column(p), np.array([2.0, 4.0]))


def test_missing_demand_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"sales": [1.0, 2.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="demand"):
        read_demand_column(p)


def test_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"demand": pd.Series([], dtype=float)}).to_parquet(p)
    with pytest.raises(ValueError, match="empty"):
        read_demand_column(p)


def test_non_finite_raises(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"demand": [1.0, np.nan, 3.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="non-finite"):
        read_demand_column(p)


def test_negative_raises(tmp_path: Path) -> None:
    p = tmp_path / "d.parquet"
    pd.DataFrame({"demand": [1.0, -2.0, 3.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="negative"):
        read_demand_column(p)
