"""Unit tests for the sensitivity parameter-spec helpers.

Pure functions over a ``RunConfig`` — no Streamlit — so the perturbable catalogue and the ±% bounds
are locked apart from the page. The catalogue test is the architect-requested guard: it pins which
config paths are offered (and, just as importantly, which structural / inert fields are not). Built
from the canonical ``example.yaml`` (the reproducibility anchor) so the expected paths are concrete.
"""

from __future__ import annotations

from pathlib import Path

from core.config import RunConfig
from ui.components.sensitivity_spec import low_high, numeric_paths

_EXAMPLE = Path(__file__).resolve().parents[2] / "data" / "scenarios" / "example.yaml"


def _config() -> RunConfig:
    return RunConfig.from_yaml(_EXAMPLE.read_text())


def test_catalogue_includes_business_inputs() -> None:
    paths = set(numeric_paths(_config()))
    assert {"demand.mean", "demand.std", "costs.ordering_fixed", "policy.reorder_point"} <= paths


def test_catalogue_excludes_structural_and_enum_fields() -> None:
    paths = set(numeric_paths(_config()))
    # str discriminators, the top-level seed, and the whole structural simulation section are out.
    assert not any(p.endswith(".kind") for p in paths)
    assert "master_seed" not in paths
    assert not any(p.startswith("simulation.") for p in paths)  # horizon / initial_* / backorder
    assert not any(p.startswith(("pattern.", "disruption.")) for p in paths)


def test_catalogue_excludes_zero_baselines() -> None:
    # The Phase-1 inert costs default to 0, so a ±% of them carries no signal — never offered.
    catalogue = numeric_paths(_config())
    assert "costs.expediting_per_unit" not in catalogue
    assert "costs.lost_sale_per_unit" not in catalogue
    assert all(v != 0 for v in catalogue.values())


def test_catalogue_values_are_real_numbers_not_bools() -> None:
    for value in numeric_paths(_config()).values():
        assert isinstance(value, (int, float))
        assert not isinstance(value, bool)


def test_low_high_floats_pass_through_unrounded() -> None:
    assert low_high(20.0, 0.25) == (15.0, 25.0)


def test_low_high_ints_round_to_whole_units() -> None:
    assert low_high(30, 0.25) == (round(22.5), round(37.5))  # PositiveInt order_quantity, say


def test_low_high_clamps_int_low_to_one() -> None:
    # baseline 1, -50% -> round(0.5) == 0 -> clamped to 1, so the bound stays a valid PositiveInt.
    assert low_high(1, 0.5) == (1, 2)
