"""Unit: scenario file I/O helpers for the Scenario Library."""

from __future__ import annotations

from pathlib import Path

import pytest

from ui.components.scenario_io import (
    list_scenarios,
    save_scenario,
    scenario_exists,
    validate_yaml,
)

_EXAMPLE = Path("data/scenarios/example.yaml")


def test_list_scenarios_sorted_yaml_only(tmp_path: Path) -> None:
    (tmp_path / "b.yaml").write_text("")
    (tmp_path / "a.yaml").write_text("")
    (tmp_path / "notes.txt").write_text("ignore me")
    assert list_scenarios(tmp_path) == [tmp_path / "a.yaml", tmp_path / "b.yaml"]


def test_list_scenarios_missing_dir(tmp_path: Path) -> None:
    assert list_scenarios(tmp_path / "nope") == []


def test_validate_yaml_happy() -> None:
    config, error = validate_yaml(_EXAMPLE.read_text())
    assert error is None
    assert config is not None
    assert config.config_hash()[:12] == "84d364adb32c"  # the curated baseline


def test_validate_yaml_malformed_yaml() -> None:
    config, error = validate_yaml("a: b: c")  # not valid YAML
    assert config is None
    assert error  # a non-empty message


def test_validate_yaml_invalid_config() -> None:
    config, error = validate_yaml("master_seed: 42\n")  # valid YAML, missing required sections
    assert config is None
    assert error


def test_save_scenario_roundtrips(tmp_path: Path) -> None:
    config, _ = validate_yaml(_EXAMPLE.read_text())
    assert config is not None
    path = save_scenario(tmp_path, "mine", config)
    assert path == tmp_path / "mine.yaml"
    reloaded, error = validate_yaml(path.read_text())
    assert error is None
    assert reloaded == config  # canonical YAML reloads to an equal config


def test_save_scenario_adds_yaml_suffix(tmp_path: Path) -> None:
    config, _ = validate_yaml(_EXAMPLE.read_text())
    assert config is not None
    assert save_scenario(tmp_path, "no_suffix", config).name == "no_suffix.yaml"


def test_save_scenario_sanitizes_traversal(tmp_path: Path) -> None:
    config, _ = validate_yaml(_EXAMPLE.read_text())
    assert config is not None
    path = save_scenario(tmp_path, "../evil", config)
    assert path == tmp_path / "evil.yaml"  # traversal stripped, stays inside the dir


def test_save_scenario_empty_name_raises(tmp_path: Path) -> None:
    config, _ = validate_yaml(_EXAMPLE.read_text())
    assert config is not None
    with pytest.raises(ValueError, match="empty"):
        save_scenario(tmp_path, "   ", config)


def test_scenario_exists(tmp_path: Path) -> None:
    config, _ = validate_yaml(_EXAMPLE.read_text())
    assert config is not None
    save_scenario(tmp_path, "here", config)
    assert scenario_exists(tmp_path, "here") is True
    assert scenario_exists(tmp_path, "here.yaml") is True
    assert scenario_exists(tmp_path, "absent") is False
