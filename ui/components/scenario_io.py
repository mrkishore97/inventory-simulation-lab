"""Scenario file I/O for the Scenario Library page.

Importable, Streamlit-free helpers over the ``RunConfig`` YAML surface
(``from_yaml`` / ``to_yaml`` / ``config_hash``), so the file operations are unit-testable
with ``tmp_path`` apart from the page. ``ui/pages/5_Scenario_Library.py`` is thin glue over
these. Saving always writes the *canonical* ``to_yaml`` of a validated config, so a saved
scenario is guaranteed reloadable and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from core.config import RunConfig


def list_scenarios(directory: Path) -> list[Path]:
    """Sorted ``*.yaml`` files in ``directory`` (empty list if it does not exist)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def validate_yaml(text: str) -> tuple[RunConfig | None, str | None]:
    """Parse + validate scenario YAML → ``(config, None)`` or ``(None, message)``.

    Catches both malformed YAML (:class:`yaml.YAMLError`) and an invalid config
    (:class:`pydantic.ValidationError`), so the page has one fail-loud error path.
    """
    try:
        return RunConfig.from_yaml(text), None
    except (ValidationError, yaml.YAMLError) as exc:
        return None, str(exc)


def _safe_filename(name: str) -> str:
    """A bare ``<stem>.yaml`` filename (no directories) — guards against path traversal."""
    stem = Path(name.strip()).name  # drop any directory components / traversal
    if not stem:
        raise ValueError("scenario name must not be empty")
    if not stem.endswith(".yaml"):
        stem += ".yaml"
    return stem


def scenario_exists(directory: Path, name: str) -> bool:
    """Whether ``name`` (sanitized) already exists in ``directory``."""
    try:
        return (directory / _safe_filename(name)).exists()
    except ValueError:
        return False


def save_scenario(directory: Path, name: str, config: RunConfig) -> Path:
    """Write ``config``'s canonical YAML to ``directory/<name>.yaml``; return the path.

    Saves :meth:`RunConfig.to_yaml` (sorted, canonical) of the *validated* config, so the
    file always reloads to an equal config. The name is sanitized to a bare filename (no
    path traversal); an empty name raises :class:`ValueError`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _safe_filename(name)
    path.write_text(config.to_yaml())
    return path
