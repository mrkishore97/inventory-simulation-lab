"""Unit tests for CLI argument parsing and dispatch.

End-to-end behavior (engine roundtrip, CSV correctness, determinism) lives in
``tests/integration/test_cli.py``. This file owns argparse's own boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inventory_twin.cli import main


class TestCLIArgparse:
    def test_no_subcommand_exits_2(self) -> None:
        # argparse exits with code 2 on a missing required subcommand.
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_run_without_config_exits_2(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["run"])
        assert exc.value.code == 2

    def test_run_with_missing_config_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            main(["run", "--config", str(missing)])
