"""End-to-end tests for the ``inventory_twin.cli`` runner.

These tests exercise the full YAML → engine → Ledger → CSV path. The
``test_determinism_two_runs`` test is the M1 deliverable gate: two runs
of the same config must produce a byte-identical CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.config import RunConfig
from core.ledger import Ledger
from inventory_twin.cli import main


class TestCLIRun:
    def test_writes_csv_with_correct_shape(self, tmp_path: Path, basic_config: RunConfig) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_path = tmp_path / "out.csv"

        rc = main(["run", "--config", str(cfg_path), "--out", str(out_path), "--quiet"])

        assert rc == 0
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert len(df) == basic_config.simulation.horizon
        # Belt-and-suspenders contract pin: column order locked at the CLI seam,
        # in addition to Ledger's own _SCHEMA-driven construction.
        assert tuple(df.columns) == ("period",) + Ledger._SCHEMA

    def test_determinism_two_runs(self, tmp_path: Path, basic_config: RunConfig) -> None:
        # M1 reproducibility gate: same config + same master seed = byte-identical CSV.
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_a = tmp_path / "a.csv"
        out_b = tmp_path / "b.csv"

        main(["run", "--config", str(cfg_path), "--out", str(out_a), "--quiet"])
        main(["run", "--config", str(cfg_path), "--out", str(out_b), "--quiet"])

        # Guard against silent pass on empty-equal-empty.
        assert out_a.stat().st_size > 0
        assert out_a.read_bytes() == out_b.read_bytes()

    def test_default_output_uses_config_hash(
        self,
        tmp_path: Path,
        basic_config: RunConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        monkeypatch.chdir(tmp_path)

        main(["run", "--config", str(cfg_path), "--quiet"])

        expected = tmp_path / f"{basic_config.config_hash()[:12]}.csv"
        assert expected.exists()
        assert expected.stat().st_size > 0

    def test_quiet_suppresses_summary(
        self,
        tmp_path: Path,
        basic_config: RunConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_path = tmp_path / "out.csv"

        main(["run", "--config", str(cfg_path), "--out", str(out_path), "--quiet"])

        captured = capsys.readouterr()
        assert "Total cost" not in captured.out
        assert "Wrote ledger to" in captured.out

    def test_default_summary_includes_key_fields(
        self,
        tmp_path: Path,
        basic_config: RunConfig,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_path = tmp_path / "out.csv"

        main(["run", "--config", str(cfg_path), "--out", str(out_path)])

        captured = capsys.readouterr()
        assert "Run:" in captured.out
        assert "Hash:" in captured.out
        assert "Horizon:" in captured.out
        assert "Total cost:" in captured.out
        assert "Wrote ledger to" in captured.out


class TestCLIMonteCarlo:
    def test_writes_kpi_distribution_csv(self, tmp_path: Path, basic_config: RunConfig) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_path = tmp_path / "mc.csv"

        rc = main(
            [
                "montecarlo",
                "--config",
                str(cfg_path),
                "--reps",
                "16",
                "--jobs",
                "1",
                "--out",
                str(out_path),
                "--quiet",
            ]
        )

        assert rc == 0
        df = pd.read_csv(out_path)
        assert len(df) == 16
        assert len(df.columns) == 14
        assert tuple(df.columns[:2]) == ("replication", "seed")

    def test_determinism_two_runs(self, tmp_path: Path, basic_config: RunConfig) -> None:
        # Reproducibility gate for the MC path: same config -> byte-identical distribution CSV.
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_a = tmp_path / "a.csv"
        out_b = tmp_path / "b.csv"

        for out in (out_a, out_b):
            main(
                [
                    "montecarlo",
                    "--config",
                    str(cfg_path),
                    "--reps",
                    "16",
                    "--jobs",
                    "1",
                    "--out",
                    str(out),
                    "--quiet",
                ]
            )

        assert out_a.stat().st_size > 0
        assert out_a.read_bytes() == out_b.read_bytes()

    def test_default_output_uses_mc_path(
        self, tmp_path: Path, basic_config: RunConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        monkeypatch.chdir(tmp_path)

        main(["montecarlo", "--config", str(cfg_path), "--reps", "8", "--jobs", "1", "--quiet"])

        expected = tmp_path / f"{basic_config.config_hash()[:12]}_mc.csv"
        assert expected.exists()
        assert expected.stat().st_size > 0

    def test_quiet_suppresses_summary(
        self, tmp_path: Path, basic_config: RunConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_path = tmp_path / "mc.csv"

        main(
            [
                "montecarlo",
                "--config",
                str(cfg_path),
                "--reps",
                "8",
                "--jobs",
                "1",
                "--out",
                str(out_path),
                "--quiet",
            ]
        )

        captured = capsys.readouterr()
        assert "Total cost" not in captured.out
        assert "Wrote KPI distribution to" in captured.out

    def test_default_summary_includes_key_fields(
        self, tmp_path: Path, basic_config: RunConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(basic_config.to_yaml())
        out_path = tmp_path / "mc.csv"

        main(
            [
                "montecarlo",
                "--config",
                str(cfg_path),
                "--reps",
                "8",
                "--jobs",
                "1",
                "--out",
                str(out_path),
            ]
        )

        captured = capsys.readouterr()
        assert "Replications:" in captured.out
        assert "Total cost:" in captured.out
        assert "Fill rate:" in captured.out
        assert "Wrote KPI distribution to" in captured.out
