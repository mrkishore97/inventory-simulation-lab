"""Tests for core.ledger — pre-allocated, append-free run buffer."""

from __future__ import annotations

import numpy as np
import pytest

from core.ledger import Ledger


class TestLedger:
    def test_horizon_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Ledger(0)
        with pytest.raises(ValueError):
            Ledger(-5)
        # Boundary: horizon=1 must construct cleanly.
        Ledger(1)

    def test_record_writes_field(self) -> None:
        led = Ledger(10)
        led.record(3, on_hand=42.0)
        df = led.to_dataframe()
        assert df.loc[3, "on_hand"] == 42.0

    def test_unwritten_periods_default_to_zero(self) -> None:
        led = Ledger(5)
        led.record(2, demand=7.0)
        df = led.to_dataframe()
        # Same-period sibling columns stay zero.
        assert df.loc[2, "sales"] == 0.0
        # Untouched periods are zero across the board.
        for t in (0, 1, 3, 4):
            for col in Ledger._SCHEMA:
                assert df.loc[t, col] == 0.0

    def test_record_out_of_range_raises_indexerror(self) -> None:
        led = Ledger(10)
        with pytest.raises(IndexError):
            led.record(10, demand=1.0)
        with pytest.raises(IndexError):
            led.record(-1, demand=1.0)
        with pytest.raises(IndexError):
            led.record(100, demand=1.0)

    def test_record_unknown_field_raises_keyerror(self) -> None:
        led = Ledger(10)
        with pytest.raises(KeyError, match="sale"):
            led.record(0, sale=5.0)  # typo: should be "sales"

    def test_record_multiple_fields_in_one_call(self) -> None:
        led = Ledger(10)
        led.record(2, demand=8.0, sales=5.0, backorders=3.0)
        df = led.to_dataframe()
        assert df.loc[2, "demand"] == 8.0
        assert df.loc[2, "sales"] == 5.0
        assert df.loc[2, "backorders"] == 3.0

    def test_record_multiple_calls_merge_across_fields(self) -> None:
        led = Ledger(10)
        led.record(2, demand=8.0)
        led.record(2, sales=5.0)
        df = led.to_dataframe()
        assert df.loc[2, "demand"] == 8.0
        assert df.loc[2, "sales"] == 5.0

    def test_record_overwrites_same_field_same_period(self) -> None:
        led = Ledger(10)
        led.record(2, demand=8.0)
        led.record(2, demand=12.0)
        df = led.to_dataframe()
        assert df.loc[2, "demand"] == 12.0

    def test_to_dataframe_shape(self) -> None:
        led = Ledger(7)
        df = led.to_dataframe()
        # 7 rows; 'period' + every column in _SCHEMA.
        assert df.shape == (7, 1 + len(Ledger._SCHEMA))

    def test_to_dataframe_column_order_matches_schema(self) -> None:
        led = Ledger(3)
        df = led.to_dataframe()
        assert df.columns[0] == "period"
        assert tuple(df.columns[1:]) == Ledger._SCHEMA

    def test_to_dataframe_period_column_is_range(self) -> None:
        led = Ledger(5)
        df = led.to_dataframe()
        assert (df["period"].to_numpy() == np.arange(5)).all()

    def test_to_dataframe_returns_copy(self) -> None:
        led = Ledger(5)
        led.record(0, demand=1.0)
        df1 = led.to_dataframe()
        df1.loc[0, "demand"] = 999.0
        df2 = led.to_dataframe()
        assert df2.loc[0, "demand"] == 1.0

    def test_schema_columns_are_float64(self) -> None:
        led = Ledger(5)
        df = led.to_dataframe()
        for name in Ledger._SCHEMA:
            assert df[name].dtype == np.float64

    def test_row_returns_all_fields(self) -> None:
        led = Ledger(5)
        led.record(2, on_hand=42.0, demand=7.0, holding_cost=3.5)
        row = led.row(2)
        assert set(row.keys()) == set(Ledger._SCHEMA)
        assert row["on_hand"] == 42.0
        assert row["demand"] == 7.0
        assert row["holding_cost"] == 3.5
        # Unwritten fields default to zero.
        assert row["sales"] == 0.0

    def test_row_negative_t_raises_indexerror(self) -> None:
        led = Ledger(10)
        with pytest.raises(IndexError, match="out of range"):
            led.row(-1)

    def test_row_out_of_range_raises_indexerror(self) -> None:
        led = Ledger(10)
        with pytest.raises(IndexError, match="out of range"):
            led.row(10)
        with pytest.raises(IndexError, match="out of range"):
            led.row(100)
