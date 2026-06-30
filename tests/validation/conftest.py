"""Shared helpers for the analytical-validation suite.

These tests assert the simulator reproduces the closed-form laws in
``analytics.classical``. ``run_to_ledger`` drives a
``RunConfig`` to completion under its policy — the same loop as
``inventory_twin.cli._run`` — and returns the materialized Ledger, so each test
can isolate whichever cost columns its law cares about. (For EOQ: holding +
ordering only — purchase cost is ``Q``-independent over a full horizon, and
stockout cost is held at zero by provisioning the reorder point above
lead-time demand.)
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from core.config import RunConfig
from experiments import single_run


def _run_to_ledger(config: RunConfig) -> pd.DataFrame:
    """Run ``config`` to completion under its policy; return the Ledger frame."""
    return single_run.run(config)


@pytest.fixture
def run_to_ledger() -> Callable[[RunConfig], pd.DataFrame]:
    """Return the engine runner (mirrors the ``make_config`` fixture idiom)."""
    return _run_to_ledger
