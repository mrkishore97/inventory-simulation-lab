"""Drive one ``RunConfig`` to a materialized Ledger frame.

``run(config)`` is the single-config convenience over
:func:`experiments.common.rollout`: build the policy and engine from the config,
roll out, and return the Ledger as a DataFrame. The CLI, the validation/integration
test helpers, and the M3 Monte Carlo runner all call this instead of inlining the
rollout.

It returns the materialized ``pd.DataFrame`` (not the ``Ledger`` object): every
consumer — CSV export, :func:`analytics.kpis.compute_kpis`, the test assertions, and
the Monte Carlo runner — works on the frame, so returning it here avoids a
``.to_dataframe()`` at every call site.
"""

from __future__ import annotations

import pandas as pd

from core.config import RunConfig
from core.simulation import InventoryEngine
from experiments.common import rollout
from policies.registry import make_policy


def run(config: RunConfig) -> pd.DataFrame:
    """Run ``config`` under its configured policy; return the Ledger frame."""
    policy = make_policy(config.policy)
    engine = InventoryEngine(config)
    rollout(engine, policy)
    return engine.ledger.to_dataframe()
