"""Performance smoke test: 365-period single run under 100 ms.

The 100 ms budget is the M1 deliverable. It exists for
two reasons:

1. **End-user latency.** Streamlit's ``Single Run`` page (M4) re-runs the engine
   on every parameter change; > 100 ms feels laggy.
2. **Phase 2 RL training viability.** RL needs millions of episodes; if a single
   horizon-365 episode costs > 100 ms, training a policy from scratch is
   prohibitive.

This test times the **end-user-perceived pipeline**: engine construction +
inner loop + ``to_dataframe()``. It excludes RunConfig validation (one-shot
setup) and CSV write (CLI concern, not engine concern).

Best-of-5 is asserted, not average — this is a capability test, not an SLA.
A single GC pause should not fail the build; the question is "is the engine
*capable* of running this fast", not "does every run hit 100 ms". A persistent
regression will push the floor up across all five reps.

Skipped under coverage / debugger tracing because ``sys.settrace`` slows
CPython 2-3× and would make the measurement meaningless. Run the perf gate
with plain ``pytest``, run coverage with ``pytest --cov ...`` separately.
"""

from __future__ import annotations

import sys
import time

import pytest

from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SQPolicyConfig,
)
from core.simulation import InventoryEngine
from policies.sQ import SQPolicy


def _perf_config() -> RunConfig:
    return RunConfig(
        simulation=SimulationConfig(horizon=365, initial_on_hand=100),
        demand=NormalDemandConfig(mean=10.0, std=2.0),
        lead_time=DeterministicLeadTimeConfig(lead_time=3),
        policy=SQPolicyConfig(reorder_point=20.0, order_quantity=30),
        costs=CostConfig(
            unit_cost=5.0,
            holding_per_unit_per_period=0.5,
            ordering_fixed=20.0,
            backorder_per_unit_per_period=2.0,
        ),
        master_seed=42,
    )


def _time_one_run(config: RunConfig) -> float:
    start = time.perf_counter()
    engine = InventoryEngine(config)
    assert isinstance(config.policy, SQPolicyConfig)
    policy = SQPolicy(config.policy)
    state = engine.initial_observation()
    done = False
    while not done:
        state, done = engine.step(policy.decide(state))
    engine.ledger.to_dataframe()
    return time.perf_counter() - start


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason="sys.settrace (coverage / pdb) invalidates perf measurement",
)
class TestPerfSmoke:
    def test_single_run_365_under_100ms(self) -> None:
        config = _perf_config()
        # Warm-up: first call pays for lazy numpy / pandas state caches.
        _time_one_run(config)

        runs_ms = [_time_one_run(config) * 1000 for _ in range(5)]
        best_ms = min(runs_ms)

        assert best_ms < 100.0, (
            f"Single run too slow: best={best_ms:.2f}ms exceeds 100ms budget. "
            f"All runs (ms): {[f'{r:.2f}' for r in runs_ms]}"
        )
