"""The one canonical rollout loop — drive an engine to completion under a policy.

the shared primitive that ``single_run``, ``policy_comparison``
(deferred), and the M3 Monte Carlo runner all compose, instead of each inlining
``engine.initial_observation()`` then ``while not done: step(decide(state))`` and
drifting apart. The engine accumulates into its own Ledger; callers read
``engine.ledger`` after the call (materialize with ``engine.ledger.to_dataframe()``).
"""

from __future__ import annotations

from core.simulation import InventoryEngine
from policies.base import Policy


def rollout(engine: InventoryEngine, policy: Policy) -> None:
    """Drive ``engine`` to completion under ``policy`` (mutates ``engine`` in place).

    Runs the locked within-period loop: read the Point-A state, let the policy
    decide an order quantity, step the engine, repeat until the horizon is
    exhausted. Returns nothing — the run's output is ``engine.ledger``.
    """
    state = engine.initial_observation()
    done = False
    while not done:
        state, done = engine.step(policy.decide(state))
