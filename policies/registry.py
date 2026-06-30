"""Policy factory: dispatch from ``PolicyConfig`` to a concrete ``Policy``.

The factory is the single construction site for policies in the CLI and
elsewhere. Keeping policy instantiation behind one function means the call
site is stable as M2 bullets add policy variants — each new policy lands as
one additional ``isinstance`` arm.

To add a new policy in a future bullet:
    1. Add an ``isinstance`` arm here mapping the new ``*Config`` to its
       concrete ``Policy`` class.
    2. Add a dispatch + passthrough test in ``tests/unit/test_registry.py``.
"""

from __future__ import annotations

from core.config import (
    BaseStockPolicyConfig,
    PeriodicReviewPolicyConfig,
    PolicyConfig,
    SQPolicyConfig,
    SSPolicyConfig,
)
from policies.base import Policy
from policies.base_stock import BaseStockPolicy
from policies.periodic_review import PeriodicReviewPolicy
from policies.sQ import SQPolicy
from policies.sS import SSPolicy


def make_policy(config: PolicyConfig) -> Policy:
    """Return the ``Policy`` concrete class for ``config``.

    Dispatches on the runtime type of ``config`` (Pydantic guarantees the
    type matches its ``kind`` discriminator). Raises ``ValueError`` if no
    arm is registered for the given config type — defensive guard against
    runtime misuse (e.g. a callsite that bypassed Pydantic discriminator
    validation).
    """
    if isinstance(config, SQPolicyConfig):
        return SQPolicy(config)
    if isinstance(config, SSPolicyConfig):
        return SSPolicy(config)
    if isinstance(config, BaseStockPolicyConfig):
        return BaseStockPolicy(config)
    if isinstance(config, PeriodicReviewPolicyConfig):
        return PeriodicReviewPolicy(config)
    raise ValueError(
        f"Unknown policy config type: {type(config).__name__}. "
        f"Register it in policies.registry.make_policy."
    )
