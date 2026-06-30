"""Tests for the Policy abstract base class."""

from __future__ import annotations

import pytest

from core.simulation import EngineState
from policies.base import Policy


class TestPolicyABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Policy()  # type: ignore[abstract]

    def test_subclass_without_decide_cannot_instantiate(self) -> None:
        class IncompletePolicy(Policy):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompletePolicy()  # type: ignore[abstract]

    def test_minimal_concrete_subclass_works(self) -> None:
        class ZeroPolicy(Policy):
            def decide(self, state: EngineState) -> float:
                return 0.0

        policy = ZeroPolicy()
        state = EngineState(
            t=0, on_hand=10.0, on_order=0.0, backorders=0.0, inventory_position=10.0
        )
        assert policy.decide(state) == 0.0
