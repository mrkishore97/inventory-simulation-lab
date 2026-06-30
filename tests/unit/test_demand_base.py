"""Tests for the ``Demand`` ABC."""

from __future__ import annotations

import pytest

from demand.base import Demand


class TestDemandABC:
    def test_demand_is_abstract(self) -> None:
        # Cannot instantiate the ABC directly; mirrors the Policy ABC contract.
        with pytest.raises(TypeError):
            Demand()  # type: ignore[abstract]

    def test_subclass_without_draw_is_still_abstract(self) -> None:
        class PartialDemand(Demand):
            pass

        with pytest.raises(TypeError):
            PartialDemand()  # type: ignore[abstract]

    def test_subclass_with_draw_is_concrete(self) -> None:
        class ConstantDemand(Demand):
            def draw(self) -> float:
                return 7.0

        instance = ConstantDemand()
        assert isinstance(instance, Demand)
        assert instance.draw() == 7.0
