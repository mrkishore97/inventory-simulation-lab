"""Tests for the ``Disruption`` ABC and the ``NoDisruption`` identity arm."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import NoDisruptionConfig
from leadtime.disruptions.base import Disruption
from leadtime.disruptions.none import NoDisruption


class TestDisruptionABC:
    def test_disruption_is_abstract(self) -> None:
        # Cannot instantiate the ABC directly; mirrors the LeadTime / Pattern ABCs.
        with pytest.raises(TypeError):
            Disruption()  # type: ignore[abstract]

    def test_subclass_without_apply_is_still_abstract(self) -> None:
        class PartialDisruption(Disruption):
            pass

        with pytest.raises(TypeError):
            PartialDisruption()  # type: ignore[abstract]

    def test_subclass_with_apply_is_concrete(self) -> None:
        class DoubleDisruption(Disruption):
            def apply(self, base_lead_time: int, t: int) -> int:
                return base_lead_time * 2

        instance = DoubleDisruption()
        assert isinstance(instance, Disruption)
        assert instance.apply(3, 0) == 6


class TestNoDisruptionIdentity:
    """The defining identity contract — ``apply(base_lead_time, t) == base_lead_time``.

    NoDisruption is the chassis arm of the disruption channel and the Pydantic default for
    ``RunConfig.disruption``. The contract is unconditional: identity does not
    depend on time, does not consume RNG, and returns its input unchanged.
    Mirrors ``StationaryPattern``.
    """

    def test_apply_returns_base_unchanged(self) -> None:
        disruption = NoDisruption(NoDisruptionConfig(), np.random.default_rng(0))
        for base in (1, 2, 3, 7, 100, 10_000):
            for t in (0, 1, 100, 10_000):
                assert disruption.apply(base, t) == base, (
                    f"identity violated at base={base}, t={t}: got {disruption.apply(base, t)}"
                )

    def test_apply_returns_int_type(self) -> None:
        disruption = NoDisruption(NoDisruptionConfig(), np.random.default_rng(0))
        assert isinstance(disruption.apply(3, 0), int)

    def test_apply_is_time_invariant(self) -> None:
        disruption = NoDisruption(NoDisruptionConfig(), np.random.default_rng(0))
        assert disruption.apply(5, 0) == disruption.apply(5, 1_000_000)

    def test_apply_does_not_consume_rng(self) -> None:
        # NoDisruption stores rng but never draws. Locked via byte-for-byte state
        # comparison at three checkpoints: before construction, after
        # construction, and after 1000 apply() calls. Mirrors the
        # StationaryPattern rng-pin — catches a future refactor that accidentally
        # wires apply() through the rng.
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        disruption = NoDisruption(NoDisruptionConfig(), rng)
        state_after_construct = rng.bit_generator.state
        for i in range(1, 1001):
            disruption.apply(i, i)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_determinism_under_same_seed(self) -> None:
        cfg = NoDisruptionConfig()
        a = NoDisruption(cfg, np.random.default_rng(42))
        b = NoDisruption(cfg, np.random.default_rng(42))
        for base in (1, 2, 5, 100):
            for t in (0, 1, 10, 100):
                assert a.apply(base, t) == b.apply(base, t)
