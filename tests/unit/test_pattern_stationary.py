"""Tests for ``demand.patterns.stationary.StationaryPattern``."""

from __future__ import annotations

import numpy as np

from core.config import StationaryPatternConfig
from demand.patterns.stationary import StationaryPattern


class TestStationaryPatternIdentity:
    """The defining identity contract — ``apply(base, t) == base`` for ALL base.

    Stationary is the chassis arm of the pattern engine and the Pydantic default
    for ``RunConfig.pattern``. The contract is unconditional: identity does
    not police input, does not depend on time, does not consume RNG, and
    produces bit-for-bit equal output to its input.
    """

    def test_apply_returns_base_unchanged(self) -> None:
        # Exact float equality — identity is bit-for-bit. Grid covers
        # zero, small / unit / moderate / large positives, and far-apart
        # time values (Stationary is time-invariant, so t should not
        # matter, but the test still varies t to lock the property).
        pattern = StationaryPattern(StationaryPatternConfig(), np.random.default_rng(0))
        for base in (0.0, 1.0, 3.7, 100.0, 1e-9, 1e9):
            for t in (0, 1, 100, 10_000):
                assert pattern.apply(base, t) == base, (
                    f"identity violated at base={base}, t={t}: got {pattern.apply(base, t)}"
                )

    def test_apply_is_time_invariant(self) -> None:
        # Identity ignores ``t`` entirely.
        pattern = StationaryPattern(StationaryPatternConfig(), np.random.default_rng(0))
        assert pattern.apply(5.0, 0) == pattern.apply(5.0, 1_000_000)

    def test_apply_does_not_consume_rng(self) -> None:
        # Stationary stores rng but never draws. Locked via byte-for-byte
        # state comparison at three checkpoints: before construction, after
        # construction, and after 1000 apply() calls. If a future refactor
        # accidentally wires apply() through the rng, this test catches it.
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        pattern = StationaryPattern(StationaryPatternConfig(), rng)
        state_after_construct = rng.bit_generator.state
        for i in range(1000):
            pattern.apply(float(i), i)
        state_after_apply = rng.bit_generator.state
        assert state_before == state_after_construct == state_after_apply

    def test_determinism_under_same_seed(self) -> None:
        # Trivially true (identity has no stochasticity), but locks the
        # symmetric API for the test seam future stochastic patterns reuse.
        cfg = StationaryPatternConfig()
        a = StationaryPattern(cfg, np.random.default_rng(42))
        b = StationaryPattern(cfg, np.random.default_rng(42))
        for base in (0.0, 1.0, 5.0, 100.0):
            for t in (0, 1, 10, 100):
                assert a.apply(base, t) == b.apply(base, t)

    def test_apply_negative_base_passthrough(self) -> None:
        # Identity is **unconditional**: ``apply(-1.0, 0) == -1.0``.
        # Non-negativity is the BASE distribution's responsibility, not
        # Stationary's. Stationary never modifies the input, so it never
        # violates the engine's non-negative contract — but only because
        # ``Demand`` never feeds it a negative value at runtime. Future
        # patterns whose math could dip below zero MUST clip at source.
        pattern = StationaryPattern(StationaryPatternConfig(), np.random.default_rng(0))
        assert pattern.apply(-1.0, 0) == -1.0
        assert pattern.apply(-1e9, 100) == -1e9

    def test_construction_with_rng_does_not_draw(self) -> None:
        # Explicit: building ``StationaryPattern`` does NOT call any RNG
        # method. Pinned via ``bit_generator.state`` byte-identity pre/post
        # ``__init__``. Stricter than the apply-state test above because
        # this isolates construction.
        rng = np.random.default_rng(7)
        state_before = rng.bit_generator.state
        _ = StationaryPattern(StationaryPatternConfig(), rng)
        state_after = rng.bit_generator.state
        assert state_before == state_after
