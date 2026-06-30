"""Tests for core.seeding — master seed cascade.

``SeedManager._COMPONENTS`` widened from a 3-tuple
(``demand``, ``lead_time``, ``disruptions``) to a 4-tuple by appending
``pattern`` at the end. The widening is safe per the numpy
``SeedSequence.spawn(N+1) ⊇ spawn(N)`` contract: existing master_seed
values produce byte-for-byte identical demand / lead_time / disruptions
streams. The critical regression test
``test_seeding_3way_backward_compat_after_4way_widening`` locks this
contract; it is the single most important test in this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.seeding import SeedManager

COMPONENTS = ("demand", "lead_time", "disruptions", "pattern")


class TestSeedManager:
    def test_same_seed_produces_same_per_component_streams(self) -> None:
        a = SeedManager(42)
        b = SeedManager(42)
        for component in COMPONENTS:
            ra = a.rng(component).integers(0, 1_000_000, size=200)
            rb = b.rng(component).integers(0, 1_000_000, size=200)
            assert (ra == rb).all(), f"Streams differ for component {component!r}"

    def test_different_seed_produces_different_streams(self) -> None:
        a = SeedManager(42)
        b = SeedManager(43)
        diffs = []
        for component in COMPONENTS:
            ra = a.rng(component).integers(0, 1_000_000, size=100)
            rb = b.rng(component).integers(0, 1_000_000, size=100)
            diffs.append(bool((ra != rb).any()))
        assert any(diffs)

    def test_components_are_independent(self) -> None:
        consumed = SeedManager(42)
        fresh = SeedManager(42)
        # Drain demand on the consumed manager only.
        _ = consumed.rng("demand").integers(0, 1_000_000, size=1000)
        # Other streams must remain identical to the fresh manager.
        for component in ("lead_time", "disruptions", "pattern"):
            ra = consumed.rng(component).integers(0, 1_000_000, size=100)
            rb = fresh.rng(component).integers(0, 1_000_000, size=100)
            assert (ra == rb).all(), f"Stream {component!r} was perturbed by demand consumption"

    def test_unknown_component_raises(self) -> None:
        sm = SeedManager(0)
        with pytest.raises(KeyError):
            sm.rng("forecast")

    def test_returns_numpy_generator(self) -> None:
        sm = SeedManager(0)
        assert isinstance(sm.rng("demand"), np.random.Generator)

    def test_all_four_components_present(self) -> None:
        sm = SeedManager(0)
        for component in COMPONENTS:
            sm.rng(component)


class TestSeedManagerBackwardCompatAfter4WayWidening:
    """The critical regression suite for the seed cascade.

    Existing scenarios with ``master_seed`` (the 8 reference YAMLs) must
    produce byte-for-byte identical demand / lead_time / disruptions
    streams after the cascade widens from 3 to 4 components. The numpy
    ``SeedSequence.spawn(N+1)`` contract guarantees that the first N
    children of a fresh ``SeedSequence(seed).spawn(N+1)`` exactly equal
    the children of a fresh ``SeedSequence(seed).spawn(N)``. Locked here.

    Critical: each spawn call uses a FRESH ``SeedSequence(seed)``. This
    mirrors what ``SeedManager.__init__`` actually does at runtime —
    consecutive ``.spawn()`` calls on the same ``SeedSequence`` instance
    advance an internal child-counter and produce DIFFERENT children.
    The backward-compat property is fresh-vs-fresh equality, not
    consecutive-spawn equality.
    """

    def test_seeding_3way_backward_compat_after_4way_widening(self) -> None:
        # For each canonical master seed, the 4-component manager's first
        # 3 streams must match a freshly-spawned 3-child SeedSequence
        # byte-for-byte. Locks the numpy spawn contract that all of
        # the seed cascade's backward-compat depends on.
        for master_seed in (0, 42, 12345, 999_999):
            manager = SeedManager(master_seed)
            # Expected: what a hypothetical pre-Bullet-10 (3-component)
            # SeedManager would have produced from the same master_seed.
            # Fresh SeedSequence(seed).spawn(3) — NOT the manager's
            # internal SeedSequence, which it consumed differently.
            expected_children = np.random.SeedSequence(master_seed).spawn(3)
            expected_rngs = [np.random.default_rng(ss) for ss in expected_children]
            for component, expected_rng in zip(
                ("demand", "lead_time", "disruptions"), expected_rngs, strict=True
            ):
                actual = manager.rng(component).integers(0, 1_000_000, size=100)
                expected = expected_rng.integers(0, 1_000_000, size=100)
                assert (actual == expected).all(), (
                    f"A later widening perturbed pre-existing stream "
                    f"{component!r} at master_seed={master_seed}"
                )

    def test_pattern_stream_reachable(self) -> None:
        # The new 4th component is reachable via the SeedManager.rng API.
        # No KeyError; returns a numpy Generator.
        sm = SeedManager(0)
        rng = sm.rng("pattern")
        assert isinstance(rng, np.random.Generator)

    def test_pattern_stream_independent_of_demand_stream(self) -> None:
        # Drawing from the pattern stream must not perturb subsequent
        # draws from the demand stream. The cascade-independence property
        # is locked by ``SeedSequence.spawn`` semantics; this test pins
        # it explicitly for the new component.
        consumed = SeedManager(42)
        fresh = SeedManager(42)
        # Drain the pattern stream on the consumed manager only.
        _ = consumed.rng("pattern").integers(0, 1_000_000, size=1000)
        # The demand stream must remain identical to the fresh manager.
        ra = consumed.rng("demand").integers(0, 1_000_000, size=100)
        rb = fresh.rng("demand").integers(0, 1_000_000, size=100)
        assert (ra == rb).all(), (
            "demand stream was perturbed by pattern stream consumption — "
            "cascade independence violated"
        )
