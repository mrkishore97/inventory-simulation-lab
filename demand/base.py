"""Demand abstract base.

A ``Demand`` is a per-period stochastic source: each call to ``draw()``
returns one period's demand realization, advancing an internal RNG. The
contract is **stateful** (the RNG mutates on every call) — distinct from
``Policy``, which is stateless because policies read but do not mutate
state. Storing the RNG on the instance keeps the engine's call site clean
(``self._demand.draw()``) and the seed-cascade plumbing
(``SeedManager.rng("demand")``) entirely inside engine ``__init__``.

Concrete arms are constructed via ``demand.registry.make_demand`` from a
``DemandConfig`` plus a spawned ``np.random.Generator``. Subclasses are
internal — consumers configure demand through YAML / config, not by
instantiating distribution classes directly.

Sample-domain note: ``draw() -> float`` is the locked return type.
Future int-valued distributions (Poisson, Negative Binomial) upcast to
float at this boundary because the engine's ``_period_demand``, the
Ledger's float64 columns, and Convention C (``sales <= demand``) all
assume float. Future continuous distributions whose support is
non-negative natively (Gamma, Lognormal) do NOT need a ``max(0.0, …)``
clip — that clip is Normal-specific. Do not cargo-cult it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Demand(ABC):
    """Abstract base for stochastic per-period demand generators."""

    @abstractmethod
    def draw(self) -> float:
        """Return one period's demand realization as a non-negative float.

        The internal RNG advances on every call. Implementations must
        return a finite ``float`` >= 0; the engine assigns the value
        directly to ``self._period_demand`` and the Ledger row's
        ``demand`` column.
        """
        ...
