"""Master seed cascade — single source of stochasticity for a simulation run.

A :class:`SeedManager` constructs one independent ``numpy.random.Generator``
per declared component from one master seed via
``numpy.random.SeedSequence.spawn()``. No engine module instantiates its own
RNG; every stochastic call goes through ``SeedManager.rng("demand")`` /
``"lead_time"`` / ``"disruptions"``.

The cascade guarantees:

  1. Two runs with identical master seed produce identical RNG streams per
     component.
  2. Consuming from one component's stream does not perturb any other
     component's stream.
  3. Adding a new component requires extending ``_COMPONENTS`` *at the end*
     of the tuple — earlier components keep their child sequences because
     ``spawn(N+1)`` produces the same first N children as ``spawn(N)``.
     Inserting in the middle would silently re-shuffle every later
     component's entropy.
"""

from __future__ import annotations

import numpy as np


class SeedManager:
    _COMPONENTS: tuple[str, ...] = ("demand", "lead_time", "disruptions", "pattern")

    def __init__(self, master_seed: int) -> None:
        master = np.random.SeedSequence(master_seed)
        children = master.spawn(len(self._COMPONENTS))
        self._rngs: dict[str, np.random.Generator] = {
            name: np.random.default_rng(ss)
            for name, ss in zip(self._COMPONENTS, children, strict=True)
        }

    def rng(self, component: str) -> np.random.Generator:
        if component not in self._rngs:
            raise KeyError(
                f"Unknown component '{component}'. Known components: {sorted(self._rngs)}"
            )
        return self._rngs[component]
