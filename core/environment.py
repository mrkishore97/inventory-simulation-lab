"""Gymnasium-compatible wrapper for :class:`InventoryEngine`.

This is the boundary at which classical policies, RL agents, and Monte Carlo
runners all talk to the same engine through a single interface.

    obs, info             = env.reset(seed=...)        # Point A snapshot
    obs, r, term, trunc, info = env.step(action)       # action -> Point A of next period

Per the locked timing convention (see :mod:`core.simulation`), the observation
returned by ``reset()`` and ``step()`` is **Observation Point A** — the post-
fulfillment, pre-order snapshot the policy reads to decide. The Ledger row
written each period is **Observation Point B** (post-cost-accrual). Same
period, two snapshots; conflating them silently corrupts policy logic or
analytics. Read the engine module docstring before changing anything here.

Reward and termination
----------------------
``reward = -(holding + ordering + purchase + stockout)`` for the period just
closed; negative-cost so reward maximization == cost minimization (PPO/SAC
default).

``terminated`` is always ``False`` — finite-horizon inventory has no MDP-
internal absorbing state. ``truncated`` is ``True`` at horizon end. The
distinction matters for Phase 2 PPO bootstrapping (truncation uses bootstrap,
termination doesn't).

Spaces
------
``observation_space`` lows = ``[0, 0, 0, -inf]``; only ``inventory_position``
can legitimately go negative (when accumulated backorders exceed pipeline).
Highs are ``inf`` — observations grow unboundedly with config. SB3-style RL
training relies on ``VecNormalize`` for obs scaling, not ``observation_space``
bounds, so ``inf`` here is conventional.

``action_space`` is finite ``[0, 1e6]``. RL action normalization (SB3
``RescaleAction``, PPO/SAC actor scaling) divides by ``high - low`` and breaks
under ``np.inf``. ``1e6`` is a generous safe constant — realistic single-
period orders for M1 configs (demand_mean ≤ 1000, lead_time ≤ 100) are below
``1e5``. M2 may parameterize via config if stochastic LT or capacity makes
this scale problematic.

RL noise and action clipping
----------------------------
``step()`` rejects non-finite actions (``nan``/``inf``) — they bypass every
comparison guard (``nan < 0`` is ``False``) and would silently corrupt the
trajectory. Out-of-bounds-but-finite actions pass through; silent env-side
clipping is intentionally NOT done because it would mask bugs in classical
policies (a buggy ``-50`` order should crash, not become ``0``). For Phase-2
RL training where Gaussian policies sample microscopically out-of-bounds,
wrap with ``gym.wrappers.ClipAction`` or rely on the library's built-in
clipping (SB3 clips before stepping).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from core.config import RunConfig
from core.ledger import Ledger
from core.simulation import EngineState, InventoryEngine

# Generous safe upper bound on a single-period order. Documented above.
_ACTION_HIGH = 1e6


class InventoryEnv(gym.Env[npt.NDArray[np.float64], npt.NDArray[np.float64]]):
    """Gymnasium-compatible wrapper for :class:`InventoryEngine`."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self._config = config
        self._engine: InventoryEngine | None = None
        self._period: int = 0

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf], dtype=np.float64),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=_ACTION_HIGH,
            shape=(1,),
            dtype=np.float64,
        )

    # ---- Public API ----

    @property
    def ledger(self) -> Ledger:
        if self._engine is None:
            raise RuntimeError("call reset() before reading ledger")
        return self._engine.ledger

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        super().reset(seed=seed)
        config = (
            self._config if seed is None else self._config.model_copy(update={"master_seed": seed})
        )
        self._engine = InventoryEngine(config)
        self._period = 0
        state = self._engine.initial_observation()
        return self._encode(state), {"period": 0}

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        if self._engine is None:
            raise RuntimeError("call reset() before step()")

        arr = np.asarray(action, dtype=np.float64)
        if arr.shape != (1,):
            raise ValueError(f"action must be shape (1,), got shape {arr.shape}")
        if not np.isfinite(arr[0]):
            raise ValueError(f"action must be finite, got {arr[0]}")
        order_placed = float(arr[0])

        period_just_closed = self._period
        state, done = self._engine.step(order_placed)
        row = self._engine.ledger.row(period_just_closed)
        reward = -(
            row["holding_cost"] + row["ordering_cost"] + row["purchase_cost"] + row["stockout_cost"]
        )
        self._period += 1
        return self._encode(state), reward, False, done, {"period": self._period}

    # ---- Internal ----

    @staticmethod
    def _encode(state: EngineState) -> npt.NDArray[np.float64]:
        return np.array(
            [state.on_hand, state.on_order, state.backorders, state.inventory_position],
            dtype=np.float64,
        )
