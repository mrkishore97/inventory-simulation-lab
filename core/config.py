"""Pydantic v2 configuration models for the inventory-twin project.

A :class:`RunConfig` bundles every parameter needed to reproduce a single
simulation run: simulation horizon and initial state, demand and lead-time
specs, the inventory policy, monetary costs, and a master seed.
Configurations serialize to canonical JSON and YAML; the SHA-256 of the JSON
form is the run's content-addressable identifier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class CostConfig(BaseModel):
    model_config = _FROZEN

    unit_cost: NonNegativeFloat
    holding_per_unit_per_period: NonNegativeFloat
    ordering_fixed: NonNegativeFloat
    backorder_per_unit_per_period: NonNegativeFloat = 0
    lost_sale_per_unit: NonNegativeFloat = 0
    expediting_per_unit: NonNegativeFloat = 0


class SimulationConfig(BaseModel):
    model_config = _FROZEN

    horizon: PositiveInt
    initial_on_hand: NonNegativeInt
    initial_on_order: NonNegativeInt = 0
    backorder_policy: Literal["backorder", "lost_sales"] = "backorder"


class NormalDemandConfig(BaseModel):
    model_config = _FROZEN

    kind: Literal["normal"] = "normal"
    mean: NonNegativeFloat
    std: NonNegativeFloat


class PoissonDemandConfig(BaseModel):
    """Poisson-distributed demand.

    The textbook integer-demand distribution: counts of independent events
    over a fixed period at constant rate ``λ``. Mean and variance are both
    equal to ``rate``. Right model for low-volume / item-count SKUs where
    Normal is wrong (a Normal(2, 1) demand can be -0.4 units; Poisson(2)
    yields {0, 1, 2, 3, …} naturally). ``PositiveFloat`` enforces
    ``rate > 0`` — degenerate λ=0 ("always zero demand") is not a Poisson
    and would belong to a separate ``ConstantDemandConfig`` if ever needed.
    """

    model_config = _FROZEN

    kind: Literal["poisson"] = "poisson"
    rate: PositiveFloat


class NegativeBinomialDemandConfig(BaseModel):
    """Negative-Binomial-distributed demand (over-dispersed integers).

    The textbook over-dispersed integer-demand distribution: mean is
    ``n(1-p)/p`` and variance is ``n(1-p)/p²``, so variance/mean = ``1/p``
    > 1 (strictly over-dispersed). Right model for real SKU demand where
    Poisson's variance=mean constraint underestimates volatility — almost
    always the case in practice because real demand carries un-modeled
    drivers (promotions, weather, basket size, weekday effects).

    The (n, p) parameterization comes from the textbook "trials until
    n-th success" interpretation but extends via the Gamma–Poisson mixture
    to non-integer ``n``: ``X ~ Poisson(Λ)`` where ``Λ ~ Gamma(n, (1-p)/p)``.
    ``PositiveFloat`` n captures this broader contract; numpy's
    ``Generator.negative_binomial(n, p)`` accepts float n natively. The
    ``p`` field is the first exclusive-bounds Pydantic field in this
    codebase: ``Field(gt=0, lt=1)`` enforces the open interval (0, 1)
    since both endpoints are degenerate (p=0 diverges; p=1 puts all mass
    at 0).
    """

    model_config = _FROZEN

    kind: Literal["negative_binomial"] = "negative_binomial"
    n: PositiveFloat
    p: Annotated[float, Field(gt=0.0, lt=1.0)]


class GammaDemandConfig(BaseModel):
    """Gamma-distributed demand (continuous, strictly positive support).

    The textbook continuous demand distribution: strictly positive support
    ``(0, ∞)``, right-skewed by construction, two-parameter family that
    decouples mean and variance: ``mean = shape × scale`` and
    ``variance = shape × scale²``. Where Normal is symmetric and can
    return negatives (requiring the ``max(0.0, …)`` clip in
    ``NormalDemand``), Gamma is naturally non-negative and matches the
    shape of most SKU demand and almost all lead-time demand distributions.

    Parameter naming is locked to ``shape, scale`` (mirroring numpy's
    ``rng.gamma(shape, scale)`` API surface) over the textbook ``(k, θ)``
    and the rate parameterization ``(α, β)``. The textbook ``k = shape``
    and ``θ = scale`` equivalence holds; the rate parameterization with
    ``β = 1/scale`` is deliberately rejected because it inverts the
    relationship with numpy's API (most error-prone form).

    Both fields are ``PositiveFloat`` — strictly positive, no upper
    bound; no cross-field invariant. Special cases: ``shape = 1`` is the
    exponential distribution (memoryless arrivals); integer ``shape`` is
    the Erlang distribution; as ``shape → ∞`` (with scale adjusted to
    preserve mean), Gamma converges to Normal — a pedagogically useful
    limit. Gamma is closed under summation when scales match, which is
    why it is the natural choice for lead-time demand modeling.
    """

    model_config = _FROZEN

    kind: Literal["gamma"] = "gamma"
    shape: PositiveFloat
    scale: PositiveFloat


class LognormalDemandConfig(BaseModel):
    """Lognormal-distributed demand (continuous, strictly positive, heavy-tailed).

    The textbook heavy-tail demand distribution: strictly positive
    support ``(0, ∞)``, sub-exponential right tail (decays slower than
    any exponential), the classical model for rare-event-dominated
    demand (pandemic stockpiling, viral surges, holiday spikes) and for
    any SKU whose demand is multiplicatively driven (X = product of
    many small independent factors → ln X approximately Normal by CLT →
    X is Lognormal).

    Parameter naming **deliberately deviates** from the Gamma rule of
    "numpy-native ASCII names". numpy's ``rng.lognormal(mean, sigma)``
    uses ``mean`` for the underlying Normal's location parameter — NOT
    the mean of the resulting Lognormal. This is the well-known numpy
    foot-gun: a user reading YAML ``mean: 2.21`` reasonably assumes the
    resulting demand averages 2.21 (it actually averages ≈ 10). We
    therefore use the textbook ``mu, sigma`` names, where ``mu`` and
    ``sigma`` are the parameters of the *underlying* log-Normal: if
    ``X ~ Lognormal(mu, sigma)`` then ``ln(X) ~ Normal(mu, sigma)``.
    The Gamma rule (prefer numpy-native) still applies when numpy's
    names are unambiguous; here numpy itself is misleading and we
    deviate to the unambiguous textbook form.

    Resulting moments (NOT the parameters):
        ``E[X]    = exp(mu + sigma²/2)``
        ``Var[X]  = (exp(sigma²) − 1) × exp(2*mu + sigma²)``

    ``mu`` is plain ``float`` — the **only demand config field that
    accepts negative values**. Mathematically correct: Lognormal supports
    any real ``mu`` (mean = ``exp(mu + sigma²/2)`` is positive for any
    real ``mu``); slow-moving SKUs with mean < 1 require ``mu < 0``
    (e.g., ``mu = -2`` gives mean ≈ 0.135). ``sigma`` is ``PositiveFloat``
    (``sigma = 0`` is the degenerate point mass at ``exp(mu)`` and is
    rejected). No cross-field invariant.
    """

    model_config = _FROZEN

    kind: Literal["lognormal"] = "lognormal"
    mu: float
    sigma: PositiveFloat


class EmpiricalDemandConfig(BaseModel):
    """Empirical demand — IID resampling from a stored history (e.g., M5 SKU sample).

    The first **non-parametric** arm of :data:`DemandConfig`. Instead of
    describing a demand distribution with parameters (mean, std, shape,
    scale, etc.), it carries a path to a parquet file whose ``demand``
    column IS the distribution: at each ``draw()`` the engine returns a
    uniformly-random element of that column (``rng.choice(history)``).

    Use cases (unlocked by future bullets):

    - **M5 archetype replay** — bundle a SKU's 5-year history and run
      classical policies against the actual marginal demand shape with
      zero parametric approximation.
    - **Historical reconstruction** — given a real sales sequence, ask
      "what would have been the optimal policy" without parametric fitting.
    - **Custom corporate data** — point ``history_path`` at any parquet
      with a ``demand`` column.

    Parameter rationale:

    - ``history_path``: ``pathlib.Path`` (Pydantic v2 coerces ``str`` →
      ``Path``; serializes to ``str`` in JSON/YAML). Path is interpreted
      **relative to the current working directory** at engine
      ``__init__`` time. Same convention as the existing CLI invocation
      (``python -m inventory_twin.cli run --config data/scenarios/…``).
      The file's existence, schema, and content validity are checked at
      :class:`EmpiricalDemand.__init__` time (NOT at config-validation
      time) — Pydantic stays I/O-free; bad files fail loudly at engine
      construction with a clear ``ValueError``.

    File-format expectations (enforced at engine init):

    - Parquet with a ``demand: float64`` column (other columns are
      ignored).
    - At least one row.
    - All values finite (no NaN, +inf, -inf).
    - All values non-negative.

    Determinism caveat (locked but worth understanding):

        **The config hash includes ``history_path`` but NOT the file
        contents.** Two scenarios with different paths hash differently;
        two scenarios with the same path but mutated file contents hash
        identically yet produce different demand streams. Users are
        responsible for treating bundled parquets as immutable
        artifacts. A future bullet may add an optional
        ``history_sha256: str`` fingerprint field that validates at
        load time.

    Resampling semantics: **IID** (`numpy.random.Generator.choice`). Each
    ``draw()`` is independent — serial correlation in the source history
    is NOT preserved. Sequential replay and block-bootstrap variants are
    out of scope for the chassis bullet and may land as additional
    config fields in future bullets if the friction warrants.
    """

    model_config = _FROZEN

    kind: Literal["empirical"] = "empirical"
    history_path: Path


DemandConfig = Annotated[
    NormalDemandConfig
    | PoissonDemandConfig
    | NegativeBinomialDemandConfig
    | GammaDemandConfig
    | LognormalDemandConfig
    | EmpiricalDemandConfig,
    Field(discriminator="kind"),
]


class StationaryPatternConfig(BaseModel):
    """Stationary (identity) pattern — equivalent to "no pattern overlay".

    The trivial arm of the ``PatternConfig`` discriminated union and the
    Pydantic-level default for ``RunConfig.pattern``. Combined with the
    locked ``Field(default_factory=lambda: StationaryPatternConfig())``
    on ``RunConfig``, every existing scenario YAML continues to validate
    and produce byte-for-byte identical demand streams without any
    schema change.

    No fields beyond the discriminator: identity is fully stateless. The
    matching ``StationaryPattern.apply(base, t)`` returns ``base``
    unchanged for all base and t. Concrete pattern arms with actual
    time-modulation logic (Seasonal, Trending, Intermittent, Lumpy) land
    as additional arms of ``PatternConfig``.
    """

    model_config = _FROZEN

    kind: Literal["stationary"] = "stationary"


class SeasonalPatternConfig(BaseModel):
    """Sinusoidal multiplicative seasonal pattern — first concrete arm of ``PatternConfig``.

    Implements cyclic demand modulation via the textbook decomposition:

        ``apply(base, t) = base * (1 + amplitude * sin(2π·t/period + phase))``

    Real seasonal demand (retail Q4 spikes, summer apparel, ice cream,
    cold-medicine flu season, etc.) is the most common deviation from
    stationarity. The multiplicative form scales seasonality with the
    demand level (a 20% Q4 surge is bigger in absolute units when the
    baseline is high) and preserves the time-averaged mean of the base
    distribution exactly (``sin`` averages to 0 over a full cycle).

    Parameter bounds rationale:

    - ``amplitude``: ``Annotated[float, Field(ge=0.0, lt=1.0)]`` — closed
      at 0, **OPEN at 1**. At ``amplitude=0`` the factor is constantly 1
      and ``apply(base, t) == base`` (degenerate-to-identity; useful for
      A/B comparisons against Stationary). At ``amplitude < 1`` the
      factor range is strictly ``(1 - amplitude, 1 + amplitude) ⊂ (0, 2)``,
      so non-negativity is guaranteed by the bound — no runtime clip in
      ``apply()``. ``amplitude=1`` is EXCLUDED because it would produce
      zero-demand troughs (sin=-1 at t = 3·period/4), which is degenerate
      intermittent behavior: use ``IntermittentPattern`` for
      stochastic zero-demand periods.
    - ``period``: ``PositiveInt`` — number of periods per full cycle.
      Common values: 7 (weekly), 12 (monthly within a year), 52 (weekly
      within a year). ``period=1`` is technically valid but degenerate
      (factor is constant across all t).
    - ``phase``: any real (radians). Default 0 means ``t=0`` sits at the
      mean of the cycle (sin(0) = 0); peak is at ``t = period/4`` and
      trough at ``t = 3·period/4``. To position the peak at ``t=0``, use
      ``phase = π/2``. To shift the cycle by ``N`` periods, use
      ``phase = 2π·N/period``.
    """

    model_config = _FROZEN

    kind: Literal["seasonal"] = "seasonal"
    amplitude: Annotated[float, Field(ge=0.0, lt=1.0)]
    period: PositiveInt
    phase: float = 0.0


class TrendingPatternConfig(BaseModel):
    """Linear-multiplicative trending pattern — second concrete arm of ``PatternConfig``.

    Implements monotone demand drift via the textbook linear trend:

        ``apply(base, t) = max(0.0, base * (1 + slope * t))``

    Real trending demand is the second-most-common deviation from
    stationarity after seasonality: growing SKUs (emerging products,
    expanding distribution, category growth) have positive slope; declining
    SKUs (sunsetting products, fashion exits, post-launch decay) have
    negative slope. The linear-multiplicative form scales with demand level
    (matching :class:`SeasonalPatternConfig`'s multiplicative chassis) and
    preserves the mean of the base distribution at ``t=0`` (factor at
    ``t=0`` is exactly ``1.0`` in IEEE 754).

    Parameter rationale:

    - ``slope``: ``FiniteFloat`` (Pydantic v2 alias for
      ``Annotated[float, Field(allow_inf_nan=False)]``). Per-period
      fractional change: ``slope=0.02`` means +2%/period growth;
      ``slope=-0.01`` means -1%/period decline. Any **finite** real
      accepted (positive, zero, or negative; no magnitude bound) — only
      NaN, +inf, and -inf are rejected at config-load time. This is **not
      a domain bound**: it is strictly a numeric sanity gate protecting
      the engine's no-NaN/inf ledger invariant from silent corruption
      (without it, ``max(0.0, base * (1 + NaN * t))`` would propagate NaN
      through the demand stream). ``slope=0`` is the degenerate-to-identity
      case (factor is constantly ``1.0``; equivalent to
      :class:`StationaryPatternConfig`).

    Non-negativity strategy: **clip-at-source** in the pattern's
    ``apply()``. At sufficiently steep negative slope, the factor
    ``1 + slope * t`` crosses zero at ``t = -1/slope`` and goes negative
    beyond; the ``max(0.0, ...)`` clip handles this. Mirrors
    :class:`NormalDemand`'s clip-at-source rule. This is the **third
    non-negativity satisfaction path** in the codebase, alongside
    Stationary's unconditional passthrough and Seasonal's bounds-based
    guarantee — picked here because bounds-based for slope would require
    cross-field validation with ``simulation.horizon``, which is brittle.

    Trends do **not** reset at horizon boundaries: ``apply()`` consumes
    the engine's monotonically increasing ``t`` directly. Sawtooth /
    periodic-reset trends are out-of-scope.
    """

    model_config = _FROZEN

    kind: Literal["trending"] = "trending"
    slope: FiniteFloat


class IntermittentPatternConfig(BaseModel):
    """Intermittent (Bernoulli zero-mask) pattern — third concrete arm, FIRST stochastic arm.

    Implements intermittent demand by masking the base draw to zero on a random
    subset of periods:

        ``apply(base, t) = base   with probability occurrence_probability``
        ``apply(base, t) = 0.0    otherwise``

    Real intermittent demand (slow-moving SKUs, spare parts, low-velocity items)
    has many zero-demand periods interspersed with active ones. This is the
    **first pattern arm that consumes the seeded ``"pattern"`` RNG stream** the
    chassis reserved — Stationary / Seasonal / Trending are all
    deterministic and never draw. The matching
    :class:`~demand.patterns.intermittent.IntermittentPattern` draws exactly one
    uniform per ``apply()`` (before the keep/zero branch), so the ``"pattern"``
    stream advances deterministically by one per period.

    Parameter rationale:

    - ``occurrence_probability``: ``Annotated[float, Field(gt=0.0, le=1.0)]`` —
      the per-period probability the base demand is kept. The expected average
      demand interval is ``ADI ≈ 1 / occurrence_probability``. **OPEN at 0**
      (``gt``): ``p=0`` would zero ALL demand forever (no orders ever) — a
      pathological config, rejected. **CLOSED at 1** (``le``): ``p=1`` is the
      degenerate-to-identity case (every period kept; ``apply == base``), ALLOWED
      for A/B parity with ``SeasonalPatternConfig.amplitude == 0`` and
      ``TrendingPatternConfig.slope == 0``. The bounded ``Field`` rejects NaN and
      ±inf intrinsically (they fail the gt/le comparisons), so no separate
      ``FiniteFloat`` gate is needed — same as ``amplitude``.

    Non-negativity is free: the output is either ``base`` (non-negative by the
    ``Demand`` contract) or ``0.0`` — no clip needed (Stationary's
    unconditional-passthrough path, not Trending's clip path).
    """

    model_config = _FROZEN

    kind: Literal["intermittent"] = "intermittent"
    occurrence_probability: Annotated[float, Field(gt=0.0, le=1.0)]


class LumpyPatternConfig(BaseModel):
    """Lumpy (Bernoulli burst-mask) pattern — fourth concrete arm; closes the M2 pattern set.

    The sibling of :class:`IntermittentPatternConfig`: the same Bernoulli mask, but the
    kept value is *amplified* by a burst multiplier:

        ``apply(base, t) = base * burst_multiplier   with probability occurrence_probability``
        ``apply(base, t) = 0.0                        otherwise``

    Models the colloquial lumpy-demand shape — long runs of no-demand periods punctuated by
    occasional large bursts (spare parts, capital goods, promotional spikes). Like Intermittent,
    it consumes the seeded ``"pattern"`` stream (one draw per ``apply()``).

    Parameter rationale:

    - ``occurrence_probability``: ``Annotated[float, Field(gt=0.0, le=1.0)]`` — identical to
      :class:`IntermittentPatternConfig` (per-period burst probability; ``p=0`` rejected as
      all-zero-forever; ``p=1`` the degenerate boundary — every period a burst).
    - ``burst_multiplier``: ``Annotated[float, Field(gt=1.0, allow_inf_nan=False)]`` — how much
      larger demand is on a burst period. **Strictly > 1** (``gt``): ``m=1`` collapses Lumpy into
      Intermittent (use that arm for the no-burst case, mirroring Seasonal's ``amplitude=1``
      exclusion); ``m<1`` would *shrink* demand (anti-lumpy). ``allow_inf_nan=False`` is required
      because ``gt=1.0`` alone admits ``+inf`` (``inf > 1.0`` is True), which would corrupt the
      ledger via ``base * inf`` — the ``FiniteFloat`` gate combined with a lower bound. No upper
      bound (a 100x burst is valid).

    **SB caveat:** a *constant* ``burst_multiplier`` scales nonzero magnitudes but does NOT by
    itself raise the **CV² of the nonzero demand sizes** (nonzero size = ``base * m``, so its CV²
    equals the base's). The realized Syntetos-Boylan quadrant therefore depends on the base
    distribution — a low-variance base may still classify as *intermittent* rather than *lumpy*.
    This arm is the colloquial lumpy shape, not a guaranteed SB-lumpy generator. A *stochastic*
    burst magnitude (which would raise the nonzero CV²) is intentionally out of scope.

    Non-negativity is free: the output is ``base * m`` (non-negative; ``base ≥ 0``, ``m > 1``) or
    ``0.0`` — no clip.
    """

    model_config = _FROZEN

    kind: Literal["lumpy"] = "lumpy"
    occurrence_probability: Annotated[float, Field(gt=0.0, le=1.0)]
    burst_multiplier: Annotated[float, Field(gt=1.0, allow_inf_nan=False)]


# 5-arm discriminated union (Lumpy landed) — the pattern set is complete
# (Stationary identity + Seasonal + Trending + Intermittent + Lumpy). The discriminator field
# name (``"kind"``) is fixed; each arm is a purely additive change to this annotation. Order =
# chronological order of bullets to keep the JSON schema stable.
PatternConfig = Annotated[
    StationaryPatternConfig
    | SeasonalPatternConfig
    | TrendingPatternConfig
    | IntermittentPatternConfig
    | LumpyPatternConfig,
    Field(discriminator="kind"),
]


class DeterministicLeadTimeConfig(BaseModel):
    model_config = _FROZEN

    kind: Literal["deterministic"] = "deterministic"
    lead_time: PositiveInt


class NormalLeadTimeConfig(BaseModel):
    """Normal-distributed lead time, rounded to an integer number of periods.

    ``NormalLeadTime.sample()`` draws ``X ~ Normal(mean, std)`` and returns
    ``max(1, round(X))`` — round-to-nearest period, clipped to >= 1 (the
    periodic model requires a strictly positive lead time, and the clip also
    absorbs the negative tail of the Normal). ``mean`` is the pre-rounding
    mean lead time and must be positive; ``std = 0`` is allowed and
    degenerates to ``round(mean)`` on every draw (a near-deterministic case).
    """

    model_config = _FROZEN

    kind: Literal["normal"] = "normal"
    mean: PositiveFloat
    std: NonNegativeFloat


class GammaLeadTimeConfig(BaseModel):
    """Gamma-distributed lead time, rounded to an integer number of periods.

    ``GammaLeadTime.sample()`` draws ``X ~ Gamma(shape, scale)`` on ``(0, ∞)``
    and returns ``max(1, round(X))`` — round-to-nearest period, clipped to >= 1.
    The textbook delivery-delay distribution (positive support, right-skewed).
    Parameter naming ``shape, scale`` mirrors numpy and ``GammaDemandConfig``;
    the pre-rounding mean is ``shape * scale`` and variance ``shape * scale²``.
    """

    model_config = _FROZEN

    kind: Literal["gamma"] = "gamma"
    shape: PositiveFloat
    scale: PositiveFloat


class LognormalLeadTimeConfig(BaseModel):
    """Lognormal-distributed lead time, rounded to an integer number of periods.

    ``LognormalLeadTime.sample()`` draws ``X ~ Lognormal(mu, sigma)`` on
    ``(0, ∞)`` and returns ``max(1, round(X))`` — round-to-nearest period,
    clipped to >= 1. Heavy right tail: the model for delivery delays that are
    usually short but occasionally very long.

    Parameter-naming foot-gun (same as ``LognormalDemandConfig``): numpy's
    ``Generator.lognormal(mean, sigma)`` parameter ``mean`` is the **underlying
    Normal's location**, NOT the mean of the resulting Lognormal. We use the
    unambiguous textbook ``mu, sigma`` (parameters of the underlying Normal: if
    ``X ~ Lognormal(mu, sigma)`` then ``ln(X) ~ Normal(mu, sigma)``). The
    resulting pre-rounding moments are ``E[X] = exp(mu + sigma²/2)`` and
    ``Var[X] = (exp(sigma²) − 1) · exp(2·mu + sigma²)``.

    ``mu`` is ``FiniteFloat`` (any finite real, NaN/±inf rejected): negative
    ``mu`` is valid and gives a sub-1 mean lead time (most draws round to 0,
    clipped to 1 — a fast, near-deterministic lead time). ``sigma`` is
    ``PositiveFloat`` (``sigma = 0`` is the degenerate point mass at
    ``exp(mu)`` and is rejected).
    """

    model_config = _FROZEN

    kind: Literal["lognormal"] = "lognormal"
    mu: FiniteFloat
    sigma: PositiveFloat


LeadTimeConfig = Annotated[
    DeterministicLeadTimeConfig
    | NormalLeadTimeConfig
    | GammaLeadTimeConfig
    | LognormalLeadTimeConfig,
    Field(discriminator="kind"),
]


class DisruptionWindow(BaseModel):
    """One lead-time disruption window: a ``(start, duration, multiplier)`` triple.

    During periods ``t`` in the half-open interval ``[start, start + duration)``,
    an order placed at ``t`` has its lead time multiplied by ``multiplier`` (then
    rounded to the nearest integer period and clipped to >= 1 via
    :func:`leadtime.base.round_to_periods`). ``multiplier > 1`` delays the order
    (the typical supply disruption), ``multiplier == 1`` is a no-op, and
    ``0 < multiplier < 1`` expedites it (the result still clips to >= 1).

    ``start`` is a ``NonNegativeInt`` (a disruption may begin at period 0);
    ``duration`` is a ``PositiveInt`` (a zero-length window is meaningless);
    ``multiplier`` is a ``PositiveFloat`` (mirrors the lead-time ``shape`` /
    ``scale`` / ``sigma`` params — strictly positive; ``+inf`` is admitted but
    would crash loudly at ``round_to_periods``, the same garbage-in property
    those siblings carry).
    """

    model_config = _FROZEN

    start: NonNegativeInt
    duration: PositiveInt
    multiplier: PositiveFloat


class NoDisruptionConfig(BaseModel):
    """No lead-time disruption — equivalent to "no disruption overlay".

    The trivial arm of the ``DisruptionConfig`` discriminated union and the
    Pydantic-level default for ``RunConfig.disruption`` (via a
    ``default_factory``), so every existing scenario YAML that omits the
    ``disruption:`` block still validates and produces byte-for-byte identical
    Ledgers. The matching ``NoDisruption.apply(base_lead_time, t)`` returns
    ``base_lead_time`` unchanged. Mirrors ``StationaryPatternConfig``.
    """

    model_config = _FROZEN

    kind: Literal["none"] = "none"


class ScheduledDisruptionConfig(BaseModel):
    """Deterministic schedule of lead-time disruption windows.

    Holds a non-empty list of :class:`DisruptionWindow` triples. An order placed
    at period ``t`` has its lead time multiplied by the product of the
    multipliers of every window active at ``t`` (windows may overlap and compose
    multiplicatively; the order of the list is irrelevant). The schedule is fully
    deterministic — ``ScheduledDisruption`` accepts but never draws from the
    ``"disruptions"`` RNG stream, which is reserved for a future stochastic
    disruption generator. Mirrors ``SeasonalPatternConfig`` as the first concrete
    arm of its union.
    """

    model_config = _FROZEN

    kind: Literal["scheduled"] = "scheduled"
    windows: list[DisruptionWindow] = Field(min_length=1)


DisruptionConfig = Annotated[
    NoDisruptionConfig | ScheduledDisruptionConfig,
    Field(discriminator="kind"),
]


class SQPolicyConfig(BaseModel):
    model_config = _FROZEN

    kind: Literal["sQ"] = "sQ"
    reorder_point: float
    order_quantity: PositiveInt


class SSPolicyConfig(BaseModel):
    """(s,S) min-max reorder policy.

    The ``model_validator`` enforces ``reorder_point < order_up_to`` so the
    triggered order quantity (``order_up_to − inventory_position``) is always
    strictly positive at the trigger boundary; the policy's ``decide`` body
    therefore needs no ``max(0, …)`` clip.
    """

    model_config = _FROZEN

    kind: Literal["sS"] = "sS"
    reorder_point: float
    order_up_to: PositiveFloat

    @model_validator(mode="after")
    def _check_s_lt_S(self) -> Self:
        if self.reorder_point >= self.order_up_to:
            raise ValueError(
                f"(s,S) policy requires reorder_point < order_up_to; "
                f"got reorder_point={self.reorder_point}, order_up_to={self.order_up_to}"
            )
        return self


class BaseStockPolicyConfig(BaseModel):
    """Base-stock (order-up-to-S) periodic-review policy.

    Single-parameter policy: every period, raise inventory position to
    ``target_level``. Distinct from (s,S): there is no reorder threshold,
    so under positive demand the policy fires every period (no dead zone).
    Optimal under the K=0 special case of the periodic-review newsvendor
    setting. ``PositiveFloat`` enforces ``target_level > 0``; no
    cross-field invariant is needed because there is only one field.
    """

    model_config = _FROZEN

    kind: Literal["base_stock"] = "base_stock"
    target_level: PositiveFloat


class PeriodicReviewPolicyConfig(BaseModel):
    """(R,S) periodic-review order-up-to policy.

    Two-parameter time-driven policy: every ``review_period`` periods,
    raise inventory position to ``order_up_to``. Categorically distinct
    from (s,Q) / (s,S) / base-stock — the firing schedule is gated by the
    clock, not the inventory level. Periods between reviews are silent
    regardless of how low IP drops. Models warehouse cycle counts, fixed
    supplier order windows, or shared-truck consolidation.

    No cross-field invariant — ``review_period`` and ``order_up_to`` are
    independent. ``PositiveInt`` enforces ``review_period >= 1`` (so
    ``state.t % review_period`` is always well-defined); ``PositiveFloat``
    enforces ``order_up_to > 0``.
    """

    model_config = _FROZEN

    kind: Literal["RS"] = "RS"
    review_period: PositiveInt
    order_up_to: PositiveFloat


PolicyConfig = Annotated[
    SQPolicyConfig | SSPolicyConfig | BaseStockPolicyConfig | PeriodicReviewPolicyConfig,
    Field(discriminator="kind"),
]


class RunConfig(BaseModel):
    model_config = _FROZEN

    simulation: SimulationConfig
    demand: DemandConfig
    # ``pattern`` is the chassis field for the demand-pattern
    # overlay (seasonal / trending / intermittent / lumpy in Bullets 11-14).
    # The ``default_factory`` returns a fresh ``StationaryPatternConfig`` per
    # construction, so every existing scenario YAML that omits the ``pattern:``
    # block still validates and produces byte-for-byte identical demand
    # streams. The factory (not a class-level default) prevents shared mutable
    # state across configs even though ``_FROZEN`` already forbids mutation.
    pattern: PatternConfig = Field(default_factory=lambda: StationaryPatternConfig())
    lead_time: LeadTimeConfig
    # ``disruption`` is the chassis field for the lead-time
    # disruption overlay — event-driven ``(start, duration, multiplier)`` windows
    # that multiply the lead time during active periods. The ``default_factory``
    # returns a fresh ``NoDisruptionConfig`` per construction, so every existing
    # scenario YAML that omits the ``disruption:`` block still validates and
    # produces byte-for-byte identical Ledgers. Mirrors the ``pattern`` field
    # above (both are overlays defaulting to their identity arm).
    disruption: DisruptionConfig = Field(default_factory=lambda: NoDisruptionConfig())
    policy: PolicyConfig
    costs: CostConfig
    master_seed: NonNegativeInt
    name: str | None = None
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> RunConfig:
        return cls.model_validate_json(s)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)

    @classmethod
    def from_yaml(cls, s: str) -> RunConfig:
        return cls.model_validate(yaml.safe_load(s))

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
