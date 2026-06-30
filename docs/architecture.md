# Architecture

How the inventory twin is built: its layers, the contracts that hold them together,
the invariants you must respect to change it safely, and the **as-built reality** —
including the scaffold modules that were deliberately routed around rather than filled in.

This is the structural companion to two sibling documents:

- [`theory.md`](theory.md) — the inventory *science* (what each formula means and why).
- [`reproducibility.md`](reproducibility.md) — the *determinism* contract end to end.

Where theory.md answers *"what is the model,"* this document answers *"how is the
system put together, and what will silently break if I get it wrong."*

## Contents

1. [Introduction and reading guide](#1-introduction-and-reading-guide)
2. [Layered architecture and the dependency rule](#2-layered-architecture-and-the-dependency-rule)
3. [The configuration contract](#3-the-configuration-contract)
4. [The periodic engine](#4-the-periodic-engine)
5. [The two observation points](#5-the-two-observation-points)
6. [The ledger buffer](#6-the-ledger-buffer)
7. [The overlay model](#7-the-overlay-model)
8. [Factories and registries](#8-factories-and-registries)
9. [The seed cascade](#9-the-seed-cascade)
10. [The rollout primitive](#10-the-rollout-primitive)
11. [The RL boundary](#11-the-rl-boundary)
12. [Analytics, recommender, and visualization](#12-analytics-recommender-and-visualization)
13. [The UI layer](#13-the-ui-layer)
14. [As-built tree and routed-around modules](#14-as-built-tree-and-routed-around-modules)

### Module → section map

Every tracked source module has a documented home. If you find code with no section,
the doc is stale.

| Module / family | Section |
|---|---|
| `core/config.py` | [§3](#3-the-configuration-contract) |
| `core/simulation.py` (`InventoryEngine`, `EngineState`) | [§4](#4-the-periodic-engine), [§5](#5-the-two-observation-points) |
| `core/ledger.py` (`Ledger`) | [§6](#6-the-ledger-buffer) |
| `core/seeding.py` (`SeedManager`) | [§9](#9-the-seed-cascade) |
| `core/environment.py` (`InventoryEnv`) | [§11](#11-the-rl-boundary) |
| `core/state.py`, `core/events.py` (empty stubs) | [§14](#14-as-built-tree-and-routed-around-modules) |
| `demand/` (base, registry, 6 distributions) | [§8](#8-factories-and-registries) |
| `demand/patterns/` (base, registry, 5 arms) | [§7](#7-the-overlay-model), [§8](#8-factories-and-registries) |
| `leadtime/` (base, registry, 4 distributions) | [§8](#8-factories-and-registries) |
| `leadtime/disruptions/` (base, registry, none/scheduled) | [§7](#7-the-overlay-model), [§8](#8-factories-and-registries) |
| `policies/` (base, registry, 4 policies) | [§8](#8-factories-and-registries), [§10](#10-the-rollout-primitive) |
| `analytics/` (kpis, classical, bullwhip, risk, monte_carlo, pareto, sensitivity) | [§12](#12-analytics-recommender-and-visualization) |
| `recommender/` (classify, advisor) | [§12](#12-analytics-recommender-and-visualization) |
| `visualization/` (7 factory modules) | [§12](#12-analytics-recommender-and-visualization), [§13](#13-the-ui-layer) |
| `experiments/` (common.rollout, single_run.run) | [§10](#10-the-rollout-primitive) |
| `inventory_twin/cli.py` | [§10](#10-the-rollout-primitive) |
| `ui/` (app, components, 9 pages) | [§13](#13-the-ui-layer) |
| `scenarios/` (data-only) | [§14](#14-as-built-tree-and-routed-around-modules) |
| `scripts/` (M5 data-prep) | [§14](#14-as-built-tree-and-routed-around-modules) |

---

## 1. Introduction and reading guide

The twin is a **deterministic, periodic-time inventory simulator** wrapped in a
Streamlit teaching UI. One simulation is fully described by a single validated
`RunConfig`; running it produces one `Ledger` of per-period state; analytics reduce
that ledger (or re-run the config) into KPIs, distributions, and charts.

Three ideas recur and are worth holding in mind while reading:

- **One typed contract.** `RunConfig` (Pydantic v2, frozen) is the only way to
  describe a run. Everything downstream is written against it, and its
  `config_hash()` is the run's reproducibility fingerprint ([§3](#3-the-configuration-contract)).
- **Two observation points per period.** Within each period there is a snapshot the
  *policy* reads and a different snapshot *analytics* read. Conflating them corrupts
  results silently — nothing throws ([§5](#5-the-two-observation-points)).
- **Additive plug-in arms.** New demand distributions, patterns, lead-time models,
  disruptions, and policies are added as *arms* behind factories, never by editing
  the engine loop ([§7](#7-the-overlay-model), [§8](#8-factories-and-registries)).

If you change one thing, read its section *and* [§5](#5-the-two-observation-points)
first.

## 2. Layered architecture and the dependency rule

The codebase is a stack of pure-compute layers with a thin Streamlit shell at the
edge. Dependencies point **inward**, toward the configuration schema:

```
  ui/            Streamlit app.py + 9 pages + components   <- only layer importing streamlit
   |  builds RunConfig, reads frames, renders figures
   v
  visualization/ pure Streamlit-free chart / table factories
   |
   v
  analytics/  recommender/        experiments/
   |  reducers + experiment          rollout, single_run
   |  generators                     (the canonical drive loop)
   v                                  |
  core/ engine . ledger . seeding . environment   <----------+
   |  + demand/  leadtime/  policies/  (plug-in arm families)
   v
  core/config.py   RunConfig - the shared schema contract
```

**The dependency rule.** `RunConfig` and its config models are the **shared schema
contract** the engine, experiments, analytics, and scenario loading are written
against; the UI builds `RunConfig` objects and reads the frames they produce. (This
is a contract, not a literal universal import — a pure reducer such as
`analytics.kpis` operates on a DataFrame and need not import `core.config` at all.)

**Streamlit lives only in `ui/`.** Every layer below it — including `visualization/`,
which produces Plotly figures and DataFrame tables — is import-clean of Streamlit and
unit-testable headless. This is what lets the compute core stay 100% coverage-gated
while the UI is exercised by lightweight `AppTest` smokes.

**Coverage and gate policy.** `core/`, `demand/`, `leadtime/`, `policies/`,
`analytics/`, `recommender/`, `visualization/`, and `experiments/` are held at 100%
line coverage (TOTAL 99%; the sole documented miss is a `cli.py` argument-parse
branch). `ui/`, `tests/`, and `docs/` are coverage-omitted: `ui/` is thin glue over
the gated core, exercised by smokes rather than line-counted. The full gate stack
(`coverage`, `ruff check`, `ruff format --check`, `mypy`) plus the `config_hash`
non-invasiveness check runs in CI ([§14](#14-as-built-tree-and-routed-around-modules)).

## 3. The configuration contract

`core/config.py` is the typed boundary of the whole system. Every model sets
`model_config = ConfigDict(frozen=True, extra="forbid")`: configs are immutable
(safe to share, hash, and cache) and **reject unknown keys** (a typo'd YAML field
fails loudly instead of being silently ignored).

**Discriminated unions.** Each pluggable family is a tagged union keyed on a `kind`
literal, so Pydantic parses the right arm from YAML/JSON automatically:

```python
DemandConfig = Annotated[
    NormalDemandConfig
    | PoissonDemandConfig
    | NegativeBinomialDemandConfig
    | GammaDemandConfig
    | LognormalDemandConfig
    | EmpiricalDemandConfig,
    Field(discriminator="kind"),
]
```

There are **five** such unions — `DemandConfig`, `PatternConfig`, `LeadTimeConfig`,
`DisruptionConfig`, and `PolicyConfig` — each `Field(discriminator="kind")`. Adding an
arm widens its union; nothing else in the parser changes ([§8](#8-factories-and-registries)).

**The root aggregate.** `RunConfig` composes one arm from each family plus
`SimulationConfig` (horizon, initial state, backorder policy), `CostConfig`, a
`master_seed`, and optional `name`/`notes`. Two of its fields are *overlays* that
default to their identity arm via `default_factory` — so a scenario YAML that omits
the `pattern:` or `disruption:` block validates and produces a byte-identical run
([§7](#7-the-overlay-model)).

**Canonical serialization and the fingerprint.** `RunConfig` round-trips through both
JSON and YAML. The reproducibility identity is computed on canonical JSON:

```python
def to_json(self) -> str:
    return json.dumps(self.model_dump(mode="json"), sort_keys=True)

def config_hash(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
```

`sort_keys=True` makes the hash independent of field order, so two YAML files that
differ only in key ordering hash identically. The canonical `example.yaml` hashes to
`84d364adb32c...`; that 12-char prefix is the **non-invasiveness gate** every
docs/test/UI bullet must hold (see [reproducibility.md](reproducibility.md) for the
full contract).

## 4. The periodic engine

`core/simulation.py::InventoryEngine` is the simulation heart: a **deterministic
forward state machine over integer periods**, not a continuous-time discrete-event
simulator. Simulation time is the period index `t`.

**The locked within-period sequence** runs five steps every
period, in this fixed order:

```
period t:
  1  receive arrivals      on_hand += order_received          (drain pending[t])
  2  observe demand        demand[t] = pattern.apply(demand.draw(), t)
  3  fulfill demand        -> sales / lost_sales / backorders / backorders_cleared
  -----------------------  Observation Point A  (policy reads here)
  4  place order           arrival_t = t + disruption.apply(L, t);  pending[arrival_t] += q
  5  accrue costs          holding + ordering + purchase + stockout
  -----------------------  Observation Point B  (ledger row t written here)
```

**Externally driven, not self-looping.** The engine exposes two methods and holds no
loop of its own:

- `initial_observation()` runs steps 1–3 of period 0 and returns the Point-A snapshot.
- `step(order_placed)` runs steps 4–5 of the current period, advances `t`, then runs
  steps 1–3 of the next period (or terminates at the horizon), returning
  `(EngineState, done)`.

The caller in between is a policy ([§10](#10-the-rollout-primitive)). This inversion
is what lets a classical policy, the Gym wrapper, and a Monte Carlo runner all drive
the *same* engine through the *same* contract.

**Arrivals are a dict, not an event heap.** Pipeline orders are tracked as
`_pending_arrivals: dict[int, float]` keyed by arrival period; an order placed at `t`
with effective lead time `L` increments `pending[t + L]`, and step 1 pops `pending[t]`
(default `0`). Multiple orders landing in one period sum into a single entry — the
same semantics as a DES event heap, with no event-ordering to reason about. (This is
why `core/events.py` is empty — see [§14](#14-as-built-tree-and-routed-around-modules).)

**Engine constraints.** `initial_on_order` must be `0` (a non-zero starting pipeline
needs a cohort-arrival convention deferred to Phase 2); lead time must be `>= 1`
period (an order placed in step 4 of period `t` cannot arrive in step 1 of the same
period). Both are enforced at construction or at config validation.

## 5. The two observation points

This is the single most error-prone invariant in the simulator, and the engine's own
module docstring (`core/simulation.py`) is its authoritative statement. Two snapshots
exist **within the same period** and mean different things:

- **Observation Point A** — post-fulfillment, *pre-order*. Returned by
  `initial_observation()` and `step()` as an `EngineState`. This is what the **policy**
  reads to decide an order; `(s, Q)` and `(s, S)` reorder triggers compare against it.
- **Observation Point B** — end-of-period, *post-cost-accrual*. Written to the
  `Ledger` row for period `t`. This is what **analytics, audit, and KPIs** read.

`EngineState` carries `t, on_hand, on_order, backorders, inventory_position`, with
`inventory_position = on_hand + on_order - backorders`. The fields diverge between A
and B because steps 4–5 happen in between:

| Field | Point A (`EngineState`) | Point B (Ledger row `t`) |
|---|---|---|
| `t` | period the *next* decision applies to | period just closed |
| `on_hand` | post-step-3 | post-step-3 (steps 4–5 don't touch it) |
| `on_order` | post-step-1, pre-step-4 (received, no new order yet) | post-step-4 (prior − received + placed) |
| `backorders` | post-step-3 | post-step-3 |
| `inventory_position` | computed from the A row | computed from the B row |

**Audit recipe.** From a ledger row you can recover what the policy saw:

```
inventory_position_policy[t] = inventory_position_ledger[t] - order_placed[t]
on_order_policy[t]           = on_order_ledger[t]           - order_placed[t]
```

Choosing the wrong snapshot does not raise — it produces plausible-looking nonsense in
either policy logic or analytics. Read this section before touching the engine, the
Gym wrapper, or any KPI.

## 6. The ledger buffer

`core/ledger.py::Ledger` is the single sink for per-period state. It is deliberately
**dumb storage**: it enforces no accounting conventions (those are engine contracts) —
it just stores columns fast.

**Why it exists.** Appending to a Pandas DataFrame inside the inner loop is O(n²) and
would blow the 100 ms single-run target at horizon = 365. Instead the Ledger
pre-allocates one fixed-length NumPy array per column at construction, offers an O(1)
`record(t, **fields)`, and materializes a DataFrame exactly once via `to_dataframe()`
at end-of-run. Downstream consumers all work on that frame.

**Fixed schema, append-at-end.** The 14-column `_SCHEMA` (on_hand, on_order,
inventory_position, demand, sales, backorders, lost_sales, backorders_cleared,
order_placed, order_received, and the four cost columns) is set at class level. An
unknown column name in `record()` raises `KeyError` loudly. New columns must be
appended at the end so existing column-position tests stay stable — the same
append-only discipline the seed cascade uses ([§9](#9-the-seed-cascade)).

## 7. The overlay model

Two parts of a run are **additive overlays** that compose onto a base draw and default
to an identity arm. Both were retrofitted onto the engine with no loop change, which is
why a pre-overlay scenario YAML still produces a byte-identical run.

**The demand-pattern overlay** multiplies the base demand draw by a period-dependent
factor. It composes at exactly one site, `InventoryEngine._draw_demand`:

```python
return self._pattern.apply(self._demand.draw(), self._t)
```

`PatternConfig` has **five arms**: `Stationary` (the identity default), `Seasonal`,
`Trending`, `Intermittent`, and `Lumpy`. The default `StationaryPattern.apply`
returns `base` unchanged, so omitting the `pattern:` block is a no-op.

**The lead-time disruption overlay** multiplies the sampled lead time during scheduled
`(start, duration, multiplier)` windows. It composes at order placement, in
`InventoryEngine._end_period`:

```python
arrival_t = self._t + self._disruption.apply(base_lead_time, self._t)
```

`DisruptionConfig` has two arms: `NoDisruption` (the identity default) and
`ScheduledDisruption`. The default returns the lead time unchanged. Because `apply` runs
once, at placement, a window scopes to orders *placed* while it is active: an order already
in the pending-arrivals queue when a window opens keeps its scheduled arrival.

**A deliberate asymmetry.** The pattern overlay perturbs **demand**; the lead-time
disruption overlay perturbs **lead time only**. There is intentionally *no* demand-
disruption overlay — time-windowed demand shocks are deferred to Phase 2 (this is the
limitation called out in theory.md §4 / §12, and the reason the four demand-windowed
stress scenarios are not yet expressible). See theory.md §3
(patterns) and §4 (lead time and disruptions) for the science.

## 8. Factories and registries

Each plug-in family has a single construction site — a `make_*` factory that dispatches
a config arm to its concrete implementation:

| Family | Factory | Base type |
|---|---|---|
| demand | `demand.registry.make_demand` | `demand.base.Demand` |
| pattern | `demand.patterns.registry.make_pattern` | `demand.patterns.base.Pattern` |
| lead time | `leadtime.registry.make_lead_time` | `leadtime.base.LeadTime` |
| disruption | `leadtime.disruptions.registry.make_disruption` | `leadtime.disruptions.base.Disruption` |
| policy | `policies.registry.make_policy` | `policies.base.Policy` |

`InventoryEngine.__init__` calls the first four factories (binding each to its seeded
RNG, [§9](#9-the-seed-cascade)); the rollout layer calls `make_policy`. The engine
**never** dispatches on `config.*.kind` inline — keeping construction behind one
function per family means the engine call sites stay stable as arms are added.

**The "add an arm" recipe** (from `demand/registry.py`, generalizes to every family):

1. Add the `*Config` model to `core/config.py` and widen the union with
   `Field(discriminator="kind")`.
2. Add the concrete subclass (e.g. a new `Demand`) under the family package.
3. Add one `isinstance` arm to the factory mapping config → class.
4. Add a dispatch test.

The factory raises `ValueError` on an unregistered config type — a defensive guard
against a call site that bypassed Pydantic's discriminator. This four-step,
loop-free extensibility is the spine that carried the project from 1 demand
distribution to 6, and the pattern set from 1 to 5.

## 9. The seed cascade

`core/seeding.py::SeedManager` is the single source of stochasticity. From one
`master_seed` it spawns one independent `numpy.random.Generator` per declared
component via `numpy.random.SeedSequence.spawn`:

```
master_seed
  └─ SeedSequence(master_seed).spawn(4) ─► [ child0, child1, child2, child3 ]
                                              |       |        |         |
                                          "demand" "lead_time" "disruptions" "pattern"
                                              v       v        v         v
                                          default_rng per component (4 independent streams)
```

No engine module instantiates its own RNG; every stochastic call goes through
`SeedManager.rng("demand")` / `"lead_time"` / `"disruptions"` / `"pattern"`. The
cascade guarantees:

1. **Reproducibility** — identical master seed ⇒ identical per-component streams.
2. **Stream independence** — consuming from one component's stream never perturbs
   another's (this is why adding a stochastic pattern arm did not shift demand draws).
3. **Append-at-end ordering** — `_COMPONENTS` must grow only at the tail, because
   `spawn(N + 1)` reproduces the same first `N` children as `spawn(N)`. `"pattern"`
   was appended last for exactly this reason; inserting in the middle would silently
   re-shuffle every later component's entropy.

The end-to-end determinism story — master seed, per-stream isolation, and the
`config_hash` fingerprint — is the subject of [reproducibility.md](reproducibility.md).

## 10. The rollout primitive

`experiments/common.py::rollout` is the one canonical drive loop, so that every caller
shares identical semantics instead of each re-inlining the engine handshake:

```python
def rollout(engine: InventoryEngine, policy: Policy) -> None:
    state = engine.initial_observation()
    done = False
    while not done:
        state, done = engine.step(policy.decide(state))
```

The policy reads the **Point-A** `state` ([§5](#5-the-two-observation-points)) and
returns an order quantity. `experiments/single_run.py::run` is the single-config
convenience over it — build the policy via `make_policy`, build the engine, roll out,
and return `engine.ledger.to_dataframe()`:

```
RunConfig ─► make_policy ──┐
            InventoryEngine ├─► rollout ─► Ledger ─► DataFrame ─► analytics ─► visualization ─► ui
                           ┘   (policy.decide on Point A)
```

`run()` is what the CLI (`inventory_twin/cli.py`), the integration tests, and the
Monte Carlo runner all call — none of them reach the engine through the Gym wrapper.
The classical path is `Policy → InventoryEngine → rollout`; the wrapper in
[§11](#11-the-rl-boundary) is a *parallel* boundary over the same engine for RL.

## 11. The RL boundary

`core/environment.py::InventoryEnv` is a `gymnasium.Env` subclass — the
**RL-compatible boundary over the same engine** used by classical policies,
experiments, Monte Carlo, and the UI. It composes the engine (it does not subclass
it): `reset(seed=…)` constructs a fresh `InventoryEngine` (optionally seed-overriding
the config via `model_copy`), runs steps 1–3 of period 0, and returns the Point-A
observation; `step(action)` validates the action, drives `engine.step`, reads the
just-closed ledger row for costs, and returns `(obs, reward, terminated, truncated,
info)`.

Design choices that matter for Phase-2 training:

- **Observation** = `[on_hand, on_order, backorders, inventory_position]` (the Point-A
  snapshot). The space lows are `[0, 0, 0, -inf]` — only `inventory_position` can go
  negative; highs are `inf` (RL relies on `VecNormalize`, not space bounds, for
  scaling).
- **Action** = a 1-D `Box[0, 1e6]`. The bound is finite (not `inf`) because SB3-style
  action rescaling divides by `high - low` and breaks under `inf`; `1e6` is a generous
  safe constant for M1-scale configs.
- **Reward** = `-(holding + ordering + purchase + stockout)` for the period just
  closed — negative cost, so reward maximization equals cost minimization.
- **Termination** = `terminated` is always `False` (finite-horizon inventory has no
  absorbing state); `truncated` is `True` at horizon end. The distinction is load-
  bearing for PPO bootstrapping (truncation bootstraps, termination does not).
- **Robustness** = non-finite actions (`nan`/`inf`) are rejected; out-of-bounds-but-
  finite actions pass through unclipped on purpose (silent clipping would mask a buggy
  classical policy). The Phase-2 readiness test drives a canonical
  `action_space.sample()` policy for 100 steps through this boundary. See theory.md §13
  for the MDP framing.

## 12. Analytics, recommender, and visualization

All three layers are pure, Streamlit-free, and 100% coverage-gated.

**Analytics splits cleanly in two** — a distinction worth preserving for Phase 2:

- **Reducers over an existing frame** — `analytics.kpis` (the service trinity, cost
  decomposition, efficiency KPIs), `analytics.risk` (VaR/CVaR over a Monte Carlo
  distribution), and `analytics.bullwhip` (`Var(orders)/Var(demand)`). They take a
  DataFrame and summarize it; two calls on the same frame return identical results.
- **Experiment generators that re-run configs** — `analytics.monte_carlo` (N seeded
  replications), `analytics.pareto` (a sweep across a config grid), and
  `analytics.sensitivity` (one-at-a-time perturbation of config paths). They take a
  `RunConfig` and produce a fresh distribution/frame by calling `run()`.

`risk_metrics` makes the split explicit in its own contract: it is a reduction over an
existing distribution and pointedly does *not* re-run a config the way pareto and
sensitivity do.

**The recommender** (`recommender/classify.py` → `recommender/advisor.py`) is a small
pipeline, not a lookup table: classify a demand history into its Syntetos-Boylan
quadrant, summarize it into mean/std, then map the quadrant to one policy with
parameters derived from `analytics.classical` plus a plain-English rationale
(theory.md §10).

**Visualization** (`visualization/`) holds seven pure, Streamlit-free **chart and
table factories** (timeseries, comparison, distribution, pareto, bullwhip,
sensitivity, waterfall) tested without the UI. Most return Plotly figures; some, like
`comparison_table`, return a DataFrame. Keeping them out of `ui/` is what makes the
visual layer unit-testable.

## 13. The UI layer

`ui/` is the only Streamlit-touching layer: `app.py` (the entry point), a `pages/`
directory of nine pages, and a `components/` directory of reusable glue. It is
coverage-omitted thin glue over the gated core, exercised by `AppTest` smokes. Each
page composes core compute + a visualization factory:

| Page | Capability | Core composed |
|---|---|---|
| 1 Single Run | one policy on one world + KPIs + scrubber | `single_run.run`, `kpis`, timeseries/waterfall |
| 2 Policy Comparison | every selected policy raced on a shared seed | `run`, comparison factory (assembled in-page) |
| 3 Pareto Explorer | cost/service frontier + click-to-load | `analytics.pareto`, pareto factory |
| 4 Stress Test | baseline vs lead-time-disrupted A/B | `run`, disruption overlay, scenario export |
| 5 Scenario Library | browse / edit / run / export YAML scenarios | `scenario_io`, `from_yaml` |
| 6 Policy Advisor | demand history → recommended policy + rationale | `recommender.classify`/`advisor` |
| 7 Monte Carlo & Risk | outcome distribution + VaR/CVaR markers | `monte_carlo`, `risk`, distribution factory |
| 8 Bullwhip | order-vs-demand variance amplification | `bullwhip`, bullwhip factory |
| 9 Sensitivity | tornado of perturbable config paths | `sensitivity`, sensitivity factory + spec |

Notable components: `config_builder` (the reusable sidebar plus a per-policy parameter
editor, so multi-policy pages reuse one editor), `run_cache` (memoized runs),
`scenario_io` (YAML load/save), `demand_input` (the parquet/CSV demand reader),
`sensitivity_spec` (a Streamlit-free catalogue of perturbable paths), and the
presentation helpers `kpi_card` and `glossary` (the single source of tooltip text).

## 14. As-built tree and routed-around modules

This document describes **what is built**, not the original scaffold's aspirational
tree. Several modules from the initial scaffold or early plan were deliberately routed
around; they are documented here so a reader does not mistake an empty file for a gap
to fill or rebuild the wrong intended design.

```
core/        config.py . simulation.py . ledger.py . seeding.py . environment.py
             state.py (EMPTY) . events.py (EMPTY)
demand/      base . registry . {normal,poisson,negative_binomial,gamma,lognormal,empirical}
             patterns/ base . registry . {stationary,seasonal,trending,intermittent,lumpy}
leadtime/    base . registry . {deterministic,normal,gamma,lognormal}
             disruptions/ base . registry . {none,scheduled}
policies/    base . registry . {sQ,sS,base_stock,periodic_review}
analytics/   kpis . classical . bullwhip . risk . monte_carlo . pareto . sensitivity
recommender/ classify . advisor
experiments/ common (rollout) . single_run (run)        [no policy_comparison.py]
visualization/  timeseries . comparison . distribution . pareto . bullwhip . sensitivity . waterfall
ui/          app.py . pages/1...9 . components/
inventory_twin/ cli.py
scenarios/   __init__.py only (data lives in data/scenarios/*.yaml)
scripts/     extract_m5_archetypes . generate_m5_sample
docs/        theory.md . architecture.md . reproducibility.md
```

**The authority note** (do not "fix" these by adding dead code):

- **`core/state.py` and `core/events.py` are empty (0-line) scaffold stubs.**
  `EngineState` lives in `core/simulation.py`, *not* `state.py`. Arrivals are a
  `dict[int, float]` ([§4](#4-the-periodic-engine)), *not* an event heap, so there is
  no event module.
- **`experiments/policy_comparison.py` was never built.** Page 2 (Policy Comparison)
  assembles its multi-policy race directly in the UI over the shared `rollout`
  primitive, by design — there is no separate experiment module for it.
- **`scenarios/` is data-only** (just `__init__.py`). Built-in scenarios are YAML files
  under `data/scenarios/`, loaded generically through `RunConfig.from_yaml`; there are
  no scenario *modules*.
- **`scripts/` is offline data-prep** for the M5 empirical archetypes, not part of the
  runtime.

**Quality gates and CI.** Changes are gated by `coverage` (TOTAL 99%; documented
`cli.py` miss), `ruff check`, `ruff format --check`, and `mypy`, plus the
`config_hash` non-invasiveness check (`example.yaml` → `84d364adb32c...`) for any
change that claims not to touch run semantics. CI runs the stack on every push
(`.github/workflows/ci.yml`). The standing discipline — **no dead wrappers, describe
the as-built reality** — is why this section exists.
