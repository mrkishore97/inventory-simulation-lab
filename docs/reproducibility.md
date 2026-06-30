# Reproducibility

This is the third of three companion documents and the **determinism pillar** of the
project. Its siblings answer different questions:

- [`theory.md`](theory.md) — *is the inventory science correct?* (the modelling pillar)
- [`architecture.md`](architecture.md) — *how is the system built?* (the structural pillar)
- **`reproducibility.md`** (this document) — *can I trust and replay a result?* (the
  determinism pillar)

The thesis, stated once and defended throughout:

> **Same code, same dependency lock, same `RunConfig`, same `master_seed` ⇒ identical
> Ledger output.** A simulation run is a pure function of its `RunConfig`.

Everything below makes that claim precise, shows how the code enforces it, tells you how
to reproduce any result yourself, and is honest about where the boundary of the guarantee
lies (Section 11).

This document deliberately does **not** re-derive what
[`architecture.md`](architecture.md) already covers structurally — the configuration
contract ([§3](architecture.md#3-the-configuration-contract)) and the single-run seed
cascade ([§9](architecture.md#9-the-seed-cascade)). It operates at a different altitude:
the determinism *contract*, the *two-level* cascade (architecture's §9 covered only the
per-run component level; here we add the Monte Carlo replication level), the
*reproduce-it-yourself* CLI recipes with real pinned numbers, the *enforcement* layer
(which tests and CI guard determinism), and the *threats* to reproducibility with their
mitigations.

## Contents

- [1. Introduction and the reproducibility claim](#1-introduction-and-the-reproducibility-claim)
- [2. The determinism contract](#2-the-determinism-contract)
- [3. Config identity and the hash](#3-config-identity-and-the-hash)
- [4. The seed cascade in two levels](#4-the-seed-cascade-in-two-levels)
- [5. Reproduce from YAML with the CLI](#5-reproduce-from-yaml-with-the-cli)
- [6. The byte-identical re-run gates](#6-the-byte-identical-re-run-gates)
- [7. Monte Carlo replication determinism](#7-monte-carlo-replication-determinism)
- [8. RL determinism](#8-rl-determinism)
- [9. Enforcement through tests and CI](#9-enforcement-through-tests-and-ci)
- [10. Reference values](#10-reference-values)
- [11. Threats to reproducibility and mitigations](#11-threats-to-reproducibility-and-mitigations)

### Module to section map

Every determinism surface in the codebase has a documented home below.

| Module / artifact | Role | Sections |
| --- | --- | --- |
| `core/seeding.py` (`SeedManager`) | the one source of stochasticity | [§2](#2-the-determinism-contract), [§4](#4-the-seed-cascade-in-two-levels) |
| `core/config.py` (`RunConfig.to_json` / `config_hash`) | config canonicalisation + identity | [§3](#3-config-identity-and-the-hash) |
| `inventory_twin/cli.py` (`run` / `montecarlo`) | reproduce from YAML | [§5](#5-reproduce-from-yaml-with-the-cli), [§6](#6-the-byte-identical-re-run-gates) |
| `analytics/monte_carlo.py` (`monte_carlo`) | replication-level seeding | [§7](#7-monte-carlo-replication-determinism) |
| `core/environment.py` (`InventoryEnv.reset`) | RL reseed path | [§8](#8-rl-determinism) |
| `.github/workflows/ci.yml` | the automated gate | [§9](#9-enforcement-through-tests-and-ci) |
| `tests/unit/test_config.py`, `tests/integration/test_{cli,scenario_io,stress_scenarios}.py` | the determinism tests | [§6](#6-the-byte-identical-re-run-gates), [§9](#9-enforcement-through-tests-and-ci) |
| `data/scenarios/*.yaml` | the reproducible reference inputs | [§10](#10-reference-values) |
| `uv.lock` | the dependency boundary | [§2](#2-the-determinism-contract), [§11](#11-threats-to-reproducibility-and-mitigations) |

## 1. Introduction and the reproducibility claim

A teaching twin is only trustworthy if a number it shows can be regenerated on demand. If
"my base-stock policy costs \$5,562.05 over this horizon" cannot be reproduced byte for
byte, it is an anecdote, not a measurement. So reproducibility is treated here as a
**first-class engineering property**, defended by tests and CI rather than asserted in
prose.

The claim has a precise scope. Hold these four things fixed and the Ledger output is
identical:

1. **the code** — the same commit;
2. **the dependency lock** — the same `uv.lock` (NumPy, pandas, and Python versions are
   part of the determinism boundary; see [§11](#11-threats-to-reproducibility-and-mitigations));
3. **the `RunConfig`** — every field of the validated configuration;
4. **the `master_seed`** — which is itself a field of the `RunConfig`.

Because the `master_seed` lives *inside* the `RunConfig`, points 3 and 4 collapse into
one: **a run is a pure function of its `RunConfig`** within a fixed code + lock
environment. There are no other inputs — no wall-clock, no environment variables, no
hidden global RNG, no thread-scheduling dependence. The rest of this document is the proof
of that sentence.

## 2. The determinism contract

The contract is the absence of every uncontrolled input. Three design choices enforce it.

**One source of stochasticity.** No engine module instantiates its own RNG. Every
stochastic draw — demand, lead time, disruption timing, pattern noise — is pulled from a
`numpy.random.Generator` handed out by `core/seeding.py::SeedManager`. The module docstring
is explicit: *"No engine module instantiates its own RNG; every stochastic call goes
through `SeedManager.rng(...)`."* If a draw cannot reach a generator without going through
the `SeedManager`, there is no back door through which nondeterminism can enter.

**No intentional wall-clock input.** The simulation result does not depend on the time of
day it is run. (Logs and file timestamps that live *outside* the run artifact may of course
record wall-clock time; the point is that the Ledger — the result — does not read it.)

**No execution-order dependence.** The seed each stochastic stream receives is fixed by
its *name* (or, for Monte Carlo, by its replication *index*), never by the order in which
work executes. This is what makes the Monte Carlo path safe to parallelise across cores
([§7](#7-monte-carlo-replication-determinism)) without changing a single output value.

`SeedManager` carries three guarantees, quoted from `core/seeding.py`:

> 1. Two runs with identical master seed produce identical RNG streams per component.
> 2. Consuming from one component's stream does not perturb any other component's stream.
> 3. Adding a new component requires extending `_COMPONENTS` *at the end* of the tuple —
>    earlier components keep their child sequences because `spawn(N+1)` produces the same
>    first N children as `spawn(N)`.

Guarantee 2 (stream independence) is the reason a feature that adds a lead-time draw cannot
silently shift the demand stream. Guarantee 3 (append-at-end) is the one real footgun, and
it is called out again in [§11](#11-threats-to-reproducibility-and-mitigations).

## 3. Config identity and the hash

If a run is a pure function of its `RunConfig`, then a fingerprint of the `RunConfig` is a
fingerprint of the run. That fingerprint is `config_hash`. From `core/config.py`:

```python
def to_json(self) -> str:
    return json.dumps(self.model_dump(mode="json"), sort_keys=True)

def config_hash(self) -> str:
    return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
```

Three properties matter:

- **Canonical, not incidental.** The hash is taken over `to_json()` with `sort_keys=True`,
  not over the YAML text. Two YAML files that differ only in key order, indentation, or
  comments parse to the *same* validated `RunConfig` and therefore the *same* JSON and the
  *same* hash. The serialization format is an input detail; the validated config is the
  identity. (`tests/unit/test_config.py::test_hash_invariant_to_yaml_field_order` pins
  this: a config and its key-reversed YAML hash identically.)
- **Deterministic.** `config_hash()` called twice on the same config returns the same
  digest (`test_hash_deterministic`). It is a SHA-256 hex digest, 64 characters; the first
  12 are used as a short human-facing run-id and default output filename.
- **Sensitive to what matters.** Changing the seed, the policy kind, or any demand /
  lead-time / pattern field changes the hash (`test_hash_changes_with_seed`,
  `test_hash_changes_with_policy_kind`, and the per-kind variants). The hash is the run's
  name, and different runs get different names.

The curated baseline scenario `data/scenarios/example.yaml` hashes to **`84d364adb32c`**.
This 12-character prefix is **test-pinned** — `tests/integration/test_scenario_io.py`
asserts `config.config_hash()[:12] == "84d364adb32c"`, so any accidental change to the
config schema, the example, or the canonicalisation that would alter the baseline's
identity breaks the build immediately.

A subtlety worth internalising: **the hash is config identity, not outcome identity.** Two
different configs can produce identical output. `example.yaml` and
`example_pattern_stationary.yaml` both cost \$5,562.05 over 90 periods — the explicit
`pattern: stationary` block is the *identity* overlay and changes nothing about the demand
stream — yet they hash differently (`84d364adb32c` vs `93eee2e748fc`), because spelling the
default out explicitly changes the serialized JSON. The hash answers "is this the same
configuration?", not "did this produce the same numbers?".

## 4. The seed cascade in two levels

All stochasticity descends from a single integer, `RunConfig.master_seed`, through a
`numpy.random.SeedSequence` tree. There are two levels.

```
                    master_seed   (RunConfig.master_seed, a NonNegativeInt)
                         │
             SeedSequence(master_seed)
                         │
   ┌─────────────────────┴──────────────────────┐
   │  LEVEL 0 — replication   (Monte Carlo only) │
   │  .spawn(N)  →  N child SeedSequences         │
   │  → N replication seeds, each < 2**53         │
   └─────────────────────┬──────────────────────┘
                         │   single run: N = 1, this level is a pass-through;
                         │   each replication seed becomes a fresh master_seed
                         ▼
   ┌─────────────────────┴──────────────────────┐
   │  LEVEL 1 — component     (every run)        │
   │  SeedManager: SeedSequence(seed).spawn(4)   │
   └──┬──────────┬───────────────┬──────────┬────┘
      ▼          ▼               ▼          ▼
   demand    lead_time      disruptions   pattern    (4 independent Generators)
```

**Level 1 — component streams (every run).** `SeedManager(master_seed)` builds
`SeedSequence(master_seed)` and calls `.spawn(4)`, one child per declared component in
`_COMPONENTS = ("demand", "lead_time", "disruptions", "pattern")`. Each child becomes an
independent `default_rng`. A single CLI `run` exercises only this level.

**Level 0 — replication streams (Monte Carlo only).** `analytics/monte_carlo.py` spawns one
level *up*: `SeedSequence(config.master_seed).spawn(N)` produces N replication seeds, and
each becomes the `master_seed` of a fresh `SeedManager` for that replication (delivered via
`config.model_copy(update={"master_seed": seed})`). The cascade is self-similar — the same
`spawn` primitive at two scales.

The append-at-end guarantee (Section 2, guarantee 3) is a property of `spawn`:
`spawn(N+1)` reproduces the first N children of `spawn(N)`. That is why a future component
must be appended to `_COMPONENTS`, and why adding a fifth Monte Carlo replication does not
disturb the first four.

## 5. Reproduce from YAML with the CLI

`inventory_twin/cli.py` is the reproduce-from-YAML entry point. Two subcommands:

```
python -m inventory_twin.cli run        --config CONFIG [--out OUT] [--quiet]
python -m inventory_twin.cli montecarlo --config CONFIG [--reps N] [--jobs J] [--out OUT] [--quiet]
```

`run` wires the config through the engine and policy and writes the Ledger CSV;
`montecarlo` runs N seeded replications and writes the per-replication KPI distribution CSV.
When `--out` is omitted, the output filename **is the run's identity**: `<hash12>.csv` for
`run`, `<hash12>_mc.csv` for `montecarlo`. The reproducible artifact names itself.

A real worked example (regenerated from the live CLI for this document):

```
$ python -m inventory_twin.cli run --config data/scenarios/example.yaml
Run:        m1-baseline
Hash:       84d364adb32c
Horizon:    90 periods
Total cost: $5,562.05
Wrote ledger to: 84d364adb32c.csv
```

The printed `Hash` matches the test-pinned baseline from [§3](#3-config-identity-and-the-hash),
and the file written is `84d364adb32c.csv`. `--quiet` suppresses the human summary (for
scripting); `--out` overrides the filename. The Monte Carlo summary additionally prints the
cost mean with p5/p95 and the mean fill rate — the distribution, not a point estimate.

## 6. The byte-identical re-run gates

The contract from [§2](#2-the-determinism-contract) is not merely documented; it is a
failing-test-if-violated gate. `tests/integration/test_cli.py` runs the CLI twice into two
temp files and asserts the bytes are equal:

```python
def test_determinism_two_runs(self, tmp_path, basic_config):
    # M1 reproducibility gate: same config + same master seed = byte-identical CSV.
    ...
    assert out_a.read_bytes() == out_b.read_bytes()
```

This gate exists on **both** CLI paths: the single-run `run` path and the `montecarlo`
path each have their own `test_determinism_two_runs` asserting `read_bytes() ==
read_bytes()`. A companion test (`test_default_output_uses_config_hash`) pins that the
default filename is exactly `<config_hash()[:12]>.csv`, so the self-naming property is
guarded too.

`read_bytes()` is deliberately stronger than comparing parsed DataFrames: it catches not
only value drift but any change in float formatting, column order, or line endings. If the
two files match to the byte, the run reproduced.

This gate was reproduced live while authoring this document — two `run` invocations of
`example.yaml` produced byte-identical CSVs (`cmp` reported no difference).

## 7. Monte Carlo replication determinism

A distribution must be as reproducible as a point estimate, *and* it must be safe to
compute in parallel. `analytics/monte_carlo.py` achieves both by fixing each replication's
seed by its **index, not its execution order**:

```python
parent = np.random.SeedSequence(config.master_seed)
seeds  = [ <53-bit int from child> for child in parent.spawn(n_replications) ]
# joblib: delayed(_run_replication)(config, i, seed) for i, seed in enumerate(seeds)
```

Two consequences follow:

- **Parallelism cannot change results.** Because replication `i` always uses `seeds[i]`
  regardless of which worker runs it or when, `n_jobs` (the joblib worker count, default
  `-1` = all cores) is a performance knob with **no** effect on the output. Run it on one
  core or sixteen; the distribution is identical.
- **Each replication is independently replayable.** The returned DataFrame carries a `seed`
  column. Any single replication reproduces exactly via
  `single_run.run(config.model_copy(update={"master_seed": seed}))` — you can lift one
  outlier replication out of a 1,000-run distribution and re-examine it deterministically.

One implementation detail earns its place: **replication seeds are kept below 2⁵³** so they
can be stored in a float-backed DataFrame column without losing integer precision. (A
DataFrame row accessed via `df.iloc[i]` upcasts mixed columns to float64; a seed larger
than 2⁵³ would no longer round-trip exactly and the `seed` column would stop reproducing
its replication.) The 53-bit width keeps the seeds high-quality and independent while
remaining exactly representable as float64.

## 8. RL determinism

The reinforcement-learning boundary reuses the same machinery rather than inventing its
own. `core/environment.py::InventoryEnv.reset` follows the Gymnasium convention — an
optional `seed` argument reseeds the episode:

```python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    config = self._config if seed is None else self._config.model_copy(
        update={"master_seed": seed}
    )
    ...
```

Passing `seed=s` constructs a fresh engine whose `master_seed` is `s` — the exact same
`model_copy` reseed mechanism Monte Carlo uses ([§7](#7-monte-carlo-replication-determinism)),
over the exact same two-level cascade ([§4](#4-the-seed-cascade-in-two-levels)). So
`env.reset(seed=s)` is reproducible: the same `s` yields the same opening observation and
the same trajectory under a fixed policy. The Phase-2 readiness gate exercises this with a
canonical Gymnasium random policy driven by a seeded `action_space` over a full episode.

A scope note, so this is not over-claimed: **the Gym environment reset path is
deterministic; full RL training reproducibility additionally depends on the learning
library, the hardware, and the algorithm's own settings** (replay-buffer sampling, GPU
non-associativity, framework seed handling). This document guarantees the environment, not
the agent.

## 9. Enforcement through tests and CI

Determinism is guarded at two levels: a suite of determinism tests, and a CI workflow that
runs the whole suite on every change.

**The determinism tests** — each guarantee maps to a test that fails if it regresses:

| Guarantee | Test |
| --- | --- |
| `config_hash` is deterministic | `tests/unit/test_config.py::test_hash_deterministic` |
| seed changes the identity | `tests/unit/test_config.py::test_hash_changes_with_seed` |
| identity is YAML-order-invariant | `tests/unit/test_config.py::test_hash_invariant_to_yaml_field_order` |
| policy/demand kind changes identity | `tests/unit/test_config.py::test_hash_changes_with_policy_kind` (+ per-kind variants) |
| the baseline hash is pinned | `tests/integration/test_scenario_io.py` → `== "84d364adb32c"` |
| same config ⇒ byte-identical CSV (run **and** MC) | `tests/integration/test_cli.py::*::test_determinism_two_runs` |
| every shipped scenario validates + hashes | `tests/integration/test_stress_scenarios.py::test_all_shipped_scenarios_validate_and_hash` |

**The CI workflow** (`.github/workflows/ci.yml`, verified against the current file) runs on
every push to `main` and on every pull request:

```yaml
- run: uv sync --all-groups        # install against the pinned uv.lock
- run: uv run ruff check .         # lint
- run: uv run ruff format --check .# format gate
- run: uv run mypy core/ policies/ # type gate (core/ and policies/)
- run: uv run pytest               # the full suite, incl. the determinism gates above
```

Two notes on accuracy. First, CI's type gate is scoped to `core/ policies/` — the
type-critical engine and policy code — not the whole tree. Second, CI runs `pytest`
directly; the **coverage** gate (TOTAL 99%, with `cli.py` the sole documented miss) is run
locally as part of the pre-commit quality bar rather than in the workflow. The
byte-identical determinism gates run in both places because they are ordinary pytest tests.

## 10. Reference values

These are the canonical reproducible outputs. Every row was **regenerated from the live
CLI** while authoring this document, not copied from notes. To reproduce any row:

```
python -m inventory_twin.cli run --config data/scenarios/<file>.yaml
```

| Scenario (`name`) | `config_hash[:12]` | Horizon | Total cost |
| --- | --- | --- | --- |
| `example.yaml` (`m1-baseline`) | `84d364adb32c` | 90 | **\$5,562.05** |
| `example_pattern_stationary.yaml` | `93eee2e748fc` | 90 | \$5,562.05 |
| `example_pattern_intermittent.yaml` | `1843ec7d868a` | 90 | \$3,834.51 |
| `example_pattern_seasonal.yaml` | `051687a6f9ef` | 90 | \$6,060.51 |
| `example_pattern_lumpy.yaml` | `9ec2a7db5d66` | 90 | \$9,457.22 |
| `example_pattern_trending.yaml` | `5ac8c6675f45` | 90 | \$13,272.07 |
| `lead_time_spike.yaml` | `14845e879b65` | 90 | \$8,391.85 |
| `seasonal_shock.yaml` | `f44d04098037` | 90 | \$6,171.33 |
| `supplier_outage.yaml` | `3a10f49dc440` | 90 | \$24,349.33 |

**What is pinned vs. documented.** Only `84d364adb32c` (the `example.yaml` baseline) is a
**test-pinned** identity — `test_scenario_io.py` asserts it literally, so it cannot drift
silently. The dollar costs and the other eleven-hex hashes are **documented reference
outputs**: reproducible from the shipped YAML on demand, but verified by regenerating them
(as above), not by a literal assertion. `test_stress_scenarios.py` does pin that *every*
shipped scenario validates and produces a full 64-character hash; it does not pin the exact
dollar figures.

The five pattern archetypes share `example.yaml`'s structure with only the demand-pattern
overlay swapped, so their costs form a clean **cost axis** holding everything else fixed:

> Intermittent \$3,834.51 < Stationary \$5,562.05 < Seasonal \$6,060.51 < Lumpy
> \$9,457.22 < Trending \$13,272.07.

And the illustration from [§3](#3-config-identity-and-the-hash) is visible directly in the
table: `example.yaml` and `example_pattern_stationary.yaml` share the cost \$5,562.05 but
not the hash — config identity is not outcome identity.

## 11. Threats to reproducibility and mitigations

An honest determinism document names what could break the guarantee.

**Dependency drift.** The single largest threat. NumPy's RNG internals, pandas' float
formatting, or Python's behaviour could change byte-level output across versions. This is
exactly why the dependency lock is part of the claim in
[§1](#1-introduction-and-the-reproducibility-claim): the boundary is *"same `uv.lock`."*
`uv sync --all-groups` in CI installs against that pinned lock, so CI reproduces against the
same dependency set the lock describes. Reproducing a historical result means checking out
its commit *and* its lock.

**Platform and floating point.** NumPy's `SeedSequence`/`spawn` cascade is specified to be
platform-independent: the same `master_seed` yields the same child sequences on Linux,
macOS, or Windows. Arithmetic is IEEE-754 double precision, deterministic for the scalar,
non-reduction-order-sensitive operations the engine performs. The engine's within-period
sequence is serial (no parallel float reduction), so there is no summation-order
nondeterminism inside a run.

**Parallelism.** The Monte Carlo path is embarrassingly parallel — replication `i` is a
self-contained function of `seeds[i]` ([§7](#7-monte-carlo-replication-determinism)) — so
joblib never performs an order-sensitive reduction across workers. Changing `--jobs` cannot
change a result.

**Reordering `_COMPONENTS` — the one real footgun.** Because each component's stream is the
*k*-th child of `spawn(len(_COMPONENTS))`, inserting a new component in the *middle* of the
tuple silently re-shuffles every later component's entropy, changing the demand/lead-time
streams of every existing scenario while their configs (and hashes) stay the same — a
reproducibility break that no hash would flag. The mitigation is the discipline encoded in
`SeedManager`'s guarantee 3 and its docstring: **append new components at the end, never
insert.** The baseline-hash pin (`84d364adb32c`) would not catch this on its own — the
config is unchanged — which is precisely why the rule is a documented invariant rather than
a test.

In short: within the boundary of *same code + same lock*, reproducibility is a guarantee
the test suite enforces; outside it, this document tells you exactly which knobs (versions,
the `_COMPONENTS` order) you must hold fixed.
