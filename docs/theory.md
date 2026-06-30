# Theory Notes

The inventory science behind the Inventory Twin. This is the conceptual pillar of
the project: **every modelling choice, formula, and metric the code implements is
derived and explained here**, grounded in the simulator's own numbers rather than
copied from a textbook.

- For *how the system is built* (engine, config, layers, the Gym wrapper) see
  [`architecture.md`](architecture.md).
- For *how to reproduce a run exactly* (seeds, `config_hash`, determinism) see
  [`reproducibility.md`](reproducibility.md).

Each section follows the same shape: **intuition → assumptions → formula → code
pointer → worked example**. Worked examples use the shipped baseline
`data/scenarios/example.yaml` — Normal($\mu=10,\sigma=2$) demand, a deterministic
lead time $L=3$, an $(s,Q)$ policy with $s=20,\ Q=30$, costs (holding $h=0.5$,
ordering $K=20$, unit $5$, backorder $2$/unit/period, lost-sale $0$), horizon $90$,
`master_seed = 42` — whose canonical run hashes to `84d364adb32c` and costs
**\$5,562.05**.

## Notation

| Symbol | Meaning | Maps to |
|---|---|---|
| $\mu$ | mean demand per period | `NormalDemandConfig.mean` |
| $\sigma$ | demand standard deviation per period | `NormalDemandConfig.std` |
| $L$ | replenishment lead time (periods) | `lead_time` config |
| $D$ | demand rate (per period) for EOQ | $\mu$ |
| $K$ | fixed ordering cost per order | `CostConfig.ordering_fixed` |
| $h$ | holding cost / unit / period | `CostConfig.holding_per_unit_per_period` |
| $C_u,\ C_o$ | underage / overage cost (newsvendor) | stockout / holding |
| $z$ | standard-normal safety factor | `z_for_service_level` |
| $\Phi$ | standard-normal CDF; $\Phi^{-1}$ its quantile | `scipy.stats.norm` |
| $\mathrm{IP}$ | inventory position $= \mathrm{OH}+\mathrm{OO}-\mathrm{BO}$ | `EngineState.inventory_position` |
| $s,\ S,\ Q,\ R$ | reorder point, order-up-to, batch, review period | policy configs |

## Table of contents

1. [Introduction](#1-introduction)
2. [The inventory system](#2-the-inventory-system)
3. [Demand modelling](#3-demand-modelling)
4. [Lead time and its disruptions](#4-lead-time-and-its-disruptions)
5. [Inventory control policies](#5-inventory-control-policies)
6. [Classical analytical laws](#6-classical-analytical-laws)
7. [Service levels — the trinity](#7-service-levels--the-trinity)
8. [Costs and efficiency](#8-costs-and-efficiency)
9. [The bullwhip effect](#9-the-bullwhip-effect)
10. [Demand classification and the Policy Advisor](#10-demand-classification-and-the-policy-advisor)
11. [Risk analytics](#11-risk-analytics)
12. [Experimental methods](#12-experimental-methods)
13. [From classical control to reinforcement learning](#13-from-classical-control-to-reinforcement-learning)
14. [References](#14-references)

---

## 1. Introduction

Inventory management is the discipline of deciding **how much to order and when**,
under uncertain demand and a non-zero replenishment lead time, so as to balance two
opposing costs: holding too much (capital, storage, obsolescence) versus holding too
little (stockouts, lost sales, expediting). Every section below is one facet of that
trade-off.

The Inventory Twin is a **single-echelon, periodic-review** simulator. "Single
echelon" means one stocking location facing external demand and replenished by an
external supplier; "periodic review" means time advances in discrete periods and the
ordering decision is taken once per period. This is the classical setting of Silver,
Pyke & Thomas and Zipkin, and the natural substrate for the Phase-2 extensions — a diagnostic
layer over the same engine, with reinforcement learning as a parallel research track (§13).

Two ideas recur throughout:

- **Inventory position, not on-hand, drives decisions.** A good policy reasons about
  stock it already owns *plus* stock in transit, otherwise it double-orders during the
  lead time.
- **Variability, not just the mean, drives cost.** Two demand streams with identical
  means but different variability need very different safety buffers; the entire
  back half of this document (service levels, bullwhip, classification, risk) is about
  variability.

---

## 2. The inventory system

**Intuition.** Each period the warehouse receives whatever it previously ordered,
meets as much demand as it can from stock, then decides whether to order more. What
it cannot meet is either *back-ordered* (remembered and filled later) or *lost*.

**Assumptions.**

- Discrete periods $t = 0, 1, \dots, T-1$ (horizon $T$).
- One decision per period (periodic review).
- Initial pipeline is empty (`initial_on_order = 0`); a non-zero starting pipeline
  needs a cohort-arrival convention that is deliberately out of scope.
- The fulfilment invariant **$\text{sales}_t \le \text{demand}_t$** always holds: every
  unit of demand is either sold from stock, lost, or becomes a backorder.

**State variables** (`core/simulation.py::EngineState`, "Observation Point A" —
post-fulfilment, pre-order). There is no separate `core/state.py`; the engine state
lives in `core/simulation.py` and the per-period ledger row in `core/ledger.py`.

$$\mathrm{IP}_t = \underbrace{\mathrm{OH}_t}_{\text{on hand}} + \underbrace{\mathrm{OO}_t}_{\text{on order}} - \underbrace{\mathrm{BO}_t}_{\text{backorders}}$$

On-hand is physical stock; on-order is the in-transit pipeline; backorders are unmet
demand owed to customers. **Inventory position** nets them into the single quantity a
policy should reason about.

**The per-period sequence** (`core/simulation.py::InventoryEngine.step`). The locked
five-step order is:

1. **Receive arrivals** scheduled for period $t$: $\mathrm{OH} \mathrel{+}= \text{received}$,
   $\mathrm{OO} \mathrel{-}= \text{received}$.
2. **Observe demand** $d_t$ (drawn from the demand model, §3).
3. **Fulfil demand and backorders** from on-hand, respecting $\text{sales}_t \le d_t$.
   Any shortfall becomes a new backorder (backorder mode) or a lost sale (lost-sales
   mode).
4. **Place an order** $q_t$ from the policy (§5), scheduled to arrive at $t + L$.
5. **Accrue costs** for the period (§8) and record the ledger row.

Arrivals are a `dict[int, float]` keyed by arrival period: an order placed in $t$ with
effective lead time $L$ increments `_pending_arrivals[t + L]`. Because step 4 runs
*after* step 1, an order can never arrive in the period it was placed (so $L \ge 1$ is
required).

**Backorder vs. lost-sales** (`SimulationConfig.backorder_policy`). Under **backorder**
unmet demand is carried forward and satisfied by later arrivals (`backorders_cleared`);
under **lost-sales** it vanishes (with an optional per-unit lost-sale penalty). The
distinction changes the cost structure and the meaning of the service levels in §7.

**Flow conservation.** The ledger satisfies, every period, the on-hand recurrence
$\mathrm{OH}_t = \mathrm{OH}_{t-1} + \text{received}_t - \text{sales}_t - \text{backorders\_cleared}_t$
and the demand balance
$d_t = \text{sales}_t + \text{lost\_sales}_t + (\mathrm{BO}_t - \mathrm{BO}_{t-1}) + \text{backorders\_cleared}_t$.
These identities are property-tested over full runs (`tests/` Hypothesis suite) and
are why the simulator can be trusted as a "twin."

> **Note — what the engine does *not* model.** Demand and the demand pattern are fixed
> for the whole horizon: there is no time-windowed *demand* shock mechanism. Only the
> **lead time** can be disrupted in-window (§4). Rectangular demand-window scenarios
> (demand surge/collapse/recovery) are a deferred demand-disruption overlay (§12).

---

## 3. Demand modelling

Demand is the simulator's primary source of uncertainty. It is built in two layers: a
**base distribution** (the per-period magnitude) and an optional **multiplicative
pattern** (how that magnitude trends or switches on and off over time).

### 3.1 Base distributions

`make_demand` (`demand/registry.py`) dispatches on config to one of six arms. The
choice encodes a modelling assumption about the product:

| Distribution | Support | Key property | Use when |
|---|---|---|---|
| **Normal** | $\mathbb{R}$ (clipped $\ge 0$) | symmetric; mean $\mu$, var $\sigma^2$ independent | smooth, high-volume demand |
| **Poisson** | $\{0,1,2,\dots\}$ | **mean = variance** ($\lambda$) | low-count arrivals, no overdispersion |
| **Negative binomial** | $\{0,1,2,\dots\}$ | **variance > mean** (overdispersion) | clustered / "bursty" counts |
| **Gamma** | $(0,\infty)$ | positive, right-skewed, continuous | continuous positive demand with skew |
| **Lognormal** | $(0,\infty)$ | heavy right tail | multiplicative / skewed demand |
| **Empirical** | resampled history | distribution-free bootstrap | when a real demand sample exists |

The recurring summary statistic is the **coefficient of variation**
$\mathrm{CV} = \sigma/\mu$ (and its square $\mathrm{CV}^2$), a scale-free measure of
variability that reappears in safety stock (§6), classification (§10), and risk (§11).
The Poisson-vs-negative-binomial choice is the classic **overdispersion** decision:
real intermittent demand almost always has variance above its mean, which Poisson
cannot represent and the negative binomial can.

The base draw is assumed **IID across periods** (independent, identically
distributed). This is what lets variance add linearly over a lead time — the
foundation of the $\sqrt{L}$ safety-stock result (§6).

### 3.2 Demand patterns

A `Pattern` (`demand/patterns/`) is a multiplicative overlay
$\tilde d_t = \text{pattern}(d_t,\ t)$ applied to each base draw, drawing at most once
per period from an independent `"pattern"` RNG stream. There are **five** arms:

| Pattern | Rule | Models |
|---|---|---|
| **Stationary** | $\tilde d_t = d_t$ (identity) | no time structure |
| **Seasonal** | $\tilde d_t = d_t \cdot (1 + a\sin(2\pi t/p + \phi))$ | holiday / seasonal swings |
| **Trending** | $\tilde d_t = d_t \cdot (1 + g t)$ | linear growth / decline |
| **Intermittent** | $\tilde d_t = d_t$ w.p. $\rho$, else $0$ | sporadic demand (many zero periods) |
| **Lumpy** | $\tilde d_t = m\,d_t$ w.p. $\rho$, else $0$ | sporadic **and** large when it occurs |

Intermittent is a Bernoulli **zero-mask** (occurrence probability $\rho$); Lumpy is a
Bernoulli **burst-mask** that additionally scales each occurrence by $m > 1$, so
$\text{lumpy}(\rho, m) = \text{intermittent}(\rho) \times m$ on the same seed. For
intermittent demand the expected **average demand interval** is $\approx 1/\rho$ — the
quantity §10 classifies on.

> **"Smooth" is not a pattern arm.** The five arms above are the only
> `PatternConfig` types. *Smooth*, *erratic*, *intermittent*, and *lumpy* are
> Syntetos-Boylan **classifications** of an observed history (§10), not generators —
> there is no `SmoothPattern`. (Intermittent and lumpy happen to share a name with two
> classes because those generators reliably produce histories in those quadrants.)

**Worked example — the cost axis.** Running `example.yaml`'s world under each pattern
(holding the policy and seed fixed) orders total cost as

$$\text{Intermittent } \$3{,}834 < \text{Stationary } \$5{,}562 < \text{Seasonal } \$6{,}061 < \text{Lumpy } \$9{,}457 < \text{Trending } \$13{,}272.$$

Intermittent is *cheaper* (long zero runs mean less to hold and fewer stockouts at a
fixed policy); lumpy and trending are dramatically more expensive because a static
$(s,Q)$ sized for mean-10 demand is badly mismatched to bursts and to growth.

---

## 4. Lead time and its disruptions

**Intuition.** The lead time $L$ is the gap between placing an order and receiving it.
It does not change the *economic batch* (§6), but it sets how far ahead the policy must
protect: longer lead times mean more demand can occur before replenishment lands, so
more safety stock.

**Lead-time models** (`leadtime/`, dispatched by `make_lead_time`). The deterministic
arm returns a fixed integer $L$ and draws no randomness. The three stochastic arms —
**Normal**, **Gamma**, **Lognormal** — draw a continuous value per order and convert it
to an integer number of periods via

$$\text{round\_to\_periods}(x) = \max\!\big(1,\ \mathrm{round}(x)\big)$$

(`leadtime/base.py::round_to_periods`; banker's rounding, clipped to $\ge 1$ because the
periodic model schedules arrivals in an integer pending-arrivals queue and an order
cannot arrive the same period it is placed). Gamma and lognormal are the realistic
choices: they are positive by construction and right-skewed, capturing the "usually
on time, occasionally very late" shape of real supply.

**Lead-time demand.** Under IID per-period demand, the demand over a lead time of $L$
periods has mean $\mu L$ and — because variances add — standard deviation $\sigma\sqrt{L}$.
This is the distribution a reorder point must cover (§6).

**Disruptions** (`leadtime/disruptions/scheduled.py`). A `ScheduledDisruption` is a list
of $(\text{start}, \text{duration}, \text{multiplier})$ windows. For an order placed in
period $t$, the effective lead time is

$$L_{\text{eff}}(t) = \text{round\_to\_periods}\!\Big(L \cdot \prod_{w\ \text{active at } t} m_w\Big),$$

the base lead time scaled by the product of every window active at $t$ (half-open
$[\text{start}, \text{start}+\text{duration})$), then rounded. The schedule is fully
deterministic (it draws no RNG), so a disrupted run shares a byte-identical demand
stream with its undisrupted baseline — enabling clean A/B stress tests.

**A window affects only orders placed while it is active.** The multiplier is applied once,
at placement (`disruption.apply(L, t)`), so an order already in transit when a window opens
keeps its original arrival period — a window delays *future* orders, not deliveries already
in the pipeline.

**The rounding cliff.** Because of the final rounding, bands of multipliers collapse
onto the same integer. With $L=3$: $3\times1.6=4.8$, $3\times1.7=5.1$, $3\times1.8=5.4$
all round to **5 periods**, so those three multipliers produce identical runs and
behaviour steps only at the rounding thresholds. The Stress Test page surfaces this
explicitly.

> **Only lead time is disruptable.** A *supplier outage* is approximated by a large
> finite multiplier (e.g. $\times 50$, pushing $L$ far past the horizon so in-window
> orders never arrive), not a literal infinite lead time. Demand-side shocks are out of
> scope here — see §12.

---

## 5. Inventory control policies

A **policy** is a stateless rule mapping the observed inventory position (and, for
periodic review, the period index) to an order quantity. The Twin implements the four
classical policies, spanning the two review regimes. All four read **only**
`state.inventory_position` (and $(R,S)$ additionally reads `state.t`).

| Policy | Review | Rule (`policies/…::decide`) | Parameters |
|---|---|---|---|
| **$(s,Q)$** | continuous | order $Q$ if $\mathrm{IP} \le s$, else $0$ | reorder point $s$, batch $Q$ |
| **$(s,S)$** | continuous | order $S-\mathrm{IP}$ if $\mathrm{IP} \le s$, else $0$ | reorder point $s$, order-up-to $S$ |
| **base-stock** | every period | order $\max(0,\ S-\mathrm{IP})$ | target level $S$ |
| **$(R,S)$** | periodic | every $R$ periods order $\max(0,\ S-\mathrm{IP})$, else $0$ | review period $R$, order-up-to $S$ |

**Continuous vs. periodic.** $(s,Q)$ and $(s,S)$ are *continuous-review* in spirit — they
react the moment inventory position crosses the reorder point $s$ (here, checked each
period). $(R,S)$ is *periodic-review*: it is silent between reviews regardless of how low
stock falls, trading responsiveness for fewer, predictable ordering occasions. **Base
stock** is a period-by-period order-up-to policy — equivalent to review every period in
this simulator, with no reorder dead zone — and is optimal in the $K=0$
(no fixed ordering cost) newsvendor special case (§6.4).

**$(s,Q)$ vs. $(s,S)$.** Both trigger at $\mathrm{IP} \le s$, but $(s,Q)$ orders a *fixed*
batch $Q$ (a single $Q$; multi-$Q$ "catch-up" ordering is a different policy and is not
implemented), whereas $(s,S)$ tops the position back up to $S$. A fixed batch shines when
each demand is small and regular; an order-up-to absorbs variable demand sizes. The
optimality of $(s,S)$ under fixed ordering costs is the classical Scarf (1960) result.

**The protection interval.** The deep idea linking policies to the formulas in §6 is *how
long the current decision must protect against demand*:

- $(s,Q)$ / $(s,S)$: protect over the **lead time $L$** (you can re-react next period).
- $(R,S)$: protect over **$R + L$** (you cannot re-order until the next review, then must
  wait $L$ more).
- base-stock: protect over **$L + 1$** (this period plus the lead time).

This is exactly the interval the Policy Advisor plugs into the safety-stock formula per
quadrant (§10).

---

## 6. Classical analytical laws

These are the closed-form textbook results the simulator is **validated against**:
`analytics/classical.py` implements them as pure functions, and the validation suite
asserts the digital twin reproduces them. They are also what the Policy Advisor uses to
derive starting parameters (§10).

### 6.1 Economic Order Quantity (EOQ)

**Intuition.** Order in big batches and you order rarely (low ordering cost) but hold a
lot (high holding cost); order in small batches and the reverse. EOQ is the batch that
minimises the sum.

**Assumptions.** Constant, known demand $D$; fixed (or zero) lead time; no stockouts; a
unit cost independent of $Q$ (so only ordering and holding depend on $Q$).

**Formula.** Per-period cost is $C(Q) = \tfrac{KD}{Q} + \tfrac{hQ}{2}$. Setting
$C'(Q) = -KD/Q^2 + h/2 = 0$ gives the **Wilson/Harris** EOQ:

$$Q^\* = \sqrt{\frac{2DK}{h}}.$$

`analytics/classical.py::eoq`. EOQ is **lead-time-independent** — lead time moves the
reorder point, not the optimal quantity.

**Worked example.** For `example.yaml` ($D=10,\ K=20,\ h=0.5$):
$Q^\* = \sqrt{2\cdot 10\cdot 20 / 0.5} = \sqrt{800} \approx 28.3$. The shipped policy uses
$Q=30$ — within rounding of the economic batch.

**A feasibility caveat (continuous vs. discrete review).** EOQ silently assumes
*continuous review* — you reorder the instant stock hits the reorder point, however often
that is. The implied order frequency is $D/Q$ per period (a cycle of $Q/D$ periods). When
the economic batch falls below one period's demand ($Q < D$), the implied cycle is
*sub-period* — more than one order per period — which a once-per-period discrete-review
$(s, Q)$ cannot deliver: it places at most one order per period, so it may order every
period and still fall behind. The classical fixes are $(s, nQ)$ (order an integer multiple
of $Q$), $(s, S)$ (order up to a level), or a shorter review period. The Policy Advisor
(§10) surfaces exactly this case as a feasibility warning.

### 6.2 Safety stock and the service factor

**Intuition.** Safety stock is the buffer held *above* expected lead-time demand to
absorb variability. How big depends on how sure you want to be of not stocking out.

**Formula.** With IID demand, lead-time demand has standard deviation $\sigma\sqrt{L}$
(variance adds over the $L$ periods of a cycle), and safety stock is the $z$-quantile
offset of that distribution:

$$\text{SS} = z\,\sigma\,\sqrt{L}, \qquad z = \Phi^{-1}(\text{service level}).$$

`analytics/classical.py::safety_stock`, `z_for_service_level`. A 95% cycle service level
gives $z \approx 1.645$; 97.5% gives $\approx 1.960$; 50% gives $z=0$ (no safety stock).
The $\sqrt{L}$ is the heart of the result — double the lead time and safety stock grows
by only $\sqrt{2}$, not $2$.

### 6.3 Reorder point

The reorder point is **cycle stock + safety stock**:

$$s = \underbrace{\mu L}_{\text{expected lead-time demand}} + \underbrace{z\,\sigma\sqrt{L}}_{\text{safety stock}}.$$

`analytics/classical.py::reorder_point`. This is the $s$ of an $(s,Q)$ policy: order when
inventory position falls to it.

**Worked example.** For `example.yaml` at 95% service:
$s = 10\cdot 3 + 1.645\cdot 2\cdot\sqrt{3} = 30 + 5.70 \approx 35.7$. The shipped policy
uses $s=20$ — *below* the 95% reorder point, a deliberate teaching contrast: the demo
runs a leaner buffer than 95% service would require, which is why it accumulates some
backorders. Raising $s$ toward 36 trades holding cost for service.

### 6.4 The newsvendor

**Intuition.** For a single selling period (or a base-stock policy with no fixed
ordering cost), how much to stock balances the cost of stocking too little ($C_u$ per
unit short) against too much ($C_o$ per unit left over).

**Formula.** The optimal in-stock probability is the **critical ratio**, and the
order-up-to level is its demand quantile:

$$\text{CR} = \frac{C_u}{C_u + C_o}, \qquad S^\* = \mu + \Phi^{-1}(\text{CR})\,\sigma.$$

`analytics/classical.py::newsvendor_level`. Balanced costs ($C_u=C_o$) give
$\text{CR}=0.5 \Rightarrow S^\*=\mu$ (no bias); when stockouts hurt more than leftovers
($C_u > C_o$) the ratio and $S^\*$ rise — stock more. This sets a base-stock policy's
target level, which is optimal in the no-fixed-cost setting.

**Worked example.** With $C_u=2$ (backorder) and $C_o=0.5$ (holding):
$\text{CR}=2/2.5=0.8$, $S^\* = 10 + \Phi^{-1}(0.8)\cdot 2 \approx 11.7$.

---

## 7. Service levels — the trinity

**Intuition.** "Service level" is ambiguous: it can mean *how often you avoid a
stockout*, *what fraction of demand you fill immediately*, or *how often you have stock
on the shelf*. These are genuinely different numbers, and reporting only one (as most
tools do) hides the others. The Twin reports all three (`analytics/kpis.py`), each in
$[0,1]$.

| Type | Name | Definition (`compute_kpis`) |
|---|---|---|
| **Type 1** | cycle service level $\alpha$ | fraction of replenishment **cycles** with no stockout |
| **Type 2** | fill rate $\beta$ | fraction of **demand** met immediately from stock |
| **Type 3** | ready rate | fraction of **periods** ending with positive on-hand |

**Fill rate is current-period only.** Type 2 is
$\sum_t \text{sales}_t / \sum_t \text{demand}_t$, where $\text{sales}_t$ is
*current-period* demand filled now from stock. It deliberately **excludes
`backorders_cleared`** (past demand satisfied late): a backorder cleared in a later
period does **not** retroactively improve the fill rate of the period that went short.
This is the honest reading — a customer who waited was not served immediately. With zero
total demand the fill rate is defined as $1.0$.

**Cycle service level uses receipt-based cycles.** A new cycle begins each time stock
arrives ($\text{order\_received} > 0$); a cycle counts as stocked-out if *any* period in
it has unmet current-period demand ($\text{demand}-\text{sales} > \text{tol}$). This
signal is valid in both backorder and lost-sales modes because the
$\text{sales} \le \text{demand}$ invariant routes every shortfall to lost sales or new
backorders. (An alternative convention delimits cycles by order *placement*; the two
agree in count under constant lead time.)

**Ready rate is time-based, not demand-weighted.** Type 3 counts *periods* with positive
on-hand, so every period counts equally regardless of its demand — a quiet period and a
peak period each contribute one period to the average. A product can post a high ready
rate yet still miss demand on the few high-volume periods that matter most, which is why
it is read alongside the demand-based fill rate, not instead of it.

**Why they diverge.** A product can have a high ready rate (usually in stock) but a poor
fill rate if the rare stockouts coincide with large demand; or a high fill rate but a
low cycle service level if many cycles have a tiny shortfall. Looking at all three at
once is the §7.6 differentiator.

---

## 8. Costs and efficiency

### 8.1 Cost decomposition

Total cost is the sum of four pre-accrued ledger columns (`analytics/kpis.py`):

$$\text{total} = \underbrace{h \cdot \text{on-hand}}_{\text{holding}} + \underbrace{K \cdot \text{order count}}_{\text{ordering}} + \underbrace{c \cdot \text{units ordered}}_{\text{purchase}} + \underbrace{\text{stockout cost}}_{\text{backorder / lost-sale}}.$$

- **Holding** $h$ per unit of end-of-period on-hand per period.
- **Ordering** a fixed $K$ each time an order is placed (so
  $\text{total ordering} = K \cdot \text{order\_count}$, where order count is *events*,
  not units).
- **Purchase** unit cost $c$ times units ordered.
- **Stockout** either a backorder penalty per unit per period of waiting, or a per-unit
  lost-sale penalty, depending on the mode (§2).

`total_cost` is the headline minimised across the Pareto and sensitivity tools (§12).
**Expediting cost** (an emergency-order premium) is deliberately *excluded* — there is no
such ledger column in Phase 1; it is a deferred invasive feature.

### 8.2 Efficiency metrics

- **Average on-hand** — mean end-of-period physical stock.
- **Inventory turns** $= \dfrac{\sum_t \text{demand}_t}{\text{avg on-hand}}$ — how many
  times the held stock "turns over" across the run. **Horizon-based, not annualised**
  (equals annual turns only at a one-year horizon); $\text{nan}$ when average on-hand is
  zero.
- **Days of supply** $= \dfrac{\text{avg on-hand}}{\text{mean demand per period}}$ — how
  many periods of demand the held stock covers.

Turns and days-of-supply are reciprocally related and both flow from **Little's Law**
(average inventory $=$ throughput $\times$ average time in system): held stock, demand
rate, and dwell time are three views of the same flow.

---

## 9. The bullwhip effect

**Intuition.** Even perfectly smooth customer demand can drive a wildly variable *order*
stream upstream, because batching and re-forecasting **manufacture variability the
customer never had**. That amplification, compounding stage by stage up a supply chain,
is the bullwhip effect (Lee, Padmanabhan & Whang 1997).

**The four causes** (Lee et al. 1997): (1) **demand-signal processing** — each tier
re-forecasts from the orders it receives, not true demand, adding its own buffer; (2)
**order batching** — fixed ordering costs make stages order in lumps; (3) the **rationing
(shortage) game** — buyers over-order when supply is scarce; (4) **price fluctuations** —
promotions pull demand forward. The single-echelon Twin exhibits the batching cause
directly.

**Measurement** (`analytics/bullwhip.py`). On one echelon the classical metric (Chen et
al. 2000) is the variance ratio

$$\text{Bullwhip} = \frac{\mathrm{Var}(\text{order\_placed})}{\mathrm{Var}(\text{demand})}$$

(population variance, $\text{ddof}=0$): $>1$ amplification, $=1$ pass-through, $<1$
smoothing, and $\text{nan}$ for constant demand (undefined). Using the raw per-period
order quantity (0 in idle periods, the batch size when an order fires) is deliberate —
its variance *includes the on/off batching lumpiness, which is exactly why bullwhip
exists*.

**Worked example.** Perfectly smooth demand
$[10,10,10,10,10,10]$ ($\mathrm{Var}=0$) drives an $(s,Q)$ order stream like
$[0,0,50,0,0,50]$ ($\mathrm{Var}$ large), so the ratio blows up: batching alone created
variability from nothing.

---

## 10. Demand classification and the Policy Advisor

### 10.1 Syntetos-Boylan classification

**Intuition.** Before choosing a policy, characterise the demand. Two scale-free axes
capture what matters: **how often** demand occurs, and **how variable** it is when it
does.

**Metrics** (`recommender/classify.py`):

- **ADI** (average demand interval) $= \dfrac{\#\text{periods}}{\#\text{non-zero periods}}$
  — intermittency. Large ADI ⇒ many zero periods.
- **$\mathrm{CV}^2$** — squared coefficient of variation of the **non-zero** demand sizes
  — erraticness of size when demand occurs.

**The 2×2.** Against the cutoffs $\text{ADI} = 1.32$ and $\mathrm{CV}^2 = 0.49$
(Syntetos, Boylan & Croston 2005), demand falls into four quadrants:

| | $\mathrm{CV}^2 < 0.49$ | $\mathrm{CV}^2 \ge 0.49$ |
|---|---|---|
| **$\text{ADI} < 1.32$** | **smooth** | **erratic** |
| **$\text{ADI} \ge 1.32$** | **intermittent** | **lumpy** |

These cutoffs are not arbitrary: Syntetos & Boylan (2005) derived them from where
**Croston's method** and the **Syntetos-Boylan Approximation (SBA)** swap which has lower
forecast mean-squared error. The categorisation is the standard segmentation step in
intermittent-demand practice. (As noted in §3.2, "smooth" here is a *class of observed
history*, not a generator.)

### 10.2 The heuristic Policy Advisor

**Intuition.** Each quadrant has a natural policy and a natural *protection interval*;
plugging the demand moments into the §6 formulas over that interval yields concrete
starting parameters and a plain-English rationale (`recommender/advisor.py`).

| SB class | Policy | Protection interval | Parameters |
|---|---|---|---|
| smooth | $(s,Q)$ | $L$ (continuous) | $Q=\text{EOQ}$, $s=\text{ROP}$ |
| erratic | $(s,S)$ | $L$ (continuous) | $s=\text{ROP}$, $S = s + \text{EOQ}$ |
| intermittent | $(R,S)$ | $R+L$ (periodic) | $R \approx \text{ADI}$, $S = \mu(R{+}L) + z\sigma\sqrt{R{+}L}$ |
| lumpy | base-stock | $L+1$ (every period) | $S = \mu(L{+}1) + z\sigma\sqrt{L{+}1}$ |

The logic: smooth demand suits a fixed economic batch; erratic demand needs an
order-up-to to absorb size variability; intermittent demand makes continuous review
wasteful, so review roughly every demand interval; lumpy demand (sporadic *and* erratic
— the hardest quadrant) gets a conservative every-period base stock. These are
**starting parameters, not guaranteed optima** — the advisor says so explicitly for the
lumpy case, where high $\mathrm{CV}^2$ can drive a large target that should be tuned
against simulated cost.

---

## 11. Risk analytics

**Intuition.** A "95% service level **on average**" can still hide catastrophic
5th-percentile outcomes. Averaging across Monte Carlo replications (§12) reports the
centre of the outcome distribution; risk measures report its **tail**.

**Definitions** (`analytics/risk.py`), at confidence level $\alpha$ (default 0.95):

- **Value-at-Risk (VaR)** — a quantile of the outcome distribution: the threshold the
  outcome does not cross with probability $\alpha$.
- **Conditional VaR (CVaR / expected shortfall)** — the *mean of the tail beyond VaR*:
  the expected outcome given that a tail event occurs (Rockafellar & Uryasev 2000). CVaR
  is more conservative and better-behaved than VaR (it sees the whole tail, not just its
  edge).

**Which tail is the risky one depends on the KPI:**

- **Upper (loss) tail — cost and other "high-is-bad" quantities** (the default, e.g.
  `total_cost`): $\text{VaR} = \text{quantile}(x, \alpha)$ (e.g. the 95th-percentile
  cost), $\text{CVaR} = \mathrm{mean}(x \mid x \ge \text{VaR})$. Ordering
  $\text{mean} \le \text{VaR} \le \text{CVaR}$.
- **Lower (gain) tail — service levels and other "low-is-bad" quantities** (e.g.
  `fill_rate`): $\text{VaR} = \text{quantile}(x, 1-\alpha)$ (e.g. the 5th-percentile
  service), $\text{CVaR} = \mathrm{mean}(x \mid x \le \text{VaR})$. Ordering
  $\text{CVaR} \le \text{VaR} \le \text{mean}$.

CVaR uses the empirical tail mask, so it always selects at least one replication and is
never $\text{nan}$ (matters at $n=1$). The Monte Carlo & Risk page draws this exactly: a
VaR marker line plus CVaR as the mean of the points beyond it.

---

## 12. Experimental methods

The single-run engine is the substrate for three sweep-style analytics, each a pure
function of a configuration.

### 12.1 Monte Carlo

A single run is one demand realisation. To estimate the *distribution* of an outcome,
re-run the same configuration across many seeds (`analytics/monte_carlo.py`): each
replication varies only the `master_seed`, yielding a sample of KPIs whose spread is the
genuine outcome uncertainty. The sample mean converges to the true mean at rate
$1/\sqrt{n}$ (central limit theorem), and the full sample feeds the risk tail measures
(§11). The performance target is 1,000 replications in well under 30 seconds.

### 12.2 The Pareto frontier

Service and cost trade off: you can almost always buy more service by holding more
stock. Sweeping a grid of policy parameters and plotting (cost, service) for each yields
a cloud of options; the **Pareto-efficient frontier** (`analytics/pareto.py`) is the
subset that is **non-dominated** — no other option is cheaper *and* better-serving. Any
point inside the frontier is strictly improvable; the frontier itself is the menu of
rational choices, and where on it to sit is a business decision about the value of
service.

### 12.3 Sensitivity analysis

**One-at-a-time (OAT)** sensitivity (`analytics/sensitivity.py`) perturbs each input by
$\pm$ a percentage while holding the others fixed, and measures the swing in an output
KPI. Ranking the inputs by swing magnitude produces a **tornado chart** (widest bar on
top) that shows which assumptions the result actually depends on — focusing attention on
the few parameters worth pinning down precisely.

> **Stress scenarios are lead-time / pattern-expressible only.** The named stress presets
> (`data/scenarios/`) cover what the static-config model can faithfully express — a
> lead-time spike, an effective supplier outage (large lead-time multiplier), and a
> seasonal demand shock (a pattern). The **demand-windowed** scenarios (demand surge,
> collapse, recovery) require a time-windowed *demand-disruption overlay* — the demand
> twin of the lead-time disruption of §4 — which is deferred to future invasive work, not
> faked with patterns.

---

## 13. From classical control to reinforcement learning

The classical policies of §5 are **fixed rules**. A Phase 2 *research track* reframes the same
system as a sequential decision problem and *learns* a policy. (The headline Phase 2 direction
is a diagnostic layer over this same engine; reinforcement learning is the parallel research
extension, not the product surface.) The bridge already exists in Phase 1:
`core/environment.py::InventoryEnv` wraps the engine in a Gymnasium environment.

The inventory problem is a **Markov Decision Process**:

- **State** — the observation the policy sees each period: on-hand, on-order, backorders,
  and inventory position (Observation Point A of §2).
- **Action** — the order quantity $q_t \ge 0$ (a one-dimensional continuous action).
- **Reward** — the negative of the period's total cost, $-\text{cost}_t$, so maximising
  cumulative reward minimises total cost.
- **Transition** — the engine's five-step period (receive, observe, fulfil, order,
  accrue), with stochasticity from the demand and lead-time draws.

The four classical policies are the **baselines** an RL agent must beat: they are
provably good under their assumptions (§6), so a learned policy earns its complexity only
by exploiting structure they ignore (non-stationarity, multi-echelon coupling, capacity).
A canonical random policy sampled from the action space already drives the environment
for 100 steps without error — the Phase-2 readiness gate — proving the hook is real, not
aspirational. The full RL theory (value functions, policy gradients, PPO) is Phase-2
scope; see Sutton & Barto (2018).

---

## 14. References

1. Harris, F. W. (1913). "How Many Parts to Make at Once." *Factory, The Magazine of Management*, 10(2), 135–136. *(EOQ.)*
2. Wilson, R. H. (1934). "A Scientific Routine for Stock Control." *Harvard Business Review*, 13, 116–128.
3. Arrow, K. J., Harris, T., & Marschak, J. (1951). "Optimal Inventory Policy." *Econometrica*, 19(3), 250–272.
4. Scarf, H. (1960). "The Optimality of $(s,S)$ Policies in the Dynamic Inventory Problem." In *Mathematical Methods in the Social Sciences*. Stanford University Press.
5. Croston, J. D. (1972). "Forecasting and Stock Control for Intermittent Demands." *Operational Research Quarterly*, 23(3), 289–303.
6. Lee, H. L., Padmanabhan, V., & Whang, S. (1997). "Information Distortion in a Supply Chain: The Bullwhip Effect." *Management Science*, 43(4), 546–558.
7. Zipkin, P. H. (2000). *Foundations of Inventory Management*. McGraw-Hill.
8. Chen, F., Drezner, Z., Ryan, J. K., & Simchi-Levi, D. (2000). "Quantifying the Bullwhip Effect in a Simple Supply Chain." *Management Science*, 46(3), 436–443.
9. Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21–41.
10. Syntetos, A. A., & Boylan, J. E. (2005). "The Accuracy of Intermittent Demand Estimates." *International Journal of Forecasting*, 21(2), 303–314.
11. Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). "On the Categorization of Demand Patterns." *Journal of the Operational Research Society*, 56(5), 495–503. *(The ADI = 1.32 / $\mathrm{CV}^2$ = 0.49 cutoffs.)*
12. Nahmias, S. (2009). *Production and Operations Analysis*, 6th ed. McGraw-Hill.
13. Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). *Inventory and Production Management in Supply Chains*, 4th ed. CRC Press.
14. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. *(Phase-2 bridge.)*
