"""Sidebar config builder — Streamlit controls → a validated ``RunConfig``.

The keystone shared component: Pages 1–4 all need "turn sidebar widgets into a
``RunConfig``." :func:`build_config_sidebar` is an orchestrator — it calls one private
``_build_*_dict`` per config section, assembles a plain dict keyed by each discriminated
union's ``kind``, and hands it to ``RunConfig.model_validate`` (the **same path as**
``RunConfig.from_yaml``, which is ``model_validate(yaml.safe_load(...))``). Union
resolution and every field / cross-field invariant are therefore validated centrally by
Pydantic — the widgets only collect values. On a validation error (the one user-reachable
case is an (s,S) policy with ``reorder_point >= order_up_to``) it shows ``st.error`` and
halts the rerun via ``st.stop`` rather than returning an invalid config.

Widget defaults mirror ``data/scenarios/example.yaml`` (the m1 baseline), so an untouched
sidebar is a known-good configuration. The disruption overlay is fixed to ``none`` here;
the Stress Test page (a later bullet) owns the richer timeline composer. Widget ``key=``
values are namespaced ``cfg_*`` so the ``AppTest`` harness can address them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import streamlit as st
from pydantic import ValidationError

from core.config import RunConfig
from ui.components.archetypes import load_archetypes

_POLICY_KINDS: Final = ["sQ", "sS", "base_stock", "RS"]
# Short, stable per-kind key infixes so build_policy_param_inputs can render once per policy on a
# multi-policy page without widget-key clashes; with key_prefix="cfg_pol" it reproduces Page 1's
# historical policy keys exactly.
_POLICY_ABBREV: Final = {"sQ": "sq", "sS": "ss", "base_stock": "bs", "RS": "rs"}
# A fixed valid policy injected when the Policy section is suppressed (include_policy=False):
# RunConfig requires a policy, but such callers override it per racer, so it is never run.
_PLACEHOLDER_POLICY: Final = {"kind": "sQ", "reorder_point": 20.0, "order_quantity": 30}

# session_state key for a Pareto-Explorer click-to-load handoff: Page 3 stashes a clicked
# point's full policy dict here, then navigates to Single Run, which consumes it (see
# consume_policy_handoff). Public — it is the cross-page contract.
POLICY_HANDOFF_KEY: Final = "cfg_policy_handoff"
# Policy field name -> the key infix build_policy_param_inputs uses for that field's widget
# (the inverse of the per-kind suffixes there; order_up_to/target_level share "S" but never
# coexist in one kind). consume_policy_handoff seeds f"cfg_pol_{abbrev}_{suffix}" from this.
_FIELD_SUFFIX: Final = {
    "reorder_point": "s",
    "order_quantity": "q",
    "order_up_to": "S",
    "target_level": "S",
    "review_period": "R",
}


def _build_simulation_dict() -> dict[str, Any]:
    st.subheader("Simulation")
    return {
        "horizon": int(
            st.number_input("Horizon (periods)", min_value=1, value=90, step=1, key="cfg_horizon")
        ),
        "initial_on_hand": int(
            st.number_input("Initial on-hand", min_value=0, value=100, step=1, key="cfg_init_oh")
        ),
        "initial_on_order": int(
            st.number_input("Initial on-order", min_value=0, value=0, step=1, key="cfg_init_oo")
        ),
        "backorder_policy": st.selectbox(
            "Unmet demand", ["backorder", "lost_sales"], index=0, key="cfg_backorder_policy"
        ),
    }


def _build_demand_dict() -> dict[str, Any]:
    st.subheader("Demand")
    kind = st.selectbox(
        "Distribution",
        ["normal", "poisson", "negative_binomial", "gamma", "lognormal", "empirical"],
        index=0,
        key="cfg_demand_kind",
    )
    d: dict[str, Any] = {"kind": kind}
    if kind == "normal":
        d["mean"] = float(st.number_input("Mean", min_value=0.0, value=10.0, key="cfg_dem_n_mean"))
        d["std"] = float(st.number_input("Std dev", min_value=0.0, value=2.0, key="cfg_dem_n_std"))
    elif kind == "poisson":
        d["rate"] = float(
            st.number_input("Rate (λ)", min_value=0.01, value=10.0, key="cfg_dem_p_rate")
        )
    elif kind == "negative_binomial":
        d["n"] = float(
            st.number_input("n (dispersion)", min_value=0.01, value=10.0, key="cfg_dem_nb_n")
        )
        d["p"] = float(
            st.number_input(
                "p (success prob)", min_value=0.01, max_value=0.99, value=0.5, key="cfg_dem_nb_p"
            )
        )
    elif kind == "gamma":
        d["shape"] = float(
            st.number_input("Shape (k)", min_value=0.01, value=5.0, key="cfg_dem_g_shape")
        )
        d["scale"] = float(
            st.number_input("Scale (θ)", min_value=0.01, value=2.0, key="cfg_dem_g_scale")
        )
    elif kind == "lognormal":
        d["mu"] = float(
            st.number_input("mu (underlying Normal mean)", value=2.0, key="cfg_dem_ln_mu")
        )
        d["sigma"] = float(
            st.number_input(
                "sigma (underlying Normal std)", min_value=0.01, value=0.5, key="cfg_dem_ln_sigma"
            )
        )
    else:  # empirical — replay a stored history IID (an M5 archetype or a custom parquet)
        if (
            st.radio(
                "History source",
                ["M5 archetype", "Custom path"],
                horizontal=True,
                key="cfg_dem_emp_source",
            )
            == "M5 archetype"
        ):
            archetypes = load_archetypes()
            arch = archetypes[st.selectbox("Archetype", list(archetypes), key="cfg_dem_emp_arch")]
            d["history_path"] = arch.history_path
            # Surface the business label vs the data-driven SB class — they need not match (a
            # teaching point: "seasonal" is SB "erratic", "promotional" is SB "lumpy"), not a bug.
            st.caption(
                f"**{arch.name.title()}** archetype · SB quadrant **{arch.sb_quadrant}** "
                f"(business label ≠ statistical class) · {arch.item_id} · CV²={arch.cv2:.2f} · "
                f"ADI={arch.adi:.2f} · zero-fraction={arch.zero_fraction:.0%}"
            )
        else:
            d["history_path"] = st.text_input(
                "History parquet path",
                value="data/m5/sample_smooth.parquet",
                key="cfg_dem_emp_path",
            )
    return d


def _build_pattern_dict() -> dict[str, Any]:
    st.subheader("Demand pattern")
    kind = st.selectbox(
        "Overlay",
        ["stationary", "seasonal", "trending", "intermittent", "lumpy"],
        index=0,
        key="cfg_pattern_kind",
    )
    d: dict[str, Any] = {"kind": kind}
    if kind == "seasonal":
        d["amplitude"] = float(
            st.number_input(
                "Amplitude (0–1)", min_value=0.0, max_value=0.99, value=0.3, key="cfg_pat_amp"
            )
        )
        d["period"] = int(
            st.number_input("Period", min_value=1, value=12, step=1, key="cfg_pat_period")
        )
        d["phase"] = float(st.number_input("Phase (radians)", value=0.0, key="cfg_pat_phase"))
    elif kind == "trending":
        d["slope"] = float(
            st.number_input(
                "Slope (per-period frac.)", value=0.01, format="%.4f", key="cfg_pat_slope"
            )
        )
    elif kind == "intermittent":
        d["occurrence_probability"] = float(
            st.number_input(
                "Occurrence probability (0–1)",
                min_value=0.01,
                max_value=1.0,
                value=0.6,
                step=0.05,
                format="%.2f",
                key="cfg_pat_occ",
            )
        )
    elif kind == "lumpy":
        d["occurrence_probability"] = float(
            st.number_input(
                "Occurrence probability (0–1)",
                min_value=0.01,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                help="How often burst periods occur.",
                key="cfg_pat_lumpy_occ",
            )
        )
        d["burst_multiplier"] = float(
            st.number_input(
                "Burst multiplier (>1)",
                min_value=1.01,
                value=5.0,
                step=0.5,
                format="%.2f",
                help="How much larger demand is during burst periods.",
                key="cfg_pat_lumpy_burst",
            )
        )
    return d


def _build_lead_time_dict() -> dict[str, Any]:
    st.subheader("Lead time")
    kind = st.selectbox(
        "Distribution",
        ["deterministic", "normal", "gamma", "lognormal"],
        index=0,
        key="cfg_lt_kind",
    )
    d: dict[str, Any] = {"kind": kind}
    if kind == "deterministic":
        d["lead_time"] = int(
            st.number_input("Lead time (periods)", min_value=1, value=3, step=1, key="cfg_lt_det")
        )
    elif kind == "normal":
        d["mean"] = float(st.number_input("Mean", min_value=0.01, value=3.0, key="cfg_lt_n_mean"))
        d["std"] = float(st.number_input("Std dev", min_value=0.0, value=1.0, key="cfg_lt_n_std"))
    elif kind == "gamma":
        d["shape"] = float(
            st.number_input("Shape", min_value=0.01, value=9.0, key="cfg_lt_g_shape")
        )
        d["scale"] = float(
            st.number_input("Scale", min_value=0.01, value=0.3333, key="cfg_lt_g_scale")
        )
    else:  # lognormal
        d["mu"] = float(st.number_input("mu", value=1.0, key="cfg_lt_ln_mu"))
        d["sigma"] = float(
            st.number_input("sigma", min_value=0.01, value=0.3, key="cfg_lt_ln_sigma")
        )
    return d


def _build_costs_dict() -> dict[str, Any]:
    st.subheader("Costs")
    d: dict[str, Any] = {
        "unit_cost": float(
            st.number_input("Unit cost", min_value=0.0, value=5.0, key="cfg_cost_unit")
        ),
        "holding_per_unit_per_period": float(
            st.number_input("Holding /unit/period", min_value=0.0, value=0.5, key="cfg_cost_hold")
        ),
        "ordering_fixed": float(
            st.number_input("Ordering fixed (K)", min_value=0.0, value=20.0, key="cfg_cost_order")
        ),
    }
    with st.expander("Advanced costs"):
        d["backorder_per_unit_per_period"] = float(
            st.number_input("Backorder /unit/period", min_value=0.0, value=2.0, key="cfg_cost_back")
        )
        d["lost_sale_per_unit"] = float(
            st.number_input("Lost-sale /unit", min_value=0.0, value=0.0, key="cfg_cost_lost")
        )
        d["expediting_per_unit"] = float(
            st.number_input("Expediting /unit", min_value=0.0, value=0.0, key="cfg_cost_exped")
        )
    return d


def build_policy_param_inputs(kind: str, *, key_prefix: str) -> dict[str, Any]:
    """Render the parameter widgets for ONE policy ``kind`` and return its config dict.

    The returned dict includes the ``"kind"`` discriminator, so it can be fed straight to
    ``RunConfig.model_validate`` (or used as a ``policy`` override). Widget keys are
    ``f"{key_prefix}_{abbrev}_{field}"`` (abbrev per :data:`_POLICY_ABBREV`), so the same editor
    can appear once per policy on a multi-policy page without key clashes; with
    ``key_prefix="cfg_pol"`` it reproduces Page 1's historical policy keys exactly.
    """
    ab = _POLICY_ABBREV[kind]
    d: dict[str, Any] = {"kind": kind}
    if kind == "sQ":
        d["reorder_point"] = float(
            st.number_input("Reorder point (s)", value=20.0, key=f"{key_prefix}_{ab}_s")
        )
        d["order_quantity"] = int(
            st.number_input(
                "Order quantity (Q)", min_value=1, value=30, step=1, key=f"{key_prefix}_{ab}_q"
            )
        )
    elif kind == "sS":
        d["reorder_point"] = float(
            st.number_input("Reorder point (s)", value=20.0, key=f"{key_prefix}_{ab}_s")
        )
        d["order_up_to"] = float(
            st.number_input(
                "Order-up-to (S)", min_value=0.01, value=60.0, key=f"{key_prefix}_{ab}_S"
            )
        )
    elif kind == "base_stock":
        d["target_level"] = float(
            st.number_input(
                "Target level (S)", min_value=0.01, value=50.0, key=f"{key_prefix}_{ab}_S"
            )
        )
    else:  # RS
        d["review_period"] = int(
            st.number_input(
                "Review period (R)", min_value=1, value=7, step=1, key=f"{key_prefix}_{ab}_R"
            )
        )
        d["order_up_to"] = float(
            st.number_input(
                "Order-up-to (S)", min_value=0.01, value=60.0, key=f"{key_prefix}_{ab}_S"
            )
        )
    return d


def policy_label(kind: str, params: Mapping[str, Any]) -> str:
    """A self-documenting legend/table label for one policy, e.g. ``sQ(s=20, Q=30)``.

    Shared by every page that compares policy *variants* on a common world (the Policy Comparison
    race and the Bullwhip Preview) so the same policy is labelled identically across them — no
    drift. ``params`` is a policy dict as produced by :func:`build_policy_param_inputs`; ``:g``
    drops a trailing ``.0`` on the float fields.
    """
    if kind == "sQ":
        return f"sQ(s={params['reorder_point']:g}, Q={params['order_quantity']})"
    if kind == "sS":
        return f"sS(s={params['reorder_point']:g}, S={params['order_up_to']:g})"
    if kind == "base_stock":
        return f"BaseStock(S={params['target_level']:g})"
    return f"RS(R={params['review_period']}, S={params['order_up_to']:g})"


def consume_policy_handoff() -> None:
    """Apply a pending Pareto-Explorer click-to-load handoff to the policy widgets, once.

    Page 3 stashes a clicked point's full policy dict in
    ``st.session_state[POLICY_HANDOFF_KEY]`` then navigates here. This seeds the policy ``kind``
    selectbox and that kind's parameter widgets *before* they instantiate (the supported way to
    set a widget value programmatically), then pops the handoff so subsequent edits are not
    clobbered on later reruns (load-once). A no-op when no handoff is pending.
    """
    handoff = st.session_state.pop(POLICY_HANDOFF_KEY, None)
    if not handoff:
        return
    kind = handoff["kind"]
    st.session_state["cfg_policy_kind"] = kind
    ab = _POLICY_ABBREV[kind]
    for field, value in handoff.items():
        if field != "kind":
            st.session_state[f"cfg_pol_{ab}_{_FIELD_SUFFIX[field]}"] = value


def _build_policy_dict() -> dict[str, Any]:
    consume_policy_handoff()  # seed widgets from a pending Pareto click-to-load, before they render
    st.subheader("Policy")
    kind = st.selectbox("Type", _POLICY_KINDS, index=0, key="cfg_policy_kind")
    return build_policy_param_inputs(kind, key_prefix="cfg_pol")


def build_config_sidebar(*, include_policy: bool = True) -> RunConfig:
    """Render the configuration sidebar and return a validated :class:`RunConfig`.

    On a ``ValidationError`` (e.g. an (s,S) policy with ``reorder_point >= order_up_to``)
    the error is shown and the rerun is halted via ``st.stop``; the function never returns
    an invalid config.

    When ``include_policy`` is ``False`` the Policy section is not rendered and a fixed
    placeholder policy is injected (``RunConfig`` requires one); callers that build their own
    policies — e.g. the multi-policy comparison page — pass ``False`` and override ``policy``.
    """
    with st.sidebar:
        st.header("Configuration")
        config_dict: dict[str, Any] = {
            "simulation": _build_simulation_dict(),
            "demand": _build_demand_dict(),
            "pattern": _build_pattern_dict(),
            "lead_time": _build_lead_time_dict(),
            "disruption": {"kind": "none"},
            "policy": _build_policy_dict() if include_policy else _PLACEHOLDER_POLICY.copy(),
            "costs": _build_costs_dict(),
            "master_seed": int(
                st.number_input("Master seed", min_value=0, value=42, step=1, key="cfg_master_seed")
            ),
        }

    try:
        return RunConfig.model_validate(config_dict)
    except ValidationError as exc:
        st.error(f"Invalid configuration:\n\n{exc}")
        st.stop()
        raise  # unreachable: st.stop() ends the script run; re-raise satisfies the type checker
