"""Integration: the config-builder sidebar component.

Tested with Streamlit's ``AppTest.from_function`` — a tiny host script calls
``build_config_sidebar`` and stashes the result in ``session_state`` so the test can
read it back, set widgets by their ``key``, and re-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from core.config import (
    CostConfig,
    DeterministicLeadTimeConfig,
    NoDisruptionConfig,
    NormalDemandConfig,
    RunConfig,
    SimulationConfig,
    SQPolicyConfig,
    StationaryPatternConfig,
)
from ui.components.config_builder import POLICY_HANDOFF_KEY


def _host() -> None:
    # AppTest.from_function execs this body as a standalone script, so imports must live
    # inside it (the test module's module-level imports are not in scope here).
    import streamlit as st

    from ui.components.config_builder import build_config_sidebar

    # Reset first so an invalid (halting) build leaves None rather than a stale value.
    st.session_state["built"] = None
    st.session_state["built"] = build_config_sidebar()


def _host_all_policies() -> None:
    # Render every policy kind's editor with ONE shared key_prefix: the abbrev infixes keep the
    # widget keys distinct, which is exactly the multi-policy-page scenario.
    import streamlit as st

    from ui.components.config_builder import build_policy_param_inputs

    st.session_state["built"] = {
        kind: build_policy_param_inputs(kind, key_prefix="t")
        for kind in ["sQ", "sS", "base_stock", "RS"]
    }


def _host_no_policy() -> None:
    import streamlit as st

    from ui.components.config_builder import build_config_sidebar

    st.session_state["built"] = None
    st.session_state["built"] = build_config_sidebar(include_policy=False)


def _host_with_handoff() -> None:
    # Simulate the Pareto Explorer's click-to-load: stash a policy handoff once (before the very
    # first sidebar render), then build. The sidebar should consume it and seed the widgets.
    import streamlit as st

    from ui.components.config_builder import POLICY_HANDOFF_KEY, build_config_sidebar

    if "handoff_seeded" not in st.session_state:
        st.session_state[POLICY_HANDOFF_KEY] = {
            "kind": "sS",
            "reorder_point": 15.0,
            "order_up_to": 80.0,
        }
        st.session_state["handoff_seeded"] = True
    st.session_state["built"] = build_config_sidebar()


def _expected_baseline() -> RunConfig:
    """The config the untouched sidebar should produce (the example.yaml m1 baseline)."""
    return RunConfig(
        simulation=SimulationConfig(
            horizon=90, initial_on_hand=100, initial_on_order=0, backorder_policy="backorder"
        ),
        demand=NormalDemandConfig(mean=10.0, std=2.0),
        pattern=StationaryPatternConfig(),
        lead_time=DeterministicLeadTimeConfig(lead_time=3),
        disruption=NoDisruptionConfig(),
        policy=SQPolicyConfig(reorder_point=20.0, order_quantity=30),
        costs=CostConfig(
            unit_cost=5.0,
            holding_per_unit_per_period=0.5,
            ordering_fixed=20.0,
            backorder_per_unit_per_period=2.0,
            lost_sale_per_unit=0.0,
            expediting_per_unit=0.0,
        ),
        master_seed=42,
    )


def _widget(at: AppTest, widget_type: str, key: str) -> Any:
    for w in getattr(at, widget_type):
        if w.key == key:
            return w
    raise KeyError(f"no {widget_type} widget with key {key!r}")


def test_default_build_matches_example_baseline() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    assert not at.exception
    assert at.session_state["built"] == _expected_baseline()


def test_default_build_is_deterministic() -> None:
    a = AppTest.from_function(_host, default_timeout=30).run()
    b = AppTest.from_function(_host, default_timeout=30).run()
    assert a.session_state["built"].config_hash() == b.session_state["built"].config_hash()


def test_switch_policy_to_ss() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_policy_kind").set_value("sS")
    at.run()
    policy = at.session_state["built"].policy
    assert policy.kind == "sS"
    assert policy.reorder_point == 20.0
    assert policy.order_up_to == 60.0


def test_switch_demand_to_poisson() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_demand_kind").set_value("poisson")
    at.run()
    demand = at.session_state["built"].demand
    assert demand.kind == "poisson"
    assert demand.rate == 10.0


def test_switch_pattern_to_seasonal() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_pattern_kind").set_value("seasonal")
    at.run()
    pattern = at.session_state["built"].pattern
    assert pattern.kind == "seasonal"
    assert pattern.period == 12


def test_switch_pattern_to_intermittent() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_pattern_kind").set_value("intermittent")
    at.run()
    pattern = at.session_state["built"].pattern
    assert pattern.kind == "intermittent"
    assert pattern.occurrence_probability == 0.6


def test_switch_pattern_to_lumpy() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_pattern_kind").set_value("lumpy")
    at.run()
    pattern = at.session_state["built"].pattern
    assert pattern.kind == "lumpy"
    assert pattern.occurrence_probability == 0.3
    assert pattern.burst_multiplier == 5.0


def test_switch_lead_time_to_gamma() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_lt_kind").set_value("gamma")
    at.run()
    lead_time = at.session_state["built"].lead_time
    assert lead_time.kind == "gamma"
    assert lead_time.shape == 9.0


def test_invalid_ss_policy_halts_with_error() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_policy_kind").set_value("sS")
    at.run()
    # s >= S violates the (s,S) cross-field invariant (default S = 60).
    _widget(at, "number_input", "cfg_pol_ss_s").set_value(70.0)
    at.run()
    assert not at.exception  # st.stop() is clean control flow, not an exception
    assert len(at.error) >= 1  # st.error was rendered
    assert at.session_state["built"] is None  # build_config_sidebar halted before returning


# --- build_policy_param_inputs (the reusable per-kind editor) ----------------------


def test_policy_param_inputs_returns_dict_per_kind() -> None:
    at = AppTest.from_function(_host_all_policies, default_timeout=30).run()
    assert not at.exception
    built = at.session_state["built"]
    assert built["sQ"] == {"kind": "sQ", "reorder_point": 20.0, "order_quantity": 30}
    assert built["sS"] == {"kind": "sS", "reorder_point": 20.0, "order_up_to": 60.0}
    assert built["base_stock"] == {"kind": "base_stock", "target_level": 50.0}
    assert built["RS"] == {"kind": "RS", "review_period": 7, "order_up_to": 60.0}


def test_policy_param_inputs_keys_namespaced_no_clash() -> None:
    # Four editors share key_prefix="t" yet render without a DuplicateWidgetID, because the
    # per-kind abbrev infixes make all seven widget keys distinct.
    at = AppTest.from_function(_host_all_policies, default_timeout=30).run()
    assert not at.exception
    keys = {w.key for w in at.number_input}
    expected = {"t_sq_s", "t_sq_q", "t_ss_s", "t_ss_S", "t_bs_S", "t_rs_R", "t_rs_S"}
    assert expected <= keys


# --- build_config_sidebar(include_policy=False) -----------------------------------


def test_include_policy_false_omits_section_but_returns_valid_config() -> None:
    at = AppTest.from_function(_host_no_policy, default_timeout=30).run()
    assert not at.exception
    # The Policy section is not rendered at all...
    assert all(w.key != "cfg_policy_kind" for w in at.selectbox)
    assert all(not (w.key or "").startswith("cfg_pol_") for w in at.number_input)
    # ...yet a valid RunConfig comes back, carrying the injected placeholder policy.
    built = at.session_state["built"]
    assert isinstance(built, RunConfig)
    assert built.policy.kind == "sQ"
    assert built.policy.reorder_point == 20.0
    assert built.policy.order_quantity == 30
    # the rest of the sidebar still builds normally.
    assert built.demand.kind == "normal"
    assert built.master_seed == 42


# --- consume_policy_handoff (the Pareto-Explorer click-to-load landing) ------------


def test_policy_handoff_seeds_widgets_and_pops() -> None:
    at = AppTest.from_function(_host_with_handoff, default_timeout=30).run()
    assert not at.exception
    policy = at.session_state["built"].policy
    assert policy.kind == "sS"  # the kind selectbox was seeded
    assert policy.reorder_point == 15.0  # cfg_pol_ss_s seeded
    assert policy.order_up_to == 80.0  # cfg_pol_ss_S seeded
    assert POLICY_HANDOFF_KEY not in at.session_state  # consumed (load-once)


def test_policy_handoff_is_load_once() -> None:
    at = AppTest.from_function(_host_with_handoff, default_timeout=30).run()
    at.run()  # a later rerun: the handoff is gone, but the seeded widget values persist
    assert not at.exception
    policy = at.session_state["built"].policy
    assert policy.kind == "sS"
    assert policy.reorder_point == 15.0
    assert policy.order_up_to == 80.0


# --- empirical demand: M5 archetype picker + custom path ---------------------------


def test_empirical_archetype_default() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_demand_kind").set_value("empirical")
    at.run()
    demand = at.session_state["built"].demand
    assert demand.kind == "empirical"
    # the archetype arm is the default source; first archetype (manifest order) is "smooth"
    assert demand.history_path == Path("data/m5/archetype_smooth.parquet")


def test_empirical_archetype_selection() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_demand_kind").set_value("empirical")
    at.run()
    _widget(at, "selectbox", "cfg_dem_emp_arch").set_value("seasonal")
    at.run()
    demand = at.session_state["built"].demand
    assert demand.history_path == Path("data/m5/archetype_seasonal.parquet")


def test_empirical_custom_path_preserved() -> None:
    at = AppTest.from_function(_host, default_timeout=30).run()
    _widget(at, "selectbox", "cfg_demand_kind").set_value("empirical")
    at.run()
    _widget(at, "radio", "cfg_dem_emp_source").set_value("Custom path")
    at.run()
    # the legacy custom-path text box still drives history_path (no capability lost)
    custom = _widget(at, "text_input", "cfg_dem_emp_path")
    custom.set_value("data/m5/archetype_intermittent.parquet")
    at.run()
    demand = at.session_state["built"].demand
    assert demand.history_path == Path("data/m5/archetype_intermittent.parquet")
