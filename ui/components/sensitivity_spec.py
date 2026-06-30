"""Streamlit-free helpers for the Sensitivity page's parameter picker.

Pure functions over a :class:`~core.config.RunConfig` — the perturbable-parameter catalogue and the
symmetric low/high bounds — factored out of ``ui/pages/9_Sensitivity.py`` so they are unit-testable
apart from the page (the :mod:`ui.components.demand_input` / :mod:`ui.components.pareto_select`
precedent).

:func:`numeric_paths` is the *loud* exclusion. Only the four **business-input** sections are walked,
and within them only non-bool ``int`` / ``float`` fields with a **non-zero** baseline qualify. So
``kind`` (the str discriminator), ``master_seed``, the structural ``simulation`` fields (horizon /
initial_* / backorder_policy), the ``disruption`` windows, and inert zero costs (expediting /
lost-sale in Phase 1) are all excluded *by construction* — none can become a fake "parameter" with
no real sensitivity. ``pattern`` is deliberately deferred until each arm's ±% behaviour is tested.
"""

from __future__ import annotations

from core.config import RunConfig

# The perturbable OAT target sections. NOT simulation / master_seed / disruption (structural — they
# change the run itself, not a business input); NOT pattern (deferred until each arm is validated).
_PERTURB_SECTIONS: tuple[str, ...] = ("demand", "lead_time", "costs", "policy")


def numeric_paths(config: RunConfig) -> dict[str, float | int]:
    """``{"section.field": baseline}`` for every perturbable numeric field of ``config``.

    A field qualifies only if it is a non-bool ``int`` / ``float`` with a non-zero baseline (a ±%
    perturbation of zero has no effect — it would be a fake parameter). String discriminators and
    the structural sections are excluded simply by walking only :data:`_PERTURB_SECTIONS`.
    """
    catalogue: dict[str, float | int] = {}
    for section in _PERTURB_SECTIONS:
        sub = getattr(config, section)
        for field, value in sub.model_dump().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value == 0:
                continue
            catalogue[f"{section}.{field}"] = value
    return catalogue


def low_high(baseline: float | int, frac: float) -> tuple[float | int, float | int]:
    """Symmetric ``±frac`` bounds around ``baseline`` (the equal-relative-perturbation tornado).

    Integer baselines — all positive in the perturbable sections (``lead_time`` / ``order_quantity``
    / ``review_period``) — round to whole units and clamp to ``>= 1`` so a perturbed value stays a
    valid ``PositiveInt``; floats pass through unrounded.
    """
    lo = baseline * (1 - frac)
    hi = baseline * (1 + frac)
    if isinstance(baseline, int):
        return max(1, round(lo)), max(1, round(hi))
    return lo, hi
