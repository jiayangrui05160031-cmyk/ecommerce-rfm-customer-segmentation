"""Deterministic marketing experiment analysis.

The module deliberately has no scipy dependency: a two-sided pooled z-test
can be computed with Python's standard library, which keeps the API small and
easy to deploy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt


@dataclass(frozen=True)
class ExperimentResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float | None
    confidence_interval_95: tuple[float, float]
    z_score: float
    p_value: float
    statistically_significant: bool
    expected_incremental_conversions: float
    expected_incremental_revenue: float
    expected_campaign_cost: float
    expected_incremental_profit: float
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_experiment(
    *,
    control_visitors: int,
    control_conversions: int,
    treatment_visitors: int,
    treatment_conversions: int,
    target_audience: int = 0,
    value_per_conversion: float = 0.0,
    cost_per_contact: float = 0.0,
    alpha: float = 0.05,
) -> ExperimentResult:
    """Analyze a two-arm conversion experiment and project its economics."""
    _validate_arm("control", control_visitors, control_conversions)
    _validate_arm("treatment", treatment_visitors, treatment_conversions)
    if target_audience < 0:
        raise ValueError("target_audience must be non-negative")
    if value_per_conversion < 0 or cost_per_contact < 0:
        raise ValueError("unit economics must be non-negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    p_control = control_conversions / control_visitors
    p_treatment = treatment_conversions / treatment_visitors
    lift = p_treatment - p_control
    relative_lift = lift / p_control if p_control else None

    # Pooled standard error is used for the null-hypothesis z statistic.
    pooled = (
        (control_conversions + treatment_conversions)
        / (control_visitors + treatment_visitors)
    )
    null_se = sqrt(
        pooled * (1 - pooled)
        * (1 / control_visitors + 1 / treatment_visitors)
    )
    z_score = lift / null_se if null_se else 0.0
    p_value = erfc(abs(z_score) / sqrt(2)) if null_se else 1.0

    # The interval estimates each arm separately instead of imposing H0.
    estimate_se = sqrt(
        p_control * (1 - p_control) / control_visitors
        + p_treatment * (1 - p_treatment) / treatment_visitors
    )
    ci = (lift - 1.96 * estimate_se, lift + 1.96 * estimate_se)
    significant = p_value < alpha

    incremental_conversions = lift * target_audience
    incremental_revenue = incremental_conversions * value_per_conversion
    campaign_cost = target_audience * cost_per_contact
    incremental_profit = incremental_revenue - campaign_cost
    if significant and lift > 0 and incremental_profit > 0:
        recommendation = "scale"
    elif significant and lift > 0:
        recommendation = "do_not_scale"
    elif significant and lift < 0:
        recommendation = "stop"
    else:
        recommendation = "continue_testing"

    return ExperimentResult(
        control_rate=round(p_control, 6),
        treatment_rate=round(p_treatment, 6),
        absolute_lift=round(lift, 6),
        relative_lift=round(relative_lift, 6) if relative_lift is not None else None,
        confidence_interval_95=(round(ci[0], 6), round(ci[1], 6)),
        z_score=round(z_score, 4),
        p_value=round(p_value, 6),
        statistically_significant=significant,
        expected_incremental_conversions=round(incremental_conversions, 2),
        expected_incremental_revenue=round(incremental_revenue, 2),
        expected_campaign_cost=round(campaign_cost, 2),
        expected_incremental_profit=round(incremental_profit, 2),
        recommendation=recommendation,
    )


def _validate_arm(name: str, visitors: int, conversions: int) -> None:
    if visitors <= 0:
        raise ValueError(f"{name}_visitors must be positive")
    if conversions < 0 or conversions > visitors:
        raise ValueError(
            f"{name}_conversions must be between 0 and {name}_visitors"
        )
