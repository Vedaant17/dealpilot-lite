"""Sensitivity analysis for DealPilot Lite DCF valuations."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from backend.app.financial.valuation import calculate_valuation


def _clone_assumptions(assumptions: Any, **updates: Any) -> Any:
    data = assumptions.model_dump()
    data.update(updates)
    return assumptions.__class__(**data)


def run_sensitivity_analysis(
    conn: Any,
    company_id: str,
    base_period: str,
    assumptions: Any,
    ranges: Any,
    overrides: Any = None,
) -> dict[str, Any]:
    """Independently recalculate a DCF valuation for every WACC/growth pair."""
    wacc_values = [
        ranges.wacc_min + ranges.wacc_step * i
        for i in range(int((ranges.wacc_max - ranges.wacc_min) / ranges.wacc_step) + 1)
    ]
    growth_values = [
        ranges.terminal_growth_min + ranges.terminal_growth_step * i
        for i in range(
            int(
                (ranges.terminal_growth_max - ranges.terminal_growth_min)
                / ranges.terminal_growth_step
            )
            + 1
        )
    ]

    if len(wacc_values) * len(growth_values) > ranges.max_pairs:
        return {
            "available": False,
            "reasons": ["sensitivity grid exceeds max_pairs"],
            "scenarios": [],
        }

    # Apply optional overrides to a base assumption set.
    base_assumptions = assumptions
    if overrides is not None:
        override_data = overrides.model_dump(exclude_none=True)
        if "revenue_growth_rates" in override_data:
            base_assumptions = _clone_assumptions(
                base_assumptions, revenue_growth_rates=override_data.pop("revenue_growth_rates")
            )
        if override_data:
            base_assumptions = _clone_assumptions(base_assumptions, **override_data)

    base_result = calculate_valuation(conn, company_id, base_period, base_assumptions)

    scenarios = []
    for wacc in wacc_values:
        for growth in growth_values:
            scenario_assumptions = _clone_assumptions(
                base_assumptions,
                direct_wacc=wacc,
                terminal_growth=growth,
            )
            result = calculate_valuation(conn, company_id, base_period, scenario_assumptions)
            scenarios.append(
                {
                    "wacc": str(wacc),
                    "terminal_growth": str(growth),
                    "available": result.get("available", False),
                    "enterprise_value": result.get("enterprise_value"),
                    "equity_value": result.get("equity_value"),
                    "per_share_value": result.get("per_share_value"),
                    "currency": result.get("currency"),
                    "reasons": result.get("reasons", []),
                }
            )

    # Compare the independently calculated base case with its corresponding grid point.
    base_wacc = base_result.get("wacc", {}).get("value")
    base_growth = str(base_assumptions.terminal_growth)
    matching = None
    if base_wacc is not None:
        for scenario in scenarios:
            if scenario["wacc"] == base_wacc and scenario["terminal_growth"] == base_growth:
                matching = scenario
                break

    comparison = {
        "base_enterprise_value": base_result.get("enterprise_value"),
        "matched_grid_enterprise_value": matching.get("enterprise_value") if matching else None,
        "difference": None,
    }
    if (
        matching is not None
        and base_result.get("enterprise_value") is not None
        and matching.get("enterprise_value") is not None
    ):
        comparison["difference"] = str(
            Decimal(base_result["enterprise_value"]) - Decimal(matching["enterprise_value"])
        )

    return {
        "available": base_result.get("available", False),
        "company_id": company_id,
        "currency": base_result.get("currency"),
        "base_case": base_result,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "base_case_grid_comparison": comparison,
        "reasons": base_result.get("reasons", []),
    }
