"""WACC calculation helpers for DealPilot Lite."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any, Optional

from backend.app.financial.decimal_utils import CalculationResult, to_decimal

CALCULATION_PRECISION = 28


@dataclass(frozen=True)
class CostOfDebtEstimate:
    value: Optional[Decimal]
    method: str
    reason: Optional[str] = None


def ensure_calculation_result(result: CalculationResult, context: str) -> None:
    """Reject ambiguous CalculationResult states."""
    if result.value is None and result.reason is None:
        raise ValueError(f"{context}: ambiguous CalculationResult (value and reason are both None)")
    if result.value is not None and result.reason is not None:
        raise ValueError(f"{context}: ambiguous CalculationResult (value and reason are both set)")


def estimate_cost_of_debt(
    stated_rate: Any = None,
    interest_expense: Any = None,
    debt: Any = None,
    average_debt: Any = None,
) -> CostOfDebtEstimate:
    """Select a cost of debt using stated rate, then average debt, then closing debt."""
    stated = to_decimal(stated_rate)
    if stated is not None:
        return CostOfDebtEstimate(value=stated, method="stated_borrowing_rate")

    interest = to_decimal(interest_expense)
    avg = to_decimal(average_debt)
    closing = to_decimal(debt)

    denominator = avg if avg is not None else closing
    if interest is None or denominator is None:
        return CostOfDebtEstimate(
            value=None, method="unavailable", reason="missing interest expense or debt"
        )
    if denominator <= 0:
        return CostOfDebtEstimate(
            value=None,
            method="unavailable",
            reason="debt must be positive for cost-of-debt estimation",
        )
    if interest < 0:
        return CostOfDebtEstimate(
            value=None, method="unavailable", reason="interest expense must be non-negative"
        )
    method = "average_debt" if avg is not None else "closing_debt"
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CostOfDebtEstimate(value=interest / denominator, method=method)


def calculate_component_wacc(
    equity: Any,
    debt: Any,
    cost_of_equity: Any,
    cost_of_debt: Any,
    interest_tax_shield_rate: Any,
    operating_tax_rate: Any = None,
) -> CalculationResult:
    """Calculate component WACC from book-value capital weights."""
    e = to_decimal(equity)
    d = to_decimal(debt)
    ke = to_decimal(cost_of_equity)
    kd = to_decimal(cost_of_debt)
    shield = to_decimal(interest_tax_shield_rate)
    tax_rate = to_decimal(operating_tax_rate)

    if e is None:
        return CalculationResult(reason="missing shareholders' equity")
    if d is None:
        return CalculationResult(reason="missing total debt")
    if ke is None:
        return CalculationResult(reason="missing cost of equity")
    if d != 0 and kd is None:
        return CalculationResult(reason="missing cost of debt")
    if d == 0:
        kd = Decimal(0)
    if e <= 0:
        return CalculationResult(reason="shareholders' equity must be positive")
    if d < 0:
        return CalculationResult(reason="total debt must be non-negative")
    if shield is None:
        if tax_rate is None:
            return CalculationResult(reason="missing interest tax shield rate")
        shield = tax_rate
    if shield < 0 or shield > 1:
        return CalculationResult(reason="interest tax shield rate must be between 0 and 1")

    total_capital = e + d
    if total_capital <= 0:
        return CalculationResult(reason="total capital must be positive")

    after_tax_cost_of_debt = kd * (Decimal(1) - shield)
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        wacc = (e / total_capital) * ke + (d / total_capital) * after_tax_cost_of_debt
        return CalculationResult(value=wacc)
