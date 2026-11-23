"""Free cash flow to firm (FCFF) calculations."""

from decimal import Decimal, localcontext
from typing import Any

from backend.app.financial.decimal_utils import CalculationResult, to_decimal

CALCULATION_PRECISION = 28


def operating_tax(ebit: Any, tax_rate: Any) -> CalculationResult:
    """Operating tax: EBIT * rate if EBIT > 0, else 0. No tax benefit for negative EBIT."""
    e = to_decimal(ebit)
    r = to_decimal(tax_rate)
    if e is None:
        return CalculationResult(reason="missing ebit input")
    if r is None:
        return CalculationResult(reason="missing tax rate input")
    if e <= 0:
        return CalculationResult(value=Decimal(0), reason=None)
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=e * r)


def nopat(ebit: Any, tax_rate: Any) -> CalculationResult:
    """NOPAT = EBIT - Tax (using operating tax policy)."""
    e = to_decimal(ebit)
    t_result = operating_tax(ebit, tax_rate)
    if e is None:
        return CalculationResult(reason="missing ebit input")
    if t_result.value is None and t_result.reason is not None:
        return CalculationResult(reason=t_result.reason)
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=e - t_result.value)


def fcff(
    ebit: Any,
    tax_rate: Any,
    da: Any,
    capex: Any,
    delta_nwc: Any,
) -> CalculationResult:
    """FCFF = NOPAT + D&A - CapEx - Delta NWC.

    CapEx must be a positive expenditure magnitude.
    Returns unavailable result if any required input is missing.
    Missing base-year NWC prevents dependent FCFF calculation.
    """
    e = to_decimal(ebit)
    r = to_decimal(tax_rate)
    d = to_decimal(da)
    c = to_decimal(capex)
    dn = to_decimal(delta_nwc)
    if e is None:
        return CalculationResult(reason="missing ebit input")
    if r is None:
        return CalculationResult(reason="missing tax rate input")
    if d is None:
        return CalculationResult(reason="missing depreciation and amortization input")
    if c is None:
        return CalculationResult(reason="missing capex input")
    if dn is None:
        return CalculationResult(reason="missing delta NWC input")
    if d < 0:
        return CalculationResult(reason="depreciation and amortization must be non-negative")
    if c < 0:
        return CalculationResult(
            reason="capex must be non-negative (positive expenditure magnitude)"
        )
    n = nopat(ebit, tax_rate)
    if n.value is None:
        return CalculationResult(reason=n.reason)
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=n.value + d - c - dn)
