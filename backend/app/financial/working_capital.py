"""Operating net working capital calculations."""

from decimal import localcontext
from typing import Any

from backend.app.financial.decimal_utils import CalculationResult, to_decimal

CALCULATION_PRECISION = 28


def operating_nwc(
    accounts_receivable: Any,
    inventory: Any,
    other_operating_current_assets: Any,
    accounts_payable: Any,
    other_operating_current_liabilities: Any,
) -> CalculationResult:
    """Operating NWC = AR + Inventory + Other Op CA - AP - Other Op CL.

    Excludes cash, interest-bearing debt, financing liabilities.
    Returns unavailable result if any required component is missing.
    """
    ar = to_decimal(accounts_receivable)
    inv = to_decimal(inventory)
    oca = to_decimal(other_operating_current_assets)
    ap = to_decimal(accounts_payable)
    ocl = to_decimal(other_operating_current_liabilities)
    if ar is None:
        return CalculationResult(reason="missing accounts receivable input")
    if inv is None:
        return CalculationResult(reason="missing inventory input")
    if oca is None:
        return CalculationResult(reason="missing other operating current assets input")
    if ap is None:
        return CalculationResult(reason="missing accounts payable input")
    if ocl is None:
        return CalculationResult(reason="missing other operating current liabilities input")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=ar + inv + oca - ap - ocl)


def nwc_change(current_nwc: Any, prior_nwc: Any) -> CalculationResult:
    """Change in NWC = Current NWC - Prior NWC. Returns unavailable if either is None."""
    c = to_decimal(current_nwc)
    p = to_decimal(prior_nwc)
    if c is None or p is None:
        return CalculationResult(reason="missing current or prior NWC input")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=c - p)
