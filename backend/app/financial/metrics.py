"""Deterministic financial metrics."""

from decimal import localcontext
from typing import Any

from backend.app.financial.decimal_utils import CalculationResult, to_decimal

CALCULATION_PRECISION = 28


def revenue_growth(current: Any, prior: Any) -> CalculationResult:
    """Calculate year-over-year revenue growth.

    Returns unavailable result if prior revenue is missing, zero, or negative,
    or if current revenue is negative.
    """
    c = to_decimal(current)
    p = to_decimal(prior)
    if c is None or p is None:
        return CalculationResult(reason="missing revenue input")
    if p <= 0:
        return CalculationResult(reason="prior revenue must be positive")
    if c < 0:
        return CalculationResult(reason="current revenue must be non-negative")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=(c - p) / p)


def ebitda(ebit: Any, da: Any) -> CalculationResult:
    """EBITDA = EBIT + D&A. Requires both inputs."""
    e = to_decimal(ebit)
    d = to_decimal(da)
    if e is None or d is None:
        return CalculationResult(reason="missing ebit or depreciation/amortization input")
    if d < 0:
        return CalculationResult(reason="depreciation and amortization must be non-negative")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=e + d)


def ebitda_margin(ebitda_val: Any, revenue: Any) -> CalculationResult:
    """EBITDA Margin = EBITDA / Revenue. Withheld for zero/negative/missing revenue."""
    e = to_decimal(ebitda_val)
    r = to_decimal(revenue)
    if e is None or r is None:
        return CalculationResult(reason="missing ebitda or revenue input")
    if r <= 0:
        return CalculationResult(reason="revenue must be positive")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=e / r)


def ebit_margin(ebit: Any, revenue: Any) -> CalculationResult:
    """EBIT Margin = EBIT / Revenue. Withheld for zero/negative/missing revenue."""
    e = to_decimal(ebit)
    r = to_decimal(revenue)
    if e is None or r is None:
        return CalculationResult(reason="missing ebit or revenue input")
    if r <= 0:
        return CalculationResult(reason="revenue must be positive")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=e / r)


def net_profit_margin(net_income: Any, revenue: Any) -> CalculationResult:
    """Net Profit Margin = Net Income / Revenue. Withheld for zero/negative/missing revenue."""
    n = to_decimal(net_income)
    r = to_decimal(revenue)
    if n is None or r is None:
        return CalculationResult(reason="missing net income or revenue input")
    if r <= 0:
        return CalculationResult(reason="revenue must be positive")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=n / r)


def debt_to_equity(total_debt: Any, equity: Any) -> CalculationResult:
    """Debt/Equity = Total Debt / Shareholders' Equity.

    Withheld if debt or equity is missing, equity is zero or negative,
    or debt is negative.
    """
    d = to_decimal(total_debt)
    e = to_decimal(equity)
    if d is None or e is None:
        return CalculationResult(reason="missing debt or equity input")
    if e <= 0:
        return CalculationResult(reason="equity must be positive")
    if d < 0:
        return CalculationResult(reason="debt must be non-negative")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=d / e)


def interest_coverage(ebit: Any, interest_expense: Any) -> CalculationResult:
    """Interest Coverage = EBIT / Interest Expense.

    Returns unavailable if EBIT or interest expense is missing.
    If interest expense is zero, returns unavailable (no debt service) rather than infinity.
    Negative interest expense is invalid.
    """
    e = to_decimal(ebit)
    i = to_decimal(interest_expense)
    if e is None:
        return CalculationResult(reason="missing ebit input")
    if i is None:
        return CalculationResult(reason="missing interest expense input")
    if i < 0:
        return CalculationResult(reason="interest expense must be non-negative")
    if i == 0:
        return CalculationResult(reason="interest expense is zero; no debt service")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=e / i)


def current_ratio(current_assets: Any, current_liabilities: Any) -> CalculationResult:
    """Current Ratio = Current Assets / Current Liabilities. Withheld for zero/missing liabilities."""
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    if ca is None or cl is None:
        return CalculationResult(reason="missing current assets or current liabilities input")
    if cl <= 0:
        return CalculationResult(reason="current liabilities must be positive")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=ca / cl)
