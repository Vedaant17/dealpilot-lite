"""Deterministic financial calculation library.

All monetary calculations use Python Decimal. No floating-point arithmetic
for monetary values. Missing inputs return CalculationResult with a reason.
"""

from backend.app.financial.decimal_utils import (
    CalculationResult,
    round_display,
    safe_divide,
    safe_subtract,
    to_decimal,
    to_string,
)
from backend.app.financial.fcff import (
    fcff,
    nopat,
    operating_tax,
)
from backend.app.financial.metrics import (
    current_ratio,
    debt_to_equity,
    ebit_margin,
    ebitda,
    ebitda_margin,
    interest_coverage,
    net_profit_margin,
    revenue_growth,
)
from backend.app.financial.validation import (
    validate_currency_match,
    validate_finite,
    validate_nonnegative,
    validate_positive,
    validate_rate,
    validate_required,
)
from backend.app.financial.working_capital import (
    nwc_change,
    operating_nwc,
)

__all__ = [
    "CalculationResult",
    "to_decimal",
    "safe_divide",
    "safe_subtract",
    "round_display",
    "to_string",
    "validate_finite",
    "validate_rate",
    "validate_currency_match",
    "validate_required",
    "validate_positive",
    "validate_nonnegative",
    "revenue_growth",
    "ebitda",
    "ebitda_margin",
    "ebit_margin",
    "net_profit_margin",
    "debt_to_equity",
    "interest_coverage",
    "current_ratio",
    "operating_nwc",
    "nwc_change",
    "operating_tax",
    "nopat",
    "fcff",
]
