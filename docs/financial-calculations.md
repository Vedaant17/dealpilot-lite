# Financial Calculation Library

## Result representation

Successful calculations return `CalculationResult(value=Decimal(...))`.
Unavailable calculations return `CalculationResult(reason="...")`.
Callers access `.value` (Decimal or None) and `.reason` (str or None).
A result is available when `result.value is not None`.

## Decimal context

- No global Decimal context mutation occurs on import.
- Calculation functions use `localcontext()` with 28-digit precision.
- Display rounding is isolated and does not affect calculation precision.
- Importing the financial package does not change the caller's Decimal context.

## Implemented formulas (M2)

- Revenue Growth = (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}
- EBITDA = EBIT + D&A (D&A must be non-negative)
- EBITDA Margin = EBITDA / Revenue (withheld for zero/negative/missing revenue)
- EBIT Margin = EBIT / Revenue (withheld for zero/negative/missing revenue)
- Net Profit Margin = Net Income / Revenue (withheld for zero/negative/missing revenue)
- Debt-to-Equity = Total Debt / Shareholders' Equity (withheld for missing/zero/negative equity or negative debt)
- Interest Coverage = EBIT / Interest Expense (None for zero interest expense to avoid infinity)
- Current Ratio = Current Assets / Current Liabilities (withheld for zero/missing liabilities)
- Operating NWC = AR + Inventory + Other Op CA - AP - Other Op CL (excludes cash, debt, financing liabilities)
- ΔNWC = Current NWC - Prior NWC
- Operating Tax = EBIT × rate if EBIT > 0, else 0 (no tax benefit for negative EBIT)
- NOPAT = EBIT - Tax
- FCFF = NOPAT + D&A - CapEx - ΔNWC (CapEx as positive expenditure, subtracted once)

## Input conventions

- Monetary values: Decimal strings or Decimal objects. Floats converted via str().
- Missing values: None (not zero). Calculations return None when required inputs missing.
- Currency: validated for compatibility across inputs.
- CapEx: positive expenditure magnitude. Normalized at ingestion.
- Interest expense: positive expense magnitude.

## Decimal and rounding policy

- All calculations use Python Decimal with 28-digit precision.
- No intermediate rounding.
- Display rounding applied only at presentation layer (2 decimals for percentages, 4 for ratios).
- Monetary values stored as TEXT Decimal strings in SQLite.

## Missing-data behavior

- Functions return None when required inputs are missing or invalid.
- Reasons for unavailability are documented in metric output structures (M7+).
- Never silently substitute zero for missing values.

## Operating tax policy

- Tax = EBIT × rate when EBIT > 0.
- Tax = 0 when EBIT ≤ 0.
- No tax benefit recognized for negative EBIT unless explicit tax-loss assumption.

## NWC exclusions

Operating NWC excludes cash, interest-bearing debt, and financing liabilities.
Only operating current assets and operating current liabilities are included.

## CapEx sign convention

CapEx is represented as a positive expenditure amount in the calculation layer.
Source cash-flow values are normalized during ingestion if necessary.
FCFF subtracts CapEx exactly once.

## Known limitations (M2 scope)

- WACC, DCF, terminal value, equity bridge, comparable multiples, and sensitivity analysis are NOT implemented in M2. They belong to M7.
- Forecast NWC assumptions are not yet implemented.
- Currency conversion is not implemented (no implicit FX conversion in MVP).
- No database or MCP integration in this module — pure functions only.