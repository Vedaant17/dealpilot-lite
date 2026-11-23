"""Tests for financial calculation library."""

from decimal import Decimal

import pytest

from backend.app.financial.decimal_utils import (
    round_display,
    safe_divide,
    safe_subtract,
    to_decimal,
    to_string,
)
from backend.app.financial.fcff import fcff, nopat, operating_tax
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
    validate_nonnegative,
    validate_positive,
    validate_rate,
    validate_required,
)
from backend.app.financial.working_capital import nwc_change, operating_nwc

# --- Decimal utilities ---


class TestToDecimal:
    def test_int(self):
        assert to_decimal(42) == Decimal(42)

    def test_str(self):
        assert to_decimal("123.45") == Decimal("123.45")

    def test_decimal(self):
        assert to_decimal(Decimal("99.99")) == Decimal("99.99")

    def test_float_via_str(self):
        assert to_decimal(0.1) == Decimal("0.1")

    def test_none(self):
        assert to_decimal(None) is None

    def test_empty_str(self):
        assert to_decimal("") is None

    def test_invalid_str(self):
        with pytest.raises(ValueError):
            to_decimal("abc")

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            to_decimal(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            to_decimal(float("inf"))

    def test_neg_inf_rejected(self):
        with pytest.raises(ValueError):
            to_decimal(float("-inf"))

    def test_decimal_nan_rejected(self):
        with pytest.raises(ValueError):
            to_decimal(Decimal("NaN"))

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            to_decimal(True)

    def test_negative_value(self):
        assert to_decimal("-50.25") == Decimal("-50.25")


class TestSafeDivide:
    def test_normal(self):
        r = safe_divide(Decimal(10), Decimal(2))
        assert r.value == Decimal(5)
        assert r.reason is None

    def test_zero_denominator(self):
        r = safe_divide(Decimal(10), Decimal(0))
        assert r.value is None
        assert r.reason is not None

    def test_none_numerator(self):
        r = safe_divide(None, Decimal(2))
        assert r.value is None
        assert r.reason is not None

    def test_none_denominator(self):
        r = safe_divide(Decimal(10), None)
        assert r.value is None
        assert r.reason is not None


class TestSafeSubtract:
    def test_normal(self):
        r = safe_subtract(Decimal(10), Decimal(3))
        assert r.value == Decimal(7)
        assert r.reason is None

    def test_none(self):
        r = safe_subtract(None, Decimal(3))
        assert r.value is None
        assert r.reason is not None


class TestRoundDisplay:
    def test_round(self):
        assert round_display(Decimal("123.456"), 2) == Decimal("123.46")

    def test_none(self):
        assert round_display(None) is None


class TestToString:
    def test_value(self):
        assert to_string(Decimal("42.5")) == "42.5"

    def test_none(self):
        assert to_string(None) is None


# --- Validation ---


class TestValidation:
    def test_rate_valid(self):
        assert validate_rate(0.5) == Decimal("0.5")

    def test_rate_boundary_zero(self):
        assert validate_rate(0) == Decimal(0)

    def test_rate_boundary_one(self):
        assert validate_rate(1) == Decimal(1)

    def test_rate_invalid(self):
        with pytest.raises(ValueError):
            validate_rate(1.5)

    def test_rate_none(self):
        assert validate_rate(None) is None

    def test_currency_match(self):
        assert validate_currency_match(["USD", "USD", None]) is True

    def test_currency_mismatch(self):
        assert validate_currency_match(["USD", "EUR"]) is False

    def test_required_none(self):
        with pytest.raises(ValueError):
            validate_required(None, "revenue")

    def test_positive_zero(self):
        with pytest.raises(ValueError):
            validate_positive(Decimal(0), "equity")

    def test_nonnegative_negative(self):
        with pytest.raises(ValueError):
            validate_nonnegative(Decimal(-1), "debt")


# --- Revenue growth ---


class TestRevenueGrowth:
    def test_normal(self):
        r = revenue_growth(120, 100)
        assert r.value == Decimal("0.2")
        assert r.reason is None

    def test_zero_prior(self):
        r = revenue_growth(120, 0)
        assert r.value is None
        assert r.reason is not None

    def test_negative_prior(self):
        r = revenue_growth(120, -10)
        assert r.value is None
        assert r.reason is not None

    def test_negative_current(self):
        r = revenue_growth(-10, 100)
        assert r.value is None
        assert r.reason is not None

    def test_missing(self):
        r = revenue_growth(None, 100)
        assert r.value is None
        assert r.reason is not None
        r = revenue_growth(120, None)
        assert r.value is None
        assert r.reason is not None

    def test_first_period(self):
        r = revenue_growth(100, None)
        assert r.value is None
        assert r.reason is not None


# --- EBITDA ---


class TestEbitda:
    def test_normal(self):
        r = ebitda(50, 10)
        assert r.value == Decimal(60)
        assert r.reason is None

    def test_missing(self):
        r = ebitda(None, 10)
        assert r.value is None
        assert r.reason is not None
        r = ebitda(50, None)
        assert r.value is None
        assert r.reason is not None

    def test_negative_da(self):
        r = ebitda(50, -5)
        assert r.value is None
        assert r.reason is not None


# --- Margins ---


class TestMargins:
    def test_ebitda_margin(self):
        r = ebitda_margin(40, 100)
        assert r.value == Decimal("0.4")
        assert r.reason is None

    def test_ebitda_margin_zero_revenue(self):
        r = ebitda_margin(40, 0)
        assert r.value is None
        assert r.reason is not None

    def test_ebitda_margin_negative_revenue(self):
        r = ebitda_margin(40, -100)
        assert r.value is None
        assert r.reason is not None

    def test_ebit_margin(self):
        r = ebit_margin(30, 100)
        assert r.value == Decimal("0.3")
        assert r.reason is None

    def test_net_profit_margin(self):
        r = net_profit_margin(15, 100)
        assert r.value == Decimal("0.15")
        assert r.reason is None

    def test_net_profit_margin_zero_revenue(self):
        r = net_profit_margin(15, 0)
        assert r.value is None
        assert r.reason is not None


# --- Debt to equity ---


class TestDebtToEquity:
    def test_normal(self):
        r = debt_to_equity(200, 100)
        assert r.value == Decimal(2)
        assert r.reason is None

    def test_zero_equity(self):
        r = debt_to_equity(200, 0)
        assert r.value is None
        assert r.reason is not None

    def test_negative_equity(self):
        r = debt_to_equity(200, -50)
        assert r.value is None
        assert r.reason is not None

    def test_negative_debt(self):
        r = debt_to_equity(-10, 100)
        assert r.value is None
        assert r.reason is not None

    def test_missing(self):
        r = debt_to_equity(None, 100)
        assert r.value is None
        assert r.reason is not None
        r = debt_to_equity(200, None)
        assert r.value is None
        assert r.reason is not None


# --- Interest coverage ---


class TestInterestCoverage:
    def test_normal(self):
        r = interest_coverage(50, 10)
        assert r.value == Decimal(5)
        assert r.reason is None

    def test_zero_interest(self):
        r = interest_coverage(50, 0)
        assert r.value is None
        assert r.reason is not None

    def test_negative_interest(self):
        r = interest_coverage(50, -5)
        assert r.value is None
        assert r.reason is not None

    def test_missing_ebit(self):
        r = interest_coverage(None, 10)
        assert r.value is None
        assert r.reason is not None

    def test_missing_interest(self):
        r = interest_coverage(50, None)
        assert r.value is None
        assert r.reason is not None


# --- Current ratio ---


class TestCurrentRatio:
    def test_normal(self):
        r = current_ratio(200, 100)
        assert r.value == Decimal(2)
        assert r.reason is None

    def test_zero_liabilities(self):
        r = current_ratio(200, 0)
        assert r.value is None
        assert r.reason is not None

    def test_missing(self):
        r = current_ratio(None, 100)
        assert r.value is None
        assert r.reason is not None


# --- Operating NWC ---


class TestOperatingNwc:
    def test_normal(self):
        r = operating_nwc(10, 5, 2, 3, 1)
        assert r.value == Decimal(13)
        assert r.reason is None

    def test_excludes_cash_and_debt(self):
        r = operating_nwc(10, 5, 2, 3, 1)
        assert r.value == Decimal(13)

    def test_missing_component(self):
        r = operating_nwc(None, 5, 2, 3, 1)
        assert r.value is None
        assert r.reason is not None
        r = operating_nwc(10, None, 2, 3, 1)
        assert r.value is None
        assert r.reason is not None
        r = operating_nwc(10, 5, None, 3, 1)
        assert r.value is None
        assert r.reason is not None
        r = operating_nwc(10, 5, 2, None, 1)
        assert r.value is None
        assert r.reason is not None
        r = operating_nwc(10, 5, 2, 3, None)
        assert r.value is None
        assert r.reason is not None

    def test_zero_values(self):
        r = operating_nwc(0, 0, 0, 0, 0)
        assert r.value == Decimal(0)
        assert r.reason is None


class TestNwcChange:
    def test_normal(self):
        r = nwc_change(Decimal(13), Decimal(10))
        assert r.value == Decimal(3)
        assert r.reason is None

    def test_missing(self):
        r = nwc_change(None, Decimal(10))
        assert r.value is None
        assert r.reason is not None
        r = nwc_change(Decimal(13), None)
        assert r.value is None
        assert r.reason is not None


# --- Operating tax and FCFF ---


class TestOperatingTax:
    def test_positive_ebit(self):
        r = operating_tax(100, Decimal("0.25"))
        assert r.value == Decimal(25)
        assert r.reason is None

    def test_zero_ebit(self):
        r = operating_tax(0, Decimal("0.25"))
        assert r.value == Decimal(0)
        assert r.reason is None

    def test_negative_ebit(self):
        r = operating_tax(-50, Decimal("0.25"))
        assert r.value == Decimal(0)
        assert r.reason is None

    def test_missing(self):
        r = operating_tax(None, Decimal("0.25"))
        assert r.value is None
        assert r.reason is not None
        r = operating_tax(100, None)
        assert r.value is None
        assert r.reason is not None


class TestNopat:
    def test_positive(self):
        r = nopat(100, Decimal("0.25"))
        assert r.value == Decimal(75)
        assert r.reason is None

    def test_negative_ebit(self):
        r = nopat(-50, Decimal("0.25"))
        assert r.value == Decimal(-50)
        assert r.reason is None

    def test_zero_ebit(self):
        r = nopat(0, Decimal("0.25"))
        assert r.value == Decimal(0)
        assert r.reason is None


class TestFcff:
    def test_normal(self):
        r = fcff(100, Decimal("0.25"), 10, 5, 3)
        assert r.value == Decimal(77)
        assert r.reason is None

    def test_negative_ebit(self):
        r = fcff(-50, Decimal("0.25"), 10, 5, 3)
        assert r.value == Decimal(-48)
        assert r.reason is None

    def test_missing_input(self):
        r = fcff(100, Decimal("0.25"), None, 5, 3)
        assert r.value is None
        assert r.reason is not None
        r = fcff(100, Decimal("0.25"), 10, None, 3)
        assert r.value is None
        assert r.reason is not None
        r = fcff(100, Decimal("0.25"), 10, 5, None)
        assert r.value is None
        assert r.reason is not None
        r = fcff(None, Decimal("0.25"), 10, 5, 3)
        assert r.value is None
        assert r.reason is not None

    def test_negative_da(self):
        r = fcff(100, Decimal("0.25"), -5, 5, 3)
        assert r.value is None
        assert r.reason is not None

    def test_negative_capex(self):
        r = fcff(100, Decimal("0.25"), 10, -5, 3)
        assert r.value is None
        assert r.reason is not None

    def test_missing_base_nwc(self):
        r = fcff(100, Decimal("0.25"), 10, 5, None)
        assert r.value is None
        assert r.reason is not None

    def test_capex_subtracted_once(self):
        r = fcff(100, Decimal("0.25"), 10, 5, 3)
        assert r.value == Decimal(77)

    def test_delta_nwc_sign(self):
        r = fcff(100, Decimal("0.25"), 10, 5, -3)
        assert r.value == Decimal(83)


# --- Decimal context regression tests ---


class TestDecimalContext:
    def test_import_does_not_change_precision(self):
        from decimal import getcontext

        before = getcontext().prec
        import backend.app.financial  # noqa: F401

        after = getcontext().prec
        assert before == after, f"Import changed precision from {before} to {after}"

    def test_calculation_does_not_change_precision(self):
        from decimal import getcontext

        before = getcontext().prec
        revenue_growth(100, 80)
        ebitda(50, 10)
        fcff(100, Decimal("0.25"), 10, 5, 3)
        operating_nwc(10, 5, 2, 3, 1)
        after = getcontext().prec
        assert before == after, f"Calculation changed precision from {before} to {after}"

    def test_calculation_precision_is_28(self):
        r = safe_divide(Decimal(1), Decimal(3))
        assert r.value is not None
        # 1/3 to 28 digits
        assert str(r.value) == "0.3333333333333333333333333333"

    def test_display_rounding_isolated(self):
        from decimal import getcontext

        before = getcontext().prec
        round_display(Decimal("123.456789"), 2)
        after = getcontext().prec
        assert before == after

    def test_currency_mismatch_behavior(self):
        # validate_currency_match returns False for mismatch
        from backend.app.financial.validation import validate_currency_match

        assert validate_currency_match(["USD", "EUR"]) is False
        assert validate_currency_match(["USD", "USD"]) is True
