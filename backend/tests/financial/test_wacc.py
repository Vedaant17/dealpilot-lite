from decimal import Decimal

from backend.app.financial.wacc import (
    calculate_component_wacc,
    estimate_cost_of_debt,
)


def test_component_wacc_zero_debt():
    result = calculate_component_wacc(
        equity=Decimal("100"),
        debt=Decimal("0"),
        cost_of_equity=Decimal("0.10"),
        cost_of_debt=None,
        interest_tax_shield_rate=None,
        operating_tax_rate=Decimal("0.25"),
    )
    assert result.value == Decimal("0.10")


def test_component_wacc_missing_debt():
    result = calculate_component_wacc(
        equity=Decimal("100"),
        debt=None,
        cost_of_equity=Decimal("0.10"),
        cost_of_debt=Decimal("0.05"),
        interest_tax_shield_rate=Decimal("0.25"),
        operating_tax_rate=Decimal("0.25"),
    )
    assert result.value is None
    assert "missing total debt" in result.reason


def test_component_wacc_negative_equity():
    result = calculate_component_wacc(
        equity=Decimal("-1"),
        debt=Decimal("10"),
        cost_of_equity=Decimal("0.10"),
        cost_of_debt=Decimal("0.05"),
        interest_tax_shield_rate=Decimal("0.25"),
        operating_tax_rate=Decimal("0.25"),
    )
    assert result.value is None


def test_cost_of_debt_stated_rate():
    estimate = estimate_cost_of_debt(
        stated_rate=Decimal("0.07"), interest_expense=Decimal("5"), debt=Decimal("100")
    )
    assert estimate.value == Decimal("0.07")
    assert estimate.method == "stated_borrowing_rate"


def test_cost_of_debt_average_then_closing():
    average = estimate_cost_of_debt(
        interest_expense=Decimal("10"), debt=Decimal("100"), average_debt=Decimal("80")
    )
    closing = estimate_cost_of_debt(interest_expense=Decimal("10"), debt=Decimal("100"))
    assert average.method == "average_debt"
    assert closing.method == "closing_debt"
    assert average.value == Decimal("0.125")
    assert closing.value == Decimal("0.10")
