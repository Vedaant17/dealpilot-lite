"""DCF valuation services for DealPilot Lite."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from typing import Any, Optional

from backend.app.financial.decimal_utils import to_decimal
from backend.app.financial.fcff import operating_tax
from backend.app.financial.metrics import ebit_margin as ebit_margin_metric
from backend.app.financial.wacc import (
    calculate_component_wacc,
    ensure_calculation_result,
    estimate_cost_of_debt,
)
from backend.app.financial.working_capital import operating_nwc
from backend.app.storage import repositories as repo
from backend.app.storage.models import BalanceSheet

CALCULATION_PRECISION = 28
BOOK_VALUE_WEIGHTS_DISCLOSURE = (
    "Book-value capital weights are used as an MVP approximation because market-value "
    "equity and debt inputs may not be available. This can materially affect WACC and "
    "the resulting illustrative valuation."
)


def _str(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else str(value)


def _current_periods(conn: sqlite3.Connection, company_id: str, period_range: str):
    incomes = repo.get_income_statements(conn, company_id, period_range)
    current = [item for item in incomes if item.is_current]
    result = []
    for income in sorted(current, key=lambda item: item.fiscal_year):
        balance = repo.get_balance_sheet(conn, company_id, income.period_end_date)
        cash_flow = repo.get_cash_flow(conn, company_id, income.period_end_date)
        result.append((income, balance, cash_flow))
    return result


def _select_base_period(periods, requested: str):
    if not periods:
        return None
    if requested == "latest":
        return periods[-1]
    for item in periods:
        income = item[0]
        if requested == income.period_end_date or requested == str(income.fiscal_year):
            return item
    return None


def _period_currency(periods) -> Optional[str]:
    currencies = [item[0].reporting_currency for item in periods]
    if not currencies:
        return None
    if len(set(currencies)) != 1:
        return None
    return currencies[0]


def _calculate_base_nwc(balance: Optional[BalanceSheet]):
    if balance is None:
        return None
    result = operating_nwc(
        balance.accounts_receivable,
        balance.inventory,
        balance.other_operating_current_assets,
        balance.accounts_payable,
        balance.other_operating_current_liabilities,
    )
    ensure_calculation_result(result, "base NWC")
    return result.value


def calculate_valuation(
    conn: sqlite3.Connection,
    company_id: str,
    base_period: str,
    assumptions: Any,
) -> dict[str, Any]:
    company = repo.get_company(conn, company_id)
    if company is None:
        return {"available": False, "reasons": ["company not found"], "warnings": []}

    periods = _current_periods(conn, company_id, "all")
    base = _select_base_period(periods, base_period)
    if base is None:
        return {"available": False, "reasons": ["requested base period not found"], "warnings": []}

    base_income, base_balance, _ = base
    if base_income.reporting_currency is None:
        return {
            "available": False,
            "reasons": ["base period reporting currency is missing"],
            "warnings": [],
        }

    currency = base_income.reporting_currency
    reasons: list[str] = []
    warnings: list[str] = []

    base_revenue = base_income.revenue
    base_ebit = base_income.operating_income
    base_da = base_income.depreciation_amortization

    if assumptions.operating_margin is not None:
        operating_margin = assumptions.operating_margin
        margin_source = "supplied"
    else:
        margin_result = ebit_margin_metric(base_ebit, base_revenue)
        ensure_calculation_result(margin_result, "historical operating margin")
        if margin_result.value is None:
            operating_margin = None
            margin_source = "unavailable"
            reasons.append(margin_result.reason or "operating margin unavailable")
        else:
            operating_margin = margin_result.value
            margin_source = "historical_base_period"

    base_nwc = _calculate_base_nwc(base_balance)
    if base_nwc is None:
        nwc_reason = "base-period operating NWC is unavailable"
    else:
        nwc_reason = None

    # WACC
    wacc_value: Optional[Decimal] = None
    wacc_method = "unavailable"
    wacc_reason = None
    cost_of_debt_method = None
    limitations: list[str] = []

    if assumptions.direct_wacc is not None:
        wacc_value = assumptions.direct_wacc
        wacc_method = "user_provided"
    else:
        if assumptions.cost_of_equity is None:
            wacc_reason = "cost of equity is required for component WACC"
        else:
            eligible = []
            latest_year = max(
                (item[0].fiscal_year for item in periods), default=base_income.fiscal_year
            )
            cutoff = latest_year - 2
            for item in periods:
                income, balance, _ = item
                if income.fiscal_year < cutoff:
                    continue
                if income.reporting_currency != currency:
                    continue
                if balance is None or balance.shareholders_equity is None:
                    continue
                if balance.shareholders_equity <= 0:
                    continue
                if balance.total_debt is None:
                    continue
                if balance.total_debt < 0:
                    continue
                eligible.append(item)

            eligible.sort(key=lambda item: item[0].fiscal_year, reverse=True)
            selected_wacc_period = None
            for item in eligible:
                income, balance, _ = item
                if balance.total_debt and balance.total_debt > 0:
                    estimate = estimate_cost_of_debt(
                        stated_rate=balance.stated_borrowing_rate,
                        interest_expense=income.interest_expense,
                        debt=balance.total_debt,
                    )
                    if estimate.value is None:
                        continue
                    cost_of_debt = estimate.value
                    cost_of_debt_method = estimate.method
                else:
                    cost_of_debt = None
                    cost_of_debt_method = "not_required_zero_debt"
                selected_wacc_period = (income, balance, cost_of_debt)
                break

            if selected_wacc_period is None:
                wacc_reason = "no eligible period with positive equity and known nonnegative debt within two years"
            else:
                income, balance, cost_of_debt = selected_wacc_period
                if balance.total_debt and balance.total_debt > 0 and cost_of_debt is None:
                    wacc_reason = "usable cost of debt is unavailable for positive debt"
                else:
                    if balance.total_debt == 0:
                        cost_of_debt = Decimal(0)
                        cost_of_debt_method = "not_required_zero_debt"
                    shield_rate = assumptions.interest_tax_shield_rate
                    if shield_rate is None:
                        shield_rate = assumptions.operating_tax_rate
                    wacc_result = calculate_component_wacc(
                        equity=balance.shareholders_equity,
                        debt=balance.total_debt,
                        cost_of_equity=assumptions.cost_of_equity,
                        cost_of_debt=cost_of_debt,
                        interest_tax_shield_rate=shield_rate,
                        operating_tax_rate=assumptions.operating_tax_rate,
                    )
                    ensure_calculation_result(wacc_result, "component WACC")
                    if wacc_result.value is None:
                        wacc_reason = wacc_result.reason
                    else:
                        wacc_value = wacc_result.value
                        wacc_method = "component"
                        limitations.append(BOOK_VALUE_WEIGHTS_DISCLOSURE)
                        if cost_of_debt_method and cost_of_debt_method not in (
                            "not_required_zero_debt",
                        ):
                            limitations.append(
                                f"cost of debt estimated using {cost_of_debt_method}"
                            )

    if wacc_value is not None and wacc_value <= assumptions.terminal_growth:
        wacc_reason = "WACC must be greater than terminal growth"
        wacc_value = None

    # Growth rates
    if assumptions.revenue_growth_rate is None and assumptions.revenue_growth_rates is None:
        return {
            "available": False,
            "reasons": ["forecast revenue growth assumption is required"],
            "warnings": [],
            "company_id": company_id,
            "currency": currency,
        }
    growth_rates: list[Decimal]
    if assumptions.revenue_growth_rates is not None:
        growth_rates = list(assumptions.revenue_growth_rates)
    else:
        growth_rates = [assumptions.revenue_growth_rate] * assumptions.forecast_horizon

    for rate in growth_rates:
        if abs(rate) > assumptions.growth_warning_threshold:
            warnings.append(
                f"forecast revenue growth {rate} exceeds warning threshold {assumptions.growth_warning_threshold}"
            )

    # Forecast
    forecast = []
    available = True
    if base_revenue is None:
        available = False
        reasons.append("base revenue is missing")
    if operating_margin is None:
        available = False
    if wacc_value is None:
        available = False
        if wacc_reason:
            reasons.append(wacc_reason)
    if base_nwc is None:
        available = False
        if nwc_reason:
            reasons.append(nwc_reason)

    if available:
        revenue = base_revenue
        nwc_prev = base_nwc
        for index, growth in enumerate(growth_rates, start=1):
            revenue = revenue * (Decimal(1) + growth)
            ebit = revenue * operating_margin
            tax_result = operating_tax(ebit, assumptions.operating_tax_rate)
            ensure_calculation_result(tax_result, "operating tax")
            tax = tax_result.value
            nopat_value = ebit - tax
            da = revenue * assumptions.da_pct_revenue
            capex = revenue * assumptions.capex_pct_revenue
            if assumptions.forecast_nwc_values and len(assumptions.forecast_nwc_values) >= index:
                nwc = assumptions.forecast_nwc_values[index - 1]
            else:
                nwc = revenue * assumptions.nwc_pct_revenue
            delta_nwc = nwc - nwc_prev
            fcff_value = nopat_value + da - capex - delta_nwc
            forecast.append(
                {
                    "year": index,
                    "revenue": _str(revenue),
                    "ebit": _str(ebit),
                    "tax": _str(tax),
                    "nopat": _str(nopat_value),
                    "da": _str(da),
                    "capex": _str(capex),
                    "nwc": _str(nwc),
                    "delta_nwc": _str(delta_nwc),
                    "fcff": _str(fcff_value),
                }
            )
            nwc_prev = nwc

        fcff_values = [to_decimal(row["fcff"]) for row in forecast]
        if any(value is None for value in fcff_values):
            available = False
            reasons.append("forecast FCFF could not be calculated")
        else:
            last_fcff = fcff_values[-1]
            terminal_value = (
                last_fcff
                * (Decimal(1) + assumptions.terminal_growth)
                / (wacc_value - assumptions.terminal_growth)
            )
            pv_fcff = []
            for index, value in enumerate(fcff_values, start=1):
                pv = value / ((Decimal(1) + wacc_value) ** index)
                pv_fcff.append(pv)
            pv_terminal = terminal_value / (
                (Decimal(1) + wacc_value) ** assumptions.forecast_horizon
            )
            enterprise_value = sum(pv_fcff, Decimal(0)) + pv_terminal

    # Equity bridge
    net_debt = None
    equity_value = None
    per_share_value = None
    if base_balance is not None:
        debt = base_balance.total_debt
        cash = base_balance.cash
        minority = base_balance.minority_interest
        if debt is not None and cash is not None:
            net_debt = debt - cash
        if available and enterprise_value is not None:
            if debt is None or cash is None or minority is None:
                pass
            else:
                equity_value = (
                    enterprise_value - debt + cash - minority + assumptions.other_adjustments
                )
                if company.shares_outstanding is not None and company.shares_outstanding > 0:
                    per_share_value = equity_value / company.shares_outstanding

    if available and (
        base_balance is None
        or base_balance.total_debt is None
        or base_balance.cash is None
        or base_balance.minority_interest is None
    ):
        if equity_value is None:
            reasons.append(
                "equity bridge unavailable because debt, cash or minority interest is missing"
            )

    return {
        "available": available,
        "company_id": company_id,
        "company_name": company.name,
        "currency": currency,
        "base_period": {
            "fiscal_year": base_income.fiscal_year,
            "period_end_date": base_income.period_end_date,
            "reporting_currency": base_income.reporting_currency,
            "units_scale": base_income.units_scale,
        },
        "base_values": {
            "revenue": _str(base_revenue),
            "operating_income": _str(base_ebit),
            "depreciation_amortization": _str(base_da),
            "operating_margin": _str(operating_margin),
            "operating_margin_source": margin_source,
            "nwc": _str(base_nwc),
            "total_debt": _str(base_balance.total_debt if base_balance else None),
            "cash": _str(base_balance.cash if base_balance else None),
            "minority_interest": _str(base_balance.minority_interest if base_balance else None),
            "shareholders_equity": _str(base_balance.shareholders_equity if base_balance else None),
            "shares_outstanding": _str(company.shares_outstanding),
            "shares_outstanding_source": company.shares_outstanding_source,
            "shares_outstanding_as_of_date": company.shares_outstanding_as_of_date,
        },
        "assumptions": assumptions.model_dump(mode="json"),
        "wacc": {
            "value": _str(wacc_value),
            "method": wacc_method,
            "cost_of_debt_method": cost_of_debt_method,
            "reason": wacc_reason,
        },
        "forecast": forecast,
        "terminal_value": _str(terminal_value) if available else None,
        "present_value_fcff": [_str(value) for value in pv_fcff] if available else [],
        "present_value_terminal": _str(pv_terminal) if available else None,
        "enterprise_value": _str(enterprise_value) if available else None,
        "net_debt": _str(net_debt),
        "equity_value": _str(equity_value),
        "per_share_value": _str(per_share_value),
        "limitations": limitations,
        "warnings": warnings,
        "reasons": reasons,
    }
