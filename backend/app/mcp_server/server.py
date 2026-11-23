"""MCP server for DealPilot Lite."""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from backend.app.config import settings
from backend.app.financial import metrics, multiples, working_capital
from backend.app.financial.sensitivity import run_sensitivity_analysis as run_sensitivity_service
from backend.app.financial.valuation import calculate_valuation as calculate_valuation_service
from backend.app.schemas.mcp_tools import (
    ComparableFilter,
    ForecastOverrides,
    SensitivityRanges,
    ValuationAssumptions,
)
from backend.app.storage import repositories as repo

logger = logging.getLogger("dealpilot.mcp_server")


def _db_path_from_env_or_settings() -> str:
    return os.getenv("DEALPILOT_SQLITE_DB_PATH", settings.sqlite_db_path)


def _open_read_connection(db_path: str) -> sqlite3.Connection:
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    uri = resolved.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _metric(value, reason, units=None, currency=None):
    return {
        "value": None if value is None else str(value),
        "reason": reason,
        "units": units,
        "currency": currency,
    }


def _income_dict(statement):
    if statement is None:
        return None
    return {
        "revenue": _str(statement.revenue),
        "cogs": _str(statement.cogs),
        "gross_profit": _str(statement.gross_profit),
        "operating_expenses": _str(statement.operating_expenses),
        "ebitda": _str(statement.ebitda),
        "depreciation_amortization": _str(statement.depreciation_amortization),
        "operating_income": _str(statement.operating_income),
        "interest_expense": _str(statement.interest_expense),
        "taxes": _str(statement.taxes),
        "net_income": _str(statement.net_income),
    }


def _balance_dict(statement):
    if statement is None:
        return None
    return {
        "cash": _str(statement.cash),
        "accounts_receivable": _str(statement.accounts_receivable),
        "inventory": _str(statement.inventory),
        "other_operating_current_assets": _str(statement.other_operating_current_assets),
        "total_current_assets": _str(statement.total_current_assets),
        "total_assets": _str(statement.total_assets),
        "accounts_payable": _str(statement.accounts_payable),
        "other_operating_current_liabilities": _str(statement.other_operating_current_liabilities),
        "total_current_liabilities": _str(statement.total_current_liabilities),
        "total_debt": _str(statement.total_debt),
        "total_liabilities": _str(statement.total_liabilities),
        "shareholders_equity": _str(statement.shareholders_equity),
        "minority_interest": _str(statement.minority_interest),
        "stated_borrowing_rate": _str(statement.stated_borrowing_rate),
    }


def _cash_flow_dict(statement):
    if statement is None:
        return None
    return {
        "operating_cash_flow": _str(statement.operating_cash_flow),
        "capex": _str(statement.capex),
        "free_cash_flow": _str(statement.free_cash_flow),
        "dividends_paid": _str(statement.dividends_paid),
    }


def _str(value):
    return None if value is None else str(value)


def _valid_period_range(value: str) -> bool:
    if value.isdigit() and len(value) == 4:
        return True
    parts = value.split("-")
    return len(parts) == 2 and all(part.isdigit() and len(part) == 4 for part in parts)


def create_mcp_server(db_path: str | None = None):
    resolved_db_path = db_path or _db_path_from_env_or_settings()
    mcp = MCPServer("DealPilot Lite")

    @mcp.tool()
    async def get_company_financials(company_id: str, period_range: str = "all") -> dict[str, Any]:
        if period_range not in {"all"} and not _valid_period_range(period_range):
            raise ValueError("period_range must be 'all', a year, or YYYY-YYYY")
        conn = _open_read_connection(resolved_db_path)
        try:
            company = repo.get_company(conn, company_id)
            if company is None:
                return {"status": "unavailable", "reasons": ["company not found"], "periods": []}
            incomes = [
                item
                for item in repo.get_income_statements(conn, company_id, period_range)
                if item.is_current
            ]
            periods = []
            currencies = []
            for income in sorted(incomes, key=lambda item: item.fiscal_year):
                balance = repo.get_balance_sheet(conn, company_id, income.period_end_date)
                cash_flow = repo.get_cash_flow(conn, company_id, income.period_end_date)
                currencies.append(income.reporting_currency)
                periods.append(
                    {
                        "fiscal_year": income.fiscal_year,
                        "period_end_date": income.period_end_date,
                        "reporting_currency": income.reporting_currency,
                        "units_scale": income.units_scale,
                        "statement_version": income.statement_version,
                        "restated": income.restated,
                        "source_as_of_date": income.source_as_of_date,
                        "is_current": income.is_current,
                        "source": income.source,
                        "income_statement": _income_dict(income),
                        "balance_sheet": _balance_dict(balance),
                        "cash_flow_statement": _cash_flow_dict(cash_flow),
                        "source_data": income.source_data,
                    }
                )
            return {
                "company_id": company_id,
                "company_name": company.name,
                "periods": periods,
                "currency_consistent": len(set(currencies)) <= 1,
                "currencies": sorted(set(currencies)),
                "reasons": [],
            }
        finally:
            conn.close()

    @mcp.tool()
    async def calculate_financial_metrics(
        company_id: str, period_range: str = "all"
    ) -> dict[str, Any]:
        if period_range not in {"all"} and not _valid_period_range(period_range):
            raise ValueError("period_range must be 'all', a year, or YYYY-YYYY")
        conn = _open_read_connection(resolved_db_path)
        try:
            company = repo.get_company(conn, company_id)
            if company is None:
                return {"status": "unavailable", "reasons": ["company not found"], "periods": []}
            current_periods = [
                item
                for item in repo.get_income_statements(conn, company_id, period_range)
                if item.is_current
            ]
            current_periods.sort(key=lambda item: item.fiscal_year)
            output_periods = []
            for index, income in enumerate(current_periods):
                balance = repo.get_balance_sheet(conn, company_id, income.period_end_date)
                prior_income = current_periods[index - 1] if index > 0 else None
                prior_balance = (
                    repo.get_balance_sheet(conn, company_id, prior_income.period_end_date)
                    if prior_income
                    else None
                )

                prior_nwc = None
                if prior_balance is not None:
                    prior_result = working_capital.operating_nwc(
                        prior_balance.accounts_receivable,
                        prior_balance.inventory,
                        prior_balance.other_operating_current_assets,
                        prior_balance.accounts_payable,
                        prior_balance.other_operating_current_liabilities,
                    )
                    if prior_result.value is not None:
                        prior_nwc = prior_result.value

                current_nwc_result = working_capital.operating_nwc(
                    balance.accounts_receivable if balance else None,
                    balance.inventory if balance else None,
                    balance.other_operating_current_assets if balance else None,
                    balance.accounts_payable if balance else None,
                    balance.other_operating_current_liabilities if balance else None,
                )

                ebitda_result = metrics.ebitda(
                    income.operating_income, income.depreciation_amortization
                )
                revenue_growth_result = metrics.revenue_growth(
                    income.revenue, prior_income.revenue if prior_income else None
                )
                if (
                    prior_income is not None
                    and prior_income.reporting_currency != income.reporting_currency
                ):
                    revenue_growth_result = None
                    revenue_growth_reason = "incompatible currencies between periods"
                else:
                    revenue_growth_reason = (
                        None if revenue_growth_result is None else revenue_growth_result.reason
                    )

                nwc_change_result = None
                nwc_change_reason = None
                if prior_income is None:
                    nwc_change_reason = "prior period is unavailable"
                elif prior_income.reporting_currency != income.reporting_currency:
                    nwc_change_reason = "incompatible currencies between periods"
                elif current_nwc_result.value is None:
                    nwc_change_reason = current_nwc_result.reason
                elif prior_nwc is None:
                    nwc_change_reason = "prior period NWC is unavailable"
                else:
                    nwc_change_result = working_capital.nwc_change(
                        current_nwc_result.value, prior_nwc
                    )

                ebitda_value = ebitda_result.value
                metrics_output = {
                    "revenue_growth": _metric(
                        revenue_growth_result.value if revenue_growth_result else None,
                        revenue_growth_reason
                        or (
                            revenue_growth_result.reason
                            if revenue_growth_result
                            else "prior period is unavailable"
                        ),
                        units="ratio",
                    ),
                    "ebitda": _metric(
                        ebitda_value,
                        ebitda_result.reason,
                        units="currency_amount",
                        currency=income.reporting_currency,
                    ),
                    "ebitda_margin": _metric(
                        metrics.ebitda_margin(ebitda_value, income.revenue).value
                        if ebitda_value is not None
                        else None,
                        metrics.ebitda_margin(ebitda_value, income.revenue).reason
                        if ebitda_value is not None
                        else ebitda_result.reason,
                        units="ratio",
                    ),
                    "ebit_margin": _metric(
                        metrics.ebit_margin(income.operating_income, income.revenue).value,
                        metrics.ebit_margin(income.operating_income, income.revenue).reason,
                        units="ratio",
                    ),
                    "net_profit_margin": _metric(
                        metrics.net_profit_margin(income.net_income, income.revenue).value,
                        metrics.net_profit_margin(income.net_income, income.revenue).reason,
                        units="ratio",
                    ),
                    "debt_to_equity": _metric(
                        metrics.debt_to_equity(
                            balance.total_debt if balance else None,
                            balance.shareholders_equity if balance else None,
                        ).value,
                        metrics.debt_to_equity(
                            balance.total_debt if balance else None,
                            balance.shareholders_equity if balance else None,
                        ).reason,
                        units="ratio",
                    ),
                    "interest_coverage": _metric(
                        metrics.interest_coverage(
                            income.operating_income, income.interest_expense
                        ).value,
                        metrics.interest_coverage(
                            income.operating_income, income.interest_expense
                        ).reason,
                        units="ratio",
                    ),
                    "current_ratio": _metric(
                        metrics.current_ratio(
                            balance.total_current_assets if balance else None,
                            balance.total_current_liabilities if balance else None,
                        ).value,
                        metrics.current_ratio(
                            balance.total_current_assets if balance else None,
                            balance.total_current_liabilities if balance else None,
                        ).reason,
                        units="ratio",
                    ),
                    "operating_nwc": _metric(
                        current_nwc_result.value,
                        current_nwc_result.reason,
                        units="currency_amount",
                        currency=income.reporting_currency,
                    ),
                    "nwc_change": _metric(
                        nwc_change_result.value if nwc_change_result else None,
                        nwc_change_reason,
                        units="currency_amount",
                        currency=income.reporting_currency,
                    ),
                }
                output_periods.append(
                    {
                        "fiscal_year": income.fiscal_year,
                        "period_end_date": income.period_end_date,
                        "reporting_currency": income.reporting_currency,
                        "metrics": metrics_output,
                    }
                )
            return {
                "company_id": company_id,
                "company_name": company.name,
                "periods": output_periods,
                "reasons": [],
            }
        finally:
            conn.close()

    @mcp.tool()
    async def calculate_valuation(
        company_id: str,
        base_period: str = "latest",
        assumptions: ValuationAssumptions = ValuationAssumptions(),
    ) -> dict[str, Any]:
        conn = _open_read_connection(resolved_db_path)
        try:
            return calculate_valuation_service(conn, company_id, base_period, assumptions)
        finally:
            conn.close()

    @mcp.tool()
    async def calculate_comparable_multiples(
        company_id: str, peer_filter: ComparableFilter | None = None
    ) -> dict[str, Any]:
        conn = _open_read_connection(resolved_db_path)
        try:
            company = repo.get_company(conn, company_id)
            if company is None:
                return {"status": "unavailable", "reasons": ["company not found"], "peers": []}
            filter_dict = peer_filter.model_dump(exclude_none=True) if peer_filter else None
            peers = repo.get_comparables(conn, company_id, filter_dict)
            peer_rows = [multiples.calculate_peer_multiples(peer) for peer in peers]
            return {
                "company_id": company_id,
                "company_name": company.name,
                "peer_count": len(peer_rows),
                "peers": peer_rows,
                "reasons": [],
            }
        finally:
            conn.close()

    @mcp.tool()
    async def run_sensitivity_analysis(
        company_id: str,
        base_period: str = "latest",
        assumptions: ValuationAssumptions = ValuationAssumptions(),
        ranges: SensitivityRanges | None = None,
        overrides: ForecastOverrides | None = None,
    ) -> dict[str, Any]:
        if ranges is None:
            return {"status": "error", "reasons": ["ranges is required"], "scenarios": []}
        conn = _open_read_connection(resolved_db_path)
        try:
            return run_sensitivity_service(
                conn, company_id, base_period, assumptions, ranges, overrides
            )
        finally:
            conn.close()

    return mcp


mcp = create_mcp_server()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()
    if args.db_path:
        mcp = create_mcp_server(args.db_path)
    mcp.run("stdio")
