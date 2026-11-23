"""Data models for SQLite storage."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass
class Company:
    id: str
    name: str
    industry: str
    description: str = ""
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    default_reporting_currency: str = "USD"
    shares_outstanding: Optional[Decimal] = None
    shares_outstanding_as_of_date: Optional[str] = None
    shares_outstanding_source: Optional[str] = None
    is_synthetic: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass
class IncomeStatement:
    company_id: str
    fiscal_year: int
    period_end_date: str
    statement_version: int = 1
    restated: bool = False
    source_as_of_date: str = ""
    is_current: bool = False
    reporting_currency: str = "USD"
    units_scale: str = "units"
    revenue: Optional[Decimal] = None
    cogs: Optional[Decimal] = None
    gross_profit: Optional[Decimal] = None
    operating_expenses: Optional[Decimal] = None
    ebitda: Optional[Decimal] = None
    depreciation_amortization: Optional[Decimal] = None
    operating_income: Optional[Decimal] = None
    interest_expense: Optional[Decimal] = None
    taxes: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    source: str = ""
    source_data: Optional[dict] = None
    id: Optional[int] = None


@dataclass
class BalanceSheet:
    company_id: str
    fiscal_year: int
    period_end_date: str
    statement_version: int = 1
    restated: bool = False
    source_as_of_date: str = ""
    is_current: bool = False
    reporting_currency: str = "USD"
    units_scale: str = "units"
    cash: Optional[Decimal] = None
    accounts_receivable: Optional[Decimal] = None
    inventory: Optional[Decimal] = None
    other_operating_current_assets: Optional[Decimal] = None
    total_current_assets: Optional[Decimal] = None
    total_assets: Optional[Decimal] = None
    accounts_payable: Optional[Decimal] = None
    other_operating_current_liabilities: Optional[Decimal] = None
    total_current_liabilities: Optional[Decimal] = None
    total_debt: Optional[Decimal] = None
    total_liabilities: Optional[Decimal] = None
    shareholders_equity: Optional[Decimal] = None
    minority_interest: Optional[Decimal] = None
    stated_borrowing_rate: Optional[Decimal] = None
    source: str = ""
    source_data: Optional[dict] = None
    id: Optional[int] = None


@dataclass
class CashFlow:
    company_id: str
    fiscal_year: int
    period_end_date: str
    statement_version: int = 1
    restated: bool = False
    source_as_of_date: str = ""
    is_current: bool = False
    reporting_currency: str = "USD"
    units_scale: str = "units"
    operating_cash_flow: Optional[Decimal] = None
    capex: Optional[Decimal] = None
    free_cash_flow: Optional[Decimal] = None
    dividends_paid: Optional[Decimal] = None
    source: str = ""
    source_data: Optional[dict] = None
    id: Optional[int] = None


@dataclass
class Document:
    id: str
    company_id: str
    title: str
    doc_type: str
    file_path_or_id: str
    page_count: Optional[int] = None
    metadata: Optional[dict] = None
    ingested_at: str = ""
    file_format: Optional[str] = None
    storage_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    content_hash: Optional[str] = None
    source_reference: Optional[str] = None
    ingestion_status: str = "pending"
    extraction_error: Optional[str] = None
    superseded_by: Optional[str] = None


@dataclass
class Comparable:
    id: Optional[int] = None
    company_id: str = ""
    peer_name: str = ""
    industry: Optional[str] = None
    measurement_date: str = ""
    currency: str = "USD"
    revenue: Optional[Decimal] = None
    ebitda: Optional[Decimal] = None
    ebit: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None
    equity_value: Optional[Decimal] = None
    net_debt: Optional[Decimal] = None
    ev_ebitda: Optional[Decimal] = None
    pe_ratio: Optional[Decimal] = None
    revenue_growth: Optional[Decimal] = None
    ebitda_margin: Optional[Decimal] = None
    source: str = ""
    calculation_method: str = "sourced"
    notes: Optional[str] = None
