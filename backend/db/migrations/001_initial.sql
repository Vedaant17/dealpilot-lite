-- Migration 001: Initial schema for DealPilot Lite
-- All financial monetary values stored as TEXT (Decimal strings).
-- No SQLite REAL for monetary values or ratios.

CREATE TABLE IF NOT EXISTS companies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    industry TEXT NOT NULL,
    description TEXT,
    founded_year INTEGER,
    headquarters TEXT,
    default_reporting_currency TEXT NOT NULL,
    shares_outstanding TEXT,
    shares_outstanding_as_of_date TEXT,
    shares_outstanding_source TEXT,
    is_synthetic INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS income_statements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id TEXT NOT NULL REFERENCES companies(id),
    fiscal_year INTEGER NOT NULL,
    period_end_date TEXT NOT NULL,
    statement_version INTEGER NOT NULL DEFAULT 1,
    restated INTEGER NOT NULL DEFAULT 0,
    source_as_of_date TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    reporting_currency TEXT NOT NULL,
    units_scale TEXT NOT NULL,
    revenue TEXT, cogs TEXT, gross_profit TEXT, operating_expenses TEXT,
    ebitda TEXT, depreciation_amortization TEXT, operating_income TEXT,
    interest_expense TEXT, taxes TEXT, net_income TEXT,
    source TEXT NOT NULL,
    source_data TEXT,
    UNIQUE(company_id, period_end_date, statement_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_income_current
    ON income_statements(company_id, period_end_date) WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS balance_sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id TEXT NOT NULL REFERENCES companies(id),
    fiscal_year INTEGER NOT NULL,
    period_end_date TEXT NOT NULL,
    statement_version INTEGER NOT NULL DEFAULT 1,
    restated INTEGER NOT NULL DEFAULT 0,
    source_as_of_date TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    reporting_currency TEXT NOT NULL,
    units_scale TEXT NOT NULL,
    cash TEXT, accounts_receivable TEXT, inventory TEXT,
    other_operating_current_assets TEXT, total_current_assets TEXT, total_assets TEXT,
    accounts_payable TEXT, other_operating_current_liabilities TEXT,
    total_current_liabilities TEXT, total_debt TEXT, total_liabilities TEXT,
    shareholders_equity TEXT, minority_interest TEXT, stated_borrowing_rate TEXT,
    source TEXT NOT NULL,
    source_data TEXT,
    UNIQUE(company_id, period_end_date, statement_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_balance_current
    ON balance_sheets(company_id, period_end_date) WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS cash_flow_statements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id TEXT NOT NULL REFERENCES companies(id),
    fiscal_year INTEGER NOT NULL,
    period_end_date TEXT NOT NULL,
    statement_version INTEGER NOT NULL DEFAULT 1,
    restated INTEGER NOT NULL DEFAULT 0,
    source_as_of_date TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    reporting_currency TEXT NOT NULL,
    units_scale TEXT NOT NULL,
    operating_cash_flow TEXT, capex TEXT, free_cash_flow TEXT, dividends_paid TEXT,
    source TEXT NOT NULL,
    source_data TEXT,
    UNIQUE(company_id, period_end_date, statement_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cashflow_current
    ON cash_flow_statements(company_id, period_end_date) WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(id),
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    file_path_or_id TEXT NOT NULL,
    page_count INTEGER,
    metadata TEXT,
    ingested_at TEXT NOT NULL,
    UNIQUE(id, company_id)
);

CREATE TABLE IF NOT EXISTS comparables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id TEXT NOT NULL REFERENCES companies(id),
    peer_name TEXT NOT NULL,
    industry TEXT,
    measurement_date TEXT NOT NULL,
    currency TEXT NOT NULL,
    revenue TEXT, ebitda TEXT, ebit TEXT,
    enterprise_value TEXT, equity_value TEXT, net_debt TEXT,
    ev_ebitda TEXT, pe_ratio TEXT, revenue_growth TEXT, ebitda_margin TEXT,
    source TEXT NOT NULL,
    calculation_method TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);