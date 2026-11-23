"""Repository functions for SQLite storage."""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from backend.app.financial.decimal_utils import to_decimal
from backend.app.storage.models import (
    BalanceSheet,
    CashFlow,
    Company,
    Comparable,
    Document,
    IncomeStatement,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dec(val) -> Optional[Decimal]:
    if val is None:
        return None
    return to_decimal(val)


def _to_str(val: Optional[Decimal]) -> Optional[str]:
    if val is None:
        return None
    return str(val)


def _parse_json(val) -> Optional[dict]:
    if val is None:
        return None
    try:
        import json as _json

        return _json.loads(val)
    except (ValueError, TypeError):
        return None


def _to_json(val) -> Optional[str]:
    if val is None:
        return None
    import json as _json

    return _json.dumps(val)


# --- Companies ---


def insert_company(conn: sqlite3.Connection, company: Company) -> None:
    conn.execute(
        """INSERT INTO companies (id, name, industry, description, founded_year,
           headquarters, default_reporting_currency, shares_outstanding,
           shares_outstanding_as_of_date, shares_outstanding_source, is_synthetic,
           created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            company.id,
            company.name,
            company.industry,
            company.description,
            company.founded_year,
            company.headquarters,
            company.default_reporting_currency,
            _to_str(company.shares_outstanding),
            company.shares_outstanding_as_of_date,
            company.shares_outstanding_source,
            1 if company.is_synthetic else 0,
            company.created_at or _now(),
            company.updated_at or _now(),
        ),
    )


def get_company(conn: sqlite3.Connection, company_id: str) -> Optional[Company]:
    row = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
    if row is None:
        return None
    return Company(
        id=row["id"],
        name=row["name"],
        industry=row["industry"],
        description=row["description"],
        founded_year=row["founded_year"],
        headquarters=row["headquarters"],
        default_reporting_currency=row["default_reporting_currency"],
        shares_outstanding=to_decimal(row["shares_outstanding"]),
        shares_outstanding_as_of_date=row["shares_outstanding_as_of_date"],
        shares_outstanding_source=row["shares_outstanding_source"],
        is_synthetic=bool(row["is_synthetic"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def get_companies(conn: sqlite3.Connection) -> list:
    rows = conn.execute("SELECT * FROM companies ORDER BY id").fetchall()
    return [get_company(conn, r["id"]) for r in rows]


def update_company(conn: sqlite3.Connection, company: Company) -> None:
    conn.execute(
        """UPDATE companies SET name=?, industry=?, description=?, founded_year=?,
           headquarters=?, default_reporting_currency=?, shares_outstanding=?,
           shares_outstanding_as_of_date=?, shares_outstanding_source=?,
           is_synthetic=?, updated_at=? WHERE id=?""",
        (
            company.name,
            company.industry,
            company.description,
            company.founded_year,
            company.headquarters,
            company.default_reporting_currency,
            _to_str(company.shares_outstanding),
            company.shares_outstanding_as_of_date,
            company.shares_outstanding_source,
            1 if company.is_synthetic else 0,
            _now(),
            company.id,
        ),
    )


# --- Income Statements ---


def _row_to_income(row) -> IncomeStatement:
    return IncomeStatement(
        id=row["id"],
        company_id=row["company_id"],
        fiscal_year=row["fiscal_year"],
        period_end_date=row["period_end_date"],
        statement_version=row["statement_version"],
        restated=bool(row["restated"]),
        source_as_of_date=row["source_as_of_date"],
        is_current=bool(row["is_current"]),
        reporting_currency=row["reporting_currency"],
        units_scale=row["units_scale"],
        revenue=to_decimal(row["revenue"]),
        cogs=to_decimal(row["cogs"]),
        gross_profit=to_decimal(row["gross_profit"]),
        operating_expenses=to_decimal(row["operating_expenses"]),
        ebitda=to_decimal(row["ebitda"]),
        depreciation_amortization=to_decimal(row["depreciation_amortization"]),
        operating_income=to_decimal(row["operating_income"]),
        interest_expense=to_decimal(row["interest_expense"]),
        taxes=to_decimal(row["taxes"]),
        net_income=to_decimal(row["net_income"]),
        source=row["source"],
        source_data=_parse_json(row["source_data"]),
    )


def insert_income_statement(conn: sqlite3.Connection, stmt: IncomeStatement) -> None:
    conn.execute(
        """INSERT INTO income_statements (company_id, fiscal_year, period_end_date,
           statement_version, restated, source_as_of_date, is_current,
           reporting_currency, units_scale, revenue, cogs, gross_profit,
           operating_expenses, ebitda, depreciation_amortization, operating_income,
           interest_expense, taxes, net_income, source, source_data)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stmt.company_id,
            stmt.fiscal_year,
            stmt.period_end_date,
            stmt.statement_version,
            1 if stmt.restated else 0,
            stmt.source_as_of_date,
            1 if stmt.is_current else 0,
            stmt.reporting_currency,
            stmt.units_scale,
            _to_str(stmt.revenue),
            _to_str(stmt.cogs),
            _to_str(stmt.gross_profit),
            _to_str(stmt.operating_expenses),
            _to_str(stmt.ebitda),
            _to_str(stmt.depreciation_amortization),
            _to_str(stmt.operating_income),
            _to_str(stmt.interest_expense),
            _to_str(stmt.taxes),
            _to_str(stmt.net_income),
            stmt.source,
            _to_json(stmt.source_data),
        ),
    )


def get_income_statement(conn, company_id, period_end_date, version=None):
    if version is not None:
        row = conn.execute(
            "SELECT * FROM income_statements WHERE company_id=? AND period_end_date=? AND statement_version=?",
            (company_id, period_end_date, version),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM income_statements WHERE company_id=? AND period_end_date=? AND is_current=1",
            (company_id, period_end_date),
        ).fetchone()
    if row is None:
        return None
    return _row_to_income(row)


def get_current_income_statement(conn, company_id, period_end_date):
    return get_income_statement(conn, company_id, period_end_date, version=None)


def get_income_statements(conn, company_id, period_range=None):
    query = "SELECT * FROM income_statements WHERE company_id=?"
    params = [company_id]
    if period_range and period_range != "all":
        parts = period_range.split("-")
        if len(parts) == 2:
            query += " AND fiscal_year >= ? AND fiscal_year <= ?"
            params.extend([int(parts[0]), int(parts[1])])
    query += " ORDER BY fiscal_year, statement_version"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_income(r) for r in rows]


def insert_restated_statement(conn, stmt, new_current_version=None):
    """Atomically insert a restated statement and reassign current version."""
    if new_current_version is None:
        new_current_version = stmt.statement_version
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE income_statements SET is_current=0 WHERE company_id=? AND period_end_date=?",
            (stmt.company_id, stmt.period_end_date),
        )
        insert_income_statement(conn, stmt)
        conn.execute(
            "UPDATE income_statements SET is_current=1 WHERE company_id=? AND period_end_date=? AND statement_version=?",
            (stmt.company_id, stmt.period_end_date, new_current_version),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --- Balance Sheets ---


def _row_to_balance(row) -> BalanceSheet:
    return BalanceSheet(
        id=row["id"],
        company_id=row["company_id"],
        fiscal_year=row["fiscal_year"],
        period_end_date=row["period_end_date"],
        statement_version=row["statement_version"],
        restated=bool(row["restated"]),
        source_as_of_date=row["source_as_of_date"],
        is_current=bool(row["is_current"]),
        reporting_currency=row["reporting_currency"],
        units_scale=row["units_scale"],
        cash=_dec(row["cash"]),
        accounts_receivable=_dec(row["accounts_receivable"]),
        inventory=_dec(row["inventory"]),
        other_operating_current_assets=_dec(row["other_operating_current_assets"]),
        total_current_assets=_dec(row["total_current_assets"]),
        total_assets=_dec(row["total_assets"]),
        accounts_payable=_dec(row["accounts_payable"]),
        other_operating_current_liabilities=_dec(row["other_operating_current_liabilities"]),
        total_current_liabilities=_dec(row["total_current_liabilities"]),
        total_debt=_dec(row["total_debt"]),
        total_liabilities=_dec(row["total_liabilities"]),
        shareholders_equity=_dec(row["shareholders_equity"]),
        minority_interest=_dec(row["minority_interest"]),
        stated_borrowing_rate=_dec(row["stated_borrowing_rate"]),
        source=row["source"],
        source_data=_parse_json(row["source_data"]),
    )


def insert_balance_sheet(conn: sqlite3.Connection, stmt: BalanceSheet) -> None:
    conn.execute(
        """INSERT INTO balance_sheets (company_id, fiscal_year, period_end_date,
           statement_version, restated, source_as_of_date, is_current,
           reporting_currency, units_scale, cash, accounts_receivable, inventory,
           other_operating_current_assets, total_current_assets, total_assets,
           accounts_payable, other_operating_current_liabilities,
           total_current_liabilities, total_debt, total_liabilities,
           shareholders_equity, minority_interest, stated_borrowing_rate,
           source, source_data)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stmt.company_id,
            stmt.fiscal_year,
            stmt.period_end_date,
            stmt.statement_version,
            1 if stmt.restated else 0,
            stmt.source_as_of_date,
            1 if stmt.is_current else 0,
            stmt.reporting_currency,
            stmt.units_scale,
            _to_str(stmt.cash),
            _to_str(stmt.accounts_receivable),
            _to_str(stmt.inventory),
            _to_str(stmt.other_operating_current_assets),
            _to_str(stmt.total_current_assets),
            _to_str(stmt.total_assets),
            _to_str(stmt.accounts_payable),
            _to_str(stmt.other_operating_current_liabilities),
            _to_str(stmt.total_current_liabilities),
            _to_str(stmt.total_debt),
            _to_str(stmt.total_liabilities),
            _to_str(stmt.shareholders_equity),
            _to_str(stmt.minority_interest),
            _to_str(stmt.stated_borrowing_rate),
            stmt.source,
            _to_json(stmt.source_data),
        ),
    )


def get_balance_sheet(conn, company_id, period_end_date, version=None):
    if version is not None:
        row = conn.execute(
            "SELECT * FROM balance_sheets WHERE company_id=? AND period_end_date=? AND statement_version=?",
            (company_id, period_end_date, version),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM balance_sheets WHERE company_id=? AND period_end_date=? AND is_current=1",
            (company_id, period_end_date),
        ).fetchone()
    if row is None:
        return None
    return _row_to_balance(row)


def get_current_balance_sheet(conn, company_id, period_end_date):
    return get_balance_sheet(conn, company_id, period_end_date, version=None)


def get_balance_sheets(conn, company_id, period_range=None):
    query = "SELECT * FROM balance_sheets WHERE company_id=?"
    params = [company_id]
    if period_range and period_range != "all":
        parts = period_range.split("-")
        if len(parts) == 2:
            query += " AND fiscal_year >= ? AND fiscal_year <= ?"
            params.extend([int(parts[0]), int(parts[1])])
    query += " ORDER BY fiscal_year, statement_version"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_balance(r) for r in rows]


# --- Cash Flow Statements ---


def _row_to_cashflow(row) -> CashFlow:
    return CashFlow(
        id=row["id"],
        company_id=row["company_id"],
        fiscal_year=row["fiscal_year"],
        period_end_date=row["period_end_date"],
        statement_version=row["statement_version"],
        restated=bool(row["restated"]),
        source_as_of_date=row["source_as_of_date"],
        is_current=bool(row["is_current"]),
        reporting_currency=row["reporting_currency"],
        units_scale=row["units_scale"],
        operating_cash_flow=_dec(row["operating_cash_flow"]),
        capex=_dec(row["capex"]),
        free_cash_flow=_dec(row["free_cash_flow"]),
        dividends_paid=_dec(row["dividends_paid"]),
        source=row["source"],
        source_data=_parse_json(row["source_data"]),
    )


def insert_cash_flow(conn: sqlite3.Connection, stmt: CashFlow) -> None:
    conn.execute(
        """INSERT INTO cash_flow_statements (company_id, fiscal_year, period_end_date,
           statement_version, restated, source_as_of_date, is_current,
           reporting_currency, units_scale, operating_cash_flow, capex,
           free_cash_flow, dividends_paid, source, source_data)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stmt.company_id,
            stmt.fiscal_year,
            stmt.period_end_date,
            stmt.statement_version,
            1 if stmt.restated else 0,
            stmt.source_as_of_date,
            1 if stmt.is_current else 0,
            stmt.reporting_currency,
            stmt.units_scale,
            _to_str(stmt.operating_cash_flow),
            _to_str(stmt.capex),
            _to_str(stmt.free_cash_flow),
            _to_str(stmt.dividends_paid),
            stmt.source,
            _to_json(stmt.source_data),
        ),
    )


def get_cash_flow(conn, company_id, period_end_date, version=None):
    if version is not None:
        row = conn.execute(
            "SELECT * FROM cash_flow_statements WHERE company_id=? AND period_end_date=? AND statement_version=?",
            (company_id, period_end_date, version),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM cash_flow_statements WHERE company_id=? AND period_end_date=? AND is_current=1",
            (company_id, period_end_date),
        ).fetchone()
    if row is None:
        return None
    return _row_to_cashflow(row)


def get_current_cash_flow(conn, company_id, period_end_date):
    return get_cash_flow(conn, company_id, period_end_date, version=None)


def get_cash_flows(conn, company_id, period_range=None):
    query = "SELECT * FROM cash_flow_statements WHERE company_id=?"
    params = [company_id]
    if period_range and period_range != "all":
        parts = period_range.split("-")
        if len(parts) == 2:
            query += " AND fiscal_year >= ? AND fiscal_year <= ?"
            params.extend([int(parts[0]), int(parts[1])])
    query += " ORDER BY fiscal_year, statement_version"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_cashflow(r) for r in rows]


# --- Documents ---


def insert_document(conn: sqlite3.Connection, doc: Document) -> None:
    conn.execute(
        """INSERT INTO documents (id, company_id, title, doc_type, file_path_or_id,
           page_count, metadata, ingested_at, file_format, storage_path,
           file_size_bytes, content_hash, source_reference, ingestion_status,
           extraction_error, superseded_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc.id,
            doc.company_id,
            doc.title,
            doc.doc_type,
            doc.file_path_or_id,
            doc.page_count,
            _to_json(doc.metadata),
            doc.ingested_at or _now(),
            doc.file_format,
            doc.storage_path,
            doc.file_size_bytes,
            doc.content_hash,
            doc.source_reference,
            doc.ingestion_status or "pending",
            doc.extraction_error,
            doc.superseded_by,
        ),
    )


def get_document(conn: sqlite3.Connection, doc_id: str) -> Optional[Document]:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return None
    return Document(
        id=row["id"],
        company_id=row["company_id"],
        title=row["title"],
        doc_type=row["doc_type"],
        file_path_or_id=row["file_path_or_id"],
        page_count=row["page_count"],
        metadata=_parse_json(row["metadata"]),
        ingested_at=row["ingested_at"],
        file_format=row["file_format"],
        storage_path=row["storage_path"],
        file_size_bytes=row["file_size_bytes"],
        content_hash=row["content_hash"],
        source_reference=row["source_reference"],
        ingestion_status=row["ingestion_status"] or "pending",
        extraction_error=row["extraction_error"],
        superseded_by=row["superseded_by"],
    )


def get_document_by_content_hash(
    conn: sqlite3.Connection, company_id: str, content_hash: str
) -> Optional[Document]:
    row = conn.execute(
        "SELECT * FROM documents WHERE company_id = ? AND content_hash = ?",
        (company_id, content_hash),
    ).fetchone()
    if row is None:
        return None
    return get_document(conn, row["id"])


def get_documents_by_source(
    conn: sqlite3.Connection, company_id: str, source_reference: str
) -> list[Document]:
    rows = conn.execute(
        "SELECT * FROM documents WHERE company_id = ? AND source_reference = ? ORDER BY ingested_at",
        (company_id, source_reference),
    ).fetchall()
    return [get_document(conn, row["id"]) for row in rows]


def update_document_status(
    conn: sqlite3.Connection,
    document_id: str,
    status: str,
    *,
    page_count: Optional[int] = None,
    extraction_error: Optional[str] = None,
    superseded_by: Optional[str] = None,
) -> None:
    conn.execute(
        """UPDATE documents
           SET ingestion_status = ?,
               page_count = COALESCE(?, page_count),
               extraction_error = ?,
               superseded_by = COALESCE(?, superseded_by)
           WHERE id = ?""",
        (status, page_count, extraction_error, superseded_by, document_id),
    )


def mark_documents_superseded(
    conn: sqlite3.Connection, document_ids: list[str], superseded_by: str
) -> None:
    for document_id in document_ids:
        if document_id != superseded_by:
            conn.execute(
                "UPDATE documents SET ingestion_status = 'superseded', superseded_by = ? WHERE id = ?",
                (superseded_by, document_id),
            )


def get_documents(conn: sqlite3.Connection, company_id: str) -> list:
    rows = conn.execute(
        "SELECT * FROM documents WHERE company_id = ? ORDER BY ingested_at", (company_id,)
    ).fetchall()
    return [get_document(conn, r["id"]) for r in rows]


# --- Comparables ---


def _row_to_comparable(row) -> Comparable:
    return Comparable(
        id=row["id"],
        company_id=row["company_id"],
        peer_name=row["peer_name"],
        industry=row["industry"],
        measurement_date=row["measurement_date"],
        currency=row["currency"],
        revenue=to_decimal(row["revenue"]),
        ebitda=to_decimal(row["ebitda"]),
        ebit=to_decimal(row["ebit"]),
        enterprise_value=to_decimal(row["enterprise_value"]),
        equity_value=to_decimal(row["equity_value"]),
        net_debt=to_decimal(row["net_debt"]),
        ev_ebitda=to_decimal(row["ev_ebitda"]),
        pe_ratio=to_decimal(row["pe_ratio"]),
        revenue_growth=to_decimal(row["revenue_growth"]),
        ebitda_margin=to_decimal(row["ebitda_margin"]),
        source=row["source"],
        calculation_method=row["calculation_method"],
        notes=row["notes"],
    )


def insert_comparable(conn: sqlite3.Connection, comp: Comparable) -> None:
    conn.execute(
        """INSERT INTO comparables (company_id, peer_name, industry, measurement_date,
           currency, revenue, ebitda, ebit, enterprise_value, equity_value, net_debt,
           ev_ebitda, pe_ratio, revenue_growth, ebitda_margin, source,
           calculation_method, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            comp.company_id,
            comp.peer_name,
            comp.industry,
            comp.measurement_date,
            comp.currency,
            _to_str(comp.revenue),
            _to_str(comp.ebitda),
            _to_str(comp.ebit),
            _to_str(comp.enterprise_value),
            _to_str(comp.equity_value),
            _to_str(comp.net_debt),
            _to_str(comp.ev_ebitda),
            _to_str(comp.pe_ratio),
            _to_str(comp.revenue_growth),
            _to_str(comp.ebitda_margin),
            comp.source,
            comp.calculation_method,
            comp.notes,
        ),
    )


def get_comparables(conn, company_id, peer_filter=None):
    query = "SELECT * FROM comparables WHERE company_id = ?"
    params = [company_id]
    if peer_filter:
        if "industry" in peer_filter:
            query += " AND industry = ?"
            params.append(peer_filter["industry"])
    query += " ORDER BY peer_name"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_comparable(r) for r in rows]


# --- Restatement helpers ---


def resolve_current_version(conn, company_id, period_end_date, table="income_statements"):
    """Resolve the current statement version for a company/period.

    Returns the row dict or None. Raises DataIntegrityError if multiple
    current versions exist. Raises AmbiguityError if fallback tie unresolved.
    """
    rows = conn.execute(
        f"SELECT * FROM {table} WHERE company_id=? AND period_end_date=?",
        (company_id, period_end_date),
    ).fetchall()

    if not rows:
        return None

    current_rows = [r for r in rows if r["is_current"] == 1]
    if len(current_rows) > 1:
        raise DataIntegrityError(
            f"Data integrity error: multiple current versions for "
            f"{company_id}/{period_end_date} in {table}"
        )
    if len(current_rows) == 1:
        return current_rows[0]

    # No current version: fallback to highest source_as_of_date, then statement_version
    max_date = max(r["source_as_of_date"] for r in rows)
    candidates = [r for r in rows if r["source_as_of_date"] == max_date]
    if len(candidates) == 1:
        return candidates[0]
    max_version = max(r["statement_version"] for r in candidates)
    final = [r for r in candidates if r["statement_version"] == max_version]
    if len(final) > 1:
        raise AmbiguityError(
            f"Ambiguity error: cannot resolve current version for "
            f"{company_id}/{period_end_date} in {table}"
        )
    return final[0]


class DataIntegrityError(ValueError):
    """Raised when multiple current versions exist for the same company/period."""

    pass


class AmbiguityError(ValueError):
    """Raised when statement version cannot be resolved unambiguously."""

    pass
