"""Tests for repository functions."""

import os
import sqlite3
import tempfile
from decimal import Decimal

import pytest

from backend.app.storage.connection import get_connection, init_db
from backend.app.storage.models import (
    BalanceSheet,
    CashFlow,
    Company,
    Comparable,
    Document,
    IncomeStatement,
)
from backend.app.storage.repositories import (
    get_balance_sheets,
    get_company,
    get_comparables,
    get_current_balance_sheet,
    get_current_cash_flow,
    get_current_income_statement,
    get_document,
    get_documents,
    get_income_statement,
    get_income_statements,
    insert_balance_sheet,
    insert_cash_flow,
    insert_company,
    insert_comparable,
    insert_document,
    insert_income_statement,
    insert_restated_statement,
    update_company,
)


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


@pytest.fixture
def db():
    path = _tmp_db()
    init_db(path)
    conn = get_connection(path)
    yield conn, path
    conn.close()
    if os.path.exists(path):
        os.remove(path)


def _seed_company(conn):
    insert_company(
        conn,
        Company(
            id="test-co",
            name="Test Co",
            industry="Tech",
            default_reporting_currency="USD",
            is_synthetic=True,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        ),
    )


class TestCompanyRepo:
    def test_insert_and_get(self, db):
        conn, _ = db
        _seed_company(conn)
        c = get_company(conn, "test-co")
        assert c is not None
        assert c.name == "Test Co"
        assert c.default_reporting_currency == "USD"

    def test_update(self, db):
        conn, _ = db
        _seed_company(conn)
        c = get_company(conn, "test-co")
        c.name = "Updated Co"
        update_company(conn, c)
        assert get_company(conn, "test-co").name == "Updated Co"

    def test_missing(self, db):
        conn, _ = db
        assert get_company(conn, "nonexistent") is None


class TestIncomeStatementRepo:
    def test_insert_and_get_current(self, db):
        conn, _ = db
        _seed_company(conn)
        stmt = IncomeStatement(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="thousands",
            revenue=Decimal("50000"),
            net_income=Decimal("5000"),
            source="test",
        )
        insert_income_statement(conn, stmt)
        result = get_current_income_statement(conn, "test-co", "2023-12-31")
        assert result is not None
        assert result.revenue == Decimal("50000")
        assert result.net_income == Decimal("5000")

    def test_decimal_roundtrip(self, db):
        conn, _ = db
        _seed_company(conn)
        stmt = IncomeStatement(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            revenue=Decimal("123456789.12345678901234567890"),
            source="test",
        )
        insert_income_statement(conn, stmt)
        result = get_current_income_statement(conn, "test-co", "2023-12-31")
        assert result.revenue == Decimal("123456789.12345678901234567890")

    def test_missing_value_preserved(self, db):
        conn, _ = db
        _seed_company(conn)
        stmt = IncomeStatement(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            revenue=Decimal("100"),
            cogs=None,
            source="test",
        )
        insert_income_statement(conn, stmt)
        result = get_current_income_statement(conn, "test-co", "2023-12-31")
        assert result.revenue == Decimal("100")
        assert result.cogs is None

    def test_period_range(self, db):
        conn, _ = db
        _seed_company(conn)
        for yr in [2021, 2022, 2023]:
            insert_income_statement(
                conn,
                IncomeStatement(
                    company_id="test-co",
                    fiscal_year=yr,
                    period_end_date=f"{yr}-12-31",
                    source_as_of_date=f"{yr}-12-31",
                    is_current=True,
                    reporting_currency="USD",
                    units_scale="units",
                    revenue=Decimal(str(yr * 1000)),
                    source="test",
                ),
            )
        results = get_income_statements(conn, "test-co", "2022-2023")
        assert len(results) == 2
        assert results[0].fiscal_year == 2022

    def test_get_by_version(self, db):
        conn, _ = db
        _seed_company(conn)
        insert_income_statement(
            conn,
            IncomeStatement(
                company_id="test-co",
                fiscal_year=2022,
                period_end_date="2022-12-31",
                statement_version=1,
                restated=False,
                source_as_of_date="2022-12-31",
                is_current=True,
                reporting_currency="USD",
                units_scale="units",
                revenue=Decimal("55000"),
                source="test",
            ),
        )
        result = get_income_statement(conn, "test-co", "2022-12-31", version=1)
        assert result is not None
        assert result.revenue == Decimal("55000")


class TestRestatement:
    def test_restated_version(self, db):
        conn, _ = db
        _seed_company(conn)
        insert_income_statement(
            conn,
            IncomeStatement(
                company_id="test-co",
                fiscal_year=2022,
                period_end_date="2022-12-31",
                statement_version=1,
                restated=False,
                source_as_of_date="2022-12-31",
                is_current=True,
                reporting_currency="USD",
                units_scale="units",
                revenue=Decimal("55000"),
                source="test",
            ),
        )
        stmt2 = IncomeStatement(
            company_id="test-co",
            fiscal_year=2022,
            period_end_date="2022-12-31",
            statement_version=2,
            restated=True,
            source_as_of_date="2023-06-30",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            revenue=Decimal("53000"),
            source="test_restated",
        )
        insert_restated_statement(conn, stmt2, 2)
        v1 = get_income_statement(conn, "test-co", "2022-12-31", version=1)
        assert v1 is not None
        assert v1.revenue == Decimal("55000")
        assert v1.is_current is False
        v2 = get_current_income_statement(conn, "test-co", "2022-12-31")
        assert v2 is not None
        assert v2.revenue == Decimal("53000")
        assert v2.is_current is True
        assert v2.restated is True

    def test_failed_restatement_rollback(self, db):
        conn, _ = db
        _seed_company(conn)
        insert_income_statement(
            conn,
            IncomeStatement(
                company_id="test-co",
                fiscal_year=2022,
                period_end_date="2022-12-31",
                statement_version=1,
                restated=False,
                source_as_of_date="2022-12-31",
                is_current=True,
                reporting_currency="USD",
                units_scale="units",
                revenue=Decimal("55000"),
                source="test",
            ),
        )
        bad_stmt = IncomeStatement(
            company_id="test-co",
            fiscal_year=2022,
            period_end_date="2022-12-31",
            statement_version=1,
            restated=True,
            source_as_of_date="2023-06-30",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            revenue=Decimal("53000"),
            source="test_bad",
        )
        try:
            insert_restated_statement(conn, bad_stmt, 1)
        except Exception:
            pass
        v1 = get_current_income_statement(conn, "test-co", "2022-12-31")
        assert v1 is not None
        assert v1.revenue == Decimal("55000")
        assert v1.is_current is True

    def test_multiple_current_raises_integrity(self, db):
        conn, _ = db
        _seed_company(conn)
        conn.execute(
            "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("c2", "Test2", "Tech", "USD", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, is_current) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c2", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, is_current) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("c2", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test2", 1),
            )


class TestBalanceSheetRepo:
    def test_insert_and_get(self, db):
        conn, _ = db
        _seed_company(conn)
        bs = BalanceSheet(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            cash=Decimal("5000"),
            total_assets=Decimal("30000"),
            shareholders_equity=Decimal("20000"),
            source="test",
        )
        insert_balance_sheet(conn, bs)
        result = get_current_balance_sheet(conn, "test-co", "2023-12-31")
        assert result is not None
        assert result.cash == Decimal("5000")
        assert result.total_assets == Decimal("30000")

    def test_missing_value(self, db):
        conn, _ = db
        _seed_company(conn)
        bs = BalanceSheet(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            cash=Decimal("5000"),
            inventory=None,
            source="test",
        )
        insert_balance_sheet(conn, bs)
        result = get_current_balance_sheet(conn, "test-co", "2023-12-31")
        assert result.cash == Decimal("5000")
        assert result.inventory is None

    def test_period_range(self, db):
        conn, _ = db
        _seed_company(conn)
        for yr in [2021, 2022, 2023]:
            insert_balance_sheet(
                conn,
                BalanceSheet(
                    company_id="test-co",
                    fiscal_year=yr,
                    period_end_date=f"{yr}-12-31",
                    source_as_of_date=f"{yr}-12-31",
                    is_current=True,
                    reporting_currency="USD",
                    units_scale="units",
                    total_assets=Decimal(str(yr * 10000)),
                    source="test",
                ),
            )
        results = get_balance_sheets(conn, "test-co", "2022-2023")
        assert len(results) == 2


class TestCashFlowRepo:
    def test_insert_and_get(self, db):
        conn, _ = db
        _seed_company(conn)
        cf = CashFlow(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            operating_cash_flow=Decimal("8000"),
            capex=Decimal("2000"),
            free_cash_flow=Decimal("6000"),
            source="test",
        )
        insert_cash_flow(conn, cf)
        result = get_current_cash_flow(conn, "test-co", "2023-12-31")
        assert result is not None
        assert result.operating_cash_flow == Decimal("8000")
        assert result.capex == Decimal("2000")

    def test_missing_value(self, db):
        conn, _ = db
        _seed_company(conn)
        cf = CashFlow(
            company_id="test-co",
            fiscal_year=2023,
            period_end_date="2023-12-31",
            source_as_of_date="2023-12-31",
            is_current=True,
            reporting_currency="USD",
            units_scale="units",
            operating_cash_flow=Decimal("8000"),
            capex=None,
            source="test",
        )
        insert_cash_flow(conn, cf)
        result = get_current_cash_flow(conn, "test-co", "2023-12-31")
        assert result.operating_cash_flow == Decimal("8000")
        assert result.capex is None


class TestDocumentRepo:
    def test_insert_and_get(self, db):
        conn, _ = db
        _seed_company(conn)
        doc = Document(
            id="doc1",
            company_id="test-co",
            title="Test Doc",
            doc_type="overview",
            file_path_or_id="test.txt",
            page_count=2,
            metadata={"synthetic": True},
            ingested_at="2024-01-01T00:00:00",
        )
        insert_document(conn, doc)
        result = get_document(conn, "doc1")
        assert result is not None
        assert result.title == "Test Doc"
        assert result.metadata == {"synthetic": True}

    def test_company_ownership(self, db):
        conn, _ = db
        _seed_company(conn)
        with pytest.raises(sqlite3.IntegrityError):
            insert_document(
                conn,
                Document(
                    id="doc2",
                    company_id="nonexistent",
                    title="Bad Doc",
                    doc_type="overview",
                    file_path_or_id="bad.txt",
                    ingested_at="2024-01-01",
                ),
            )

    def test_get_documents_by_company(self, db):
        conn, _ = db
        _seed_company(conn)
        insert_document(
            conn,
            Document(
                id="d1",
                company_id="test-co",
                title="Doc1",
                doc_type="overview",
                file_path_or_id="d1.txt",
                ingested_at="2024-01-01",
            ),
        )
        insert_document(
            conn,
            Document(
                id="d2",
                company_id="test-co",
                title="Doc2",
                doc_type="risk",
                file_path_or_id="d2.txt",
                ingested_at="2024-01-02",
            ),
        )
        docs = get_documents(conn, "test-co")
        assert len(docs) == 2
        assert docs[0].title == "Doc1"


class TestComparableRepo:
    def test_insert_and_get(self, db):
        conn, _ = db
        _seed_company(conn)
        comp = Comparable(
            company_id="test-co",
            peer_name="Peer Inc",
            industry="Tech",
            measurement_date="2023-12-31",
            currency="USD",
            revenue=Decimal("10000000"),
            ebitda=Decimal("2000000"),
            ev_ebitda=Decimal("5.5"),
            source="test",
            calculation_method="sourced",
        )
        insert_comparable(conn, comp)
        results = get_comparables(conn, "test-co")
        assert len(results) == 1
        assert results[0].peer_name == "Peer Inc"
        assert results[0].revenue == Decimal("10000000")
        assert results[0].ev_ebitda == Decimal("5.5")

    def test_decimal_preserved(self, db):
        conn, _ = db
        _seed_company(conn)
        comp = Comparable(
            company_id="test-co",
            peer_name="Peer2",
            industry="Tech",
            measurement_date="2023-12-31",
            currency="USD",
            ev_ebitda=Decimal("12.3456789"),
            pe_ratio=Decimal("25.5"),
            source="test",
            calculation_method="sourced",
        )
        insert_comparable(conn, comp)
        results = get_comparables(conn, "test-co")
        assert results[0].ev_ebitda == Decimal("12.3456789")
        assert results[0].pe_ratio == Decimal("25.5")

    def test_peer_filter(self, db):
        conn, _ = db
        _seed_company(conn)
        insert_comparable(
            conn,
            Comparable(
                company_id="test-co",
                peer_name="P1",
                industry="Tech",
                measurement_date="2023-12-31",
                currency="USD",
                source="t",
                calculation_method="sourced",
            ),
        )
        insert_comparable(
            conn,
            Comparable(
                company_id="test-co",
                peer_name="P2",
                industry="Finance",
                measurement_date="2023-12-31",
                currency="USD",
                source="t",
                calculation_method="sourced",
            ),
        )
        results = get_comparables(conn, "test-co", peer_filter={"industry": "Tech"})
        assert len(results) == 1
        assert results[0].peer_name == "P1"
