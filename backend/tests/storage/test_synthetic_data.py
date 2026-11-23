"""Tests for synthetic data integrity and seed script."""

import os
import tempfile
from decimal import Decimal

import pytest

from backend.app.financial.decimal_utils import to_decimal
from backend.app.storage.connection import get_connection
from backend.scripts.seed_db import seed_database


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


@pytest.fixture
def seeded_db():
    path = _tmp_db()
    seed_database(path)
    conn = get_connection(path)
    yield conn, path
    conn.close()
    if os.path.exists(path):
        os.remove(path)


class TestSyntheticData:
    def test_company_count(self, seeded_db):
        conn, _ = seeded_db
        count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        assert count == 3

    def test_company_ids(self, seeded_db):
        conn, _ = seeded_db
        ids = [r[0] for r in conn.execute("SELECT id FROM companies ORDER BY id").fetchall()]
        assert ids == ["aurora-retail", "borealis-industrial", "cedar-health"]

    def test_all_synthetic(self, seeded_db):
        conn, _ = seeded_db
        rows = conn.execute("SELECT is_synthetic FROM companies").fetchall()
        assert all(r[0] == 1 for r in rows)

    def test_income_statement_counts(self, seeded_db):
        conn, _ = seeded_db
        aurora = conn.execute(
            "SELECT COUNT(*) FROM income_statements WHERE company_id='aurora-retail'"
        ).fetchone()[0]
        borealis = conn.execute(
            "SELECT COUNT(*) FROM income_statements WHERE company_id='borealis-industrial'"
        ).fetchone()[0]
        cedar = conn.execute(
            "SELECT COUNT(*) FROM income_statements WHERE company_id='cedar-health'"
        ).fetchone()[0]
        assert aurora == 4  # 3 periods + 1 restated
        assert borealis == 3
        assert cedar == 3

    def test_balance_sheet_counts(self, seeded_db):
        conn, _ = seeded_db
        for cid in ["aurora-retail", "borealis-industrial", "cedar-health"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM balance_sheets WHERE company_id=?", (cid,)
            ).fetchone()[0]
            assert count == 3, f"{cid} has {count} balance sheets (expected 3)"

    def test_cash_flow_counts(self, seeded_db):
        conn, _ = seeded_db
        for cid in ["aurora-retail", "borealis-industrial", "cedar-health"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM cash_flow_statements WHERE company_id=?", (cid,)
            ).fetchone()[0]
            assert count == 3, f"{cid} has {count} cash flows (expected 3)"

    def test_distinct_reporting_periods(self, seeded_db):
        conn, _ = seeded_db
        for cid in ["aurora-retail", "borealis-industrial", "cedar-health"]:
            periods = conn.execute(
                "SELECT COUNT(DISTINCT period_end_date) FROM income_statements WHERE company_id=?",
                (cid,),
            ).fetchone()[0]
            assert periods == 3, f"{cid} has {periods} distinct periods (expected 3)"

    def test_aurora_restated_revenue(self, seeded_db):
        conn, _ = seeded_db
        v1 = conn.execute(
            "SELECT revenue FROM income_statements WHERE company_id='aurora-retail' AND fiscal_year=2022 AND statement_version=1"
        ).fetchone()[0]
        v2 = conn.execute(
            "SELECT revenue FROM income_statements WHERE company_id='aurora-retail' AND fiscal_year=2022 AND statement_version=2"
        ).fetchone()[0]
        assert v1 == "55000000"
        assert v2 == "53000000"

    def test_aurora_current_version(self, seeded_db):
        conn, _ = seeded_db
        cur = conn.execute(
            "SELECT statement_version FROM income_statements WHERE company_id='aurora-retail' AND fiscal_year=2022 AND is_current=1"
        ).fetchone()[0]
        assert cur == 2

    def test_aurora_v1_retained(self, seeded_db):
        conn, _ = seeded_db
        v1 = conn.execute(
            "SELECT COUNT(*) FROM income_statements WHERE company_id='aurora-retail' AND fiscal_year=2022 AND statement_version=1"
        ).fetchone()[0]
        assert v1 == 1

    def test_aurora_restated_consistency(self, seeded_db):
        conn, _ = seeded_db
        # Verify restated version has independently consistent values
        v2 = conn.execute(
            "SELECT revenue, ebitda, depreciation_amortization, operating_income, net_income FROM income_statements WHERE company_id='aurora-retail' AND fiscal_year=2022 AND statement_version=2"
        ).fetchone()
        assert v2["revenue"] == "53000000"
        assert v2["ebitda"] == "7500000"
        assert v2["depreciation_amortization"] == "1100000"
        assert v2["operating_income"] == "6400000"
        # EBIT = EBITDA - D&A
        assert Decimal(v2["operating_income"]) == Decimal(v2["ebitda"]) - Decimal(
            v2["depreciation_amortization"]
        )

    def test_borealis_zero_debt_2023(self, seeded_db):
        conn, _ = seeded_db
        row = conn.execute(
            "SELECT total_debt, stated_borrowing_rate, interest_expense FROM balance_sheets bs JOIN income_statements i ON bs.company_id=i.company_id AND bs.period_end_date=i.period_end_date WHERE bs.company_id='borealis-industrial' AND bs.fiscal_year=2023"
        ).fetchone()
        assert row["total_debt"] == "0"
        assert row["stated_borrowing_rate"] is None
        assert row["interest_expense"] == "0"

    def test_borealis_negative_ebit_2022(self, seeded_db):
        conn, _ = seeded_db
        row = conn.execute(
            "SELECT ebitda, depreciation_amortization, operating_income, interest_expense, taxes, net_income FROM income_statements WHERE company_id='borealis-industrial' AND fiscal_year=2022"
        ).fetchone()
        ebitda = Decimal(row["ebitda"])
        da = Decimal(row["depreciation_amortization"])
        ebit = Decimal(row["operating_income"])
        tax = Decimal(row["taxes"])
        net = Decimal(row["net_income"])
        assert ebit == ebitda - da
        assert ebit < 0
        assert tax == 0  # no tax benefit
        interest = Decimal(row["interest_expense"])
        assert net == ebit - interest  # net income = EBIT - interest

    def test_borealis_danda_consistency(self, seeded_db):
        conn, _ = seeded_db
        rows = conn.execute(
            "SELECT fiscal_year, ebitda, depreciation_amortization, operating_income FROM income_statements WHERE company_id='borealis-industrial'"
        ).fetchall()
        for r in rows:
            if (
                r["ebitda"] is not None
                and r["depreciation_amortization"] is not None
                and r["operating_income"] is not None
            ):
                assert Decimal(r["operating_income"]) == Decimal(r["ebitda"]) - Decimal(
                    r["depreciation_amortization"]
                )

    def test_cedar_minority_interest(self, seeded_db):
        conn, _ = seeded_db
        row2021 = conn.execute(
            "SELECT minority_interest FROM balance_sheets WHERE company_id='cedar-health' AND fiscal_year=2021"
        ).fetchone()[0]
        row2022 = conn.execute(
            "SELECT minority_interest FROM balance_sheets WHERE company_id='cedar-health' AND fiscal_year=2022"
        ).fetchone()[0]
        assert row2021 == "0"
        assert row2022 == "1000000"

    def test_missing_values_preserved(self, seeded_db):
        conn, _ = seeded_db
        row = conn.execute(
            "SELECT inventory FROM balance_sheets WHERE company_id='aurora-retail' AND fiscal_year=2021"
        ).fetchone()[0]
        assert row is None
        row = conn.execute(
            "SELECT interest_expense FROM income_statements WHERE company_id='borealis-industrial' AND fiscal_year=2021"
        ).fetchone()[0]
        assert row is None

    def test_currency_consistency(self, seeded_db):
        conn, _ = seeded_db
        for cid, expected in [
            ("aurora-retail", "USD"),
            ("borealis-industrial", "USD"),
            ("cedar-health", "EUR"),
        ]:
            currencies = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT reporting_currency FROM income_statements WHERE company_id=?",
                    (cid,),
                ).fetchall()
            ]
            assert currencies == [expected], f"{cid} has currencies {currencies}"
            currencies = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT reporting_currency FROM balance_sheets WHERE company_id=?",
                    (cid,),
                ).fetchall()
            ]
            assert currencies == [expected]
            currencies = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT reporting_currency FROM cash_flow_statements WHERE company_id=?",
                    (cid,),
                ).fetchall()
            ]
            assert currencies == [expected]

    def test_balance_sheet_equation(self, seeded_db):
        conn, _ = seeded_db
        rows = conn.execute(
            "SELECT company_id, fiscal_year, total_assets, total_liabilities, shareholders_equity, minority_interest FROM balance_sheets"
        ).fetchall()
        for r in rows:
            assets = Decimal(r["total_assets"])
            liab = Decimal(r["total_liabilities"])
            equity = Decimal(r["shareholders_equity"])
            minority = Decimal(r["minority_interest"]) if r["minority_interest"] else Decimal(0)
            assert abs(assets - (liab + equity + minority)) < Decimal("0.01"), (
                f"{r['company_id']} {r['fiscal_year']}: assets={assets} != liab+equity+minority={liab + equity + minority}"
            )

    def test_ebitda_consistency(self, seeded_db):
        conn, _ = seeded_db
        rows = conn.execute(
            "SELECT company_id, fiscal_year, ebitda, depreciation_amortization, operating_income FROM income_statements"
        ).fetchall()
        for r in rows:
            if (
                r["ebitda"] is not None
                and r["depreciation_amortization"] is not None
                and r["operating_income"] is not None
            ):
                ebitda = Decimal(r["ebitda"])
                da = Decimal(r["depreciation_amortization"])
                ebit = Decimal(r["operating_income"])
                assert ebit == ebitda - da, (
                    f"{r['company_id']} {r['fiscal_year']}: ebit={ebit} != ebitda-da={ebitda - da}"
                )

    def test_idempotent_seed(self, seeded_db):
        conn, path = seeded_db
        conn.close()
        from backend.scripts.seed_db import seed_database

        seed_database(path)
        conn2 = get_connection(path)
        count = conn2.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        assert count == 3
        conn2.close()

    def test_deterministic_seed(self):
        path1 = _tmp_db()
        path2 = _tmp_db()
        seed_database(path1)
        seed_database(path2)
        conn1 = get_connection(path1)
        conn2 = get_connection(path2)
        rows1 = conn1.execute("SELECT * FROM companies ORDER BY id").fetchall()
        rows2 = conn2.execute("SELECT * FROM companies ORDER BY id").fetchall()
        assert [dict(r) for r in rows1] == [dict(r) for r in rows2]
        conn1.close()
        conn2.close()
        os.remove(path1)
        os.remove(path2)

    def test_comparables_count(self, seeded_db):
        conn, _ = seeded_db
        for cid in ["aurora-retail", "borealis-industrial", "cedar-health"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM comparables WHERE company_id=?", (cid,)
            ).fetchone()[0]
            assert count == 3, f"{cid} has {count} comparables (expected 3)"

    def test_documents_count(self, seeded_db):
        conn, _ = seeded_db
        for cid in ["aurora-retail", "borealis-industrial", "cedar-health"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE company_id=?", (cid,)
            ).fetchone()[0]
            assert count == 5, f"{cid} has {count} documents (expected 5)"

    def test_decimal_roundtrip(self, seeded_db):
        conn, _ = seeded_db
        row = conn.execute(
            "SELECT revenue FROM income_statements WHERE company_id='aurora-retail' AND fiscal_year=2023"
        ).fetchone()[0]
        val = to_decimal(row)
        assert val == Decimal("60000000")
        assert str(val) == "60000000"

    def test_comparables_decimal_preserved(self, seeded_db):
        conn, _ = seeded_db
        row = conn.execute(
            "SELECT ev_ebitda FROM comparables WHERE company_id='aurora-retail' LIMIT 1"
        ).fetchone()[0]
        val = to_decimal(row)
        assert val == Decimal("5.2")

    def test_shares_outstanding(self, seeded_db):
        conn, _ = seeded_db
        row = conn.execute(
            "SELECT shares_outstanding, shares_outstanding_source FROM companies WHERE id='aurora-retail'"
        ).fetchone()
        assert row["shares_outstanding"] == "10000000"
        assert row["shares_outstanding_source"] == "synthetic_seed"
        row = conn.execute(
            "SELECT shares_outstanding FROM companies WHERE id='borealis-industrial'"
        ).fetchone()
        assert row["shares_outstanding"] is None

    def test_borealis_borrowing_rate(self, seeded_db):
        conn, _ = seeded_db
        row2021 = conn.execute(
            "SELECT stated_borrowing_rate FROM balance_sheets WHERE company_id='borealis-industrial' AND fiscal_year=2021"
        ).fetchone()[0]
        row2022 = conn.execute(
            "SELECT stated_borrowing_rate FROM balance_sheets WHERE company_id='borealis-industrial' AND fiscal_year=2022"
        ).fetchone()[0]
        row2023 = conn.execute(
            "SELECT stated_borrowing_rate FROM balance_sheets WHERE company_id='borealis-industrial' AND fiscal_year=2023"
        ).fetchone()[0]
        assert row2021 == "0.065"
        assert row2022 == "0.075"
        assert row2023 is None

    def test_no_zero_substitution(self, seeded_db):
        conn, _ = seeded_db
        # Aurora 2021 inventory is NULL (not zero)
        row = conn.execute(
            "SELECT inventory FROM balance_sheets WHERE company_id='aurora-retail' AND fiscal_year=2021"
        ).fetchone()[0]
        assert row is None
        # Cedar 2021 inventory is "0" (explicit zero, not missing)
        row = conn.execute(
            "SELECT inventory FROM balance_sheets WHERE company_id='cedar-health' AND fiscal_year=2021"
        ).fetchone()[0]
        assert row == "0"
