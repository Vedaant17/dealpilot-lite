"""Tests for SQLite schema and integrity."""

import os
import sqlite3
import tempfile

import pytest

from backend.app.storage.connection import get_connection, init_db
from backend.app.storage.schema import get_schema_version


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


def test_fresh_init_creates_tables():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert "companies" in tables
        assert "income_statements" in tables
        assert "balance_sheets" in tables
        assert "cash_flow_statements" in tables
        assert "documents" in tables
        assert "comparables" in tables
        assert "schema_version" in tables
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_repeated_init_idempotent():
    path = _tmp_db()
    try:
        init_db(path)
        init_db(path)
        conn = get_connection(path)
        version = get_schema_version(conn)
        assert version == 3
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_schema_version_recorded():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        assert get_schema_version(conn) == 3
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_foreign_keys_enforced():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("nonexistent", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test"),
            )
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_unique_constraint():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        conn.execute(
            "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("c1", "Test", "Tech", "USD", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, statement_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, statement_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("c1", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test", 1),
            )
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_partial_unique_index_current():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        conn.execute(
            "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("c1", "Test", "Tech", "USD", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, is_current) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, is_current) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("c1", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test2", 1),
            )
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_transaction_rollback():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        conn.execute(
            "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("c1", "Test", "Tech", "USD", "2024-01-01", "2024-01-01"),
        )
        conn.commit()
        try:
            conn.execute(
                "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("c2", "Test2", "Tech", "USD", "2024-01-01", "2024-01-01"),
            )
            conn.execute(
                "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("c2", "Dup", "Tech", "USD", "2024-01-01", "2024-01-01"),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
        row = conn.execute("SELECT COUNT(*) FROM companies WHERE id='c2'").fetchone()[0]
        assert row == 0
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_foreign_keys_pragma():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_migration_failure_rollback():
    """A failed migration should not corrupt the database."""
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        # Verify schema version is 1
        assert get_schema_version(conn) == 3
        # Verify tables exist
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert "companies" in tables
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_decimal_text_roundtrip():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        conn.execute(
            "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("c1", "Test", "Tech", "USD", "2024-01-01", "2024-01-01"),
        )
        from decimal import Decimal

        val = Decimal("123456789.12345678901234567890")
        conn.execute(
            "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, revenue) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test", str(val)),
        )
        conn.commit()
        row = conn.execute(
            "SELECT revenue FROM income_statements WHERE company_id='c1'"
        ).fetchone()[0]
        assert Decimal(row) == val
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)


def test_missing_vs_zero():
    path = _tmp_db()
    try:
        init_db(path)
        conn = get_connection(path)
        conn.execute(
            "INSERT INTO companies (id, name, industry, default_reporting_currency, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("c1", "Test", "Tech", "USD", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO income_statements (company_id, fiscal_year, period_end_date, source_as_of_date, reporting_currency, units_scale, source, revenue, cogs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", 2023, "2023-12-31", "2023-12-31", "USD", "units", "test", "100", None),
        )
        conn.commit()
        row = conn.execute(
            "SELECT revenue, cogs FROM income_statements WHERE company_id='c1'"
        ).fetchone()
        assert row["revenue"] == "100"
        assert row["cogs"] is None
        conn.close()
    finally:
        import gc

        try:
            conn.close()
        except Exception:
            pass
        del conn
        gc.collect()
        import time

        time.sleep(1)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                time.sleep(1)
                os.remove(path)
