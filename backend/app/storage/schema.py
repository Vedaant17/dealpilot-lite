"""Schema creation and migration management."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List

MIGRATIONS_DIR = "backend/db/migrations"


def _parse_sql_statements(sql: str) -> List[str]:
    """Split SQL into individual statements, respecting string literals."""
    statements = []
    current = []
    in_string = False
    string_char = None
    for char in sql:
        if in_string:
            current.append(char)
            if char == string_char:
                in_string = False
                string_char = None
        elif char in ("'", '"'):
            in_string = True
            string_char = char
            current.append(char)
        elif char == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)
    remaining = "".join(current).strip()
    if remaining:
        statements.append(remaining)
    return statements


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they do not exist."""
    migration_path = Path(MIGRATIONS_DIR) / "001_initial.sql"
    if not migration_path.exists():
        raise FileNotFoundError(f"Migration file not found: {migration_path}")
    sql = migration_path.read_text(encoding="utf-8")
    statements = _parse_sql_statements(sql)

    conn.execute("BEGIN IMMEDIATE")
    try:
        for stmt in statements:
            if stmt.strip():
                conn.execute(stmt)
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (1, datetime.now(timezone.utc).isoformat()),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version, or 0 if not initialized."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def apply_migrations(conn: sqlite3.Connection, migrations_dir: str = MIGRATIONS_DIR) -> None:
    """Apply pending migrations transactionally and idempotently."""
    migrations_path = Path(migrations_dir)
    if not migrations_path.exists():
        raise FileNotFoundError(f"Migrations directory not found: {migrations_path}")

    current_version = get_schema_version(conn)
    migration_files = sorted(migrations_path.glob("*.sql"))

    for mf in migration_files:
        version = int(mf.stem.split("_")[0])
        if version <= current_version:
            continue

        sql = mf.read_text(encoding="utf-8")
        statements = _parse_sql_statements(sql)

        conn.execute("BEGIN IMMEDIATE")
        try:
            for stmt in statements:
                if stmt.strip():
                    conn.execute(stmt)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
