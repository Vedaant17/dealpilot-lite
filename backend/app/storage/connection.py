"""SQLite connection management."""

import sqlite3
from pathlib import Path


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with FK enforcement and busy timeout."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Explicitly initialize the database schema. Idempotent."""
    from backend.app.storage.schema import apply_migrations, create_schema

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        create_schema(conn)
        apply_migrations(conn, "backend/db/migrations")
        conn.commit()
    finally:
        conn.close()


def close_connection(conn: sqlite3.Connection) -> None:
    """Close a SQLite connection."""
    conn.close()
