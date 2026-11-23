# Storage Design

## SQLite Schema

SQLite is the authoritative structured data store. No ORM (SQLAlchemy) or migration tool (Alembic) is used. Schema is managed via sequential SQL migration files in `backend/db/migrations/`.

### Connection Management
- `get_connection(db_path)`: Opens connection, sets `PRAGMA foreign_keys = ON`, `PRAGMA busy_timeout = 5000`, returns `sqlite3.Row` factory.
- `init_db(db_path)`: Explicit initialization. Creates schema if not exists. Idempotent.
- No database is created by importing a module. Initialization is explicit.

### Schema Versioning
- `schema_version` table tracks applied migrations.
- Each migration is a `.sql` file in `backend/db/migrations/` named `NNN_description.sql`.
- Migrations are applied transactionally (BEGIN IMMEDIATE / COMMIT / ROLLBACK).
- `executescript()` is NOT used (it auto-commits, breaking atomicity).
- Repeated initialization does not reapply completed migrations.

### Monetary Precision
- All monetary values stored as TEXT (Decimal strings), not SQLite REAL.
- Conversion via `to_decimal()` from `backend.app.financial.decimal_utils`.
- NaN and infinity rejected on write.
- Missing values stored as NULL (distinguishable from explicit zero).
- Unit normalization applied exactly once during ingestion.

### Source Unit Normalization
- Original source value preserved in `source_data` JSON under `original_values`.
- Original unit preserved under `original_units`.
- Normalization scale preserved under `normalization_factor`.
- Normalized (canonical) value stored in the monetary TEXT column.
- Normalization method recorded under `extraction_method`.
- Repository reads return canonical values without reapplying normalization.

### Statement Versioning
- `UNIQUE(company_id, period_end_date, statement_version)` on each statement table.
- Partial unique index: `WHERE is_current = 1` on `(company_id, period_end_date)`.
- Current-version selection: explicit `is_current` → highest `source_as_of_date` → highest `statement_version` → ambiguity error.
- Restatement insert + current reassignment in one transaction.
- Multiple current versions → `DataIntegrityError`.
- Fallback tie → `AmbiguityError`.

### Currency Policy
- `reporting_currency` on each statement is authoritative.
- `companies.default_reporting_currency` is default preference only.
- No implicit FX conversion.
- Incompatible currencies withhold dependent calculations.

### Document Metadata
- Only document metadata stored in M3. No chunks, embeddings, or vector IDs.
- `documents` table with composite FK support (UNIQUE(id, company_id)).
- Document-chunk and embedding tables deferred to M6 (RAG).

### Comparable Records
- All monetary values and ratios stored as TEXT (Decimal strings), not REAL.
- `calculation_method` distinguishes 'sourced' from 'calculated'.
- EV/EBITDA, P/E, EV/Revenue only stored when underlying inputs available.

### Synthetic Data
- Three fictional companies with 3 years of financial statements.
- All data labeled as synthetic (`is_synthetic = 1`).
- Deterministic seed script with hardcoded values.
- Idempotent seeding.
- Explicit missing values (not zeros) for testing.
- Internal consistency verified (assets = liabilities + equity, EBITDA = EBIT + D&A).
- Aurora has restatement scenario (4 income statements, 3 balance sheets, 3 cash flows).
- Borealis has negative EBIT in 2022 and zero debt in 2023.
- Cedar uses EUR currency and has minority interest.