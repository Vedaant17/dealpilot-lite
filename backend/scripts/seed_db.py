"""Deterministic synthetic data seed script for DealPilot Lite.

Seeds a database with synthetic company and financial data.
Idempotent: skips seeding if companies already exist.
Uses temporary databases during testing; persistent DB only when explicitly authorized.
"""

import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    insert_balance_sheet,
    insert_cash_flow,
    insert_company,
    insert_comparable,
    insert_document,
    insert_income_statement,
)

SOURCE = "synthetic_seed_v1"
NOW = datetime.now(timezone.utc).isoformat()

# --- Company definitions ---

COMPANIES = [
    Company(
        id="aurora-retail",
        name="Aurora Retail Solutions",
        industry="Retail",
        description="Mid-market retail chain with steady growth",
        founded_year=2010,
        headquarters="Chicago, IL",
        default_reporting_currency="USD",
        shares_outstanding=Decimal("10000000"),
        shares_outstanding_as_of_date="2023-12-31",
        shares_outstanding_source="synthetic_seed",
        is_synthetic=True,
        created_at=NOW,
        updated_at=NOW,
    ),
    Company(
        id="borealis-industrial",
        name="Borealis Industrial Components",
        industry="Manufacturing",
        description="Industrial components manufacturer, cyclical demand",
        founded_year=2005,
        headquarters="Detroit, MI",
        default_reporting_currency="USD",
        shares_outstanding=None,
        is_synthetic=True,
        created_at=NOW,
        updated_at=NOW,
    ),
    Company(
        id="cedar-health",
        name="Cedar Health Analytics",
        industry="Health Technology",
        description="Health analytics SaaS platform, high growth",
        founded_year=2018,
        headquarters="Boston, MA",
        default_reporting_currency="EUR",
        shares_outstanding=Decimal("5000000"),
        shares_outstanding_as_of_date="2023-12-31",
        shares_outstanding_source="synthetic_seed",
        is_synthetic=True,
        created_at=NOW,
        updated_at=NOW,
    ),
]

# --- Financial data (all values in whole currency units, stored as TEXT Decimal) ---

AURORA_DATA = {
    "2021": {
        "revenue": "50000000",
        "cogs": "32000000",
        "gross_profit": "18000000",
        "operating_expenses": "12000000",
        "ebitda": "6000000",
        "da": "1000000",
        "ebit": "5000000",
        "interest": "2500000",
        "taxes": "625000",
        "net_income": "1875000",
        "cash": "5000000",
        "ar": "4000000",
        "inventory": None,
        "other_oca": "1000000",
        "total_ca": "10000000",
        "total_assets": "44000000",
        "ap": "3000000",
        "other_ocl": "1000000",
        "total_cl": "4000000",
        "total_debt": "15000000",
        "total_liab": "19000000",
        "equity": "25000000",
        "minority": "0",
        "ocf": "7000000",
        "capex": "1500000",
        "fcf": "5500000",
    },
    "2022": {
        "revenue": "55000000",
        "cogs": "34000000",
        "gross_profit": "21000000",
        "operating_expenses": "13500000",
        "ebitda": "7500000",
        "da": "1100000",
        "ebit": "6400000",
        "interest": "2000000",
        "taxes": "1100000",
        "net_income": "3300000",
        "cash": "6000000",
        "ar": "4500000",
        "inventory": "3000000",
        "other_oca": "1200000",
        "total_ca": "14700000",
        "total_assets": "44700000",
        "ap": "3500000",
        "other_ocl": "1200000",
        "total_cl": "4700000",
        "total_debt": "12000000",
        "total_liab": "16700000",
        "equity": "28000000",
        "minority": "0",
        "ocf": "8500000",
        "capex": "1650000",
        "fcf": "6850000",
    },
    "2023": {
        "revenue": "60000000",
        "cogs": "36000000",
        "gross_profit": "24000000",
        "operating_expenses": "14400000",
        "ebitda": "9600000",
        "da": "1200000",
        "ebit": "8400000",
        "interest": "1800000",
        "taxes": "1680000",
        "net_income": "4920000",
        "cash": "8000000",
        "ar": "5000000",
        "inventory": "3500000",
        "other_oca": "1500000",
        "total_ca": "18000000",
        "total_assets": "47500000",
        "ap": "4000000",
        "other_ocl": "1500000",
        "total_cl": "5500000",
        "total_debt": "10000000",
        "total_liab": "15500000",
        "equity": "32000000",
        "minority": "0",
        "ocf": "10000000",
        "capex": "1800000",
        "fcf": "8200000",
    },
}

# Aurora 2022 restated: revenue 53M (restated from 55M)
# Dependent values recalculated for consistency
AURORA_2022_RESTATE = {
    "revenue": "53000000",
    "cogs": "32500000",
    "gross_profit": "20500000",
    "operating_expenses": "13000000",
    "ebitda": "7500000",
    "da": "1100000",
    "ebit": "6400000",
    "interest": "2000000",
    "taxes": "1100000",
    "net_income": "3300000",
    "cash": "6000000",
    "ar": "4500000",
    "inventory": "3000000",
    "other_oca": "1200000",
    "total_ca": "14700000",
    "total_assets": "35000000",
    "ap": "3500000",
    "other_ocl": "1200000",
    "total_cl": "4700000",
    "total_debt": "12000000",
    "total_liab": "16700000",
    "equity": "28000000",
    "minority": "0",
    "ocf": "8500000",
    "capex": "1650000",
    "fcf": "6850000",
}

BOREALIS_DATA = {
    "2021": {
        "revenue": "80000000",
        "cogs": "52000000",
        "gross_profit": "28000000",
        "operating_expenses": "13600000",
        "ebitda": "14400000",
        "da": "3200000",
        "ebit": "11200000",
        "interest": None,
        "taxes": "2800000",
        "net_income": "8400000",
        "cash": "3000000",
        "ar": "6000000",
        "inventory": "8000000",
        "other_oca": "2000000",
        "total_ca": "19000000",
        "total_assets": "57000000",
        "ap": "5000000",
        "other_ocl": "2000000",
        "total_cl": "7000000",
        "total_debt": "30000000",
        "total_liab": "37000000",
        "equity": "20000000",
        "minority": "0",
        "ocf": "12000000",
        "capex": "4000000",
        "fcf": "8000000",
        "borrowing_rate": "0.065",
    },
    "2022": {
        "revenue": "72000000",
        "cogs": "50000000",
        "gross_profit": "22000000",
        "operating_expenses": "16240000",
        "ebitda": "5760000",
        "da": "7200000",
        "ebit": "-1440000",
        "interest": "4500000",
        "taxes": "0",
        "net_income": "-5940000",
        "cash": "2000000",
        "ar": "5500000",
        "inventory": "7000000",
        "other_oca": "1500000",
        "total_ca": "16000000",
        "total_assets": "46000000",
        "ap": "4500000",
        "other_ocl": "1500000",
        "total_cl": "6000000",
        "total_debt": "25000000",
        "total_liab": "31000000",
        "equity": "15000000",
        "minority": "0",
        "ocf": "6000000",
        "capex": "3600000",
        "fcf": "2400000",
        "borrowing_rate": "0.075",
    },
    "2023": {
        "revenue": "85000000",
        "cogs": "55000000",
        "gross_profit": "30000000",
        "operating_expenses": "13000000",
        "ebitda": "17000000",
        "da": "3400000",
        "ebit": "13600000",
        "interest": "0",
        "taxes": "3400000",
        "net_income": "10200000",
        "cash": "5000000",
        "ar": "7000000",
        "inventory": "9000000",
        "other_oca": "2500000",
        "total_ca": "23500000",
        "total_assets": "30500000",
        "ap": "6000000",
        "other_ocl": "2500000",
        "total_cl": "8500000",
        "total_debt": "0",
        "total_liab": "8500000",
        "equity": "22000000",
        "minority": "0",
        "ocf": "15000000",
        "capex": "4250000",
        "fcf": "10750000",
        "borrowing_rate": None,
    },
}

CEDAR_DATA = {
    "2021": {
        "revenue": "10000000",
        "cogs": "4000000",
        "gross_profit": "6000000",
        "operating_expenses": "3500000",
        "ebitda": "2500000",
        "da": "300000",
        "ebit": "2200000",
        "interest": "500000",
        "taxes": "550000",
        "net_income": "1150000",
        "cash": "3000000",
        "ar": "1500000",
        "inventory": "0",
        "other_oca": "500000",
        "total_ca": "5000000",
        "total_assets": "11500000",
        "ap": "1000000",
        "other_ocl": "500000",
        "total_cl": "1500000",
        "total_debt": "2000000",
        "total_liab": "3500000",
        "equity": "8000000",
        "minority": "0",
        "ocf": "2800000",
        "capex": "600000",
        "fcf": "2200000",
    },
    "2022": {
        "revenue": "18000000",
        "cogs": "6500000",
        "gross_profit": "11500000",
        "operating_expenses": "6460000",
        "ebitda": "5040000",
        "da": "540000",
        "ebit": "4500000",
        "interest": "500000",
        "taxes": "1125000",
        "net_income": "2875000",
        "cash": "8000000",
        "ar": "2500000",
        "inventory": "0",
        "other_oca": "1000000",
        "total_ca": "11500000",
        "total_assets": "20500000",
        "ap": "1500000",
        "other_ocl": "1000000",
        "total_cl": "2500000",
        "total_debt": "2000000",
        "total_liab": "4500000",
        "equity": "15000000",
        "minority": "1000000",
        "ocf": "5500000",
        "capex": "1080000",
        "fcf": "4420000",
    },
    "2023": {
        "revenue": "30000000",
        "cogs": "10000000",
        "gross_profit": "20000000",
        "operating_expenses": "10400000",
        "ebitda": "9600000",
        "da": "900000",
        "ebit": "8700000",
        "interest": "500000",
        "taxes": "2175000",
        "net_income": "6025000",
        "cash": "15000000",
        "ar": "4000000",
        "inventory": "0",
        "other_oca": "2000000",
        "total_ca": "21000000",
        "total_assets": "32000000",
        "ap": "2000000",
        "other_ocl": "2000000",
        "total_cl": "4000000",
        "total_debt": "2000000",
        "total_liab": "6000000",
        "equity": "25000000",
        "minority": "1000000",
        "ocf": "9500000",
        "capex": "1800000",
        "fcf": "7700000",
    },
}

AURORA_PEERS = [
    ("RetailMart Inc", "Retail", "5.2", "12.5", "0.08", "0.11"),
    ("ShopDirect LLC", "Retail", "4.8", "11.0", "0.06", "0.10"),
    ("MegaStore Corp", "Retail", "6.0", "14.0", "0.10", "0.13"),
]
BOREALIS_PEERS = [
    ("SteelForge Ltd", "Manufacturing", "7.5", "15.0", "0.05", "0.16"),
    ("MetalWorks Co", "Manufacturing", "6.0", "12.0", "0.03", "0.12"),
    ("PrecisionParts Inc", "Manufacturing", "8.0", "16.0", "0.07", "0.18"),
]
CEDAR_PEERS = [
    ("MedData Systems", "Health Technology", "12.0", "35.0", "0.40", "0.22"),
    ("ClinicalAI Corp", "Health Technology", "15.0", "40.0", "0.50", "0.25"),
    ("HealthMetrics LLC", "Health Technology", "10.0", "30.0", "0.35", "0.20"),
]


def _d(val):
    return Decimal(val) if val is not None else None


def seed_database(db_path):
    """Seed a database with synthetic data. Idempotent."""
    conn = get_connection(db_path)
    try:
        init_db(db_path)
        existing = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        if existing > 0:
            conn.close()
            return

        for c in COMPANIES:
            insert_company(conn, c)

        _seed_company(conn, "aurora-retail", AURORA_DATA, "USD", "thousands")
        _insert_restated_aurora(conn)
        _seed_company(conn, "borealis-industrial", BOREALIS_DATA, "USD", "thousands")
        _seed_company(conn, "cedar-health", CEDAR_DATA, "EUR", "thousands")

        _seed_comparables(conn, "aurora-retail", AURORA_PEERS, "USD", "2023-12-31")
        _seed_comparables(conn, "borealis-industrial", BOREALIS_PEERS, "USD", "2023-12-31")
        _seed_comparables(conn, "cedar-health", CEDAR_PEERS, "EUR", "2023-12-31")

        _seed_documents(conn)
        conn.commit()
    finally:
        conn.close()


def _seed_company(conn, company_id, data, currency, units):
    for year_str, d in data.items():
        year = int(year_str)
        period_end = f"{year}-12-31"
        source_as_of = f"{year}-12-31"
        src_json = json.dumps(
            {
                "original_values": {
                    k: v for k, v in d.items() if v is not None and k not in ("borrowing_rate",)
                },
                "original_units": units,
                "normalization_factor": 1000,
                "normalized_values": {
                    k: v for k, v in d.items() if v is not None and k not in ("borrowing_rate",)
                },
                "value_type": "reported",
                "source_reference": "synthetic_seed_v1",
                "extraction_method": "manual_seed",
            }
        )

        insert_income_statement(
            conn,
            IncomeStatement(
                company_id=company_id,
                fiscal_year=year,
                period_end_date=period_end,
                statement_version=1,
                restated=False,
                source_as_of_date=source_as_of,
                is_current=True,
                reporting_currency=currency,
                units_scale=units,
                revenue=_d(d["revenue"]),
                cogs=_d(d["cogs"]),
                gross_profit=_d(d["gross_profit"]),
                operating_expenses=_d(d["operating_expenses"]),
                ebitda=_d(d["ebitda"]),
                depreciation_amortization=_d(d["da"]),
                operating_income=_d(d["ebit"]),
                interest_expense=_d(d["interest"]),
                taxes=_d(d["taxes"]),
                net_income=_d(d["net_income"]),
                source=SOURCE,
                source_data=json.loads(src_json),
            ),
        )

        insert_balance_sheet(
            conn,
            BalanceSheet(
                company_id=company_id,
                fiscal_year=year,
                period_end_date=period_end,
                statement_version=1,
                restated=False,
                source_as_of_date=source_as_of,
                is_current=True,
                reporting_currency=currency,
                units_scale=units,
                cash=_d(d["cash"]),
                accounts_receivable=_d(d["ar"]),
                inventory=_d(d["inventory"]),
                other_operating_current_assets=_d(d["other_oca"]),
                total_current_assets=_d(d["total_ca"]),
                total_assets=_d(d["total_assets"]),
                accounts_payable=_d(d["ap"]),
                other_operating_current_liabilities=_d(d["other_ocl"]),
                total_current_liabilities=_d(d["total_cl"]),
                total_debt=_d(d["total_debt"]),
                total_liabilities=_d(d["total_liab"]),
                shareholders_equity=_d(d["equity"]),
                minority_interest=_d(d["minority"]),
                stated_borrowing_rate=_d(d.get("borrowing_rate")),
                source=SOURCE,
                source_data=json.loads(src_json),
            ),
        )

        insert_cash_flow(
            conn,
            CashFlow(
                company_id=company_id,
                fiscal_year=year,
                period_end_date=period_end,
                statement_version=1,
                restated=False,
                source_as_of_date=source_as_of,
                is_current=True,
                reporting_currency=currency,
                units_scale=units,
                operating_cash_flow=_d(d["ocf"]),
                capex=_d(d["capex"]),
                free_cash_flow=_d(d["fcf"]),
                dividends_paid=None,
                source=SOURCE,
                source_data=json.loads(src_json),
            ),
        )


def _insert_restated_aurora(conn):
    d = AURORA_2022_RESTATE
    period_end = "2022-12-31"
    source_as_of = "2023-06-30"
    src_json = json.dumps(
        {
            "original_values": {"revenue": "55000000"},
            "original_units": "thousands",
            "normalization_factor": 1000,
            "normalized_values": {"revenue": "53000000"},
            "value_type": "reported",
            "source_reference": "synthetic_seed_v1_restated",
            "extraction_method": "manual_seed",
        }
    )
    conn.execute(
        "UPDATE income_statements SET is_current=0 WHERE company_id=? AND period_end_date=?",
        ("aurora-retail", period_end),
    )
    insert_income_statement(
        conn,
        IncomeStatement(
            company_id="aurora-retail",
            fiscal_year=2022,
            period_end_date=period_end,
            statement_version=2,
            restated=True,
            source_as_of_date=source_as_of,
            is_current=True,
            reporting_currency="USD",
            units_scale="thousands",
            revenue=_d(d["revenue"]),
            cogs=_d(d["cogs"]),
            gross_profit=_d(d["gross_profit"]),
            operating_expenses=_d(d["operating_expenses"]),
            ebitda=_d(d["ebitda"]),
            depreciation_amortization=_d(d["da"]),
            operating_income=_d(d["ebit"]),
            interest_expense=_d(d["interest"]),
            taxes=_d(d["taxes"]),
            net_income=_d(d["net_income"]),
            source=SOURCE + "_restated",
            source_data=json.loads(src_json),
        ),
    )


def _seed_comparables(conn, company_id, peers, currency, measurement_date):
    for peer_name, industry, ev_ebitda, pe, rev_growth, ebitda_margin in peers:
        insert_comparable(
            conn,
            Comparable(
                company_id=company_id,
                peer_name=peer_name,
                industry=industry,
                measurement_date=measurement_date,
                currency=currency,
                ev_ebitda=_d(ev_ebitda),
                pe_ratio=_d(pe),
                revenue_growth=_d(rev_growth),
                ebitda_margin=_d(ebitda_margin),
                source=SOURCE,
                calculation_method="sourced",
                notes="Synthetic comparable data",
            ),
        )


def _seed_documents(conn):
    docs = [
        (
            "aurora-overview",
            "aurora-retail",
            "Aurora Retail Solutions — Company Overview",
            "overview",
            "aurora_overview.txt",
            2,
        ),
        (
            "aurora-financials",
            "aurora-retail",
            "Aurora Retail Solutions — Financial Summary",
            "financial_summary",
            "aurora_financials.txt",
            3,
        ),
        (
            "aurora-risks",
            "aurora-retail",
            "Aurora Retail Solutions — Risk Memo",
            "risk_memo",
            "aurora_risks.txt",
            2,
        ),
        (
            "aurora-mgmt",
            "aurora-retail",
            "Aurora Retail Solutions — Management Discussion",
            "management_discussion",
            "aurora_mgmt.txt",
            4,
        ),
        (
            "aurora-contradiction",
            "aurora-retail",
            "Aurora Retail Solutions — Supplemental Notes",
            "supplemental",
            "aurora_contradiction.txt",
            1,
        ),
        (
            "borealis-overview",
            "borealis-industrial",
            "Borealis Industrial Components — Company Overview",
            "overview",
            "borealis_overview.txt",
            2,
        ),
        (
            "borealis-financials",
            "borealis-industrial",
            "Borealis Industrial Components — Financial Summary",
            "financial_summary",
            "borealis_financials.txt",
            3,
        ),
        (
            "borealis-risks",
            "borealis-industrial",
            "Borealis Industrial Components — Risk Memo",
            "risk_memo",
            "borealis_risks.txt",
            2,
        ),
        (
            "borealis-mgmt",
            "borealis-industrial",
            "Borealis Industrial Components — Management Discussion",
            "management_discussion",
            "borealis_mgmt.txt",
            4,
        ),
        (
            "borealis-contradiction",
            "borealis-industrial",
            "Borealis Industrial Components — Supplemental Notes",
            "supplemental",
            "borealis_contradiction.txt",
            1,
        ),
        (
            "cedar-overview",
            "cedar-health",
            "Cedar Health Analytics — Company Overview",
            "overview",
            "cedar_overview.txt",
            2,
        ),
        (
            "cedar-financials",
            "cedar-health",
            "Cedar Health Analytics — Financial Summary",
            "financial_summary",
            "cedar_financials.txt",
            3,
        ),
        (
            "cedar-risks",
            "cedar-health",
            "Cedar Health Analytics — Risk Memo",
            "risk_memo",
            "cedar_risks.txt",
            2,
        ),
        (
            "cedar-mgmt",
            "cedar-health",
            "Cedar Health Analytics — Management Discussion",
            "management_discussion",
            "cedar_mgmt.txt",
            4,
        ),
        (
            "cedar-contradiction",
            "cedar-health",
            "Cedar Health Analytics — Supplemental Notes",
            "supplemental",
            "cedar_contradiction.txt",
            1,
        ),
    ]
    for doc_id, company_id, title, doc_type, file_path, page_count in docs:
        insert_document(
            conn,
            Document(
                id=doc_id,
                company_id=company_id,
                title=title,
                doc_type=doc_type,
                file_path_or_id=file_path,
                page_count=page_count,
                metadata={"synthetic": True},
                ingested_at=NOW,
            ),
        )


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/dealpilot.db"
    seed_database(db_path)
    print(f"Seeded database: {db_path}")
