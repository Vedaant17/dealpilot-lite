from decimal import Decimal

import pytest

from backend.app.financial.valuation import calculate_valuation
from backend.app.schemas.mcp_tools import ValuationAssumptions
from backend.app.storage.connection import get_connection
from backend.scripts.seed_db import seed_database


@pytest.fixture
def seeded_conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


def test_valuation_requires_growth(seeded_conn):
    assumptions = ValuationAssumptions(
        revenue_growth_rate=None, revenue_growth_rates=None, cost_of_equity=Decimal("0.10")
    )
    # schema allows both None; service must reject
    result = calculate_valuation(seeded_conn, "aurora-retail", "2023", assumptions)
    assert result["available"] is False
    assert any("growth" in reason.lower() for reason in result["reasons"])


def test_valuation_aurora(seeded_conn):
    assumptions = ValuationAssumptions(
        revenue_growth_rate=Decimal("0.05"), cost_of_equity=Decimal("0.10")
    )
    result = calculate_valuation(seeded_conn, "aurora-retail", "2023", assumptions)
    assert result["available"] is True
    assert result["currency"] == "USD"
    assert result["enterprise_value"] is not None
    assert result["equity_value"] is not None
    assert result["per_share_value"] is not None


def test_valuation_zero_debt_borealis(seeded_conn):
    assumptions = ValuationAssumptions(
        revenue_growth_rate=Decimal("0.03"), cost_of_equity=Decimal("0.12")
    )
    result = calculate_valuation(seeded_conn, "borealis-industrial", "2023", assumptions)
    assert result["available"] is True
    assert result["wacc"]["value"] is not None


def test_valuation_missing_base_nwc_does_not_invalidate_unrelated_period(seeded_conn):
    # Aurora 2023 has complete NWC; valuation remains available.
    assumptions = ValuationAssumptions(
        revenue_growth_rate=Decimal("0.05"), cost_of_equity=Decimal("0.10")
    )
    result = calculate_valuation(seeded_conn, "aurora-retail", "2023", assumptions)
    assert result["available"] is True
