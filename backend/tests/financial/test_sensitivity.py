from decimal import Decimal

from backend.app.financial.sensitivity import run_sensitivity_analysis
from backend.app.schemas.mcp_tools import SensitivityRanges, ValuationAssumptions
from backend.app.storage.connection import get_connection
from backend.scripts.seed_db import seed_database


def test_sensitivity_grid_validation():
    try:
        SensitivityRanges(
            wacc_min=Decimal("0.10"),
            wacc_max=Decimal("0.10"),
            wacc_step=Decimal("0.01"),
            terminal_growth_min=Decimal("0.11"),
            terminal_growth_max=Decimal("0.11"),
            terminal_growth_step=Decimal("0.01"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid grid accepted")


def test_sensitivity_runs(tmp_path):
    db_path = str(tmp_path / "test.db")
    seed_database(db_path)
    conn = get_connection(db_path)
    assumptions = ValuationAssumptions(
        revenue_growth_rate=Decimal("0.05"), cost_of_equity=Decimal("0.10")
    )
    ranges = SensitivityRanges(
        wacc_min=Decimal("0.10"),
        wacc_max=Decimal("0.10"),
        wacc_step=Decimal("0.01"),
        terminal_growth_min=Decimal("0.02"),
        terminal_growth_max=Decimal("0.02"),
        terminal_growth_step=Decimal("0.01"),
    )
    result = run_sensitivity_analysis(conn, "aurora-retail", "2023", assumptions, ranges)
    assert result["scenario_count"] == 1
    assert result["scenarios"][0]["available"] is True
    conn.close()
