from decimal import Decimal

from backend.app.financial.multiples import calculate_peer_multiples


class Peer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_peer_multiples_withheld_for_invalid_denominators():
    peer = Peer(
        peer_name="P",
        industry="Tech",
        measurement_date="2023-12-31",
        currency="USD",
        revenue=Decimal("0"),
        ebitda=Decimal("-1"),
        enterprise_value=Decimal("100"),
        equity_value=Decimal("50"),
        source="test",
        calculation_method="sourced",
        ev_ebitda=None,
        pe_ratio=None,
    )
    result = calculate_peer_multiples(peer)
    assert result["calculated"]["ev_ebitda"] is None
    assert result["calculated"]["ev_revenue"] is None
    assert any("EBITDA" in reason for reason in result["reasons"])
    assert any("revenue" in reason for reason in result["reasons"])
