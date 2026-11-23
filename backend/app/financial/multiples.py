"""Comparable multiples helpers for DealPilot Lite."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Any, Optional

from backend.app.financial.decimal_utils import to_decimal


def _str(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else str(value)


def calculate_peer_multiples(peer: Any) -> dict[str, Any]:
    """Return sourced and calculated peer multiples with unavailability reasons."""
    revenue = to_decimal(getattr(peer, "revenue", None))
    ebitda = to_decimal(getattr(peer, "ebitda", None))
    enterprise_value = to_decimal(getattr(peer, "enterprise_value", None))
    equity_value = to_decimal(getattr(peer, "equity_value", None))
    earnings = to_decimal(
        getattr(peer, "ebit", None)
    )  # M3 peer earnings are not stored; EBIT is the closest available proxy.

    reasons: list[str] = []

    calculated_ev_ebitda = None
    if enterprise_value is not None and ebitda is not None:
        if ebitda <= 0:
            reasons.append("EV/EBITDA withheld: EBITDA is zero or negative")
        else:
            with localcontext() as ctx:
                ctx.prec = 28
                calculated_ev_ebitda = enterprise_value / ebitda

    calculated_ev_revenue = None
    if enterprise_value is not None and revenue is not None:
        if revenue <= 0:
            reasons.append("EV/Revenue withheld: revenue is zero or negative")
        else:
            with localcontext() as ctx:
                ctx.prec = 28
                calculated_ev_revenue = enterprise_value / revenue

    calculated_pe = None
    if equity_value is not None and earnings is not None:
        if earnings <= 0:
            reasons.append("P/E withheld: earnings are zero or negative")
        else:
            with localcontext() as ctx:
                ctx.prec = 28
                calculated_pe = equity_value / earnings

    return {
        "peer_name": getattr(peer, "peer_name", None),
        "industry": getattr(peer, "industry", None),
        "measurement_date": getattr(peer, "measurement_date", None),
        "currency": getattr(peer, "currency", None),
        "source": getattr(peer, "source", None),
        "calculation_method": getattr(peer, "calculation_method", None),
        "sourced": {
            "ev_ebitda": _str(getattr(peer, "ev_ebitda", None)),
            "pe_ratio": _str(getattr(peer, "pe_ratio", None)),
            "revenue_growth": _str(getattr(peer, "revenue_growth", None)),
            "ebitda_margin": _str(getattr(peer, "ebitda_margin", None)),
        },
        "calculated": {
            "ev_ebitda": _str(calculated_ev_ebitda),
            "ev_revenue": _str(calculated_ev_revenue),
            "pe_ratio": _str(calculated_pe),
        },
        "underlying": {
            "revenue": _str(revenue),
            "ebitda": _str(ebitda),
            "enterprise_value": _str(enterprise_value),
            "equity_value": _str(equity_value),
        },
        "reasons": reasons,
    }
