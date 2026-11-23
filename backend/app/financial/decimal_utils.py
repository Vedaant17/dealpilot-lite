"""Decimal utilities for precise financial calculations."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Optional


@dataclass(frozen=True)
class CalculationResult:
    """Result of a financial calculation.

    ``value`` is the Decimal result when the calculation succeeds.
    ``reason`` is an explanation when the calculation cannot be performed.
    A successful result has a non-None value and reason is None.
    An unavailable result has value None and a non-None reason.
    """

    value: Optional[Decimal] = None
    reason: Optional[str] = None


CALCULATION_PRECISION = 28


def to_decimal(value: Any) -> Optional[Decimal]:
    """Convert a supported input to Decimal, preserving precision.

    Accepted: int, str (numeric), Decimal.
    Floats are converted via str() to avoid binary-float artifacts.
    None returns None. Invalid strings raise ValueError.
    NaN and Infinity are rejected.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite Decimal (NaN or Infinity) is not allowed")
        return value
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid numeric input")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not (value == value):  # NaN check
            raise ValueError("NaN is not allowed")
        if value == float("inf") or value == float("-inf"):
            raise ValueError("Infinity is not allowed")
        return Decimal(str(value))
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            d = Decimal(s)
        except InvalidOperation:
            raise ValueError(f"Invalid numeric string: {value!r}")
        if not d.is_finite():
            raise ValueError("Non-finite value (NaN or Infinity) is not allowed")
        return d
    raise ValueError(f"Unsupported type for Decimal conversion: {type(value).__name__}")


def safe_divide(numerator: Any, denominator: Any) -> CalculationResult:
    """Divide numerator by denominator with controlled precision.

    Returns CalculationResult with value on success, or reason on failure.
    """
    n = to_decimal(numerator)
    d = to_decimal(denominator)
    if n is None or d is None:
        return CalculationResult(reason="missing numerator or denominator")
    if d == 0:
        return CalculationResult(reason="division by zero")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=n / d)


def safe_subtract(a: Any, b: Any) -> CalculationResult:
    """Subtract b from a with controlled precision."""
    da = to_decimal(a)
    db = to_decimal(b)
    if da is None or db is None:
        return CalculationResult(reason="missing operand")
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return CalculationResult(value=da - db)


def round_display(value: Optional[Decimal], places: int = 2) -> Optional[Decimal]:
    """Round a Decimal for display purposes only. Does not modify calculation values."""
    if value is None:
        return None
    with localcontext() as ctx:
        ctx.prec = CALCULATION_PRECISION
        return value.quantize(Decimal(1).scaleb(-places))


def to_string(value: Optional[Decimal]) -> Optional[str]:
    """Serialize Decimal to canonical string. Returns None for None."""
    if value is None:
        return None
    return str(value)
