"""Financial input validation utilities."""

from decimal import Decimal
from typing import Any, Optional

from backend.app.financial.decimal_utils import to_decimal


def validate_finite(value: Any) -> Optional[Decimal]:
    """Validate that a value is a finite Decimal. Returns None for None input."""
    if value is None:
        return None
    d = to_decimal(value)
    if d is None:
        return None
    if not d.is_finite():
        raise ValueError("Non-finite value (NaN or Infinity) is not allowed")
    return d


def validate_rate(value: Any, name: str = "rate") -> Optional[Decimal]:
    """Validate a rate in [0, 1]. Returns None for None input."""
    d = validate_finite(value)
    if d is None:
        return None
    if d < 0 or d > 1:
        raise ValueError(f"{name} must be between 0 and 1, got {d}")
    return d


def validate_currency_match(currencies: list) -> bool:
    """Return True if all non-None currencies are the same."""
    non_none = [c for c in currencies if c is not None]
    if not non_none:
        return True
    return len(set(non_none)) == 1


def validate_required(value: Any, name: str) -> None:
    """Raise ValueError if a required value is None."""
    if value is None:
        raise ValueError(f"Required input '{name}' is missing")


def validate_positive(value: Optional[Decimal], name: str) -> None:
    """Raise ValueError if value is not positive (when provided)."""
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_nonnegative(value: Optional[Decimal], name: str) -> None:
    """Raise ValueError if value is negative (when provided)."""
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
