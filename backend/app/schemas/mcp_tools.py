"""Pydantic contracts for M4 MCP tools."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.financial.decimal_utils import to_decimal


def _decimal(value: Any, name: str) -> Decimal:
    result = to_decimal(value)
    if result is None:
        raise ValueError(f"{name} must be a finite number")
    return result


def _optional_decimal(value: Any, name: str) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, name)


class PeriodRange(BaseModel):
    value: str = Field(default="all", pattern=r"^(all|\d{4}|\d{4}-\d{4})$")


class MetricValue(BaseModel):
    value: str | None = None
    reason: str | None = None
    units: str | None = None
    currency: str | None = None


class ValuationAssumptions(BaseModel):
    revenue_growth_rate: Decimal | None = Field(default=None, ge=-0.99, le=10.0)
    revenue_growth_rates: list[Decimal] | None = None
    operating_margin: Decimal | None = Field(default=None, ge=-1.0, le=1.0)
    operating_tax_rate: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    da_pct_revenue: Decimal = Field(default=Decimal("0.03"), ge=0, le=1)
    capex_pct_revenue: Decimal = Field(default=Decimal("0.04"), ge=0, le=1)
    nwc_pct_revenue: Decimal = Field(default=Decimal("0.10"), ge=0, le=1)
    terminal_growth: Decimal = Field(default=Decimal("0.02"), ge=0, le=1)
    forecast_horizon: int = Field(default=5, ge=1, le=10)
    direct_wacc: Decimal | None = Field(default=None, ge=0, le=1)
    cost_of_equity: Decimal | None = Field(default=None, ge=0, le=1)
    cost_of_debt: Decimal | None = Field(default=None, ge=0, le=1)
    interest_tax_shield_rate: Decimal | None = Field(default=None, ge=0, le=1)
    other_adjustments: Decimal = Decimal(0)
    forecast_nwc_values: list[Decimal] | None = None
    growth_warning_threshold: Decimal = Field(default=Decimal("0.50"), ge=0, le=10)

    @field_validator(
        "revenue_growth_rate",
        "operating_margin",
        "operating_tax_rate",
        "da_pct_revenue",
        "capex_pct_revenue",
        "nwc_pct_revenue",
        "terminal_growth",
        "direct_wacc",
        "cost_of_equity",
        "cost_of_debt",
        "interest_tax_shield_rate",
        "other_adjustments",
        "growth_warning_threshold",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, value, info):
        if value is None:
            return None
        return _decimal(value, info.field_name)

    @field_validator("revenue_growth_rates", "forecast_nwc_values", mode="before")
    @classmethod
    def _coerce_growth_list(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("value must be a list")
        return [_decimal(item, "forecast override") for item in value]

    @model_validator(mode="after")
    def _validate_growth_inputs(self):
        has_constant = self.revenue_growth_rate is not None
        has_list = self.revenue_growth_rates is not None
        if has_constant and has_list:
            raise ValueError(
                "only one of revenue_growth_rate or revenue_growth_rates may be supplied"
            )
        if has_constant == has_list and has_constant is False:
            # Both are absent: allowed at schema level; service layer requires one.
            pass
        if has_list:
            if len(self.revenue_growth_rates) != self.forecast_horizon:
                raise ValueError("revenue_growth_rates length must equal forecast_horizon")
        return self

    @model_validator(mode="after")
    def _validate_wacc_inputs(self):
        if self.direct_wacc is None and self.cost_of_equity is None:
            # Component WACC may be unavailable at runtime; this is not a schema error.
            pass
        if self.direct_wacc is not None and self.terminal_growth >= self.direct_wacc:
            raise ValueError("direct_wacc must be greater than terminal_growth")
        return self


class ComparableFilter(BaseModel):
    industry: str | None = None
    peer_name: str | None = None


class SensitivityRanges(BaseModel):
    wacc_min: Decimal = Field(ge=0, le=1)
    wacc_max: Decimal = Field(ge=0, le=1)
    wacc_step: Decimal = Field(gt=0, le=1)
    terminal_growth_min: Decimal = Field(ge=0, le=1)
    terminal_growth_max: Decimal = Field(ge=0, le=1)
    terminal_growth_step: Decimal = Field(gt=0, le=1)
    max_pairs: int = Field(default=100, ge=1, le=100)

    @field_validator(
        "wacc_min",
        "wacc_max",
        "wacc_step",
        "terminal_growth_min",
        "terminal_growth_max",
        "terminal_growth_step",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, value, info):
        return _decimal(value, info.field_name)

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.wacc_min > self.wacc_max:
            raise ValueError("wacc_min must be <= wacc_max")
        if self.terminal_growth_min > self.terminal_growth_max:
            raise ValueError("terminal_growth_min must be <= terminal_growth_max")
        wacc_count = int((self.wacc_max - self.wacc_min) / self.wacc_step) + 1
        growth_count = (
            int((self.terminal_growth_max - self.terminal_growth_min) / self.terminal_growth_step)
            + 1
        )
        if wacc_count * growth_count > self.max_pairs:
            raise ValueError("sensitivity grid exceeds max_pairs")
        # Validate every generated WACC is greater than every generated growth.
        wacc_values = [self.wacc_min + self.wacc_step * i for i in range(wacc_count)]
        growth_values = [
            self.terminal_growth_min + self.terminal_growth_step * i for i in range(growth_count)
        ]
        for wacc in wacc_values:
            for growth in growth_values:
                if not wacc > growth:
                    raise ValueError("every WACC must be greater than every terminal growth")
        return self


class ForecastOverrides(BaseModel):
    revenue_growth_rates: list[Decimal] | None = None
    operating_margin: Decimal | None = Field(default=None, ge=-1.0, le=1.0)
    da_pct_revenue: Decimal | None = Field(default=None, ge=0, le=1)
    capex_pct_revenue: Decimal | None = Field(default=None, ge=0, le=1)
    nwc_pct_revenue: Decimal | None = Field(default=None, ge=0, le=1)
    forecast_nwc_values: list[Decimal] | None = None

    @field_validator(
        "operating_margin",
        "da_pct_revenue",
        "capex_pct_revenue",
        "nwc_pct_revenue",
        mode="before",
    )
    @classmethod
    def _coerce_optional_decimal(cls, value, info):
        if value is None:
            return None
        return _decimal(value, info.field_name)

    @field_validator("revenue_growth_rates", "forecast_nwc_values", mode="before")
    @classmethod
    def _coerce_optional_list(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("value must be a list")
        return [_decimal(item, "forecast override") for item in value]


class ToolResultEnvelope(BaseModel):
    tool: str
    status: Literal["ok", "unavailable", "error"] = "ok"
    data: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
