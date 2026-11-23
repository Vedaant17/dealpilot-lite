"""Pydantic schemas for the M8 screening API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScreeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def _trim_and_validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class EvidenceItem(BaseModel):
    company_id: str
    document_id: str
    chunk_id: int
    chunk_index: int
    text_preview: str
    source: str
    filename: str
    page_start: int
    page_end: int
    similarity_score: float
    embedding_model: str


class ExecutionMetadata(BaseModel):
    duration_ms: int
    tool_call_count: int
    round_count: int
    tools_used: list[str]
    termination_reason: str | None = None


class ScreeningResponse(BaseModel):
    company_id: str
    company_name: str
    query: str
    status: Literal["completed", "partial", "failed"]
    narrative: str
    evidence: list[EvidenceItem]
    citation_warnings: list[str]
    warnings: list[str]
    limitations: list[str]
    execution: ExecutionMetadata
    disclaimers: list[str]
