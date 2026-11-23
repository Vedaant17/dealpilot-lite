"""Screening API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.app.agent.service import AgentService, get_agent_service
from backend.app.config import settings
from backend.app.schemas.screening import (
    EvidenceItem,
    ExecutionMetadata,
    ScreeningRequest,
    ScreeningResponse,
)
from backend.app.storage.connection import get_connection
from backend.app.storage.repositories import get_company

router = APIRouter(prefix="/screenings", tags=["screenings"])


def _get_company_or_404(company_id: str):
    db_path = settings.sqlite_db_path
    if not Path(db_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database is not available"
        )
    conn = get_connection(db_path)
    try:
        company = get_company(conn, company_id)
    finally:
        conn.close()
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return company


@router.post("", response_model=ScreeningResponse)
async def create_screening(
    request: ScreeningRequest,
    service: AgentService = Depends(get_agent_service),
) -> ScreeningResponse:
    company = await run_in_threadpool(_get_company_or_404, request.company_id)
    try:
        result = await service.run_result(
            request.query,
            company_id=request.company_id,
            db_path=settings.sqlite_db_path,
        )
    except Exception as exc:  # noqa: BLE001
        detail = "Agent request failed"
        error_text = str(exc).lower()
        if "timeout" in error_text:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Agent request timed out"
            )
        if "mcp" in error_text or "server" in error_text:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
        if "openai" in error_text or "model" in error_text or "language model" in error_text:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

    if result.status == "failed":
        detail = "Agent request failed"
        error_text = (result.error or "").lower()
        if "timeout" in error_text:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Agent request timed out"
            )
        if "mcp" in error_text or "server" in error_text:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
        if "openai" in error_text or "model" in error_text or "language model" in error_text:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

    evidence = [
        EvidenceItem(
            company_id=item["company_id"],
            document_id=item["document_id"],
            chunk_id=item["chunk_id"],
            chunk_index=item["chunk_index"],
            text_preview=item["text_preview"],
            source=item["source"],
            filename=item["filename"],
            page_start=item["page_start"],
            page_end=item["page_end"],
            similarity_score=item["similarity_score"],
            embedding_model=item["embedding_model"],
        )
        for item in result.retrieved_evidence
    ]

    warnings = [f"Tool error: {message}" for message in result.tool_errors]
    limitations = []
    if result.termination_reason:
        limitations.append(result.termination_reason)
    if result.status == "partial":
        limitations.append("The screening result is partial.")

    tools_used = result.tools_used
    execution = ExecutionMetadata(
        duration_ms=result.duration_ms,
        tool_call_count=result.tool_call_count,
        round_count=result.round_count,
        tools_used=tools_used,
        termination_reason=result.termination_reason,
    )

    disclaimers = [
        "All company data is synthetic and fictional.",
        "Valuation and screening results are illustrative.",
        "This output is decision support, not investment, legal, tax, or financial advice.",
    ]

    return ScreeningResponse(
        company_id=request.company_id,
        company_name=company.name,
        query=request.query,
        status=result.status,
        narrative=result.narrative,
        evidence=evidence,
        citation_warnings=list(result.citation_warnings),
        warnings=warnings,
        limitations=limitations,
        execution=execution,
        disclaimers=disclaimers,
    )
