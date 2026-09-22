"""
routers/query_router.py
=========================

The primary RAG query endpoint. Wires together hybrid retrieval
(:mod:`services.search_service`), grounded generation
(:mod:`services.llm_orchestrator`), the audit-trail repository
(:mod:`repositories.audit_repository`), and MLflow telemetry
(:mod:`services.mlflow_tracker`) behind a single, strictly-typed
FastAPI endpoint.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db_session
from repositories.audit_repository import AuditRepository
from schemas import QueryRequest, QueryResponse
from services.llm_orchestrator import LLMOrchestrationError, LLMOrchestrator
from services.mlflow_tracker import MLflowTracker
from services.search_service import SearchService, SearchServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    """Provide a process-wide singleton :class:`SearchService`.

    Reusing a single instance avoids reconstructing HTTP clients (and
    their connection pools) on every request.

    :return: The shared :class:`SearchService` instance.
    """
    return SearchService()


@lru_cache(maxsize=1)
def get_llm_orchestrator() -> LLMOrchestrator:
    """Provide a process-wide singleton :class:`LLMOrchestrator`.

    :return: The shared :class:`LLMOrchestrator` instance.
    """
    return LLMOrchestrator()


@lru_cache(maxsize=1)
def get_mlflow_tracker() -> MLflowTracker:
    """Provide a process-wide singleton :class:`MLflowTracker`.

    :return: The shared :class:`MLflowTracker` instance.
    """
    return MLflowTracker()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a RAG query and receive a fully-cited, audit-logged answer",
)
async def submit_query(
    payload: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    search_service: SearchService = Depends(get_search_service),
    llm_orchestrator: LLMOrchestrator = Depends(get_llm_orchestrator),
    mlflow_tracker: MLflowTracker = Depends(get_mlflow_tracker),
) -> QueryResponse:
    """Execute the full retrieve -> generate -> audit -> track pipeline.

    :param payload: The validated :class:`~schemas.QueryRequest`.
    :param session: Injected async database session (transactional per request).
    :param search_service: Injected singleton hybrid search service.
    :param llm_orchestrator: Injected singleton LLM orchestration service.
    :param mlflow_tracker: Injected singleton MLflow telemetry service.
    :raises HTTPException: ``502`` if retrieval or generation fails
        against the upstream Azure services; ``500`` for unexpected
        persistence failures.
    :return: A fully-populated :class:`~schemas.QueryResponse` including
        the generated answer and its exact source citations.
    """
    pipeline_start = time.perf_counter()

    try:
        retrieval_start = time.perf_counter()
        retrieved_chunks = await search_service.hybrid_search(
            query=payload.query,
            role=payload.role,
            top_k=payload.top_k,
            document_filter=payload.document_filter,
        )
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)
    except SearchServiceError as exc:
        logger.exception("Retrieval stage failed for user_id=%s", payload.user_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document retrieval service is currently unavailable.",
        ) from exc

    try:
        generation_start = time.perf_counter()
        generation_result = await llm_orchestrator.generate_answer(
            query=payload.query,
            retrieved_chunks=retrieved_chunks,
        )
        generation_latency_ms = int((time.perf_counter() - generation_start) * 1000)
    except LLMOrchestrationError as exc:
        logger.exception("Generation stage failed for user_id=%s", payload.user_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Answer generation service is currently unavailable.",
        ) from exc

    total_latency_ms = int((time.perf_counter() - pipeline_start) * 1000)

    try:
        audit_repo = AuditRepository(session)
        user_query_record = await audit_repo.create_user_query(
            user_id=payload.user_id,
            query_text=payload.query,
            response_text=generation_result.answer,
            latency_ms=total_latency_ms,
        )
        await audit_repo.create_audit_logs(
            user_query_id=user_query_record.id,
            citations=generation_result.citations,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean 500 to the caller
        logger.exception("Audit persistence failed for user_id=%s", payload.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist the audit trail for this query.",
        ) from exc

    await mlflow_tracker.log_query_execution(
        user_id=payload.user_id,
        query=payload.query,
        role=payload.role.value,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        num_retrieved_chunks=len(retrieved_chunks),
        relevance_scores=[chunk.score for chunk in retrieved_chunks],
        prompt_tokens=generation_result.prompt_tokens,
        completion_tokens=generation_result.completion_tokens,
        model=generation_result.model,
        citations=generation_result.citations,
    )

    return QueryResponse(
        query_id=user_query_record.id,
        answer=generation_result.answer,
        citations=generation_result.citations,
        latency_ms=total_latency_ms,
        model=generation_result.model,
        created_at=datetime.now(timezone.utc),
    )
