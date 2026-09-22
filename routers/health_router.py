"""
routers/health_router.py
==========================

Liveness/readiness probe endpoints, exposed unauthenticated for use by
container orchestrators (Kubernetes, Azure Container Apps, etc.).
"""

from __future__ import annotations

from fastapi import APIRouter

from config import settings
from schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness probe")
async def health_check() -> HealthResponse:
    """Report basic service liveness.

    :return: A :class:`~schemas.HealthResponse` indicating the service is up.
    """
    return HealthResponse(status="ok", app_name=settings.APP_NAME, app_env=settings.APP_ENV)
