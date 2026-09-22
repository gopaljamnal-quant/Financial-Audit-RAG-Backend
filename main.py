"""
main.py
=======

Main FastAPI application entry point. Wires together lifespan-managed
resources (database engine, MLflow experiment), CORS, global exception
handlers, and the versioned API routers.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import settings
from database import dispose_engine
from routers import health_router, query_router
from schemas import ErrorResponse

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources.

    On startup, logs the resolved configuration profile. On shutdown,
    disposes of the database engine's connection pool cleanly so no
    connections are leaked when the process terminates.

    :param app: The FastAPI application instance.
    :yield: Control back to FastAPI for the lifetime of the application.
    """
    logger.info(
        "Starting %s in %s environment (search_index=%s, chat_deployment=%s)",
        settings.APP_NAME,
        settings.APP_ENV,
        settings.AZURE_SEARCH_INDEX_NAME,
        settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
    )
    yield
    logger.info("Shutting down %s; disposing database engine.", settings.APP_NAME)
    await dispose_engine()


def create_app() -> FastAPI:
    """Construct and fully configure the FastAPI application instance.

    :return: A configured :class:`~fastapi.FastAPI` application, ready
        to be served by an ASGI server such as Uvicorn/Gunicorn.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description="Enterprise-grade, audit-ready Retrieval-Augmented Generation backend.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def add_request_id_header(request: Request, call_next):
        """Attach a correlation id to every request for cross-log traceability.

        :param request: The incoming Starlette request.
        :param call_next: The next handler in the middleware chain.
        :return: The response, with an ``X-Request-ID`` header attached.
        """
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return a structured 422 response for request validation failures.

        :param request: The incoming request that failed validation.
        :param exc: The raised :class:`~fastapi.exceptions.RequestValidationError`.
        :return: A JSON response conforming to :class:`~schemas.ErrorResponse`.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.warning("Validation error [request_id=%s]: %s", request_id, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail="Request validation failed.",
                error_code="VALIDATION_ERROR",
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Return a structured JSON response for raised HTTP exceptions.

        :param request: The incoming request.
        :param exc: The raised :class:`~starlette.exceptions.HTTPException`.
        :return: A JSON response conforming to :class:`~schemas.ErrorResponse`.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.warning("HTTP exception [request_id=%s]: %s", request_id, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=str(exc.detail),
                error_code=f"HTTP_{exc.status_code}",
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler ensuring no unhandled exception ever leaks internals.

        :param request: The incoming request.
        :param exc: The unhandled exception raised during request processing.
        :return: A generic ``500`` JSON response conforming to
            :class:`~schemas.ErrorResponse`, with full details captured
            in the server-side logs (never in the response body).
        """
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception [request_id=%s]", request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An internal error occurred while processing the request.",
                error_code="INTERNAL_SERVER_ERROR",
                request_id=request_id,
            ).model_dump(),
        )

    app.include_router(health_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(query_router.router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
