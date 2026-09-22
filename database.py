"""
database.py
============

Asynchronous SQLAlchemy 2.0 engine initialisation and session lifecycle
management. Exposes a single :func:`get_db_session` FastAPI dependency
that yields a transactional :class:`~sqlalchemy.ext.asyncio.AsyncSession`
and guarantees commit/rollback/close semantics on every request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in :mod:`models`."""


def _build_engine() -> AsyncEngine:
    """Construct the process-wide asynchronous SQLAlchemy engine.

    :return: A configured :class:`~sqlalchemy.ext.asyncio.AsyncEngine`
        with pooling parameters sourced from application settings.
    """
    return create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        future=True,
    )


engine: AsyncEngine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Create all tables declared against :class:`Base`'s metadata.

    Intended for local development and CI bootstrap only. Production
    deployments should manage schema migrations via Alembic rather than
    calling this function against a live database.

    :raises sqlalchemy.exc.SQLAlchemyError: If the DDL statements fail
        to execute against the configured database.
    """
    import models  # noqa: F401  (ensures models are registered on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialised for %s", settings.APP_NAME)


async def dispose_engine() -> None:
    """Dispose of the async engine's connection pool on application shutdown.

    :raises sqlalchemy.exc.SQLAlchemyError: If pool disposal fails.
    """
    await engine.dispose()
    logger.info("Database engine connection pool disposed cleanly.")


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional database session.

    Commits the transaction on successful completion of the request
    scope, rolls back on any exception, and always closes the session
    to return the underlying connection to the pool.

    :yield: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
    :raises Exception: Re-raises any exception encountered by the
        caller after performing a rollback, so upstream error handlers
        retain full visibility of the original failure.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database session rolled back due to an unhandled exception.")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context-manager variant of :func:`get_db_session` for non-request code paths.

    Useful inside background tasks, MLflow telemetry hooks, or startup
    scripts where FastAPI's dependency injection is not available.

    :yield: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
