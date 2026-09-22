"""
repositories/audit_repository.py
==================================

Repository layer encapsulating all direct database access for the
audit trail: persisting user queries and their supporting citations.
Keeping this isolated from :mod:`services` and :mod:`routers` enforces
a clean separation of concerns and gives the audit-log write path a
single, testable seam.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from models import AuditLog, UserQuery
from schemas import CitationSchema

logger = logging.getLogger(__name__)


class AuditRepository:
    """Persists user queries and their citation audit trail.

    :ivar session: The bound :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an active database session.

        :param session: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`,
            typically injected via the :func:`database.get_db_session` dependency.
        """
        self.session = session

    async def create_user_query(
        self,
        user_id: str,
        query_text: str,
        response_text: str,
        latency_ms: int,
    ) -> UserQuery:
        """Persist a completed user query and its generated response.

        :param user_id: Identifier of the requesting user.
        :param query_text: The raw natural-language question submitted.
        :param response_text: The final, cited answer returned to the user.
        :param latency_ms: End-to-end pipeline latency in milliseconds.
        :raises sqlalchemy.exc.SQLAlchemyError: If the insert fails.
        :return: The persisted, flushed :class:`~models.UserQuery` instance
            (with its generated primary key populated).
        """
        record = UserQuery(
            user_id=user_id,
            query_text=query_text,
            response_text=response_text,
            latency_ms=latency_ms,
        )
        self.session.add(record)
        await self.session.flush()
        logger.debug("Persisted UserQuery id=%s for user_id=%s", record.id, user_id)
        return record

    async def create_audit_logs(
        self,
        user_query_id: uuid.UUID,
        citations: Sequence[CitationSchema],
    ) -> List[AuditLog]:
        """Persist the citation audit trail supporting a user query's answer.

        :param user_query_id: Primary key of the parent :class:`~models.UserQuery`.
        :param citations: The citations returned alongside the generated answer.
        :raises sqlalchemy.exc.SQLAlchemyError: If any insert fails.
        :return: The list of persisted, flushed :class:`~models.AuditLog` instances.
        """
        records = [
            AuditLog(
                user_query_id=user_query_id,
                document_id=citation.document_id,
                document_title=citation.document_title,
                page_number=citation.page_number,
                chunk_id=citation.chunk_id,
                relevance_score=citation.relevance_score,
            )
            for citation in citations
        ]
        self.session.add_all(records)
        await self.session.flush()
        logger.debug(
            "Persisted %d AuditLog records for user_query_id=%s",
            len(records),
            user_query_id,
        )
        return records
