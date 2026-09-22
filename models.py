"""
models.py
=========

Strict SQLAlchemy 2.0 declarative models backing the audit trail of the
RAG system: every user query, every retrieved/cited source, and the
role-based metadata mappings used for access-controlled retrieval.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class UserQuery(Base):
    """Represents a single end-user question submitted to the RAG pipeline.

    :ivar id: Primary key (UUID).
    :ivar user_id: Identifier of the authenticated caller, used for RBAC
        filtering and traceability in a financial-audit context.
    :ivar query_text: The raw natural-language question submitted.
    :ivar response_text: The final, cited answer returned to the user.
    :ivar latency_ms: End-to-end pipeline latency in milliseconds.
    :ivar created_at: UTC timestamp of query receipt.
    :ivar audit_logs: Related :class:`AuditLog` rows for this query.
    """

    __tablename__ = "user_queries"
    __table_args__ = (Index("ix_user_queries_user_id_created_at", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    audit_logs: Mapped[List["AuditLog"]] = relationship(
        back_populates="user_query",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"<UserQuery id={self.id} user_id={self.user_id!r}>"


class AuditLog(Base):
    """Immutable audit record of a single retrieved source cited in a response.

    Captures exactly which document/page/chunk contributed to an answer,
    satisfying the zero-hallucination / exact-citation requirement of a
    financial-audit deployment.

    :ivar id: Primary key (UUID).
    :ivar user_query_id: Foreign key to the originating :class:`UserQuery`.
    :ivar document_id: Source document identifier as indexed in Azure AI Search.
    :ivar document_title: Human-readable document title.
    :ivar page_number: Page number within the source document, if applicable.
    :ivar chunk_id: Identifier of the specific retrieved chunk.
    :ivar relevance_score: Hybrid search relevance/rerank score for this chunk.
    :ivar retrieved_at: UTC timestamp of retrieval.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_document_id", "document_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user_queries.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[str] = mapped_column(String(256), nullable=False)
    document_title: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(nullable=True)
    chunk_id: Mapped[str] = mapped_column(String(256), nullable=False)
    relevance_score: Mapped[float] = mapped_column(nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    user_query: Mapped["UserQuery"] = relationship(back_populates="audit_logs")

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"<AuditLog document_id={self.document_id!r} chunk_id={self.chunk_id!r}>"


class MetadataMapping(Base):
    """Role-based access-control mapping between documents and allowed roles.

    Used by :mod:`services.search_service` to construct hard OData
    filters on Azure AI Search queries, ensuring a caller only ever
    retrieves chunks from documents their role is entitled to see.

    :ivar id: Primary key (UUID).
    :ivar document_id: Source document identifier as indexed in Azure AI Search.
    :ivar allowed_roles: JSONB array of role names permitted to access the document.
    :ivar classification: Sensitivity classification, e.g. "internal", "restricted".
    :ivar updated_at: UTC timestamp of the last mapping update.
    """

    __tablename__ = "metadata_mappings"
    __table_args__ = (Index("ix_metadata_mappings_document_id", "document_id", unique=True),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[str] = mapped_column(String(256), nullable=False)
    allowed_roles: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    classification: Mapped[str] = mapped_column(String(64), nullable=False, default="internal")
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"<MetadataMapping document_id={self.document_id!r} classification={self.classification!r}>"
