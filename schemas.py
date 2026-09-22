"""
schemas.py
==========

Pydantic v2 validation models for API requests and responses. These
schemas define the strict contract between clients and the RAG
pipeline, including mandatory citation structures for financial-audit
traceability.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRole(str, Enum):
    """Roles recognised by the RBAC filter applied at retrieval time."""

    ANALYST = "analyst"
    AUDITOR = "auditor"
    ADMIN = "admin"
    COMPLIANCE_OFFICER = "compliance_officer"


class QueryRequest(BaseModel):
    """Inbound request payload for a RAG query.

    :ivar user_id: Identifier of the requesting user, used for audit
        trail and RBAC enforcement.
    :ivar role: The user's role, used to filter retrievable documents.
    :ivar query: The natural-language question in English.
    :ivar document_filter: Optional explicit document IDs to restrict
        retrieval to, further narrowed by the role-based filter.
    :ivar top_k: Number of chunks to retrieve; overrides the configured default.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    user_id: str = Field(..., min_length=1, max_length=128)
    role: UserRole = Field(...)
    query: str = Field(..., min_length=3, max_length=4000)
    document_filter: Optional[List[str]] = Field(default=None, max_length=50)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _query_must_be_english_ascii_dominant(cls, v: str) -> str:
        """Perform a lightweight structural sanity check on the query text.

        This is not a full language classifier; it rejects clearly
        malformed input (empty after stripping, control characters)
        while leaving true language detection to
        :mod:`services.text_processor`.

        :param v: The raw query string.
        :raises ValueError: If the query is empty after normalisation.
        :return: The normalised query string.
        """
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class CitationSchema(BaseModel):
    """A single, exact citation backing a claim in the generated answer.

    :ivar document_id: Source document identifier as indexed in search.
    :ivar document_title: Human-readable document title.
    :ivar page_number: Page number within the source document, if known.
    :ivar chunk_id: Identifier of the specific retrieved chunk.
    :ivar relevance_score: Retrieval relevance score in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_title: str
    page_number: Optional[int] = None
    chunk_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Outbound response payload for a RAG query.

    :ivar query_id: Server-generated identifier for this query/response pair.
    :ivar answer: The generated, citation-grounded answer in English.
    :ivar citations: Ordered list of :class:`CitationSchema` backing the answer.
    :ivar latency_ms: End-to-end pipeline latency in milliseconds.
    :ivar model: Name of the LLM deployment used to generate the answer.
    :ivar created_at: UTC timestamp of response generation.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: uuid.UUID
    answer: str
    citations: List[CitationSchema] = Field(default_factory=list)
    latency_ms: int
    model: str
    created_at: datetime


class ChunkMetadata(BaseModel):
    """Metadata attached to a single text chunk produced during preprocessing.

    :ivar document_id: Source document identifier.
    :ivar document_title: Human-readable document title.
    :ivar page_number: Page number within the source document, if applicable.
    :ivar chunk_index: Zero-based position of this chunk within the document.
    :ivar token_count: Number of tokens in this chunk under the configured encoding.
    :ivar classification: Sensitivity classification inherited from the source document.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_title: str
    page_number: Optional[int] = None
    chunk_index: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    classification: str = Field(default="internal")


class TextChunk(BaseModel):
    """A single chunk of text produced by :mod:`services.text_processor`.

    :ivar chunk_id: Deterministic, unique identifier for the chunk.
    :ivar text: The chunk's textual content.
    :ivar metadata: The chunk's associated :class:`ChunkMetadata`.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """A chunk returned from Azure AI Search hybrid retrieval.

    :ivar chunk: The underlying :class:`TextChunk`.
    :ivar score: The hybrid search relevance score in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: TextChunk
    score: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Liveness/readiness probe response.

    :ivar status: ``"ok"`` when the service is healthy.
    :ivar app_name: The configured application name.
    :ivar app_env: The configured deployment environment.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    app_name: str
    app_env: str


class ErrorResponse(BaseModel):
    """Standardised error envelope returned by global exception handlers.

    :ivar detail: Human-readable error description.
    :ivar error_code: Machine-readable error identifier.
    :ivar request_id: Correlation identifier for cross-referencing logs.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str
    error_code: str
    request_id: Optional[str] = None
