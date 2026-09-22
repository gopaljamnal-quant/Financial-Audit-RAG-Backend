"""
test_rag.py
===========

Automated test suite for the RAG backend using ``pytest`` and
``pytest-asyncio``. The database layer, embedding generation, and
Azure AI Search indexing responses are all mocked so the suite runs
fully offline and deterministically in CI, with no live Azure
resources required.
"""

from __future__ import annotations

import os
import uuid
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# --- Environment must be populated before importing application modules ---
# so that `config.Settings` (which forbids extra/undeclared env vars and
# requires several fields) can be constructed successfully in a CI runner
# with no real Azure credentials present.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test-openai.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("AZURE_OPENAI_CHAT_DEPLOYMENT", "test-chat-deployment")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://test-search.search.windows.net")
os.environ.setdefault("AZURE_SEARCH_API_KEY", "test-search-key")
os.environ.setdefault("AZURE_SEARCH_INDEX_NAME", "test-index")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test_db")

from schemas import (  # noqa: E402
    ChunkMetadata,
    CitationSchema,
    QueryRequest,
    RetrievedChunk,
    TextChunk,
    UserRole,
)
from services.text_processor import TextProcessingError, TextProcessor  # noqa: E402
from services.search_service import SearchService  # noqa: E402
from services.llm_orchestrator import LLMOrchestrator  # noqa: E402
from repositories.audit_repository import AuditRepository  # noqa: E402


# ---------------------------------------------------------------------------
# services/text_processor.py
# ---------------------------------------------------------------------------


class TestTextProcessor:
    """Unit tests for token-based chunking with overlap and metadata preservation."""

    @pytest.mark.asyncio
    async def test_chunking_produces_overlapping_windows(self) -> None:
        """Chunks should be produced with the configured size and overlap."""
        processor = TextProcessor(chunk_size=50, overlap=10)
        text = " ".join(f"word{i}" for i in range(500))

        chunks = await processor.chunk_document(
            text=text,
            document_id="doc-1",
            document_title="Test Document",
            classification="internal",
        )

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata.token_count <= 50
            assert chunk.metadata.document_id == "doc-1"
            assert chunk.metadata.classification == "internal"

    @pytest.mark.asyncio
    async def test_chunk_ids_are_deterministic(self) -> None:
        """Re-chunking identical input must yield identical chunk ids (audit reproducibility)."""
        processor = TextProcessor(chunk_size=30, overlap=5)
        text = "The quick brown fox jumps over the lazy dog. " * 20

        first_pass = await processor.chunk_document(text, "doc-2", "Doc Two")
        second_pass = await processor.chunk_document(text, "doc-2", "Doc Two")

        assert [c.chunk_id for c in first_pass] == [c.chunk_id for c in second_pass]

    @pytest.mark.asyncio
    async def test_rejects_non_english_dominant_content(self) -> None:
        """Content dominated by non-Latin scripts must be rejected by the ingestion guard."""
        processor = TextProcessor(chunk_size=50, overlap=5)
        non_english_text = "こんにちは世界。" * 200

        with pytest.raises(TextProcessingError):
            await processor.chunk_document(non_english_text, "doc-3", "Non-English Doc")

    def test_invalid_overlap_configuration_raises(self) -> None:
        """Overlap configured >= chunk size must fail fast at construction time."""
        with pytest.raises(ValueError):
            TextProcessor(chunk_size=20, overlap=20)


# ---------------------------------------------------------------------------
# services/search_service.py
# ---------------------------------------------------------------------------


class _FakeAsyncSearchResultsIterator:
    """Minimal async iterator standing in for Azure Search's async result pager."""

    def __init__(self, results: List[dict]) -> None:
        self._results = results

    def __aiter__(self) -> "_FakeAsyncSearchResultsIterator":
        self._iter = iter(self._results)
        return self

    async def __anext__(self) -> dict:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class TestSearchService:
    """Unit tests for hybrid search retrieval and RBAC filter construction."""

    def _make_service(self, search_results: List[dict], embedding_vector: List[float]) -> SearchService:
        """Build a :class:`SearchService` with fully mocked Azure clients.

        :param search_results: Fake documents to be yielded by the mocked search client.
        :param embedding_vector: Fake embedding vector returned by the mocked embedding client.
        :return: A :class:`SearchService` wired to mocks instead of live Azure resources.
        """
        mock_search_client = MagicMock()
        mock_search_client.search = AsyncMock(return_value=_FakeAsyncSearchResultsIterator(search_results))
        mock_search_client.close = AsyncMock()

        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=embedding_vector)]

        mock_embedding_client = MagicMock()
        mock_embedding_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        mock_embedding_client.close = AsyncMock()

        return SearchService(search_client=mock_search_client, embedding_client=mock_embedding_client)

    def test_rbac_filter_excludes_restricted_for_analyst(self) -> None:
        """Analysts must not be granted access to 'restricted' classified documents."""
        odata_filter = SearchService._build_rbac_filter(role=UserRole.ANALYST)
        assert "restricted" not in odata_filter
        assert "internal" in odata_filter

    def test_rbac_filter_includes_restricted_for_auditor(self) -> None:
        """Auditors must be permitted to see 'restricted' classified documents."""
        odata_filter = SearchService._build_rbac_filter(role=UserRole.AUDITOR)
        assert "restricted" in odata_filter

    def test_rbac_filter_narrows_but_does_not_widen_with_document_filter(self) -> None:
        """An explicit document_filter must combine with, not replace, the RBAC clause."""
        odata_filter = SearchService._build_rbac_filter(role=UserRole.ANALYST, document_filter=["doc-1", "doc-2"])
        assert "restricted" not in odata_filter
        assert "doc-1" in odata_filter and "doc-2" in odata_filter

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_parsed_chunks(self) -> None:
        """A mocked search response should be parsed into RetrievedChunk objects."""
        fake_results = [
            {
                "chunk_id": "chunk-abc",
                "content": "Revenue recognition policy excerpt.",
                "document_id": "doc-10",
                "document_title": "FY24 Audit Report",
                "page_number": 12,
                "chunk_index": 3,
                "token_count": 128,
                "classification": "internal",
                "@search.score": 0.87,
            }
        ]
        service = self._make_service(search_results=fake_results, embedding_vector=[0.1] * 3072)

        results = await service.hybrid_search(query="What is the revenue recognition policy?", role=UserRole.ANALYST)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].chunk.metadata.document_title == "FY24 Audit Report"
        assert 0.0 <= results[0].score <= 1.0


# ---------------------------------------------------------------------------
# services/llm_orchestrator.py
# ---------------------------------------------------------------------------


class TestLLMOrchestrator:
    """Unit tests for grounded generation with mandatory citations."""

    def _make_retrieved_chunk(self) -> RetrievedChunk:
        return RetrievedChunk(
            score=0.91,
            chunk=TextChunk(
                chunk_id="chunk-xyz",
                text="The audit committee approved the revised revenue recognition policy in Q3.",
                metadata=ChunkMetadata(
                    document_id="doc-99",
                    document_title="Audit Committee Minutes",
                    page_number=4,
                    chunk_index=1,
                    token_count=64,
                    classification="internal",
                ),
            ),
        )

    @pytest.mark.asyncio
    async def test_generate_answer_returns_citations_from_context(self) -> None:
        """The orchestrator must surface citations for every chunk supplied as context."""
        mock_message = MagicMock()
        mock_message.content = (
            "The audit committee approved the revised policy in Q3. "
            "[Doc: Audit Committee Minutes, p.4, chunk:chunk-xyz]"
        )
        mock_choice = MagicMock(message=mock_message)
        mock_usage = MagicMock(prompt_tokens=120, completion_tokens=40)
        mock_response = MagicMock(choices=[mock_choice], usage=mock_usage)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        orchestrator = LLMOrchestrator(client=mock_client)
        result = await orchestrator.generate_answer(
            query="When was the revenue policy approved?",
            retrieved_chunks=[self._make_retrieved_chunk()],
        )

        assert "Q3" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].document_title == "Audit Committee Minutes"
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 40

    @pytest.mark.asyncio
    async def test_generate_answer_with_no_context_declines(self) -> None:
        """With zero retrieved chunks, the orchestrator must short-circuit to the standard decline message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()

        orchestrator = LLMOrchestrator(client=mock_client)
        result = await orchestrator.generate_answer(query="Anything?", retrieved_chunks=[])

        assert "does not contain sufficient information" in result.answer
        mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# repositories/audit_repository.py
# ---------------------------------------------------------------------------


class TestAuditRepository:
    """Unit tests for audit persistence against a fully mocked database session."""

    @pytest_asyncio.fixture
    async def mock_session(self) -> AsyncIterator[MagicMock]:
        """Provide a mocked :class:`~sqlalchemy.ext.asyncio.AsyncSession`."""
        session = MagicMock()
        session.add = MagicMock()
        session.add_all = MagicMock()
        session.flush = AsyncMock()
        yield session

    @pytest.mark.asyncio
    async def test_create_user_query_flushes_session(self, mock_session: MagicMock) -> None:
        """Creating a user query record must add it to the session and flush."""
        repo = AuditRepository(mock_session)

        record = await repo.create_user_query(
            user_id="user-1",
            query_text="What is the capitalisation policy?",
            response_text="Per section 4.2 [Doc: Policy, p.2, chunk:abc].",
            latency_ms=250,
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()
        assert record.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_create_audit_logs_persists_all_citations(self, mock_session: MagicMock) -> None:
        """Every citation returned by the LLM must be persisted as an AuditLog row."""
        repo = AuditRepository(mock_session)
        citations = [
            CitationSchema(
                document_id="doc-1",
                document_title="Doc One",
                page_number=1,
                chunk_id="chunk-1",
                relevance_score=0.75,
            ),
            CitationSchema(
                document_id="doc-2",
                document_title="Doc Two",
                page_number=5,
                chunk_id="chunk-2",
                relevance_score=0.60,
            ),
        ]

        records = await repo.create_audit_logs(user_query_id=uuid.uuid4(), citations=citations)

        assert len(records) == 2
        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_awaited_once()


# ---------------------------------------------------------------------------
# schemas.py -- request validation
# ---------------------------------------------------------------------------


class TestQueryRequestValidation:
    """Unit tests for strict Pydantic v2 request validation."""

    def test_valid_request_parses(self) -> None:
        request = QueryRequest(user_id="user-1", role=UserRole.ANALYST, query="What is the accrual policy?")
        assert request.role is UserRole.ANALYST

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(Exception):
            QueryRequest(user_id="user-1", role=UserRole.ANALYST, query="   ")

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(Exception):
            QueryRequest.model_validate(
                {
                    "user_id": "user-1",
                    "role": "analyst",
                    "query": "Valid question?",
                    "unexpected_field": "should fail",
                }
            )
