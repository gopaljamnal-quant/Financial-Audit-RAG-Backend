"""
services/search_service.py
============================

Asynchronous wrapper around Azure AI Search providing hybrid retrieval
(vector similarity over ``text-embedding-3-large`` embeddings combined
with BM25 keyword scoring) plus hard, role-based OData filtering over
document metadata so that a caller can never retrieve chunks from
documents their role is not entitled to see.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI

from config import settings
from schemas import ChunkMetadata, RetrievedChunk, TextChunk, UserRole

logger = logging.getLogger(__name__)

# Roles permitted to see documents classified as "restricted".
_RESTRICTED_ROLES: frozenset[str] = frozenset(
    {UserRole.AUDITOR.value, UserRole.ADMIN.value, UserRole.COMPLIANCE_OFFICER.value}
)


class SearchServiceError(Exception):
    """Raised when Azure AI Search retrieval fails irrecoverably."""


class SearchService:
    """Async hybrid-search wrapper with embedded RBAC enforcement.

    :ivar client: The underlying async :class:`~azure.search.documents.aio.SearchClient`.
    :ivar embedding_client: Async Azure OpenAI client used to embed queries.
    """

    def __init__(
        self,
        search_client: Optional[SearchClient] = None,
        embedding_client: Optional[AsyncAzureOpenAI] = None,
    ) -> None:
        """Initialise the service, optionally injecting clients for testing.

        :param search_client: Pre-constructed search client; a default
            client bound to ``settings.AZURE_SEARCH_ENDPOINT`` is
            created when omitted.
        :param embedding_client: Pre-constructed Azure OpenAI client
            used for query embedding; a default client is created when
            omitted.
        """
        self.client: SearchClient = search_client or SearchClient(
            endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
        )
        self.embedding_client: AsyncAzureOpenAI = embedding_client or AsyncAzureOpenAI(
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    async def close(self) -> None:
        """Release the underlying HTTP transports held by both clients."""
        await self.client.close()
        await self.embedding_client.close()

    async def _embed_query(self, query: str) -> List[float]:
        """Embed the user's query using the configured embedding deployment.

        :param query: The natural-language query text.
        :raises SearchServiceError: If the embedding call fails.
        :return: The dense embedding vector for the query.
        """
        try:
            response = await self.embedding_client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=query,
                dimensions=settings.AZURE_OPENAI_EMBEDDING_DIMENSIONS,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception("Query embedding failed.")
            raise SearchServiceError("Failed to embed query for vector search.") from exc
        return response.data[0].embedding

    @staticmethod
    def _build_rbac_filter(
        role: UserRole,
        document_filter: Optional[Sequence[str]] = None,
    ) -> str:
        """Construct a hard OData filter enforcing role-based access control.

        Non-privileged roles (:attr:`UserRole.ANALYST`) are restricted
        to documents classified as ``internal`` or ``public``.
        Privileged roles (auditor, admin, compliance officer) may also
        see ``restricted`` documents. When an explicit
        ``document_filter`` is supplied it is combined with (never
        substituted for) the role-based clause, so a caller can narrow
        but never widen their entitlement.

        :param role: The requesting user's role.
        :param document_filter: Optional explicit allow-list of document ids.
        :return: A valid Azure AI Search OData ``$filter`` expression.
        """
        if role.value in _RESTRICTED_ROLES:
            classification_clause = (
                "(classification eq 'internal' or classification eq 'public' or classification eq 'restricted')"
            )
        else:
            classification_clause = "(classification eq 'internal' or classification eq 'public')"

        clauses = [classification_clause]

        if document_filter:
            escaped_ids = [doc_id.replace("'", "''") for doc_id in document_filter]
            id_clause = " or ".join(f"document_id eq '{doc_id}'" for doc_id in escaped_ids)
            clauses.append(f"({id_clause})")

        return " and ".join(clauses)

    async def hybrid_search(
        self,
        query: str,
        role: UserRole,
        top_k: Optional[int] = None,
        document_filter: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        """Execute a hybrid (vector + BM25) search with RBAC filtering applied.

        :param query: The natural-language query text.
        :param role: The requesting user's role, used to build the RBAC filter.
        :param top_k: Number of results to retrieve; defaults to
            ``settings.AZURE_SEARCH_TOP_K``.
        :param document_filter: Optional explicit allow-list of document ids
            to further narrow retrieval within the caller's entitlement.
        :raises SearchServiceError: If the underlying search request fails.
        :return: Retrieved chunks ordered by descending relevance score.
        """
        k = top_k or settings.AZURE_SEARCH_TOP_K
        odata_filter = self._build_rbac_filter(role=role, document_filter=document_filter)
        vector = await self._embed_query(query)

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=k,
            fields=settings.AZURE_SEARCH_VECTOR_FIELD,
        )

        start = time.perf_counter()
        try:
            results_iterator = await self.client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=odata_filter,
                query_type="semantic",
                semantic_configuration_name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
                top=k,
                select=[
                    "chunk_id",
                    settings.AZURE_SEARCH_CONTENT_FIELD,
                    "document_id",
                    "document_title",
                    "page_number",
                    "chunk_index",
                    "token_count",
                    "classification",
                ],
            )
        except HttpResponseError as exc:
            logger.exception("Azure AI Search hybrid query failed.")
            raise SearchServiceError("Hybrid search request to Azure AI Search failed.") from exc

        retrieved: List[RetrievedChunk] = []
        async for result in results_iterator:
            score = float(result.get("@search.reranker_score") or result.get("@search.score") or 0.0)
            normalised_score = (
                max(0.0, min(score / 4.0, 1.0)) if result.get("@search.reranker_score") else max(0.0, min(score, 1.0))
            )
            retrieved.append(
                RetrievedChunk(
                    score=normalised_score,
                    chunk=TextChunk(
                        chunk_id=result["chunk_id"],
                        text=result[settings.AZURE_SEARCH_CONTENT_FIELD],
                        metadata=ChunkMetadata(
                            document_id=result["document_id"],
                            document_title=result["document_title"],
                            page_number=result.get("page_number"),
                            chunk_index=result.get("chunk_index", 0),
                            token_count=result.get("token_count", 0),
                            classification=result.get("classification", "internal"),
                        ),
                    ),
                )
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "Hybrid search returned %d chunks in %d ms (role=%s, filter=%s)",
            len(retrieved),
            elapsed_ms,
            role.value,
            odata_filter,
        )
        return retrieved
