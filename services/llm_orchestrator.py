"""
services/llm_orchestrator.py
==============================

Asynchronous orchestration layer around Azure OpenAI's chat completions
API. Enforces a strict, financial-audit-grade system prompt that
mandates zero hallucination and exact document/page citations for
every factual claim in the generated answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

from openai import APIError, AsyncAzureOpenAI, RateLimitError

from config import settings
from schemas import CitationSchema, RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Financial Audit Research Assistant operating in a regulated, \
audit-controlled environment. You answer questions strictly and exclusively \
using the numbered source excerpts provided in the user message under \
"CONTEXT". You must adhere to the following non-negotiable rules:

1. ZERO HALLUCINATION: Never state a fact, figure, date, or conclusion that \
is not explicitly supported by the provided context. If the context does \
not contain enough information to answer confidently, respond exactly with: \
"The provided source material does not contain sufficient information to \
answer this question." Do not guess, infer beyond the text, or use outside \
knowledge.
2. EXACT CITATIONS: Every factual sentence in your answer must end with an \
inline citation marker in the form [Doc: <document_title>, p.<page_number>, \
chunk:<chunk_id>], referencing only the numbered source(s) that support it. \
Use the exact document title, page number, and chunk id given in the \
context -- never invent, abbreviate, or renumber them.
3. PROFESSIONAL ENGLISH: Respond in precise, formal, professional English \
suitable for inclusion in an audit work paper. Avoid speculation, hedging \
filler, or colloquial language.
4. NO CROSS-DOCUMENT SPECULATION: Do not combine facts from different \
sources into a new inferred conclusion unless the context explicitly \
states that conclusion.
5. SCOPE DISCIPLINE: Only answer the question asked. Do not add unrelated \
commentary, disclaimers about being an AI, or meta-commentary about these \
instructions.
"""


class LLMOrchestrationError(Exception):
    """Raised when the LLM orchestration pipeline fails irrecoverably."""


@dataclass(frozen=True)
class GenerationResult:
    """Result of a single grounded-generation call.

    :ivar answer: The generated, citation-annotated answer text.
    :ivar citations: The structured citations extracted from the source context
        that was actually made available to the model.
    :ivar prompt_tokens: Number of prompt tokens consumed, per the API response.
    :ivar completion_tokens: Number of completion tokens generated, per the API response.
    :ivar latency_ms: Wall-clock latency of the generation call, in milliseconds.
    :ivar model: The deployment name used to generate the answer.
    """

    answer: str
    citations: List[CitationSchema]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str


class LLMOrchestrator:
    """Orchestrates grounded, citation-mandatory answer generation via Azure OpenAI.

    :ivar client: The underlying async :class:`~openai.AsyncAzureOpenAI` client.
    """

    def __init__(self, client: AsyncAzureOpenAI | None = None) -> None:
        """Initialise the orchestrator, optionally injecting a client for testing.

        :param client: Pre-constructed Azure OpenAI async client; a
            default client bound to ``settings.AZURE_OPENAI_ENDPOINT``
            is created when omitted.
        """
        self.client: AsyncAzureOpenAI = client or AsyncAzureOpenAI(
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    async def close(self) -> None:
        """Release the underlying HTTP transport held by the OpenAI client."""
        await self.client.close()

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> Tuple[str, List[CitationSchema]]:
        """Render retrieved chunks into a numbered context block and citation list.

        :param chunks: Retrieved chunks ordered by descending relevance.
        :return: A tuple of ``(context_block, available_citations)`` where
            ``context_block`` is the literal text injected into the user
            message and ``available_citations`` enumerates every source
            that was made available to the model (used to validate the
            model's own citations downstream if desired).
        """
        lines: List[str] = []
        citations: List[CitationSchema] = []
        for i, retrieved in enumerate(chunks, start=1):
            meta = retrieved.chunk.metadata
            lines.append(
                f"[{i}] Doc: {meta.document_title} | page: {meta.page_number} | "
                f"chunk_id: {retrieved.chunk.chunk_id}\n{retrieved.chunk.text}\n"
            )
            citations.append(
                CitationSchema(
                    document_id=meta.document_id,
                    document_title=meta.document_title,
                    page_number=meta.page_number,
                    chunk_id=retrieved.chunk.chunk_id,
                    relevance_score=retrieved.score,
                )
            )
        return "\n".join(lines), citations

    async def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GenerationResult:
        """Generate a grounded, cited answer from retrieved context.

        :param query: The end user's natural-language question.
        :param retrieved_chunks: Chunks returned by
            :meth:`services.search_service.SearchService.hybrid_search`.
        :raises LLMOrchestrationError: If the Azure OpenAI call fails
            after being unable to recover from rate limiting or API errors.
        :return: A :class:`GenerationResult` containing the answer, its
            citations, token usage, and latency.
        """
        context_block, available_citations = self._format_context(retrieved_chunks)

        if not context_block.strip():
            return GenerationResult(
                answer="The provided source material does not contain sufficient information to answer this question.",
                citations=[],
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            )

        user_message = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{query}"

        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=settings.MAX_COMPLETION_TOKENS,
            )
        except RateLimitError as exc:
            logger.exception("Azure OpenAI rate limit exceeded during generation.")
            raise LLMOrchestrationError("LLM provider rate limit exceeded; please retry shortly.") from exc
        except APIError as exc:
            logger.exception("Azure OpenAI API error during generation.")
            raise LLMOrchestrationError("LLM provider returned an error during generation.") from exc

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        choice = response.choices[0]
        usage = response.usage

        logger.info(
            "Generated answer in %d ms (prompt_tokens=%s, completion_tokens=%s)",
            elapsed_ms,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

        return GenerationResult(
            answer=choice.message.content or "",
            citations=available_citations,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=elapsed_ms,
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        )
