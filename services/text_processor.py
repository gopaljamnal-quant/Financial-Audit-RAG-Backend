"""
services/text_processor.py
===========================

Enterprise English text preprocessing service. Performs deterministic,
token-based chunking using ``tiktoken`` (``cl100k_base`` encoding) so
that chunk boundaries are stable across model versions and reproducible
for audit purposes. Chunking is CPU-bound; it is offloaded to a worker
thread via :func:`asyncio.to_thread` so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import unicodedata
from typing import List

import tiktoken

from config import settings
from schemas import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Non-ASCII-Latin scripts that indicate the source text is very unlikely
# to be English. This is a coarse, fast pre-filter -- not a full language
# identification model -- appropriate as a guardrail before indexing.
_NON_ENGLISH_SCRIPT_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]")


class TextProcessingError(Exception):
    """Raised when a document cannot be safely chunked (e.g. non-English content)."""


class TextProcessor:
    """Chunks raw English document text into overlapping, metadata-tagged blocks.

    :ivar encoding: The ``tiktoken`` encoding used for tokenisation,
        configured via ``TIKTOKEN_ENCODING`` (default ``cl100k_base``).
    :ivar chunk_size: Target chunk size in tokens.
    :ivar overlap: Number of overlapping tokens between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
        encoding_name: str | None = None,
    ) -> None:
        """Initialise the processor with explicit or configuration-derived parameters.

        :param chunk_size: Target chunk size in tokens; defaults to
            ``settings.CHUNK_TOKEN_SIZE``.
        :param overlap: Overlap in tokens between consecutive chunks;
            defaults to ``settings.CHUNK_TOKEN_OVERLAP``.
        :param encoding_name: ``tiktoken`` encoding name; defaults to
            ``settings.TIKTOKEN_ENCODING``.
        """
        self.chunk_size = chunk_size or settings.CHUNK_TOKEN_SIZE
        self.overlap = overlap if overlap is not None else settings.CHUNK_TOKEN_OVERLAP
        self.encoding = tiktoken.get_encoding(encoding_name or settings.TIKTOKEN_ENCODING)

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be strictly smaller than chunk_size")

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise unicode form and collapse extraneous whitespace.

        :param text: Raw input text.
        :return: Normalised text with NFC unicode form, collapsed
            horizontal whitespace, and capped consecutive blank lines.
        """
        text = unicodedata.normalize("NFC", text)
        text = _WHITESPACE_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()

    @classmethod
    def _assert_english_dominant(cls, text: str, sample_len: int = 4000) -> None:
        """Guard against indexing content that is clearly not English.

        Financial-audit deployments require an English-only corpus for
        deterministic citation and compliance review. This performs a
        cheap script-based sniff test rather than full language ID.

        :param text: The normalised document text.
        :param sample_len: Number of characters sampled for the check.
        :raises TextProcessingError: If a significant fraction of the
            sampled text is composed of non-Latin scripts.
        """
        sample = text[:sample_len]
        if not sample:
            return
        non_english_hits = len(_NON_ENGLISH_SCRIPT_RE.findall(sample))
        if non_english_hits / max(len(sample), 1) > 0.15:
            raise TextProcessingError(
                "Document content does not appear to be predominantly English; "
                "rejected by the English-only ingestion policy."
            )

    def _chunk_token_ids_sync(self, token_ids: List[int]) -> List[List[int]]:
        """Slide a fixed-size, overlapping window across a token-id sequence.

        :param token_ids: The full sequence of token ids for a document.
        :return: A list of token-id windows, each of length at most
            ``self.chunk_size``, advancing by ``chunk_size - overlap``
            tokens per step.
        """
        if not token_ids:
            return []

        step = self.chunk_size - self.overlap
        windows: List[List[int]] = []
        start = 0
        total = len(token_ids)
        while start < total:
            end = min(start + self.chunk_size, total)
            windows.append(token_ids[start:end])
            if end == total:
                break
            start += step
        return windows

    def _build_chunks_sync(
        self,
        text: str,
        document_id: str,
        document_title: str,
        classification: str,
        page_number: int | None,
    ) -> List[TextChunk]:
        """Synchronous, CPU-bound core of the chunking algorithm.

        :param text: Normalised source text for a single document (or page).
        :param document_id: Source document identifier.
        :param document_title: Human-readable document title.
        :param classification: Sensitivity classification to propagate to chunk metadata.
        :param page_number: Page number this text originates from, if applicable.
        :return: A list of fully-populated :class:`TextChunk` instances.
        """
        token_ids = self.encoding.encode(text, disallowed_special=())
        windows = self._chunk_token_ids_sync(token_ids)

        chunks: List[TextChunk] = []
        for index, window in enumerate(windows):
            chunk_text = self.encoding.decode(window).strip()
            if not chunk_text:
                continue
            digest_source = f"{document_id}:{page_number}:{index}:{chunk_text[:64]}"
            chunk_id = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:32]
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        document_title=document_title,
                        page_number=page_number,
                        chunk_index=index,
                        token_count=len(window),
                        classification=classification,
                    ),
                )
            )
        return chunks

    async def chunk_document(
        self,
        text: str,
        document_id: str,
        document_title: str,
        classification: str = "internal",
        page_number: int | None = None,
    ) -> List[TextChunk]:
        """Asynchronously normalise, validate, and chunk a document's text.

        The CPU-bound tokenisation work is delegated to a worker thread
        via :func:`asyncio.to_thread` to keep the event loop responsive
        under concurrent ingestion load.

        :param text: Raw document (or single-page) text.
        :param document_id: Source document identifier, propagated to every chunk.
        :param document_title: Human-readable document title.
        :param classification: Sensitivity classification propagated to chunk metadata.
        :param page_number: Page number this text originates from, if applicable.
        :raises TextProcessingError: If the content fails the English-dominance guard.
        :return: A list of overlapping :class:`TextChunk` instances covering the document.
        """
        normalised = self._normalise(text)
        self._assert_english_dominant(normalised)

        chunks = await asyncio.to_thread(
            self._build_chunks_sync,
            normalised,
            document_id,
            document_title,
            classification,
            page_number,
        )
        logger.info(
            "Chunked document_id=%s page=%s into %d chunks (chunk_size=%d, overlap=%d)",
            document_id,
            page_number,
            len(chunks),
            self.chunk_size,
            self.overlap,
        )
        return chunks

    async def chunk_pages(
        self,
        pages: List[str],
        document_id: str,
        document_title: str,
        classification: str = "internal",
    ) -> List[TextChunk]:
        """Chunk a multi-page document, preserving per-page provenance.

        :param pages: Ordered list of page texts, one entry per page
            (1-indexed page numbers are assigned in order).
        :param document_id: Source document identifier.
        :param document_title: Human-readable document title.
        :param classification: Sensitivity classification propagated to chunk metadata.
        :return: The concatenation of chunks produced for every page, in order.
        """
        results: List[TextChunk] = []
        for page_number, page_text in enumerate(pages, start=1):
            page_chunks = await self.chunk_document(
                text=page_text,
                document_id=document_id,
                document_title=document_title,
                classification=classification,
                page_number=page_number,
            )
            results.extend(page_chunks)
        return results
