"""
Knowledge Indexer -- Phase 1C

Processes knowledge inputs (Q&A pairs, text snippets) into vector
embeddings stored in ChromaDB via the KnowledgeRetriever.

Phase 1C: Simple Q&A pair and text chunk indexing.
Phase 2: Full document processing (PDF, TXT) with chunking.

Pipeline:
  Text input -> Split into chunks -> Store in ChromaDB (auto-embedded)
"""

import logging
import re
import time
from typing import Optional

logger = logging.getLogger("liveself.engine.indexer")

# Chunking defaults
DEFAULT_CHUNK_SIZE = 512  # characters (not tokens, simpler for Phase 1C)
DEFAULT_CHUNK_OVERLAP = 50


class KnowledgeIndexer:
    """
    Indexes knowledge for a persona's RAG collection.

    Takes raw text or Q&A pairs and stores them in ChromaDB through
    the KnowledgeRetriever.

    Usage:
        from knowledge.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever()
        retriever.load()
        retriever.set_persona("abc123")

        indexer = KnowledgeIndexer(retriever)
        indexer.add_qa_pairs([
            {"q": "What do you do?", "a": "I am a software engineer."},
        ])
        indexer.add_text("I have 5 years of Python experience...")
    """

    def __init__(self, retriever, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Args:
            retriever: A loaded KnowledgeRetriever with a persona selected.
            chunk_size: Max characters per chunk when splitting long text.
            chunk_overlap: Overlap between adjacent chunks in characters.
        """
        self._retriever = retriever
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        logger.info("KnowledgeIndexer created")

    def add_qa_pairs(self, pairs: list[dict]) -> int:
        """
        Index Q&A pairs as knowledge chunks.

        Each pair is stored as "Q: {question}\nA: {answer}" so the
        retriever can match questions semantically.

        Args:
            pairs: List of dicts with "q" and "a" keys.

        Returns:
            Number of chunks indexed.
        """
        if not pairs:
            return 0

        start = time.perf_counter()

        texts = []
        metadatas = []
        for pair in pairs:
            q = pair.get("q", "").strip()
            a = pair.get("a", "").strip()
            if q and a:
                texts.append(f"Q: {q}\nA: {a}")
                metadatas.append({"type": "qa", "question": q[:200]})

        if texts:
            self._retriever.add_knowledge(texts=texts, metadatas=metadatas)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Indexed {len(texts)} Q&A pairs in {elapsed_ms:.0f}ms")
        return len(texts)

    def add_text(self, text: str, source: str = "manual") -> int:
        """
        Index a block of text by splitting it into chunks.

        Long text is split on sentence boundaries into chunks of
        approximately chunk_size characters, with overlap.

        Args:
            text: Raw text to index.
            source: Label for the source (e.g., "upload", "manual").

        Returns:
            Number of chunks indexed.
        """
        if not text.strip():
            return 0

        start = time.perf_counter()

        chunks = self._split_text(text)

        if chunks:
            metadatas = [{"type": "text", "source": source} for _ in chunks]
            self._retriever.add_knowledge(texts=chunks, metadatas=metadatas)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Indexed {len(chunks)} text chunks in {elapsed_ms:.0f}ms (source: {source})")
        return len(chunks)

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks on sentence boundaries.

        Tries to keep chunks under chunk_size characters while splitting
        at sentence endings (. ! ?). Falls back to splitting on whitespace
        if no sentence boundary is found.

        Args:
            text: Raw text to split.

        Returns:
            List of text chunks.
        """
        text = text.strip()
        if len(text) <= self._chunk_size:
            return [text]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self._chunk_size:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If a single sentence exceeds chunk_size, split on whitespace
                if len(sentence) > self._chunk_size:
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= self._chunk_size:
                            current_chunk = f"{current_chunk} {word}".strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
