"""
Knowledge Retriever -- Phase 1C

Queries ChromaDB to find the most relevant knowledge chunks for a given
question. This is the RAG (Retrieval-Augmented Generation) layer that
makes the avatar sound like YOU, not a generic AI.

Input: question text from ASR
Output: top-k most relevant knowledge chunks for LLM context

Each persona has its own ChromaDB collection. The retriever queries
the correct collection based on the active persona ID.

ChromaDB runs in-process (no server needed). Data is stored on disk
in engine/data/chroma/.
"""

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("liveself.engine.retriever")

# Default ChromaDB persistence directory
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"

# How many chunks to retrieve per query
DEFAULT_TOP_K = 3

# Collection naming convention: persona_{persona_id}
COLLECTION_PREFIX = "persona_"


class KnowledgeRetriever:
    """
    RAG retriever backed by ChromaDB.

    Each persona gets its own ChromaDB collection. When the avatar receives
    a question, this module finds the most relevant knowledge chunks to
    include in the LLM prompt.

    Usage:
        retriever = KnowledgeRetriever()
        retriever.load()
        retriever.set_persona("abc123")
        chunks = retriever.query("What is your experience with Python?")
    """

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Args:
            chroma_path: Directory where ChromaDB stores its data.
                         Defaults to engine/data/chroma/.
            top_k: Number of knowledge chunks to retrieve per query.
        """
        self._chroma_path = Path(chroma_path) if chroma_path else DEFAULT_CHROMA_PATH
        self._top_k = top_k

        self._client = None
        self._collection = None
        self._persona_id = None
        self._is_loaded = False

        logger.info("KnowledgeRetriever created (not loaded yet)")

    def load(self) -> None:
        """
        Initialize the ChromaDB client with persistent storage.

        Creates the data directory if it does not exist.

        Raises:
            ImportError: If chromadb is not installed.
        """
        start = time.perf_counter()

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required but not installed. "
                "Install with: pip install chromadb"
            )

        # Create persistence directory
        self._chroma_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._chroma_path))
        self._is_loaded = True

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"ChromaDB client loaded in {elapsed_ms:.0f}ms (path: {self._chroma_path})")

    def set_persona(self, persona_id: str) -> None:
        """
        Switch to a persona's knowledge collection.

        Creates the collection if it does not exist yet (empty persona).

        Args:
            persona_id: Unique persona identifier.

        Raises:
            RuntimeError: If ChromaDB client is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("ChromaDB not loaded. Call load() first.")

        collection_name = f"{COLLECTION_PREFIX}{persona_id}"

        # get_or_create_collection is idempotent
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._persona_id = persona_id
        count = self._collection.count()
        logger.info(f"Persona set: {persona_id} ({count} chunks in collection)")

    def query(self, question: str, top_k: Optional[int] = None) -> list[str]:
        """
        Find the most relevant knowledge chunks for a question.

        Uses ChromaDB's built-in embedding function (all-MiniLM-L6-v2 by
        default) to embed the question and find similar chunks.

        Args:
            question: The user's question text from ASR.
            top_k: Override the default number of results.

        Returns:
            List of relevant knowledge chunk strings, ordered by relevance.
            Returns empty list if the collection is empty.

        Raises:
            RuntimeError: If no persona collection is selected.
        """
        if self._collection is None:
            raise RuntimeError("No persona selected. Call set_persona() first.")

        k = top_k or self._top_k
        count = self._collection.count()

        if count == 0:
            logger.debug(f"Empty collection for persona {self._persona_id}")
            return []

        # Don't request more results than exist
        k = min(k, count)

        start = time.perf_counter()

        results = self._collection.query(
            query_texts=[question],
            n_results=k,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # results["documents"] is a list of lists (one per query)
        chunks = results["documents"][0] if results["documents"] else []

        logger.info(
            f"Retrieved {len(chunks)} chunks in {elapsed_ms:.0f}ms "
            f"for: '{question[:60]}'"
        )

        return chunks

    def add_knowledge(
        self,
        texts: list[str],
        ids: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """
        Add knowledge chunks to the current persona's collection.

        This is called by the indexer when the user uploads Q&A pairs or
        documents. For Phase 1C, this handles simple text chunks directly.

        Args:
            texts: List of knowledge text chunks to store.
            ids: Optional unique IDs for each chunk. Auto-generated if not provided.
            metadatas: Optional metadata dicts for each chunk.

        Raises:
            RuntimeError: If no persona collection is selected.
        """
        if self._collection is None:
            raise RuntimeError("No persona selected. Call set_persona() first.")

        if not texts:
            return

        # Auto-generate IDs if not provided
        if ids is None:
            existing_count = self._collection.count()
            ids = [f"chunk_{existing_count + i}" for i in range(len(texts))]

        start = time.perf_counter()

        self._collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Added {len(texts)} chunks to persona {self._persona_id} "
            f"in {elapsed_ms:.0f}ms (total: {self._collection.count()})"
        )

    def get_collection_count(self) -> int:
        """Return the number of chunks in the current persona's collection."""
        if self._collection is None:
            return 0
        return self._collection.count()

    @property
    def is_ready(self) -> bool:
        """Check if the retriever is loaded and a persona is selected."""
        return self._is_loaded and self._collection is not None

    def unload(self) -> None:
        """Release ChromaDB client."""
        self._client = None
        self._collection = None
        self._persona_id = None
        self._is_loaded = False
        logger.info("KnowledgeRetriever unloaded")
