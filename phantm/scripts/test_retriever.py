"""
Test script for the KnowledgeRetriever and KnowledgeIndexer (Phase 1C).

Run with: python scripts/test_retriever.py

Tests ChromaDB-backed RAG retrieval. Uses a temporary directory so
tests don't pollute the real knowledge base.

No GPU needed -- ChromaDB runs on CPU.
"""

import os
import sys
import shutil
import tempfile
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_retriever")


def test_retriever_init():
    """Verify KnowledgeRetriever initializes without loading."""
    from knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    assert not retriever.is_ready
    assert retriever.get_collection_count() == 0
    logger.info("PASS: retriever init")


def test_retriever_errors_before_load():
    """Verify proper errors before loading."""
    from knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()

    try:
        retriever.set_persona("test")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not loaded" in str(e).lower()
        logger.info("PASS: set_persona raises RuntimeError before load()")

    try:
        retriever.query("test question")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        logger.info("PASS: query raises RuntimeError before set_persona()")


def test_retriever_full_workflow():
    """Test the full add -> query workflow with a temp ChromaDB."""
    from knowledge.retriever import KnowledgeRetriever

    # Use a temp directory for ChromaDB
    temp_dir = tempfile.mkdtemp(prefix="liveself_test_chroma_")

    try:
        retriever = KnowledgeRetriever(chroma_path=temp_dir)
        retriever.load()
        assert retriever._is_loaded
        logger.info("PASS: retriever load")

        # Set persona
        retriever.set_persona("test_persona_123")
        assert retriever.is_ready
        assert retriever.get_collection_count() == 0
        logger.info("PASS: set persona (empty collection)")

        # Add knowledge
        retriever.add_knowledge(
            texts=[
                "I have been programming in Python for 5 years.",
                "My favorite framework is FastAPI for building APIs.",
                "I studied Physics and Computer Science at university.",
                "I live in Lagos, Nigeria and work on AI projects.",
            ]
        )
        assert retriever.get_collection_count() == 4
        logger.info("PASS: add knowledge (4 chunks)")

        # Query -- should find relevant chunks
        results = retriever.query("What programming languages do you know?")
        assert len(results) > 0, "Should find at least one result"
        assert any("Python" in r for r in results), "Should find the Python chunk"
        logger.info(f"PASS: query returned {len(results)} results")

        # Query with different question
        results = retriever.query("Where do you live?")
        assert len(results) > 0
        assert any("Lagos" in r for r in results), "Should find the Lagos chunk"
        logger.info("PASS: query relevance (found Lagos)")

        # Query empty topic
        results = retriever.query("What is your favorite color?")
        assert len(results) > 0  # Will return something, just less relevant
        logger.info("PASS: query on unrelated topic still returns results")

        retriever.unload()
        assert not retriever.is_ready
        logger.info("PASS: retriever unload")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_indexer_qa_pairs():
    """Test KnowledgeIndexer Q&A pair indexing."""
    from knowledge.retriever import KnowledgeRetriever
    from knowledge.indexer import KnowledgeIndexer

    temp_dir = tempfile.mkdtemp(prefix="liveself_test_indexer_")

    try:
        retriever = KnowledgeRetriever(chroma_path=temp_dir)
        retriever.load()
        retriever.set_persona("test_indexer")

        indexer = KnowledgeIndexer(retriever)

        # Add Q&A pairs
        count = indexer.add_qa_pairs([
            {"q": "What do you do for work?", "a": "I am an AI engineer building digital twins."},
            {"q": "What is your hobby?", "a": "I enjoy playing chess and reading sci-fi novels."},
            {"q": "What is your goal?", "a": "I want to make AI accessible to everyone in Africa."},
        ])
        assert count == 3
        assert retriever.get_collection_count() == 3
        logger.info("PASS: indexer Q&A pairs")

        # Query through retriever
        results = retriever.query("What is your job?")
        assert any("AI engineer" in r for r in results)
        logger.info("PASS: Q&A retrieval works")

        retriever.unload()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_indexer_text_chunking():
    """Test text chunking in KnowledgeIndexer."""
    from knowledge.retriever import KnowledgeRetriever
    from knowledge.indexer import KnowledgeIndexer

    temp_dir = tempfile.mkdtemp(prefix="liveself_test_chunks_")

    try:
        retriever = KnowledgeRetriever(chroma_path=temp_dir)
        retriever.load()
        retriever.set_persona("test_chunks")

        indexer = KnowledgeIndexer(retriever, chunk_size=100)

        # Long text that should be split into multiple chunks
        long_text = (
            "I started learning to code when I was 16 years old. "
            "My first language was Python, which I picked up from YouTube tutorials. "
            "After a year, I moved on to JavaScript and built my first website. "
            "In university, I studied both Physics and Computer Science. "
            "My final year project was about using machine learning for medical imaging. "
            "Now I work on AI digital twins, which combines everything I have learned."
        )

        count = indexer.add_text(long_text, source="test")
        assert count > 1, f"Long text should be split into multiple chunks, got {count}"
        logger.info(f"PASS: text chunking ({count} chunks from long text)")

        # Short text should stay as one chunk
        short_text = "I like coffee."
        count2 = indexer.add_text(short_text, source="test")
        assert count2 == 1
        logger.info("PASS: short text stays as one chunk")

        retriever.unload()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("KnowledgeRetriever + Indexer Tests -- Phase 1C")
    logger.info("=" * 60)

    # Check if chromadb is installed
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb not installed. Install with: pip install chromadb")
        logger.info("Skipping retriever tests (chromadb not available)")
        # Still run the init test
        test_retriever_init()
        logger.info("=" * 60)
        logger.info("Partial tests passed (chromadb not installed)")
        logger.info("=" * 60)
        return

    test_retriever_init()
    test_retriever_errors_before_load()
    test_retriever_full_workflow()
    test_indexer_qa_pairs()
    test_indexer_text_chunking()

    logger.info("=" * 60)
    logger.info("All tests passed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
