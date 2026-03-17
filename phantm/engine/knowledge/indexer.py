"""
Knowledge Indexer — Phase 1C / Phase 2

Processes uploaded documents (PDFs, text files, Q&A pairs) into
vector embeddings stored in ChromaDB.

Pipeline:
  Document → Chunk (512 tokens, 50 overlap) → Embed → Store in ChromaDB

For Phase 1C: just index simple Q&A pairs manually.
For Phase 2: handle PDF/TXT uploads from the dashboard.
"""

# TODO: Phase 1C — simple Q&A pair indexing
# TODO: Phase 2 — full document processing pipeline
