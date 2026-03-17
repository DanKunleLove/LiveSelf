"""
Knowledge Retriever — Phase 1C

Queries ChromaDB to find the most relevant knowledge chunks
for a given question. This is the RAG (Retrieval-Augmented Generation) layer.

Input: question text
Output: top 3 most relevant knowledge chunks

This is what makes the avatar sound like YOU specifically,
not just a generic AI. It retrieves YOUR answers to questions.
"""

# TODO: Phase 1C implementation
# 1. Connect to ChromaDB collection for this persona
# 2. Embed the question using sentence-transformers
# 3. Query ChromaDB for top 3 similar chunks
# 4. Return chunks as context for the LLM prompt
