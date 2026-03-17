"""
Knowledge base and document request/response models.
"""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.models.base import StrictModel, DocumentFileType, DocumentStatus


__all__ = [
    "KnowledgeBaseCreate",
    "KnowledgeBaseResponse",
    "KnowledgeBaseListResponse",
    "DocumentResponse",
]


class KnowledgeBaseCreate(StrictModel):
    """Body for creating a new knowledge base."""
    name: str = Field(min_length=1, max_length=100)


class KnowledgeBaseResponse(StrictModel):
    """Single knowledge base returned by API."""
    id: UUID
    user_id: UUID
    name: str
    chroma_collection_id: str
    document_count: int
    chunk_count: int
    last_updated_at: datetime | None
    created_at: datetime


class KnowledgeBaseListResponse(StrictModel):
    """Wrapper for listing knowledge bases."""
    knowledge_bases: list[KnowledgeBaseResponse]
    count: int


class DocumentResponse(StrictModel):
    """Single document within a knowledge base."""
    id: UUID
    kb_id: UUID
    filename: str
    file_url: str
    file_type: DocumentFileType
    status: DocumentStatus
    chunk_count: int | None
    created_at: datetime
