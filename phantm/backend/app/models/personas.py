"""
Persona request/response models.
"""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.models.base import StrictModel, LLMProvider


__all__ = [
    "PersonaCreate",
    "PersonaUpdate",
    "PersonaResponse",
    "PersonaListResponse",
]


class PersonaCreate(StrictModel):
    """Body for POST /api/personas. Photo and voice are uploaded separately."""
    name: str = Field(min_length=1, max_length=100)
    system_prompt: str | None = Field(default=None, max_length=2000)
    knowledge_base_id: UUID | None = None
    llm_provider: LLMProvider = LLMProvider.OLLAMA


class PersonaUpdate(StrictModel):
    """Body for PUT /api/personas/:id. All fields optional."""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    system_prompt: str | None = Field(default=None, max_length=2000)
    knowledge_base_id: UUID | None = None
    llm_provider: LLMProvider | None = None
    is_active: bool | None = None


class PersonaResponse(StrictModel):
    """Single persona returned by API."""
    id: UUID
    user_id: UUID
    name: str
    photo_url: str | None
    voice_sample_url: str | None
    voice_model_id: str | None
    knowledge_base_id: UUID | None
    system_prompt: str | None
    llm_provider: LLMProvider
    is_active: bool
    created_at: datetime


class PersonaListResponse(StrictModel):
    """Wrapper for GET /api/personas list endpoint."""
    personas: list[PersonaResponse]
    count: int
