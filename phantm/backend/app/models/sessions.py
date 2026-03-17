"""
Session and exchange request/response models.
"""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.models.base import StrictModel, SessionStatus, TargetPlatform


__all__ = [
    "SessionCreate",
    "SessionResponse",
    "SessionListResponse",
    "SessionEndResponse",
    "ExchangeResponse",
]


class SessionCreate(StrictModel):
    """Body for POST /api/sessions. Starts a new live avatar session."""
    persona_id: UUID
    target_platform: TargetPlatform | None = None


class SessionResponse(StrictModel):
    """Single session returned by API."""
    id: UUID
    user_id: UUID
    persona_id: UUID
    status: SessionStatus
    target_platform: TargetPlatform | None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: int | None
    exchange_count: int
    created_at: datetime


class SessionListResponse(StrictModel):
    """Wrapper for GET /api/sessions list endpoint."""
    sessions: list[SessionResponse]
    count: int


class SessionEndResponse(StrictModel):
    """Returned by PUT /api/sessions/:id/end with computed stats."""
    id: UUID
    status: SessionStatus
    ended_at: datetime
    duration_seconds: int
    exchange_count: int


class ExchangeResponse(StrictModel):
    """Single Q&A exchange within a session."""
    id: UUID
    session_id: UUID
    question_text: str
    answer_text: str
    kb_chunks_used: list[str] | None
    latency_ms: int | None
    user_edited: bool
    created_at: datetime
