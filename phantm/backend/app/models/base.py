"""
Shared base model configuration and common types.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that forbids extra fields and uses enum values in serialization."""
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )


# -- Shared enums used across multiple models --

class UserPlan(str, Enum):
    """Subscription plan tier."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class LLMProvider(str, Enum):
    """Which LLM backend powers the persona."""
    OLLAMA = "ollama"
    CLAUDE = "claude"
    OPENAI = "openai"


class SessionStatus(str, Enum):
    """Lifecycle status of a live avatar session."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class TargetPlatform(str, Enum):
    """Video call platform the avatar connects to."""
    ZOOM = "zoom"
    MEET = "meet"
    WHATSAPP = "whatsapp"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    OTHER = "other"


class DocumentFileType(str, Enum):
    """Supported document formats for knowledge base upload."""
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    QA_PAIRS = "qa_pairs"


class DocumentStatus(str, Enum):
    """Indexing pipeline status for an uploaded document."""
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
