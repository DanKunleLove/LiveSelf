"""
User and auth request/response models.
"""

from datetime import datetime
from uuid import UUID

from pydantic import EmailStr, Field

from app.models.base import StrictModel, UserPlan


__all__ = [
    "RegisterRequest",
    "LoginRequest",
    "AuthResponse",
    "UserResponse",
    "UserUpdate",
]


# -- Auth requests --

class RegisterRequest(StrictModel):
    """Body for POST /api/auth/register."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(StrictModel):
    """Body for POST /api/auth/login."""
    email: EmailStr
    password: str


# -- Auth response --

class AuthResponse(StrictModel):
    """Returned after successful register or login. Tokens go in httpOnly cookies."""
    user: "UserResponse"
    access_token: str
    refresh_token: str


# -- User profile --

class UserResponse(StrictModel):
    """Public user profile returned by GET /api/auth/me and in AuthResponse."""
    id: UUID
    email: str
    display_name: str | None
    plan: UserPlan
    minutes_used: int
    minutes_limit: int
    created_at: datetime
    last_active_at: datetime | None


class UserUpdate(StrictModel):
    """Body for updating user profile fields."""
    display_name: str | None = Field(default=None, max_length=100)
