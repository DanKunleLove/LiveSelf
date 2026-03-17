"""
Auth routes: register, login, get current user.

Authentication is handled by Supabase Auth. We call the Supabase client
to create accounts and sign in, then return the JWT tokens. The frontend
stores tokens in httpOnly cookies (never localStorage).
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.database.supabase import get_supabase_auth_client, get_supabase_client
from app.middleware.auth import get_current_user_id
from app.models.users import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)

router = APIRouter()


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    """Create a new account with email + password.

    Supabase Auth creates the auth.users row. Our DB trigger automatically
    creates the public.users profile row. We then update display_name if provided.
    """
    supabase = get_supabase_auth_client()

    try:
        auth_response = supabase.auth.sign_up({
            "email": body.email,
            "password": body.password,
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}",
        )

    if not auth_response.user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed: no user returned",
        )

    # Update display_name if provided (the trigger only sets id + email)
    if body.display_name:
        db = get_supabase_client()
        db.table("users").update({
            "display_name": body.display_name
        }).eq("id", str(auth_response.user.id)).execute()

    # Fetch the full user profile
    db = get_supabase_client()
    result = db.table("users").select("*").eq("id", str(auth_response.user.id)).single().execute()

    return AuthResponse(
        user=UserResponse(**result.data),
        access_token=auth_response.session.access_token,
        refresh_token=auth_response.session.refresh_token,
    )


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    """Login with email + password. Returns JWT access and refresh tokens."""
    supabase = get_supabase_auth_client()

    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Login failed: {str(e)}",
        )

    if not auth_response.user or not auth_response.session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Fetch full user profile
    db = get_supabase_client()
    result = db.table("users").select("*").eq("id", str(auth_response.user.id)).single().execute()

    # Update last_active_at
    db.table("users").update({
        "last_active_at": "now()"
    }).eq("id", str(auth_response.user.id)).execute()

    return AuthResponse(
        user=UserResponse(**result.data),
        access_token=auth_response.session.access_token,
        refresh_token=auth_response.session.refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user_id: UUID = Depends(get_current_user_id)):
    """Get the current authenticated user's profile."""
    db = get_supabase_client()
    result = db.table("users").select("*").eq("id", str(user_id)).single().execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found",
        )

    return UserResponse(**result.data)
