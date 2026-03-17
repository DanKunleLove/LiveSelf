"""
Session routes: create, list, end.

A session represents one live avatar call. The user picks a persona,
we create a session record, and the engine takes over for real-time
face/voice/brain processing.
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.database.supabase import get_supabase_client
from app.middleware.auth import get_current_user_id
from app.models.sessions import (
    SessionCreate,
    SessionEndResponse,
    SessionListResponse,
    SessionResponse,
)

router = APIRouter()


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create a new live session for a persona.

    Validates that the persona belongs to the user and is active,
    then checks the user has minutes remaining on their plan.
    Returns a session record in 'initializing' status.
    """
    db = get_supabase_client()

    # Verify persona belongs to user and is active
    persona = db.table("personas").select("id, is_active").eq(
        "id", str(body.persona_id)
    ).eq("user_id", str(user_id)).single().execute()

    if not persona.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona not found",
        )

    if not persona.data.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Persona is inactive",
        )

    # Check minutes remaining
    user = db.table("users").select("minutes_used, minutes_limit").eq(
        "id", str(user_id)
    ).single().execute()

    if user.data["minutes_used"] >= user.data["minutes_limit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Monthly minutes limit reached. Upgrade your plan for more.",
        )

    # Create session
    result = db.table("sessions").insert({
        "user_id": str(user_id),
        "persona_id": str(body.persona_id),
        "target_platform": body.target_platform,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "active",
    }).execute()

    # Update user last_active_at
    db.table("users").update({
        "last_active_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", str(user_id)).execute()

    return SessionResponse(**result.data[0])


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user_id: UUID = Depends(get_current_user_id),
):
    """List past sessions for the current user, newest first."""
    db = get_supabase_client()
    result = db.table("sessions").select("*").eq(
        "user_id", str(user_id)
    ).order("created_at", desc=True).range(offset, offset + limit - 1).execute()

    sessions = [SessionResponse(**row) for row in result.data]

    # Get total count for pagination
    count_result = db.table("sessions").select(
        "id", count="exact"
    ).eq("user_id", str(user_id)).execute()

    return SessionListResponse(sessions=sessions, count=count_result.count or 0)


@router.put("/{session_id}/end", response_model=SessionEndResponse)
async def end_session(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """End an active session. Computes duration and updates user minutes_used.

    Only sessions with status 'active' or 'paused' can be ended.
    """
    db = get_supabase_client()

    # Fetch session and verify ownership
    session = db.table("sessions").select("*").eq(
        "id", str(session_id)
    ).eq("user_id", str(user_id)).single().execute()

    if not session.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    current_status = session.data.get("status")
    if current_status not in ("active", "paused"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot end session with status '{current_status}'",
        )

    # Calculate duration
    now = datetime.now(timezone.utc)
    started_at = datetime.fromisoformat(session.data["started_at"])
    duration_seconds = int((now - started_at).total_seconds())
    duration_minutes = max(1, duration_seconds // 60)  # Round up, minimum 1 minute

    # Update session
    result = db.table("sessions").update({
        "status": "ended",
        "ended_at": now.isoformat(),
        "duration_seconds": duration_seconds,
    }).eq("id", str(session_id)).execute()

    # Add minutes to user's usage
    user = db.table("users").select("minutes_used").eq(
        "id", str(user_id)
    ).single().execute()

    db.table("users").update({
        "minutes_used": user.data["minutes_used"] + duration_minutes,
    }).eq("id", str(user_id)).execute()

    return SessionEndResponse(
        id=session_id,
        status="ended",
        ended_at=now,
        duration_seconds=duration_seconds,
        exchange_count=result.data[0].get("exchange_count", 0),
    )
