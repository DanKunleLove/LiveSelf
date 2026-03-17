"""
Persona routes: list, create, get, upload photo, upload voice.

A persona is a configured avatar -- face photo, voice sample, knowledge base,
and LLM instructions. Users can have multiple personas.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from app.database.supabase import get_supabase_client
from app.middleware.auth import get_current_user_id
from app.models.personas import (
    PersonaCreate,
    PersonaListResponse,
    PersonaResponse,
)

router = APIRouter()


@router.get("", response_model=PersonaListResponse)
async def list_personas(user_id: UUID = Depends(get_current_user_id)):
    """List all personas belonging to the current user."""
    db = get_supabase_client()
    result = db.table("personas").select("*").eq(
        "user_id", str(user_id)
    ).eq("is_active", True).order("created_at", desc=True).execute()

    personas = [PersonaResponse(**row) for row in result.data]
    return PersonaListResponse(personas=personas, count=len(personas))


@router.post("", response_model=PersonaResponse, status_code=status.HTTP_201_CREATED)
async def create_persona(
    body: PersonaCreate,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create a new persona. Photo and voice are uploaded separately after creation."""
    db = get_supabase_client()

    # Validate knowledge_base_id belongs to user if provided
    if body.knowledge_base_id:
        kb_check = db.table("knowledge_bases").select("id").eq(
            "id", str(body.knowledge_base_id)
        ).eq("user_id", str(user_id)).execute()

        if not kb_check.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge base not found or does not belong to you",
            )

    result = db.table("personas").insert({
        "user_id": str(user_id),
        "name": body.name,
        "system_prompt": body.system_prompt,
        "knowledge_base_id": str(body.knowledge_base_id) if body.knowledge_base_id else None,
        "llm_provider": body.llm_provider,
    }).execute()

    return PersonaResponse(**result.data[0])


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(
    persona_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get a single persona by ID. Must belong to the current user."""
    db = get_supabase_client()
    result = db.table("personas").select("*").eq(
        "id", str(persona_id)
    ).eq("user_id", str(user_id)).single().execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona not found",
        )

    return PersonaResponse(**result.data)


@router.post("/{persona_id}/photo", response_model=PersonaResponse)
async def upload_photo(
    persona_id: UUID,
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
):
    """Upload a face photo for a persona.

    Accepts JPEG or PNG. The file gets stored in Cloudflare R2 and
    the persona's photo_url is updated with the storage URL.
    For now, stores a placeholder URL until R2 service is wired up.
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Photo must be JPEG or PNG",
        )

    # Verify persona belongs to user
    db = get_supabase_client()
    check = db.table("personas").select("id").eq(
        "id", str(persona_id)
    ).eq("user_id", str(user_id)).execute()

    if not check.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona not found",
        )

    # TODO: Upload to Cloudflare R2 via storage service
    # For now, store a placeholder path that will be replaced when R2 is wired up
    photo_url = f"r2://liveself-assets/personas/{persona_id}/photo/{file.filename}"

    result = db.table("personas").update({
        "photo_url": photo_url,
    }).eq("id", str(persona_id)).execute()

    return PersonaResponse(**result.data[0])


@router.post("/{persona_id}/voice", response_model=PersonaResponse)
async def upload_voice(
    persona_id: UUID,
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
):
    """Upload a voice sample for a persona.

    Accepts WAV or MP3 (10+ seconds recommended). Gets stored in R2 and
    later used by CosyVoice for voice cloning.
    For now, stores a placeholder URL until R2 service is wired up.
    """
    # Validate file type
    allowed_types = ("audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav")
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice sample must be WAV or MP3",
        )

    # Verify persona belongs to user
    db = get_supabase_client()
    check = db.table("personas").select("id").eq(
        "id", str(persona_id)
    ).eq("user_id", str(user_id)).execute()

    if not check.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persona not found",
        )

    # TODO: Upload to Cloudflare R2 via storage service
    voice_url = f"r2://liveself-assets/personas/{persona_id}/voice/{file.filename}"

    result = db.table("personas").update({
        "voice_sample_url": voice_url,
    }).eq("id", str(persona_id)).execute()

    return PersonaResponse(**result.data[0])
